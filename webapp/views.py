import json
import threading
from pathlib import Path

import nh3
from flask import (Flask, Response, abort, jsonify, render_template, request,
                   send_file, stream_with_context, url_for)
from langchain_chroma import Chroma
from markdown import markdown

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent import (build_agent, collect_queries, collect_sources,
                       current_turn, validate_citations)
from src.tools import load_mcp_tools
from src.config import load_config, llm_client, embedding_client, vlm_client
from src.ingest import DOCS_DIRS, sync
from src.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(ROOT / "vectordb")

app = Flask(__name__)

# RAG pipeline built once at startup, shared across requests
config = load_config()
chat_llm = llm_client(config)
vectordb = Chroma(persist_directory=PERSIST_DIR,
                  embedding_function=embedding_client(config))

agent = None
# Kept outside the agent so conversations survive a reindex rebuild; sqlite
# so they also survive server restarts. check_same_thread: Flask threads +
# reindex thread share the connection, SqliteSaver serializes the accesses.
checkpointer = SqliteSaver(sqlite3.connect(str(ROOT / "conversations.db"),
                                           check_same_thread=False))
# Loaded once at startup: reindex rebuilds reuse the same MCP tools
mcp_tools = load_mcp_tools(config)


def build_rag_agent():
    """(Re)build the agent and its retriever. BM25 is an in-memory index, so
    it must be rebuilt after every vector store update. Stays None while the
    store is empty (first launch before any indexing)."""
    global agent
    if not vectordb.get(limit=1)["ids"]:
        agent = None
        return
    retriever = HybridRetriever.from_vectordb(vectordb, **config["retriever"])
    agent = build_agent(retriever, chat_llm, checkpointer, extra_tools=mcp_tools)


build_rag_agent()

# One reindex at a time, running in a background thread
reindex_state = {"running": False, "done": 0, "total": 0, "current": None,
                 "result": None, "error": None}
reindex_lock = threading.Lock()


def on_progress(done, total, current):
    reindex_state.update(done=done, total=total, current=current)


def run_reindex():
    try:
        result = sync(DOCS_DIRS, vectordb, vlm_client(config),
                      chunk_size=config["ingestion"]["chunk_size"],
                      chunk_overlap=config["ingestion"]["chunk_overlap"],
                      on_progress=on_progress)
        build_rag_agent()
        reindex_state["result"] = result
    except Exception as exc:
        reindex_state["error"] = str(exc)
    finally:
        reindex_state["running"] = False


def sources_payload(sources):
    """Source metadata ready for the front end: name, page and a URL served
    by the /source route (file:// links are blocked on HTTP pages)."""
    return [{"name": Path(info["path"]).name,
             "page": info["page"],
             "url": url_for("source", path=info["path"])}
            for info in sources]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """NDJSON stream: one line per new batch of search queries as the agent
    works, then a final line with the full answer payload."""
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()
    thread_id = payload.get("thread_id", "").strip() or "default"
    if not question:
        return jsonify(error="Question vide."), 400
    if agent is None:
        return jsonify(error="Index vide : lance une ré-indexation."), 503

    def generate():
        try:
            run_config = {"configurable": {"thread_id": thread_id}}
            state, seen_calls, seen_docs = None, set(), 0
            for state in agent.stream({"messages": [{"role": "user", "content": question}]},
                                      config=run_config, stream_mode="values"):
                turn = current_turn(state["messages"])
                for message in turn:
                    for call in getattr(message, "tool_calls", None) or []:
                        if call["id"] not in seen_calls:
                            seen_calls.add(call["id"])
                            yield json.dumps({"tool": call["name"], "args": call["args"]}) + "\n"
                docs = len(collect_sources(turn))
                if docs > seen_docs:
                    seen_docs = docs
                    yield json.dumps({"retrieved": docs}) + "\n"

            turn = current_turn(state["messages"])
            retrieved = collect_sources(turn)
            structured = state.get("structured_response")
            if structured is not None:
                text = structured.response
                # Validated against the whole thread: a follow-up answered from
                # memory may legitimately cite an earlier turn's document
                sources = validate_citations(structured.sources,
                                             collect_sources(state["messages"]))
            else:
                text = turn[-1].content
                sources = retrieved
            # nh3 strips raw HTML that python-markdown lets through (XSS via corpus)
            yield json.dumps({
                "response": nh3.clean(markdown(text, extensions=["tables"])),
                "sources": sources_payload(sources),
                "consulted": len(retrieved),
                "queries": collect_queries(turn)}) + "\n"
        except Exception:
            app.logger.exception("agent failed")
            yield json.dumps({"error": "Le modèle n'a pas réussi à produire une réponse, réessaie."}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.route("/source")
def source():
    """Serve a source document in the browser, restricted to indexed roots."""
    path = Path(request.args.get("path", "")).resolve()
    if not any(path.is_relative_to(root) for root in DOCS_DIRS):
        abort(403)
    if not path.is_file():
        abort(404)
    return send_file(path)


@app.route("/reindex", methods=["POST"])
def reindex():
    """Start an incremental reindex in the background, one run at a time.
    A full rebuild is a long batch reserved for the CLI (--full)."""
    with reindex_lock:
        if not reindex_state["running"]:
            reindex_state.update(running=True, done=0, total=0, current=None,
                                 result=None, error=None)
            threading.Thread(target=run_reindex, daemon=True).start()
    return jsonify(**reindex_state)


@app.route("/reindex/status")
def reindex_status():
    return jsonify(**reindex_state)
