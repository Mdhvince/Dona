import json
import threading
import time
from pathlib import Path

import nh3
from flask import (Flask, Response, abort, jsonify, render_template, request,
                   send_file, stream_with_context, url_for)
from langchain_chroma import Chroma
from markdown import markdown

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.agent import (build_agent, collect_queries, collect_sources,
                       current_turn, parse_citations, source_label,
                       tool_failures)
from src.tools import confirmed_tool_names, load_mcp_tools, make_calendar_finder
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
calendar_finder = make_calendar_finder(mcp_tools)
extra_tools = mcp_tools + ([calendar_finder] if calendar_finder else [])


def build_rag_agent():
    """(Re)build the agent and its retriever. BM25 is an in-memory index, so
    it must be rebuilt after every vector store update. Stays None while the
    store is empty (first launch before any indexing)."""
    global agent
    if not vectordb.get(limit=1)["ids"]:
        agent = None
        return
    retriever = HybridRetriever.from_vectordb(vectordb, **config["retriever"])
    agent = build_agent(retriever, chat_llm, checkpointer, extra_tools=extra_tools,
                        confirm_tools=confirmed_tool_names(config))


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
    return [{"name": source_label(info),
             "page": info["page"],
             "url": url_for("source", path=info["path"])}
            for info in sources]


@app.route("/")
def index():
    return render_template("index.html")


def agent_events(agent_input, thread_id):
    """NDJSON lines for one run - a new question or a resumed confirmation.
    Throttled {phase, tokens} heartbeats come from the token stream, {tool,
    args} and {retrieved} from the state stream. The run ends either with a
    {confirm} request (a side-effecting tool awaits approval) or with the
    full answer payload."""
    run_config = {"configurable": {"thread_id": thread_id}}
    state, seen_calls, seen_docs = None, set(), 0
    thinking_tokens = answer_tokens = 0
    last_beat = 0.0

    def heartbeat(payload):
        nonlocal last_beat
        if time.monotonic() - last_beat < 0.4:
            return None
        last_beat = time.monotonic()
        return json.dumps(payload) + "\n"

    for mode, data in agent.stream(agent_input, config=run_config,
                                   stream_mode=["messages", "values"]):
        if mode == "messages":
            chunk, _ = data
            beat = None
            reasoning = (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content")
            if reasoning:
                thinking_tokens += 1
                beat = heartbeat({"phase": "thinking", "tokens": thinking_tokens})
            elif getattr(chunk, "tool_call_chunks", None):
                beat = heartbeat({"phase": "tool_prep"})
            elif getattr(chunk, "content", None):
                answer_tokens += 1
                beat = heartbeat({"phase": "answer", "tokens": answer_tokens})
            if beat:
                yield beat
            continue

        state = data
        if isinstance(state, dict) and state.get("__interrupt__"):
            # The args shown are the ones that would really be sent, never a
            # paraphrase by the model
            requests = state["__interrupt__"][0].value.get("action_requests", [])
            yield json.dumps({"confirm": [{"tool": r["name"], "args": r["args"]}
                                          for r in requests]}) + "\n"
            return

        turn = current_turn(state["messages"])
        for message in turn:
            for call in getattr(message, "tool_calls", None) or []:
                if call["id"] not in seen_calls:
                    seen_calls.add(call["id"])
                    thinking_tokens = 0
                    yield json.dumps({"tool": call["name"], "args": call["args"]}) + "\n"
        docs = len(collect_sources(turn))
        if docs > seen_docs:
            seen_docs = docs
            yield json.dumps({"retrieved": docs}) + "\n"

    turn = current_turn(state["messages"])
    retrieved = collect_sources(turn)
    text = turn[-1].content
    # Candidates from the whole thread: a follow-up answered from memory may
    # legitimately cite an earlier turn's document
    text, sources = parse_citations(text, collect_sources(state["messages"]))
    failed, succeeded = tool_failures(turn)
    status = "ok" if not failed else ("partial" if succeeded else "error")
    # nh3 strips raw HTML that python-markdown lets through (XSS via corpus)
    yield json.dumps({
        "response": nh3.clean(markdown(text, extensions=["tables"])),
        "status": status,
        "failed_tools": sorted(set(failed)),
        "sources": sources_payload(sources),
        "consulted": len(retrieved),
        "queries": collect_queries(turn)}) + "\n"


def stream_run(agent_input, thread_id):
    def generate():
        try:
            yield from agent_events(agent_input, thread_id)
        except Exception:
            app.logger.exception("agent failed")
            yield json.dumps({"error": "Le modèle n'a pas réussi à produire une réponse, réessaie."}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()
    thread_id = payload.get("thread_id", "").strip() or "default"
    if not question:
        return jsonify(error="Question vide."), 400
    if agent is None:
        return jsonify(error="Index vide : lance une ré-indexation."), 503
    return stream_run({"messages": [{"role": "user", "content": question}]}, thread_id)


@app.route("/confirm", methods=["POST"])
def confirm():
    """Resume a run paused on a side-effecting tool. Anything other than an
    explicit approval rejects the action: the default is to do nothing."""
    payload = request.get_json(silent=True) or {}
    thread_id = payload.get("thread_id", "").strip() or "default"
    approved = payload.get("approved") is True
    if agent is None:
        return jsonify(error="Index vide : lance une ré-indexation."), 503
    decision = ({"type": "approve"} if approved
                else {"type": "reject", "message": "Action refusée par Medhy."})
    return stream_run(Command(resume={"decisions": [decision]}), thread_id)


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
