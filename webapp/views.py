import sys
import threading
from pathlib import Path

import nh3
from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from markdown import markdown

# Modules in src/ use flat imports (from config import ...)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever

from config import load_config, llm_client, embedding_client, vlm_client
from ingest import DOCS_DIRS, sync
from response_helper import answer
from retrieval import HybridRetriever, MULTI_QUERY_PROMPT

PERSIST_DIR = str(ROOT / "vectordb")

app = Flask(__name__)

# RAG pipeline built once at startup, shared across requests
config = load_config()
chat_llm = llm_client(config)
vectordb = Chroma(persist_directory=PERSIST_DIR,
                  embedding_function=embedding_client(config))

retriever = None


def build_retriever():
    """(Re)build the retrieval chain. BM25 is an in-memory index, so it must
    be rebuilt after every vector store update. Stays None while the store
    is empty (first launch before any indexing)."""
    global retriever
    if not vectordb.get(limit=1)["ids"]:
        retriever = None
        return
    hybrid_retriever = HybridRetriever.from_vectordb(vectordb, **config["retriever"])
    retriever = MultiQueryRetriever.from_llm(retriever=hybrid_retriever, llm=chat_llm,
                                             prompt=MULTI_QUERY_PROMPT,
                                             include_original=True)


build_retriever()

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
        build_retriever()
        reindex_state["result"] = result
    except Exception as exc:
        reindex_state["error"] = str(exc)
    finally:
        reindex_state["running"] = False


def sources_payload(result):
    """Source metadata ready for the front end: name, page and a URL served
    by the /source route (file:// links are blocked on HTTP pages)."""
    return [{"name": Path(info["path"]).name,
             "page": info["page"],
             "url": url_for("source", path=info["path"])}
            for info in result.sources_info]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="Question vide."), 400
    if retriever is None:
        return jsonify(error="Index vide : lance une ré-indexation."), 503
    try:
        result = answer(question, retriever, chat_llm)
    except Exception:
        app.logger.exception("answer() failed")
        return jsonify(error="Le modèle n'a pas réussi à produire une réponse, réessaie."), 500
    # nh3 strips raw HTML that python-markdown lets through (XSS via corpus)
    return jsonify(response=nh3.clean(markdown(result.response, extensions=["tables"])),
                   sources=sources_payload(result))


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
