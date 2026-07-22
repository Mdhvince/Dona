import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from markdown import markdown

# Les modules de src/ utilisent des imports plats (from config import ...)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever

from config import load_config, llm_client, embedding_client
from ingest import DOCS_DIRS
from response_helper import answer
from retrieval import HybridRetriever

PERSIST_DIR = str(ROOT / "vectordb")

app = Flask(__name__)

# Pipeline RAG construit une seule fois au démarrage, partagé par les requêtes
load_dotenv()
config = load_config()
chat_llm = llm_client(config)
vectordb = Chroma(persist_directory=PERSIST_DIR,
                  embedding_function=embedding_client(config, os.environ.get("OPENROUTER_API_KEY")))
hybrid_retriever = HybridRetriever.from_vectordb(vectordb, **config["retriever"])
retriever = MultiQueryRetriever.from_llm(retriever=hybrid_retriever, llm=chat_llm,
                                         include_original=True)


def source_payload(result):
    """Prépare la source d'une réponse pour le front : nom, page et URL
    servie par la route /source (les liens file:// sont bloqués en HTTP)."""
    if not result.source_path:
        return None
    return {"name": Path(result.source_path).name,
            "page": result.source_page,
            "url": url_for("source", path=result.source_path)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="Question vide."), 400
    result = answer(question, retriever, chat_llm)
    return jsonify(response=markdown(result.response),
                   source=source_payload(result))


@app.route("/source")
def source():
    """Sert un document source dans le navigateur, restreint aux racines indexées."""
    path = Path(request.args.get("path", "")).resolve()
    if not any(path.is_relative_to(root) for root in DOCS_DIRS):
        abort(403)
    if not path.is_file():
        abort(404)
    return send_file(path)
