# Personal Assistant

Assistant personnel de question/réponse (RAG) sur mes documents Google Drive,
100% local : Ollama (LLM, vision, embeddings), Chroma, LangChain, Flask.

L'architecture complète et ses justifications sont dans [docs/rag.md](docs/rag.md).

## Prérequis

- [Ollama](https://ollama.com) démarré, avec les modèles `gpt-oss:20b`,
  `qwen3.5:9b` et `qwen3-embedding:8b` (voir `config.toml`).
- `uv sync` pour les dépendances Python.

## Utilisation

```bash
uv run python src/ingest.py          # ingestion incrémentale
uv run python src/ingest.py --full   # reconstruction complète (~30s/page)
uv run python run.py                 # webapp sur http://127.0.0.1:5000
uv run pytest                        # tests
```

La webapp permet de poser une question, voir la réponse avec ses sources
(aperçu PDF ouvert à la page citée), et relancer l'ingestion incrémentale.
