# Personal Assistant

Assistant personnel agentique sur mes données (documents Google Drive,
agendas), 100% local : Ollama (LLM, vision, embeddings), Chroma, LangChain/
LangGraph, Flask.

## Documentation

- [docs/rag.md](docs/rag.md) - ingestion des documents, index, retrieval
- [docs/agentic.md](docs/agentic.md) - couche agentique : agent, outils, serveurs MCP, conversation
- [docs/webapp.md](docs/webapp.md) - interface, routes, streaming

## Prérequis

- [Ollama](https://ollama.com) démarré, avec les modèles listés dans
  `config.toml` ([llm], [vlm], [embedding]).
- `uv sync` pour les dépendances Python.

## Utilisation

```bash
uv run python -m src.ingest    # ingestion incrémentale
uv run python run.py           # webapp sur http://127.0.0.1:5001
uv run pytest                  # tests
```
