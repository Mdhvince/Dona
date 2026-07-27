# Personal Assistant

Assistant personnel agentique sur mes données (documents Google Drive,
agendas), 100% local : Ollama (LLM, vision, embeddings), Chroma, LangChain/
LangGraph, Flask.

## Documentation

- [docs/rag.md](docs/rag.md) - ingestion des documents, index, retrieval
- [docs/agentic.md](docs/agentic.md) - couche agentique : agent, outils, serveurs MCP, conversation
- [docs/webapp.md](docs/webapp.md) - interface, routes, streaming

### Visuel (ouvrir dans un navigateur)

- [docs/map.html](docs/map.html) - carte interactive du système, drill-down jusqu'au fichier
- [docs/sim-ingestion.html](docs/sim-ingestion.html) - du fichier Drive au vecteur, pas à pas
- [docs/sim-retrieval.html](docs/sim-retrieval.html) - recherche hybride et fusion RRF, avec le calcul détaillé
- [docs/sim-agent.html](docs/sim-agent.html) - routage, middlewares, outils, confirmation
- [docs/sim-reponse.html](docs/sim-reponse.html) - streaming, citations, sources, échecs d'outils

## Prérequis

- [Ollama](https://ollama.com) démarré, avec les modèles listés dans
  `config.toml` ([llm], [vlm], [embedding]).
- `uv sync` pour les dépendances Python.

## Utilisation

```bash
uv run python -m src.ingestor    # ingestion incrémentale
uv run python run.py           # webapp sur http://127.0.0.1:5001
uv run pytest                  # tests
```
