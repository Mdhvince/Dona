import os
import re
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
ENV_PATH = Path(__file__).parent.parent / ".env"

_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


def _expand(value):
    """Expand ${VAR} placeholders from the environment (.env): personal
    data (emails, paths, credentials) stays out of the tracked config."""
    if isinstance(value, str):
        def repl(match):
            var = match.group(1)
            if var not in os.environ:
                raise KeyError(
                    f"Variable d'environnement manquante : {var} (voir .env.example)")
            return os.environ[var]
        return _PLACEHOLDER.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config():
    load_dotenv(ENV_PATH)
    with open(CONFIG_PATH, "rb") as f:
        return _expand(tomllib.load(f))


def docs_dirs():
    """Drive roots to index, from DOCS_DIRS in .env (os.pathsep-separated).
    Raises rather than returning empty: sync() against zero roots would
    read it as 'everything deleted' and wipe the index."""
    load_dotenv(ENV_PATH)
    raw = os.environ.get("DOCS_DIRS", "")
    dirs = [Path(p) for p in raw.split(os.pathsep) if p.strip()]
    if not dirs:
        raise SystemExit("DOCS_DIRS absent de .env (voir .env.example) : "
                         "racines d'indexation inconnues")
    return dirs


def llm_client(config):
    """Keys in [llm] (model, base_url, temperature, num_predict, reasoning...)
    map directly to ChatOllama arguments and are passed through as-is."""
    return ChatOllama(**config["llm"])


def vlm_client(config):
    """Vision model, through Ollama's native API which exposes reasoning
    control: transcribes PDF pages and images at ingestion time."""
    return ChatOllama(**config["vlm"])


def embedding_client(config, api_key=None):
    """Embedding client shared by ingestion and querying: both must use the
    exact same model, otherwise the vectors are not comparable."""
    return OpenAIEmbeddings(model=config["embedding"]["model"],
                            api_key=api_key or "ollama",
                            base_url=config["embedding"]["base_url"],
                            check_embedding_ctx_length=False)
