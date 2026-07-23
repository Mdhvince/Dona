import tomllib
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


def load_config():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def llm_client(config, api_key="ollama"):
    """Keys in [llm] (model, base_url, temperature, max_tokens...) map
    directly to ChatOpenAI arguments and are passed through as-is."""
    return ChatOpenAI(api_key=api_key, **config["llm"])


def rewriter_client(config, api_key="ollama"):
    """LLM for query rewriting: deterministic one-liner output, used only
    to prepare the retrieval query, never shown to the user."""
    return ChatOpenAI(api_key=api_key, **config["rewriter"])


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
