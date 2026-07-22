import tomllib
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


def load_config():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def llm_client(config, api_key="ollama"):
    """Les clés de [llm] (model, base_url, temperature, max_tokens...)
    correspondent aux arguments de ChatOpenAI et sont passées telles quelles."""
    return ChatOpenAI(api_key=api_key, **config["llm"])


def embedding_client(config, api_key):
    """Client d'embedding partagé par l'ingestion et la requête : les deux
    doivent utiliser exactement le même modèle, sinon les vecteurs ne sont
    plus comparables."""
    return OpenAIEmbeddings(model=config["embedding"]["model"],
                            api_key=api_key,
                            base_url=config["embedding"]["base_url"],
                            check_embedding_ctx_length=False)
