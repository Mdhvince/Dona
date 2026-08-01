from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import chat_client


def test_chat_client_defaults_to_ollama():
    client = chat_client({"conversation": {"model": "qwen", "base_url": "http://localhost:11434"}},
                         "conversation")
    assert isinstance(client, ChatOllama)


def test_chat_client_builds_an_openai_compatible_client():
    client = chat_client({"router": {"provider": "openai", "model": "gpt-oss-120b",
                                     "base_url": "http://melious.test/v1", "api_key": "k"}},
                         "router")
    assert isinstance(client, ChatOpenAI)
    assert client.model_name == "gpt-oss-120b"
