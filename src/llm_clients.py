from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def llm(api_key, model_id, base_url="https://openrouter.ai/api/v1"):
    return ChatOpenAI(model=model_id,
                      api_key=api_key,
                      temperature=0.15,
                      max_tokens=2000,
                      base_url=base_url)


def embedding_model(api_key, model_id="openai/text-embedding-3-small"):
    return OpenAIEmbeddings(
        model=model_id,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,
    )
