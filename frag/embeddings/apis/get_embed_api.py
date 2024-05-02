from frag.embeddings.apis import OAIEmbedAPI, EmbedAPI, SentenceEmbedAPI


def get_embed_api(
    embed_api: EmbedAPI | str, max_tokens: int = 512, api_key: str | None = None
) -> EmbedAPI:
    """
    Retrieves an embedding API instance based on the input.

    Args:
        embed_api (EmbedAPI|str): The embedding API instance or a string identifier for the API.
            For OpenAI APIs, prepend 'oai:' to the API name.

    Returns:
        EmbedAPI: An instance of the requested embedding API.

    Raises:
        ValueError: If no embedding API is provided.
    """
    if not embed_api:
        raise ValueError("Embedding model and chunking settings must be provided")
    if isinstance(embed_api, str):
        if embed_api.startswith("oai:"):
            return OAIEmbedAPI(name=embed_api.replace("oai:", ""), api_key=api_key)
        try:
            embed_api = SentenceEmbedAPI(name=embed_api, max_tokens=max_tokens)
        except ValueError:
            raise ValueError(f"Invalid embedding API: {embed_api}")

    if isinstance(type(embed_api), EmbedAPI):
        return embed_api

    raise ValueError("Invalid embedding API type: %s" % type(embed_api))
