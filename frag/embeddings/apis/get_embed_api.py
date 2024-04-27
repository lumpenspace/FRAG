from frag.embeddings.apis import OAIEmbedAPI, EmbedAPI, SentenceEmbedAPI


def get_embed_api(embed_api: EmbedAPI | str) -> EmbedAPI:
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
    if isinstance(embed_api, EmbedAPI):
        return embed_api
    if isinstance(embed_api, str):
        if embed_api.startswith("oai:"):
            return OAIEmbedAPI(name=embed_api.replace("oai:", ""))
        return SentenceEmbedAPI(name=embed_api)
    if issubclass(embed_api, EmbedAPI):
        return embed_api()
