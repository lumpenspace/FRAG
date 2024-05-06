from frag.embeddings.apis import OAIEmbedAPI, EmbedAPI, HFEmbedAPI
from frag.typedefs.embed_types import ApiSource


def get_embed_api(
    api_name: str, api_source: ApiSource, api_key: str | None = None
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

    if not api_name:
        raise ValueError("Embedding model and chunking settings must be provided")
    if api_source == "OpenAI":
        return OAIEmbedAPI(name=api_name, api_key=api_key)
    elif api_source == "HuggingFace":
        try:
            embed_api = HFEmbedAPI(name=api_name)
        except ValueError:
            raise ValueError(f"Invalid embedding API: {api_name} on {api_source}")

    raise ValueError("Invalid embedding API type: %s" % type(embed_api))
