from llama_index.core.embeddings import BaseEmbedding
from frag.typedefs.embed_types import ApiSource


def get_embed_api(
    api_source: ApiSource, api_model: str | None, api_key: str | None = None
) -> BaseEmbedding:
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

    if not api_model:
        raise ValueError("Embedding model and chunking settings must be provided")
    if api_source == "OpenAI":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=api_model, api_key=api_key)
    elif api_source == "HuggingFace":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        try:
            embed_api = HuggingFaceEmbedding(model=api_model)
        except ValueError:
            raise ValueError(f"Invalid embedding API: {api_model} on {api_source}")

    raise ValueError("Invalid embedding API type: %s" % type(embed_api))
