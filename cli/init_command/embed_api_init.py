from typing import Callable

from frag.settings import EmbedApiSettingsDict
from frag.embeddings.apis import openai_embedding_models

from rich.prompt import Prompt


def embed_api_init(path: str) -> EmbedApiSettingsDict:

    api_source: str = Prompt.ask(
        choices=["OpenAI", "HuggingFace"],
        default="OpenAI",
        prompt="API provider for embeddings:",
    )

    if api_source == "OpenAI":
        embed_model: str = Prompt.ask(
            choices=openai_embedding_models,
            prompt="Select OpenAI embedding model:",
            default="text-embedding-3-small",
        )
        api_name: str = embed_model
    else:
        # free text
        embed_model = Prompt.ask(
            prompt="Enter the Hugging Face embedding model name:",
            default="all-MiniLM-L6-v2",
        )
        api_name = embed_model

    return EmbedApiSettingsDict(api_name=api_name, api_source=api_source)
