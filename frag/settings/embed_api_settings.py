from typing import TypedDict
from pydantic_settings import BaseSettings
from frag.embeddings.apis import EmbedAPI, get_embed_api
from frag.typedefs.embed_types import ApiSource


EmbedApiSettingsDict = TypedDict(
    "EmbedApiSettingsDict",
    {
        "api_name": str,
        "api_source": ApiSource,
    },
)


class EmbedAPISettings(BaseSettings):
    api_name: str = "text-embedding-3-large"
    api_source: ApiSource = "OpenAI"
    max_tokens: int = 1000
    api: EmbedAPI

    def __init__(self, api_name: str, api_source: ApiSource) -> None:
        api: EmbedAPI = get_embed_api(api_name=api_name, api_source=api_source)
        max_tokens: int = api.max_tokens
        super().__init__(
            api_name=api_name, api_source=api_source, max_tokens=max_tokens
        )

    @classmethod
    def from_dict(cls, data: EmbedApiSettingsDict) -> "EmbedAPISettings":
        return cls(**data)
