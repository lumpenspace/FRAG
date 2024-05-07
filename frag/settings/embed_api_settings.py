from typing import Self, Dict, Any
from typing_extensions import TypedDict
from pydantic_settings import BaseSettings
from frag.embeddings.apis import EmbedAPI, get_embed_api
from frag.typedefs.embed_types import ApiSource
from frag.console import console

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
    max_tokens: int

    @property
    def api(self) -> EmbedAPI:
        return get_embed_api(api_name=self.api_name, api_source=self.api_source)

    @classmethod
    def from_dict(
        cls,
        embeds_dict: Dict[str, Any],
    ) -> Self:
        console.log("embeds_dict", embeds_dict)
        api_name: str = embeds_dict.get("api_name", "text-embedding-3-large")
        api_source: ApiSource = embeds_dict.get("api_source", "OpenAI")
        api: EmbedAPI = get_embed_api(api_name=api_name, api_source=api_source)
        max_tokens: int = embeds_dict.get("max_tokens", api.max_tokens)

        return cls(api_name=api_name, api_source=api_source, max_tokens=max_tokens)
