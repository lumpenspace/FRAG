from pathlib import Path
from typing import Self, Dict, Any
from typing_extensions import TypedDict
from pydantic_settings import BaseSettings
from pydantic import field_validator, ValidationInfo
import logging

from frag.embeddings.get_embed_api import get_embed_api
from llama_index.core.embeddings import BaseEmbedding
from frag.typedefs.embed_types import ApiSource
from frag.utils.console import console, error_console

EmbedSettingsDict = TypedDict(
    "EmbedSettingsDict",
    {
        "api_name": str,
        "api_source": ApiSource,
        "chunk_overlap": int,
        "path": Path,
        "default_collection": str,
    },
)


class EmbedSettings(BaseSettings):
    api_source: ApiSource = "OpenAI"
    api_model: str = "text-embedding-3-large"
    chunk_overlap: int = 0
    path: Path = Path("./db")
    default_collection: str = "default"

    @field_validator("default_collection")
    @classmethod
    def validate_default_collection(cls, v: str) -> str:
        """
        Validate the collection name.
        """
        if not v:
            v = "default_collection"
            logging.warning(
                "Collection name not provided, using default collection name: %s",
                v,
            )
        return v

    @property
    def api(self) -> BaseEmbedding:
        if not hasattr(self, "_api"):
            self._api: BaseEmbedding = get_embed_api(
                api_model=self.api_model, api_source=self.api_source
            )
        return self._api

    @classmethod
    def from_dict(
        cls,
        embeds_dict: Dict[str, Any],
    ) -> Self:
        console.log(f"[b]EmbedApiSettings:[/] {embeds_dict}")
        api_model: str = embeds_dict.get("api_model", "text-embedding-3-large")
        api_source: ApiSource = embeds_dict.get("api_source", "OpenAI")
        instance: Self = cls(
            api_model=api_model,
            api_source=api_source,
            chunk_overlap=embeds_dict.get("chunk_overlap", 0),
            default_collection=embeds_dict.get("default_collection", "default"),
            path=Path(embeds_dict.get("path", "./db")),
        )
        try:
            instance.api
        except Exception as e:
            error_console.log("Error getting embed api", e)

        return instance
