"""
Settings for the embedding API and database.
"""

import logging
import os
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings

from frag.embeddings.apis import EmbedAPI, get_embed_api

from .chunker_settings import ChunkerSettings


class EmbedSettings(BaseSettings):
    """
    Settings for the embedding API and database.
    """

    db_path: str = os.path.join(os.path.dirname(__file__), "db")
    collection_name: str = "default_collection"
    api_name: str = "oai:text-embedding-3-large"

    chunker: ChunkerSettings = ChunkerSettings()

    _api: EmbedAPI

    def __init__(self, api_name: str | None = None, **data: Any) -> None:
        super().__init__(**data)
        # this could also be done with default_factory
        self._api = get_embed_api(api_name or self.api_name)

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        """
        Check if the database path exists and is writable. If not, create it.
        """
        if not os.path.exists(v):
            if not os.access(v, os.W_OK):
                raise ValueError(f"Database path {v} is not writable")
            os.makedirs(v)
        return v

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
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

    field_validator("embed_model")

    @classmethod
    def validate_embed_model(cls, v: str) -> str:
        """
        Validate the embed model name. If none is provided, use the default.
        """
        if not v:
            v = "oai:text-embedding-3-large"
            logging.warning(
                "Embed model not provided, using default embed model: %s", v
            )
        else:
            logging.info("Using embed model: %s", v)
        return v
