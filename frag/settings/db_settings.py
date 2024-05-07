"""
Settings for the embedding API and database.
"""

import logging
import os
from typing import Any, Dict
from typing_extensions import TypedDict
from frag.console import console
from pydantic import field_validator
from pydantic_settings import BaseSettings

DBSettingsDict = TypedDict(
    "DBSettingsDict",
    {
        "path": str,
        "collection_name": str,
    },
)


class DBSettings(BaseSettings):
    path: str = "./db"
    default_collection: str = "default"

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """
        Check if the database path exists and is writable. If not, create it.
        """
        if not os.path.exists(v):
            if not os.access(v, os.W_OK):
                raise ValueError(f"Database path {v} is not writable")
            os.makedirs(v)
        return v

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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DBSettings":
        """
        Create a DBSettings object from a dictionary, using the model to validate it.
        """
        console.log(f"[b]DBSettings:[/] {d}")
        return cls(**d)
