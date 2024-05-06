"""
Settings for the embedding API and database.
"""

import logging
import os
from typing import Any, Dict

from pydantic import field_validator
from pydantic_settings import BaseSettings


class DBSettings(BaseSettings):
    path: str = "./db"
    collection_name: str = "default"

    @field_validator("path")
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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DBSettings":
        """
        Create a DBSettings object from a dictionary, using the model to validate it.
        """
        return cls(**d)
