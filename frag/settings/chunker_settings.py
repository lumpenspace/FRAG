"""
Module containing the settings used for the chunker.

They are contained under embed.chunker in the .fragrc file.
"""

from distutils.command import sdist
from logging import Logger, getLogger
from re import S
from typing import Any, Dict

from pydantic_settings import BaseSettings

logger: Logger = getLogger(__name__)


class ChunkerSettings(BaseSettings):
    """
    Settings for the chunker.
    """

    chunk_size: int = 10000
    chunk_overlap: int = 0
    preserve_sentences: bool = False
    preserve_paragraphs: bool = True
    max_length: int = 512
    buffer_before: int = 0
    buffer_after: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkerSettings":
        """
        Create a ChunkerSettings object from a dictionary, using the model to validate it.
        """
        if "max_length" in d:
            if d["max_length"] > d["api_max_tokens"]:
                d["max_length"] = d["api_max_tokens"]
                logger.warning(
                    "max_length is greater than the maximum number of tokens for the model.\
                    Setting max_length to the maximum number of tokens."
                )
        return cls(**d)
