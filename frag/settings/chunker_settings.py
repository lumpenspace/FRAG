"""
Module containing the settings used for the chunker.

They are contained under embed.chunker in the .fragrc file.
"""

from typing import Any, Dict, Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings
from frag.console import console


class ChunkerSettings(BaseSettings):
    """
    Settings for the chunker.
    """

    chunk_size: int = 0
    chunk_overlap: int = 0
    preserve_sentences: bool = False
    preserve_paragraphs: bool = True
    max_length: int = 512

    @model_validator(mode="after")
    def validate_chunk_size(self) -> Self:
        if self.chunk_size == 0:
            self.chunk_size = self.max_length - self.chunk_overlap * 2
        if self.chunk_size < self.chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")
        if self.chunk_size + self.chunk_overlap * 2 > self.max_length:
            console.log(
                "max_length is greater than the maximum number of tokens for the model.\
                Setting max_length to the maximum number of tokens."
            )
            self.chunk_size = self.max_length - self.chunk_overlap * 2
        return self

    @classmethod
    def from_dict(cls, d: Dict[str, Any], api_max_tokens: int) -> Self:
        """
        Create a ChunkerSettings object from a dictionary, using the model to validate it.
        """
        if "max_length" in d:
            if d["max_length"] > api_max_tokens:
                d["max_length"] = api_max_tokens
                console.log(
                    "max_length is greater than the maximum number of tokens for the model.\
                    Setting max_length to the maximum number of tokens."
                )
        return cls(**d)
