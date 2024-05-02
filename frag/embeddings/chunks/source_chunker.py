"""
This module contains the `SourceChunker` class, which is responsible for
breaking down text into manageable chunks for embedding.

It leverages settings defined in `ChunkerSettings` to customize the
chunking process, such as preserving paragraphs or sentences, and setting buffer
sizes before and after chunks. The class also validates these settings against
the capabilities of the specified embedding model to ensure effective chunking.

Additionally, it provides methods for splitting text into chunks and adding
buffer content around chunks for context preservation.
"""

import re

from typing import List
from pydantic import BaseModel, Field, model_validator

from .chunk import SourceChunk
from frag.embeddings.apis.openai_embed_api import EmbedAPI
from frag.typedefs.llm_settings import ChunkerSettings


class SourceChunker(BaseModel):
    """
    `SourceChunker` class, responsible for
    breaking down text into manageable chunks for embedding.

    It leverages settings defined in `ChunkerSettings` to customize the
    chunking process, such as preserving paragraphs or sentences, and setting buffer
    sizes before and after chunks. The class also validates these settings against
    the capabilities of the specified embedding model to ensure effective chunking.

    Additionally, it provides methods for splitting text into chunks and adding
    buffer content around chunks for context preservation.
    """

    settings: ChunkerSettings = Field(
        default_factory=ChunkerSettings,
        description="The settings for the chunking process",
    )
    embed_api: EmbedAPI = Field(..., description="The embedding model to use")
    buffered_max_tokens: int = Field(
        default=0,
        description="The maximum number of tokens allowed in a chunk, adjusted for\
             buffer settings",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_settings(cls, values):
        """
        Validates the chunk settings against the embedding model's capabilities.

        Args:
            values (dict): The initial values for the model.

        Returns:
            dict: The validated values with adjusted `buffered_max_tokens`.

        Raises:
            ValueError: If buffer settings are invalid or the total buffer size exceeds
            the model's max tokens.
        """
        settings: ChunkerSettings = values.get("settings")
        embed_api: EmbedAPI = values.get("embed_api")
        buffered_max_tokens = 0
        if settings and embed_api:
            if settings.buffer_before < 0 or settings.buffer_after < 0:
                raise ValueError(
                    "buffer_before and buffer_after must be greater than 0"
                )
            buffered_max_tokens = embed_api.max_tokens - (
                settings.buffer_before + settings.buffer_after
            )
            if buffered_max_tokens <= 0:
                raise ValueError(
                    f"The available tokens must be greater than the sum of buffer_before and \
                      buffer_after.\n\nRequired: \
                      {settings.buffer_before + settings.buffer_after}, but got: \
                      {embed_api.max_tokens}"
                )
        return {
            **values,
            "buffered_max_tokens": buffered_max_tokens,
            "settings": settings,
            "embed_api": embed_api,
        }

    def _split_text_into_chunks(self, text: str) -> List[List[int]]:
        """
        Splits the given text into chunks based on the embedding model's token limits
        and buffer settings.

        Args:
            text (str): The text to be chunked.

        Returns:
            List[List[int]]: A list of token lists, each representing a chunk.
        """
        tokens = self.embed_api.encode(text)
        max_tokens = self.embed_api.max_tokens
        chunk_size = max_tokens - (
            self.settings.buffer_before + self.settings.buffer_after
        )

        token_chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            token_chunks.append(chunk_tokens)

        return token_chunks

    def _add_chunk(
        self,
        unit: str,
        current_chunk_tokens: List[int],
        chunks: List[SourceChunk],
        max_tokens: int,
    ):
        """
        Adds a chunk of tokens to the list of chunks, handling buffer zones and splitting
        large units.

        Args:
            unit (str): The text unit to be chunked.
            current_chunk_tokens (List[int]): The current list of tokens in the chunk being built.
            chunks (List[dict]): The list of chunks built so far.
            max_tokens (int): The maximum number of tokens allowed in a chunk.

        Returns:
            Tuple[List[int], List[dict]]: The updated current chunk tokens and chunks list.
        """
        unit_tokens = self.embed_api.encode(unit)
        if len(unit_tokens) > max_tokens:
            unit_chunks = self._split_text_into_chunks(unit)
            for i, chunk_tokens in enumerate(unit_chunks):
                before_buffer = (
                    self.embed_api.decode(
                        unit_chunks[i - 1][-self.settings.buffer_before :]  # noqa
                    )
                    if i > 0
                    else ""
                )
                after_buffer = (
                    self.embed_api.decode(
                        unit_chunks[i + 1][: self.settings.buffer_after]
                    )
                    if i < len(unit_chunks) - 1
                    else ""
                )
                chunk = self._create_chunk(chunk_tokens, before_buffer, after_buffer)
                chunks.append(chunk)
        else:
            if len(current_chunk_tokens) + len(unit_tokens) <= max_tokens:
                current_chunk_tokens.extend(unit_tokens)
            else:
                chunks.append(self._create_chunk(current_chunk_tokens))
                current_chunk_tokens = unit_tokens

        return current_chunk_tokens, chunks

    def _create_chunk(
        self, tokens: List[int], before_buffer: str = "", after_buffer: str = ""
    ) -> SourceChunk:
        """
        Creates a chunk with optional before and after buffer content.

        Args:
            tokens (List[int]): The list of tokens for the chunk.
            before_buffer (str, optional): The text to prepend for context. Defaults to ''.
            after_buffer (str, optional): The text to append for context. Defaults to ''.

        Returns:
            dict: A dictionary representing the chunk with text and buffer content.
        """
        return SourceChunk(
            text=self.embed_api.decode(tokens), before=before_buffer, after=after_buffer
        )

    def chunk_text(self, text: str) -> List[SourceChunk]:
        """
        Breaks down the given text into chunks based on the chunking
        settings and embedding model's capabilities.

        Args:
            text (str): The text to be chunked.

        Returns:
            List[SourceChunk]: A list of `SourceChunk` objects representing the chunked text.
        """
        chunks = []
        if text == "":
            return chunks

        max_tokens = self.buffered_max_tokens
        if self.settings.preserve_paragraphs:
            text_units = re.split(r"\n\s*\n", text)
        elif self.settings.preserve_sentences:
            text_units = re.split(r"(?<=[.!?])\s+", text)
        else:
            text_units = [text]

        current_chunk_tokens = []
        for unit in text_units:
            current_chunk_tokens, chunks = self._add_chunk(
                unit, current_chunk_tokens, chunks, max_tokens
            )

        if current_chunk_tokens:
            chunks.append(self._create_chunk(current_chunk_tokens))

        return chunks
