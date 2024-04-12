"""
This module contains the `SourceChunker` class, which is responsible for
breaking down text into manageable chunks for embedding.

It leverages settings defined in `ChunkingSettings` to customize the
chunking process, such as preserving paragraphs or sentences, and setting buffer
sizes before and after chunks. The class also validates these settings against
the capabilities of the specified embedding model to ensure effective chunking.

Additionally, it provides methods for splitting text into chunks and adding
buffer content around chunks for context preservation.
"""

import re

from typing import List
from pydantic import BaseModel, Field, model_validator

from frag.embeddings.Chunk import SourceChunk
from frag.embeddings.apis.openai_embed_api import EmbedAPI

class ChunkingSettings(BaseModel):
    """
    Settings for the chunking process.
    """
    preserve_paragraphs: bool = Field(False, description="Whether to preserve paragraphs in the embedding process")
    preserve_sentences: bool = Field(True, description="Whether to preserve sentences in the embedding process")
    buffer_before: int = Field(10, description="The number of tokens to buffer before the current sentence")
    buffer_after: int = Field(10, description="The number of tokens to buffer after the current sentence")

class SourceChunker(BaseModel):
    """
    A class for chunking text into manageable pieces for embedding.
    """
    settings: ChunkingSettings = Field(default_factory=ChunkingSettings, description="The settings for the chunking process")
    embedding_model: EmbedAPI = Field(..., description="The embedding model to use")
    buffered_max_tokens: int = Field(0, description="Maximum tokens per chunk, adjusted for buffers")

    @model_validator(mode='before')
    def validate_settings(cls, values: dict):
        settings: ChunkingSettings = values.get("settings")
        embedding_model: EmbedAPI = values.get("embedding_model")
        if settings and embedding_model:
            if (settings.buffer_before < 0 or settings.buffer_after < 0):
                raise ValueError(f"buffer_before and buffer_after must be greater than 0")
            buffered_max_tokens = embedding_model.max_tokens - (settings.buffer_before + settings.buffer_after)
            if buffered_max_tokens <= 0:
                raise ValueError(f"The available tokens must be greater than the sum of buffer_before and buffer_after.\n\nRequired: {settings.buffer_before + settings.buffer_after}, but got: {embedding_model.max_tokens}")
        return {
            **values,
            "buffered_max_tokens": buffered_max_tokens,
            "settings": settings,
            "embedding_model": embedding_model
        }


    def _split_text_into_chunks(self, text: str) -> List[List[int]]:
        tokens = self.embedding_model.tokenize(text)
        max_tokens = self.embedding_model.max_tokens
        chunk_size = max_tokens - (self.settings.buffer_before + self.settings.buffer_after)
        
        token_chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            token_chunks.append(chunk_tokens)
        
        return token_chunks

    def _add_chunk(self, unit: str, current_chunk_tokens: List[int], chunks: List[dict], max_tokens: int):
        unit_tokens = self.embedding_model.tokenize(unit)
        if len(unit_tokens) > max_tokens:
            unit_chunks = self._split_text_into_chunks(unit)
            for i, chunk_tokens in enumerate(unit_chunks):
                before_buffer = self.embedding_model.decode(unit_chunks[i-1][-self.settings.buffer_before:]) if i > 0 else ''
                after_buffer = self.embedding_model.decode(unit_chunks[i+1][:self.settings.buffer_after]) if i < len(unit_chunks) - 1 else ''
                chunks.append(self._create_chunk(chunk_tokens, before_buffer, after_buffer))
        else:
            if len(current_chunk_tokens) + len(unit_tokens) <= max_tokens:
                current_chunk_tokens.extend(unit_tokens)
            else:
                chunks.append(self._create_chunk(current_chunk_tokens))
                current_chunk_tokens = unit_tokens

        return current_chunk_tokens, chunks

    def _create_chunk(self, tokens: List[int], before_buffer: str = '', after_buffer: str = '') -> dict:
        return SourceChunk(
            text=self.embedding_model.decode(tokens),
            before=before_buffer,
            after=after_buffer
        )

    def chunk_text(self, text: str) -> List[SourceChunk]:
        """
        Breaks down the given text into chunks based on the chunking 
        settings and embedding model's capabilities.
        """
        chunks = []
        if text == "":
            return chunks

        max_tokens = self.buffered_max_tokens
        if self.settings.preserve_paragraphs:
            text_units = re.split(r'\n\s*\n', text)
        elif self.settings.preserve_sentences:
            text_units = re.split(r'(?<=[.!?])\s+', text)
        else:
            text_units = [text]

        current_chunk_tokens = []
        for unit in text_units:
            current_chunk_tokens, chunks = self._add_chunk(unit, current_chunk_tokens, chunks, max_tokens)

        if current_chunk_tokens:
            chunks.append(self._create_chunk(current_chunk_tokens))

        return chunks