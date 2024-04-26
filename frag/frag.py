

import os

from pydantic_settings import BaseSettings

from frag.embeddings.chunks.source_chunker import SourceChunker
from frag.embeddings.embedding_store import EmbeddingStore
from frag.embeddings.read.embed_reader import EmbeddingsReader
from frag.embeddings.write.embed_writer import EmbedWriter

from frag.settings.settings import Settings

class Frag:
    def __init__(self, settings: Settings = None):
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.embedding_store = self._create_embedding_store()

    def _create_embedding_store(self):
        chunk_settings = {
            "preserve_paragraphs": self.settings.preserve_paragraphs,
            "token_limit": self.settings.token_limit,
            "buffer_before": self.settings.buffer_before,
            "buffer_after": self.settings.buffer_after,
        }

        embedding_settings = {
            "model": self.settings.embed_model,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
        }

        return EmbeddingStore(
            path=self.settings.db_path,
            collection_name=self.settings.default_collection,
            embed_api=embedding_settings["model"],
            chunk_settings=chunk_settings,
        )

    def writer(self):
        """
        Returns an EmbedWriter instance
        """
        return EmbedWriter(store=self.embedding_store, chunker=self.embedding_store.chunker)

    def chunker(self):
        """
        Returns a SourceChunker instance
        """
        return self.embedding_store.chunker

    def reader(self):
        """
        Returns an EmbeddingsReader instance
        """
        return EmbeddingsReader(store=self.embedding_store)
