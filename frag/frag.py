"""
fRAG: Main class for the fRAG library
"""

from frag.embeddings.embedding_store import EmbeddingStore
from frag.embeddings.read.embed_reader import EmbeddingsReader
from frag.embeddings.write.embed_writer import EmbedWriter

from frag.settings.settings import Settings

class Frag:
    """
    Main class for the fRAG library
    """
    def __init__(self, settings: Settings|dict = None):
        if settings is None:
            settings = Settings()
        if isinstance(settings, dict):
            settings = Settings(**settings)

        self.settings = settings
        self.embedding_store = self._create_embedding_store()

    def _create_embedding_store(self):
        chunk_settings = {
            "preserve_paragraphs": self.settings.chunker.preserve_paragraphs,
            "max_length": self.settings.chunker.max_length,
            "buffer_before": self.settings.chunker.buffer_before,
            "buffer_after": self.settings.chunker.buffer_after,
        }

        embedding_settings = {
            "model": self.settings.embed.model,
            "chunk_size": self.settings.embed.chunk_size,
            "chunk_overlap": self.settings.embed.chunk_overlap,
        }

        return EmbeddingStore(
            db_path=self.settings.db.db_path,
            collection_name=self.settings.db.default_collection,
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
