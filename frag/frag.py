"""
fRAG: Main class for the fRAG library
"""

from frag.embeddings.embedding_store import EmbeddingStore
from frag.embeddings.read.embed_reader import EmbeddingsReader
from frag.embeddings.write.embed_writer import EmbedWriter

from frag.settings import Settings
from frag.settings.settings import SettingsDict


class Frag:
    """
    Main class for the fRAG library
    """

    def __init__(self, settings: Settings | SettingsDict | None = None) -> None:
        if settings is None:
            settings = Settings.from_default()
        if isinstance(settings, SettingsDict):
            settings = Settings.from_dict(settings)

        self.settings = settings
        self.embedding_store = self._create_embedding_store()

    def _create_embedding_store(self):

        return EmbeddingStore(
            db_path=self.settings.db.db_path,
            collection_name=self.settings.db.default_collection,
            embed_settings=self.settings.embed,
            chunker_settings=self.settings.chunker,
        )

    def writer(self):
        """
        Returns an EmbedWriter instance
        """
        return EmbedWriter(
            store=self.embedding_store, chunker=self.embedding_store.chunker
        )

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
