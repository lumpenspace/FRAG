"""
fRAG: Main class for the fRAG library
"""

from frag.embeddings.store import EmbeddingStore
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

    def _create_embedding_store(self) -> EmbeddingStore:

        return EmbeddingStore(
            db_path=self.settings.db.db_path,
            collection_name=self.settings.db.default_collection,
            embed_settings=self.settings.embed,
            chunker_settings=self.settings.chunker,
        )
