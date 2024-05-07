"""
fRAG: Main class for the fRAG library
"""

from pathlib import Path
from frag.embeddings.store import EmbeddingStore
from frag.settings import Settings
from frag.settings.settings import SettingsDict


class Frag:
    """
    Main class for the fRAG library
    """

    def __init__(self, settings: Settings | SettingsDict | str) -> None:
        if isinstance(settings, str):
            if settings.startswith("~"):
                settings = settings.replace("~", str(Path.home()))
            settings = Settings.from_path(settings)
        if isinstance(settings, SettingsDict):
            settings = Settings.from_dict(settings)
        self.settings: Settings = settings
        self.embedding_store: EmbeddingStore = EmbeddingStore.create(
            settings=self.settings.embeds,
        )
