from pydantic import root_validator
from pydantic_settings import BaseSettings
import os
import yaml

class DBSettings(BaseSettings):
    db_path: str = os.path.join(os.path.dirname(__file__), "db")
    default_collection: str = "default_collection"

class EmbedSettings(BaseSettings):
    model: str = "oai:text-embedding-3-large"
    chunk_size: int = 10000
    chunk_overlap: int = 0

class ChunkerSettings(BaseSettings):
    preserve_sentences: bool = False
    preserve_paragraphs: bool = True
    max_length: int = 512
    buffer_before: int = 0
    buffer_after: int = 0

class Settings(BaseSettings):
    db: DBSettings = DBSettings()
    embed: EmbedSettings = EmbedSettings()
    chunker: ChunkerSettings = ChunkerSettings()

    @property
    def embed_model(self):
        return self.embed.model
    @property
    def max_length(self):
        return self.chunker.max_length
    @property
    def chunk_overlap(self):
        return self.chunker.chunk_overlap
    @property
    def preserve_paragraphs(self):
        return self.chunker.preserve_paragraphs
    @property
    def max_length(self):
        return self.chunker.max_length
    @property
    def buffer_before(self):
        return self.chunker.buffer_before
    @property
    def buffer_after(self):
        return self.chunker.buffer_after
    @property
    def db_path(self):
        return self.db.db_path
    @property
    def default_collection(self):
        return self.db.default_collection

    @root_validator(pre=True)
    def load_default_settings(cls, values):
        """
        Load settings from .fragrc file if it exists and no settings are provided.
        """
        if not values:  # Check if no settings are provided
            current_dir = os.path.dirname(__file__)
            while True:
                fragrc_path = os.path.join(current_dir, '.fragrc')
                if os.path.isfile(fragrc_path):
                    with open(fragrc_path, 'r', encoding='utf-8') as file:
                        fragrc_settings = yaml.safe_load(file)
                        for section, settings in fragrc_settings.items():
                            if section in values:
                                for key, value in settings.items():
                                    if key in values[section]:
                                        values[section][key] = value
                    break
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # reached the root directory
                    break
                current_dir = parent_dir
        return values