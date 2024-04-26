import os
import yaml

from pydantic_settings import BaseSettings

class DBSettings(BaseSettings):
    """
    Settings for the database
    """
    db_path: str = '/db'
    default_collection: str = "default_collection"

class EmbedSettings(BaseSettings):
    """
    Settings for the embeddings
    """
    model: str = "text-embedding-3-large"
    chunk_size: int = 10000
    chunk_overlap: int = 0

class ChunkerSettings(BaseSettings):
    """
    Settings for the chunker
    """
    preserve_sentences: bool = False
    preserve_paragraphs: bool = True
    token_limit: int = 512
    buffer_before: int = 0
    buffer_after: int = 0

class Settings(BaseSettings):
    """
    Settings for the Frag application
    """
    db: DBSettings = DBSettings()
    embed: EmbedSettings = EmbedSettings()
    chunker: ChunkerSettings = ChunkerSettings()


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_fragrc_settings()

    def load_fragrc_settings(self):
        """
        Load settings from .fragrc file if it exists in the directory of frag.py or any parent directories.
        """
        current_dir = os.path.dirname(__file__)
        while True:
            fragrc_path = os.path.join(current_dir, '.fragrc')
            if os.path.isfile(fragrc_path):
                with open(fragrc_path, 'r', encoding='utf-8') as file:
                    fragrc_settings = yaml.safe_load(file)
                    for section, settings in fragrc_settings.items():
                        for key, value in settings.items():
                            if hasattr(self, section):
                                setattr(getattr(self, section), key, value)
                break
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # reached the root directory
                break
            current_dir = parent_dir