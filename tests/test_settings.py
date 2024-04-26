# pylint: disable=[missing-docstring,W0621:redefined-outer-name]

from unittest.mock import mock_open, patch
from frag.settings.settings import Settings

# Sample content of a .fragrc file
SAMPLE_FRAGRC = """
db:
  db_path: ./custom_db
  default_collection: custom_collection
chunker:
  preserve_paragraphs: false
  token_limit: 1024
  buffer_before: 1
  buffer_after: 1
embed:
  model: custom-model
  chunk_size: 5000
  chunk_overlap: 100
"""

def test_load_settings_from_fragrc():
    with patch("builtins.open", mock_open(read_data=SAMPLE_FRAGRC)):
        with patch("os.path.isfile", return_value=True):
            settings = Settings()
            assert settings.db.db_path == "./custom_db"
            assert settings.db.default_collection == "custom_collection"
            assert settings.chunker.preserve_paragraphs is False
            assert settings.chunker.token_limit == 1024
            assert settings.chunker.buffer_before == 1
            assert settings.chunker.buffer_after == 1
            assert settings.embed.model == "custom-model"
            assert settings.embed.chunk_size == 5000
            assert settings.embed.chunk_overlap == 100

def test_default_settings_when_no_fragrc():
    with patch("os.path.isfile", return_value=False):
        settings = Settings()
        assert settings.db.db_path.split('/')[-1] == "db"
        assert settings.db.default_collection == "default_collection"
        assert settings.chunker.preserve_paragraphs is True
        assert settings.chunker.token_limit == 512
        assert settings.chunker.buffer_before == 0
        assert settings.chunker.buffer_after == 0
        assert settings.embed.model == "text-embedding-3-large"
        assert settings.embed.chunk_size == 10000
        assert settings.embed.chunk_overlap == 0