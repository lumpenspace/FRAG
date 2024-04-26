# pylint: disable=[missing-docstring,W0621:redefined-outer-name]

from unittest.mock import mock_open, patch
from frag.settings.settings import Settings


def test_default_settings_when_no_fragrc():
    with patch("os.path.isfile", return_value=False):
        settings = Settings()
        assert settings.db.db_path.split('/')[-1] == "db"
        assert settings.db.default_collection == "default_collection"
        assert settings.chunker.preserve_paragraphs is True
        assert settings.chunker.max_length == 512
        assert settings.chunker.buffer_before == 0
        assert settings.chunker.buffer_after == 0
        assert settings.embed.model == "oai:text-embedding-3-large"
        assert settings.embed.chunk_size == 10000
        assert settings.embed.chunk_overlap == 0