import pytest
from unittest.mock import patch
from frag.frag import Frag
from frag.settings.settings import Settings
from frag.embeddings.write.embed_writer import EmbedWriter
from frag.embeddings.read.embed_reader import EmbeddingsReader

def test_initialization_with_default_settings():
    with patch('frag.settings.settings.Settings') as mock_settings:
        mock_settings.return_value = Settings()
        frag_instance = Frag()
        assert isinstance(frag_instance, Frag)
        assert isinstance(frag_instance.settings, Settings)

def test_initialization_with_custom_settings():
    custom_settings = {
        'db': { 
          'db_path': 'custom_db_path',
          'default_collection': 'custom_collection'
        },
        'chunker': {
          'preserve_paragraphs': True,
          'buffer_before': 10,
          'buffer_after': 10,
          'max_length': 1000
        },
        'embed': {
          'model': 'oai:text-embedding-ada-002',
        }
    }
    frag_instance = Frag(settings=custom_settings)
    assert frag_instance.settings.db_path == 'custom_db_path'
    assert frag_instance.settings.default_collection == 'custom_collection'
    assert frag_instance.settings.preserve_paragraphs is True
    assert frag_instance.settings.max_length == 1000
    assert frag_instance.settings.buffer_before == 10
    assert frag_instance.settings.buffer_after == 10
    assert frag_instance.settings.embed_model == 'oai:text-embedding-ada-002'

def test_writer_method():
    frag_instance = Frag()
    writer = frag_instance.writer()
    assert isinstance(writer, EmbedWriter)

def test_chunker_method():
    frag_instance = Frag()
    chunker = frag_instance.chunker()
    assert chunker == frag_instance.embedding_store.chunker

def test_reader_method():
    frag_instance = Frag()
    reader = frag_instance.reader()
    assert isinstance(reader, EmbeddingsReader)
