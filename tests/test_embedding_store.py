from pydantic import ValidationError
import pytest
import uuid

from frag.embeddings.embedding_store import EmbeddingStore
from frag.embeddings.chunks import ChunkSettings

from tests.utils import EmbedAPITest

def test_path_validation(tmpdir):
  print(tmpdir.join("test_db"))
  store = EmbeddingStore(
    path=str(tmpdir.join("test_db")),
    chunk_settings=ChunkSettings(),
    embed_api=EmbedAPITest,
    collection_name = f"test_collection_{uuid.uuid4()}"  
  )

def test_settings_validation():
    with pytest.raises(ValidationError):
        EmbeddingStore(embed_api="oai:text-embedding-small", chunk_settings={'buffer_before': -1, 'buffer_after': 2})