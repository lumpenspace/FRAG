import pytest
import uuid

from frag.embeddings.embedding_store import EmbeddingStore
from frag.embeddings.write.source_chunker import ChunkingSettings

from tests.utils import EmbedAPITest

def test_path_validation(tmpdir):
  store = EmbeddingStore(
    path=str(tmpdir.join("test_db")),
    chunking_settings=ChunkingSettings(),
    embeddings_api=EmbedAPITest,
    collection_name = f"test_collection_{uuid.uuid4()}"  
  )

def test_settings_validation():
    with pytest.raises(TypeError):
        EmbeddingStore(embeddings_api="OpenAI", chunking_settings={'buffer_before': -1, 'buffer_after': 2})