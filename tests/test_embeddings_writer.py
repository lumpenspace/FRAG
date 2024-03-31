import pytest
import chromadb
import uuid
from chromadb import Collection
from frag.embeddings.embeddings_metadata import Chunk, Metadata
from frag.embeddings.write.embeddings_writer import EmbeddingsWriter
from frag.embeddings.write.source_chunker import ChunkingSettings
from .utils import MyEmbeddingModel

@pytest.fixture
def embeddings_writer(tmpdir):
    return EmbeddingsWriter(
        database_path=str(tmpdir.join("test_db")),
        chunking_settings=ChunkingSettings(),
        embedding_model=MyEmbeddingModel,
        collection_name = f"test_collection_{uuid.uuid4()}"
    )

def test_create_embeddings_for_document(embeddings_writer):
    text = "This is a sample document."
    metadata = Metadata(source="test", source_id="1", url="https://example.com", title="title")
    
    item = embeddings_writer.create_embeddings_for_document(text, metadata)
    
    # Assert that the embeddings are stored in the database
    collection = embeddings_writer.client.get_collection(embeddings_writer.collection_name)
    results = collection.get(ids=[item.id])
    assert len(results) > 0
