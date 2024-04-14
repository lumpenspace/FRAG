import pytest
import uuid

from frag.embeddings.embedding_store import EmbeddingStore

from frag.embeddings.embeddings_metadata import Metadata
from frag.embeddings.write.embeddings_writer import EmbeddingsWriter
from frag.embeddings.write.source_chunker import ChunkSettings

from tests.utils import EmbedAPITest

@pytest.fixture
def embeddings_writer(tmpdir):
    store = EmbeddingStore(
        path=str(tmpdir.join("test_db")),
        chunk_settings=ChunkSettings(),
        embed_api=EmbedAPITest,
        collection_name = f"test_collection_{uuid.uuid4()}"  
    )
    return EmbeddingsWriter(store=store)

def test_create_embeddings_for_document(embeddings_writer:EmbeddingsWriter):
    text = "This is a sample document."
    metadata = Metadata(
        title="title",
        source="test",
        source_id="1",
        url="https://example.com",
    )

    chunks = embeddings_writer.create_embeddings_for_document(text, metadata)
    # Assert that the embeddings are stored in the database
    results = embeddings_writer.store.get(ids=[ chunk.id for chunk in chunks])
    assert len(results) > 0
    assert results.get('documents')[0] == chunks[0].text

def test_metadata_serialization_excludes_none():
    # Create a Metadata instance with some fields set to None
    metadata = Metadata(title="Test Document", url=None, author="Test Author", publish_date=None)
    
    # Serialize the metadata and exclude None values
    serialized_metadata = metadata.model_dump(exclude_none=True)
    
    # Assert that the serialized metadata does not contain keys with None values
    assert 'url' not in serialized_metadata
    assert 'publish_date' not in serialized_metadata
    assert 'title' in serialized_metadata
    assert 'author' in serialized_metadata
    assert serialized_metadata['title'] == "Test Document"
    assert serialized_metadata['author'] == "Test Author"