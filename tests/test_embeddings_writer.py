import pytest
import chromadb
import uuid
from chromadb import Collection
from frag.embeddings.Chunk import Chunk
from frag.embeddings.embeddings_metadata import Metadata
from frag.embeddings.write.embeddings_writer import EmbeddingsWriter
from frag.embeddings.write.source_chunker import ChunkingSettings
from .utils import EmbeddingModelTest

@pytest.fixture
def embeddings_writer(tmpdir):
    return EmbeddingsWriter(
        database_path=str(tmpdir.join("test_db")),
        chunking_settings=ChunkingSettings(),
        embeddings_source=EmbeddingModelTest,
        collection_name = f"test_collection_{uuid.uuid4()}"
    )

def test_create_embeddings_for_document(embeddings_writer: EmbeddingsWriter):
    text = "This is a sample document."
    metadata = Metadata(
        title="title",
        source="test",
        source_id="1",
        url="https://example.com",
    )

    chunks = embeddings_writer.create_embeddings_for_document(text, metadata)

    # Assert that the embeddings are stored in the database
    results = embeddings_writer.chroma_collection.get(ids=[ chunk.id for chunk in chunks])
    assert len(results) > 0
    assert results.get('documents')[0] == chunks[0].text

from datetime import datetime
from frag.embeddings.embeddings_metadata import Metadata, ChunkMetadata

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