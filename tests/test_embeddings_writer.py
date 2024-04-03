import pytest
import chromadb
import uuid
from chromadb import Collection
from frag.embeddings.Chunk import Chunk
from frag.embeddings.embeddings_metadata import Metadata
from frag.embeddings.write.embeddings_writer import EmbeddingsWriter
from frag.embeddings.write.source_chunker import ChunkingSettings
from .utils import TestEmbeddingModel

@pytest.fixture
def embeddings_writer(tmpdir):
    return EmbeddingsWriter(
        database_path=str(tmpdir.join("test_db")),
        chunking_settings=ChunkingSettings(),
        embeddings_source=TestEmbeddingModel,
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

