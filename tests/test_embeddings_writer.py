import pytest
import uuid

from frag.embeddings import ChunkInfo, EmbedWriter, EmbeddingStore
from frag.types import ChunkerSettings


from tests.utils import EmbedAPITest


@pytest.fixture
def embeddings_writer(tmpdir):
    """
    Create an EmbedWriter instance with a temporary directory as the store path.
    """
    store = EmbeddingStore(
        db_path=str(tmpdir.join("test_db")),
        chunk_settings=ChunkerSettings(),
        embed_api=EmbedAPITest,
        collection_name=f"test_collection_{uuid.uuid4()}",
    )
    return EmbedWriter(store=store)


def test_create_embeddings_for_document(embeddings_writer):
    """
    Test that embeddings are created for a document and stored in the database.
    """
    text = "This is a sample document."
    metadata = ChunkInfo(
        title="title",
        author="author",
        publish_date="2023-01-01",
        url="https://example.com",
    )

    chunks = embeddings_writer.create_embeddings_for_document(text, metadata)
    # Assert that the embeddings are stored in the database
    results = embeddings_writer.store.get(ids=[chunk.id for chunk in chunks])
    assert len(results) > 0
    assert results.get("documents")[0] == chunks[0].text


def test_metadata_serialization_excludes_none():
    """
    Test that the metadata serialization excludes None values.
    """
    # Create a Metadata instance with some fields set to None
    metadata = ChunkInfo(
        title="Test Document", url=None, author="Test Author", publish_date=None
    )

    # Serialize the metadata and exclude None values
    serialized_metadata = metadata.model_dump(exclude_none=True)

    # Assert that the serialized metadata does not contain keys with None values
    assert "url" not in serialized_metadata
    assert "publish_date" not in serialized_metadata
    assert "title" in serialized_metadata
    assert "author" in serialized_metadata
    assert serialized_metadata["title"] == "Test Document"
    assert serialized_metadata["author"] == "Test Author"
