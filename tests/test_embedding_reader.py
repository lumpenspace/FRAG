import pytest
import uuid

from frag.embeddings.embedding_store import EmbeddingStore

from frag.embeddings.embeddings_metadata import Metadata
from frag.embeddings.read.embed_reader import EmbeddingsReader
from frag.embeddings.write.embed_writer import EmbedWriter
from frag.types import ChunkerSettings

from tests.utils import EmbedAPITest

@pytest.fixture
def embeddings_reader(tmpdir):
    store = EmbeddingStore(
        db_path=str(tmpdir.join("test_db")),
        chunk_settings=ChunkerSettings(),
        embed_api=EmbedAPITest,
        collection_name = f"test_collection_{uuid.uuid4()}"  
    )
    embeddings_writer = EmbedWriter(store=store)
    text = "This is a sample document."
    metadata = Metadata(
        title="title",
        source="test",
        source_id="1",
        url="https://example.com",
    )

    embeddings_writer.create_embeddings_for_document(text, metadata)

    return EmbeddingsReader(store=store)

def test_get_similar(embeddings_reader:EmbeddingsReader):
    """Test fetching similar embeddings from the store."""
    similar_embeddings = embeddings_reader.get_similar(text="my_text", n_results=1)
    assert similar_embeddings is not None, "Failed to fetch similar embeddings"

