from frag.embeddings import SourceChunk, Chunk, DBChunk, ChunkInfo, ChunkMetadata


def test_source_chunk_creation():
    text = "Sample text"
    before = "Before text"
    after = "After text"
    source_chunk = SourceChunk(text=text, before=before, after=after)
    assert source_chunk.text == text
    assert source_chunk.before == before
    assert source_chunk.after == after


def test_chunk_creation():
    text = "Chunk text"
    metadata = ChunkInfo(
        title="Test Title",
        url="http://chunkurl.com",
        author="Test Author",
        publish_date="2023-01-01",
    )
    chunk = Chunk(text=text, metadata=metadata, id="12345")
    assert chunk.text == text
    assert chunk.metadata == metadata
    assert chunk.id == "12345"


def test_id_generation():
    text = "Chunk text for ID generation"
    metadata = ChunkMetadata(
        title="ID Test",
        parts=1,
        part=1,
        before="Before",
        after="After",
        extra_metadata={},
        url="http://chunkurl.com",
        author="Test Author",
        publish_date="2023-01-01",
    )
    part = 1
    generated_id = Chunk.make_id(text=text, metadata=metadata, part=part)
    assert isinstance(generated_id, str)
    assert len(generated_id) > 0


def test_from_source_chunk():
    source_chunk = SourceChunk(text="Source chunk text", before="Before", after="After")
    metadata = ChunkMetadata(
        title="From Source",
        url="http://chunkurl.com",
        author="Test Author",
        publish_date="2023-01-01",
        parts=1,
        part=1,
        before="Before",
        after="After",
        extra_metadata={},
    )
    part = 1
    chunk = Chunk.from_source_chunk(
        source_chunk=source_chunk, metadata=metadata, part=part
    )
    assert chunk.text == source_chunk.text
    assert chunk.metadata == metadata
    assert isinstance(chunk.id, str)
    assert len(chunk.id) > 0


def test_from_db_result():
    db_result = {
        "documents": [["Chunk text"]],
        "metadatas": [[{"title": "From DB", "parts": 1, "part": 1}]],
        "ids": [["12345"]],
        "distances": [[0.5]],
    }
    chunk = DBChunk.from_db_results(**db_result)
    assert chunk[0].text == db_result["documents"][0][0]
    assert chunk[0].metadata == db_result["metadatas"][0][0]
    assert chunk[0].id == db_result["ids"][0][0]
    assert chunk[0].distance == db_result["distances"][0][0]
