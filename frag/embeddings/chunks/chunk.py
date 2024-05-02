from pydantic import BaseModel, Field

import hashlib

from frag.embeddings.embeddings_metadata import ChunkInfo, ChunkMetadata


class SourceChunk(BaseModel):
    """
    Represents a chunk of source text, including the text itself and the context around it.

    Attributes:
        text (str): Text of the chunk.
        before (str): Text before the chunk.
        after (str): Text after the chunk.

    Example:
        >>> source_chunk = SourceChunk(
            text="Hello, world!",
            before="Greetings: ",
            after=" That's all.")
        >>> print(source_chunk)
    """

    text: str = Field(..., description="Text of the chunk")
    before: str = Field(..., description="Text before the chunk")
    after: str = Field(..., description="Text after the chunk")


class Chunk(BaseModel):
    """
    Represents a chunk with its text, metadata, and a unique identifier.

    Attributes:
        text (str): Text of the chunk.
        metadata (Metadata): Metadata of the chunk, including title and parts.
        id (str): Unique identifier of the chunk, generated based on text and metadata.

    Example:
        >>> metadata = Metadata(title="Example", parts=1)
        >>> chunk = Chunk.from_source_chunk(
            source_chunk=SourceChunk(
                text="Hello, world!",
                before="Greetings: ",
                after=" That's all."),
            metadata=metadata, part=1
        )
        >>> print(chunk.id)
    """

    text: str = Field(..., description="Text of the chunk")
    metadata: ChunkInfo = Field(..., description="Metadata of the chunk")
    id: str = Field(..., description="ID of the chunk")

    @classmethod
    def make_id(cls, text: str, metadata: ChunkMetadata, part: int) -> str:
        """
        Generates a unique identifier for a chunk based on its text, metadata, and part number.

        Parameters:
            text (str): Text of the chunk.
            metadata (Metadata): Metadata associated with the chunk.
            part (int): Part number of the chunk.

        Returns:
            str: A unique identifier for the chunk.
        """
        text_hash = hashlib.sha256((text).encode("utf-8")).hexdigest()
        return f"{metadata.title}::{part}:{metadata.parts}::{text_hash[:8]}"

    @classmethod
    def from_source_chunk(
        cls, source_chunk: SourceChunk, metadata: ChunkMetadata, part: int
    ):
        """
        Creates a Chunk instance from a SourceChunk instance,
        incorporating additional metadata and generating a unique ID.

        Parameters:
            source_chunk (SourceChunk): The source chunk to convert.
            metadata (Metadata): Metadata to associate with the chunk.
            part (int): Part number of the chunk.

        Returns:
            Chunk: A new Chunk instance with the given text, metadata, and a unique ID.
        """
        metadata.part = part
        return Chunk(
            text=source_chunk.text,
            metadata=metadata,
            id=Chunk.make_id(text=source_chunk.text, metadata=metadata, part=part),
        )
