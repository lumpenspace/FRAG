from curses import meta
from pydantic import BaseModel, Field
import hashlib

from frag.embeddings.embeddings_metadata import ChunkMetadata
from frag.embeddings.write.source_chunker import SourceChunk

from frag.embeddings.embeddings_metadata import Metadata

class Chunk(BaseModel):
    text: str = Field(..., description="Text of the chunk")
    metadata: Metadata = Field(..., description="Metadata of the chunk")
    id: str = Field(..., description="ID of the chunk")

    @classmethod
    def make_id(cls, text: str, metadata: Metadata, part: int) -> str:
        text_hash = hashlib.sha256((text).encode('utf-8')).hexdigest()
        return f"{metadata.title}::{part}:{metadata.parts}::{text_hash[:8]}"

    @classmethod
    def from_source_chunk(cls, source_chunk: SourceChunk, metadata: Metadata, part: int):
        # separate extra metadata from rest of metadata,
        # and add the rest of metadata to the chunk metadata
        metadata.part = part
        return Chunk(
            text=source_chunk.text,
            metadata=metadata,
            id=Chunk.make_id(text=source_chunk.text, metadata=metadata, part=part)
        )

