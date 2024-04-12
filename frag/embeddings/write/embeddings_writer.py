"""
This module contains the `EmbeddingsWriter` class, which is responsible for
writing embeddings to a vector database using various models.

It handles the creation, storage, and management of embeddings,
leveraging the Chroma database for persistence.

The class also provides functionality for chunking text documents,
fetching embeddings from models, and updating or deleting embeddings in the database.
"""

import logging
from typing import List
from openai import BaseModel

from pydantic import  Field

from frag.embeddings.embedding_store import EmbeddingStore
from frag.embeddings.Chunk import SourceChunk
from frag.embeddings.write.source_chunker import SourceChunker
from frag.embeddings.Chunk import Chunk
from frag.embeddings.embeddings_metadata import Metadata

logger = logging.getLogger(__name__)

class EmbeddingsWriter(BaseModel):
    
    store: EmbeddingStore = Field(default=None)
    chunker: SourceChunker = Field(default=None)

    def create_embeddings_for_document(self, text: str, metadata: Metadata):
        chunks = []
        source_chunks = self.store.chunker.chunk_text(text)
        metadata.parts = len(source_chunks)
        for index, source_chunk in enumerate(source_chunks):
            metadata = metadata.model_copy(update={'part': index + 1})
            chunk = Chunk.from_source_chunk(source_chunk=source_chunk, metadata=metadata, part=index + 1)
            self.fetch_and_store_embedding(chunk)
            chunks.append(chunk)

        return chunks

    def fetch_and_store_embedding(self, chunk: SourceChunk) -> List[float]:
        """
        Stores the embedding in a Chroma database and returns it.

        Parameters:
            chunk: The Chunk object containing text and metadata.

        Returns:
            The embedding vector.
        """
        try:
            collection_result = self.store.get(ids=[chunk.id])
                
            if collection_result.get('id'):
                logging.info("Embedding found in db")
                return collection_result[0]

            logging.info("getting embeddings") 
            embedding = self.store.fetch(chunk.text)
            result = self.store.add(
                ids=chunk.id,
                embeddings=embedding,
                documents=chunk.text,
                metadatas=chunk.metadata.model_dump(exclude_none=True)  # Use the serialized metadata with None values excluded
            )
            return embedding
        except Exception as e:
            raise e
        