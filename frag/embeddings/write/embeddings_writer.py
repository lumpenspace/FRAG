"""
This module contains the `EmbeddingsWriter` class, which is responsible for
writing embeddings to a vector database using various models.

It handles the creation, storage, and management of embeddings,
leveraging the Chroma database for persistence.

The class also provides functionality for chunking text documents,
fetching embeddings from models, and updating or deleting embeddings in the database.
"""

import os
import logging
import chromadb
from typing import List, Type

from pydantic import Field, computed_field, model_validator

from frag.embeddings.embed_api import EmbedAPI

from frag.embeddings.embedding_store import EmbeddingStore
from frag.embeddings.source_chunk import SourceChunk
from frag.embeddings.Chunk import Chunk
from frag.embeddings.embeddings_metadata import Metadata

from .source_chunker import SourceChunker

logger = logging.getLogger(__name__)
class EmbeddingsWriter(EmbeddingStore):
    """
    A class for writing embeddings to a vector db using various models.
    """
    chunker: SourceChunker = Field(..., description="Chunker for the embeddings")
    embeddings_source: Type[EmbedAPI]|str = Field(..., description="Embedding Source")
    embedding_model: EmbedAPI = Field(..., description="Embedding API client. Create one with `$ frag api name` or use `$ frag api help`")
            

    def create_embeddings_for_document(self, text: str, metadata: Metadata):
        chunks = []
        source_chunks = self.chunker.chunk_text(text)
        metadata.parts = len(source_chunks)
        for index, source_chunk in enumerate(source_chunks):
            ## TODO: Add infor anbout the parts and current part
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
            collection_result = self.chroma_collection.get(ids=chunk.id)
                
            if collection_result.get('id'):
                logging.info("Embedding found in db")
                return collection_result[0]

            logging.info("getting embeddings") 
            embedding = self.fetch_embedding(chunk.text)
            self.chroma_collection.add(
                ids=chunk.id,
                embeddings=embedding,
                documents=chunk.text,
                metadatas=chunk.metadata.model_dump(exclude_none=True)  # Use the serialized metadata with None values excluded
            )

            return embedding
        except Exception as e:
            raise e

            
    def update_metadata(self, chunk_id: str, metadata: Metadata) -> bool:
        """
        Updates the metadata for an embedding in the Chroma database.
        """
        # if the current metadata has a different schema, throw an error.
        result = self.chroma_collection.get(ids=['chunk_id']).get(chunk_id)
        if result.metadata and not metadata.model_validate(result.metadata):
            raise ValueError(f"The metadata schema is different from the current metadata schema")
        self.chroma_collection.update(ids=chunk_id, metadatas=metadata.model_dump())
        return True

    def delete_embedding(self, chunk_id: str) -> bool:
        """
        Deletes an embedding from the Chroma database based on the chunk ID.

        Parameters:
            chunk_id: The ID of the chunk whose embedding is to be deleted.

        Returns:
            A boolean indicating whether the deletion was successful.
        """
        try:
            delete_result = self.chroma_collection.delete(ids=[chunk_id])
            if delete_result:
                logging.info(f"Successfully deleted embedding with ID: {chunk_id}")
                return True
            else:
                logging.warning(f"Failed to delete embedding with ID: {chunk_id}")
                return False
        except Exception as e:
            logging.error(f"Error deleting embedding: {e}")
            return False