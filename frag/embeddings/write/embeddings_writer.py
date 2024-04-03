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
from datetime import date
from typing import List, Annotated, Type

from pydantic import BaseModel, Field, computed_field, model_validator

import chromadb
from regex import P

from frag.embeddings.embedding_model import EmbeddingModel, OpenAiEmbeddingModel

from frag.embeddings.source_chunk import SourceChunk
from frag.embeddings.Chunk import Chunk
from frag.embeddings.embeddings_metadata import ChunkMetadata, Metadata

from .source_chunker import ChunkingSettings, SourceChunker

logger = logging.getLogger(__name__)
class EmbeddingsWriter(BaseModel):
    """
    A class for writing embeddings to a vector db using various models.
    """
    database_path: Annotated[str, Field(os.path.join(os.path.dirname(__file__), "chroma_db"), description="Path to the Chroma database")]
    chunker: SourceChunker = Field(..., description="Chunker for the embeddings")
    embeddings_source: Type[EmbeddingModel]|str = Field(..., description="Embedding Source")
    chroma_collection: chromadb.Collection = Field(..., description="Chroma client for the database")
    embedding_model: EmbeddingModel = Field(..., description="Embedding model")

    @model_validator(mode='before')
    def validate_settings(cls, values):
        embeddings_source = values["embeddings_source"]
        settings: ChunkingSettings = values.get('chunking_settings')

        if settings and embeddings_source:
            if (settings.buffer_before < 0 or settings.buffer_after < 0):
                raise ValueError(f"buffer_before and buffer_after must be greater than 0")
            if isinstance(embeddings_source, str):
                embedding_model = OpenAiEmbeddingModel(embeddings_source)
            else:
                embedding_model = embeddings_source()

            if not hasattr(embedding_model, 'max_tokens'):
                raise ValueError(f"Embedding model {embeddings_source} does not have a max_tokens attribute")

            buffered_max_tokens = embedding_model.max_tokens - (settings.buffer_before + settings.buffer_after)
            if buffered_max_tokens <= 0:
                raise ValueError(f"The available tokens must be greater than the sum of buffer_before and buffer_after.\n\nRequired: {settings.buffer_before + settings.buffer_after}, but got: {embedding_model.max_tokens}")
            chunker = SourceChunker(settings=settings, embedding_model=embedding_model)
            values = {
                **values,
                "buffered_max_tokens": buffered_max_tokens,
                "embedding_model": embedding_model,
                "chunker": chunker
            }
            return values
        else:
            raise ValueError("Embedding model and chunking settings must be provided")

    @model_validator(mode='before')
    def validate_chroma_client(cls, values):
        database_path: str = values.get('database_path')
        collection_name: str = values.get('collection_name')
        if not collection_name:
            collection_name = 'default_collection'
            logging.warning(f"Collection name not provided, using default collection name: {collection_name}")
        if not os.path.exists(database_path):
            os.makedirs(database_path, exist_ok=True)
        if not os.access(database_path, os.W_OK):
            raise ValueError(f"Database path {database_path} is not writable")
        try:
            client = chromadb.PersistentClient(path=database_path)
        except Exception as e:
            logger.error(f"Error creating chromadb client: {e}")
            raise e
        try:
            collection = client.get_or_create_collection(collection_name)
        except Exception as e:
            logger.error(f"Error creating chromadb collection: {e}")
            raise e
        values['chroma_collection'] = collection  # Assign the collection instead of validating it as a dictionary
        return values          
      
    @computed_field
    def name(self) -> str:
        """Returns the name of the embedding model."""
        return self.embedding_model.name

    def fetch_embedding(self, text: str) -> List[float]:
        """Returns the embedding vector for the given text."""
        return self.embedding_model.embed(text)
            

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
        result = self.chroma_collection.get(ids=chunk_id)[0]
        if not metadata.model_validate(result.metadata):
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