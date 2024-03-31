from http import client
import os
import logging
import chromadb
from typing import Any, List, Optional, Type, Optional, Annotated

from pydantic import BaseModel, Field, model_validator

import chromadb
from chromadb.types import Collection
from frag.embeddings.embedding_model import EmbeddingModel, OpenAiEmbeddingModel

from frag.embeddings.embeddings_metadata import Chunk, Metadata
from frag.embeddings.write.source_chunker import SourceChunker, ChunkingSettings


logger = logging.getLogger(__name__)
class EmbeddingsWriter(BaseModel):
    """
    A class for writing embeddings to a vector db using various models.
    """
    database_path: Annotated[str, Field(os.path.join(os.path.dirname(__file__), "chroma_db"), description="Path to the Chroma database")]
    chunker: SourceChunker = Field(..., description="Chunker for the embeddings")
    embedding_model: EmbeddingModel = Field(..., description="Embedding model")
    client: Any = Field(..., description="Chroma client for the database")
    collection_name: str = Field(..., description="Name of the collection")
    
    @model_validator(mode='before')
    def validate_settings(cls, values):
        embedding_class: EmbeddingModel = values["embedding_model"]
        settings: ChunkingSettings = values.get('chunking_settings')

        if settings and embedding_class:
            if (settings.buffer_before < 0 or settings.buffer_after < 0):
                raise ValueError(f"buffer_before and buffer_after must be greater than 0")
            embedding_model = OpenAiEmbeddingModel(embedding_class) if isinstance(embedding_class, str) else embedding_class()
            if hasattr(embedding_model, 'max_tokens'):
                buffered_max_tokens = embedding_model.max_tokens - (settings.buffer_before + settings.buffer_after)
                if buffered_max_tokens <= 0:
                    raise ValueError(f"The available tokens must be greater than the sum of buffer_before and buffer_after.\n\nRequired: {settings.buffer_before + settings.buffer_after}, but got: {embedding_model.max_tokens}")
                chunker =SourceChunker(settings=settings, embedding_model=embedding_model)

            else:
                raise ValueError(f"Embedding model {embedding_class} does not have a max_tokens attribute")
        return {
            **values,
            "buffered_max_tokens": buffered_max_tokens,
            "embedding_model": embedding_model,
            "chunker": chunker
        }

    @model_validator(mode='before')
    def validate_chroma_client(cls, values):
        database_path: str = values.get('database_path')

        if not os.path.exists(database_path):
            os.makedirs(database_path, exist_ok=True)
        if not os.access(database_path, os.W_OK):
            raise ValueError(f"Database path {database_path} is not writable")
        client =  chromadb.PersistentClient(path=database_path)
        client.get_or_create_collection(values.get('collection_name', 'default_collection'))
        values['client'] = client
        return values

    @property
    def name(self) -> str:
        """Returns the name of the embedding model."""
        return self.embedding_model.name

    def fetch_embedding(self, text: str) -> List[float]:
        """Returns the embedding vector for the given text."""
        return self.embedding_model.embed(text)
        
        
    def create_embeddings_for_document(self, text: str, metadata: Metadata):
        """
        Creates embeddings for a document and stores them in a Chroma database.
        """
        chunks = self.chunker.chunk_text(text)
        parts = len(chunks)
        for index, source_chunk in enumerate(chunks):
            chunk = Chunk(
                part=index + 1,
                parts=parts,
                **metadata.model_dump(),
                **self.chunker.settings.model_dump(),
                **source_chunk.model_dump()
            )
            self.fetch_and_store_embedding(chunk=chunk)
        return chunk

    def fetch_and_store_embedding(self, chunk: Chunk) -> List[float]:
        """
        Stores the embedding in a Chroma database and returns it.

        Parameters:
            chunk: The Chunk object containing text and metadata.

        Returns:
            The embedding vector.
        """
        try:
            collection_result = self.client.collection.get(ids=chunk.id)
                
            if collection_result and collection_result[0]:
                logging.info("Embedding found in db")
                return collection_result[0]

            logging.info("getting embeddings") 
            embedding = self.fetch_embedding(chunk.text)
            
            self.client.collection.add(
                ids=chunk.id,
                embeddings=embedding,
                documents=chunk.text,
                metadatas=chunk.metadata()
            )

            return embedding
        except Exception as e:
            logging.error(f"Error storing embedding: {e}")
            return None


            
    def update_metadata(self, chunk_id: str, metadata: Metadata) -> bool:
        """
        Updates the metadata for an embedding in the Chroma database.
        """
        # if the current metadata has a different schema, throw an error.
        result = self.client.collection.get(ids=chunk_id)[0]
        if not metadata.model_validate(result.metadata):
            raise ValueError(f"The metadata schema is different from the current metadata schema")
        self.client.collection.update(ids=chunk_id, metadatas=metadata.model_dump())
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
            delete_result = self.client.collection.delete(ids=[chunk_id])
            if delete_result:
                logging.info(f"Successfully deleted embedding with ID: {chunk_id}")
                return True
            else:
                logging.warning(f"Failed to delete embedding with ID: {chunk_id}")
                return False
        except Exception as e:
            logging.error(f"Error deleting embedding: {e}")
            return False