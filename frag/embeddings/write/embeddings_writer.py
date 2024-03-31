import os
import logging

from typing import List, Optional, Type, Optional
from pydantic import BaseModel, Field, model_validator

from chromadb import ClientAPI, Collection, Client

from frag.embeddings.embeddings_metadata import Chunk, Metadata
from frag.embeddings.write.source_chunker import SourceChunker, ChunkingSettings
from frag.embeddings.embedding_model import EmbeddingModel, OpenAiEmbeddingModel, openai_embedding_models

logger = logging.getLogger(__name__)

class EmbeddingsWriter(BaseModel):
    """
    A class for writing embeddings to a vector db using various models.
    """
    database_path: str = Field("./chroma_db", description="Path to the Chroma database")
    chunking_settings: ChunkingSettings = Field(..., description="Settings for chunking the text")
    chunker: SourceChunker = Field(..., description="Chunker for the embeddings")
    chroma_client: Type[ClientAPI] = Field(..., description="Chroma client for the database")
    embedding_model: EmbeddingModel = Field(..., description="Embedding model")
    collection_name: Optional[str] = Field('default_collection', description="Collection name")

    @model_validator(mode='before')
    def validate_chunking_settings(cls, values):
        chunking_settings = values.get('chunking_settings')
        if not isinstance(chunking_settings, ChunkingSettings):
            raise ValueError(f"Expected 'chunking_settings' to be of type 'ChunkingSettings', got {type(chunking_settings)}")
        return values

    @model_validator(mode='before')
    def validate_settings(cls, values):
        settings: ChunkingSettings = values["settings"]
        embedding_class: EmbeddingModel = values["embedding_model"]
        
        if settings and embedding_class:
            if (settings.buffer_before < 0 or settings.buffer_after < 0):
                raise ValueError(f"buffer_before and buffer_after must be greater than 0")
            embedding_model = OpenAiEmbeddingModel(embedding_class) if embedding_class is str else EmbeddingModel()
            if hasattr(embedding_model, 'max_tokens'):
                buffered_max_tokens = embedding_model.max_tokens - (settings.buffer_before + settings.buffer_after)
                if buffered_max_tokens <= 0:
                    raise ValueError(f"The available tokens must be greater than the sum of buffer_before and buffer_after.\n\nRequired: {settings.buffer_before + settings.buffer_after}, but got: {embedding_model.max_tokens}")
                values["buffered_max_tokens"] = buffered_max_tokens
                values["embedding_model"]
        return values

    @model_validator(mode='before')
    def validate_chunker(cls, values):
        chunking_settings = values.get('chunking_settings')
        embedding_model = values.get('embedding_model')

        if chunking_settings and embedding_model:
            values['chunker'] = SourceChunker(chunking_settings=chunking_settings, embedding_model=embedding_model)

        return values

    @model_validator(mode='before')
    def validate_chroma_client(cls, values):
        database_path = values.get('database_path')

        # check the path's accessibility
        if database_path and os.path.exists(database_path) and os.access(database_path, os.W_OK):
            values['chroma_client'] = Client(database_path=database_path)
        else:
            raise ValueError(f"Database path {database_path} is not writable or does not exist")

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
                text=source_chunk.text,
                before=source_chunk.before, 
                after=source_chunk.after,
                part=index + 1,
                parts=parts,
                **metadata.model_dump()
            )
            self.fetch_and_store_embedding(chunk=chunk)


    def fetch_and_store_embedding(self, chunk: Chunk) -> Optional[List[float]]:
        """
        Stores the embedding in a Chroma database and returns it.

        Parameters:
            chunk: The Chunk object containing text and metadata.

        Returns:
            The embedding vector.
        """
        try:
            collection:Collection = self.chroma_client.get_or_create_collection(self.collection_name)

            collection_result = collection.get(ids=chunk.id)
            
            if collection_result and collection_result[0]:
                logging.info("Embedding found in db")
                return collection_result[0]

            logging.info("getting embeddings") 
            embedding = self.fetch_embedding(chunk.text)
            
            collection.add(
                ids=chunk.id,
                embeddings=embedding,
                documents=chunk.text,
                metadatas=chunk.metadata()
            )

            return embedding
        except Exception as e:
            logging.error(f"Error storing embedding: {e}")
            return None

    def delete_embedding(self, chunk_id: str) -> bool:
        """
        Deletes an embedding from the Chroma database based on the chunk ID.

        Parameters:
            chunk_id: The ID of the chunk whose embedding is to be deleted.

        Returns:
            A boolean indicating whether the deletion was successful.
        """
        try:
            collection: Collection = self.chroma_client.get_or_create_collection(self.collection_name)
            delete_result = collection.delete(ids=[chunk_id])
            if delete_result:
                logging.info(f"Successfully deleted embedding with ID: {chunk_id}")
                return True
            else:
                logging.warning(f"Failed to delete embedding with ID: {chunk_id}")
                return False
        except Exception as e:
            logging.error(f"Error deleting embedding: {e}")
            return False