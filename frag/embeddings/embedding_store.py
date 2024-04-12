import os
import chromadb
import logging
from pydantic import BaseModel, Field, model_validator, computed_field
from typing import List, Type

from frag.embeddings.embeddings_metadata import Metadata
from frag.embeddings.embed_api import  OpenAIEmbedAPI, EmbedAPI
from frag.embeddings.write.source_chunker import SourceChunker, ChunkingSettings

logger = logging.getLogger(__name__)

class EmbeddingStore(BaseModel):
    path: str = Field(os.path.join(os.path.dirname(__file__), "chroma_db"), description="Path to the Chroma database")
    collection_name: str = Field(default="default_collection", description="Name of the collection")
    collection: chromadb.Collection = Field(..., description="Chroma client for the database")
    chunker: SourceChunker = Field(..., description="Chunker for the embeddings")
    embeddings_api: Type[EmbedAPI]|str = Field(..., description="Embedding Source")
    chunking_settings: ChunkingSettings = Field(..., description="Chunking settings")
    embedding_model: EmbedAPI 

    @model_validator(mode='before')
    @classmethod
    def validate_embedding_source(cls, values):
        if "embeddings_api" not in values or "chunking_settings" not in values:
            missing_keys = [key for key in ["embeddings_api", "chunking_settings"] if key not in values]
            raise ValueError(f"Missing required parameters: {', '.join(missing_keys)}")
        embeddings_api = values["embeddings_api"]
        settings: ChunkingSettings = values['chunking_settings']
        if settings and embeddings_api:
            if (settings.buffer_before < 0 or settings.buffer_after < 0):
                raise ValueError(f"buffer_before and buffer_after must be greater than 0")
            if isinstance(embeddings_api, str):
                embedding_model = OpenAIEmbedAPI(embeddings_api)
            else:
                embedding_model = embeddings_api()

            if not hasattr(embedding_model, 'max_tokens'):
                raise ValueError(f"Embedding model {embeddings_api} does not have a max_tokens attribute")

            buffered_max_tokens = embedding_model.max_tokens - (settings.buffer_before + settings.buffer_after)
            if buffered_max_tokens <= 0:
                raise ValueError(f"The available tokens must be greater than the sum of buffer_before and buffer_after.\n\nRequired: {settings.buffer_before + settings.buffer_after}, but got: {embedding_model.max_tokens}")
  
            return {
                **values,
                "buffered_max_tokens": buffered_max_tokens,
                "embedding_model": embedding_model,
                "chunking_settings": settings,
                "chunker": SourceChunker(settings=settings, embedding_model=embedding_model)
            }
        else:
            raise ValueError("Embedding model and chunking settings must be provided")
 

    @model_validator(mode='before')
    @classmethod
    def validate_chroma_client(cls, values: dict):
        path: str = values.get('path') 
        collection_name: str = values.get('collection_name')

        if not collection_name:
            collection_name = 'default_collection'
            logging.warning(f"Collection name not provided, using default collection name: {collection_name}")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise ValueError(f"Database path {path} is not writable")
        try:
            client = chromadb.PersistentClient(path=path)
        except Exception as e:
            logger.error(f"Error creating chromadb client: {e}")
            raise e
        try:
            collection = client.get_or_create_collection(collection_name)
        except Exception as e:
            logger.error(f"Error creating chromadb collection: {e}")
            raise e
        
        return {
            **values,
            "collection": collection
        }


    @computed_field
    def name(self) -> str:
        """Returns the name of the embedding model."""
        return self.embedding_model.name


    @property
    def chroma_collection(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        if not os.access(self.path, os.W_OK):
            raise ValueError(f"Database path {self.path} is not writable")
        try:
            client = chromadb.PersistentClient(path=self.path)
            collection = client.get_or_create_collection(self.collection_name)
            return collection
        except Exception as e:
            logging.error(f"Error creating chromadb client or collection: {e}")
            raise e
        
    @property
    def add(self):
        return self.collection.add

    @property
    def get(self):
        return self.collection.get
    
    @property
    def query(self):
        return self.collection.query

    def fetch(self, text: str) -> List[float]:
        """Returns the embedding vector for the given text."""
        return self.embedding_model.embed(text)

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