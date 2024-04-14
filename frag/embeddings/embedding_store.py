import os
import chromadb
import logging
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field
from typing import List, Type
from frag.embeddings.apis.base_embed_api import DBEmbedFunction

from frag.embeddings.embeddings_metadata import Metadata
from frag.embeddings.apis import EmbedAPI, get_embed_api
from frag.embeddings.chunks import SourceChunker, ChunkSettings

logger = logging.getLogger(__name__)

"""
This module defines the EmbeddingStore class, which is responsible for storing and managing embeddings in a Chroma database.
It utilizes Pydantic for data validation and Chroma for database interactions. The EmbeddingStore class provides functionality
to validate embedding sources, manage Chroma client and collection instances, and perform operations such as fetching, updating,
and deleting embeddings.
"""

class EmbeddingStore(BaseModel):
    """
    A class for storing and managing embeddings in a Chroma database.

    Attributes:
        path (str): Path to the Chroma database.
        collection_name (str): Name of the collection within the Chroma database.
        collection (chromadb.Collection): Chroma client for the database.
        embed_api (Type[EmbedAPI]|str): Embedding Source.
            For HuggingFace models, use the model name; for OpenAI, the model name is prefixed with 'oai:'.
            For instance: "oai:text-embedding-ada-002".
        chunk_settings (ChunkSettings): Chunking settings.

    Methods:
        validate_embedding_source: Validates the embedding source and chunking settings.
        validate_chroma_client: Validates and initializes the Chroma client and collection.
        name: Returns the name of the embedding model.
        chroma_collection: Returns the Chroma collection instance.
        add: Shortcut method to add an item to the Chroma collection.
        get: Shortcut method to get an item from the Chroma collection.
        query: Shortcut method to query the Chroma collection.
        fetch: Returns the embedding vector for the given text.
        update_metadata: Updates the metadata for an embedding in the Chroma database.
        delete_embedding: Deletes an embedding from the Chroma database based on the chunk ID.
    """
    path: str = Field(str(os.path.join(os.path.dirname(__file__), "chroma_db")), description="Path to the Chroma database")
    collection_name: str = Field(default="default_collection", description="Name of the collection")
    embed_api: Type[EmbedAPI]|str|EmbedAPI = Field(..., description="Embedding Source")
    chunk_settings: ChunkSettings = Field(..., description="Chunking settings")

    collection: chromadb.Collection = None
    chunker: SourceChunker= None
    client: chromadb.PersistentClient= None

    @field_validator('path')
    def validate_path(cls, v):
        if not os.path.exists(v):
            os.makedirs(v, exist_ok=True)
        return v
    
    @field_validator('embed_api')
    def validate_embeddings_model(cls, v):
        if not v:
            raise ValueError("Embedding model and chunking settings must be provided")
        if isinstance(v, EmbedAPI):
            return v
        return get_embed_api(v)
    
    @field_validator('collection_name')
    def validate_collection_name(cls, v):
        if not v:
            v = 'default_collection'
            logging.warning(f"Collection name not provided, using default collection name: {v}")
        return v

    @model_validator(mode='after')
    def validate_chroma_client(self):
        if type(self.chunk_settings) is dict:
            self.chunk_settings = ChunkSettings(**self.chunk_settings)

        try:
            self.client = chromadb.PersistentClient(path=self.path)
        except Exception as e:
            logger.error(f"Error creating chromadb client: {e}")
            raise e
        try:
            self.collection = self.client.get_or_create_collection(
                self.collection_name,
                embedding_function=DBEmbedFunction(embed=self.embed_api.embed)
            )
        except Exception as e:
            logger.error(f"Error creating chromadb collection: {e}")
            raise e
        
        if not hasattr(self.embed_api, 'max_tokens'):
            raise ValueError(f"Embedding model {self.embed_api.__repr_name__} does not have a max_tokens attribute")

        self.chunker = SourceChunker(settings=self.chunk_settings, embed_api=self.embed_api)
        return self


    @computed_field
    def name(self) -> str:
        """Returns the name of the embedding model."""
        return self.embed_api.name


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
        return self.embed_api.embed(text)
    
    def find_similar(self, text: str|List[str], n_results: int = 1) -> chromadb.QueryResult:
        """Returns the most similar embeddings to the given text."""
        return self.collection.query(
            query_texts=text if isinstance(text, list) else [text],
            n_results=n_results
        )

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