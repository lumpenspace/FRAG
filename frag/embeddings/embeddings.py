import json
import re
import os

from typing import List, Union, Dict, Any, Type
from pydantic import BaseModel, Field, ConfigDict

from chromadb import Client
from urllib.request import urlcleanup

from .source_chunker import SourceChunker, ChunkingSettings
from .embeddings_model import EmbeddingModel, OpenAiEmbeddingModel, openai_embedding_models

class Metadata(BaseModel):
    url: str = Field(..., description="URL of the document")
    title: str = Field(..., description="Title of the document")

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "remove_titles": True,
            "validate_types": ["string", "number"]
        }
    )

    @staticmethod
    def schema_extra(schema: Dict[str, Any], model: Type['Metadata']) -> None:
        if model.config_dict.json_schema_extra.get("remove_titles", False):
            for prop in schema.get('properties', {}).values():
                prop.pop('title', None)
        if "validate_types" in model.config_dict.json_schema_extra:
            allowed_types = model.config_dict.json_schema_extra["validate_types"]
            for prop in schema.get('properties', {}).values():
                if prop['type'] not in allowed_types:
                    raise ValueError("Only string or number fields are allowed.")

class Embeddings(BaseModel):
    """
    A class for handling embeddings using various models.
    Methods:
        name(self) -> str: Returns the name of the embedding model.
        fetch_embedding(self, text: str) -> List[float]: Returns the embedding vector for the given text.
        fetch_and_store_embedding(self, exchange: List[str], name: str, metadata: T, chroma_client: Client): Stores the embedding in a Chroma database and returns it.
        store_grounding_embeddings(self, name: str, chroma_client: Client): Stores embeddings for a collection of documents in a Chroma database.
    """
    model: Union[str, Type[EmbeddingModel]] = Field(..., description="Embedding model to use")
    api_key: str = Field("", description="API key for the embedding model")
    database_path: str = Field("./chroma_db", description="Path to the Chroma database")
    chunker: SourceChunker = Field(..., description="Chunker for the embeddings")

    def __init__(self, chunking_settings: ChunkingSettings,  **data):
        # Fetch the API key from the environment variable
        api_key_from_env = os.getenv('OPENAI_API_KEY', '')
        # If an API key is provided in the data, use it; otherwise, use the one from the environment
        data['api_key'] = data.get('api_key', api_key_from_env)

        chunker = SourceChunker(chunking_settings=chunking_settings, embedding_model=self.model)
        
        # Transform the model before passing it to super().__init__
        if isinstance(data.get('model'), str):
            if data.get('model') in openai_embedding_models:
                data['model'] = OpenAiEmbeddingModel(name=data.get('model'), api_key=data['api_key'])
            else:
                raise ValueError(f"Unsupported embedding model: {data.get('model')}")
            
        # create or initialise the database at database_path
        self.chroma_client = Client(database_path=data.database_path)    
        
        super().__init__(**data, chunker=chunker)

    @property
    def name(self) -> str:
        """Returns the name of the embedding model."""
        if isinstance(self.model, str):
            return self.model
        else:
            return self.model.name

    def fetch_embedding(self, text: str) -> List[float]:
        """Returns the embedding vector for the given text."""
        if isinstance(self.model, str):
            raise ValueError(f"Unsupported embedding model: {self.model}")
        else:
            return self.model.embed(text)
        
    def create_embeddings_for_document(self, text: str, metadata: Metadata):
        """
        Creates embeddings for a document and stores them in a Chroma database.
        """
        chunks = self.chunker.chunk_text(text)
        for chunk in chunks:
            self.fetch_and_store_embedding(chunk['text'], metadata=metadata)

    def fetch_and_store_embedding(self, text: str, metadata: Metadata):
        """
        Stores the embedding in a Chroma database and returns it.

        Parameters:
            text: The text of the document
            metadata: The metadata associated with the embedding.

        Returns:
            The embedding vector.
        """
        try:
            collection = self.chroma_client.fetch_or_create_collection(metadata.name)
            id = urlcleanup(metadata.url).lower()

            stored_embedding = collection.get(ids=id).get('embeddings')

            if stored_embedding and len(stored_embedding):
                print("Embedding found in db")
                return stored_embedding[0]

            print("getting embeddings")
            # Get the embedding for the text
            embedding = self.fetch_embedding(text)
            
            # Store the text and its embedding in Chroma
            collection.add(ids=id, embeddings=embedding, documents=text, metadatas=metadata.model_dump())

            return embedding
        except Exception as e:
            print(f"Error storing embedding: {e}")
            # Log the error or handle it appropriately
            return None

    def update_embedding_metadata(self, id: str, metadata: Metadata):
        """
        Updates the metadata of an existing embedding in the Chroma database.

        Parameters:
            id: The ID of the embedding to update.
            metadata: The new metadata for the embedding.
        """
        collection = self.chroma_client.get_collection(metadata.name)
        if collection:
            collection.update(ids=id, metadatas=metadata.model_dump())

