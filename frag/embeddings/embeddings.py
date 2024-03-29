import json
import re
import os

from typing import List, Union, TypeVar, Generic, Protocol
from pydantic import BaseModel, Field

from chromadb import Client
from openai import OpenAI

from .embeddings_model import EmbeddingModel, OpenAiEmbeddingModel, openai_embedding_models

T = TypeVar('T', bound=BaseModel)

class EmbeddingModelProtocol(Protocol):
    def embed(self, text: str) -> List[float]:
        ...

class Embeddings(BaseModel):
    """
    A class for handling embeddings using various models.
    Attributes:
        model: The embedding model to use. Can be a string identifier or an EmbeddingModel instance.
        api_key: API key for the embedding model if required.
        metadata_type: The type of metadata associated with the embeddings.

    Methods:
        model_name(self) -> str: Returns the name of the embedding model.
        get_embedding(self, text: str) -> List[float]: Returns the embedding vector for the given text.
        get_and_store_embedding(self, exchange: List[str], name: str, metadata: T, chroma_client: Client): Stores the embedding in a Chroma database and returns it.
        store_grounding_embeddings(self, name: str, chroma_client: Client): Stores embeddings for a collection of documents in a Chroma database.
    """
    model: Union[str, EmbeddingModelProtocol] = Field(..., description="Embedding model to use")
    api_key: str = Field("", description="API key for the embedding model")
    metadata_type: T = Field(..., description="Metadata type for the embeddings")

    def __init__(self, **data):
        # Fetch the API key from the environment variable
        api_key_from_env = os.getenv('OPENAI_API_KEY', '')
        # If an API key is provided in the data, use it; otherwise, use the one from the environment
        data['api_key'] = data.get('api_key', api_key_from_env)
        
        # Transform the model before passing it to super().__init__
        if isinstance(data.get('model'), str):
            if data.get('model') in openai_embedding_models:
                data['model'] = OpenAiEmbeddingModel(model_name=data.get('model'), api_key=data['api_key'])
            else:
                raise ValueError(f"Unsupported embedding model: {data.get('model')}")
        
        super().__init__(**data)

    @property
    def model_name(self) -> str:
        """Returns the name of the embedding model."""
        if isinstance(self.model, str):
            return self.model
        else:
            return self.model.model_name

    def get_embedding(self, text: str) -> List[float]:
        """Returns the embedding vector for the given text."""
        if isinstance(self.model, str):
            raise ValueError(f"Unsupported embedding model: {self.model}")
        else:
            return self.model.embed(text)

    def get_and_store_embedding(self, exchange: List[str], name: str, metadata: T, chroma_client: Client):
        """
        Stores the embedding in a Chroma database and returns it.

        Parameters:
            exchange: A list containing the question and answer.
            name: The name of the collection in the Chroma database.
            metadata: The metadata associated with the embedding.
            chroma_client: The Chroma database client.

        Returns:
            The embedding vector.
        """
        question, answer = exchange

        collection = chroma_client.get_or_create_collection(name)
        id = re.sub(r'\W+', '', metadata.url + question[:20]).lower()

        stored_embedding = collection.get(ids=id).get('embeddings')

        if stored_embedding and len(stored_embedding):
            print("Embedding found in db")
            return stored_embedding[0]

        print("getting embeddings")
        # Get the embedding for the text
        embedding = self.get_embedding(question)
        
        text = f"In a past interview, you answered '{question}' with:\n\n {answer}"
        # Store the text and its embedding in Chroma
        collection.add(ids=id, embeddings=embedding, documents=text, metadatas=metadata.model_dump())

        return embedding
