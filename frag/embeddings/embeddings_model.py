from typing import List, Protocol

import os

import tiktoken
from openai import OpenAI

class EmbeddingModel(Protocol):
    """
    A protocol for embedding models.

    Attributes:
        dimensions: The dimensionality of the embedding vectors.
        max_tokens: The maximum number of tokens to consider for embedding.

    Methods:
        tokenize(text: str) -> List[int]: Tokenizes the given text and returns a list of token IDs.
        embed(text: str) -> List[float]: Returns an embedding vector for the given text.
    """
    dimensions: int
    max_tokens: int

    def tokenize(self, text: str) -> List[int]:
        ...

    def embed(self, text: str) -> List[float]:
        ...

openai_embedding_models = ["text-embedding-ada-002", "text-embedding-large", "text-embedding-small"]

class OpenAiEmbeddingModel(EmbeddingModel):
    """
    A class to interact with OpenAI's embedding models.

    This class initializes with a specific model name and creates an OpenAI object
    using the API key fetched from environment variables. It provides methods to
    tokenize and embed text using the specified OpenAI model.

    Attributes:
        model_name: The name of the OpenAI embedding model.
        openai_client: The OpenAI client object initialized with the API key.
    """
    def __init__(self, model_name: str = "text-embedding-small", tokenizer_name: str = "cl100k_base"):

        if not(model_name in openai_embedding_models):
            raise ValueError(f"Unsupported OpenAI embedding model: {model_name}")
        
        api_key = os.getenv("OPENAI_API_KEY")  # API key fetched from environment variable

        self.dimensions = 1536
        self.max_tokens = 8191
        self.model_name = model_name
        self.openai_client = OpenAI(api_key=api_key)
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def embed(self, text: str) -> List[float]:
        try:
            embedding_object = self.openai_client.embeddings.create(input=text, model=self.model_name)
        except Exception as e:
            raise ValueError(f"OpenAiEmbeddingsModel: error embedding text: {e}")

        embedding_vector = embedding_object.data[0].embedding
        
        if len(embedding_vector) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(embedding_vector)}")
        
        return embedding_vector