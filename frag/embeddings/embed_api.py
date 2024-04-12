"""
This module defines the base and OpenAI-specific embedding model classes.

The `EmbeddingModel` class serves as a base for creating embedding models with
methods for tokenization, embedding, and decoding.

The `OpenAiEmbeddingModel` class extends this base class to interact with OpenAI's
embedding models, providing implementations for the base class's abstract methods
using OpenAI's API and a specified tokenizer.
"""

from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator
from openai import OpenAI
import tiktoken

class EmbedAPI(BaseModel):
    """
    A base class for embedding models.

    Attributes:
        dimensions: The dimensionality of the embedding vectors.
        max_tokens: The maximum number of tokens to consider for embedding.
    """
    dimensions: int = Field(..., description="The dimensionality of the embedding vectors")
    max_tokens: int = Field(..., description="The maximum number of tokens to consider for embedding")

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError("Subclasses must implement the tokenize method")

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError("Subclasses must implement the embed method")

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError("Subclasses must implement the decode method")

openaiembedding_models = ["text-embedding-ada-002", "text-embedding-large", "text-embedding-small"]

class OpenAIEmbedAPI(EmbedAPI):
    """
    A class to interact with OpenAI's embedding models.

    Attributes:
        name: The name of the OpenAI embedding model.
        api_key: The API key for the OpenAI client.
        tokenizer_name: The name of the tokenizer to use.
    """
    name: str = Field("text-embedding-small", description="The name of the OpenAI embedding model")
    api_key: str = Field(..., description="The API key for the OpenAI client")
    tokenizer_name: str = Field("cl100k_base", description="The name of the tokenizer to use")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v not in openaiembedding_models:
            raise ValueError(f"Unsupported OpenAI embedding model: {v}")
        return v
    
    @model_validator(mode="before")
    @classmethod
    def validate_api_key(cls, values):
        if not values.api_key:
            raise ValueError("OpenAI API key is required")
        values["openai_client"] = OpenAI(api_key=values.api_key)
        return values

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def embed(self, text: str) -> List[float]:
        try:
            embedding_object = self.openai_client.embeddings.create(input=text, model=self.name)
        except Exception as e:
            raise ValueError(f"OpenAiEmbeddingsModel: error embedding text: {e}")

        embedding_vector = embedding_object.data[0].embedding

        return embedding_vector

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)