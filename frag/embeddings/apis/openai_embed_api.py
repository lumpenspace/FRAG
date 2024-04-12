"""
This module defines the base and OpenAI-specific embedding model classes.

The `EmbeddingModel` class serves as a base for creating embedding models with
methods for tokenization, embedding, and decoding.

The `OpenAiEmbeddingModel` class extends this base class to interact with OpenAI's
embedding models, providing implementations for the base class's abstract methods
using OpenAI's API and a specified tokenizer.
"""

from typing import List, Optional
from pydantic import Field, field_validator, model_validator
from openai import OpenAI, OpenAIError
import tiktoken


from .base_embed_api import EmbedAPI

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
    tokenizer_name: str = Field("cl10k_base", description="The name of the tokenizer to use")
    tokenizer: Optional[type(tiktoken.Encoding)] = Field(default=None, description="The tokenizer to use")

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

    @model_validator(mode="before")
    @classmethod
    def validate_tokenizer_name(cls, values):
        if  values.tokenizer_name and values.tokenizer_name in tiktoken.tokenizer_names:
            tokenizer = tiktoken.get_encoding(values.tokenizer_name)
        elif tiktoken.encoding_name_for_model(values.name):
            tokenizer = tiktoken.encoding_for_model(values.name)
        else:
            raise ValueError(f"Unsupported tokenizer: {values.tokenizer_name}")
        return {
            **values,
            "tokenizer": tokenizer
        }

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def embed(self, text: str) -> List[float]:
        try:
            embedding_object = self.openai_client.embeddings.create(input=text, model=self.name)
        except OpenAIError as e:
            raise ValueError(f"OpenAiEmbeddingsModel: error embedding text with OpenAI API: {e}")
        except Exception as e:
            raise ValueError(f"OpenAiEmbeddingsModel: unexpected error embedding text: {e}")

        embedding_vector = embedding_object.data[0].embedding

        return embedding_vector

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)