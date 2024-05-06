import os
from typing import List
from pydantic import Field, field_validator
from openai import OpenAI
from chromadb.utils import embedding_functions
from chromadb.api.types import EmbeddingFunction
from frag.typedefs import Documents

import tiktoken


from .embed_api import EmbedAPI

openai_embedding_models: list[str] = [
    "text-embedding-ada-002",
    "text-embedding-large",
    "text-embedding-small",
    "text-embedding-3-large",
    "text-embedding-3-small",
]


class OAIEmbedAPI(EmbedAPI):
    """
    A class to interact with OpenAI's embedding models.

    Extends this base class to interact with OpenAI's
    embedding models, providing implementations for
    the base class's abstract methods
    using OpenAI's API and a specified tokenizer.
    """

    name: str = Field(
        "text-embedding-small", description="Name of the OAI embedding model"
    )
    api_key: str | None = Field(
        default=None, description="The API key for the OpenAI client"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if v not in openai_embedding_models:
            raise ValueError(f"Unsupported OpenAI embedding model: {v}")
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str:
        key: str | None = v or os.getenv("OPENAI_API_KEY")
        if (key is None or len(key) == 0) and os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OpenAI API key is required")
        try:
            OpenAI(api_key=key).models.list()
        except Exception as e:
            raise ValueError(f"Invalid OpenAI API key: {e}")
        return str(key)

    @property
    def tokenizer(self) -> tiktoken.Encoding:
        return tiktoken.encoding_for_model(self.name)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text=text)

    @property
    def embed_function(self) -> EmbeddingFunction[Documents]:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name=self.name,
        )

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
