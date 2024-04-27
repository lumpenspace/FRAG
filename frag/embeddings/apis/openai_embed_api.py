import os
from typing import List
from pydantic import Field, field_validator, model_validator
from openai import OpenAI, OpenAIError
import tiktoken


from .base_embed_api import EmbedAPI

openaiembedding_models = [
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
    tokenizer: tiktoken.Encoding = Field(
        default=None, description="The tokenizer to use"
    )
    openai_client: OpenAI = Field(default=None, description="OAI client")

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v not in openaiembedding_models:
            raise ValueError(f"Unsupported OpenAI embedding model: {v}")
        return v

    @field_validator("tokenizer")
    @classmethod
    def validate_tokenizer(cls, v):
        if v in tiktoken.list_encoding_names():
            v = tiktoken.get_encoding(v)
        elif v is None:
            try:
                v = tiktoken.get_encoding(cls.name)
            except Exception:
                raise ValueError(f"Unsupported tokenizer for {cls.name}: {v}")
        else:
            raise ValueError(f"Unsupported tokenizer for: {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_api_key(cls, values):
        key = values.get("api_key") or os.getenv("OPENAI_API_KEY")

        if values.get("openai_client") is None:
            if not key:
                raise ValueError("OpenAI API key is required")
            values["openai_client"] = OpenAI(api_key=key)
        return values

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text=text)

    def embed(self, input: list[str]) -> List[List[float]]:
        try:
            embedding_object = self.openai_client.embeddings.create(
                input=input, model=self.name
            )
        except OpenAIError as e:
            raise ValueError(f"OpenAiEmbeddingsModel: error embedding {input} {e}")
        except Exception as e:
            raise ValueError(
                f"OpenAiEmbeddingsModel: unexpected error embedding text: {e}"
            )
        return [data.embedding for data in embedding_object.data]

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
