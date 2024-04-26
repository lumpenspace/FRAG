import os
from typing import List, Optional
from chromadb import Embeddings
from chromadb.types import Vector
from pydantic import Field, field_validator, model_validator
from openai import OpenAI, OpenAIError
import tiktoken


from .base_embed_api import EmbedAPI

openaiembedding_models = ["text-embedding-ada-002", "text-embedding-large", "text-embedding-small", "text-embedding-3-large", "text-embedding-3-small"]

class OpenAIEmbedAPI(EmbedAPI):
    """
    A class to interact with OpenAI's embedding models.

    Extends this base class to interact with OpenAI's
    embedding models, providing implementations for the base class's abstract methods
    using OpenAI's API and a specified tokenizer.
    """
    name: str = Field("text-embedding-small", description="The name of the OpenAI embedding model")
    api_key: str|None = Field(None, description="The API key for the OpenAI client")
    tokenizer: Optional[type(tiktoken.Encoding)] = Field(default=None, description="The tokenizer to use")
    openai_client: OpenAI = Field(default=None, description="The OpenAI client to use")
   
    model_config = {
        "arbitrary_types_allowed": True
    }
 
    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v not in openaiembedding_models:
            raise ValueError(f"Unsupported OpenAI embedding model: {v}")
        return v
    
    @field_validator("tokenizer")
    def validate_tokenizer(cls, v):
        if (v in tiktoken.tokenizer_names):
            v = tiktoken.get_encoding(v)
        elif (v is None):
            try:
                v = tiktoken.get_encoding(cls.name)
            except:
                raise ValueError(f"Unsupported tokenizer for: {cls.name}; specify a tokenizer")
        else:
            raise ValueError(f"Unsupported tokenizer for: {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_api_key(cls, values):
        key = values.get('api_key') or os.getenv("OPENAI_API_KEY")

        if (values.get("openai_client") is None):
            if not key:
              raise ValueError("OpenAI API key is required")
            values["openai_client"] = OpenAI(api_key=key)
        return values

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def embed(self, input: List[str]) -> List[Embeddings]:
        try:
            embedding_object = self.openai_client.embeddings.create(input=input, model=self.name)
        except OpenAIError as e:
            raise ValueError(f"OpenAiEmbeddingsModel: error embedding text with OpenAI API: {e}")
        except Exception as e:
            raise ValueError(f"OpenAiEmbeddingsModel: unexpected error embedding text: {e}")

        result:Embeddings = [Vector(embedding=embedding_object.data[i].embedding) for i in range(len(input))]

        return result

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)