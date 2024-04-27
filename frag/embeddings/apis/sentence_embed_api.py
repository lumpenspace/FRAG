"""
This module allows for embedding with HuggingFace models.
"""

from typing import List, Optional
from chromadb import Embeddings
from pydantic import ConfigDict, Field, model_validator
from logging import getLogger

logger = getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .base_embed_api import EmbedAPI

class SentenceEmbedAPI(EmbedAPI):
    """
    A class to interact with HuggingFace's embedding models.

    Attributes:
        name: The name of the HuggingFace embedding model, for example "gpt2". 
        max_tokens: The maximum number of tokens to embed.
        model: The SentenceTransformer model to use.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field("all-MiniLM-L6-v2", description="The name of the HuggingFace embedding model")
    max_tokens: int = Field(..., description="The maximum number of tokens to embed")
    model: Optional[SentenceTransformer] = Field(None, description="The SentenceTransformer model to use")


    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, v):
        if not SentenceTransformer:
            raise ValueError("SentenceTransformer not installed; use `poetry add -E oss` to install it and run open source models")
        try:
            model = SentenceTransformer(v.get('name'))
        except Exception as e:
            raise ValueError(f"Error initializing SentenceTransformer model: {e}")
        return {
            **v,
            "model": model,
            "max_tokens": v.get('max_tokens', model.get_max_seq_length())
        }

    def encode(self, text: str) -> List[int]:
        return self.model.tokenizer.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.model.tokenizer.decode(tokens)

    def embed(self, input: List[str]) -> List[Embeddings]:
        try:
            return self.model.encode(input)
        except Exception as e:
            raise ValueError(f"Error embedding texts with HuggingFace model: {e}")
  
