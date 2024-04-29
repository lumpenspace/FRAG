"""
This module allows for embedding with HuggingFace models.
"""

from typing import List
from pydantic import ConfigDict, Field, model_validator
from logging import getLogger
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .base_embed_api import EmbedAPI  # noqa


class SentenceEmbedAPI(EmbedAPI):
    """
    A class to interact with HuggingFace's embedding models.

    Attributes:
        name: The name of the HuggingFace embedding model, for example "gpt2".
        max_tokens: The maximum number of tokens to embed.
        model: The SentenceTransformer model to use.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(
        "all-MiniLM-L6-v2", description="Name for HuggingFace embeddings model"
    )
    max_tokens: int = Field(..., description="Maximum tokens to embed")
    model: "SentenceTransformer" = Field(  # type: ignore
        description="The SentenceTransformer model to use"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, v):
        if not SentenceTransformer:
            raise ValueError(
                "SentenceTransformer not installed; use `poetry add -E oss` \
                    to install it and run open source models"
            )
        try:
            model = SentenceTransformer(v.get("name"))
        except Exception as e:
            raise ValueError(f"Error: init SentenceTransformer model:{e}")
        return {
            **v,
            "model": model,
            "max_tokens": v.get("max_tokens", model.get_max_seq_length()),
        }

    def encode(self, text: str) -> List[int]:
        return self.model.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.model.tokenizer.decode(tokens)

    def embed(self) -> EmbeddingFunction:
        try:
            return SentenceTransformerEmbeddingFunction(model_name=self.model)
        except Exception as e:
            raise ValueError(f"Error embedding text with HF model: {e}")
