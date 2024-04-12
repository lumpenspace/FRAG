"""
This module allows for embedding with HuggingFace models.
"""

from typing import List
from pydantic import ConfigDict, Field, model_validator
from transformers import AutoTokenizer, AutoModel, pipeline

from .base_embed_api import EmbedAPI

openaiembedding_models = ["text-embedding-ada-002", "text-embedding-large", "text-embedding-small"]

class HuggingFaceEmbedAPI(EmbedAPI):
    """
    A class to interact with HuggingFace's embedding models.

    Attributes:
        name: The name of the HuggingFace embedding model, for example "gpt2"
    """
    name: str = Field("text-embedding-small", description="The name of the HuggingFace embedding model")
    tokenizer: AutoTokenizer = Field(..., description="The tokenizer to use")
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def validate_name(cls, v):
        tokenizer = AutoTokenizer.from_pretrained(v.get('name'))
        model = AutoModel.from_pretrained(v.get('name'))
        pipe = pipeline("feature_extraction", model=model, tokenizer=tokenizer)
        return {
            "tokenizer": tokenizer,
            "model": model,
            "pipe": pipe
        }

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def embed(self, text: str) -> List[float]:
        try:
            return self.pipe(text)
        except Exception as e:
            raise ValueError(f"Error embedding text with HuggingFace model: {e}")

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

