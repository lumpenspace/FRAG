"""
This module allows for embedding with HuggingFace models.
"""

from typing import List
from pydantic import ConfigDict, Field
from logging import getLogger, Logger
from chromadb.api.types import EmbeddingFunction, Documents
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

logger: Logger = getLogger(__name__)

from .embed_api import EmbedAPI  # noqa


class HFEmbedAPI(EmbedAPI):
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
    max_tokens: int = Field(512, description="Maximum tokens to embed")

    _api: "SentenceTransformer" | None = None

    @property
    def model(self) -> "SentenceTransformer":

        if self._api is None:
            try:
                self._api = SentenceTransformer(self.name)
            except ImportError:
                raise ImportError(
                    "Unable to import SentenceTransformer. Please install it using ",
                    "`pip install sentence-transformers`",
                )
            except Exception as e:
                raise ValueError(f"Error embedding text with HF model: {e}")
        return self._api

    def encode(self, text: str) -> List[int]:
        return self.model.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.model.tokenizer.decode(tokens)

    @property
    def embed_function(self) -> EmbeddingFunction[Documents]:
        try:
            return SentenceTransformerEmbeddingFunction(model_name=self.name)
        except Exception as e:
            raise ValueError(f"Error embedding text with HF model: {e}")
