from typing import List, TypeVar, Protocol, Callable
from pydantic import BaseModel, Field
from chromadb import Documents, Embeddings, EmbeddingFunction
from chromadb.types import EmbeddingRecord


Embeddable = Documents
D = TypeVar("D", bound=Embeddable)


class DBEmbedFunction(Protocol[D]):
    embed: Callable[[D], Embeddings]

    def __init__(self, embed: Callable[[D], Embeddings]) -> None:
        self.embed = embed

    def __call__(self, input: D) -> Embeddings:
        return self.embed(input)


class EmbedAPI(BaseModel):
    """
    Base class for embedding APIs. This class defines the interface that all
    embedding API implementations should follow.

    A base class for embedding models.

    Methods:
        encode: Takes a text input and returns a list of tokens.
        embed: Takes a text input and returns an embedding vector.
        decode: Takes a list of tokens and returns a text string.
    """

    dimensions: int = Field(
        default=125, description="The dimensionality of the embedding vectors"
    )
    max_tokens: int = Field(
        default=125,
        description="The maximum number of tokens to consider for embedding",
    )

    def encode(self, text: str) -> List[int]:
        """Takes a text input and returns a list of tokens."""
        raise NotImplementedError("Subclasses must implement `encode`")

    def decode(self, tokens: List[int]) -> str:
        """Takes a list of tokens and returns a text string."""
        raise NotImplementedError("Subclasses must implement the `decode`")

    @property
    def embed_function(self) -> EmbeddingFunction[Documents]:
        """
        Generates an embedding for the given text.

        Parameters:
            text (str): The text to embed.

        Returns:
            Dict[str, Any]: The embedding vector.
        """
        raise NotImplementedError("Subclasses must implement `embed`")
