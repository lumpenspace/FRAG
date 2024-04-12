from typing import Any, Dict, List

from pydantic import BaseModel, Field

class EmbedAPI(BaseModel):
    """
    Base class for embedding APIs. This class defines the interface that all embedding API implementations should follow.
    
    A base class for embedding models.

    Methods:
        tokenize: Takes a text input and returns a list of tokens.
        embed: Takes a text input and returns an embedding vector.
        decode: Takes a list of tokens and returns a text string.
    """

    dimensions: int = Field(..., description="The dimensionality of the embedding vectors")
    max_tokens: int = Field(..., description="The maximum number of tokens to consider for embedding")


    def tokenize(self, text: str) -> List[int]:
        """Takes a text input and returns a list of tokens."""
        raise NotImplementedError("Subclasses must implement the tokenize method")

    def decode(self, tokens: List[int]) -> str:
        """Takes a list of tokens and returns a text string."""
        raise NotImplementedError("Subclasses must implement the decode method")

    def embed(self, text: str) -> Dict[str, Any]:
        """
        Generates an embedding for the given text.
        
        Parameters:
            text (str): The text to embed.
            
        Returns:
            Dict[str, Any]: The embedding vector.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
