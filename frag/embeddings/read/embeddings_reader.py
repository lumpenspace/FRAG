"""
`EmbeddingsReader` is responsible for reading embeddings from a vector database.

It fetches the closest n embeddings to a given input and, if they are contiguous, returns them as a single text block.
"""
from unittest import result
from chromadb import Metadata
from openai import BaseModel
from pydantic import Field
from typing import List

from frag.embeddings.Chunk import Chunk
from frag.embeddings.embedding_store import EmbeddingStore


class EmbeddingsReader(BaseModel):
    
    store: EmbeddingStore = Field(default=None)

    def get_similar(self, text:str, **kwargs) -> List[Chunk]:
        """
        Fetches embeddings similar to the given string.

        Parameters:
            string: The input string to find similar embeddings for.

        Returns:
            A list of Chunk objects with similar embeddings.
        """
        
        result = self.store.query(query_embeddings=self.store.fetch(text), **kwargs)
        return result

