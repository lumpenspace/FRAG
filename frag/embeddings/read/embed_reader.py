"""
`EmbeddingsReader` is responsible for reading embeddings from a vector database.

It fetches the closest n embeddings to a given input and, if they are contiguous, returns them as a single text block.
"""
from pydantic import BaseModel, Field
from typing import List
import logging

from frag.embeddings.chunks import Chunk,  DBChunk
from frag.embeddings.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)

class EmbeddingsReader(BaseModel):
    """
    A class to read and fetch similar embeddings from an embedding store.
    
    Attributes:
        store (EmbeddingStore): The embedding store to fetch embeddings from.
        n_results (int): The number of results to fetch.
        min_similarity (float): The minimum similarity threshold for fetched embeddings.
    """
    
    store: EmbeddingStore = Field(default=None)
    n_results: int = Field(default=3)
    min_similarity: float = Field(default=0.5)

    def get_similar(self, text:str, n_results: int = None, **kwargs) -> List[Chunk]:
        """
        Fetches embeddings similar to the given string.

        Parameters:
            text: The input string to find similar embeddings for.

        Returns:
            A list of Chunk objects with similar embeddings.

        Example:
            >>> reader = EmbeddingsReader(store=embedding_store)
            >>> similar_chunks = reader.get_similar("example text")
            >>> print(similar_chunks)

        Raises:
            ConnectionError: If there is an issue connecting to the database.
            QueryError: If the query fails for any reason.
        """
        
        try:
            db_results = self.store.find_similar(text=text, n_results=n_results or self.n_results)
            return DBChunk.from_db_results(**db_results)
                                           
        except Exception as e:
            raise e
