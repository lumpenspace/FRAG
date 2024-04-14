"""
`EmbeddingsReader` is responsible for reading embeddings from a vector database.

It fetches the closest n embeddings to a given input and, if they are contiguous, returns them as a single text block.
"""
from chromadb import Metadata
from pydantic import BaseModel, Field
from typing import List

from frag.embeddings.Chunk import Chunk
from frag.embeddings.embedding_store import EmbeddingStore


class EmbeddingsReader(BaseModel):
    
    store: EmbeddingStore = Field(default=None)
    n_results: int = Field(default=3)
    min_similarity: float = Field(default=0.5)

    def get_similar(self, text:str, **kwargs) -> List[Chunk]:
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
            db_results = self.store.query(
                query_texts=[text],
                n_results=self.n_results,
                **kwargs)
        except Exception as e:
            raise ConnectionError("Failed to fetch from database") from e
        results = zip(db_results["ids"], db_results["distances"])

        return results

