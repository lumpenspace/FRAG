"""
`EmbeddingsReader` is responsible for reading embeddings from a vector database.

It fetches the closest n embeddings to a given input and, if they are contiguous, returns them as a single text block.
"""
from chromadb import Metadata
from openai import BaseModel
from pydantic import Field
from typing import List

from frag.embeddings.Chunk import Chunk

from frag.embeddings.embedding_store import EmbeddingStore

class EmbeddingResult(BaseModel()):
    text: Field(str)
    metadata: Field(Metadata)
    chunks: Field(List[Chunk()])

class EmbeddingsReader(EmbeddingStore):
    pass
    def fetch_similar(self, string: str) -> List[Chunk]:
        """
        Fetches embeddings similar to the given string.

        Parameters:
            string: The input string to find similar embeddings for.

        Returns:
            A list of Chunk objects with similar embeddings.
        """
        embedding = self.fetch_embedding(string)
        similar_embeddings = self.chroma_collection.find_similar(embeddings=embedding)
        similar_chunks = [Chunk.from_source_chunk(**e) for e in similar_embeddings]
        return similar_chunks

