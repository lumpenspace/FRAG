from typing import List
import logging

from pydantic import Field
from .chunk import Chunk

logger = logging.getLogger(__name__)

class DBChunk(Chunk):
    score: float = Field(default=0.0, description="The score of the chunk.")

    @classmethod
    def from_db_results(cls, ids: List[List[str]], documents: List[List[str]], metadatas: List[List[dict]], distances: List[List[float]], **kwargs):
        """
        Creates a Chunk instance from a database result.
        """
        
        db_chunks = []
        for i in range(len(ids)):
            chunk = cls(
                parts = metadatas[i][0].pop('parts', 1),
                part = metadatas[i][0].pop('part', 1),
                metadata = metadatas[i][0],
                score = distances[i][0],
                text = documents[i][0],
                id = ids[i][0]
            )
            db_chunks.append(chunk)
        return db_chunks

        

