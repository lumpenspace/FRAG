from typing import List
import logging
from pydantic import Field
from .chunk import Chunk, Metadata

logger = logging.getLogger(__name__)


class DBChunk(Chunk):
    """
    A database chunk that extends the base Chunk model with a score attribute.
    """

    score: float = Field(default=0.0, description="The score of the chunk.")
    parts: int = Field(1, description="The number of parts the chunk is split into")
    part: int = Field(1, description="The part number of the chunk")

    @classmethod
    def from_db_results(
        cls,
        ids: List[List[str]],
        documents: List[List[str]],
        metadatas: List[List[Metadata]],
        distances: List[List[float]],
        **kwargs
    ):
        """
        Creates a list of DBChunk instances from database results.

        Args:
            ids (List[List[str]]): The list of chunk ids.
            documents (List[List[str]]): The list of chunk documents.
            metadatas (List[List[dict]]): The list of chunk metadatas.
            distances (List[List[float]]): The list of chunk distances.
            **kwargs: Additional keyword arguments.

        Returns:
            List[DBChunk]: A list of DBChunk instances.
        """

        db_chunks = []
        for i in range(len(ids)):
            chunk = cls(
                parts=metadatas[i][0].to_dict().get("parts", 1),
                part=metadatas[i][0].to_dict().get("part", 1),
                metadata=metadatas[i][0],
                score=distances[i][0],
                text=documents[i][0],
                id=ids[i][0],
            )
            db_chunks.append(chunk)
        return db_chunks
