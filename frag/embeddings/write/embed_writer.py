"""
This module contains the `EmbeddingsWriter` class, which is responsible for
writing embeddings to a vector database using various models.

It handles the creation, storage, and management of embeddings,
leveraging the Chroma database for persistence.

The class also provides functionality for chunking text documents,
fetching embeddings from models, and updating or deleting embeddings in the database.
"""

import logging

from pydantic import BaseModel, Field
from frag.embeddings.embedding_store import EmbeddingStore
from frag.embeddings.chunks import SourceChunker, Chunk
from frag.embeddings.embeddings_metadata import Metadata, ChunkMetadata

logger = logging.getLogger(__name__)


class EmbedWriter(BaseModel):

    store: EmbeddingStore = Field(default=None)
    chunker: SourceChunker = Field(default=None)

    def create_embeddings_for_document(self, text: str, metadata: Metadata):
        """
        Creates embeddings for a given document and stores them in the database.

        Parameters:
            text (str): The text of the document to create embeddings for.
            metadata (Metadata): Metadata associated with the document.

        Returns:
            A list of Chunk objects representing the embedded chunks of the document.

        Example:
            >>> writer = EmbeddingsWriter(store=embedding_store, chunker=source_chunker)
            >>> metadata = Metadata(title="Sample Document", parts=3)
            >>> chunks = writer.create_embeddings_for_document(
                "This is a sample document.",
                metadata)
            >>> for chunk in chunks:
            ...     print(chunk.id)
        """
        chunks = []
        source_chunks = self.store.chunker.chunk_text(text)
        metadata.parts = len(source_chunks)
        for index, source_chunk in enumerate(source_chunks):
            metadata = ChunkMetadata(
                **metadata.model_copy(update={"part": index + 1}).model_dump()
            )
            chunk = Chunk.from_source_chunk(
                source_chunk=source_chunk, metadata=metadata, part=index + 1
            )
            self.fetch_and_store_embedding(chunk)
            chunks.append(chunk)

        return chunks

    def fetch_and_store_embedding(self, chunk: Chunk):
        """
        Stores the embedding in a Chroma database and returns it.

        Parameters:
            chunk: The Chunk object containing text and metadata.

        Returns:
            The embedding vector.

        Raises:
            ConnectionError: If there is an issue connecting to the database.
            ValueError: If the embedding cannot be fetched or stored.

        Example:
            >>> writer = EmbeddingsWriter(store=embedding_store, chunker=source_chunker)
            >>> metadata = Metadata(title="Sample Document", parts=1)
            >>> source_chunk = SourceChunk(text="This is a sample document.")
            >>> chunk = Chunk.from_source_chunk(
                source_chunk=source_chunk,
                metadata=metadata,
                part=1)
            >>> embedding = writer.fetch_and_store_embedding(chunk)
            >>> print(embedding)
        """
        try:
            collection_result = self.store.get(ids=[chunk.id])
            embedding = collection_result.get("embeddings")
            if embedding is not None:
                logging.info("Embedding found in db")
                return embedding

            logging.info("getting embeddings")
            self.store.add(
                ids=chunk.id,
                documents=chunk.text,
                metadatas=chunk.metadata.model_dump(
                    exclude_none=True
                ),  # Use the serialized metadata with None values excluded
            )

        except ConnectionError as e:
            logger.error(f"Failed to connect to the database: {e}")
            raise ConnectionError("Failed to connect to the database") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching or storing embedding: {e}")
            raise ValueError("Unexpected error fetching or storing embedding") from e
