import os
from chromadb.api import ClientAPI
from chromadb import Collection, PersistentClient
from chromadb.api.types import QueryResult, Embeddings
from chromadb.errors import ChromaError, InvalidCollectionException
import logging
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    computed_field,
    ConfigDict,
)

from frag.embeddings.apis import EmbedAPI, get_embed_api
from frag.embeddings.chunks import SourceChunker
from frag.typedefs import EmbedSettings

from typing import List, Callable

logger = logging.getLogger(__name__)

"""
This module defines the EmbeddingStore class, which is responsible for storing and managingi
embeddings in a Chroma database.
It utilizes Pydantic for data validation and Chroma for database interactions. The EmbeddingStore
class provides functionality
to validate embedding sources, manage Chroma client and collection instances, and perform
operations such as fetching, updating,
and deleting embeddings.
"""


class EmbeddingStore(BaseModel):
    """
    A class for storing and managing embeddings in a Chroma database.

    Attributes:
        db_path (str): Path to the Chroma database.
        collection_name (str): Name of the collection within the Chroma database.
        collection (chromadb.Collection): Chroma client for the database.
        embed_api (Type[EmbedAPI]|str): Embedding Source.
            For HuggingFace models, use the model name; for OpenAI, the model name is prefixed
                with 'oai:'.
            For instance: "oai:text-embedding-ada-002".
        chunk_settings (ChunkerSettings): Chunking settings.

    Methods:
        validate_embedding_source: Validates the embedding source and chunking settings.
        validate_chroma_client: Validates and initializes the Chroma client and collection.
        name: Returns the name of the embedding model.
        chroma_collection: Returns the Chroma collection instance.
        add: Shortcut method to add an item to the Chroma collection.
        get: Shortcut method to get an item from the Chroma collection.
        query: Shortcut method to query the Chroma collection.
        fetch: Returns the embedding vector for the given text.
        update_metadata: Updates the metadata for an embedding in the Chroma database.
        delete_embedding: Deletes an embedding from the Chroma database based on the chunk ID.
    """

    _api: EmbedAPI | None = None
    _chunker: SourceChunker | None = None
    _collection: Collection | None = None
    settings: EmbedSettings

    model_config = ConfigDict({"arbitrary_types_allowed": True})

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._api = get_embed_api(self.settings.embed_model_name)
        self._chunker = SourceChunker(settings=self.settings, embed_api=self._api)
        self._client = PersistentClient(path=self.settings.db_path)
        self._collection = self._client.get_or_create_collection(
            self.settings.collection_name
        )

    @property
    def add(self):
        return self.collection.add

    @property
    def get(self) -> Callable[..., Embeddings]:
        return self._collection.get

    @property
    def query(self) -> Callable[..., QueryResult]:
        query: Callable[, QueryResult] = self._collection.query
        return query

    def fetch(self, text: str) -> Embeddings:
        """Returns the embedding vector for the given text."""
        return self.embed_api.embed_function().embed_with_retries(text)

    def find_similar(self, text: str | List[str], n_results: int = 1) -> QueryResult:
        """Returns the most similar embeddings to the given text."""
        return self._collection.query(
            query_texts=text if isinstance(text, list) else [text], n_results=n_results
        )

    def delete_embedding(self, chunk_id: str) -> bool:
        """
        Deletes an embedding from the Chroma database based on the chunk ID.

        Parameters:
            chunk_id: The ID of the chunk whose embedding is to be deleted.

        Returns:
            A boolean indicating whether the deletion was successful.
        """
        try:
            delete_result = self.collection.delete(ids=[chunk_id])
            if delete_result:
                logging.info("Successfully deleted embedding with ID: %s", chunk_id)
                return True
            else:
                logging.warning("Failed to delete embedding with ID: %s", chunk_id)
                return False
        except InvalidCollectionException as e:
            logging.error("Invalid Collection: %s", e)
            return False
        except ChromaError as e:
            logging.error("Chroma database error: %s", e)
            return False
