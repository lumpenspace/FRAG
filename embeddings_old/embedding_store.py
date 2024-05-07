from typing import Any
from chromadb import Collection, GetResult, PersistentClient
from chromadb.api import ClientAPI
from chromadb.api.types import QueryResult, Embeddings
from chromadb.errors import ChromaError, InvalidCollectionException
from frag.console import console, error_console
from frag.embeddings_old.apis import EmbedAPI
from frag.embeddings_old.chunks import SourceChunker
from frag.settings import DBSettings, ChunkerSettings, EmbedSettings

from typing import List, Callable

"""
This module defines the EmbeddingStore class, which is responsible for storing and managingi
embeddings in a Chroma database.
It utilizes Pydantic for data validation and Chroma for database interactions. The EmbeddingStore
class provides functionality
to validate embedding sources, manage Chroma client and collection instances, and perform
operations such as fetching, updating,
and deleting embeddings.
"""


class EmbeddingStore:
    """
    A class for storing and managing embeddings in a Chroma database.

    Attributes:
        embed_api_settings (EmbedAPISettings): Embedding API settings.
        db_settings (DBSettings): Database settings.
        chunker_settings (ChunkerSettings): Chunking settings.

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

    api: EmbedAPI
    chunker: SourceChunker
    collection: Collection
    embed_api_settings: EmbedSettings
    db_settings: DBSettings
    chunker_settings: ChunkerSettings
    _collection_name: str

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.api: EmbedAPI = self.embed_api_settings.api
        self.chunker = SourceChunker(settings=self.chunker_settings, embed_api=self.api)
        self.client: ClientAPI = PersistentClient(path=self.db_settings.path)
        self.collection: Collection = self.client.get_or_create_collection(
            self.db_settings.default_collection,
            embedding_function=self.api.embed_function,
        )
        self._collection_name = self.db_settings.default_collection

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @collection_name.setter
    def collection_name(self, name: str) -> None:
        self._collection_name = name
        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=self.api.embed_function,
        )
        console.log("Collection set to: %s", name)

    @property
    def get(self) -> Callable[..., GetResult]:
        return self.collection.get

    def fetch(self, text: str) -> Embeddings:
        """Returns the embedding vector for the given text."""
        return self.api.embed_function(input=[text])

    def find_similar(self, text: str | List[str], n_results: int = 1) -> QueryResult:
        """Returns the most similar embeddings to the given text."""
        return self.collection.query(
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
            delete_result: None = self.collection.delete(ids=[chunk_id])
            if delete_result:
                console.log("Successfully deleted embedding with ID: %s", chunk_id)
                return True
            else:
                console.log("Failed to delete embedding with ID: %s", chunk_id)
                return False
        except InvalidCollectionException as e:
            error_console.log("Invalid Collection: %s", e)
            return False
        except ChromaError as e:
            error_console.log("Chroma database error: %s", e)
            return False
