from typing import List, TypedDict, Self

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.extractors import BaseExtractor
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import TransformComponent
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb import PersistentClient
from trio import Path
from frag.settings.embed_settings import EmbedSettings
from frag.utils import SingletonMixin
from frag.typedefs.embed_types import BaseEmbedding

ArgType = TypedDict(
    "ArgType",
    {"settings": EmbedSettings},
)

AddOns = TypedDict(
    "AddOns",
    {"extractors": List[BaseExtractor], "preprocessors": List[TransformComponent]},
)


class EmbeddingStore(SingletonMixin[type(ArgType)]):
    embed_model: BaseEmbedding
    db: ClientAPI
    collection: Collection
    text_splitter: SentenceSplitter = SentenceSplitter()
    docstore: SimpleDocumentStore = SimpleDocumentStore()
    vector_store: ChromaVectorStore
    settings: EmbedSettings
    index: BaseRetriever
    collection_name: str
    text_splitter: SentenceSplitter
    docstore: SimpleDocumentStore
    vector_store: ChromaVectorStore

    def __init__(
        self,
        settings: EmbedSettings,
        collection_name: str | None = None,
    ) -> None:
        self.settings = settings
        self.collection_name = collection_name or self.settings.default_collection
        self.db = PersistentClient(
            path=str(settings.path / "db"),
        )
        self.embed_model = settings.api
        self.change_collection(collection_name=collection_name)
        self.text_splitter = SentenceSplitter()
        self.docstore = SimpleDocumentStore()
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

    @classmethod
    def create(
        cls, settings: EmbedSettings, collection_name: str | None = None
    ) -> Self:
        """
        Create a new embedding store
        """
        cls.reset()
        instance: Self = cls.__new__(cls, settings=settings)
        return instance

    def get_index(self) -> BaseRetriever:
        """
        Get the index
        """
        vector_store: ChromaVectorStore = ChromaVectorStore(
            chroma_collection=self.collection
        )
        storage_context: StorageContext = StorageContext.from_defaults(
            vector_store=vector_store
        )
        index: VectorStoreIndex = VectorStoreIndex.from_documents(
            documents=[],
            storage_context=storage_context,
            embed_model=self.embed_model,
        )
        return index.as_retriever()

    def change_collection(self, collection_name: str | None = None) -> None:
        """
        Change the collection name
        """
        self.collection_name = collection_name or self.settings.default_collection
        self.collection = self.db.get_or_create_collection(name=self.collection_name)
        self.index = self.get_index()

    def get_pipeline(
        self,
        addons: AddOns,
    ) -> IngestionPipeline:
        """
        Ingest a URL into the embedding store.
        """
        return IngestionPipeline(
            transformations=[
                *addons["preprocessors"],
                self.text_splitter,
                self.embed_model,
                *addons["extractors"],
            ],
            cache=IngestionCache(
                collection=f"{self.collection_name}-{self.embed_model.model_name}"
            ),
            vector_store=self.vector_store,
            docstore=self.docstore,
        )
