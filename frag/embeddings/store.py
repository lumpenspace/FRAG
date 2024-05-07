from typing import List, TypedDict

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.extractors import BaseExtractor
from llama_index.core.node_parser import NodeParser, SentenceSplitter
from llama_index.core.schema import TransformComponent
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import BaseRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from openai import BaseModel
from frag import settings

from frag.utils.console import error_console
from frag.settings import EmbedSettings
from frag.utils import SingletonMixin
from frag.typedefs.embed_types import BaseEmbedding

ArgType = TypedDict(
    "ArgType",
    {"settings": EmbedSettings},
)


class EmbeddingStore(SingletonMixin[ArgType]):
    embed_model: BaseEmbedding
    db: ChromaVectorStore
    text_splitter: NodeParser
    docstore: SimpleDocumentStore

    def __init__(self, settings: EmbedSettings) -> None:
        self.embed_model = settings.api
        self.db = ChromaVectorStore(
            persist_directory=settings.db_path,
            collection_name=settings.default_collection,
        )
        self.text_splitter = SentenceSplitter(
            chunk_overlap=20,
        )

    def change_collection(self, collection_name: str) -> None:
        """
        Change the collection name
        """
        self.db = ChromaVectorStore(
            persist_directory=self.db.persist_dir,
            collection_name=collection_name,
        )

    @property
    def collection_name(self) -> str:
        """
        Get the collection name
        """
        return f"{self.db.collection_name}"

    @property
    def index(self) -> BaseRetriever:
        """
        Get the index
        """
        storage_context: StorageContext = StorageContext.from_defaults(
            vector_store=self.db
        )
        index: VectorStoreIndex = VectorStoreIndex.from_documents(
            documents=[],
            storage_context=storage_context,
            embed_model=self.embed_model,
        )
        return index.as_retriever()

    def get_pipeline(
        self,
        extractors: List[BaseExtractor] = [],
        preprocessors: List[TransformComponent] = [],
    ) -> IngestionPipeline:
        """
        Ingest a URL into the embedding store.
        """
        return IngestionPipeline(
            transformations=[
                *preprocessors,
                self.text_splitter,
                self.embed_model,
                *extractors,
            ],
            cache=IngestionCache(
                collection=f"{self.db.collection_name}-{self.embed_model.model_name}"
            ),
            vector_store=self.db,
            docstore=self.docstore,
        )
