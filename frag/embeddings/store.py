from frag.settings import EmbedSettings, DBSettings

from frag.typedefs.embed_types import BaseEmbedding
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser import SentenceSplitter

from llama_index.vector_stores.chroma import ChromaVectorStore


class EmbeddingStore:
    embed_model: BaseEmbedding
    db: ChromaVectorStore
    text_splitter: NodeParser

    def __init__(self, settings: EmbedSettings, db_settings: DBSettings):
        self.embed_model = settings.api
        self.db = ChromaVectorStore(
            persist_directory=db_settings.path,
            collection_name=db_settings.default_collection,
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
