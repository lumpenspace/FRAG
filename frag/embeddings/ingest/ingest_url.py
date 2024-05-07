from typing import Callable, List, Dict, Any
from frag.embeddings.store import EmbeddingStore, AddOns
from llama_index.core.schema import Document
from llama_index.readers.web import (
    BeautifulSoupWebReader,
)
from llama_index.core.ingestion import IngestionPipeline
from pydantic import BaseModel, field_validator, Field, ConfigDict


class URLIngestor(BaseModel):
    """
    Ingest a URL into the embedding store.
    """

    store: EmbeddingStore | None = Field(default=None)

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def pipeline_addons(self) -> AddOns:
        return {
            "extractors": [],
            "preprocessors": [],
        }

    def ingest(self, data: List[str] | str) -> None:
        """
        Ingest the URL into the embedding store.
        """
        if isinstance(data, str):
            data = [data]
        self.store.get_pipeline(self.pipeline_addons).run(
            documents=BeautifulSoupWebReader().load_data(urls=data),
        )
