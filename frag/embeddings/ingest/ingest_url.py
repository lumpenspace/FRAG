from typing import Callable, List
from llama_index.core.schema import Document
from llama_index.readers.web import (
    BeautifulSoupWebReader,
)
from llama_index.core.ingestion import IngestionPipeline
from pydantic import BaseModel, field_validator


class URLIngestor(BaseModel):
    """
    Ingest a URL into the embedding store.
    """

    pipeline: Callable[[List[IngestionPipeline]], IngestionPipeline]
    _documents: List[Document]

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline(cls, v: IngestionPipeline) -> IngestionPipeline:
        return v

    def ingest(self, data: List[str] | str) -> None:
        """
        Ingest the URL into the embedding store.
        """
        if isinstance(data, str):
            data = [data]
        self.pipeline(self.pipeline_addons()).run(
            documents=BeautifulSoupWebReader().load_data(urls=data),
        )
