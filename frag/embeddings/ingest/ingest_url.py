from re import I
from typing import List
from llama_index.core.schema import Document
from llama_index.readers.web import (
    BeautifulSoupWebReader,
)
from llama_index.core.ingestion import IngestionPipeline
from pydantic import BaseModel, Field, field_validator
from frag.typedefs.embed_types import PipelineAddons


class URLIngestor(BaseModel):
    """
    Ingest a URL into the embedding store.
    """

    pipeline: IngestionPipeline
    _documents: List[Document]

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline(cls, v) -> IngestionPipeline:

        return v

    def ingest(self, data: List[str] | str) -> None:
        """
        Ingest the URL into the embedding store.
        """
        if isinstance(data, str):
            data = [data]
        self.pipeline.ingest(
            documents=BeautifulSoupWebReader().load_data(urls=data),
            addons=self.pipeline_addons(),
        )
