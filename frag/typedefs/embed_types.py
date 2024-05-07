"""
Pydantic models for document metadata.

It includes the following models:
- DocMeta: A model representing metadata for a document, including title, URL, author, 
    and publish date.
- RecordMeta: A model representing metadata for a document chunk, including part number, text
    before and after the chunk, and extra metadata.
"""

from datetime import date, datetime
from typing import Dict, Optional, Union, Literal
from llama_index.core.embeddings import BaseEmbedding

from pydantic import BaseModel, Field, field_validator

ApiSource = Literal["OpenAI", "HuggingFace"]

BaseEmbedding = BaseEmbedding


class DocMeta(BaseModel):
    """
    A model representing metadata for a document, including title, URL, author, and publish date.
    """

    title: str = Field(..., description="Title of the document")
    url: Optional[str] = Field(None, description="URL of the document")
    author: str = Field(None, description="Author of the document")
    publish_date: Optional[datetime | str] = Field(
        None, description="Publish date of the document"
    )
    extra_metadata: Dict[str, Union[str, int, float, date]] = Field(
        {}, description="Extra metadata"
    )


class RecordMeta(DocMeta):
    """
    A model representing metadata for a document, including title, URL, author, and publish date.
    """

    part: int = Field(1, description="Part of the document")
    parts: int = Field(1, description="Total parts of the document")
    before: str | None = Field(..., description="Text before the chunk")
    after: str | None = Field(..., description="Text after the chunk")

    extra_metadata: Dict[str, Union[str, int, float, date]] = Field(
        {}, description="Extra metadata"
    )

    @field_validator("publish_date", mode="before")
    @classmethod
    def parse_publish_date(cls, v: datetime | str) -> datetime:
        """
        Parses the publish date into a datetime object.
        """
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d")
        return v

    def to_dict(self) -> dict[str, int | str]:
        """
        Converts the model instance into a dictionary.
        """
        return self.model_dump()

    def to_json(self) -> str:
        """
        Converts the model instance into a JSON string.
        """
        return self.model_dump_json()
