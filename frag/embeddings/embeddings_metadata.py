from datetime import date
from flatten_dict import flatten
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Union, Optional

from typing import Any, Dict

from datetime import datetime


class Metadata(BaseModel):
    """
    A model representing metadata for a document, including title, URL, author, and publish date.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "remove_titles": True,
            "validate_types": ["string", "number"],
        },
    )
    title: str = Field(..., description="Title of the document")
    url: Optional[str] = Field(None, description="URL of the document")
    author: str = Field(None, description="Author of the document")
    publish_date: Optional[datetime | str] = Field(
        None, description="Publish date of the document"
    )

    @field_validator("publish_date", mode="before")
    @classmethod
    def parse_publish_date(cls, v):
        """
        Parses the publish date into a datetime object.
        """
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d")
        return v

    def to_dict(self):
        """
        Converts the model instance into a dictionary.
        """
        return self.model_dump()

    def to_json(self):
        """
        Converts the model instance into a JSON string.
        """
        return self.model_dump_json()


class ChunkMetadata(Metadata):
    """
    A model representing metadata for a chunk of a document, including part number, total parts,
    and extra metadata.
    """

    part: int = Field(..., description="Part of the document")
    parts: int = Field(..., description="Total parts of the document")
    before: str = Field(..., description="Text before the chunk")
    after: str = Field(..., description="Text after the chunk")
    extra_metadata: Dict[str, Union[str, int, float, date]] = Field(
        {}, description="Extra metadata"
    )

    def combine_with_document_metadata(self, document_metadata: Metadata):
        """
        Combines chunk-specific metadata with document-level metadata for database saving.
        """
        combined_metadata = {
            **self.model_dump(),
            **document_metadata.model_dump(),
        }
        return combined_metadata

    @classmethod
    def separate_from_combined_metadata(cls, combined_metadata: Dict[str, Any]):
        """
        Separates combined metadata back into document-level and chunk-specific metadata
        when reading from the database.
        """
        document_metadata_keys = Metadata.model_fields.keys()
        document_metadata = {
            key: combined_metadata[key]
            for key in document_metadata_keys
            if key in combined_metadata
        }
        chunk_metadata = {
            key: combined_metadata[key]
            for key in combined_metadata
            if key not in document_metadata_keys
        }
        return cls(**chunk_metadata), Metadata(**document_metadata)

    @field_validator("extra_metadata", mode="after")
    def flatten_extra_metadata(cls, v):
        """
        Flattens the extra metadata dictionary.
        """
        if isinstance(v, dict):
            return flatten(v)
        return v

    def dict(self, *args, **kwargs):
        """
        Converts the model instance into a dictionary, with extra metadata flattened.
        """
        data = super().model_dump(*args, **kwargs)
        data["extra_metadata"] = flatten(data["extra_metadata"])
        return data

    @classmethod
    def parse_obj(cls, data: Any):
        """
        Parses object data into a model instance, including flattening extra metadata.
        """
        extra_metadata = data.get("extra_metadata", {})
        if isinstance(extra_metadata, dict):
            data += {k: v for k, v in extra_metadata.items()}
        return super().model_dump(data)
