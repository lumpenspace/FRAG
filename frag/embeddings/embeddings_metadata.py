from datetime import date
from flatten_dict import flatten_dict
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Union, Optional

from typing import Any, Dict, Type

from datetime import datetime

class Metadata(BaseModel):
    title: str = Field(..., description="Title of the document")
    url: Optional[str] = Field(None, description="URL of the document")
    author: str = Field(None, description="Author of the document")
    publish_date: Optional[datetime] = Field(None, description="Publish date of the document")

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "remove_titles": True,
            "validate_types": ["string", "number"]
        }
    )

    def to_dict(self):
        return self.model_dump()
    
    def to_json(self):
        return self.model_dump_json()
    

    @staticmethod
    def schema_extra(schema: Dict[str, Any], model: Type['Metadata']) -> None:
        if model.config_dict.json_schema_extra.get("remove_titles", False):
            for prop in schema.get('properties', {}).values():
                prop.pop('title', None)
        if "validate_types" in model.config_dict.json_schema_extra:
            allowed_types = model.config_dict.json_schema_extra["validate_types"]
            for prop in schema.get('properties', {}).values():
                if prop['type'] not in allowed_types:
                    raise ValueError("Only string or number fields are allowed.")

class ChunkMetadata(Metadata):
    part: int = Field(..., description="Part of the document")
    parts: int = Field(..., description="Total parts of the document")
    before: str = Field(..., description="Text before the chunk")
    after: str = Field(..., description="Text after the chunk")
    extra_metadata: Dict[str, Union[str, int, float, date]] = Field({}, description="Extra metadata")

    def combine_with_document_metadata(self, document_metadata: Metadata):
        # Combine chunk-specific metadata with document-level metadata for database saving
        combined_metadata = {
            **self.model_dump(exclude=self.model_dump().keys()),
            **document_metadata.model_dump(exclude=self.model_dump().keys())
        }
        return combined_metadata

    @classmethod
    def separate_from_combined_metadata(cls, combined_metadata: Dict[str, Any]):
        # Separate combined metadata back into document-level and chunk-specific metadata when reading from the database
        document_metadata_keys = Metadata.__fields__.keys()
        document_metadata = {key: combined_metadata[key] for key in document_metadata_keys if key in combined_metadata}
        chunk_metadata = {key: combined_metadata[key] for key in combined_metadata if key not in document_metadata_keys}
        return cls(**chunk_metadata), Metadata(**document_metadata)

    @field_validator('extra_metadata', mode="after")
    def flatten_extra_metadata(cls, v):
        if isinstance(v, dict):
            return flatten_dict(v)
        return v

    def dict(self, *args, **kwargs):
        data = super().model_dump()(*args, **kwargs)
        data['extra_metadata'] = flatten_dict()(data['extra_metadata'])
        return data

    @classmethod
    def parse_obj(cls, data: Any):
        extra_metadata = data.get('extra_metadata', {})
        if isinstance(extra_metadata, dict):
            data += {k: v for k, v in extra_metadata.items()}
        return super().model_dump(data)
    