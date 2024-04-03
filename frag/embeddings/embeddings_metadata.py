from copy import copy
from datetime import date
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Union

from typing import Any, Dict, Type

from sympy import flatten

class Metadata(BaseModel):
    title: str = Field(..., description="Title of the document")

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "remove_titles": True,
            "validate_types": ["string", "number"]
        }
    )


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

    @field_validator('extra_metadata', mode="after")
    def flatten_extra_metadata(cls, v):
        if isinstance(v, dict):
            return flatten(v)
        return v

    def dict(self, *args, **kwargs):
        data = super().dict(*args, **kwargs)
        data['extra_metadata'] = flatten(data['extra_metadata'])
        return data

    @classmethod
    def parse_obj(cls, data: Any):
        extra_metadata = data.get('extra_metadata', {})
        if isinstance(extra_metadata, dict):
            data += {k: v for k, v in extra_metadata.items()}
        return super().model_dump(data)
    