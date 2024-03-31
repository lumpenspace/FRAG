from datetime import date
from importlib import metadata
from pydantic import BaseModel, ConfigDict, Field, model_validator
import hashlib

from typing import Any, Dict, Type

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


class Chunk(BaseModel):
    part: int = Field(..., description="Part of the document")
    parts: int = Field(..., description="Total parts of the document")
    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text of the chunk")
    before: str = Field(..., description="Text before the chunk")
    after: str = Field(..., description="Text after the chunk")
    title: str = Field(..., description="Title of the document")
    extra_metadata: Dict[str, str|int|float|date] = Field({}, description="Extra metadata")
    

    @model_validator(mode="before")
    def validate_id(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        text_hash = hashlib.sha256(v.get('text').encode('utf-8')).hexdigest()
        v['id'] = f"{v.get('title')}::{v.get('part')}:{v.get('parts')}::{text_hash[:8]}"
        return v
    
    @property
    def metadata(self) -> Metadata:
        return Metadata(
            **self.extra_metadata,
            title=self.title,
            part=self.part,
            parts=self.parts,
            tokens=self.tokens,
            before=self.before,
            after=self.after
        )
