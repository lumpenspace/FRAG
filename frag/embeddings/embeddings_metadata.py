from pydantic import BaseModel, ConfigDict, Field


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


class ChunkMetadata(Metadata):
    part: int = Field(..., description="Part of the document")
    parts: int = Field(..., description="Total parts of the document")