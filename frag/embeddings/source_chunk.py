from pydantic import BaseModel, Field

class SourceChunk(BaseModel):
    text: str = Field(..., description="Text of the chunk")
    before: str = Field(..., description="Text before the chunk")
    after: str = Field(..., description="Text after the chunk")