from pydantic import BaseModel

class Note(BaseModel):
    id: str
    source: str
    title: str
    summary: str
    complete: bool
