import os
from frag.settings import DBSettingsDict

from rich.prompt import Prompt


def embed_db_init(path: str) -> DBSettingsDict:

    db_path: str = Prompt.ask(
        "Embeddings DB path",
        default=os.path.join(path, "db"),
    )
    collection_name: str = Prompt.ask("Default collection name", default="default")

    return DBSettingsDict(path=db_path, collection_name=collection_name)
