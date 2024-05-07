"""
.. include:: ../README.md
"""

from frag import typedefs
from frag.frag import Frag
from frag.embeddings.store import EmbeddingStore
from frag.completions import Prompter
from frag.console import console, error_console, live

from dotenv import load_dotenv

load_dotenv()

__all__: list[str] = [
    "Frag",
    "typedefs",
    "EmbeddingStore",
    "Prompter",
    "console",
    "live",
    "error_console",
]
