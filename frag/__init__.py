"""
.. include:: ../README.md
"""

from frag import typedefs
from frag.frag import Frag
from frag.embeddings import EmbeddingStore
from frag.completions import Prompter

from dotenv import load_dotenv

load_dotenv()

__all__ = ["Frag", "typedefs", "EmbeddingStore", "Prompter"]
