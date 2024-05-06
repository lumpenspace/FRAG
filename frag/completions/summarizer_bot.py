"""
The SummarizerBot class is responsible for summarizing chat messages.
It uses internal methods to render system and user messages based on
the latest messages.
"""

from typing import List, Literal, Dict, Any
from frag.typedefs import LLMSettings, MessageParam, Document, DocMeta
from .base_bot import BaseBot


class SummarizerBot(BaseBot):
    """
    The Summarizer class is responsible for summarizing chat messages.
    It uses internal methods to render system and user messages based on
    the latest messages.
    """

    client_type: Literal["summarizer"] = "summarizer"

    def __init__(self, settings: LLMSettings, template_dir: str):
        super().__init__(settings, template_dir=template_dir)

    def _render(
        self,
        *_,
        messages: List[MessageParam],
        document: Document,
        doc_meta: DocMeta,
        **kwargs: Dict[str, Any]
    ) -> List[MessageParam]:
        return [
            self._render_message(messages, role="system", **kwargs),
            self._render_message(messages, role="user", **kwargs),
        ]
