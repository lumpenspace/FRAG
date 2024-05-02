"""
The SummarizerBot class is responsible for summarizing chat messages.
It uses internal methods to render system and user messages based on
the latest messages.
"""

from typing import List, Literal
from frag.typedefs import LLMSettings, MessageParam, Document, DocMeta
from .base_api_client import BaseBot


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
        self, messages: List[MessageParam], document: Documents, doc_meta: DocMeta
    ) -> List[MessageParam]:
        return [
            self._render_message(messages, role="system", **kwargs),
            self._render_message(messages, role="user", **kwargs),
        ]
