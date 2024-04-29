from typing import List, Literal

from frag.types import LLMSettings, MessageParam
from .base_api_client import BaseApiClient


class Summarizer(BaseApiClient):
    """
    The Summarizer class is responsible for summarizing chat messages.
    It uses internal methods to render system and user messages based on
    the latest messages.
    """

    client_type: Literal["summarizer"] = "summarizer"

    def __init__(self, settings: LLMSettings, template_dir: str):
        super().__init__(settings, template_dir=template_dir)

    def render(
        self, latest_messages: List[MessageParam], **kwargs
    ) -> List[MessageParam]:
        return [
            self._render_message(latest_messages, role="system", **kwargs),
            self._render_message(latest_messages, role="user", **kwargs),
        ]
