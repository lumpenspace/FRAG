from typing import List, Literal
from openai.types.chat import ChatCompletionMessage as Message

from frag.types import LLMSettings
from .base_api_client import BaseApiClient


class Summarizer(BaseApiClient):
    """
    The Summarizer class is responsible for summarizing chat messages.
    It uses internal methods to render system and user messages based on
    the latest messages.
    """

    client_type: Literal["summarizer"] = "summarizer"

    def __init__(
        self, settings: LLMSettings, system_template_path: str, user_template_path: str
    ):
        super().__init__(settings, system_template_path, user_template_path)

    def render(self, latest_messages: List[Message], **kwargs) -> List[Message]:
        return [
            self._render_system(latest_messages, **kwargs),
            self._render_user(latest_messages[-1], **kwargs),
        ]
