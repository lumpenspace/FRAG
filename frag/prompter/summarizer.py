from typing import List
from openai.types.chat import ChatCompletionMessage as Message

from frag.types import ModelSettings
from .base_api_client import BaseApiClient

class Summarizer(BaseApiClient):
    client_type = "summarizer"

    def __init__(self, settings: ModelSettings, system_template_path: str, user_template_path: str):
        super().__init__(settings, system_template_path, user_template_path)


    def render(self, latest_messages:List[Message], **kwargs) -> List[Message]:
        return [
            self._render_system(latest_messages, **kwargs),
            self._render_user(latest_messages, **kwargs)
        ]
