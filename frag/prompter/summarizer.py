
from pydantic import BaseModel
import logging
from litellm.main import completion
from litellm.types.completion import CompletionRequest
from typing import List
from openai.types.chat import ChatCompletionMessage as Message, ChatCompletionRole as Role
from ..types import SummarizerSettings
import jinja2
from .base_api_client import BaseApiClient

class Summarize(BaseApiClient):
    client_type = "summarizer"
    settings: SummarizerSettings
    system_template: jinja2.Template
    user_template: jinja2.Template


    def render(self, latest_messages:List[Message], **kwargs) -> List[Message]:
        return [
            self._render_system(latest_messages, **kwargs),
            self._render_user(latest_messages, **kwargs)
        ]
