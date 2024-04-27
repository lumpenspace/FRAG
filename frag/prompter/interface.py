from pydantic import BaseModel
import logging
from litellm.main import completion
from litellm.types.completion import CompletionRequest
from typing import List
from openai.types.chat import ChatCompletionMessage as Message, ChatCompletionRole as Role
from frag.types import ModelSettings, Note
from .base_api_client import BaseApiClient
import jinja2

class Interface(BaseApiClient):
    client_type = "prompter"
    settings: ModelSettings
    system_template: jinja2.Template
    user_template: jinja2.Template

    def load_templates(self, system_template_path: str, user_template_path: str):
        self.system_template = jinja2.Template(open(system_template_path).read())
        self.user_template = jinja2.Template(open(user_template_path).read())

    def run(self, messages: List[Message], notes: List[Note]):
        rendered_messages = self.render(messages, notes)
        result = completion(self.settings.model, messages=rendered_messages, **self.settings)
        return result

    def render(self, messages:List[Message], **kwargs) -> List[Message]:
        [last_message, *messages] = messages
        return [
            self._render_system(messages=messages, **kwargs),
            *messages,
            self._render_user(messages=last_message.content, **kwargs)
        ]

    def _render_system(self, latest_messages:List[Message], **kwargs) -> Message:
        return Message(self.system_template.render(latest_messages=latest_messages, **kwargs), role=Role.SYSTEM)

    def _render_user(self, latest_messages:List[Message], **kwargs) -> Message:
        return Message(self.user_template.render(latest_messages=latest_messages, **kwargs), role=Role.USER)

