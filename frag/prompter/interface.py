from litellm.main import completion
from litellm.types.completion import CompletionRequest
from typing import List, Literal
from openai.types.chat import (
    ChatCompletionMessage as Message,
    ChatCompletionRole as Role,
)
from frag.types import LLMSettings, Note
from .base_api_client import BaseApiClient
import jinja2
import logging


class Interface(BaseApiClient):
    client_type: Literal["prompter"] = "prompter"
    settings: LLMSettings
    system_template: jinja2.Template
    user_template: jinja2.Template

    def __init__(
        self, settings: LLMSettings, system_template_path: str, user_template_path: str
    ):
        super().__init__(settings, system_template_path, user_template_path)

    def run(self, messages: List[Message], notes: List[Note]):
        try:
            rendered_messages = self.render(messages, notes=notes)
            request = CompletionRequest(
                model=self.settings.model,
                messages=rendered_messages,
                **self.settings.model_dump(),
            )
            result = completion(request)
            return result
        except jinja2.TemplateError as te:
            logging.error(f"Template rendering error: {te}")
            raise
        except Exception as e:
            logging.error(f"General error during completion: {e}")
            raise

    def render(self, messages: List[Message], **kwargs) -> List[Message]:
        try:
            last_message = messages[-1]
            return [
                self._render_system(messages=messages[:-1], **kwargs),
                *messages[:-1],
                self._render_user(messages=last_message.content, **kwargs),
            ]
        except IndexError as ie:
            logging.error(
                f"Rendering error - possibly due to empty messages list: {ie}"
            )
            raise
        except Exception as e:
            logging.error(f"General rendering error: {e}")
            raise

    def _render_system(self, latest_messages: List[Message], **kwargs) -> Message:
        return SystemMessage(
            self.system_template.render(latest_messages=latest_messages, **kwargs)
        )

    def _render_user(self, latest_message: Message, **kwargs) -> Message:
        return Message(
            content=self.user_template.render(message=latest_message, **kwargs),
            role="user",
        )
