from litellm.main import completion
from typing import List, Literal
from frag.types import LLMSettings, Note, MessageParam
from .base_api_client import BaseApiClient
import jinja2
import logging


class Interface(BaseApiClient):
    client_type: Literal["prompter"] = "prompter"
    settings: LLMSettings
    system_template: jinja2.Template
    user_template: jinja2.Template

    def __init__(self, settings: LLMSettings, template_dir: str):
        super().__init__(settings, template_dir=template_dir)

    def run(self, messages: List[MessageParam], notes: List[Note]):
        try:
            rendered_messages = self.render(messages, notes=notes)

            result = completion(
                messages=rendered_messages,
                model=self.settings.llm,
                **self.settings.model_dump(exclude=set(["llm"])),
            )
            return result
        except jinja2.TemplateError as te:
            logging.error(f"Template rendering error: {te}")
            raise
        except Exception as e:
            logging.error(f"General error during completion: {e}")
            raise

    def render(self, messages: List[MessageParam], **kwargs) -> List[MessageParam]:
        try:
            last_message = messages[-1]
            return [
                self._render_message(messages=messages[:-1], role="system", **kwargs),
                *messages[:-1],
                self._render_message(
                    messages=last_message.get("content"), role="user", **kwargs
                ),
            ]
        except IndexError as ie:
            logging.error(
                f"Rendering error - possibly due to empty messages list: {ie}"
            )
            raise
        except Exception as e:
            logging.error(f"General rendering error: {e}")
            raise
