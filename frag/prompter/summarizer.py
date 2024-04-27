
from pydantic import BaseModel
import logging
from litellm import completion
from litellm.types.completion import CompletionRequest
from typing import List
from openai.types.chat import ChatCompletionMessage as Message, ChatCompletionRole as Role
from ..settings import SummarizerSettings
import jinja2

logger = logging.getLogger(__name__)

class Summarizer(BaseModel):
    settings: SummarizerSettings
    system_template: jinja2.Template
    user_template: jinja2.Template

    def __init__(self, **data):
        super().__init__(**data)
        self.load_templates()
        logger.info("Summarizer initialized")

    def run(self, input_messages:List[Message], **kwargs):
        request = CompletionRequest(
            messages=self.render(input_messages, **kwargs),
            **self.settings.model_settings.model_dump(),
        )
        completion(request)

    def load_templates(self):
        if self.settings.system_message_template and self.settings.user_message_template:
            try:
                self.system_template = jinja2.Template(open(self.settings.system_message_template).read())
                self.user_template = jinja2.Template(open(self.settings.user_message_template).read())
            except Exception as e:
                logger.error(f"Error loading templates: {e}")
                raise e
        else:
            # Default templates path assumed to be in the same directory
            self.system_template = jinja2.Template(open('templates/summarizer.system.md').read())
            self.user_template = jinja2.Template(open('templates/summarizer.user.md').read())

    def render(self, latest_messages:List[Message], **kwargs) -> List[Message]:
        return [
            self._render_system(latest_messages, **kwargs),
            self._render_user(latest_messages, **kwargs)
        ]

    def _render_system(self, latest_messages:List[Message], **kwargs) -> Message:
        return Message(self.system_template.render(latest_messages=latest_messages, **kwargs), role=Role.SYSTEM)

    def _render_user(self, latest_messages:List[Message], **kwargs) -> Message:
        return Message(self.user_template.render(latest_messages=latest_messages, **kwargs), role=Role.USER)

