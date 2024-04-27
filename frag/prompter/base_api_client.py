
import logging
from litellm.main import completion
from litellm.types.completion import CompletionRequest
from typing import List
from openai.types.chat import ChatCompletionMessage as Message, ChatCompletionRole as Role
from frag.types import ModelSettings
import jinja2

class BaseApiClient():
    settings: ModelSettings
    client_type: None
    system_template: jinja2.Template
    user_template: jinja2.Template
    logger: logging.Logger

    def __init__(self, settings: ModelSettings, system_template_path: str, user_template_path: str):

        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.load_templates(system_template_path, user_template_path)

    def run(self, messages:List[Message], **kwargs):
        request = CompletionRequest(
            messages=self.render(messages, **kwargs),
            **self.settings.llm_model.model_dump(),
        )
        completion(request)

    def render(self, messages:List[Message], **kwargs):
        raise NotImplementedError

    def load_templates(self):
        if self.client_type is None:
            raise ValueError("[dev] client_type must be set")
        if self.settings.system_message_template and self.settings.user_message_template:
            try:
                self.system_template = jinja2.Template(open(self.settings.system_message_template).read())
                self.user_template = jinja2.Template(open(self.settings.user_message_template).read())
            except Exception as e:
                self.logger.error(f"Error loading templates: {e}")
                raise e
        else:
            # Default templates path assumed to be in the same directory
            self.system_template = jinja2.Template(open(f'templates/{self.client_type}.system.md').read())
            self.user_template = jinja2.Template(open(f'templates/{self.client_type}.user.md').read())

    def _render_system(self, latest_messages:List[Message], **kwargs) -> Message:
        return Message(self.system_template.render(latest_messages=latest_messages, **kwargs), role=Role.SYSTEM)

    def _render_user(self, latest_messages:List[Message], **kwargs) -> Message:
        return Message(self.user_template.render(latest_messages=latest_messages, **kwargs), role=Role.USER)

