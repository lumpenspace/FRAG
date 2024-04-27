import logging
import os
from litellm.main import completion
from litellm.types.completion import CompletionRequest
from typing import List
from openai.types.chat import (
    ChatCompletionMessage as Message,
    ChatCompletionRole as Role,
)
from frag.types import LLMSettings
import jinja2


class BaseApiClient:
    """
    Base API client class that handles interactions with the LLM model.
    """

    settings: LLMSettings
    client_type: None
    system_template: jinja2.Template
    user_template: jinja2.Template
    logger: logging.Logger

    def __init__(
        self, settings: LLMSettings, system_template_path: str, user_template_path: str
    ):
        """
        Initializes the BaseApiClient with the given settings, template paths, and client type.

        :param settings: LLMSettings object containing configuration for the model.
        :param system_template_path: Path to the system message template file.
        :param user_template_path: Path to the user message template file.
        :param client_type: Type of the client, used to determine the template files.
        """
        if self.client_type is None:
            raise ValueError("client_type must be provided")
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.load_templates(system_template_path, user_template_path)

    def run(self, messages: List[Message], **kwargs):
        """
        Processes the given messages and performs completion using the LLM model.

        :param messages: List of ChatCompletionMessage objects to be processed.
        """
        try:
            rendered_messages = [
                msg.content for msg in self._render(messages, **kwargs)
            ]
            model_settings = (
                self.settings.llm
                if hasattr(self.settings, "llm")
                else {}
            )
            request = CompletionRequest(
                messages=rendered_messages,
                **model_settings,
            )
            completion(self.settings.llm request)
        except Exception as e:
            self.logger.error(f"Error during completion: {e}")
            raise

    def _render(self, messages: List[Message], **kwargs) -> List[Message]:
        """
        Renders the messages based on the templates. This method should be implemented by subclasses.

        :param messages: List of ChatCompletionMessage objects to be rendered.
        :return: List of rendered ChatCompletionMessage objects.
        """
        raise NotImplementedError

    def load_templates(self, system_template_path: str, user_template_path: str):
        """
        Loads the message templates from the specified paths.

        :param system_template_path: Path to the system message template file.
        :param user_template_path: Path to the user message template file.
        """
        if self.client_type is None:
            raise ValueError("[dev] client_type must be set")

        if (
            self.settings.system_message_template
            and self.settings.user_message_template
        ):
            try:
                with open(
                    os.path.join(self.settings.system_message_template), "r"
                ) as file:
                    self.system_template = jinja2.Template(file.read())
                with open(
                    os.path.join(self.settings.user_message_template), "r"
                ) as file:
                    self.user_template = jinja2.Template(file.read())
            except FileNotFoundError as e:
                self.logger.error(f"Template file not found: {e}")
                raise e
            except Exception as e:
                self.logger.error(f"Error loading templates: {e}")
                raise e
        else:
            # Default templates path assumed to be in the same directory
            try:
                with open(
                    os.path.join("templates", f"{self.client_type}.system.md"), "r"
                ) as file:
                    self.system_template = jinja2.Template(file.read())
                with open(
                    os.path.join("templates", f"{self.client_type}.user.md"), "r"
                ) as file:
                    self.user_template = jinja2.Template(file.read())
            except FileNotFoundError as e:
                self.logger.error(f"Template file not found: {e}")
                raise e
            except Exception as e:
                self.logger.error(f"Error loading templates: {e}")
                raise e

    def _render_message(
        self, latest_messages: List[Message], role: Role, **kwargs
    ) -> Message:
        """
        Renders a message based on the role and the latest messages.

        :param latest_messages: List of the most recent ChatCompletionMessage objects.
        :param role: Role of the message to be rendered (SYSTEM or USER).
        :return: Rendered ChatCompletionMessage object.
        """
        template = self.system_template if role == Role.SYSTEM else self.user_template
        return Message(
            template.render(latest_messages=latest_messages, **kwargs), role=role
        )

    def _render_system(self, latest_messages: List[Message], **kwargs) -> Message:
        return self._render_message(message=latest_messages, role=Role.SYSTEM, **kwargs)

    def _render_user(self, latest_message: Message, **kwargs) -> Message:
        return self._render_message(message=latest_message, role=Role.USER, **kwargs)
