import logging
import os
from litellm.main import completion
from typing import List
from frag.types import (
    LLMSettings,
    MessageParam,
    Role,
    SystemMessage,
    UserMessage,
)
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

    def __init__(self, settings: LLMSettings, template_dir: str):
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
        self.responder = settings["responder"]
        self.load_templates(template_dir)

    def run(self, messages: List[MessageParam], **kwargs):
        """
        Processes the given messages and performs completion using the LLM model.

        :param messages: List of ChatCompletionMessage objects to be processed.
        """
        if self.settings.llm is None:
            raise ValueError("llm must be provided")
        if len(messages) == 0 or messages is None:
            raise ValueError("messages must be provided")
        try:
            rendered_messages: List[MessageParam] = [
                msg for msg in self._render(messages, **kwargs)
            ]

            completion(self.settings.llm, messages=rendered_messages)
        except Exception as e:
            self.logger.error(f"Error during completion: {e}")
            raise

    def _render(self, messages: List[MessageParam], **kwargs) -> List[MessageParam]:
        """
        Renders the messages based on the templates. This method should be implemented
        by subclasses.

        :param messages: List of ChatCompletionMessage objects to be rendered.
        :return: List of rendered ChatCompletionMessage objects.
        """
        raise NotImplementedError

    def load_templates(self, template_dir: str):
        """
        Loads the message templates from the specified paths.

        :param system_template_path: Path to the system message template file.
        :param user_template_path: Path to the user message template file.
        """
        if self.client_type is None:
            raise ValueError("[dev] client_type must be set")

        template_dir = template_dir if template_dir else "templates"
        try:
            with open(
                os.path.join(template_dir, f"{self.client_type}.system.html"), "r"
            ) as file:
                self.system_template = jinja2.Template(file.read())
            with open(
                os.path.join(template_dir, f"{self.client_type}.user.html"), "r"
            ) as file:
                self.user_template = jinja2.Template(file.read())
        except FileNotFoundError as e:
            self.logger.error(f"Template file not found: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"Error loading templates: {e}")
            raise e

    def _render_message(
        self, latest_messages: List[MessageParam], role: Role, **kwargs
    ) -> MessageParam:
        """
        Renders a message based on the role and the latest messages.

        :param latest_messages: List of the most recent ChatCompletionMessage objects.
        :param role: Role of the message to be rendered (SYSTEM or USER).
        :return: Rendered ChatCompletionMessage object.
        """
        if role == SystemMessage:
            return SystemMessage(
                content=self.system_template.render(
                    latest_messages=latest_messages, **kwargs
                ),
                role="system",
            )
        elif role == UserMessage:
            return UserMessage(
                content=self.user_template.render(
                    latest_messages=latest_messages, **kwargs
                ),
                role="user",
            )
        else:
            raise ValueError("MessageType must be SystemMessage or UserMessage")
