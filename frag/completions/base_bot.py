"""
Base API client, from which both the prompter and the summariser inherit.
"""

import os
from typing import List, Dict, Any, Literal

import jinja2
from litellm.main import ModelResponse, completion

from frag.typedefs import MessageParam, Role, SystemMessage, UserMessage
from frag.settings import BotModelSettings
from frag.utils.console import error_console


class BaseBot:
    """
    Base API client class that handles interactions with the LLM model.
    """

    settings: BotModelSettings
    client_type: Literal["interface", "summarizer"] | None = None
    system_template: jinja2.Template
    user_template: jinja2.Template
    messages: List[MessageParam]
    responder: str

    def __init__(self, settings: BotModelSettings, template_dir: str) -> None:
        """
        Initializes the BaseApiClient with the given settings, template paths, and client type.

        :param settings: LLMSettings object containing configuration for the model.
        :param system_template_path: Path to the system message template file.
        :param user_template_path: Path to the user message template file.
        :param client_type: Type of the client, used to determine the template files.
        """
        if self.client_type is None:
            raise ValueError("client_type must be provided")
        self.settings: BotModelSettings = settings
        self.load_templates(template_dir)

    def run(
        self, messages: List[MessageParam], **kwargs: Dict[str, Any]
    ) -> ModelResponse:
        """
        Processes the given messages and performs completion using the LLM model.

        :param messages: List of ChatCompletionMessage objects to be processed.
        """
        try:
            rendered_messages: List[MessageParam] = [
                msg for msg in self._render(messages, **kwargs)
            ]

            return ModelResponse(
                completion(
                    model=self.settings.api,
                    messages=rendered_messages,
                    **self.settings.completion_kwargs,
                )
            )
        except Exception as e:
            error_console.log(f"Error during completion: {e}")
            raise

    def _render(
        self, messages: List[MessageParam], **kwargs: Dict[str, Any]
    ) -> List[MessageParam]:
        """
        Renders the messages based on the templates. This method should be implemented
        by subclasses.

        :param messages: List of ChatCompletionMessage objects to be rendered.
        :return: List of rendered ChatCompletionMessage objects.
        """
        raise NotImplementedError

    def load_templates(self, template_dir: str) -> None:
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
                os.path.join(template_dir, f"{self.client_type}.system.html"),
                "r",
                encoding="utf-8",
            ) as file:
                self.system_template = jinja2.Template(file.read())
            with open(
                os.path.join(template_dir, f"{self.client_type}.user.html"),
                "r",
                encoding="utf-8",
            ) as file:
                self.user_template = jinja2.Template(file.read())
        except FileNotFoundError as e:
            error_console.log("Template file not found: %s", e)
            raise e
        except Exception as e:
            error_console.log("Error loading templates: %s", e)
            raise e

    def _render_message(
        self, latest_messages: List[MessageParam], role: Role, **kwargs: Dict[str, Any]
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
