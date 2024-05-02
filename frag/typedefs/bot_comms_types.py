"""
Module containing types pertinent to bot communications.

It exports the relevant OpenAI Chat types, as well as a Note model -
used to pass information from the summarizer to the interface bot.
"""

# pylint: disable=unused-import
# pylance: disable=unused-import

from pydantic import BaseModel
from openai.types.chat import (  # noqa: F401
    ChatCompletionMessage as Message,
    ChatCompletionAssistantMessageParam as AssistantMessage,
    ChatCompletionMessageParam as MessageParam,
    ChatCompletionUserMessageParam as UserMessage,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionRole as Role,
    ChatCompletionFunctionMessageParam as FunctionMessage,
)
from openai.types.chat.completion_create_params import (  # noqa: F401
    CompletionCreateParamsBase as CompletionParams,
)


class Note(BaseModel):
    """
    A model representing a note.
    """

    id: str
    source: str
    title: str
    summary: str
    complete: bool


__all__: list[str] = [
    "Note",
    "Message",
    "AssistantMessage",
    "UserMessage",
    "MessageParam",
    "Role",
    "SystemMessage",
    "FunctionMessage",
    "CompletionParams",
]
