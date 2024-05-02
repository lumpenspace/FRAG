"""
Module for settings
"""

# pylint: disable=unused-import
# pylint: disable=import-error
# flake8: noqa

from typing import Type

from chromadb import Documents

from .bot_comms_types import (
    AssistantMessage,
    CompletionParams,
    Message,
    MessageParam,
    Note,
    Role,
    SystemMessage,
    UserMessage,
)
from .embed_meta import DocMeta, RecordMeta
from .embed_settings import EmbedSettings
from .llm_settings import LLMSettings

Document = Type[str]
