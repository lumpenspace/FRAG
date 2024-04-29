from pydantic import model_validator
from pydantic_settings import BaseSettings
from litellm.types.completion import CompletionRequest
from litellm.utils import get_supported_openai_params, get_model_info

import os
import yaml
from pathlib import Path
from typing import Literal


class DBSettings(BaseSettings):
    db_path: str = os.path.join(os.path.dirname(__file__), "db")
    default_collection: str = "default_collection"


class EmbedSettings(BaseSettings):
    model: str = "oai:text-embedding-3-large"
    chunk_size: int = 10000
    chunk_overlap: int = 0


class ChunkerSettings(EmbedSettings):
    preserve_sentences: bool = False
    preserve_paragraphs: bool = True
    max_length: int = 512
    buffer_before: int = 0
    buffer_after: int = 0


class LLMSettings(CompletionRequest):
    llm: str = "gpt-3.5-turbo"
    interface_bot: "ResponderSettings"
    summarizer_bot: "ResponderSettings"

    def bot_settings(self, bot_type: Literal["interface", "summarizer"]):
        return self[f"{bot_type}_bot"].model_dump(exclude_unset=True, exclude_none=True)

    def dump(self):
        return self.model_dump(exclude_unset=True, exclude_none=True)

    def __getitem__(self, item):
        return self.model_dump(exclude_unset=True, exclude_none=True)[item]

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, values):
        other_values = {k: v for k, v in values.items() if k != "llm"}
        # check if values are supported
        model_string = values.get("model", "gpt-3.5-turbo")
        info = get_model_info(model_string)
        supported_params = get_supported_openai_params(
            model_string, info.get("litellm_provider", None)
        )
        for k, v in other_values.items():
            if k not in supported_params:
                raise ValueError(
                    f"Unsupported parameter {k} for model {values.get('model', 'gpt-3.5-turbo')}"
                )
        return values


class ResponderSettings(LLMSettings):
    template_dir: str = os.path.join(os.path.dirname(__file__), "templates")


class Settings(BaseSettings):
    db: DBSettings = DBSettings()
    embed: EmbedSettings = EmbedSettings()
    chunker: ChunkerSettings = ChunkerSettings()
    summarizer: LLMSettings
    interface: LLMSettings

    @property
    def embed_model(self):
        return self.embed.model

    @property
    def preserve_paragraphs(self):
        return self.chunker.preserve_paragraphs

    @property
    def max_length(self):
        return self.chunker.max_length

    @property
    def buffer_before(self):
        return self.chunker.buffer_before

    @property
    def buffer_after(self):
        return self.chunker.buffer_after

    @property
    def db_path(self):
        return self.db.db_path

    @property
    def default_collection(self):
        return self.db.default_collection

    @model_validator(mode="before")
    @classmethod
    def load_default_settings(cls, values: dict) -> dict:
        """
        Load settings from .fragrc file if it exists and merge with default settings using
         pathlib for better path handling.
        """
        current_dir = Path(__file__).parent
        while True:
            fragrc_path = current_dir / ".fragrc"
            if fragrc_path.is_file():
                with open(fragrc_path, "r", encoding="utf-8") as file:
                    fragrc_settings = yaml.safe_load(file) or {}
                    for section, settings in fragrc_settings.items():
                        if section in values:
                            for key, value in settings.items():
                                values[section][key] = value.get(
                                    key, values[section].get(key)
                                )
                break
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # reached the root directory
                break
            current_dir = parent_dir
        return values
