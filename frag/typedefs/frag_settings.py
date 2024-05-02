"""
Full settings for the frag system.

They are by default read from a .fragrc file, which you can generate by running:

```sh
frag init
```
"""

from typing import TypedDict, Dict, Any
from pathlib import Path
from logging import getLogger, Logger
import yaml
from pydantic_settings import BaseSettings

from .embed_settings import EmbedSettings
from .llm_settings import LLMSettings


SettingsDict = TypedDict(
    "SettingsDict",
    {"embed": Dict[str, Any], "llm": Dict[str, Any]},
)

logger: Logger = getLogger(__name__)


class Settings(BaseSettings):
    """
    Read settings from a .frag file or load them programmatically, and validate them.

    those settings include:
    """

    embed: EmbedSettings
    llm: LLMSettings

    def __new__(cls, *args: Any, **kwargs: Any) -> "Settings":
        if not hasattr(cls, "instance"):
            logger.debug("Creating a new instance of the Settings class.")
            cls._instance: Settings | None = super().__new__(cls)
        if cls._instance is None:
            raise ValueError("The instance of the Settings class is None.")
        return cls._instance

    @classmethod
    def from_dict(cls, settings: SettingsDict) -> "Settings":
        """
        Load settings from a dictionary and merge them with the default settings.
        """
        if hasattr(cls, "instance"):
            logger.debug("Returning the existing instance of the Settings class.")
            if cls._instance is None:
                raise ValueError("The instance of the Settings class is None.")
            return cls._instance

        default_settings: SettingsDict = yaml.safe_load(".fragrc_default")
        merged_settings: SettingsDict = {**default_settings, **settings}

        embed: EmbedSettings = EmbedSettings(**merged_settings.get("embed", {}))
        llm: LLMSettings = LLMSettings.from_dict(merged_settings.get("llm", {}))
        instance: Settings = cls(embed=embed, llm=llm)
        cls._instance = instance
        return instance

    @classmethod
    def from_rc(cls, path: Path) -> "Settings":
        """
        Create a new instance of the class from a .fragrc file.
        """
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        with open(path, "r", encoding="utf-8") as f:
            rc: str = f.read()
        return cls.from_dict(yaml.safe_load(rc))

    @classmethod
    def reset(cls) -> None:
        """
        Reset Settings instance, allowing for a new instance to be created
          the next time it is accessed.
        """
        cls._instance = None
