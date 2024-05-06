"""
Full settings for the frag system.

They are by default read from a .fragrc file, which you can generate by running:

```sh
frag init
```
"""

from typing import Dict, Any
from typing_extensions import TypedDict
from pathlib import Path
from logging import getLogger, Logger
import yaml
from pydantic_settings import BaseSettings

from .llm_settings import LLMSettings
from .db_settings import DBSettings, DBSettingsDict
from .embed_api_settings import EmbedAPISettings, EmbedApiSettingsDict
from .chunker_settings import ChunkerSettings

EmbedApiSettingsDict = EmbedApiSettingsDict
DBSettingsDict = DBSettingsDict

SettingsDict = TypedDict(
    "SettingsDict",
    {
        "db": None | DBSettingsDict,
        "embed_api": None | EmbedApiSettingsDict,
        "chunker": None | Dict[str, Any],
        "bots": None | Dict[str, Any],
    },
)

logger: Logger = getLogger(__name__)


class Settings(BaseSettings):
    """
    Read settings from a .frag file or load them programmatically, and validate them.

    those settings include:
    """

    db: DBSettings | DBSettingsDict
    embed_api: EmbedAPISettings | EmbedApiSettingsDict
    chunker: ChunkerSettings
    bots: LLMSettings

    _frag_dir: Path | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Settings":
        if not hasattr(cls, "instance"):
            logger.debug("Creating a new instance of the Settings class.")
            cls._instance: Settings | None = super().__new__(cls)
        if cls._instance is None:
            raise ValueError("The instance of the Settings class is None.")
        return cls._instance

    @classmethod
    def set_dir(cls, frag_dir: str) -> None:
        """
        Set the directory in which the fragrc file is located.
        """
        cls._frag_dir = Path(frag_dir)

    @classmethod
    @property
    def frag_dir(cls) -> Path:
        """
        Get the directory in which the fragrc file is located.
        """
        if cls._frag_dir is None:
            raise ValueError("The frag directory is not set.")
        return cls._frag_dir

    @classmethod
    def from_dict(cls, settings_path: Path | None) -> "Settings":
        """
        Load settings from a dictionary and merge them with the default settings.
        """
        if hasattr(cls, "instance"):
            logger.debug("Returning the existing instance of the Settings class.")
            if cls._instance is None:
                raise ValueError("The instance of the Settings class is None.")
            return cls._instance

        default_settings: SettingsDict = yaml.safe_load(".fragrc_default")

        if settings_path is not None and settings_path.exists():
            settings: SettingsDict = yaml.safe_load(settings_path.read_text())
        else:
            settings: SettingsDict = {}

        bots: LLMSettings = LLMSettings.from_dict(settings.get("bots", {}))

        embed_api: EmbedAPISettings = EmbedAPISettings.from_dict(
            settings.get("embedAPI", default_settings.get("embedAPI", {}))
        )
        chunker: ChunkerSettings = ChunkerSettings.from_dict(
            {
                **settings.get("chunker", default_settings.get("chunker", {})),
                "api_max_tokens": embed_api.max_tokens,
            }
        )
        db: DBSettings = DBSettings.from_dict(
            settings.get(
                "db", default_settings.get("db", default_settings.get("db", {}))
            )
        )
        instance: Settings = cls(db=db, embed_api=embed_api, chunker=chunker, bots=bots)
        cls._instance = instance
        return instance

    @classmethod
    def from_rc(cls, path: Path) -> "Settings":
        """
        Create a new instance of the class from a .fragrc file.
        """
        if cls._instance is not None:
            logger.debug("Returning the existing instance of the Settings class.")
            return cls._instance
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        with open(path, "r", encoding="utf-8") as f:
            rc: str = f.read()
        return cls.from_dict(yaml.safe_load(Path(rc).read_text()))

    @classmethod
    def defaults(cls) -> SettingsDict:
        """
        Return the default settings for the class.
        """
        return yaml.safe_load(".fragrc_default")

    @classmethod
    def reset(cls) -> None:
        """
        Reset Settings instance, allowing for a new instance to be created
          the next time it is accessed.
        """
        cls._frag_dir = None
        cls._instance = None
