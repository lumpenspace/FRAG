"""
Full settings for the frag system.

They are by default read from a .fragrc file, which you can generate by running:

```sh
frag init
```
"""

import os
from typing import Dict, Any, Self
from typing_extensions import TypedDict
from pathlib import Path
import yaml
import json
from pydantic_settings import BaseSettings

from .bots_settings import BotsSettings
from .db_settings import DBSettings, DBSettingsDict
from .embed_api_settings import EmbedAPISettings, EmbedApiSettingsDict
from .chunker_settings import ChunkerSettings
from frag.console import console, error_console, live


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


class Settings(BaseSettings):
    """
    Read settings from a .frag file or load them programmatically, and validate them.

    those settings include:
    """

    db: DBSettings | DBSettingsDict
    embed_api: EmbedAPISettings | EmbedApiSettingsDict
    chunker: ChunkerSettings
    bots: BotsSettings

    _frag_dir: Path | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Settings":
        if not hasattr(cls, "instance"):
            console.log("Creating a new instance of the Settings class.")
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
    def defaults_path(cls) -> Path:
        dir_path: str = os.path.dirname(os.path.realpath(__file__))
        return Path(dir_path) / ".fragrc_default.yaml"

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
    def from_path(cls, frag_dir: str) -> Self:
        """
        Load settings from a dictionary and merge them with the default settings.
        """
        cls.set_dir(frag_dir)
        if hasattr(cls, "instance"):
            console.log("Returning the existing instance of the Settings class.")
            if cls._instance is None:
                raise ValueError("The instance of the Settings class is None.")
            return cls._instance

        if Path(cls.frag_dir, "config.yaml").exists():
            console.log(f"Config found in: {Path(cls.frag_dir, 'config.yaml')}")
            settings: SettingsDict = yaml.safe_load(
                Path(cls.frag_dir, "config.yaml").read_text()
            )
        else:
            console.log(
                f"No config found in: {Path(cls.frag_dir, 'config')}, using defaults"
            )
            settings: SettingsDict = {
                "db": None,
                "embed_api": None,
                "chunker": None,
                "bots": None,
            }
        try:
            with live(console=console):
                result: Self = cls.from_dict(settings)

            return result
        except ValueError as e:
            error_console.log(f"Error validating settings: \n {json.dumps(settings)}")
            raise e

    @classmethod
    def from_dict(cls, settings: SettingsDict) -> Self:
        default_settings: SettingsDict = yaml.safe_load(cls.defaults_path.read_text())

        embed_api: EmbedAPISettings | None = None
        chunker: ChunkerSettings | None = None
        bots: BotsSettings | None = None
        db: DBSettings | None = None
        console.log("[b]Validating:[/b]")
        try:
            bots = BotsSettings.from_dict(settings.get("bots", {}))
        except ValueError as e:
            error = f"Error getting bot settings for:\n\
                    {json.dumps(settings.get('bots', {}))}"
            error_console.log(f"Error: {error}\n\n {e}\n")
        if bots is None:
            raise ValueError("Bots settings are required")

        try:
            embed_api = EmbedAPISettings.from_dict(
                settings.get("embed_api", default_settings.get("embed_api", {}))
            )
        except ValueError as e:
            error: str = (
                f"Error getting embed_api settings for:\n\
                    {json.dumps(settings.get('embed_api', {}))}"
            )
            error_console.log(f"Error: {error}\n\n {e}\n")
        if embed_api is None:
            raise ValueError("Embed API settings are required")

        try:
            chunker = ChunkerSettings.from_dict(
                settings.get("chunker", default_settings.get("chunker", {})),
                embed_api.max_tokens,
            )
        except ValueError as e:
            error: str = (
                f"Error getting chunker settings for:\n\
                    {json.dumps(settings.get('chunker', {}))}"
            )
            error_console.log(f"Error: {error}\n\n {e}\n")
        if chunker is None:
            raise ValueError("Chunker settings are required")

        try:
            db = DBSettings.from_dict(
                settings.get(
                    "db", default_settings.get("db", default_settings.get("db", {}))
                )
            )
        except ValueError as e:
            error: str = (
                f"Error getting db settings for:\n\
                    {json.dumps(settings.get('db', {}))}"
            )
            error_console.log(f"Error: {error}\n\n {e}\n")
        if db is None:
            raise ValueError("DB settings are required")

        instance: Settings = cls(db=db, embed_api=embed_api, chunker=chunker, bots=bots)
        cls._instance = instance
        console.log("[b][green]Success![/green][/b]")
        return instance

    @classmethod
    def from_rc(cls, path: Path) -> "Settings":
        """
        Create a new instance of the class from a .fragrc file.
        """
        if cls._instance is not None:
            console.log("Returning the existing instance of the Settings class.")
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
