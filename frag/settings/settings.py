"""
Full settings for the frag system.

They are by default read from a .fragrc file, which you can generate by running:

```sh
frag init
```
"""

import os
import pickle
from typing import Dict, Any, Self
from typing_extensions import TypedDict
from pathlib import Path
import yaml
import json
from pydantic_settings import BaseSettings

from frag.utils.singleton import SingletonMixin

from .bots_settings import BotsSettings
from .embed_settings import EmbedSettings, EmbedSettingsDict
from frag.utils.console import console, error_console, live


EmbedSettingsDict = EmbedSettingsDict

SettingsDict = TypedDict(
    "SettingsDict",
    {
        "embeds": None | EmbedSettingsDict,
        "bots": None | Dict[str, Any],
    },
)


class Settings(SingletonMixin, BaseSettings):
    """
    Read settings from a .frag file or load them programmatically, and validate them.

    those settings include:
    """

    embeds: EmbedSettings
    bots: BotsSettings

    _frag_dir: Path | None = None

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
    def pickle_path(cls) -> Path:
        return Path(cls.frag_dir, "config.lock")

    def save_lock(self) -> None:
        """
        Save the current settings to a lock file.
        """
        with open(self.pickle_path(), "wb") as f:
            pickle.dump(self, f)
            console.log(f"Saved config lock to: {self.pickle_path()}")

    @classmethod
    def load_lock(cls) -> None:
        """
        Load the current settings from a lock file.
        """
        if cls.pickle_path().exists():
            console.log(f"Config lock found in: {cls.pickle_path()}")
            try:
                return pickle.load(cls.pickle_path().open("rb"))
            except ValueError as e:
                error_console.log(
                    f"Error loading config lock: {e}\n\n Trying to load from .yaml"
                )
        return None

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

        lock: None | Settings = cls.load_lock()
        if lock is not None:
            return lock

        settings: SettingsDict = {
            "embeds": None,
            "bots": None,
        }

        if Path(cls.frag_dir, "config.yaml").exists():
            console.log(f"Config found in: {Path(cls.frag_dir, 'config.yaml')}")
            settings = yaml.safe_load(Path(cls.frag_dir, "config.yaml").read_text())
        else:
            console.log(
                f"No config found in: {Path(cls.frag_dir, 'config')}, using defaults"
            )
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

        embeds: EmbedSettings | None = None
        bots: BotsSettings | None = None
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
            embeds = EmbedSettings.from_dict(
                settings.get("embeds", default_settings.get("embeds", {}))
            )
        except ValueError as e:
            error: str = (
                f"Error getting embed_api settings for:\n\
                    {json.dumps(settings.get('embeds', {}))}"
            )
            error_console.log(f"Error: {error}\n\n {e}\n")
        if embeds is None:
            raise ValueError("Embed API settings are required")

        instance: Settings = cls(embeds=embeds, bots=bots)
        cls._instance = instance
        console.log("[b][green]Success![/green][/b]")
        instance.save_lock()
        return instance

    @classmethod
    def defaults(cls) -> SettingsDict:
        """
        Return the default settings for the class.
        """
        return yaml.safe_load(".fragrc_default")
