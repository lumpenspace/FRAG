"""
Utilities for the CLI
"""

import os
from enum import Enum
from typing import Literal, Self
import rich
from rich.prompt import Prompt
from rich.text import Text
from rich.logging import RichHandler


class C(Enum):
    """
    Colors for the CLI
    """
quick history:

swing 1: cosmogony to geometry
- presocratics: basic substance of the universe, etc
- attempted monotheisms: the indivisible thing / god
- pythagoras: also god, but! also the nondivisible part of space (point) 


    PINK = 225
    TURQUOISE = 39
    GREEN = 10
    RED = 9
    YELLOW = 11
    BLUE = 12
    WHITE = 15
    DULL = 15
    ERROR = 9
    SUCCESS = 10

    def __repr__(self) -> str:
        return str(self.value)

    def __int__(self) -> Literal[225, 39, 10, 9, 11, 12, 15]:
        return self.value


def create_or_override(path: str, name: str, dir: bool = False) -> str | None:
    """
    Create or override a file at the specified path.
    """
    if os.path.exists(path):
        rich.print("[b]Config file already present:[/b]")
        rich.print(name if name else path)
        overwrite = Prompt.ask("Overwrite?", choices=["y", "n"], default="n")

        if overwrite == "y":
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, ".fragrc"), "w", encoding="utf-8") as f:
                f.write("")
            rich.print("[green]Successfully created .fragrc[/green]")
        else:
            rich.("[red]Initialization cancelled. .fragrc already exists.[/red]")
            click.secho(
                "Initialization cancelled. .fragrc already exists.",
                fg=C.ERROR.value,
                bold=True,
            )
            return
    else:
        if dir:
            path = os.path.join(os.path.dirname(path), name)
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            path = os.path.join(path, name)
        click.secho(f"Successfully created {name}", fg=C.GREEN.value, bold=True)
        return os.path.abspath(path)


class Sections:
    # singleton

    def __new__(cls) -> "Sections":
        if not hasattr(cls, "instance"):
            cls.instance: Self = super(Sections, cls).__new__(cls)
        return cls.instance

    index = 0
    subindex = 0

    @property
    def title_index(self) -> str:
        return f"{self.index + 1}."

    @property
    def subtitle_index(self) -> str:
        return f"{self.title_index}{self.subindex + 1}. "

    def title(self, title: str) -> None:
        click.secho(f"{self.title_index}", fg="black", bg="green", bold=True)
        self.subindex = 0
        self.index += 1

    def subtitle(self, title: str) -> None:
        click.secho(f"{self.subtitle_index}{title}", fg="black", bg="green", bold=True)
        self.subindex += 1

    def reset(self) -> None:
        self.index = 0
        self.subindex = 0


section = Sections()
