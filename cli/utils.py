"""
Utilities for the CLI
"""

import os
from enum import Enum
from typing import Literal, Self
from rich.prompt import Prompt
from rich.panel import Panel
from frag.console import console


class C(Enum):
    """
    Colors for the CLI
    """

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
    check_path: str = os.path.join(path, name)
    abs_path: str = os.path.abspath(check_path)
    if os.path.exists(check_path):
        console.log(
            f"[b]{'directory' if dir else 'file'} already present:[/b] {abs_path}"
        )
        overwrite: str = Prompt.ask("Overwrite?", choices=["y", "n"], default="n")

        if overwrite != "y":
            console.log(
                f"[bold {C.ERROR.value}]Initialization cancelled. .fragrc already exists.[/]"
            )
            console.log(
                f"[bold {C.ERROR.value}]Initialization cancelled. .fragrc already exists.[/]"
            )
            return

    console.log(f"[b]Creating {check_path}[/b]")
    os.makedirs(path, exist_ok=True)
    if dir:
        os.makedirs(check_path, exist_ok=True)
    else:
        with open(check_path, "w", encoding="utf-8") as f:
            f.write("")
    console.print(f"[bold {C.GREEN.value}]Successfully created {name}[/]")
    return check_path


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

    def section(self, title: str, subtitle: str | None = None) -> None:
        self.title(title)
        if subtitle:
            self.subtitle(subtitle)

    def subsection(self, title: str) -> None:
        self.subtitle(title)

    def title(self, title: str) -> None:
        console.print(Panel(f"[bold green]{self.title_index}[/][bold]{title}[/]"))
        self.subindex = 0
        self.index += 1

    def subtitle(self, title: str) -> None:
        console.print(f"[bold green]{self.subtitle_index}[/][bold]{title}[/]")
        self.subindex += 1

    def reset(self) -> None:
        self.index = 0
        self.subindex = 0


section = Sections
