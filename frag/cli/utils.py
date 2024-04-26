"""
Utilities for the CLI
"""
import os
from enum import Enum

import click
from beaupy import select

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
    DULL= 15
    ERROR = 9
    SUCCESS = 10

    def __repr__(self):
        return str(self.value)
    def __int__(self):
        return self.value

def create_or_override(path: str, name: None | str = None):
    """
    Create or override a file at the specified path.
    """
    if os.path.exists(path):
        click.secho("Config file already present:", fg=C.PINK.value, bold=True)
        click.secho(name if name else path,  fg=C.DULL.value)
        click.secho("Override?", fg=C.PINK.value, bold=True, nl=True)
        if select(["yes", "no"], cursor_index=1) == "yes":

            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, '.fragrc'), 'w', encoding="utf-8") as f:
                f.write("")
            click.secho("Successfully created .fragrc", fg=C.SUCCESS.value, bold=True)
        else:
            click.secho("Initialization cancelled. .fragrc already exists.", fg=C.ERROR.value, bold=True)
    else:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, '.fragrc'), 'w', encoding="utf-8") as f:
            f.write("")
        click.secho("Successfully created .fragrc", fg=C.GREEN.value, bold=True)
