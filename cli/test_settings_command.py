import os
import click
import json

from pathlib import Path
from rich.panel import Panel
from rich.pretty import Pretty

from frag.settings import Settings
from frag.console import console


@click.command()
@click.argument(
    "path",
    default=Path(os.getcwd(), ".frag"),
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option("--json_string", "-j", default=None, type=click.STRING)
def test_settings(path: str, json_string: str | None = None) -> None:
    """
    Test the settings class

    Args:
        path (str): The path to the config file to test
        json_string (str, optional): The json string to test
    """
    if json_string is not None:
        settings: Settings = Settings.from_dict(json.loads(json_string))
    else:
        settings: Settings = Settings.from_path(path)
    console.log(Panel(Pretty(settings.model_dump()), title="Settings"))
