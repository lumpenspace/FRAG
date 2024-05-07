import os
import click
import json

from pathlib import Path
from rich.prompt import Prompt
from rich.panel import Panel
from rich.pretty import Pretty

from .utils import C

from frag.settings import Settings
from frag.utils.console import console, error_console


def test_settings(
    path: str, json_string: str | None = None, return_settings: bool = False
) -> Settings | None:
    """
    Test the settings class

    Args:
        path (str): The path to the config file to test
        json_string (str, optional): The json string to test
    """
    settings: Settings | None = None
    if json_string is not None:
        settings = Settings.from_dict(json.loads(json_string))
    else:
        if Path(path).exists():

            settings = Settings.from_path(path)
        else:
            console.print(
                f"🤷 [{C.WARNING.value}]Settings path does not exist[/]:[dark_orange] {path} [/]\n",
                emoji=True,
            )

            overwrite: str = Prompt.ask(
                "👷 Create a new config?", choices=["y", "n"], default="y"
            )

            if overwrite != "y":
                error_console.log(
                    f"[bold {C.ERROR.value}]Test cancelled. Create a new config with\
                        [code]frag init[/code][/]"
                )
                return
            else:
                console.log("Creating a new config...")
                from .init_command import init

                init(path=path)

    if settings:
        console.log(Panel(Pretty(settings.model_dump()), title="Settings"))
        if return_settings:
            return settings


@click.command("test:settings")
@click.argument(
    "path",
    default=Path(os.getcwd(), ".frag"),
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option("--json_string", "-j", default=None, type=click.STRING)
def main(path: str, json_string: str | None = None) -> None:
    test_settings(path=path, json_string=json_string)
