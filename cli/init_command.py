import os
import click

import jinja2

from pathlib import Path

from frag.settings import Settings, SettingsDict
from frag.console import console

from .utils import create_or_override, section


@click.command()
@click.argument(
    "path",
    default=os.getcwd(),
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
def init(path: str) -> SettingsDict | None:
    """
    Initialize a new frag project

    Args:
        path (str): The path to the project's root
    """
    sections = section()

    sections.section(title="Configuring project settings")

    dir_path: str | None = create_or_override(path=path, name=".frag", dir=True)

    if dir_path is None:
        return

    default_settings: str = Settings.defaults_path.read_text()

    console.log("[b]creating default config file[/b]")
    Path(dir_path, "config.yaml").write_text(
        jinja2.Template(default_settings).render(
            {"frag_dir": Path(dir_path).relative_to(path)}
        )
    )

    console.log(f"created config file: {Path(dir_path, 'config.yaml')}")

    db_path: str | None = create_or_override(path=dir_path, name="db", dir=True)
    if db_path is None:
        return
    else:
        console.log(f"created db path: {db_path}")