import os
import click

import jinja2
import shutil
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
    if db_path is not None:
        console.log(f"created db path: {db_path}")

    console.log("[b]copying default templates[/b]")

    templates_src_path: Path = Path(__file__).parent.parent / "frag" / "templates"
    templates_dest_path: Path = Path(dir_path) / "templates"

    for src_file in templates_src_path.glob("*"):
        dest_file: Path = templates_dest_path / src_file.name
        if dest_file.exists():
            dest_file.unlink()
        shutil.copy(src_file, dest_file)
