import os
import click

from .embed_api_init import embed_api_init
from .embed_db_init import embed_db_init

from frag.settings import (
    Settings,
    EmbedApiSettingsDict,
    DBSettingsDict,
    SettingsDict,
)

from rich.console import Console

from ..utils import create_or_override, section

console = Console()


@click.command()
@click.argument(
    "path",
    default=os.getcwd(),
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
def init(path: str | None) -> SettingsDict | None:

    path = create_or_override(path=path or ".", name=".frag", dir=True)

    if path is None:
        return

    Settings.set_dir(path)

    sections = section()

    sections.section(title="Configuring embeddings settings")

    sections.subsection("Embeddings API")
    api_settings: EmbedApiSettingsDict = embed_api_init(path=path)

    sections.subsection("Embeddings DB")
    db_settings: DBSettingsDict = embed_db_init(
        path=path,
    )

    settings = SettingsDict(db=db_settings, embed_api=api_settings, chunker={}, bots={})

    sections.section(title="Configuting generation settings")

    return settings
