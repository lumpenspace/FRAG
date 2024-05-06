import os
import click
import rich
from rich.prompt import Prompt
from sympy import true
import yaml

from frag.settings import (
    Settings,
    EmbedAPISettings,
    LLMSettings,
    embed_api_settings,
    SettingsDict,
)
from frag.embeddings.apis import openai_embed_models, hf_embed_models
from frag.cli.utils import create_or_override, section
from frag.typedefs import ApiSource
from openai import OpenAI


@click.command()
@click.argument(
    "path",
    default=os.getcwd(),
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
def init(path: str | None) -> None:

    path = create_or_override(path=path or ".", name=".frag", dir=True)

    if path is None:
        return

    Settings.set_dir(path)

    defaults: SettingsDict = settings.defaults()

    section(
        title="Configuring embeddings settings",
        subtitle="Embedding will be stored in a local database",
        items=["db_path", "collection_name", "model_name"],
    )

    embed_provider = Prompt.ask(
        choices=[ApiSource.OpenAI, ApiSource.HuggingFace],
        default=ApiSource.OpenAI,
        prompt="API provider for embeddings:",
    )

    if embed_provider == "OpenAI":

        embed_model = select(
            openai_embed_models,
            cursor_index=openai_embed_models.index(embed_settings.model_name),
            prompt="Select OpenAI embedding model:",
        )
        embed_settings.model_name = f"oai:{embed_model}"
    else:
        embed_model = select(
            hf_embed_models, prompt="Select Hugging Face embedding model:"
        )
        embed_settings.model_name = embed_model

    embed_settings.db_path = click.prompt(
        "Embeddings DB path",
        default=embed_settings.db_path,
        type=click.Path(file_okay=False),
    )
    embed_settings.collection_name = click.prompt(
        "Default collection name", default=embed_settings.collection_name
    )

    # Validate embed settings
    embed_settings.validate()

    # LLMs
    llm_settings = settings.bots
    click.echo("Configuring LLM settings...")
    # TODO: Prompt for LLM settings and validate

    # Write config
    config = {"embed": embed_settings.model_dump(), "llm": llm_settings.model_dump()}

    create_or_override(os.path.join(path, ".frag"), name="config.yaml")

    click.echo(f"Config written to {config_path}")
