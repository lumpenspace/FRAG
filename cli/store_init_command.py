import click
from frag.utils import console
from pathlib import Path
from typing import List, Tuple
from frag.settings import Settings
from frag.embeddings.store import EmbeddingStore


def init_store(items: Tuple[str]) -> None:
    # Load settings
    settings: Settings = Settings.from_path()
    console.log(f"Settings: {settings.model_dump()}")
    # Initialize the embedding store
    store: EmbeddingStore = EmbeddingStore.instance or EmbeddingStore.create(
        settings.embeds
    )

    urls: List[str] = []
    paths: List[str] = []

    # Determine if each item is a URL or a file path
    for item in items:
        if item.startswith("http://") or item.startswith("https://"):
            urls.append(item)
        elif Path(item).exists():
            paths.append(item)
        else:
            console.log(f"Invalid item detected, neither URL nor path: {item}")

    console.log(f"URLs: {urls}")
    console.log(f"Paths: {paths}")
    console.log(f"Embedding store: {store}")

    if len(urls) > 0:
        from frag.embeddings.ingest.ingest_url import URLIngestor

        ingestor = URLIngestor(store=store)
        ingestor.ingest(urls)

    # Further processing can be added here to handle URLs and paths with the store


@click.command("init:store")
@click.argument("items", nargs=-1, type=str)
def main(items: Tuple[str]) -> None:
    init_store(items=items)
