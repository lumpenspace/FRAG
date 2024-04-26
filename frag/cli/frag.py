"""
The main CLI entrypoint for frag.
"""
import click

from .init_command import init


@click.group()
def frag():
    """
    Main CLI group for frag.
    """

frag.add_command(init)

main = frag
