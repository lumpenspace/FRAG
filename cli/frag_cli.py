"""

The main CLI entrypoint for frag.
"""

import click

from .init_command import init

from .test_settings_command import test_settings


@click.group()
def frag() -> None:
    """
    Main CLI group for frag.
    """


frag.add_command(init)
frag.add_command(test_settings)

main = frag
