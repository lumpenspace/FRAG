"""

The main CLI entrypoint for frag.
"""

import click
from click.core import Group

from .init_command import main as init
from .test_settings_command import main as test_settings
from .store_init_command import main as store_init


@click.group()
def frag() -> None:
    """
    Main CLI group for frag.
    """


frag.add_command(init)
frag.add_command(test_settings)
frag.add_command(store_init)

main: Group = frag
