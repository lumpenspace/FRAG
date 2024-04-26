"""
Initialize the .fragrc file in the specified directory.
"""

import os
import click
from .utils import create_or_override


@click.command()
@click.option("--path", default=os.getcwd(), help="Directory path to initialize .fragrc")
def init(path):
    """
    Initialize the .fragrc file in the specified directory.
    """
    create_or_override(os.path.join(path, '.fragrc'), name= '.fragrc')
