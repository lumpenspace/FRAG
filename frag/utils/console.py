from rich.console import Console
from rich.live import Live

console = Console()
error_console = Console(stderr=True)


def live(console: Console) -> Live:
    return Live(console=console)
