"""Command-line entry point for AIVIA."""

from aivia import __version__


def main() -> int:
    """Print the installed version and exit successfully."""
    print(f"aivia {__version__}")
    return 0
