"""
Demo CLI wrapper entrypoint for the sound-recognition demo.

This script provides a simple entry point for the example/demo. It calls
the existing `main()` function from `main.py` so the package can expose a
`sound-demo` console script in `pyproject.toml`.
"""
from main import main


def main_cli():
    return main()


if __name__ == "__main__":
    main()
