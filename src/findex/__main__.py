"""``python -m findex`` == the ``findex`` command (handy under ``python -m cProfile``)."""

from findex.cli import app

if __name__ == "__main__":  # required: under "spawn", worker processes re-import this module
    app()
