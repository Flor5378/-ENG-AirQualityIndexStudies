"""Build the v2 notebooks from their jupytext sources and execute them.

The analyses live in ``notebooks/py/*.py`` (percent format, diff-friendly); this
script turns each into ``notebooks/<name>.ipynb`` and runs it so that the
committed notebooks carry their outputs and figures.

    python build_notebooks.py            # all notebooks
    python build_notebooks.py 03         # only the notebooks whose name starts with 03
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def build(src: Path) -> Path:
    nb = jupytext.read(src)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = NB_DIR / f"{src.stem}.ipynb"
    t0 = time.time()
    NotebookClient(nb, timeout=-1, kernel_name="python3",
                   resources={"metadata": {"path": str(NB_DIR)}}).execute()
    nbformat.write(nb, out)
    print(f"{out.name}: executed in {time.time() - t0:.0f} s", flush=True)
    return out


def main(argv: list[str]) -> int:
    prefix = argv[0] if argv else ""
    for src in sorted((NB_DIR / "py").glob(f"{prefix}*.py")):
        build(src)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
