#!/usr/bin/env python3
"""Regenerate experiments.ipynb.

The notebook is generated rather than hand-edited so its cells stay in sync with
`slr.experiment`, and so nobody has to diff JSON by hand. Run this after changing
the experiment stages, then commit the .ipynb.
"""

import json
from pathlib import Path

REPO = "https://github.com/AlgoArtist06/signer-honest-asl.git"
OUT = Path(__file__).with_name("experiments.ipynb")


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": s.strip("\n").splitlines(keepends=True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


def build() -> dict:
    cells = json.loads(Path(__file__).with_name("cells.json").read_text())
    out = []
    for kind, body in cells:
        out.append(md(body) if kind == "md" else code(body))
    return {"cells": out,
            "metadata": {"kernelspec": {"display_name": "Python 3.12 (slr)",
                                        "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"},
                         "hardware": {"machine": "MacBook Pro",
                                      "chip": "Apple M4 Pro",
                                      "cpu_cores": "12 (8P+4E)",
                                      "gpu_cores": 16,
                                      "memory_gb": 24,
                                      "torch_device": "mps"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1))
    print(f"wrote {OUT}")
