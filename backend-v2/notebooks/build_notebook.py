#!/usr/bin/env python3
"""Regenerate the notebooks.

The notebook is generated rather than hand-edited so its cells stay in sync with
`slr.experiment`, and so nobody has to diff JSON by hand. Run this after changing
the experiment stages, then commit the .ipynb.
"""

import json
from pathlib import Path

REPO = "https://github.com/AlgoArtist06/signer-honest-asl.git"
NOTEBOOKS = {"cells.json": "experiments.ipynb",
             "cells_video.json": "video_experiments.ipynb"}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": s.strip("\n").splitlines(keepends=True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


def build(src: str) -> dict:
    cells = json.loads(Path(__file__).with_name(src).read_text())
    out = []
    for kind, body in cells:
        out.append(md(body) if kind == "md" else code(body))
    return {"cells": out,
            "metadata": {"accelerator": "GPU",
                         "colab": {"provenance": [], "gpuType": "T4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    for src, dest in NOTEBOOKS.items():
        out = Path(__file__).with_name(dest)
        out.write_text(json.dumps(build(src), indent=1))
        print(f"wrote {out}")
