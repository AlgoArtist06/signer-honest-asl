#!/usr/bin/env python3
"""Execute experiments.ipynb cell by cell; stop on first error."""
from __future__ import annotations

import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB = Path(__file__).resolve().parent / "experiments.ipynb"
LOG = Path(__file__).resolve().parent / "_run_nb.log"


def log(msg: str) -> None:
    line = msg + "\n"
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(line)


def main() -> int:
    LOG.write_text("")
    nb = nbformat.read(NB, as_version=4)
    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="slr",
        resources={"metadata": {"path": str(NB.parent)}},
        allow_errors=False,
    )
    client.create_kernel_manager()
    client.start_new_kernel()
    client.start_new_kernel_client()
    try:
        n_code = 0
        for i, cell in enumerate(nb.cells):
            kind = cell.cell_type
            if kind != "code":
                log(f"[skip] cell {i} ({kind})")
                continue
            n_code += 1
            preview = "".join(cell.source).strip().splitlines()[:2]
            log(f"[run ] cell {i} (code #{n_code}): {preview}")
            try:
                client.execute_cell(cell, i)
            except CellExecutionError as e:
                log(f"[FAIL] cell {i}\n{e}")
                nbformat.write(nb, NB)
                return 1
            except Exception:
                log(f"[FAIL] cell {i}\n{traceback.format_exc()}")
                nbformat.write(nb, NB)
                return 1
            outs = cell.get("outputs") or []
            texts = []
            for o in outs:
                if o.get("output_type") in ("stream",):
                    texts.append(o.get("text", "")[:2000])
                elif o.get("output_type") == "error":
                    texts.append("\n".join(o.get("traceback", [])[-20:]))
            if texts:
                log("[out ] " + "\n".join(texts)[:3000])
            log(f"[ok  ] cell {i}")
            nbformat.write(nb, NB)
        log("[done] all cells")
        return 0
    finally:
        client._cleanup_kernel()


if __name__ == "__main__":
    raise SystemExit(main())
