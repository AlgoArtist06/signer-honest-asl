"""Score `leakage.recover_groups` against ground-truth signer ids.

`sl_digits` is the one corpus where the true signer is recoverable exactly (the
camera's frame counter encodes it, see `sources.group_sl_digits`), which makes
it the only place the near-duplicate recovery used everywhere else can be
checked rather than asserted. Run it over a ladder of thresholds and report how
badly each one fragments the signers it is supposed to reconstruct.

    python -m slr.signer_check dataset/manifests/sl_digits.csv

The headline in README - every signer fragments at any usable threshold - is
this script's output, not a claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import leakage, sources

LADDER = (2, 6, 10, 12)


def score(manifest: Path, ladder=LADDER) -> list[dict]:
    truth = sources.read_manifest(manifest)
    n_signers = len({r["group"] for r in truth})
    print(f"[truth] {len(truth)} images, {n_signers} signers from the frame counter\n")

    out = []
    for max_dist in ladder:
        rows = [dict(r) for r in truth]
        leakage.recover_groups(rows, max_dist=max_dist, workers=8)
        by_signer: dict[str, set[str]] = {}
        for t, r in zip(truth, rows):
            by_signer.setdefault(t["group"], set()).add(r["group"])
        fragmented = sum(1 for v in by_signer.values() if len(v) > 1)
        out.append({
            "max_dist": max_dist,
            "components": len({r["group"] for r in rows}),
            "signers_fragmented": fragmented,
            "signers": n_signers,
        })
        print(f"  max_dist={max_dist:<3} components={out[-1]['components']:<6} "
              f"fragmented={fragmented}/{n_signers}")

    print(f"\n{'max_dist':>10}{'components':>14}{'signers fragmented':>22}")
    print("-" * 46)
    for r in out:
        frac = f"{r['signers_fragmented']} / {r['signers']}"
        print(f"{r['max_dist']:>10}{r['components']:>14}{frac:>22}")
    return out


if __name__ == "__main__":
    score(Path(sys.argv[1] if len(sys.argv) > 1
               else "dataset/manifests/sl_digits.csv"))
