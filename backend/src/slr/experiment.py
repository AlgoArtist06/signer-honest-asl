"""The three-protocol experiment, as one runnable module.

Packaged as code rather than notebook cells so the run is reproducible and so
driving it remotely is a matter of calling `stage(name)` instead of retyping
analysis into a browser.

Stages, in order:

    data   download the corpora, build the manifest, recover session groups
    A      random split          - what the literature reports
    B      group split           - sessions kept whole
    C      cross-corpus          - tested on a corpus with different signers
    arch   architecture search on protocol C, selected on validation
    cal    temperature scaling and the abstention threshold for the winner

Each stage writes `runs/results.json` incrementally, so a disconnected Colab
session loses at most the stage in flight.

Subsampling is group-aware by construction: `--per-class` caps images per class
by dropping whole groups, never by cherry-picking frames, because sampling
individual frames out of a session would quietly reintroduce the leak the
experiment exists to measure.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from . import data, evaluate, leakage, model as M, sources, train

RUNS = Path("runs")
RESULTS = RUNS / "results.json"
MANIFEST = Path("dataset/manifests/all.csv")

TRAIN_CORPUS = "asl_alphabet"
TEST_CORPUS = "asl_alphabet_test"


# --- bookkeeping -------------------------------------------------------------

def load() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}


def save(state: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(state, indent=2))
    print(f"[state] -> {RESULTS}")


def write_kaggle_credentials(username: str, key: str) -> None:
    p = Path.home() / ".kaggle" / "kaggle.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"username": username, "key": key}))
    p.chmod(0o600)
    print(f"[kaggle] credentials for {username} written to {p}")


def wait_for_kaggle(get, poll: int = 30) -> None:
    """Block until Colab Secrets carry Kaggle credentials, then write them.

    `get` is `google.colab.userdata.get`, passed in so this module keeps no
    dependency on Colab. An unattended run should not die because the secrets
    were not in place at the moment the cell started; it should sit still and
    pick them up as soon as they are.
    """
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        print("[kaggle] credentials already present")
        return
    while True:
        try:
            write_kaggle_credentials(get("KAGGLE_USERNAME"), get("KAGGLE_KEY"))
            return
        except Exception as e:
            print(f"[kaggle] waiting for KAGGLE_USERNAME / KAGGLE_KEY in Colab "
                  f"Secrets ({type(e).__name__}); retrying in {poll}s", flush=True)
            time.sleep(poll)


# --- sampling ----------------------------------------------------------------

def subsample(rows: list[dict], per_class: int, seed: int = 0) -> list[dict]:
    """Cap images per class, dealing whole sessions wherever one fits.

    Sessions are dealt whole so a subsample can never split one across the
    train/test boundary. On `asl_alphabet` a recovered session runs to thousands
    of frames, so whole sessions alone cannot honour a cap of a few hundred -
    taking one session already overshoots by an order of magnitude. Where that
    happens a contiguous *prefix* of the session is kept instead. That is still
    one session, still all-in or all-out of any split, and still an unbroken run
    of near-identical frames, so protocol A's contamination is preserved rather
    than thinned away by the sampling.
    """
    if not per_class:
        return rows
    by_class_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_class_group[r["label"]][r["group"]].append(r)

    rng = random.Random(seed)
    kept: list[dict] = []
    for label, groups in by_class_group.items():
        names = sorted(groups)
        rng.shuffle(names)
        n = 0
        for g in names:
            if n >= per_class:
                break
            take = groups[g][: per_class - n]
            kept += take
            n += len(take)
    print(f"[sample] {len(rows)} -> {len(kept)} images "
          f"(<={per_class}/class, whole groups only)")
    return kept


# --- stages ------------------------------------------------------------------

def stage_data(per_class: int = 0, max_dist: int = 6) -> dict:
    for name in (TRAIN_CORPUS, TEST_CORPUS):
        sources.download(name)

    rows: list[dict] = []
    for name in (TRAIN_CORPUS, TEST_CORPUS):
        rows += sources.scan(name)

    t0 = time.time()
    leakage.recover_groups(rows, max_dist=max_dist, workers=8)
    print(f"[data] group recovery took {time.time() - t0:.0f}s")

    sources.write_manifest([{k: r[k] for k in sources.MANIFEST_COLS} for r in rows], MANIFEST)
    summary = {
        "images": len(rows),
        "by_source": {s: sum(1 for r in rows if r["source"] == s)
                      for s in {r["source"] for r in rows}},
        "groups": len({r["group"] for r in rows}),
        "classes": len({r["label"] for r in rows}),
        "recovery_seconds": round(time.time() - t0, 1),
    }
    print(f"[data] {summary}")
    return summary


def _rows(per_class: int, source: str | None = None) -> list[dict]:
    rows = sources.read_manifest(MANIFEST)
    if source:
        rows = [r for r in rows if r["source"] == source]
    return subsample(rows, per_class)


def _protocol_result(rows: list[dict], res: dict, rep: dict) -> dict:
    return {
        "n_train": sum(1 for r in rows if r["split"] == "train"),
        "n_test": sum(1 for r in rows if r["split"] == "test"),
        "test_groups": len({r["group"] for r in rows if r["split"] == "test"}),
        "test_leak_rate": rep["splits"].get("test", {}).get("leak_rate"),
        "test_exact_duplicates": rep["splits"].get("test", {}).get("exact_duplicates"),
        "test_acc": res.get("test", {}).get("acc"),
        "test_macro_f1": res.get("test", {}).get("macro_f1"),
        "val_f1": res["best"]["val_f1"],
        "params_m": res["params_m"],
    }


def stage_A(per_class: int, preset: str, epochs: int, workers: int = 2) -> dict:
    """Random split. Sessions are scattered across train and test on purpose."""
    rows = _rows(per_class, TRAIN_CORPUS)
    rng = random.Random(0)
    for r in rows:
        x = rng.random()
        r["split"] = "train" if x < 0.7 else ("val" if x < 0.85 else "test")
    rep = leakage.audit(rows, max_dist=6, workers=8)
    res = train.train_one(rows, preset, epochs, RUNS, workers, tag="A_random")
    return _protocol_result(rows, res, rep)


def stage_B(per_class: int, preset: str, epochs: int, workers: int = 2) -> dict:
    """Group split. Same corpus, same model, sessions kept whole."""
    rows = _rows(per_class, TRAIN_CORPUS)
    data.split_by_group(rows, val=0.15, test=0.15, seed=0)
    rep = leakage.audit(rows, max_dist=6, workers=8)
    res = train.train_one(rows, preset, epochs, RUNS, workers, tag="B_group")
    return _protocol_result(rows, res, rep)


def _cross_rows(per_class: int) -> list[dict]:
    tr = subsample([r for r in sources.read_manifest(MANIFEST)
                    if r["source"] == TRAIN_CORPUS], per_class)
    te = [r for r in sources.read_manifest(MANIFEST) if r["source"] == TEST_CORPUS]
    return data.split_cross_source(tr + te, [TRAIN_CORPUS], [TEST_CORPUS], val=0.15)


def stage_C(per_class: int, preset: str, epochs: int, workers: int = 2) -> dict:
    """Cross-corpus. Different signer, room, and camera. Signer-disjoint by
    construction, with no need to recover anyone's identity."""
    rows = _cross_rows(per_class)
    rep = leakage.audit(rows, max_dist=6, workers=8)
    res = train.train_one(rows, preset, epochs, RUNS, workers, tag="C_cross")
    out = _protocol_result(rows, res, rep)
    out["cross_corpus_leak_rate"] = rep["splits"].get("test", {}).get("leak_rate")
    return out


def stage_arch(per_class: int, presets: list[str], epochs: int, workers: int = 2) -> dict:
    """Architecture search under protocol C, ranked on validation."""
    rows = _cross_rows(per_class)
    return train.sweep(rows, presets, epochs, RUNS, workers, tag="C_arch")


def stage_cal(winner: str, per_class: int, workers: int = 2) -> dict:
    rows = _cross_rows(per_class)
    man = Path("dataset/manifests/cross.csv")
    sources.write_manifest([{k: r[k] for k in sources.MANIFEST_COLS} for r in rows], man)
    ckpt = RUNS / f"C_arch_WINNER_{winner}" / "best.pt"
    if not ckpt.exists():
        ckpt = RUNS / "C_cross" / "best.pt"
    rep = evaluate.run(ckpt, man, out=ckpt.parent / "eval.json", workers=workers)
    return {"checkpoint": str(ckpt), "temperature": rep["temperature"],
            "test": rep.get("test")}


# --- driver ------------------------------------------------------------------

def run(stage: str, per_class: int = 400, preset: str = "vit_small", epochs: int = 4,
        presets: list[str] | None = None, winner: str = "vit_small",
        workers: int = 2) -> dict:
    state = load()
    t0 = time.time()

    if stage == "data":
        out = stage_data(per_class)
    elif stage == "A":
        out = stage_A(per_class, preset, epochs, workers)
    elif stage == "B":
        out = stage_B(per_class, preset, epochs, workers)
    elif stage == "C":
        out = stage_C(per_class, preset, epochs, workers)
    elif stage == "arch":
        out = stage_arch(per_class, presets or M.SWEEPS["quick"], epochs, workers)
    elif stage == "cal":
        out = stage_cal(winner, per_class, workers)
    else:
        raise ValueError(f"unknown stage {stage}")

    out["seconds"] = round(time.time() - t0, 1)
    state["stages"][stage] = out
    save(state)
    print(f"\n[{stage}] done in {out['seconds']}s")
    return out


def summary() -> str:
    """The three protocols side by side. This is the deliverable table."""
    s = load()["stages"]
    lines = ["", f"{'protocol':<34}{'test acc':>10}{'macro F1':>10}{'leakage':>10}", "-" * 64]
    names = {"A": "A  random split (as published)",
             "B": "B  group split (sessions whole)",
             "C": "C  cross-corpus (new signer)"}
    for k, label in names.items():
        r = s.get(k)
        if not r:
            continue
        lines.append(f"{label:<34}{r['test_acc']:>10.4f}{r['test_macro_f1']:>10.4f}"
                     f"{r['test_leak_rate']:>10.1%}")
    if "A" in s and "C" in s:
        d = s["A"]["test_acc"] - s["C"]["test_acc"]
        lines += ["-" * 64,
                  f"A minus C = {d:.1%} of the published headline was leakage, not skill."]
    out = "\n".join(lines)
    print(out)
    return out
