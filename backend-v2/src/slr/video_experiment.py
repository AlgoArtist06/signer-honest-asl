"""The video experiment, as one runnable module.

Same three-protocol argument as the image backend, with one important upgrade.
On the image corpora protocol B had to *recover* capture sessions because no
corpus records who signed each frame, and `signer_check` measures how badly
that recovery fragments real signers. Every video corpus here except WLASL
ships a signer id, so the honest split is not approximated:

    R  random clip split      what most video SLR papers report
    S  signer-disjoint split  signers dealt whole. Exact, not recovered.
    O  official benchmark     the corpus's own published signer-independent
                              split, so the number is comparable to the
                              literature rather than to a split of our own

R minus S is the same contribution as A minus C on the image side: the share of
the published headline that was signer memorisation rather than sign
recognition. Because S is exact here, that gap is a measurement rather than an
estimate - which is the reason this backend exists.

Stages, in order:

    data   download the corpora, build the clip manifest
    R      random clip split
    S      signer-disjoint split
    O      official benchmark split, where the corpus publishes one
    arch   temporal architecture search on S, selected on validation
    cal    temperature scaling and the abstention threshold for the winner
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from . import evaluate, sources, video_data, video_model as VM, video_train

RUNS = Path("runs")
RESULTS = RUNS / "video_results.json"
MANIFEST = Path("dataset/manifests/clips.csv")

# PopSign is the default because it is the only corpus with a real signer id for
# every one of its 21 signers, which is precisely what protocol S needs.
DEFAULT_CORPUS = "popsign_islr"


# --- bookkeeping -------------------------------------------------------------

def load() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}


def save(state: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(state, indent=2))
    print(f"[state] -> {RESULTS}")


def subsample(rows: list[dict], per_class: int, seed: int = 0) -> list[dict]:
    """Cap clips per sign, dealing whole signers wherever one fits.

    Whole signers, for the same reason the image side deals whole sessions: a
    subsample must not be able to put one signer on both sides of a split.
    Where a single signer already exceeds the cap their clips are truncated
    rather than the signer dropped, so the vocabulary stays covered.
    """
    if not per_class:
        return rows
    by_sign: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_sign.setdefault(r["label"], {}).setdefault(r["signer"], []).append(r)

    rng = random.Random(seed)
    kept: list[dict] = []
    for sign, signers in by_sign.items():
        names = sorted(signers)
        rng.shuffle(names)
        n = 0
        for s in names:
            if n >= per_class:
                break
            take = signers[s][: per_class - n]
            kept += take
            n += len(take)
    print(f"[sample] {len(rows)} -> {len(kept)} clips (<={per_class}/sign)")
    return kept


# --- stages ------------------------------------------------------------------

def stage_data(corpus: str = DEFAULT_CORPUS) -> dict:
    sources.download(corpus)
    rows = video_data.scan_clips(corpus)
    video_data.write_manifest(rows, MANIFEST)
    summary = {
        "corpus": corpus,
        "clips": len(rows),
        "signs": len({r["label"] for r in rows}),
        "signers": len({r["signer"] for r in rows if r["signer"]}),
    }
    print(f"[data] {summary}")
    return summary


def _rows(per_class: int) -> list[dict]:
    return subsample(video_data.read_manifest(MANIFEST), per_class)


def _result(rows: list[dict], res: dict) -> dict:
    return {
        "n_train": sum(1 for r in rows if r["split"] == "train"),
        "n_test": sum(1 for r in rows if r["split"] == "test"),
        "train_signers": len({r["signer"] for r in rows if r["split"] == "train"}),
        "test_signers": len({r["signer"] for r in rows if r["split"] == "test"}),
        "shared_signers": len(
            {r["signer"] for r in rows if r["split"] == "train"}
            & {r["signer"] for r in rows if r["split"] == "test"}),
        "test_acc": res.get("test", {}).get("acc"),
        "test_macro_f1": res.get("test", {}).get("macro_f1"),
        "val_f1": res["best"]["val_f1"],
        "params_m": res["params_m"],
    }


def stage_R(per_class: int, preset: str, epochs: int, workers: int = 2) -> dict:
    """Random clip split. Every signer appears on both sides, on purpose.

    `shared_signers` in the result is the whole point: it counts how many
    signers the model saw in training and is then tested on. For a random split
    that number is all of them.
    """
    rows = _rows(per_class)
    rng = random.Random(0)
    for r in rows:
        x = rng.random()
        r["split"] = "train" if x < 0.7 else ("val" if x < 0.85 else "test")
    video_data._report(rows)
    res = video_train.train_one(rows, preset, epochs, RUNS, workers, tag="R_random")
    return _result(rows, res)


def stage_S(per_class: int, preset: str, epochs: int, workers: int = 2) -> dict:
    """Signer-disjoint split. Same corpus, same model, no shared signer."""
    rows = _rows(per_class)
    video_data.split_by_signer(rows, val=0.15, test=0.15, seed=0)
    res = video_train.train_one(rows, preset, epochs, RUNS, workers, tag="S_signer")
    out = _result(rows, res)
    assert out["shared_signers"] == 0, "signer-disjoint split leaked a signer"
    return out


def stage_O(per_class: int, preset: str, epochs: int, mapping: dict[str, str],
            workers: int = 2) -> dict:
    """The corpus's own published signer-independent split."""
    rows = _rows(per_class)
    video_data.official_split(rows, mapping)
    res = video_train.train_one(rows, preset, epochs, RUNS, workers, tag="O_official")
    return _result(rows, res)


def stage_arch(per_class: int, presets: list[str], epochs: int,
               workers: int = 2) -> dict:
    """Temporal architecture search under the signer-disjoint protocol."""
    rows = _rows(per_class)
    video_data.split_by_signer(rows, val=0.15, test=0.15, seed=0)
    return video_train.sweep(rows, presets, epochs, RUNS, workers, tag="S_arch")


def stage_cal(winner: str, per_class: int, workers: int = 2) -> dict:
    """Temperature and the abstention threshold, fitted on validation only.

    Abstention matters more here than on stills. A ten-second upload is cut
    into windows and most windows contain no complete sign at all, so a model
    that cannot say "nothing here" will label the gaps confidently.
    """
    rows = _rows(per_class)
    video_data.split_by_signer(rows, val=0.15, test=0.15, seed=0)

    ckpt = RUNS / f"S_arch_WINNER_{winner}" / "best.pt"
    if not ckpt.exists():
        ckpt = RUNS / "S_signer" / "best.pt"
    blob = torch.load(ckpt, map_location="cpu")
    dev = video_train.device()
    net = VM.build(blob["preset"], len(blob["classes"]))
    net.load_state_dict(blob["state_dict"])
    net.to(dev).eval()

    p = VM.PRESETS[blob["preset"]]
    dl = video_data.loaders(rows, kind=p.kind, batch_size=p.batch_size,
                            workers=workers, n_frames=p.n_frames,
                            **({"img_size": p.img_size} if p.kind == "rgb" else {}))

    vl, vy = evaluate.collect_logits(net, dl["val"], dev)
    temp = evaluate.fit_temperature(vl, vy)
    tl, ty = evaluate.collect_logits(net, dl["test"], dev)
    probs = torch.softmax(tl / temp, dim=1)

    report = {
        "checkpoint": str(ckpt),
        "temperature": temp,
        "test": {
            "ece_before": evaluate.expected_calibration_error(
                torch.softmax(tl, dim=1), ty),
            "ece_after": evaluate.expected_calibration_error(probs, ty),
            "risk_coverage": evaluate.risk_coverage(probs, ty),
        },
    }
    (ckpt.parent / "eval.json").write_text(json.dumps(report, indent=2))
    print(f"[cal] temperature={temp:.3f} -> {ckpt.parent / 'eval.json'}")
    return report


# --- driver ------------------------------------------------------------------

def run(stage: str, per_class: int = 0, preset: str = "lm_transformer",
        epochs: int = 10, presets: list[str] | None = None,
        winner: str = "lm_transformer", corpus: str = DEFAULT_CORPUS,
        mapping: dict[str, str] | None = None, workers: int = 2) -> dict:
    state = load()
    t0 = time.time()

    if stage == "data":
        out = stage_data(corpus)
    elif stage == "R":
        out = stage_R(per_class, preset, epochs, workers)
    elif stage == "S":
        out = stage_S(per_class, preset, epochs, workers)
    elif stage == "O":
        out = stage_O(per_class, preset, epochs, mapping or {}, workers)
    elif stage == "arch":
        out = stage_arch(per_class, presets or VM.SWEEPS["quick"], epochs, workers)
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
    """The protocols side by side. This is the deliverable table."""
    s = load()["stages"]
    lines = ["", f"{'protocol':<38}{'test acc':>10}{'macro F1':>10}{'shared signers':>16}",
             "-" * 74]
    names = {"R": "R  random clip split (as published)",
             "S": "S  signer-disjoint (exact)",
             "O": "O  official benchmark split"}
    for k, label in names.items():
        r = s.get(k)
        if not r or r.get("test_acc") is None:
            continue
        lines.append(f"{label:<38}{r['test_acc']:>10.4f}"
                     f"{r['test_macro_f1']:>10.4f}{r['shared_signers']:>16}")
    if "R" in s and "S" in s and s["R"].get("test_acc") and s["S"].get("test_acc"):
        d = s["R"]["test_acc"] - s["S"]["test_acc"]
        lines += ["-" * 74,
                  f"R minus S = {d:.1%} of the published headline was signer "
                  "memorisation, not sign recognition."]
    out = "\n".join(lines)
    print(out)
    return out
