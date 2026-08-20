"""Clip manifests, signer-disjoint splits, and the torch Datasets for video.

The split policy here is stricter than the image side's and it is not a matter
of taste. On the image corpora nobody records who signed each frame, so
`leakage.recover_groups` reconstructs capture *sessions* and the README has to
admit those are not signers. Every video corpus in the registry except WLASL
ships a real signer id, so the honest split is available directly:

    `split_by_signer`  - a signer's clips are all in exactly one of train/val/
                         test. Not recovered, not approximated. Exact.

Where a corpus publishes its own signer-independent benchmark split - AUTSL
does - `official_split` uses that instead, so the number is comparable to the
published one rather than to a split of our own invention.

WLASL has no signer labels and severe link rot, so it is a train-side corpus
only and never the thing a headline number is measured on.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import clips, landmarks
from .sources import DEFAULT_ROOT, REGISTRY

MANIFEST_COLS = ["path", "label", "label_idx", "source", "signer", "group", "split"]


# --- manifests ---------------------------------------------------------------

def write_manifest(rows: list[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in MANIFEST_COLS} for r in rows])
    print(f"[ok] {len(rows)} clips -> {out}")
    return out


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label_idx"] = int(r["label_idx"])
    return rows


def label_space(rows: list[dict]) -> list[str]:
    """The sorted vocabulary actually present, since every corpus differs.

    Derived from the manifest rather than hard-coded: 250 signs for PopSign,
    226 for AUTSL, 64 for LSA64. A fixed CLASSES list like the image side's
    29-letter alphabet would be a lie in three of the four cases.
    """
    return sorted({r["label"] for r in rows})


# --- scanning ----------------------------------------------------------------

def scan_clips(name: str, root: Path = DEFAULT_ROOT) -> list[dict]:
    """Walk one downloaded video/landmark corpus and emit manifest rows.

    Each branch reads whatever that corpus publishes as a signer id. Where it
    publishes none, `signer` is left empty and the caller must not pretend
    otherwise - `split_by_signer` refuses such a corpus rather than silently
    falling back to a random split.
    """
    src = REGISTRY[name]
    base = root / name
    if not base.exists():
        raise FileNotFoundError(
            f"{base} missing - run `python -m slr.sources download {name}` first")

    if name == "popsign_islr":
        return _scan_popsign(base)
    if name == "lsa64":
        return _scan_lsa64(base)
    if name == "autsl":
        return _scan_autsl(base)
    if name == "wlasl":
        return _scan_wlasl(base)
    raise ValueError(f"{name} is not a video corpus")


def _rows(paths, labels, signers, source) -> list[dict]:
    vocab = {c: i for i, c in enumerate(sorted(set(labels)))}
    return [{"path": str(p), "label": lab, "label_idx": vocab[lab],
             "source": source, "signer": str(sg),
             "group": f"{source}/{sg}", "split": ""}
            for p, lab, sg in zip(paths, labels, signers)]


def _scan_popsign(base: Path) -> list[dict]:
    """PopSign ships `train.csv` plus one parquet of landmarks per sequence."""
    import pandas as pd

    csv_path = next(base.rglob("train.csv"))
    df = pd.read_csv(csv_path)
    root = csv_path.parent
    paths = [root / p for p in df["path"]]
    print(f"[scan] popsign_islr: {len(df)} clips, "
          f"{df['participant_id'].nunique()} signers, {df['sign'].nunique()} signs")
    return _rows(paths, df["sign"].tolist(), df["participant_id"].tolist(),
                 "popsign_islr")


def _scan_lsa64(base: Path) -> list[dict]:
    """LSA64 encodes everything in the filename: `NNN_SSS_RRR.mp4`.

    `NNN` is the sign, `SSS` the signer, `RRR` the repetition. Ten signers,
    sixty-four signs, five repetitions each.
    """
    paths, labels, signers = [], [], []
    for p in sorted(base.rglob("*.mp4")):
        parts = p.stem.split("_")
        if len(parts) < 3:
            continue
        paths.append(p)
        labels.append(f"sign{int(parts[0]):03d}")
        signers.append(f"signer{int(parts[1]):03d}")
    print(f"[scan] lsa64: {len(paths)} clips, {len(set(signers))} signers, "
          f"{len(set(labels))} signs")
    return _rows(paths, labels, signers, "lsa64")


def _scan_autsl(base: Path) -> list[dict]:
    """AUTSL ships `*_labels.csv` alongside `signer<N>_sample<M>_color.mp4`."""
    import re

    label_files = list(base.rglob("*labels*.csv"))
    if not label_files:
        raise FileNotFoundError(f"no AUTSL label csv under {base}")
    lookup: dict[str, str] = {}
    for lf in label_files:
        with lf.open() as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    lookup[row[0]] = row[1]

    paths, labels, signers = [], [], []
    for p in sorted(base.rglob("*_color.mp4")):
        key = p.stem.replace("_color", "")
        if key not in lookup:
            continue
        m = re.match(r"signer(\d+)", key)
        paths.append(p)
        labels.append(f"sign{int(lookup[key]):03d}")
        signers.append(f"signer{int(m.group(1)):03d}" if m else "unknown")
    print(f"[scan] autsl: {len(paths)} clips, {len(set(signers))} signers, "
          f"{len(set(labels))} signs")
    return _rows(paths, labels, signers, "autsl")


def _scan_wlasl(base: Path) -> list[dict]:
    """WLASL: gloss directories of mp4s, and no signer id anywhere.

    The empty `signer` is the point. Link rot means the clips present are a
    subset nobody else has exactly, so this corpus trains and never scores.
    """
    paths, labels = [], []
    for p in sorted(base.rglob("*.mp4")):
        paths.append(p)
        labels.append(p.parent.name)
    print(f"[scan] wlasl: {len(paths)} clips, {len(set(labels))} glosses, "
          "NO signer ids (train-side only)")
    return _rows(paths, labels, [""] * len(paths), "wlasl")


# --- splitting ---------------------------------------------------------------

def split_by_signer(rows: list[dict], val: float = 0.15, test: float = 0.15,
                    seed: int = 0) -> list[dict]:
    """Deal whole signers to train/val/test. Exact, not recovered.

    Signers are placed largest-first into whichever split is furthest below its
    target. A first-fit walk would hand an unusually prolific signer to
    whichever split it reached first and blow that split's budget in one go -
    the failure the image-side group split hit on `asl_alphabet`, where one
    recovered session held most of the corpus and emptied train entirely.
    """
    missing = [r for r in rows if not r.get("signer")]
    if missing:
        raise ValueError(
            f"{len(missing)} clips carry no signer id (e.g. {missing[0]['source']}); "
            "a signer-disjoint split cannot be built from them. Use them as a "
            "train-only source, or split a corpus that ships signer labels."
        )

    by_signer: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_signer[r["signer"]].append(i)

    names = sorted(by_signer)
    random.Random(seed).shuffle(names)                 # tie-break only
    n = len(rows)
    targets = {"train": (1 - val - test) * n, "val": val * n, "test": test * n}
    filled = {"train": 0, "val": 0, "test": 0}

    for s in sorted(names, key=lambda g: -len(by_signer[g])):
        dest = max(filled, key=lambda k: targets[k] - filled[k])
        for i in by_signer[s]:
            rows[i]["split"] = dest
        filled[dest] += len(by_signer[s])

    _report(rows)
    return rows


def official_split(rows: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Apply a corpus's own published benchmark split.

    AUTSL publishes a signer-independent train/val/test partition with a known
    baseline (62.02%). Reusing it makes our number comparable to the literature;
    inventing our own would not be.
    """
    for r in rows:
        r["split"] = mapping.get(r["signer"], "train")
    _report(rows)
    return rows


def _report(rows: list[dict]) -> None:
    counts = Counter(r["split"] for r in rows)
    print(f"[split] {dict(counts)}")
    vocab = set(label_space(rows))
    for s in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == s]
        if not part:
            print(f"[split] {s:>6}: EMPTY")
            continue
        seen = {r["label"] for r in part}
        missing = sorted(vocab - seen)
        note = f", MISSING {len(missing)} labels" if missing else ""
        print(f"[split] {s:>6}: {len(part)} clips, "
              f"{len({r['signer'] for r in part})} signers, "
              f"{len(seen)}/{len(vocab)} labels{note}")


# --- datasets ----------------------------------------------------------------

class LandmarkClips(Dataset):
    """Landmark sequences: `(T, FEATURE_DIM)` float32 per clip.

    Handles both shapes the registry produces - PopSign parquet, already
    extracted, and an mp4 that needs Holistic run over it once and cached.
    """

    def __init__(self, rows: list[dict], n_frames: int = clips.N_FRAMES,
                 cache: Path = Path("dataset/landmarks"), train: bool = False):
        self.rows = rows
        self.n_frames = n_frames
        self.cache = Path(cache)
        self.train = train
        self.vocab = {c: i for i, c in enumerate(label_space(rows))}

    def __len__(self) -> int:
        return len(self.rows)

    def _sequence(self, row: dict) -> np.ndarray:
        p = Path(row["path"])
        if p.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(p, columns=["frame", "x", "y", "z"])
            n = df["frame"].nunique()
            seq = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            seq = seq.reshape(n, landmarks.N_POINTS, 3)
            idx = clips.sample_indices(n, self.n_frames)
            return seq[idx]
        return landmarks.load_or_extract(p, self.cache, self.n_frames)

    def __getitem__(self, i: int):
        row = self.rows[i]
        feats = landmarks.to_features(self._sequence(row))
        x = torch.from_numpy(np.ascontiguousarray(feats))
        if self.train:
            # Scale and time jitter only. No horizontal flip, for the same
            # reason the image backend refuses it: signing is handed, and
            # mirroring changes what some signs mean.
            x = x * (1.0 + 0.1 * torch.randn(1))
        return x, self.vocab[row["label"]]


class RGBClips(Dataset):
    """Sampled RGB frames: `(T, 3, H, W)` float32 per clip.

    Kept so the eighteen image backbones stay usable on video - a frame
    backbone plus temporal pooling is the control that says whether motion
    modelling earned anything over classifying stills.
    """

    def __init__(self, rows: list[dict], img_size: int = 224,
                 n_frames: int = clips.N_FRAMES, train: bool = False):
        from torchvision import transforms as T

        self.rows = rows
        self.n_frames = n_frames
        self.vocab = {c: i for i, c in enumerate(label_space(rows))}
        norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.tf = T.Compose([T.ToTensor(), T.Resize((img_size, img_size),
                                                    antialias=True), norm])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        frames = clips.read_clip(row["path"], self.n_frames)
        x = torch.stack([self.tf(f) for f in frames])      # (T, 3, H, W)
        return x, self.vocab[row["label"]]


def loaders(rows: list[dict], kind: str = "landmark", batch_size: int = 64,
            workers: int = 2, **kw) -> dict[str, DataLoader]:
    """One DataLoader per non-empty split."""
    cls = {"landmark": LandmarkClips, "rgb": RGBClips}[kind]
    out: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        if not part:
            continue
        ds = cls(part, train=(split == "train"), **kw)
        out[split] = DataLoader(ds, batch_size=batch_size,
                                shuffle=(split == "train"),
                                num_workers=workers, pin_memory=True,
                                drop_last=(split == "train"))
    return out
