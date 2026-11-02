"""Dataset registry, downloads and manifests.

Each source says how to fetch it, where its labels are, and how to compute a
group key. Rows that share a group (e.g. frames from one session) are kept in
the same split. Sources without one get groups from `slr.leakage`.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

# --- label space -------------------------------------------------------------

# J and Z are left out: they are drawn in the air, so a photo cannot hold them.
LETTERS = [c for c in map(chr, range(ord("A"), ord("Z") + 1)) if c not in "JZ"]
CLASSES = LETTERS + ["space", "del", "nothing"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Datasets name the three non-letter classes differently.
_ALIASES = {
    "space": "space", "sp": "space", "blank": "space", "_": "space",
    "del": "del", "delete": "del", "dl": "del",
    "nothing": "nothing", "none": "nothing", "empty": "nothing", "bg": "nothing",
}


def normalise_label(raw: str) -> str | None:
    """Map a directory or column name onto the class vocabulary, or None."""
    s = raw.strip()
    if len(s) == 1 and s.upper() in CLASS_TO_IDX:
        return s.upper()
    return _ALIASES.get(s.lower().replace(" ", "").replace("-", ""))


# --- group-key rules ---------------------------------------------------------

def group_by_parent_dir(path: Path, root: Path) -> str:
    """The whole corpus is one group, so it is never split internally."""
    return "all"


_INDEXED_RE = re.compile(r"^([A-Za-z_]+)(\d+)")


def group_by_frame_block(block: int) -> Callable[[Path, Path], str]:
    """Group `<class><frame>.jpg` files into blocks of `block` frames.

    The class stays in the key because the frame counter restarts per class.
    """

    def _fn(path: Path, root: Path) -> str:
        m = _INDEXED_RE.match(path.stem)
        if not m:
            return f"unknown:{path.name}"
        return f"{m.group(1)}blk{int(m.group(2)) // block:05d}"

    return _fn


# --- source registry ---------------------------------------------------------

@dataclass(frozen=True)
class Source:
    name: str
    kind: str                      # "kaggle" | "zip" | "local"
    ref: str                       # dataset slug / url
    role: str                      # "train" | "holdout" | "either"
    group_fn: Callable[[Path, Path], str] = group_by_parent_dir
    subdir: str = ""               # path inside the download to scan
    note: str = ""
    sha256: str = ""               # archive digest for zip; see `download`


REGISTRY: dict[str, Source] = {s.name: s for s in [
    Source(
        name="asl_alphabet",
        kind="kaggle", ref="grassknoted/asl-alphabet", role="train",
        subdir="asl_alphabet_train/asl_alphabet_train",
        group_fn=group_by_frame_block(200),
        note="87k images, 29 classes, mostly one signer on a webcam. Its 28-image "
             "test set is too small to use.",
    ),
    Source(
        name="asl_alphabet_test",
        kind="kaggle", ref="danrasband/asl-alphabet-test", role="holdout",
        group_fn=group_by_parent_dir,
        note="870 images by a different person on varied backgrounds, made to "
             "test models trained on asl_alphabet. Main test set.",
    ),
    Source(
        name="mine",
        kind="local", ref="photos taken in the app and labelled in the page",
        role="either",
        group_fn=lambda p, root: p.relative_to(root).parts[0],
        note="Photos taken in the app, filed as <session>/<LABEL>/. The session "
             "is the group. One signer only. Local, never committed.",
    ),
    Source(
        name="signalphaset",
        kind="zip",
        ref="https://data.mendeley.com/public-files/datasets/8fmvr9m98w/files/"
            "a12e2b20-74de-4301-858f-35ab312d77b7/file_downloaded",
        role="train", subdir="SignAlphaSet",
        group_fn=group_by_frame_block(100),
        note="SignAlphaSet (Data in Brief 2025, doi:10.17632/8fmvr9m98w.3). 26k "
             "images, 1000 per letter, black background, letters only. Used for "
             "its left-hand images: asl_alphabet is 92.5% right hands. There are "
             "no signer or handedness labels, so groups are filename blocks.",
    ),
]}

REPO = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = REPO / "dataset" / "raw"


# --- downloading -------------------------------------------------------------

def _check_digest(src: Source, got: str, archive: Path) -> None:
    """Check a download against its pinned sha256, or print the hash to pin."""
    if not src.sha256:
        print(f"[hash] {src.name}: sha256={got} (not pinned)")
        return
    if got.lower() != src.sha256.lower():
        archive.unlink(missing_ok=True)
        raise SystemExit(
            f"{src.name}: sha256 mismatch\n"
            f"  expected {src.sha256}\n  got      {got}")
    print(f"[hash] {src.name}: sha256 matches the pin")


def download(name: str, root: Path = DEFAULT_ROOT) -> Path:
    """Fetch one source into `root/<name>`. Idempotent: skips a non-empty dir."""
    src = REGISTRY[name]
    dest = root / name
    if dest.exists() and any(dest.rglob("*")):
        print(f"[skip] {name} already at {dest}")
        return dest
    if src.kind == "local":
        print(f"[local] {name} is not downloaded; label photos in the app and "
              f"they go to {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    has_kaggle_creds = (bool(os.environ.get("KAGGLE_USERNAME"))
                        or (Path.home() / ".kaggle" / "kaggle.json").exists())
    if src.kind == "kaggle" and not has_kaggle_creds:
        raise SystemExit(
            f"{name} needs Kaggle credentials: put your token in "
            "~/.kaggle/kaggle.json or set KAGGLE_USERNAME / KAGGLE_KEY."
        )

    if src.kind == "kaggle":
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", src.ref, "-p", str(dest), "--unzip"],
            check=True,
        )
    elif src.kind == "zip":
        print(f"[get] {src.ref}")
        # Stream to disk, archives can be huge.
        tmp = dest / "_download.zip"
        digest = hashlib.sha256()
        with urllib.request.urlopen(src.ref, timeout=300) as r, tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                digest.update(chunk)
                f.write(chunk)
        _check_digest(src, digest.hexdigest(), tmp)
        zipfile.ZipFile(tmp).extractall(dest)
        tmp.unlink()
    else:
        raise ValueError(f"unknown source kind {src.kind}")

    print(f"[ok] {name} -> {dest}")
    return dest


# --- manifest ----------------------------------------------------------------

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Unlabelled app captures wait here. Never part of a manifest.
STAGING = "_inbox"
MANIFEST_COLS = ["path", "label", "label_idx", "source", "group", "split"]


def scan(name: str, root: Path = DEFAULT_ROOT) -> list[dict]:
    """Walk one downloaded source and emit manifest rows with an empty `split`."""
    src = REGISTRY[name]
    base = root / name / src.subdir if src.subdir else root / name
    if not base.exists():
        raise FileNotFoundError(
            f"{base} missing - run `python -m slr.sources download {name}` first")

    rows: list[dict] = []
    skipped: set[str] = set()
    staged = 0
    for p in sorted(base.rglob("*")):
        # Skip macOS `._` metadata files.
        if p.suffix.lower() not in IMG_EXT or p.name.startswith("._"):
            continue
        parts = p.relative_to(base).parts
        if len(parts) < 2:
            continue
        raw = parts[-2]  # the class name is the file's parent directory
        if raw == STAGING:
            staged += 1
            continue
        label = normalise_label(raw)
        if label is None:
            skipped.add(raw)
            continue
        idx = CLASS_TO_IDX[label]
        rows.append({
            "path": str(p.resolve()),
            "label": label,
            "label_idx": idx,
            "source": name,
            "group": f"{name}/{src.group_fn(p, base)}",
            "split": "",
        })
    if staged:
        print(f"[scan] {name}: skipped {staged} unlabelled captures in {STAGING}/")
    if skipped:
        print(f"[warn] {name}: dropped dirs outside the class vocabulary: {sorted(skipped)}")
    print(f"[scan] {name}: {len(rows)} images, {len({r['group'] for r in rows})} groups")
    return rows


def repo_relative(path: str | Path) -> str:
    """Path relative to REPO if it is inside it, so manifests survive a move."""
    try:
        return Path(path).relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def resolve(path: str | Path) -> str:
    """A stored path as an absolute one. Absolute paths pass through unchanged."""
    return str(REPO / path)


def write_manifest(rows: Iterable[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    # Fail on a missing column rather than write an empty group.
    rows = [{**{k: r[k] for k in MANIFEST_COLS}, "path": repo_relative(r["path"])}
            for r in rows]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {len(rows)} rows -> {out}")
    return out


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = resolve(r["path"])
        r["label_idx"] = int(r["label_idx"])
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="fetch corpora and build a manifest")
    ap.add_argument("cmd", choices=["list", "download", "manifest"])
    ap.add_argument("names", nargs="*", help="source names, or 'all'")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    names = list(REGISTRY) if (not a.names or a.names == ["all"]) else a.names
    if a.cmd == "list":
        for s in REGISTRY.values():
            print(f"{s.name:20s} {s.kind:7s} {s.role:8s} {s.ref}")
            print(f"{'':20s} {s.note}")
        return 0

    if a.cmd == "download":
        for n in names:
            download(n, a.root)
        return 0

    rows: list[dict] = []
    for n in names:
        try:
            rows += scan(n, a.root)
        except FileNotFoundError as e:
            print(f"[skip] {e}", file=sys.stderr)
    out = a.out or (a.root.parent / "manifests" / "all.csv")
    write_manifest(rows, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
