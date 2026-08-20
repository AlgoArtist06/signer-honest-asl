"""Splits, transforms, and the torch Dataset.

Two split policies, and only two, because these are the only two that produce a
number worth quoting:

  `group`        - groups (recovered signer/session clusters) are dealt whole to
                   exactly one of train/val/test. No frame of a session can face
                   its neighbour across the split boundary.

  `cross_source` - train on one corpus, test on a different corpus shot by
                   different people in a different room. This is the number that
                   predicts what the phone app does on a stranger's hand, and it
                   is the number nobody reports.

A plain random split is deliberately not implemented. If you want one, you want
`leakage.audit` instead.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .sources import CLASSES, REGISTRY

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- splitting ---------------------------------------------------------------

def split_by_group(rows: list[dict], val: float = 0.15, test: float = 0.15,
                   seed: int = 0) -> list[dict]:
    """Assign splits so that no group straddles a boundary.

    Greedy: shuffle the groups, then walk them filling test, then val, then the
    rest into train. Greedy on group size rather than exact stratification -
    class balance is checked and warned about afterwards rather than optimised
    for, because forcing per-class balance would require splitting groups, which
    is the exact thing we are refusing to do.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["group"]].append(i)

    names = sorted(groups)
    random.Random(seed).shuffle(names)          # tie-break between equal sizes

    n = len(rows)
    targets = {"train": (1 - val - test) * n, "val": val * n, "test": test * n}
    filled = {"train": 0, "val": 0, "test": 0}

    # Largest group first, into whichever split is furthest below its target.
    #
    # A first-fit walk that filled test, then val, then train looks reasonable
    # until group sizes are heavy-tailed, which after `leakage.recover_groups`
    # they always are: on `asl_alphabet` 87k frames collapse into 191 sessions
    # and a handful of those hold nearly all of it. First-fit handed one such
    # session to test, blew the 15% budget by an order of magnitude in a single
    # step, and left train with no rows at all. Placing the biggest session
    # first, where the deficit is largest, keeps it in train where 70% of the
    # budget is and fills val and test from the tail.
    for g in sorted(names, key=lambda g: -len(groups[g])):
        s = max(filled, key=lambda k: targets[k] - filled[k])
        for i in groups[g]:
            rows[i]["split"] = s
        filled[s] += len(groups[g])

    _report(rows)
    return rows


def split_cross_source(rows: list[dict], train_sources: list[str],
                       test_sources: list[str], val: float = 0.1,
                       seed: int = 0) -> list[dict]:
    """Train on one set of corpora, test on a disjoint set.

    Validation is carved group-disjointly out of the *training* corpora, so the
    test corpora stay completely untouched until the final evaluation.
    """
    overlap = set(train_sources) & set(test_sources)
    if overlap:
        raise ValueError(f"train and test sources overlap: {sorted(overlap)}")

    keep, dropped = [], Counter()
    for r in rows:
        if r["source"] in train_sources:
            r["split"] = "train"
            keep.append(r)
        elif r["source"] in test_sources:
            r["split"] = "test"
            keep.append(r)
        else:
            dropped[r["source"]] += 1
    if dropped:
        print(f"[split] ignoring sources not named: {dict(dropped)}")

    train_rows = [r for r in keep if r["split"] == "train"]
    groups = sorted({r["group"] for r in train_rows})
    random.Random(seed).shuffle(groups)
    quota, taken = val * len(train_rows), 0
    val_groups: set[str] = set()
    for g in groups:
        if taken >= quota:
            break
        val_groups.add(g)
        taken += sum(1 for r in train_rows if r["group"] == g)
    for r in train_rows:
        if r["group"] in val_groups:
            r["split"] = "val"

    _report(keep)
    return keep


def _report(rows: list[dict]) -> None:
    by_split = Counter(r["split"] for r in rows)
    print(f"[split] {dict(by_split)}")
    for s in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == s]
        if not sub:
            continue
        missing = set(CLASSES) - {r["label"] for r in sub}
        g = len({r["group"] for r in sub})
        print(f"[split]   {s}: {len(sub)} imgs, {g} groups, "
              f"{len(CLASSES) - len(missing)}/{len(CLASSES)} classes"
              + (f", MISSING {sorted(missing)}" if missing else ""))
    # The invariant this whole module exists to guarantee.
    seen: dict[str, str] = {}
    for r in rows:
        if r["split"] and seen.setdefault(r["group"], r["split"]) != r["split"]:
            raise AssertionError(f"group {r['group']} straddles splits - split is invalid")


# --- dataset -----------------------------------------------------------------

def build_transforms(img_size: int, train: bool):
    """Augmentation is deliberately heavy on the train side.

    The failure mode of these corpora is background memorisation, so the train
    pipeline attacks every cue that is not the hand: crop, rotate, recolour,
    desaturate, and erase patches. The eval pipeline does nothing but resize.
    """
    from torchvision import transforms as T

    if not train:
        return T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
        T.RandomApply([T.RandomRotation(18)], p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])


class SignDataset(Dataset):
    """Manifest-backed image dataset. Never applies a horizontal flip.

    ASL is handed. Mirroring an image turns a right-handed sign into a
    left-handed one, and for a few letter pairs it changes what the sign means.
    torchvision's default recipes flip; ours must not.
    """

    def __init__(self, rows: list[dict], img_size: int = 224, train: bool = False):
        bad = [r for r in rows if r["label_idx"] < 0]
        if bad:
            raise ValueError(
                f"{len(bad)} rows carry label_idx=-1, meaning they come from an "
                f"audit-only source outside the 29-class vocabulary "
                f"(e.g. {sorted({r['source'] for r in bad})}). Exclude them before "
                "training."
            )
        self.rows = rows
        self.tf = build_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        with Image.open(r["path"]) as im:
            img = im.convert("RGB")
        return self.tf(img), r["label_idx"]


def loaders(rows: list[dict], img_size: int, batch_size: int, workers: int = 4
            ) -> dict[str, DataLoader]:
    out: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        is_train = split == "train"
        out[split] = DataLoader(
            SignDataset(sub, img_size, train=is_train),
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train,
            persistent_workers=workers > 0,
        )
    return out


# ponytail: no hand detection/cropping yet. Cropping to the hand with MediaPipe
# is the single biggest expected gain on cross-corpus transfer, since it deletes
# the background shortcut outright rather than augmenting around it. Add it as a
# preprocessing pass writing crops beside the originals, once the uncropped
# cross-corpus baseline is on the board and there is something to compare to.


def main(argv: list[str] | None = None) -> int:
    import argparse

    from . import sources

    ap = argparse.ArgumentParser(description="assign leakage-safe splits")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--policy", choices=["group", "cross_source"], default="group")
    ap.add_argument("--train-sources", nargs="*", default=None)
    ap.add_argument("--test-sources", nargs="*", default=None)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    rows = sources.read_manifest(a.manifest)
    if a.policy == "group":
        rows = split_by_group(rows, a.val, a.test, a.seed)
    else:
        tr = a.train_sources or [n for n, s in REGISTRY.items() if s.role == "train"]
        te = a.test_sources or [n for n, s in REGISTRY.items() if s.role == "holdout"]
        print(f"[split] train on {tr} -> test on {te}")
        rows = split_cross_source(rows, tr, te, a.val, a.seed)

    sources.write_manifest(rows, a.out or a.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
