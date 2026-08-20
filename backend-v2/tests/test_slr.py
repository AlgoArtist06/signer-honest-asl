"""The checks that fail if the honesty guarantees break.

Everything here runs on synthetic images in a tmpdir - no dataset download, no
GPU, no network. Fast enough to run on every commit.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from slr import data, evaluate, leakage, sources


# --- label + group recovery --------------------------------------------------

def test_normalise_label_covers_the_29_class_vocabulary():
    assert sources.normalise_label("A") == "A"
    assert sources.normalise_label("z") == "Z"
    assert sources.normalise_label("Space") == "space"
    assert sources.normalise_label("nothing") == "nothing"
    assert sources.normalise_label("Del") == "del"
    assert sources.normalise_label("random_folder") is None
    assert len(sources.CLASSES) == 29


def test_digits_signer_recovery_matches_the_capture_order():
    """IMG_1118..IMG_1127 is one student shooting digits 0..9; 1128 starts the next."""
    g = sources.group_sl_digits
    first = [g(Path(f"IMG_{n}.JPG"), Path(".")) for n in range(1118, 1128)]
    assert len(set(first)) == 1, "one student's ten shots must share a group"
    assert g(Path("IMG_1128.JPG"), Path(".")) != first[0], "next block is a new student"
    assert g(Path("IMG_1138.JPG"), Path(".")) != g(Path("IMG_1128.JPG"), Path("."))


# --- near-duplicate machinery ------------------------------------------------

def test_hamming_matches_python_bit_counting():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**63, size=64, dtype=np.uint64)
    b = rng.integers(0, 2**63, size=64, dtype=np.uint64)
    want = [bin(int(x) ^ int(y)).count("1") for x, y in zip(a, b)]
    assert list(leakage.hamming(a, b)) == want


@pytest.mark.parametrize("max_dist", [1, 3, 6])
def test_candidate_pairs_misses_no_true_near_duplicate(max_dist):
    """The pigeonhole guarantee is the whole reason we can skip the O(n^2) scan.

    If banding ever drops a real match, the audit silently under-reports leakage,
    which is the one failure mode this project cannot tolerate.
    """
    rng = random.Random(7)
    base = [rng.getrandbits(64) for _ in range(120)]
    # Plant near-duplicates by flipping exactly `max_dist` bits.
    planted = []
    for i in range(0, 40, 2):
        h = base[i]
        for bit in rng.sample(range(64), max_dist):
            h ^= 1 << bit
        planted.append((i, len(base)))
        base.append(h)
    h = np.array(base, dtype=np.uint64)

    truth = {
        (i, j)
        for i in range(len(h)) for j in range(i + 1, len(h))
        if leakage.hamming(h[i : i + 1], h[j : j + 1])[0] <= max_dist
    }
    found = {
        (min(i, j), max(i, j))
        for i, j in leakage.candidate_pairs(h, max_dist)
        if leakage.hamming(h[i : i + 1], h[j : j + 1])[0] <= max_dist
    }
    assert truth <= found, f"banding lost {sorted(truth - found)}"
    for a, b in planted:
        assert (min(a, b), max(a, b)) in found


# --- end to end on synthetic images -----------------------------------------

def _write_corpus(root: Path, source: str, n_groups: int, per_group: int) -> list[dict]:
    """Each group is one 'session': a base pattern plus imperceptible jitter, so
    within-group images are near-duplicates and across-group images are not."""
    rng = np.random.default_rng(1)
    rows = []
    for g in range(n_groups):
        base = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
        label = sources.CLASSES[g % len(sources.CLASSES)]
        d = root / source / label
        d.mkdir(parents=True, exist_ok=True)
        for k in range(per_group):
            arr = np.clip(base.astype(int) + rng.integers(-1, 2, base.shape), 0, 255)
            p = d / f"g{g}_{k}.png"
            Image.fromarray(arr.astype(np.uint8), "L").convert("RGB").save(p)
            rows.append({"path": str(p), "label": label,
                         "label_idx": sources.CLASS_TO_IDX[label],
                         "source": source, "group": f"{source}/g{g}", "split": ""})
    return rows


def test_group_split_never_lets_a_group_straddle(tmp_path):
    rows = _write_corpus(tmp_path, "synth", n_groups=30, per_group=6)
    data.split_by_group(rows, val=0.2, test=0.2, seed=3)
    seen = {}
    for r in rows:
        assert seen.setdefault(r["group"], r["split"]) == r["split"]
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_random_split_leaks_and_group_split_does_not(tmp_path):
    """The claim the whole project rests on, made falsifiable.

    Split the same corpus two ways. A random split scatters each session across
    train and test, so the audit should find near-duplicates for most test
    images. A group split keeps sessions whole, so it should find almost none.
    """
    rows = _write_corpus(tmp_path, "synth", n_groups=25, per_group=8)

    rng = random.Random(0)
    for r in rows:
        r["split"] = "test" if rng.random() < 0.25 else "train"
        r.pop("dhash", None)
    leaky = leakage.audit(rows, max_dist=6)["splits"]["test"]["leak_rate"]

    for r in rows:
        r["split"] = ""
    data.split_by_group(rows, val=0.0, test=0.25, seed=0)
    honest = leakage.audit(rows, max_dist=6)["splits"]["test"]["leak_rate"]

    assert leaky > 0.8, f"random split should be badly contaminated, got {leaky}"
    assert honest < 0.05, f"group split should be clean, got {honest}"


def test_recover_groups_finds_the_sessions(tmp_path):
    rows = _write_corpus(tmp_path, "synth", n_groups=12, per_group=5)
    truth = {r["path"]: r["group"] for r in rows}
    for r in rows:
        r["group"] = "synth/all"          # pretend the corpus shipped no ids
    leakage.recover_groups(rows, max_dist=6)

    by_recovered: dict[str, set[str]] = {}
    for r in rows:
        by_recovered.setdefault(r["group"], set()).add(truth[r["path"]])
    assert all(len(v) == 1 for v in by_recovered.values()), \
        "a recovered component merged two real sessions"
    assert len(by_recovered) == 12


def test_cross_source_split_refuses_overlapping_sources(tmp_path):
    rows = _write_corpus(tmp_path, "a", 4, 3) + _write_corpus(tmp_path, "b", 4, 3)
    with pytest.raises(ValueError):
        data.split_cross_source(rows, ["a"], ["a", "b"])
    kept = data.split_cross_source(rows, ["a"], ["b"], val=0.25)
    assert {r["source"] for r in kept if r["split"] == "test"} == {"b"}
    assert {r["source"] for r in kept if r["split"] in ("train", "val")} == {"a"}


def test_purge_drops_contaminated_eval_rows_only(tmp_path):
    rows = _write_corpus(tmp_path, "synth", 10, 6)
    rng = random.Random(1)
    for r in rows:
        r["split"] = "test" if rng.random() < 0.3 else "train"
    leakage.audit(rows, max_dist=6)
    kept = leakage.purge(rows)
    assert all(r["split"] == "train" for r in kept if "leak_dist" in r)
    assert sum(r["split"] == "train" for r in kept) == sum(r["split"] == "train" for r in rows)


# --- calibration / abstention ------------------------------------------------

def test_temperature_scaling_tames_an_overconfident_model():
    import torch

    torch.manual_seed(0)
    y = torch.randint(0, 5, (600,))
    logits = torch.randn(600, 5)
    logits[torch.arange(600), y] += 1.2       # right often, but not always
    logits *= 6.0                              # ... and wildly overconfident

    t = evaluate.fit_temperature(logits, y)
    assert t > 1.5, f"expected the fit to cool the logits, got T={t}"
    before = evaluate.expected_calibration_error(logits.softmax(1), y)
    after = evaluate.expected_calibration_error((logits / t).softmax(1), y)
    assert after < before


def test_risk_coverage_is_monotone():
    import torch

    torch.manual_seed(0)
    y = torch.randint(0, 5, (400,))
    logits = torch.randn(400, 5)
    logits[torch.arange(400), y] += 2.0
    rc = evaluate.risk_coverage(logits.softmax(1), y)
    # Answering fewer, more confident photos must not raise the error rate.
    errs = [r["error"] for r in rc]
    assert errs == sorted(errs, reverse=True), errs
    assert rc[0]["coverage"] == 1.0


# --- architecture search discipline ------------------------------------------

def test_every_preset_in_the_zoo_builds_and_reports_its_size():
    """A typo in a timm checkpoint name should fail here, not eight hours into a sweep."""
    from slr import model as M

    for name, p in M.PRESETS.items():
        net = M.build(name, len(sources.CLASSES), pretrained=False)
        got = M.n_params(net) / 1e6
        assert abs(got - p.approx_params_m) < 1.0, \
            f"{name}: PRESETS says {p.approx_params_m}M, actual {got:.1f}M"
        assert p.family in {"vit", "hybrid", "cnn"}
        assert p.tier in {"scale", "arch", "mobile"}


def test_named_sweeps_reference_real_presets():
    from slr import model as M

    for name, members in M.SWEEPS.items():
        assert members, f"sweep {name} is empty"
        unknown = set(members) - set(M.PRESETS)
        assert not unknown, f"sweep {name} names missing presets {unknown}"
    # The CNN control has to be in the headline sweeps, or the comparison is rigged.
    assert any(M.PRESETS[m].family == "cnn" for m in M.SWEEPS["quick"])
    assert any(M.PRESETS[m].family == "cnn" for m in M.SWEEPS["arch"])


def test_sweep_refuses_to_select_without_a_validation_split(tmp_path):
    """Selecting an architecture on test is leakage through the back door."""
    from slr import train

    rows = _write_corpus(tmp_path, "synth", n_groups=6, per_group=4)
    for r in rows:
        r["split"] = "train" if r["group"].endswith(("g0", "g1", "g2")) else "test"
    with pytest.raises(ValueError, match="validation"):
        train.sweep(rows, ["vit_tiny"], epochs=1, out_dir=tmp_path / "runs")


def test_sweep_resumes_from_a_partial_leaderboard(tmp_path, monkeypatch):
    """A full-zoo sweep outlives a Colab session, so a restart must not retrain
    candidates that already have a score."""
    import json

    from slr import train

    rows = _write_corpus(tmp_path, "synth", n_groups=9, per_group=4)
    for r in rows:
        r["split"] = ("train" if r["group"].endswith(("g0", "g1", "g2", "g3"))
                      else "val" if r["group"].endswith(("g4", "g5")) else "test")

    trained: list[str] = []

    def fake_train_one(_rows, preset, *a, **kw):
        trained.append(preset)
        score = {"vit_tiny": 0.9, "vit_small": 0.5}[preset]
        return {"preset": preset, "params_m": 1.0,
                "best": {"val_f1": score, "val_acc": score},
                "history": [{"secs": 1.0}], "test": {"acc": score}}

    monkeypatch.setattr(train, "train_one", fake_train_one)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "sweep_leaderboard.json").write_text(json.dumps({"leaderboard": [
        {"preset": "vit_tiny", "family": "vit", "tier": "scale", "params_m": 5.5,
         "img_size": 224, "val_f1": 0.9, "val_acc": 0.9, "secs_per_epoch": 1.0},
    ]}))

    res = train.sweep(rows, ["vit_tiny", "vit_small"], epochs=1, out_dir=runs)

    # vit_tiny appears once and only as the winner's final test run, never as a
    # search candidate: its score came off disk.
    assert trained == ["vit_small", "vit_tiny"]
    assert {b["preset"] for b in res["leaderboard"]} == {"vit_tiny", "vit_small"}
    assert res["winner"] == "vit_tiny"
    assert res["complete"] is True


# --- what group recovery can and cannot do -----------------------------------

def test_recovery_fragments_signers_when_each_shot_is_unique(tmp_path):
    """The documented limitation, pinned so nobody quietly forgets it.

    A corpus with one shot per person per class has no near-duplicates linking
    one person's images, so component recovery cannot see the person. It
    fragments them instead. Measured on the real sl_digits corpus this happens to
    211 of 217 signers; here it is reproduced in miniature.

    If this test ever starts failing because recovery got smarter, that is good
    news - but the README's claims need rewriting to match.
    """
    rng = np.random.default_rng(2)
    rows = []
    for signer in range(8):
        for cls_i in range(5):                 # one shot per class per signer
            label = sources.CLASSES[cls_i]
            d = tmp_path / "oneshot" / label
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"s{signer}_{cls_i}.png"
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
            rows.append({"path": str(p), "label": label,
                         "label_idx": sources.CLASS_TO_IDX[label],
                         "source": "oneshot", "group": f"oneshot/s{signer}",
                         "split": ""})

    truth = {r["path"]: r["group"] for r in rows}
    for r in rows:
        r["group"] = "oneshot/all"
    leakage.recover_groups(rows, max_dist=6)

    spread: dict[str, set[str]] = {}
    for r in rows:
        spread.setdefault(truth[r["path"]], set()).add(r["group"])
    fragmented = sum(1 for v in spread.values() if len(v) > 1)
    assert fragmented == 8, (
        "recovery is expected to fragment every signer here; if it no longer "
        "does, update recover_groups' docstring and the README"
    )


def test_audit_only_sources_cannot_be_trained_on(tmp_path):
    """sl_digits carries label_idx=-1. Feeding it to a trainer must fail loudly."""
    rows = _write_corpus(tmp_path, "synth", n_groups=2, per_group=2)
    for r in rows:
        r["label_idx"] = -1
        r["source"] = "sl_digits"
    with pytest.raises(ValueError, match="audit-only"):
        data.SignDataset(rows)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "dataset" / "manifests" / "sl_digits.csv").exists(),
    reason="run `python -m slr.sources download sl_digits && python -m slr.sources "
           "manifest sl_digits --out dataset/manifests/sl_digits.csv` first",
)
def test_signer_rule_holds_on_the_real_digits_corpus():
    """Ground truth against real data: the frame-counter rule really does split
    2062 images into the ~218 students the paper describes."""
    man = Path(__file__).resolve().parents[2] / "dataset" / "manifests" / "sl_digits.csv"
    rows = sources.read_manifest(man)
    by_signer: dict[str, list[str]] = {}
    for r in rows:
        by_signer.setdefault(r["group"], []).append(r["label"])

    assert 210 <= len(by_signer) <= 225, f"expected ~218 signers, got {len(by_signer)}"
    # A signer shooting each digit once must not show the same digit twice. A few
    # do, where the photographer retook a shot and slipped the counter.
    dupes = sum(1 for v in by_signer.values() if len(v) != len(set(v)))
    assert dupes / len(by_signer) < 0.05, f"{dupes}/{len(by_signer)} signers have a repeated digit"
    complete = sum(1 for v in by_signer.values() if set(v) == set("0123456789"))
    assert complete > 0.7 * len(by_signer), f"only {complete}/{len(by_signer)} signers are complete"


def test_subsample_honours_the_cap_even_when_one_session_exceeds_it():
    """A recovered `asl_alphabet` session runs to thousands of frames, so a cap
    of a few hundred is only meetable by keeping part of one session. The part
    kept must stay contiguous, so it is still a single unbroken run that any
    split deals to exactly one side."""
    from slr import experiment

    rows = [{"label": "A", "group": "g0", "path": f"a{i}.jpg"} for i in range(5000)]
    rows += [{"label": "B", "group": "g1", "path": f"b{i}.jpg"} for i in range(120)]
    rows += [{"label": "B", "group": "g2", "path": f"c{i}.jpg"} for i in range(120)]

    kept = experiment.subsample(rows, per_class=200)
    by_label = {}
    for r in kept:
        by_label.setdefault(r["label"], []).append(r)

    assert len(by_label["A"]) == 200, "one oversized session must be cut to the cap"
    assert [r["path"] for r in by_label["A"]] == [f"a{i}.jpg" for i in range(200)], \
        "the kept frames must be a contiguous prefix, not a scatter"
    # B's two sessions are small enough to deal whole: 120 fits, 120+120 exceeds
    # 200, so the second is trimmed to 80 rather than dropped or taken entire.
    assert len(by_label["B"]) == 200
    assert len({r["group"] for r in by_label["B"]}) == 2


def test_group_split_survives_one_session_holding_most_of_the_corpus():
    """`recover_groups` on `asl_alphabet` collapses 87k frames into 191 sessions,
    a few of which hold nearly all of it. A first-fit walk gave one such session
    to test, overshot the 15% budget by an order of magnitude, and left train
    empty - which is what actually happened on Colab."""
    rows = [{"path": f"big{i}.jpg", "label": "A", "group": "huge", "split": ""}
            for i in range(9000)]
    for g in range(60):
        rows += [{"path": f"s{g}_{i}.jpg", "label": "B", "group": f"small{g}",
                  "split": ""} for i in range(20)]

    data.split_by_group(rows, val=0.15, test=0.15, seed=0)
    counts = Counter(r["split"] for r in rows)
    for s in ("train", "val", "test"):
        assert counts[s] > 0, f"{s} is empty: {dict(counts)}"
    assert counts["train"] >= counts["test"], \
        f"the giant session belongs in train, not test: {dict(counts)}"
    # the invariant the whole split exists to preserve
    for g in {r["group"] for r in rows}:
        assert len({r["split"] for r in rows if r["group"] == g}) == 1
