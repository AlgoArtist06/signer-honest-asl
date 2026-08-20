"""Video pipeline tests. No dataset, no GPU, no MediaPipe, no network.

MediaPipe and OpenCV are deliberately not imported here: the parts worth
testing are the geometry, the sampling policy and the split, and all three are
pure array work. Anything that needs to decode an actual mp4 is an integration
concern, not a unit one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from slr import clips, landmarks, video_data, video_model as VM


# --- frame sampling ----------------------------------------------------------

@pytest.mark.parametrize("total,n", [(100, 16), (16, 16), (5, 16), (1, 16)])
def test_sample_indices_always_returns_exactly_n_valid_frames(total, n):
    idx = clips.sample_indices(total, n)
    assert len(idx) == n
    assert idx.min() >= 0 and idx.max() < total
    assert (np.diff(idx) >= 0).all(), "sampling must not run time backwards"


def test_short_clips_pad_by_repeating_the_last_frame_not_by_looping():
    """Looping would show the model a gesture that reverses halfway through."""
    idx = clips.sample_indices(3, 8)
    assert list(idx) == [0, 1, 2, 2, 2, 2, 2, 2]


def test_sample_indices_rejects_an_empty_clip():
    with pytest.raises(ValueError):
        clips.sample_indices(0, 8)


def test_duration_warning_fires_only_for_uploads_far_longer_than_a_sign():
    assert clips.duration_warning(1.8, 2.0) is None
    assert clips.duration_warning(3.5, 2.0) is None
    msg = clips.duration_warning(10.0, 2.0)
    assert msg is not None and "10.0s" in msg


# --- landmark geometry -------------------------------------------------------

def _fake_sequence(t: int = 4, shift: float = 0.0, scale: float = 1.0):
    """A holistic sequence with shoulders placed where `normalise` expects."""
    seq = np.zeros((t, landmarks.N_POINTS, 3), dtype=np.float32)
    li = landmarks.SELECTED[len(landmarks.LIPS) + 21]
    ri = landmarks.SELECTED[len(landmarks.LIPS) + 21 + 1]
    seq[:, li, :2] = np.array([-0.1, 0.0]) * scale + shift
    seq[:, ri, :2] = np.array([0.1, 0.0]) * scale + shift
    rng = np.random.default_rng(0)
    hand = slice(landmarks.LEFT_HAND.start, landmarks.LEFT_HAND.stop)
    seq[:, hand, :] = rng.normal(0, 0.05, (t, 21, 3)).astype(np.float32) * scale + shift
    return seq


def test_selection_keeps_hands_and_arms_and_drops_the_face_mesh():
    """468 of 543 points are face. Keeping them would let the model spend its
    capacity on face shape, which is signer identity wearing a disguise."""
    assert landmarks.N_SELECTED == len(landmarks.SELECTED)
    assert landmarks.N_SELECTED < 150, "the face mesh must not survive selection"
    for h in (landmarks.LEFT_HAND, landmarks.RIGHT_HAND):
        assert set(range(h.start, h.stop)) <= set(landmarks.SELECTED.tolist())


def test_normalise_removes_camera_distance_and_signer_size():
    """Two signers of different build at different distances must land on the
    same coordinates, or the model can read identity straight off the input."""
    near = landmarks.normalise(landmarks.select(_fake_sequence(shift=0.0, scale=1.0)))
    far = landmarks.normalise(landmarks.select(_fake_sequence(shift=0.4, scale=2.0)))
    assert np.allclose(near, far, atol=1e-4)


def test_normalise_fills_missing_points_at_the_body_centre_not_the_frame_origin():
    seq = _fake_sequence()
    seq[:, landmarks.RIGHT_HAND, :] = np.nan      # hand left the frame
    out = landmarks.normalise(landmarks.select(seq))
    assert np.isfinite(out).all(), "NaN must not reach the model"
    off = len(landmarks.LIPS) + 21 + len(landmarks.ARMS)
    assert np.allclose(out[:, off:, :], 0.0)


def test_features_carry_motion_as_well_as_position():
    feats = landmarks.to_features(_fake_sequence(t=6))
    assert feats.shape == (6, landmarks.FEATURE_DIM)
    assert np.isfinite(feats).all()


def test_a_still_clip_has_zero_motion_channel():
    """A frozen hand must produce zero deltas, so the model can tell a held
    handshape from a moving one rather than inferring it from noise."""
    still = _fake_sequence(t=5)
    still[:] = still[:1]
    feats = landmarks.to_features(still)
    half = landmarks.N_SELECTED * 3
    assert np.allclose(feats[:, half:], 0.0)


# --- splitting ---------------------------------------------------------------

def _clip_rows(n_signers=10, per_signer=20):
    rows = []
    for s in range(n_signers):
        for i in range(per_signer):
            rows.append({"path": f"s{s}_{i}.mp4", "label": f"sign{i % 5}",
                         "label_idx": i % 5, "source": "lsa64",
                         "signer": f"signer{s:03d}", "group": f"lsa64/signer{s:03d}",
                         "split": ""})
    return rows


def test_signer_split_shares_no_signer_between_train_and_test():
    rows = _clip_rows()
    video_data.split_by_signer(rows, val=0.2, test=0.2, seed=0)
    tr = {r["signer"] for r in rows if r["split"] == "train"}
    te = {r["signer"] for r in rows if r["split"] == "test"}
    va = {r["signer"] for r in rows if r["split"] == "val"}
    assert tr and te and va
    assert not (tr & te) and not (tr & va) and not (te & va)


def test_signer_split_survives_one_signer_dominating_the_corpus():
    """The heavy-tail failure that emptied train on the image side, in the
    signer domain: one prolific signer must not swallow a whole split."""
    rows = _clip_rows(n_signers=6, per_signer=10)
    rows += [{"path": f"whale_{i}.mp4", "label": f"sign{i % 5}", "label_idx": i % 5,
              "source": "lsa64", "signer": "signer999",
              "group": "lsa64/signer999", "split": ""} for i in range(400)]
    video_data.split_by_signer(rows, val=0.15, test=0.15, seed=0)
    from collections import Counter
    counts = Counter(r["split"] for r in rows)
    assert all(counts[s] > 0 for s in ("train", "val", "test")), dict(counts)


def test_signer_split_refuses_a_corpus_with_no_signer_labels():
    """WLASL has none. Falling back to a random split silently would produce
    exactly the inflated number this project exists to refuse."""
    rows = _clip_rows()
    for r in rows:
        r["signer"] = ""
    with pytest.raises(ValueError, match="signer"):
        video_data.split_by_signer(rows)


def test_label_space_is_read_from_the_manifest_not_hard_coded():
    rows = _clip_rows()
    assert video_data.label_space(rows) == [f"sign{i}" for i in range(5)]


# --- models ------------------------------------------------------------------

@pytest.mark.parametrize("preset", VM.SWEEPS["landmark"])
def test_every_landmark_backbone_builds_and_maps_a_clip_to_logits(preset):
    p = VM.PRESETS[preset]
    net = VM.build(preset, n_classes=250)
    x = torch.randn(2, p.n_frames, landmarks.FEATURE_DIM)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (2, 250)
    assert torch.isfinite(out).all()


def test_the_order_blind_control_really_is_order_blind():
    """`lm_mean` is the control that answers 'does motion matter at all'. If it
    ever became order-sensitive the comparison would quietly stop meaning
    anything, so the property is pinned here."""
    net = VM.build("lm_mean", n_classes=10).eval()
    x = torch.randn(1, 16, landmarks.FEATURE_DIM)
    with torch.no_grad():
        a = net(x)
        b = net(x.flip(dims=[1]))
    assert torch.allclose(a, b, atol=1e-5)


def test_the_sequence_models_are_not_order_blind():
    net = VM.build("lm_transformer", n_classes=10).eval()
    x = torch.randn(1, 16, landmarks.FEATURE_DIM)
    with torch.no_grad():
        a, b = net(x), net(x.flip(dims=[1]))
    assert not torch.allclose(a, b, atol=1e-4)
