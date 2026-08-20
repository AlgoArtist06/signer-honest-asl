"""MediaPipe Holistic landmarks: the representation the video pipeline trains on.

Why landmarks rather than pixels, given the image backend spent eighteen
backbones on pixels: the image work established that the thing wrecking
published accuracy is the model memorising a room. Landmarks delete that
shortcut at the input rather than augmenting around it - there is no wallpaper
in a list of joint coordinates - which is exactly the gap `backend/README`
flagged and did not close.

It is also what the data allows. PopSign, the one corpus here with real signer
ids for all 21 signers, is *released* as MediaPipe Holistic landmarks and no
video at all. Matching its layout exactly means the same model consumes a
PopSign sample and a frame extracted from a phone upload.

Layout, in PopSign's order, 543 points per frame:

    face        468     indices    0..467
    left_hand    21     indices  468..488
    pose         33     indices  489..521
    right_hand   21     indices  522..542

`SELECTED` keeps 118 of those. The face mesh is 468 of the 543 points and
almost all of it is cheekbone and forehead that no sign depends on; keeping it
would let the model spend its capacity on face shape, which is signer identity
wearing a disguise. Hands, arms and the mouth carry the sign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_POINTS = 543
FACE = slice(0, 468)
LEFT_HAND = slice(468, 489)
POSE = slice(489, 522)
RIGHT_HAND = slice(522, 543)

# Mouth outline from the MediaPipe face mesh. Mouthing distinguishes signs that
# are manually identical, so it earns its place where the rest of the face
# does not.
LIPS = [61, 291, 0, 17, 78, 308, 13, 14, 82, 87, 312, 317, 40, 270, 88, 318]
# Shoulders, elbows and wrists. Everything below the hips is off-camera in a
# seated capture and contributes nothing but noise.
ARMS = [489 + i for i in (11, 12, 13, 14, 15, 16, 23, 24)]

SELECTED = np.array(
    LIPS
    + list(range(LEFT_HAND.start, LEFT_HAND.stop))
    + ARMS
    + list(range(RIGHT_HAND.start, RIGHT_HAND.stop)),
    dtype=np.int64,
)
N_SELECTED = len(SELECTED)


def extract(frames: np.ndarray) -> np.ndarray:
    """Run MediaPipe Holistic over `(T, H, W, 3)` uint8 frames.

    Returns `(T, 543, 3)` float32 in PopSign's layout and order, with NaN where
    a part was not detected - which is the same sentinel PopSign itself uses for
    a hand that left the frame, so downstream code needs one missing-data rule
    rather than two.
    """
    import mediapipe as mp

    out = np.full((len(frames), N_POINTS, 3), np.nan, dtype=np.float32)
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1, refine_face_landmarks=False)
    try:
        for t, frame in enumerate(frames):
            res = holistic.process(frame)
            for part, sl in ((res.face_landmarks, FACE),
                             (res.left_hand_landmarks, LEFT_HAND),
                             (res.pose_landmarks, POSE),
                             (res.right_hand_landmarks, RIGHT_HAND)):
                if part is None:
                    continue
                pts = np.array([[p.x, p.y, p.z] for p in part.landmark],
                               dtype=np.float32)
                out[t, sl][: len(pts)] = pts[: sl.stop - sl.start]
    finally:
        holistic.close()
    return out


def select(seq: np.ndarray) -> np.ndarray:
    """Keep the `SELECTED` points: `(T, 543, 3)` -> `(T, N_SELECTED, 3)`."""
    return seq[:, SELECTED, :]


def normalise(seq: np.ndarray) -> np.ndarray:
    """Centre on the shoulders and scale by shoulder width, per frame.

    Without this the model can read the signer's distance from the camera and
    their body size straight off the coordinates, which is signer identity by
    another route - the same failure the image side found in backgrounds.
    Shoulder width is the one length that is visible in every frame and does not
    change as the hands move.

    NaNs are filled with zero *after* centring, so a missing hand sits at the
    body centre rather than at the origin of the frame, and the model can tell
    "absent" from "at the far corner".
    """
    seq = seq.astype(np.float32).copy()
    # Shoulders are the first two ARMS entries, offset past LIPS in SELECTED.
    li, ri = len(LIPS) + 21, len(LIPS) + 21 + 1
    left, right = seq[:, li, :2], seq[:, ri, :2]

    centre = np.nanmean(np.stack([left, right]), axis=0)          # (T, 2)
    width = np.linalg.norm(left - right, axis=-1)                 # (T,)
    scale = np.nanmedian(width[np.isfinite(width)]) if np.isfinite(width).any() else 0.0
    if not scale or not np.isfinite(scale):
        scale = 1.0

    seq[..., :2] -= centre[:, None, :]
    seq[..., :2] /= scale
    seq[..., 2] /= scale
    return np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)


def to_features(seq: np.ndarray) -> np.ndarray:
    """`(T, 543, 3)` raw holistic -> `(T, N_SELECTED * 3)` model input.

    Motion is what separates two signs made with the same handshape, so the
    per-frame delta is appended rather than left for the model to rediscover
    from position alone.
    """
    xyz = normalise(select(seq))
    delta = np.diff(xyz, axis=0, prepend=xyz[:1])
    return np.concatenate([xyz, delta], axis=-1).reshape(len(xyz), -1)


FEATURE_DIM = N_SELECTED * 6      # xyz + delta, flattened


def cache_path(video: str | Path, root: Path) -> Path:
    """Where the landmarks for one clip live once extracted.

    Holistic runs at roughly video speed on CPU, so re-extracting every epoch
    would dominate training. Extract once, train many times.
    """
    stem = Path(video).stem
    return root / f"{stem}.npy"


def load_or_extract(video: str | Path, root: Path, n_frames: int) -> np.ndarray:
    """Cached `(T, 543, 3)` for one clip, extracting on first use."""
    from . import clips

    root.mkdir(parents=True, exist_ok=True)
    p = cache_path(video, root)
    if p.exists():
        return np.load(p)
    seq = extract(clips.read_clip(video, n_frames))
    np.save(p, seq)
    return seq
