from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from . import sources

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = sources.DEFAULT_ROOT.parent / "models" / "hand_landmarker.task"

MARGIN = 0.3
MIN_CONFIDENCE = 0.2

_DETECTOR = None


def model_path() -> Path:
    """Return the landmarker bundle, downloading it once if it is not here."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[hands] fetching landmarker -> {MODEL_PATH}")
        with urllib.request.urlopen(MODEL_URL, timeout=300) as r:
            MODEL_PATH.write_bytes(r.read())
    return MODEL_PATH


def available() -> bool:
    """True when this machine can actually detect hands."""
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        return True
    except ImportError:
        return False


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        from mediapipe.tasks.python import BaseOptions, vision

        _DETECTOR = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path())),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=MIN_CONFIDENCE,
            )
        )
    return _DETECTOR


def _landmarks(arr: np.ndarray):
    import mediapipe as mp

    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))
    res = _detector().detect(img)
    if not res.hand_landmarks:
        return None
    side = res.handedness[0][0].category_name if res.handedness else "Right"
    return res.hand_landmarks[0], side


def detect_hand(arr: np.ndarray, retry: bool = True
                ) -> tuple[np.ndarray, str] | None:
    """The 21 landmarks as (21, 3) pixel coordinates, plus which hand.

    Pixels rather than MediaPipe's 0-1 coordinates, which squash the hand on a
    non-square photo. z is scaled like x.

    If nothing is found, retry once on a padded, upscaled copy, since MediaPipe
    struggles when the hand fills the frame. `retry=False` skips that.
    """
    import cv2

    h, w = arr.shape[:2]
    got = _landmarks(arr)
    if got is not None:
        lm, side = got
        return np.array([[p.x * w, p.y * h, p.z * w] for p in lm],
                        dtype=np.float32), side

    if not retry:
        return None

    pad = int(max(h, w) * MARGIN)
    big = cv2.resize(cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_REPLICATE),
                     (384, 384), interpolation=cv2.INTER_CUBIC)
    got = _landmarks(big)
    if got is None:
        return None
    lm, side = got
    sx, sy = (w + 2 * pad) / 384, (h + 2 * pad) / 384
    return np.array([[p.x * 384 * sx - pad, p.y * 384 * sy - pad, p.z * 384 * sx]
                     for p in lm], dtype=np.float32), side


def _box(pts: list[tuple[float, float]], w: int, h: int) -> tuple[int, int, int, int]:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * MARGIN)
    n = min(round(side), w, h)
    x0 = min(max(round(cx - n / 2), 0), w - n)
    y0 = min(max(round(cy - n / 2), 0), h - n)
    return (x0, y0, x0 + n, y0 + n)
