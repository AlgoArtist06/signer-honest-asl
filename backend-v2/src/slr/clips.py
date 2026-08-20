"""Reading video clips, and the policy for an upload longer than one sign.

The awkward fact this module exists to handle: **no isolated-sign corpus
contains ten-second samples.** WLASL clips average 2.4s, AUTSL about 1.8s,
PopSign 1.4s. A ten-second upload therefore holds either one very slow sign or
several signs in a row, and a model trained on 1-2s samples has never seen
either.

So a long upload is not fed to the classifier as one sample. It is cut into
overlapping windows the length of a training clip, every window is scored, and
the caller gets the whole timeline back plus the best window. That is honest
about what was actually classified, and it degrades sensibly: on a clip that
really does hold one sign, every window agrees and the timeline is flat.

Frame sampling is uniform across the window rather than a contiguous run.
Signs are slow relative to 25-60 fps, so consecutive frames are near-identical -
the same redundancy `leakage` measures on the image corpora - and a contiguous
run would spend the whole frame budget on a fraction of the gesture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# A training clip's worth of video. Matches the corpora rather than the upload:
# see the module docstring for why those are not the same number.
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5
N_FRAMES = 16


@dataclass(frozen=True)
class Window:
    """One scored slice of a longer upload."""
    start: float
    end: float
    frames: np.ndarray          # (T, H, W, 3) uint8


def probe(path: str | Path) -> tuple[int, float]:
    """Return `(n_frames, fps)` without decoding the whole file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    cap.release()
    return n, fps


def sample_indices(total: int, n: int) -> np.ndarray:
    """`n` frame indices spread evenly over `total` frames.

    Short clips are padded by repeating the last frame rather than by looping
    back to the start: a sign is not periodic, and looping would show the model
    a gesture that reverses halfway through.
    """
    if total <= 0:
        raise ValueError("clip has no frames")
    if total >= n:
        return np.linspace(0, total - 1, n).round().astype(int)
    idx = np.arange(total)
    return np.concatenate([idx, np.full(n - total, total - 1)])


def read_clip(path: str | Path, n_frames: int = N_FRAMES,
              start: float = 0.0, end: float | None = None) -> np.ndarray:
    """Decode `n_frames` uniformly sampled RGB frames from `[start, end)` seconds.

    Returns `(T, H, W, 3)` uint8. Frames are pulled by index in one forward
    pass rather than by seeking to each one: seeking in a compressed stream
    lands on the nearest keyframe, which silently returns the wrong frame.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lo = max(0, int(start * fps))
    hi = total if end is None else min(total, int(end * fps))
    wanted = set((lo + sample_indices(max(hi - lo, 1), n_frames)).tolist())

    out: list[np.ndarray] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
        if i >= hi:
            break
    cap.release()

    if not out:
        raise ValueError(f"decoded no frames from {path} in [{start}, {end})")
    while len(out) < n_frames:           # ragged tail, or a truncated file
        out.append(out[-1])
    return np.stack(out[:n_frames])


def windows(path: str | Path, window: float = WINDOW_SECONDS,
            stride: float = STRIDE_SECONDS, n_frames: int = N_FRAMES
            ) -> list[Window]:
    """Cut an upload into overlapping training-clip-length windows.

    A clip at or under `window` seconds yields exactly one window covering all
    of it, so the short case costs nothing extra.
    """
    total, fps = probe(path)
    duration = total / fps if fps else 0.0
    if duration <= window:
        return [Window(0.0, duration, read_clip(path, n_frames))]

    starts = np.arange(0.0, duration - window + 1e-6, stride)
    return [Window(float(s), float(s + window),
                   read_clip(path, n_frames, float(s), float(s + window)))
            for s in starts]


def duration_warning(seconds: float, mean_clip_seconds: float = 2.0) -> str | None:
    """Say plainly when an upload is far longer than anything trained on.

    Returned rather than logged so the API can hand it to the caller: a user who
    filmed ten seconds deserves to know the model is reading a two-second slice
    of it, not silently getting one label for the whole thing.
    """
    if seconds <= mean_clip_seconds * 2:
        return None
    return (f"clip is {seconds:.1f}s but the corpora average "
            f"{mean_clip_seconds:.1f}s per sign; scoring "
            f"{WINDOW_SECONDS:.0f}s windows every {STRIDE_SECONDS:.1f}s and "
            f"reporting the best, plus the full timeline")
