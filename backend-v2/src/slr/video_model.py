"""Temporal backbones for isolated sign recognition, and their controls.

Same discipline as the image zoo: candidates are ranked on **validation** and
only the winner ever touches test, and the cheap baselines are here as controls
rather than filler.

Three questions the zoo is built to answer, in order of how much they would
change the write-up:

  * **Does motion matter at all?** `frame_mean` classifies each frame with an
    image backbone and averages the logits - it cannot see order, so a clip
    played backwards scores identically. If it ties the sequence models, then
    these signs are separable from a single handshape and the whole video
    apparatus was unnecessary.
  * **Does attention matter, or just recurrence?** `lm_gru` against
    `lm_transformer` at matched width. Sixteen frames is a short sequence and
    a GRU may well be enough.
  * **Do landmarks beat pixels?** `lm_*` against `frame_*`. Landmarks throw
    away the background, which is the shortcut the image backend measured; if
    pixels still win, the shortcut was worth more than the geometry.

Parameter counts are small by design. Sixteen frames of 118 points is a tiny
input next to a 224x224 image, and the corpora are tens of thousands of clips,
not millions - a 300M-parameter backbone would memorise the training signers
and the signer-disjoint test split would say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .landmarks import FEATURE_DIM


@dataclass(frozen=True)
class VideoPreset:
    kind: str                      # "landmark" | "rgb"
    family: str                    # "transformer" | "rnn" | "conv" | "pool"
    dim: int = 256
    depth: int = 4
    heads: int = 8
    n_frames: int = 16
    batch_size: int = 64
    lr: float = 3e-4
    img_size: int = 224            # rgb only
    frame_backbone: str = ""       # rgb only, a key into model.PRESETS
    approx_params_m: float = 0.0
    note: str = ""


PRESETS: dict[str, VideoPreset] = {
    # --- landmark tier: the primary path -------------------------------------
    "lm_mean": VideoPreset(
        "landmark", "pool", dim=256, depth=0, batch_size=128, lr=1e-3,
        approx_params_m=0.3,
        note="CONTROL. Mean-pools the frames, then an MLP. Order-blind by "
             "construction. If this ties the sequence models, the signs in "
             "this vocabulary do not need motion to be told apart."),
    "lm_gru": VideoPreset(
        "landmark", "rnn", dim=256, depth=2, batch_size=128, lr=1e-3,
        approx_params_m=1.6,
        note="CONTROL for attention. Sixteen frames is short enough that "
             "recurrence may be all the sequence modelling this needs."),
    "lm_transformer": VideoPreset(
        "landmark", "transformer", dim=256, depth=4, heads=8,
        batch_size=128, lr=1e-3, approx_params_m=3.4,
        note="The primary candidate. Matches the PopSign input layout exactly, "
             "so it trains on landmarks and serves phone uploads unchanged."),
    "lm_transformer_lg": VideoPreset(
        "landmark", "transformer", dim=512, depth=8, heads=8,
        batch_size=64, lr=5e-4, approx_params_m=25.6,
        note="Capacity check. Included to find where returns stop on a corpus "
             "of this size, not because it is expected to win."),

    # --- rgb tier: reuses the image zoo, one backbone per frame ---------------
    "frame_mean": VideoPreset(
        "rgb", "pool", n_frames=8, batch_size=8, lr=1e-4,
        frame_backbone="vit_small", approx_params_m=21.7,
        note="CONTROL. Per-frame logits, averaged. Cannot see order at all."),
    "frame_transformer": VideoPreset(
        "rgb", "transformer", dim=384, depth=2, heads=6, n_frames=8,
        batch_size=8, lr=1e-4, frame_backbone="vit_small",
        approx_params_m=24.5,
        note="Frozen-ish image features with a temporal transformer over them. "
             "The bridge between the two backends: if this beats the landmark "
             "models, pixels carry something the geometry threw away."),
}

SWEEPS: dict[str, list[str]] = {
    "landmark": [k for k, v in PRESETS.items() if v.kind == "landmark"],
    "rgb": [k for k, v in PRESETS.items() if v.kind == "rgb"],
    "quick": ["lm_mean", "lm_gru", "lm_transformer"],
    "all": list(PRESETS),
}


# --- modules -----------------------------------------------------------------

class _TemporalPool(nn.Module):
    """Masked mean over time. The order-blind control's entire architecture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, T, D) -> (B, D)
        return x.mean(dim=1)


class LandmarkNet(nn.Module):
    """Project landmarks per frame, model time, pool, classify.

    A learned positional embedding rather than a sinusoid: clips are resampled
    to a fixed sixteen frames, so position is an index into a normalised
    gesture, not an absolute time, and there is nothing to extrapolate to.
    """

    def __init__(self, n_classes: int, family: str, dim: int = 256,
                 depth: int = 4, heads: int = 8, n_frames: int = 16,
                 in_dim: int = FEATURE_DIM, drop: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.zeros(1, n_frames, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.family = family

        if family == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=drop, batch_first=True, norm_first=True,
                activation="gelu")
            self.temporal: nn.Module = nn.TransformerEncoder(layer, depth)
            self.pool: nn.Module = _TemporalPool()
        elif family == "rnn":
            self.temporal = nn.GRU(dim, dim // 2, num_layers=depth,
                                   batch_first=True, bidirectional=True,
                                   dropout=drop if depth > 1 else 0.0)
            self.pool = _TemporalPool()
        elif family == "pool":
            self.temporal = nn.Identity()
            self.pool = _TemporalPool()
        else:
            raise ValueError(f"unknown family {family}")

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, T, in_dim)
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = self.temporal(h)
        if isinstance(h, tuple):                          # GRU returns (out, hn)
            h = h[0]
        return self.head(self.norm(self.pool(h)))


class FrameNet(nn.Module):
    """One image backbone applied to every frame, then temporal aggregation.

    This is what makes backend-v2 a superset rather than a sibling: the whole
    eighteen-backbone zoo from `model.PRESETS` is reachable here, so the video
    result is measured against the image work rather than beside it.
    """

    def __init__(self, n_classes: int, frame_backbone: str, family: str,
                 dim: int = 384, depth: int = 2, heads: int = 6,
                 n_frames: int = 8, drop: float = 0.1):
        super().__init__()
        from . import model as M

        self.backbone = M.build(frame_backbone, n_classes=0)
        feat = getattr(self.backbone, "num_features", dim)
        self.family = family
        if family == "transformer":
            self.proj = nn.Linear(feat, dim)
            self.pos = nn.Parameter(torch.zeros(1, n_frames, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=drop,
                batch_first=True, norm_first=True, activation="gelu")
            self.temporal: nn.Module = nn.TransformerEncoder(layer, depth)
            self.head = nn.Linear(dim, n_classes)
        else:
            self.proj = nn.Identity()
            self.temporal = nn.Identity()
            self.head = nn.Linear(feat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, T, 3, H, W)
        b, t = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).reshape(b, t, -1)
        if self.family == "transformer":
            f = self.proj(f) + self.pos[:, :t]
            f = self.temporal(f)
        return self.head(f.mean(dim=1))


def build(preset: str, n_classes: int) -> nn.Module:
    p = PRESETS[preset]
    if p.kind == "landmark":
        m: nn.Module = LandmarkNet(n_classes, p.family, p.dim, p.depth,
                                   p.heads, p.n_frames)
    else:
        m = FrameNet(n_classes, p.frame_backbone, p.family, p.dim, p.depth,
                     p.heads, p.n_frames)
    m.slr_preset = preset  # type: ignore[attr-defined]
    return m


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
