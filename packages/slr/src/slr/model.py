from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Preset:
    timm_name: str
    img_size: int = 224
    batch_size: int = 64
    lr: float = 3e-4
    layer_decay: float | None = None
    approx_params_m: float = 0.0


# ponytail: only the served backbone. The architecture search that picked it is in git history.
PRESETS: dict[str, Preset] = {
    "deit3_base": Preset("deit3_base_patch16_224.fb_in22k_ft_in1k", batch_size=48, lr=1e-4,
                         layer_decay=0.75, approx_params_m=85.8),
}


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build(preset: str, n_classes: int, drop_path: float = 0.1,
          pretrained: bool = True) -> nn.Module:
    import timm

    m = timm.create_model(PRESETS[preset].timm_name, pretrained=pretrained,
                          num_classes=n_classes, drop_path_rate=drop_path)
    m.slr_preset = preset
    return m


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def param_groups(m: nn.Module, lr: float, weight_decay: float = 0.05,
                 layer_decay: float | None = None) -> list[dict]:
    """Optimiser groups: no weight decay on norms and biases, optional layer-wise LR decay."""
    n_layers = len(getattr(m, "blocks", []) or [])
    if layer_decay is not None and not n_layers:
        print(f"[model] {type(m).__name__} has no flat .blocks; "
              "ignoring layer_decay and using a uniform learning rate")
        layer_decay = None

    def _decay(name: str) -> float:
        if layer_decay is None:
            return 1.0
        if name.startswith(("cls_token", "pos_embed", "patch_embed")):
            depth = 0
        elif name.startswith("blocks."):
            depth = int(name.split(".")[1]) + 1
        else:
            depth = n_layers + 1
        return layer_decay ** (n_layers + 1 - depth)

    groups: dict[tuple[float, bool], dict] = {}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = p.ndim <= 1 or name.endswith(".bias")
        key = (_decay(name), no_wd)
        g = groups.setdefault(key, {
            "params": [], "lr": lr * key[0],
            "weight_decay": 0.0 if no_wd else weight_decay,
        })
        g["params"].append(p)
    return list(groups.values())


@torch.enable_grad()
def grad_attention(model: nn.Module, x: torch.Tensor, class_idx: int | None = None
                   ) -> torch.Tensor:
    """Chefer et al. gradient-weighted attention rollout: a (B, H, W) map."""
    blocks = getattr(model, "blocks", None)
    if not blocks:
        raise TypeError(f"{type(model).__name__} has no .blocks; not a timm ViT")

    maps: list[torch.Tensor] = []
    handles, saved_fused = [], []

    def keep(_m, inp, _out):
        att = inp[0]
        att.retain_grad()
        maps.append(att)

    for blk in blocks:
        saved_fused.append(getattr(blk.attn, "fused_attn", None))
        blk.attn.fused_attn = False
        handles.append(blk.attn.attn_drop.register_forward_hook(keep))
    try:
        out = model(x)
        idx = int(out[0].argmax()) if class_idx is None else class_idx
        model.zero_grad(set_to_none=True)
        out[0, idx].backward()
    finally:
        for h in handles:
            h.remove()
        for blk, f in zip(blocks, saved_fused):
            if f is not None:
                blk.attn.fused_attn = f

    n = maps[0].shape[-1]
    eye = torch.eye(n, device=x.device)
    roll = eye.clone()
    for att in maps:
        if att.grad is None:
            continue
        w = (att.grad * att).clamp(min=0).mean(dim=1)[0] + eye
        roll = (w / w.sum(dim=-1, keepdim=True)) @ roll

    mask = roll[0, 1:].detach()
    side = int(mask.numel() ** 0.5)
    mask = mask[: side * side].reshape(1, side, side)
    mask = mask - mask.amin()
    return mask / mask.amax().clamp_min(1e-8)
