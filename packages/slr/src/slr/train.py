"""Fine-tuning loop for the served pixel model.

Model selection is on validation macro-F1 rather than accuracy: the corpora are
imbalanced and a few classes (`del`, `nothing`) are tiny, so plain accuracy can
be bought by ignoring them.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import data, model as M, safe_load, sources


def _macro_f1(conf: torch.Tensor) -> float:
    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    present = (conf.sum(1) > 0)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-9)
    return f1[present].mean().item() if present.any() else 0.0


@torch.no_grad()
def evaluate_split(net: nn.Module, loader, dev: torch.device, n_classes: int
                   ) -> tuple[float, float, float]:
    """Return (loss, accuracy, macro-F1) over one loader."""
    net.eval()
    crit = nn.CrossEntropyLoss()
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = net(x)
            loss = crit(out, y)
        total_loss += loss.item() * y.numel()
        n += y.numel()
        for t, p in zip(y.cpu(), out.argmax(1).cpu()):
            conf[t, p] += 1
    acc = conf.diag().sum().item() / max(n, 1)
    return total_loss / max(n, 1), acc, _macro_f1(conf)


def train_one(rows: list[dict], preset: str, epochs: int = 12, out_dir: Path = Path("runs"),
              workers: int = 4, label_smoothing: float = 0.1, seed: int = 0,
              tag: str = "") -> dict:
    """Fine-tune one backbone, keep the best epoch on validation, then score test."""
    torch.manual_seed(seed)
    p = M.PRESETS[preset]
    dev = M.device()
    n_classes = len(sources.CLASSES)

    dl = data.loaders(rows, p.img_size, p.batch_size, workers)
    if "train" not in dl:
        raise ValueError("no train rows in manifest")

    net = M.build(preset, n_classes).to(dev)
    print(f"[train] {preset} ({M.n_params(net) / 1e6:.1f}M params) on {dev}, "
          f"{len(dl['train'].dataset)} train imgs, batch {p.batch_size}")

    opt = torch.optim.AdamW(
        M.param_groups(net, p.lr, layer_decay=p.layer_decay), lr=p.lr, betas=(0.9, 0.999))
    steps = max(1, epochs * len(dl["train"]))
    warmup = min(500, steps // 10)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    run = out_dir / (tag or f"{preset}_{int(time.time())}")
    run.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = {"val_f1": -1.0}

    for ep in range(epochs):
        net.train()
        t0, seen, run_loss = time.time(), 0, 0.0
        for x, y in dl["train"]:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = crit(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * y.numel()
            seen += y.numel()

        rec = {"epoch": ep, "train_loss": run_loss / max(seen, 1),
               "lr": sched.get_last_lr()[0], "secs": round(time.time() - t0, 1)}
        if "val" in dl:
            vl, va, vf = evaluate_split(net, dl["val"], dev, n_classes)
            rec |= {"val_loss": vl, "val_acc": va, "val_f1": vf}
            if vf > best["val_f1"]:
                best = {"epoch": ep, "val_f1": vf, "val_acc": va}
                torch.save({"preset": preset, "classes": sources.CLASSES,
                            "img_size": p.img_size,
                            "state_dict": net.state_dict()},
                           run / "best.pt")
        history.append(rec)
        print(f"[train] ep{ep:02d} " + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))

    result = {"preset": preset, "params_m": round(M.n_params(net) / 1e6, 2),
              "epochs": epochs, "batch_size": p.batch_size,
              "best": best, "history": history}

    if "test" in dl:
        if (run / "best.pt").exists():
            net.load_state_dict(safe_load(run / "best.pt", map_location=dev)["state_dict"])
        tl, ta, tf = evaluate_split(net, dl["test"], dev, n_classes)
        result["test"] = {"loss": tl, "acc": ta, "macro_f1": tf}
        print(f"[train] TEST acc={ta:.4f} macro_f1={tf:.4f}")

    (run / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[ok] {run}")
    return result
