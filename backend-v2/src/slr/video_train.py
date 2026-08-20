"""Fine-tuning loop and architecture search for the video backbones.

Deliberately the same shape as `train.py`: one loop, one optimiser, one
schedule, selection on validation macro-F1, and a leaderboard rewritten after
every candidate so a dropped Colab session costs the model in flight rather
than the sweep. The interesting decisions are upstream in how the split was
made, not in here.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import video_data, video_model as VM


def _macro_f1(conf: torch.Tensor) -> float:
    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    present = conf.sum(1) > 0
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-9)
    return f1[present].mean().item() if present.any() else 0.0


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_split(net: nn.Module, loader, dev: torch.device, n_classes: int
                   ) -> tuple[float, float, float]:
    net.eval()
    crit = nn.CrossEntropyLoss()
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev.type, dtype=torch.float16,
                            enabled=dev.type == "cuda"):
            out = net(x)
            loss = crit(out, y)
        total += loss.item() * y.numel()
        n += y.numel()
        for t, p in zip(y.cpu(), out.argmax(1).cpu()):
            conf[t, p] += 1
    return total / max(n, 1), conf.diag().sum().item() / max(n, 1), _macro_f1(conf)


def train_one(rows: list[dict], preset: str, epochs: int = 10,
              out_dir: Path = Path("runs"), workers: int = 2,
              label_smoothing: float = 0.1, seed: int = 0, tag: str = "",
              eval_test: bool = True) -> dict:
    """Fine-tune one temporal backbone.

    `eval_test=False` during a search: candidates are compared on validation
    alone and the test split stays sealed until one winner is chosen.
    """
    torch.manual_seed(seed)
    p = VM.PRESETS[preset]
    dev = device()
    classes = video_data.label_space(rows)
    n_classes = len(classes)

    dl = video_data.loaders(rows, kind=p.kind, batch_size=p.batch_size,
                            workers=workers, n_frames=p.n_frames,
                            **({"img_size": p.img_size} if p.kind == "rgb" else {}))
    if "train" not in dl:
        raise ValueError("no train rows in manifest")

    net = VM.build(preset, n_classes).to(dev)
    print(f"[train] {preset} ({VM.n_params(net) / 1e6:.1f}M params) on {dev}, "
          f"{len(dl['train'].dataset)} clips, {n_classes} signs")

    decay = [q for q in net.parameters() if q.requires_grad and q.ndim > 1]
    no_decay = [q for q in net.parameters() if q.requires_grad and q.ndim <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.05},
         {"params": no_decay, "weight_decay": 0.0}], lr=p.lr)

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
            with torch.autocast(dev.type, dtype=torch.float16,
                                enabled=dev.type == "cuda"):
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
                torch.save({"preset": preset, "classes": classes,
                            "n_frames": p.n_frames, "kind": p.kind,
                            "state_dict": net.state_dict()}, run / "best.pt")
        history.append(rec)
        print("[train] ep{:02d} ".format(ep) + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in rec.items()))

    result = {"preset": preset, "params_m": round(VM.n_params(net) / 1e6, 2),
              "epochs": epochs, "n_classes": n_classes, "best": best,
              "history": history}

    if eval_test and "test" in dl:
        if (run / "best.pt").exists():
            net.load_state_dict(
                torch.load(run / "best.pt", map_location=dev)["state_dict"])
        tl, ta, tf = evaluate_split(net, dl["test"], dev, n_classes)
        result["test"] = {"loss": tl, "acc": ta, "macro_f1": tf}
        print(f"[train] TEST acc={ta:.4f} macro_f1={tf:.4f}")

    (run / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[ok] {run}")
    return result


def sweep(rows: list[dict], presets: list[str], epochs: int = 10,
          out_dir: Path = Path("runs"), workers: int = 2, seed: int = 0,
          tag: str = "video_sweep") -> dict:
    """Rank temporal backbones on validation, then score the winner once."""
    if not any(r["split"] == "val" for r in rows):
        raise ValueError("an architecture search needs a validation split")

    out_dir.mkdir(parents=True, exist_ok=True)
    board_path = out_dir / f"{tag}_leaderboard.json"
    board: list[dict] = []
    if board_path.exists():
        board = json.loads(board_path.read_text()).get("leaderboard", [])
        print(f"[sweep] resuming: {len(board)} candidate(s) already scored")

    for name in presets:
        if any(b["preset"] == name for b in board):
            print(f"[sweep] {name} already scored; skipping")
            continue
        try:
            r = train_one(rows, name, epochs, out_dir, workers, seed=seed,
                          tag=f"{tag}_{name}", eval_test=False)
        except RuntimeError as e:
            print(f"[sweep] {name} failed ({type(e).__name__}: {e}); skipping")
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        p = VM.PRESETS[name]
        board.append({
            "preset": name, "kind": p.kind, "family": p.family,
            "params_m": r["params_m"], "n_frames": p.n_frames,
            "val_f1": r["best"]["val_f1"], "val_acc": r["best"]["val_acc"],
            "secs_per_epoch": round(
                sum(h["secs"] for h in r["history"]) / max(len(r["history"]), 1), 1),
        })
        board_path.write_text(json.dumps({"leaderboard": board,
                                          "complete": False}, indent=2))

    if not board:
        raise RuntimeError("every candidate failed")

    board.sort(key=lambda b: -b["val_f1"])
    winner = board[0]["preset"]
    print(f"\n[sweep] winner on validation: {winner}")
    final = train_one(rows, winner, epochs, out_dir, workers, seed=seed,
                      tag=f"{tag}_WINNER_{winner}", eval_test=True)
    board_path.write_text(json.dumps(
        {"leaderboard": board, "winner": winner, "complete": True}, indent=2))
    return {"leaderboard": board, "winner": winner,
            "winner_test": final.get("test")}
