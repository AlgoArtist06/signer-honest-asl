"""HTTP inference service - the surface the React Native app talks to.

One meaningful endpoint, `POST /predict`, taking a photo and returning a letter,
a calibrated confidence, and an explicit `abstain` flag. The app should render
the abstain case as "hold steady, try again" rather than showing a letter it has
no business showing.

The temperature and the confidence threshold are read from the evaluation report
produced by `slr.evaluate`, so the threshold the service uses is the same one
whose error rate was measured, rather than a number somebody guessed.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from . import data, model as M

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STATE: dict = {}


class Prediction(BaseModel):
    label: str
    confidence: float
    abstain: bool
    top_k: list[dict]
    latency_ms: float
    attention_png_b64: str | None = None


def _load() -> None:
    """Read the checkpoint plus, if present, the calibration report beside it."""
    ckpt = Path(os.environ.get("SLR_CHECKPOINT", "runs/best.pt"))
    if not ckpt.exists():
        raise RuntimeError(
            f"no checkpoint at {ckpt}. Train one first, or point SLR_CHECKPOINT at it."
        )
    dev = M.device()
    blob = torch.load(ckpt, map_location=dev)
    net = M.build(blob["preset"], len(blob["classes"]), pretrained=False).to(dev).eval()
    net.load_state_dict(blob["state_dict"])

    temp, thresh = 1.0, float(os.environ.get("SLR_THRESHOLD", 0.0))
    report = Path(os.environ.get("SLR_EVAL_REPORT", ckpt.parent / "eval.json"))
    if report.exists():
        r = json.loads(report.read_text())
        temp = r.get("temperature", 1.0)
        if not os.environ.get("SLR_THRESHOLD"):
            # Default to the threshold measured at 90% coverage on the honest
            # test split: answer nine photos in ten, at a known error rate.
            for row in r.get("test", {}).get("risk_coverage", []):
                if abs(row["coverage"] - 0.9) < 0.03:
                    thresh = row["threshold"]
        print(f"[api] calibration from {report}: T={temp:.3f} threshold={thresh:.3f}")
    else:
        print(f"[api] no eval report at {report}; confidences are UNCALIBRATED")

    STATE.update(net=net, dev=dev, classes=blob["classes"], temp=temp,
                 threshold=thresh, preset=blob["preset"],
                 tf=data.build_transforms(blob["img_size"], train=False),
                 img_size=blob["img_size"], ckpt=str(ckpt))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    STATE.clear()


app = FastAPI(title="Sign Language Recognition API", version="0.1.0", lifespan=lifespan)
# ponytail: wide-open CORS. Fine while the only client is the RN app over a LAN;
# pin allow_origins to the deployed host before this faces the internet.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {
        "ok": bool(STATE),
        "preset": STATE.get("preset"),
        "checkpoint": STATE.get("ckpt"),
        "device": str(STATE.get("dev")),
        "n_classes": len(STATE.get("classes", [])),
        "temperature": STATE.get("temp"),
        "abstain_below": STATE.get("threshold"),
    }


def _decode(raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()                       # cheap structural check
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(415, f"not a decodable image: {e}") from e


def _heatmap_png(mask: torch.Tensor, base: Image.Image) -> str:
    """Overlay the attention rollout on the photo and return it base64-encoded."""
    import numpy as np

    m = (mask.squeeze(0).cpu().numpy() * 255).astype("uint8")
    heat = Image.fromarray(m, "L").resize(base.size, Image.Resampling.BILINEAR)
    # Simple red-channel overlay; avoids pulling matplotlib in for a colourmap.
    arr = np.asarray(base.convert("RGB")).astype("float32")
    h = np.asarray(heat).astype("float32")[..., None] / 255.0
    arr = arr * (1 - 0.5 * h) + np.array([255.0, 40.0, 40.0]) * (0.5 * h)
    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8")).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/predict", response_model=Prediction)
async def predict(
    image: UploadFile = File(..., description="a photo of one hand making one sign"),
    top_k: int = Query(3, ge=1, le=10),
    explain: bool = Query(False, description="also return an attention heatmap"),
) -> Prediction:
    if not STATE:
        raise HTTPException(503, "model not loaded")
    t0 = time.perf_counter()
    img = _decode(await image.read())

    x = STATE["tf"](img).unsqueeze(0).to(STATE["dev"])
    with torch.no_grad():
        logits = STATE["net"](x).float()
    probs = (logits / STATE["temp"]).softmax(1)[0]

    k = min(top_k, probs.numel())
    vals, idx = probs.topk(k)
    conf = float(vals[0])

    heat = None
    if explain:
        mask = M.attention_rollout(STATE["net"], x)
        heat = _heatmap_png(mask, img.resize((STATE["img_size"], STATE["img_size"])))

    return Prediction(
        label=STATE["classes"][int(idx[0])],
        confidence=round(conf, 4),
        abstain=conf < STATE["threshold"],
        top_k=[{"label": STATE["classes"][int(i)], "p": round(float(v), 4)}
               for v, i in zip(vals, idx)],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        attention_png_b64=heat,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="serve the recognition API")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args(argv)
    if a.checkpoint:
        os.environ["SLR_CHECKPOINT"] = a.checkpoint
    uvicorn.run(app, host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
