# ASL Sign Reader

A web app that reads ASL fingerspelling from a photo, one static hand shape at a time.
It knows 24 letters plus space, delete and nothing.
J and Z are left out, because they are drawn in the air and a photo cannot hold them.

- `apps/sign-reader` is the FastAPI app and web page.
- `packages/slr` is the library for datasets, training and evaluation.

## Run

```sh
uv sync --frozen --no-dev --all-packages
uv run --no-sync python -m sign_reader.main
```

Then open http://localhost:8000.

The model weights are not in git, because they are too big for GitHub.
Put `deit3_base_fused.pt` (or `deit3_base.pt`) in `apps/sign-reader/weights/` before you start the app.
Or set `SLR_WEIGHTS_URL` to download it on first start.

Without weights the page still loads, and `/api/health` says what is missing.
The hand detector downloads itself on first use.

## Train the photo model

```sh
uv sync --frozen --all-packages
uv run --no-sync python -m slr.sources download asl_alphabet asl_alphabet_test
uv run --no-sync python -m slr.sources manifest
uv run --no-sync python -m slr.leakage groups dataset/manifests/all.csv
uv run --no-sync python -m slr.data dataset/manifests/all.csv --policy cross_source
uv run --no-sync python -m sign_reader.train dataset/manifests/all.csv
```

This needs a GPU and Kaggle credentials in `~/.kaggle/kaggle.json`.

## Accuracy

Static letters: 91.11% on 720 unique held-out photos.
N is still weak (3/30).
