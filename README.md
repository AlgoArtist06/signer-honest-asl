# Signer-honest sign language recognition

A vision-transformer backend that classifies a still photo of an ASL handshape into
one of 29 classes (A-Z plus `space`, `del`, `nothing`), served over HTTP so a React
Native app can post a camera frame and get a letter back.

The model is ordinary.
What is not ordinary is the evaluation, and that is the point of the project.

## The problem this project is actually about

Published static-ASL results cluster around 99.9%.
A widely cited 2023 Vision Transformer paper reports 99.98% on the ASL alphabet.
Those numbers do not survive contact with a phone camera, and the reason is not the architecture.

The standard corpora are continuous webcam captures.
`asl_alphabet`, the most-used Kaggle set, is 87,000 frames of essentially one person in one room.
Split those frames at random and frame 1041 lands in train while frame 1042 lands in test.
The two images differ by a few pixels of hand tremor.
A classifier that memorises the wallpaper scores a perfect mark.

Sign Language MNIST is worse: its official test split is produced by augmenting the same base photographs as its train split.
It is leaky by construction, and it is still used as a benchmark.

So this repository reports three numbers for the same model on the same data:

| protocol | what it measures | what to expect |
| --- | --- | --- |
| **A** random split | what everyone else publishes | very high |
| **B** group split | sessions kept whole | substantially lower |
| **C** cross-corpus | trained on one corpus, tested on another shot by different people | the honest number |

The gap between A and C is the contribution.
Everything in `backend/` exists to make that gap measurable rather than arguable.

## What is here that other projects do not have

**A leakage audit that produces a number.**
`slr.leakage.audit` hashes every image and reports what fraction of the test split has a near-duplicate sitting in train.
Run it on any split, including someone else's, and the honesty of that split stops being a matter of opinion.
Banded hashing with a pigeonhole guarantee makes it exact and fast enough for 90k images.

**Group recovery for corpora that ship no signer labels, and an honest account of its limits.**
Almost no static ASL corpus records who signed each image, which is why almost nobody splits on signer.
Two things help, and it matters which is which.

For `Sign-Language-Digits-Dataset`, signer identity is recoverable exactly.
218 students were photographed back to back on one camera, ten shots each, digits 0-9 in order, so the camera's own frame counter encodes the student: `signer = (IMG_number - 1118) // 10`.
Verified on the real download: 2062 images resolve into 217 signers, 158 of whom hold all ten digits, with 5 showing a repeated digit where the photographer retook a shot and slipped the counter.
The dataset has been public since 2017 and is not distributed with signer labels.

For everything else, `recover_groups` clusters near-duplicates into connected components.
That recovers **capture sessions**, not signers, and the distinction is the difference between a real claim and a marketing one.
Scored against the digits corpus's ground-truth signer ids:

| max_dist | components | signers fragmented across components |
| --- | --- | --- |
| 2 | 2021 | 217 / 217 |
| 6 | 953 | 211 / 217 |
| 10 | 141 | 88 / 217 |
| 12 | 48 | 37 / 217 |

Every signer fragments at any usable threshold, and raising the threshold until they stop simply collapses the corpus into a few blobs.
Where a corpus is a continuous webcam capture, sessions are the entire leak and this fixes it.
Where each person contributes one shot per class, there is nothing linking their images and the method is blind to them.

That limitation is exactly why the headline metric below is cross-corpus rather than an in-corpus group split.

**Cross-corpus evaluation as the headline metric.**
`asl_alphabet_test` was shot by a different person against varied backgrounds specifically to validate models trained on `asl_alphabet`.
Training on one and testing on the other is a protocol the data supports and almost no write-up uses.
It also sidesteps the recovery limitation above entirely: two corpora shot by different people are signer-disjoint by construction, with no need to work out who anyone is.

**Calibration and abstention, because the app needs them.**
A confident wrong letter is worse than "I'm not sure".
`slr.evaluate` fits a temperature on validation, reports expected calibration error before and after, and produces a risk-coverage table: at 90% coverage the error rate is X, at 80% it is Y.
The API reads its abstention threshold from that table, so the threshold it serves is one whose error rate was actually measured.

**An architecture search that cannot cheat.**
Eighteen backbones are registered: ViT at four scales, plus DeiT3, Swin, ConvNeXt, ConvNeXtV2, BEiTv2, CAFormer, EVA-02, SigLIP, CLIP, DINOv2, and three mobile models.
Candidates are ranked on **validation** macro-F1 and only the winner is ever run against the test split.
Ranking architectures by their test score would leak the test set into the experiment through the selection step, which is the same mistake as a random split wearing a different hat.
Two plain CNNs are in the search as controls rather than filler: if `convnext_base` ties the transformers on the cross-corpus test, then attention was not what carried this task, and the write-up has to say so.

**Attention rollout as evidence.**
`model.attention_rollout` implements Abnar & Zuidema, and `POST /predict?explain=true` returns the heatmap overlaid on the photo.
It is a feature for the app and a diagnostic for us: a model attending to the wall behind the signer is visible immediately.

## Two backends

`backend/` classifies a **still photo** of a handshape into 29 fingerspelling classes.

`backend-v2/` is a superset: everything above, plus isolated **sign** recognition from a video clip.
It exists because the still backend's honest split had to *recover* capture sessions, and `signer_check` shows that recovery fragments 211 of 217 real signers.
The video corpora ship real signer ids, so there the signer-disjoint split is exact rather than approximated.
See `backend-v2/README.md`.

## Layout

```
backend/
  src/slr/
    sources.py    corpus registry, downloaders, label normalisation, group-key rules
    leakage.py    perceptual hashing, group recovery, the train/test contamination audit
    data.py       group-disjoint and cross-corpus splits, augmentation, torch Dataset
    model.py      18-backbone zoo (5.5M-303M params), layer-wise LR decay, attention rollout
    train.py      fine-tuning loop, and the select-on-validation architecture search
    evaluate.py   temperature scaling, ECE, risk-coverage, per-class confusion
    api.py        FastAPI service the React Native app calls
  tests/          19 tests, no dataset or GPU needed, ~20 seconds
  notebooks/      experiments.ipynb - the full pipeline end to end on a Colab GPU
dataset/
  raw/            downloaded corpora (gitignored)
  manifests/      CSV: path, label, source, group, split (gitignored)
docs/
  BIBLIOGRAPHY.md 26 papers, auto-generated, each with a line saying which decision it supports
  pdf/            the PDFs themselves (gitignored)
  fetch_papers.py regenerates both
```

## Running it

Training happens on Colab.
Open `backend/notebooks/experiments.ipynb`, set the runtime to GPU, and work down the cells.
It downloads the corpora, recovers groups, runs protocols A, B and C, sweeps model scale, searches architectures, calibrates, and hands back a checkpoint.

The search is driven from the command line too:

```bash
python -m slr.train dataset/manifests/cross_corpus.csv --preset quick   # 4 models, one session
python -m slr.train dataset/manifests/cross_corpus.csv --preset scale   # ViT 5.5M -> 303M
python -m slr.train dataset/manifests/cross_corpus.csv --preset arch    # all 11 base-scale backbones
python -m slr.train dataset/manifests/cross_corpus.csv --preset mobile  # on-device candidates
```

Each writes a leaderboard with validation score, parameter count, and seconds per epoch, names the winner, and evaluates that one on test.

The leaderboard is rewritten after every candidate and re-read on restart, so a sweep that outlives its session resumes instead of starting over.
`--preset all` runs the whole zoo; on a single T4 that is several sessions' work, and losing one costs the model in flight rather than the sweep.

Locally, for inference and tests:

```bash
uv venv --python 3.12 ~/.venvs/slr
uv pip install --python ~/.venvs/slr/bin/python \
    torch torchvision timm numpy pillow pytest fastapi 'uvicorn[standard]' python-multipart

PYTHONPATH=backend/src ~/.venvs/slr/bin/python -m pytest backend/tests -q
```

The venv lives outside the project on purpose; anywhere works.
The test suite needs no dataset, no network, and no GPU, and it builds all 18 backbones to catch a bad checkpoint name before a sweep does.
One test additionally scores the signer rule against the real digits corpus, and skips if you have not downloaded it.

Serving a trained checkpoint:

```bash
SLR_CHECKPOINT=runs/C_quick_WINNER_vit_small/best.pt \
  PYTHONPATH=backend/src ~/.venvs/slr/bin/python -m slr.api --port 8000
```

`GET /health` reports the loaded preset, the fitted temperature, and the abstention threshold.
`POST /predict` takes `multipart/form-data` with an `image` field:

```bash
curl -F image=@hand.jpg 'http://localhost:8000/predict?top_k=3'
```

```json
{
  "label": "B",
  "confidence": 0.9134,
  "abstain": false,
  "top_k": [{"label": "B", "p": 0.9134}, {"label": "F", "p": 0.0421}],
  "latency_ms": 61.3
}
```

Add `&explain=true` for a base64 PNG of the attention heatmap.

The React Native side posts the camera photo as multipart and renders `label` when `abstain` is false, and a "hold steady" prompt when it is true.

## Reading list

`python docs/fetch_papers.py` downloads all 26 PDFs and regenerates `docs/BIBLIOGRAPHY.md`.
The list lives in the script, so the bibliography cannot drift from what is on disk.
Each entry says which design decision in `backend/` it supports, which is the only reason a paper is on the list.

## Known gaps

No hand detection yet.
Cropping to the hand with MediaPipe should be the single largest gain on cross-corpus transfer, because it deletes the background shortcut outright instead of augmenting around it.
The uncropped cross-corpus baseline goes on the board first so there is something to compare against.
Marked in `data.py`.

No horizontal flip augmentation, deliberately.
ASL is handed and mirroring changes what some signs mean.

The 29-class vocabulary covers fingerspelling only, not signed words.
A still photo cannot carry the motion that most signs need, so this is a limit of the input, not of the model.
