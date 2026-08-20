# backend-v2: signs from video, on top of handshapes from stills

Everything `backend/` did, plus isolated **sign** recognition from a clip.

The still-image backend classifies a photo of a handshape into 29 fingerspelling classes.
That is a limit of the input, not of the model: a still frame cannot carry motion, and most signs are motion.
This backend adds the video path and keeps the image path intact, so the two are measured against each other rather than beside each other.

## What changes, and why it matters more than "we added video"

The image project's contribution was the gap between a random split and an honest one.
Protocol B there had to **recover** capture sessions by clustering near-duplicates, because no static ASL corpus records who signed each frame.
`slr.signer_check` measures how badly that recovery does at the job people assume it does: at the usable threshold it fragments 211 of 217 real signers across components.
The README had to say so, and that limitation is why the headline metric was cross-corpus.

Video fixes this at the source.

PopSign ships a signer id for every one of its 21 Deaf signers.
AUTSL ships 43 signers and an official signer-independent benchmark split.
So the honest split is not reconstructed from pixels, it is read off the metadata:

| | protocol | what it measures |
| --- | --- | --- |
| **R** | random clip split | what most video SLR papers report |
| **S** | signer-disjoint | signers dealt whole. Exact, not recovered |
| **O** | official benchmark | the corpus's own published signer-independent split |

`R - S` is the same contribution as `A - C` on the image side.
The difference is that S is a measurement rather than an estimate, and `stage_S` asserts `shared_signers == 0` instead of trusting the split code.

## The corpora, and what is actually obtainable

| corpus | size | signers | media | signer split | how to get it |
| --- | --- | --- | --- | --- | --- |
| `popsign_islr` | ~100k clips, 250 signs | **21, labelled** | MediaPipe landmarks | exact | Kaggle competition, rules must be accepted once |
| `autsl` | 38,336 clips, 226 signs | 43, labelled | RGB + depth | **official**, baseline 62.02% | registration, manual download |
| `wlasl` | 21,083 clips, 2,000 glosses | none recorded | YouTube video | none | Kaggle mirror; see below |
| `lsa64` | 3,200 clips, 64 signs | 10, in the filename | RGB | derivable | manual download |

Two honest notes.

**WLASL is published as a list of YouTube links and most of them are dead.**
What you download is therefore not the corpus the papers benchmarked, so a number measured on it is not comparable to a published one.
It is registered as a train-side corpus and never scores a headline.
This is the same class of problem as a leaky split: a benchmark that quietly differs per downloader.

**`kind="manual"` sources refuse to download rather than pretending.**
AUTSL and LSA64 sit behind registration, so `sources.download` raises with the URL and instructions instead of failing obscurely three steps later.

## Landmarks, not pixels

The primary path is MediaPipe Holistic landmarks, and that is a direct answer to the gap left open in `backend/README`:

> No hand detection yet. Cropping to the hand with MediaPipe should be the single largest gain on cross-corpus transfer, because it deletes the background shortcut outright instead of augmenting around it.

Landmarks go further than cropping.
There is no wallpaper in a list of joint coordinates, so the shortcut the image work spent eighteen backbones measuring cannot be learned at all.
It is also what the data allows: PopSign *is* landmarks, so the same tensor layout serves a training sample and a phone upload.

`slr.landmarks` keeps 118 of the 543 holistic points.
The face mesh is 468 of those 543 and almost all of it is cheekbone and forehead that no sign depends on; keeping it would let the model spend capacity on face shape, which is signer identity wearing a disguise.
Lips survive, because mouthing separates signs that are manually identical.
Coordinates are centred on the shoulders and scaled by shoulder width, so camera distance and body size — signer identity by another route — are removed before the model sees anything.

## Ten-second uploads

**No isolated-sign corpus contains ten-second samples.**
WLASL averages 2.4s per clip, AUTSL 1.8s, PopSign 1.4s.
A ten-second upload holds either one very slow sign or several signs in a row, and a model trained on 1-2s clips has seen neither.

So a long upload is not collapsed into one sample.
`slr.clips` cuts it into overlapping windows the length of a training clip, scores every window, and returns the whole timeline alongside the best one.
`POST /predict_video` hands back `timeline`, `windows_scored` and a `note` saying plainly what was done, so a user who filmed ten seconds learns the model read a two-second slice of it.
On a clip that really is one sign, every window agrees and the timeline is flat, so the short case costs nothing.

## The model zoo, and its controls

Same selection discipline as the image side: ranked on **validation** macro-F1, and only the winner ever touches test.

| preset | family | what it is for |
| --- | --- | --- |
| `lm_mean` | pool | **CONTROL.** Order-blind by construction. If it ties the sequence models, these signs do not need motion |
| `lm_gru` | rnn | **CONTROL for attention.** 16 frames is short; recurrence may be enough |
| `lm_transformer` | transformer | the primary candidate |
| `lm_transformer_lg` | transformer | capacity check, to find where returns stop |
| `frame_mean` | pool + image backbone | **CONTROL.** Per-frame logits averaged, cannot see order |
| `frame_transformer` | transformer + image backbone | the bridge: do pixels carry what geometry threw away? |

`frame_*` reuses `slr.model`, so all eighteen image backbones are reachable from the video path.
That is what makes this a superset rather than a sibling.

`test_video.py` pins the control's defining property directly: `lm_mean` must score a clip and its reverse identically, and `lm_transformer` must not.
A control that quietly stopped being a control would make the whole comparison meaningless.

## Layout

```
backend-v2/
  src/slr/
    sources.py          one registry, image + video + landmark corpora
    leakage.py          perceptual hashing, group recovery, contamination audit
    data.py             image splits, group-disjoint and cross-corpus
    model.py            the 18-backbone image zoo
    train.py            image fine-tuning and architecture search
    evaluate.py         temperature scaling, ECE, risk-coverage
    experiment.py       image protocols A / B / C
    signer_check.py     scores group recovery against ground-truth signer ids
    clips.py            video decoding, frame sampling, the long-upload policy
    landmarks.py        MediaPipe Holistic, point selection, normalisation
    video_data.py       clip manifests, signer-disjoint splits, Datasets
    video_model.py      temporal zoo and its controls
    video_train.py      video fine-tuning and architecture search
    video_experiment.py video protocols R / S / O
    api.py              POST /predict (image) and POST /predict_video
  tests/                image suite plus test_video.py
  notebooks/            experiments.ipynb and video_experiments.ipynb
```

## Running it

Training happens on Colab.
Open `backend-v2/notebooks/video_experiments.ipynb`, set the runtime to T4, and work down the cells.

The test suite needs no dataset, no GPU, no network, and no MediaPipe:

```bash
PYTHONPATH=backend-v2/src pytest backend-v2/tests -q
```

MediaPipe and OpenCV are only needed to turn pixels into landmarks.
Training on PopSign, which ships landmarks, does not touch them, and neither do the tests.

Serving:

```bash
SLR_VIDEO_CHECKPOINT=runs/S_arch_WINNER_lm_transformer/best.pt \
  PYTHONPATH=backend-v2/src python -m slr.api --port 8000

curl -F video=@sign.mp4 'http://localhost:8000/predict_video?top_k=3'
```

```json
{
  "label": "thankyou",
  "confidence": 0.8140,
  "abstain": false,
  "top_k": [{"label": "thankyou", "p": 0.814}, {"label": "please", "p": 0.061}],
  "duration_s": 9.8,
  "windows_scored": 16,
  "timeline": [{"start": 0.0, "end": 2.0, "label": "hello", "p": 0.402}],
  "note": "clip is 9.8s but the corpora average 2.0s per sign; scoring 2s windows every 0.5s and reporting the best, plus the full timeline"
}
```

## Known gaps

No continuous sign language recognition.
The timeline shows *where* a confident window sits, but nothing segments a sentence into signs or models grammar.
Continuous SLR needs a corpus like PHOENIX-2014T or How2Sign and a CTC or transducer head, which is a different project.

No fusion of the two backends.
A fingerspelled name inside a signed sentence needs the image model and the video model to hand off to each other, and nothing here does that yet.

WLASL cannot support a headline number, for the link-rot reason above.
It is useful for pretraining and nothing else, and the registry says so.
