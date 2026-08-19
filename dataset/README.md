# dataset/

Nothing here is committed.
`raw/` and `manifests/` are gitignored, because corpora are gigabytes and licensed by their original authors.

## Getting the data

```bash
PYTHONPATH=backend/src python -m slr.sources list                 # what is registered
PYTHONPATH=backend/src python -m slr.sources download asl_alphabet
PYTHONPATH=backend/src python -m slr.sources manifest all
```

Kaggle sources need an API token at `~/.kaggle/kaggle.json` (chmod 600).
Get it from kaggle.com -> Settings -> API -> Create New Token.
Kaggle returns a bare 404 for unauthenticated downloads, so a missing token looks like a missing dataset.
HuggingFace and GitHub sources need no credentials.

The registry, including each corpus's licence origin and its group-key rule, is `backend/src/slr/sources.py`.

## What the manifest is

One CSV, one row per image:

| column | meaning |
| --- | --- |
| `path` | absolute path on disk |
| `label` | one of the 29 classes (A-Z, `space`, `del`, `nothing`) |
| `label_idx` | index into `sources.CLASSES` |
| `source` | which corpus it came from |
| `group` | the unit that must never straddle a split |
| `split` | `train` / `val` / `test`, assigned by `slr.data` |

`group` is the column that matters.
It holds a signer id where the corpus provides one, a recovered session id where it does not, and the whole corpus as a single group where neither is available, which forces that corpus to be used whole rather than split.

Splits are stored in the manifest rather than as separate directories, so a split can be recomputed without re-walking or copying hundreds of thousands of files, and so `slr.leakage.audit` can check any split anyone proposes.

## Space

`asl_alphabet` alone is ~1.1 GB compressed and ~2 GB extracted.
The full registry is well over 3 GB.
Download on Colab, not on a laptop.
