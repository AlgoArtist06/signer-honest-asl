#!/usr/bin/env python3
"""Fetch the reading list as PDFs and regenerate BIBLIOGRAPHY.md.

The list is code, not prose, so the bibliography can never drift from what is
actually on disk. Run `python docs/fetch_papers.py` to populate `docs/pdf/`.

Grouped by what each paper is here to answer, not by year, because the point of
this folder is to justify design decisions in `backend/`.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF_DIR = HERE / "pdf"
ARXIV_API = "http://export.arxiv.org/api/query?"


@dataclass(frozen=True)
class Paper:
    arxiv: str
    slug: str
    why: str


SECTIONS: dict[str, tuple[str, list[Paper]]] = {
    "leakage": (
        "Evaluation honesty: signer independence, splits, and shortcut learning.\n"
        "These are the papers that justify why this project refuses a random split.",
        [
            Paper("1608.08339", "asl-fingerspelling-signer-independence",
                  "The reference statement of the signer-independence problem. Reports "
                  "letter error rates near 60% once the test signer is unseen, against "
                  "near-perfect numbers when they are not. The gap this project measures."),
            Paper("1602.04278", "signer-independent-dnn-adaptation",
                  "Signer adaptation as an explicit step, which only makes sense once you "
                  "accept that signer identity is a confound rather than noise."),
            Paper("2004.07780", "shortcut-learning-in-dnns",
                  "Geirhos et al. The general theory of why a model scores 99.9% on a "
                  "benchmark and fails on a phone photo: it solved an easier problem that "
                  "the benchmark accidentally rewarded."),
            Paper("1811.12231", "imagenet-cnns-texture-biased",
                  "Concrete evidence that vision models latch onto texture and background "
                  "statistics. Motivates the aggressive colour/grayscale augmentation in "
                  "`data.build_transforms` and the planned hand-crop step."),
            Paper("2308.12419", "asl-processing-in-the-real-world",
                  "Shi et al. on the gap between curated ASL corpora and deployment, and "
                  "the data/task design needed to close it."),
        ],
    ),
    "datasets": (
        "The corpora themselves: what exists, how it was collected, and who signed it.",
        [
            Paper("2203.03859", "27-class-173-individuals",
                  "Mavi & Dikle. The static ASL corpus with genuine signer diversity "
                  "(173 volunteers), and the reason `asl_27class` is in the registry."),
            Paper("2407.15806", "fsboard",
                  "3M+ characters of ASL fingerspelling captured on smartphones by 147 "
                  "signers. The closest public data to this project's actual deployment "
                  "condition: a phone camera held by the signer."),
            Paper("2606.19352", "sign-language-datasets-survey",
                  "Survey of sign-language resources, benchmarks, and annotation "
                  "standards. Use it to check a corpus before adding it to the registry."),
            Paper("2506.03615", "isharah",
                  "Large multi-scene continuous corpus. Multi-scene is the property that "
                  "makes cross-condition evaluation possible at all."),
            Paper("2008.00932", "autsl",
                  "AUTSL, with an explicit signer-independent benchmark protocol worth "
                  "copying."),
            Paper("2007.12131", "bsl-1k",
                  "Scaling isolated sign recognition via mouthing cues, and a careful "
                  "discussion of how episode-level splits avoid leakage."),
            Paper("2608.10588", "hamnosys-handshape-recognition",
                  "Fine-grained isolated handshape recognition. Handshape is exactly what "
                  "a still photo can carry, so this is the closest task to ours."),
        ],
    ),
    "transformers": (
        "Vision transformers: the backbone, how to fine-tune it, and how to read it.",
        [
            Paper("2010.11929", "vit-an-image-is-worth-16x16-words",
                  "Dosovitskiy et al. The architecture. Note the data-hunger finding, "
                  "which is why every preset here starts from ImageNet-21k weights."),
            Paper("2106.10270", "how-to-train-your-vit",
                  "Steiner et al. The augreg study behind the exact checkpoints in "
                  "`model.PRESETS`, and the source of the augmentation-vs-data tradeoffs "
                  "that matter on corpora this small."),
            Paper("2012.12877", "deit-data-efficient-transformers",
                  "Training ViTs without hundreds of millions of images. The recipe this "
                  "project's training loop is a stripped-down version of."),
            Paper("2304.07193", "dinov2",
                  "Self-supervised features that transfer without task-specific "
                  "fine-tuning. The `dinov2_small` preset exists to test whether "
                  "self-supervised pretraining survives the cross-corpus jump better."),
            Paper("2005.00928", "quantifying-attention-flow",
                  "Abnar & Zuidema. Attention rollout, implemented in "
                  "`model.attention_rollout`. Raw per-layer attention is misleading "
                  "because of the residual stream; rollout corrects for it."),
        ],
    ),
    "slr_methods": (
        "Applied sign-language recognition with attention architectures. The direct\n"
        "comparison points, several of which report the inflated numbers this project\n"
        "is designed to contextualise.",
        [
            Paper("2509.03467", "saudi-slr-vision-transformer",
                  "Continuous Saudi sign language with a ViT. Recent, and a useful check "
                  "on how the field currently reports splits."),
            Paper("2504.07792", "video-vit-word-level-slr",
                  "Video ViTs for word-level recognition; the temporal counterpart to "
                  "what we do on stills."),
            Paper("2503.16855", "stack-transformer-spatiotemporal",
                  "Stacked spatial-temporal attention for dynamic signs and "
                  "fingerspelling."),
            Paper("2505.10267", "handreader",
                  "Efficient fingerspelling recognition; relevant to the on-device "
                  "latency budget."),
            Paper("2311.12128", "fingerspelling-posenet",
                  "Pose-based transformer. The main architectural alternative to raw "
                  "pixels, and the reason hand cropping is the planned next step."),
            Paper("2105.07625", "fine-grained-visual-attention-fingerspelling",
                  "Attention over the hand region in the wild - the published version of "
                  "the crop-the-hand idea marked as future work in `data.py`."),
            Paper("2511.13126", "recurrent-vs-attention-isolated-slr",
                  "Direct architectural comparison on isolated SLR; a sanity check on "
                  "whether attention is worth its cost here."),
        ],
    ),
    "deployment": (
        "Calibration and abstention: turning a classifier into something an app can\n"
        "responsibly show a user.",
        [
            Paper("1706.04599", "on-calibration-of-modern-nns",
                  "Guo et al. Modern networks are overconfident; temperature scaling "
                  "fixes it with one parameter. Implemented in `evaluate.fit_temperature`."),
            Paper("1705.08500", "selective-classification-for-dnns",
                  "Geifman & El-Yaniv. The risk-coverage framing behind the abstention "
                  "threshold the API serves."),
        ],
    ),
}

ALL = [p for _, ps in SECTIONS.values() for p in ps]


def fetch_meta(ids: list[str]) -> dict[str, dict]:
    """One batched arXiv API call. Returns {id: {title, authors, published, summary}}."""
    q = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    with urllib.request.urlopen(ARXIV_API + q, timeout=60) as r:
        root = ET.fromstring(r.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: dict[str, dict] = {}
    for e in root.findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/abs/", 1)[-1]
        out[aid.split("v")[0]] = {
            "title": " ".join(e.find("a:title", ns).text.split()),
            "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
            "published": e.find("a:published", ns).text[:10],
            "version_id": aid,
        }
    return out


def download_pdf(paper: Paper, meta: dict, dest: Path) -> bool:
    out = dest / f"{paper.slug}.{paper.arxiv}.pdf"
    if out.exists() and out.stat().st_size > 20_000:
        return True
    url = f"https://arxiv.org/pdf/{meta.get('version_id', paper.arxiv)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "slr-docs/0.1"})
        with urllib.request.urlopen(req, timeout=180) as r:
            blob = r.read()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"[fail] {paper.arxiv}: {e}")
        return False
    if not blob.startswith(b"%PDF"):
        print(f"[fail] {paper.arxiv}: response was not a PDF")
        return False
    out.write_bytes(blob)
    print(f"[ok] {out.name} ({len(blob) / 1e6:.1f} MB)")
    return True


def write_bibliography(meta: dict[str, dict], have: set[str]) -> None:
    lines = [
        "# Bibliography",
        "",
        "<!-- AUTO-GENERATED by docs/fetch_papers.py. Edit the list in that file, not here. -->",
        "",
        "Every entry below is downloadable with `python docs/fetch_papers.py`.",
        "PDFs land in `docs/pdf/` and are gitignored, so the repository stays small.",
        "",
        "Each `Why` line states what decision in `backend/` the paper supports.",
        "That is the only reason a paper is in this list.",
        "",
    ]
    for key, (blurb, papers) in SECTIONS.items():
        lines += [f"## {key.replace('_', ' ').title()}", "", blurb, ""]
        for p in papers:
            m = meta.get(p.arxiv, {})
            title = m.get("title", f"(metadata unavailable for {p.arxiv})")
            authors = m.get("authors", [])
            who = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            mark = "" if p.arxiv in have else "  `[pdf not downloaded]`"
            lines += [
                f"### {title}",
                "",
                f"{who} ({m.get('published', 'n.d.')}). "
                f"[arXiv:{p.arxiv}](https://arxiv.org/abs/{p.arxiv}).{mark}",
                "",
                f"**Why:** {p.why}",
                "",
            ]
    lines += [
        "## Not on arXiv",
        "",
        "These are worth reading but are behind publisher paywalls, so only the link is recorded.",
        "",
        "- [HGR-ViT: Hand Gesture Recognition with Vision Transformer]"
        "(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10303839/) (Sensors, 2023).",
        "  Reports 99.98% on the ASL alphabet. Read it as the clearest example of what a",
        "  random split buys you, and compare against the cross-corpus numbers this",
        "  project produces on the same data.",
        "- [FluentSigners-50: A signer independent benchmark dataset]"
        "(https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0273649) (PLOS ONE, 2022).",
        "  A benchmark built signer-independent from the start; the protocol worth imitating.",
        "- [SignViT: an enhanced vision transformer framework]"
        "(https://www.sciencedirect.com/science/article/abs/pii/S1746809425011139)",
        "  (Biomedical Signal Processing and Control, 2025).",
        "- [Investigating Signer-Independent Sign Language Recognition on LSA64]"
        "(https://www.researchgate.net/publication/363174384) (2022).",
        "",
    ]
    (HERE / "BIBLIOGRAPHY.md").write_text("\n".join(lines))
    print(f"[ok] BIBLIOGRAPHY.md ({len(ALL)} entries, {len(have)} PDFs present)")


def main() -> int:
    ap = argparse.ArgumentParser(description="fetch the reading list")
    ap.add_argument("--no-pdf", action="store_true", help="regenerate the bibliography only")
    a = ap.parse_args()

    ids = [p.arxiv for p in ALL]
    meta: dict[str, dict] = {}
    for i in range(0, len(ids), 25):
        meta |= fetch_meta(ids[i : i + 25])
        time.sleep(3)  # arXiv asks for one request per three seconds
    missing = [i for i in ids if i not in meta]
    if missing:
        print(f"[warn] no arXiv metadata for: {missing}")

    have: set[str] = set()
    if not a.no_pdf:
        PDF_DIR.mkdir(exist_ok=True)
        for p in ALL:
            if download_pdf(p, meta.get(p.arxiv, {}), PDF_DIR):
                have.add(p.arxiv)
            time.sleep(1)
    else:
        have = {p.arxiv for p in ALL
                if any(PDF_DIR.glob(f"{p.slug}.{p.arxiv}.pdf"))}

    (HERE / "papers.json").write_text(json.dumps(
        {p.arxiv: {"slug": p.slug, "why": p.why, **meta.get(p.arxiv, {})} for p in ALL},
        indent=2))
    write_bibliography(meta, have)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
