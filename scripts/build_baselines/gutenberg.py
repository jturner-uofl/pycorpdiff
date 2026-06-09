"""Build the bundled Gutenberg-fiction baseline frequency list.

This script is the source-of-truth for
``src/pycorpdiff/datasets/_data/baseline_gutenberg_fiction.parquet``.
It is **not** run at install or import time — the parquet is committed
and shipped with the package. The script lives here so reviewers can
verify the baseline is reproducible and so we can regenerate it when
the source set, the tokenizer, or the filtering rules change.

Run with::

    python -m scripts.build_baselines.gutenberg
    # or, from the repo root:
    python scripts/build_baselines/gutenberg.py

Outputs:
    src/pycorpdiff/datasets/_data/baseline_gutenberg_fiction.parquet
    src/pycorpdiff/datasets/_data/baseline_gutenberg_fiction.json

Why these books
---------------
Five public-domain (Project Gutenberg) English fiction texts spanning
late-Georgian to Edwardian British fiction. Genre is intentionally
narrow so the keyness signal a user sees against this baseline is
interpretable — "what makes my corpus differ from 19th-century British
fiction?" is a well-defined question. The combined token count
(~520K) gives a usable expected-frequency estimate across a vocabulary
of ~30K types.

This is a *starter* baseline. Users with domain-specific reference
corpora (e.g. the BNC, COCA, news-archive crawls) should build their
own via :func:`pycorpdiff.datasets.baselines.baseline_from_corpus`.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Source set
# ----------------------------------------------------------------------

# (gutenberg_id, short_title, author, year).  Plain-text URLs follow the
# canonical https://www.gutenberg.org/cache/epub/<ID>/pg<ID>.txt pattern.
SOURCES: list[tuple[int, str, str, int]] = [
    (1342, "Pride and Prejudice", "Austen, Jane", 1813),
    (11,   "Alice's Adventures in Wonderland", "Carroll, Lewis", 1865),
    (1661, "The Adventures of Sherlock Holmes", "Doyle, Arthur Conan", 1892),
    (84,   "Frankenstein", "Shelley, Mary", 1818),
    (345,  "Dracula", "Stoker, Bram", 1897),
]

# Gutenberg wraps each text with a START/END marker pair we strip out
# before tokenizing so the boilerplate (licence text, transcriber notes)
# doesn't leak into the frequency list.
GUTENBERG_START = re.compile(
    r"\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE
)
GUTENBERG_END = re.compile(
    r"\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE
)

URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
USER_AGENT = "pycorpdiff-baseline-builder/0.1 (jason.s.turner@gmail.com)"


def fetch_text(book_id: int) -> str:
    """Download one Gutenberg book as plain UTF-8 text."""
    url = URL_TEMPLATE.format(book_id=book_id)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    # Gutenberg files are predominantly UTF-8 these days; fall back to
    # latin-1 if a stray byte ever sneaks in.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def strip_boilerplate(text: str) -> str:
    """Drop the Project Gutenberg header / footer wrappers."""
    start = GUTENBERG_START.search(text)
    end = GUTENBERG_END.search(text)
    if start is not None:
        text = text[start.end() :]
    if end is not None:
        text = text[: end.start()]
    return text


def build() -> None:
    # Late import so this script remains runnable from a checkout even
    # before the package is fully installed.
    from pycorpdiff.tokenize import RegexTokenizer

    tokenizer = RegexTokenizer()  # default: \w+, lowercase, NFC

    counter: Counter[str] = Counter()
    total_tokens = 0
    book_summaries: list[dict[str, object]] = []

    for book_id, title, author, year in SOURCES:
        print(f"  fetching #{book_id}: {title}")
        raw = fetch_text(book_id)
        body = strip_boilerplate(raw)
        tokens = tokenizer(body)
        counter.update(tokens)
        total_tokens += len(tokens)
        book_summaries.append(
            {
                "gutenberg_id": book_id,
                "title": title,
                "author": author,
                "year": year,
                "tokens": len(tokens),
            }
        )
        print(f"    {len(tokens):,} tokens, {len(counter):,} types so far")

    # Drop hapax legomena.  These are dominated by names, OCR artefacts,
    # and language-specific noise — keeping them would inflate the
    # baseline vocabulary without contributing to robust expected
    # frequencies.  Two-occurrence threshold matches Rayson's LL Wizard
    # documentation and roughly halves the on-disk size.
    rows = sorted(
        ((term, count) for term, count in counter.items() if count >= 2),
        key=lambda row: (-row[1], row[0]),
    )
    df = pd.DataFrame(rows, columns=["term", "count"])
    df["count"] = df["count"].astype("int64")

    # Write parquet (compressed) + JSON metadata sidecar.
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "pycorpdiff"
        / "datasets"
        / "_data"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "baseline_gutenberg_fiction.parquet"
    json_path = out_dir / "baseline_gutenberg_fiction.json"

    df.to_parquet(parquet_path, index=False, compression="snappy")

    metadata = {
        "name": "gutenberg_fiction",
        "description": (
            "Aggregated term frequencies from five public-domain Project "
            "Gutenberg English fiction texts (late-Georgian to Edwardian, "
            "1813-1897). Tokenization: pycorpdiff.RegexTokenizer defaults "
            "(\\w+, lowercase, NFC). Hapax legomena (count < 2) removed."
        ),
        "license": "Public Domain (Project Gutenberg)",
        "source_url": "https://www.gutenberg.org/",
        "n_documents": len(SOURCES),
        "total_tokens": int(total_tokens),
        "n_terms": int(len(df)),
        "min_count": 2,
        "books": book_summaries,
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"wrote {parquet_path}  ({parquet_path.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {json_path}  ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"total tokens: {total_tokens:,}  vocab: {len(df):,}")


if __name__ == "__main__":
    build()
