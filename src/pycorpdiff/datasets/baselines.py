"""Pre-computed reference-corpus frequency baselines.

A *baseline* is an aggregated term-frequency list — ``(term, count)``
pairs plus a ``total_tokens`` and provenance metadata — derived from a
fixed reference corpus. Keyness against a baseline (:func:`compare_to_baseline`
on the keyness side, exposed publicly as
:func:`pycorpdiff.against_baseline`) answers the question:

    "Which terms are over- or under-represented in my corpus
    compared to general-purpose / domain-typical language?"

This is the canonical use case for the BNC, COCA, or in-house
reference corpora. Shipping the reference *frequency list* rather than
the source text is the right unit of distribution: it's typically two
to three orders of magnitude smaller, it side-steps reference-corpus
licence complications, and the user never needs to (re-)tokenize
hundreds of megabytes at every analysis.

Bundled baselines
-----------------

- ``"gutenberg_fiction"`` — five Project Gutenberg public-domain English
  fiction texts (late-Georgian to Edwardian, 1813-1897), ~500K tokens,
  ~11K types. Useful as an out-of-the-box "general English fiction"
  reference; clearly inappropriate for any corpus where the contrast
  with 19th-century fiction is itself the signal (e.g. comparing two
  modern news archives).

For domain-specific work, prepare a custom baseline from a
:class:`pycorpdiff.Corpus` via :func:`baseline_from_corpus`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..corpus import Corpus, CorpusSlice

_DATA_DIR = Path(__file__).parent / "_data"

# Registry of bundled baselines.  Adding a new baseline = one row here
# plus the corresponding parquet + json in ``_data/``.  Keep this dict
# the single source of truth for ``list_baselines()``.
_BUNDLED: dict[str, tuple[str, str]] = {
    "gutenberg_fiction": (
        "baseline_gutenberg_fiction.parquet",
        "baseline_gutenberg_fiction.json",
    ),
}


@dataclass(frozen=True)
class Baseline:
    """A pre-computed reference-corpus frequency baseline.

    Attributes
    ----------
    counts
        Series indexed by term, valued by integer token count.
    total_tokens
        Sum of ``counts`` *plus* any tokens excluded during the build
        (e.g. hapax legomena dropped to compress the file). Always the
        true source-corpus total — this is what the keyness math wants
        as ``n_b``.
    n_documents
        Number of source documents the baseline was aggregated from.
        ``1`` is fine — it just means the baseline doesn't support
        document-level statistics like dispersion or bootstrap CIs.
    name
        Short identifier (``"gutenberg_fiction"``, ``"bnc_written"``, …).
    metadata
        Free-form provenance dict: description, license, source URL,
        per-source breakdown, build-time tokenization choices.
    """

    counts: pd.Series
    total_tokens: int
    n_documents: int
    name: str
    metadata: Mapping[str, Any]

    def __repr__(self) -> str:
        return (
            f"Baseline(name={self.name!r}, n_terms={len(self.counts):,}, "
            f"total_tokens={self.total_tokens:,}, n_documents={self.n_documents})"
        )


def list_baselines() -> list[str]:
    """Return the names of every bundled baseline."""
    return sorted(_BUNDLED)


def load_baseline(name: str) -> Baseline:
    """Load a bundled reference-corpus frequency baseline by name.

    Parameters
    ----------
    name
        One of the names returned by :func:`list_baselines`.

    Returns
    -------
    Baseline
        The frequency list + metadata, ready to pass to
        :func:`pycorpdiff.against_baseline`.

    Raises
    ------
    KeyError
        If ``name`` is not a bundled baseline.
    FileNotFoundError
        If the parquet/metadata files are missing (indicates an
        incomplete install — the data ships in the wheel).
    """
    if name not in _BUNDLED:
        raise KeyError(
            f"unknown baseline {name!r}; available baselines: "
            f"{list_baselines()}"
        )
    parquet_name, json_name = _BUNDLED[name]
    parquet_path = _DATA_DIR / parquet_name
    json_path = _DATA_DIR / json_name
    if not parquet_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"baseline data files missing under {_DATA_DIR}; the package "
            "may have been installed without bundled data. Re-run "
            "`python scripts/build_baselines/gutenberg.py` from the "
            "source checkout to regenerate."
        )
    df = pd.read_parquet(parquet_path)
    counts = pd.Series(df["count"].to_numpy(), index=df["term"].to_numpy(), name=name)
    counts = counts.astype("int64")
    metadata: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    return Baseline(
        counts=counts,
        total_tokens=int(metadata["total_tokens"]),
        n_documents=int(metadata["n_documents"]),
        name=name,
        metadata=metadata,
    )


def baseline_from_corpus(
    corpus: Corpus | CorpusSlice,
    *,
    name: str = "custom",
    min_count: int = 2,
    metadata: Mapping[str, Any] | None = None,
) -> Baseline:
    """Build a :class:`Baseline` from a user-supplied :class:`Corpus`.

    Use this to roll your own reference baseline — e.g. aggregate a
    domain-specific reference corpus (your own news archive, a slice of
    the BNC, a HuggingFace dataset) once, then keep comparing later
    analyses against it without re-tokenizing.

    Parameters
    ----------
    corpus
        Source corpus or slice. Tokens are aggregated across all
        documents into a single frequency list.
    name
        Identifier surfaced on the returned baseline. Default
        ``"custom"``.
    min_count
        Hapax-legomenon threshold. Terms with fewer than ``min_count``
        occurrences are dropped to suppress noise (names, OCR errors,
        rare typos). Set to 1 to keep every term.
    metadata
        Optional provenance dict merged into the returned baseline's
        ``.metadata``.

    Returns
    -------
    Baseline
    """
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1; got {min_count}")
    dtm = corpus.doc_term_counts(min_count=1)
    if dtm.shape[0] == 0:
        raise ValueError("baseline_from_corpus needs at least one document")
    counts = dtm.sum(axis=0).astype("int64")
    total_tokens = int(counts.sum())
    if min_count > 1:
        counts = counts[counts >= min_count]
    meta: dict[str, Any] = {
        "description": f"User-supplied baseline {name!r}",
        "min_count": int(min_count),
        "n_terms": int(len(counts)),
        "total_tokens": total_tokens,
        "n_documents": int(dtm.shape[0]),
    }
    if metadata:
        meta.update(metadata)
    return Baseline(
        counts=counts,
        total_tokens=total_tokens,
        n_documents=int(dtm.shape[0]),
        name=name,
        metadata=meta,
    )
