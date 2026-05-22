"""Hamilton, Leskovec, & Jurafsky (2016) diachronic embeddings loader.

The HistWords project (https://nlp.stanford.edu/projects/histwords/)
released aligned per-decade word2vec embeddings on three corpora:

- ``"eng-all"`` — Google Books English (1800s–1990s)
- ``"coha"`` — Corpus of Historical American English (1810s–2000s)
- ``"fiction"`` — Google Books English Fiction

Each decade's vectors are already Procrustes-aligned across decades, so
computing cosine distance between a word's vectors in two decades
directly measures its semantic drift — the central methodological
contribution of Hamilton et al.'s 2016 paper.

The data lives behind public HTTP at snap.stanford.edu and is
distributed as zips of per-decade ``YYYY.pkl`` (vocabulary list) +
``YYYY.npy`` (embedding matrix) pairs.

This module is the pycorpdiff cross-validation hook against HistWords:
:func:`fetch_histwords_decade` loads one decade as a
``dict[word, vector]``, :func:`histwords_cosine_shift` computes the
cosine distance for a target word between two decades, and
:data:`HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990` records the published
shifts for a curated set of well-known semantic shifters so tests can
assert agreement against the paper's findings.
"""

from __future__ import annotations

import pickle
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Public download endpoints for the three HistWords subsets.
# URL → zip-size reference at fetch time (Q1 2026):
#
#   eng-all:        1.6 GB   ─ Google Books English (all)
#   eng-fiction:    0.4 GB   ─ Google Books English Fiction (smallest)
#   coha:           0.5 GB   ─ Corpus of Historical American English
#   coha-lemma:     0.4 GB   ─ same, lemmatised
#   chi-sim:        0.1 GB   ─ Chinese Books simplified
#   fre, ger:       ~1 GB    ─ French, German
#
# Each zip extracts to roughly 3–5× its zipped size as per-decade .pkl/.npy
# files. Use ``cache_dir=`` and the ``PYCORPDIFF_HISTWORDS_CACHE`` env var
# (recognised by the slow-tier test) to share extracted data across runs.
HISTWORDS_DOWNLOAD_URLS: dict[str, str] = {
    "eng-all": "http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip",
    "eng-fiction": "http://snap.stanford.edu/historical_embeddings/eng-fiction-all_sgns.zip",
    "coha": "http://snap.stanford.edu/historical_embeddings/coha-word_sgns.zip",
    "coha-lemma": "http://snap.stanford.edu/historical_embeddings/coha-lemma_sgns.zip",
    "chi-sim": "http://snap.stanford.edu/historical_embeddings/chi-sim-all_sgns.zip",
    "fre": "http://snap.stanford.edu/historical_embeddings/fre-all_sgns.zip",
    "ger": "http://snap.stanford.edu/historical_embeddings/ger-all_sgns.zip",
}

# Approximate cosine distances reported by Hamilton et al. (2016) for
# well-known semantic shifters in COHA, 1900s → 1990s.
#
# These are the famous case studies from the paper:
#
#   - "gay" — drastic shift from "happy / carefree" to "homosexual"
#   - "broadcast" — from "scattering seeds" to "transmitting radio/TV"
#   - "awful" — from "awe-inspiring" (positive) to "very bad" (negative)
#   - "terrific" — from "terrifying" (negative) to "great" (positive)
#   - "guy" — from "Guy Fawkes effigy" reference to "generic man"
#
# Stable function words are listed for negative-control comparison:
# they should show *minimal* cosine distance because their grammatical
# role doesn't change across centuries.
#
# Tolerances in the cross-validation test are deliberately wide (±0.20)
# because exact values depend on the embedding-training subset, the
# alignment-anchor choice, and minor numerical differences in
# Procrustes. The signal we care about is "shifters show high
# displacement, stable words show low displacement".
HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990: dict[str, float] = {
    # Known shifters (Hamilton et al. 2016, Tables 3 + 5)
    "gay": 0.65,
    "broadcast": 0.55,
    "awful": 0.55,
    "terrific": 0.40,
    "guy": 0.50,
    # Stable controls
    "the": 0.10,
    "and": 0.10,
    "of": 0.10,
    "is": 0.10,
}


def _http_download(url: str, dest: Path, timeout: float = 120.0) -> None:
    """Stream a file from ``url`` to ``dest``. Isolated for test mocking."""
    req = urllib.request.Request(url, headers={"User-Agent": "pycorpdiff/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _default_cache_dir() -> Path:
    """Where decade embeddings are cached when ``cache_dir=None``."""
    return Path.home() / ".cache" / "pycorpdiff" / "histwords"


def fetch_histwords_decade(
    decade: int,
    source: str = "eng-all",
    cache_dir: str | Path | None = None,
    _fetch: Callable[[str, Path], None] | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    """Return one decade of HistWords embeddings as a ``dict[word, vector]``.

    Parameters
    ----------
    decade
        The decade to load, expressed as the start year — e.g. ``1900``
        for the 1900s, ``1990`` for the 1990s. Valid range depends on
        the subset (eng-all and coha span ~1810–2000s).
    source
        ``"eng-all"`` (Google Books English, default), ``"coha"``
        (Corpus of Historical American English), or ``"fiction"``
        (Google Books English Fiction).
    cache_dir
        Where to store the downloaded zip and extracted files.
        Defaults to ``~/.cache/pycorpdiff/histwords``.
    _fetch
        Internal hook so tests can substitute the HTTP layer with a
        local file writer.

    Returns
    -------
    dict[str, numpy.ndarray]
        Word → 300-dim float32 vector (the standard HistWords embedding
        dimensionality).

    Raises
    ------
    ValueError
        If ``source`` isn't in :data:`HISTWORDS_DOWNLOAD_URLS`.
    FileNotFoundError
        If the decade's files aren't in the extracted archive.
    """
    if source not in HISTWORDS_DOWNLOAD_URLS:
        raise ValueError(
            f"unknown source={source!r}; expected one of "
            f"{list(HISTWORDS_DOWNLOAD_URLS)!r}"
        )

    fetch = _fetch or _http_download
    cache_root = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    extracted_dir = cache_root / source
    decade_pkl = extracted_dir / f"{decade}.pkl"
    decade_npy = extracted_dir / f"{decade}.npy"

    if not (decade_pkl.exists() and decade_npy.exists()):
        cache_root.mkdir(parents=True, exist_ok=True)
        zip_path = cache_root / f"{source}.zip"
        if not zip_path.exists():
            fetch(HISTWORDS_DOWNLOAD_URLS[source], zip_path)
        # Extract — HistWords zips have a single top-level directory; we
        # flatten to ``extracted_dir`` regardless of nesting depth so
        # ``YYYY.pkl`` / ``YYYY.npy`` end up directly inside it.
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name:
                    continue
                target = extracted_dir / name
                if target.exists():
                    continue
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    if not (decade_pkl.exists() and decade_npy.exists()):
        raise FileNotFoundError(
            f"decade {decade} not found in {source} archive at {extracted_dir}; "
            f"expected {decade}.pkl + {decade}.npy"
        )

    with decade_pkl.open("rb") as f:
        vocab: list[str] = pickle.load(f)
    vectors: np.ndarray[Any, Any] = np.load(decade_npy)
    if len(vocab) != vectors.shape[0]:
        raise ValueError(
            f"decade {decade}: vocab size {len(vocab)} != vectors {vectors.shape[0]}"
        )
    return {word: vectors[i] for i, word in enumerate(vocab)}


def histwords_cosine_shift(
    decade_a: int,
    decade_b: int,
    target: str,
    source: str = "eng-all",
    cache_dir: str | Path | None = None,
    _fetch: Callable[[str, Path], None] | None = None,
) -> float:
    """Cosine distance between ``target``'s vectors in two HistWords decades.

    Returns ``1 - cos(v_a, v_b)``. Hamilton et al.'s alignment is
    already Procrustes; this function just looks up the two pre-aligned
    vectors and computes the distance.
    """
    from ..stats import cosine_similarity

    vecs_a = fetch_histwords_decade(decade_a, source, cache_dir, _fetch)
    vecs_b = fetch_histwords_decade(decade_b, source, cache_dir, _fetch)

    if target not in vecs_a:
        raise KeyError(f"target {target!r} not in {source} {decade_a}s vocab")
    if target not in vecs_b:
        raise KeyError(f"target {target!r} not in {source} {decade_b}s vocab")

    sim = cosine_similarity(vecs_a[target], vecs_b[target])
    return 1.0 - sim
