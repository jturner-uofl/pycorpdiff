"""Semantic shift and neighborhood drift between corpora.

The default strategy is *averaged contextual embeddings*: for each
occurrence of the target term in a corpus, encode its surrounding
window as a sentence, then average across occurrences. The
corpus-specific representation that comes out is what we compare.

This works with any shared-space embedder (SBERT, multilingual SBERT,
HuggingFace encoders). For Hamilton-style independently-trained
embeddings, supply ``align="procrustes"`` to rotate the source space
onto the target space before comparison.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..corpus import Corpus, CorpusSlice
from ..stats import cosine_similarity
from .embed import Embedder, SBERTEmbedder

AlignmentKind = Literal["none", "procrustes"]


def _window_texts(
    corpus: Corpus | CorpusSlice, target: str, window: int
) -> list[str]:
    """Extract every window around ``target`` as a space-joined string."""
    docs_tokens = corpus.tokens()
    out: list[str] = []
    for tokens in docs_tokens:
        for i, tok in enumerate(tokens):
            if tok != target:
                continue
            lo = max(0, i - window)
            hi = min(len(tokens), i + window + 1)
            out.append(" ".join(tokens[lo:hi]))
    return out


def _centroid(vectors: np.ndarray) -> np.ndarray:
    out: np.ndarray = vectors.mean(axis=0)
    return out


def semantic_shift(
    a: Corpus | CorpusSlice,
    b: Corpus | CorpusSlice,
    target: str | list[str],
    embedder: Embedder | None = None,
    window: int = 5,
    align: AlignmentKind = "none",
) -> pd.DataFrame:
    """Embedding-space displacement of target term(s) between corpora.

    Parameters
    ----------
    a, b
        The two corpora (or slices) to compare.
    target
        One or more target terms.
    embedder
        Anything satisfying :class:`Embedder`. Defaults to
        :class:`SBERTEmbedder` (lazy-loaded sentence-transformers).
        Pass :class:`HashEmbedder` for deterministic offline demos.
    window
        Tokens of context on each side of the target.
    align
        ``"none"`` (default) assumes both corpora are embedded with the
        same model and skips alignment. ``"procrustes"`` rotates the
        source's window-vector matrix onto the target's via orthogonal
        Procrustes — appropriate when the embedder produces independent
        per-corpus spaces.

    Returns
    -------
    pandas.DataFrame
        One row per target term with ``cosine_similarity``,
        ``cosine_distance``, ``n_contexts_a``, ``n_contexts_b``.
    """
    targets = [target] if isinstance(target, str) else list(target)
    if embedder is None:
        embedder = SBERTEmbedder()

    rows: list[dict[str, object]] = []
    for tgt in targets:
        wins_a = _window_texts(a, tgt, window=window)
        wins_b = _window_texts(b, tgt, window=window)
        if not wins_a or not wins_b:
            rows.append(
                {
                    "target": tgt,
                    "cosine_similarity": float("nan"),
                    "cosine_distance": float("nan"),
                    "n_contexts_a": len(wins_a),
                    "n_contexts_b": len(wins_b),
                }
            )
            continue

        vecs_a = np.asarray(embedder.encode(wins_a), dtype=np.float64)
        vecs_b = np.asarray(embedder.encode(wins_b), dtype=np.float64)

        if align == "procrustes":
            # Procrustes wants two matrices of the same shape. Pad / truncate
            # the smaller side to make this well-defined; downstream we only
            # care about the rotated centroid.
            from .alignment import procrustes_align

            n = min(len(vecs_a), len(vecs_b))
            vecs_a = procrustes_align(vecs_a[:n], vecs_b[:n])
            vecs_b = vecs_b[:n]

        centroid_a = _centroid(vecs_a)
        centroid_b = _centroid(vecs_b)
        sim = cosine_similarity(centroid_a, centroid_b)
        rows.append(
            {
                "target": tgt,
                "cosine_similarity": sim,
                "cosine_distance": 1.0 - sim,
                "n_contexts_a": len(wins_a),
                "n_contexts_b": len(wins_b),
            }
        )
    return pd.DataFrame(rows)


def neighborhood_drift(
    a: Corpus | CorpusSlice,
    b: Corpus | CorpusSlice,
    target: str,
    k: int = 10,
    embedder: Embedder | None = None,
    window: int = 5,
    min_count: int = 2,
) -> pd.DataFrame:
    """Compare the top-k contextual neighbours of ``target`` in each corpus.

    For each candidate term that appears in the target's windows (in
    either corpus) at least ``min_count`` times, computes a contextual
    centroid in each corpus and ranks by cosine similarity to the
    target's own centroid. Returns the union of the two top-k sets with
    similarity in each corpus and a signed drift score.

    Returns
    -------
    pandas.DataFrame
        Columns ``neighbor``, ``sim_a``, ``sim_b``, ``rank_a``,
        ``rank_b``, ``drift = sim_a - sim_b``, ``status``
        (``"shared"`` / ``"gained_in_a"`` / ``"lost_in_a"``).
    """
    if embedder is None:
        embedder = SBERTEmbedder()

    wins_a = _window_texts(a, target, window=window)
    wins_b = _window_texts(b, target, window=window)
    if not wins_a:
        raise ValueError(f"target {target!r} not found in corpus a")
    if not wins_b:
        raise ValueError(f"target {target!r} not found in corpus b")

    target_vec_a = np.asarray(embedder.encode(wins_a), dtype=np.float64).mean(axis=0)
    target_vec_b = np.asarray(embedder.encode(wins_b), dtype=np.float64).mean(axis=0)

    # Candidate vocabulary: words appearing in any window in either corpus,
    # excluding the target itself.
    candidates: set[str] = set()
    for w in wins_a + wins_b:
        candidates.update(w.split())
    candidates.discard(target)

    sims_a: dict[str, float] = {}
    sims_b: dict[str, float] = {}
    for cand in candidates:
        cand_wins_a = _window_texts(a, cand, window=window)
        cand_wins_b = _window_texts(b, cand, window=window)
        if len(cand_wins_a) >= min_count:
            cv_a = np.asarray(embedder.encode(cand_wins_a), dtype=np.float64).mean(axis=0)
            sims_a[cand] = cosine_similarity(target_vec_a, cv_a)
        if len(cand_wins_b) >= min_count:
            cv_b = np.asarray(embedder.encode(cand_wins_b), dtype=np.float64).mean(axis=0)
            sims_b[cand] = cosine_similarity(target_vec_b, cv_b)

    top_a = sorted(sims_a.items(), key=lambda x: -x[1])[:k]
    top_b = sorted(sims_b.items(), key=lambda x: -x[1])[:k]
    rank_a = {term: i + 1 for i, (term, _) in enumerate(top_a)}
    rank_b = {term: i + 1 for i, (term, _) in enumerate(top_b)}

    union = set(rank_a) | set(rank_b)
    rows: list[dict[str, object]] = []
    for term in union:
        s_a = sims_a.get(term, float("nan"))
        s_b = sims_b.get(term, float("nan"))
        in_a = term in rank_a
        in_b = term in rank_b
        if in_a and in_b:
            status = "shared"
        elif in_a:
            status = "gained_in_a"
        else:
            status = "lost_in_a"
        rows.append(
            {
                "neighbor": term,
                "sim_a": s_a,
                "sim_b": s_b,
                "rank_a": rank_a.get(term),
                "rank_b": rank_b.get(term),
                "drift": (s_a if not np.isnan(s_a) else 0.0)
                - (s_b if not np.isnan(s_b) else 0.0),
                "status": status,
            }
        )
    return (
        pd.DataFrame(rows)
        .assign(_abs=lambda d: d["drift"].abs())
        .sort_values("_abs", ascending=False, kind="stable")
        .drop(columns="_abs")
        .reset_index(drop=True)
    )
