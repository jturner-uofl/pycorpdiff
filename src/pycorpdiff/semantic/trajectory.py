"""Multi-period semantic trajectories of target terms.

Where :func:`semantic_shift` answers "how did the meaning of *target*
differ between two corpora", :func:`semantic_trajectory` answers "how
did the meaning of *target* evolve across a sequence of time periods".
The algorithm is the per-period extension of averaged contextual
embeddings:

1. Slice the corpus by time period (yearly / quarterly / monthly).
2. For each period, extract every KWIC window around the target.
3. Encode each window via the embedder; average into a per-period
   centroid.
4. Pick a baseline period (the first populated period by default) and
   report cosine similarity / distance from each period's centroid to
   the baseline.

The output is a tidy DataFrame ready to plot — periods on x, distance
on y, one line per target if you tracked multiple at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..corpus import Corpus, CorpusSlice
from ..stats import cosine_similarity
from .embed import Embedder, SBERTEmbedder
from .shift import _window_texts


def semantic_trajectory(
    corpus: Corpus | CorpusSlice,
    target: str | list[str],
    time_col: str = "date",
    freq: str = "Y",
    embedder: Embedder | None = None,
    window: int = 5,
    baseline_period: str | pd.Period | None = None,
) -> pd.DataFrame:
    """Track each target's contextual centroid across time periods.

    Parameters
    ----------
    corpus
        A :class:`Corpus` or :class:`CorpusSlice` with a parseable
        ``time_col``.
    target
        One or more target terms. For a single string, the output
        carries one row per period; for a list, one row per
        (period, term) pair.
    time_col
        Column carrying the document date.
    freq
        Pandas offset alias for the bucketing (``"Y"``, ``"Q"``, ``"M"``,
        ``"W"``, ``"D"``).
    embedder
        Anything satisfying :class:`Embedder`. Defaults to
        :class:`SBERTEmbedder` (lazy-loaded sentence-transformers).
        Pass :class:`HashEmbedder` for deterministic offline demos.
    window
        Tokens of context on each side of the target.
    baseline_period
        Which period to anchor the trajectory at. Strings (e.g.
        ``"2015"``) are coerced to :class:`pandas.Period` with the
        ``freq`` above. Defaults to the *first populated* period for
        each target — so two targets with different period coverage
        each get their own appropriate baseline.

    Returns
    -------
    pandas.DataFrame
        Columns ``period``, ``target``, ``n_contexts``,
        ``similarity_to_baseline``, ``distance_from_baseline``.
        Periods with no occurrences of the target carry NaN
        similarity/distance and ``n_contexts == 0``.
    """
    if embedder is None:
        embedder = SBERTEmbedder()

    targets = [target] if isinstance(target, str) else list(target)
    temporal = corpus.by_time(time_col, freq)
    periods = temporal.periods()

    # First pass: per-target, per-period centroids + context counts.
    per_target: dict[str, dict[pd.Period, np.ndarray | None]] = {}
    per_target_counts: dict[str, dict[pd.Period, int]] = {}
    for tgt in targets:
        centroids: dict[pd.Period, np.ndarray | None] = {}
        counts: dict[pd.Period, int] = {}
        for period, slice_ in temporal.iter_slices():
            windows = _window_texts(slice_, tgt, window=window)
            counts[period] = len(windows)
            if not windows:
                centroids[period] = None
                continue
            vecs = np.asarray(embedder.encode(windows), dtype=np.float64)
            centroids[period] = vecs.mean(axis=0)
        per_target[tgt] = centroids
        per_target_counts[tgt] = counts

    # Second pass: choose baseline, compute similarities, emit rows.
    rows: list[dict[str, object]] = []
    for tgt in targets:
        centroids = per_target[tgt]
        populated = [p for p in periods if centroids[p] is not None]
        if not populated:
            if baseline_period is not None:
                raise ValueError(
                    f"baseline_period={baseline_period!r} has no contexts "
                    f"for target {tgt!r}"
                )
            # Target never appears anywhere — emit NaN rows for completeness.
            for period in periods:
                rows.append(
                    {
                        "period": period,
                        "target": tgt,
                        "n_contexts": 0,
                        "similarity_to_baseline": float("nan"),
                        "distance_from_baseline": float("nan"),
                    }
                )
            continue

        if baseline_period is None:
            baseline = populated[0]
        else:
            baseline = (
                pd.Period(baseline_period, freq=freq)
                if isinstance(baseline_period, str)
                else baseline_period
            )
            if baseline not in centroids or centroids[baseline] is None:
                raise ValueError(
                    f"baseline_period={baseline_period!r} has no contexts "
                    f"for target {tgt!r}"
                )

        baseline_vec = centroids[baseline]
        assert baseline_vec is not None  # for mypy
        for period in periods:
            vec = centroids[period]
            if vec is None:
                sim = float("nan")
                dist = float("nan")
            else:
                sim = cosine_similarity(baseline_vec, vec)
                dist = 1.0 - sim
            rows.append(
                {
                    "period": period,
                    "target": tgt,
                    "n_contexts": per_target_counts[tgt][period],
                    "similarity_to_baseline": sim,
                    "distance_from_baseline": dist,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["target", "period"], kind="stable")
        .reset_index(drop=True)
    )
