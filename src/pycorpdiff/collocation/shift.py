"""Cross-corpus collocation shift — gained / lost collocates of a target."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from ..corpus import Corpus, CorpusSlice
from .cooccurrence import collocate_counts
from .measures import logdice, mi_three, pmi, t_score

CollocationMeasure = Literal["logDice", "PMI", "t_score", "MI3"]

_MEASURES_NEED_N: dict[str, bool] = {
    "logDice": False,
    "PMI": True,
    "t_score": True,
    "MI3": True,
}


def collocation_shift(
    a: Corpus | CorpusSlice,
    b: Corpus | CorpusSlice,
    target: str,
    window: int = 5,
    measure: CollocationMeasure = "logDice",
    min_count: int = 5,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """Compute the change in target-term collocates between two corpora.

    For every collocate that meets the combined ``min_count`` threshold,
    the chosen association measure is computed in each corpus and the
    difference (``score_a - score_b``) is reported. Laplace ``smoothing``
    is applied to joint and marginal counts before scoring so collocates
    absent on one side yield finite scores; the default α=0.5 mirrors
    Hardie's LogRatio smoothing and Rychlý's logDice convention.

    Parameters
    ----------
    a, b
        The two corpora (or slices) to compare. ``target`` must appear
        in both — its complete absence on one side makes the shift
        undefined.
    target
        The pivot term whose collocates we are tracking.
    window
        Context size on each side of the target.
    measure
        Which association measure to apply.
    min_count
        Drop collocates whose ``count_a + count_b`` is below this.
    smoothing
        Laplace constant added to joint / marginal counts before
        scoring. Must be > 0.

    Returns
    -------
    pandas.DataFrame
        Indexed by collocate, columns: ``count_a``, ``count_b``,
        ``score_a``, ``score_b``, ``shift``. Sorted by ``|shift|``
        descending.
    """
    if smoothing <= 0:
        raise ValueError(f"smoothing must be > 0; got {smoothing}")
    if measure not in _MEASURES_NEED_N:
        raise ValueError(
            f"unknown measure={measure!r}; expected one of {list(_MEASURES_NEED_N)}"
        )

    tokens_a = a.tokens()
    tokens_b = b.tokens()
    cocount_a, fx_a = collocate_counts(tokens_a, target, window=window)
    cocount_b, fx_b = collocate_counts(tokens_b, target, window=window)

    if fx_a == 0:
        raise ValueError(f"target {target!r} not found in corpus a")
    if fx_b == 0:
        raise ValueError(f"target {target!r} not found in corpus b")

    all_collocates = sorted(set(cocount_a) | set(cocount_b))
    fxy_a_raw = pd.Series(
        {c: cocount_a.get(c, 0) for c in all_collocates}, dtype="int64"
    )
    fxy_b_raw = pd.Series(
        {c: cocount_b.get(c, 0) for c in all_collocates}, dtype="int64"
    )
    keep = (fxy_a_raw + fxy_b_raw) >= min_count
    fxy_a_raw = fxy_a_raw[keep]
    fxy_b_raw = fxy_b_raw[keep]

    if len(fxy_a_raw) == 0:
        return pd.DataFrame(
            columns=["count_a", "count_b", "score_a", "score_b", "shift"]
        ).rename_axis("collocate")

    vocab_a = a.vocab()
    vocab_b = b.vocab()
    fy_a_raw = vocab_a.reindex(fxy_a_raw.index, fill_value=0).astype(float)
    fy_b_raw = vocab_b.reindex(fxy_b_raw.index, fill_value=0).astype(float)

    n_a = a.total_tokens()
    n_b = b.total_tokens()

    # Laplace smoothing across joint and marginal counts — keeps every
    # measure finite even for collocates absent on one side. f_x (the
    # target count) is also smoothed for symmetry.
    fxy_a = fxy_a_raw + smoothing
    fxy_b = fxy_b_raw + smoothing
    fy_a = fy_a_raw + smoothing
    fy_b = fy_b_raw + smoothing
    fx_a_s = fx_a + smoothing
    fx_b_s = fx_b + smoothing

    if measure == "logDice":
        score_a = logdice(fxy_a, fx_a_s, fy_a)
        score_b = logdice(fxy_b, fx_b_s, fy_b)
    elif measure == "PMI":
        score_a = pmi(fxy_a, fx_a_s, fy_a, n_a)
        score_b = pmi(fxy_b, fx_b_s, fy_b, n_b)
    elif measure == "t_score":
        score_a = t_score(fxy_a, fx_a_s, fy_a, n_a)
        score_b = t_score(fxy_b, fx_b_s, fy_b, n_b)
    else:  # MI3
        score_a = mi_three(fxy_a, fx_a_s, fy_a, n_a)
        score_b = mi_three(fxy_b, fx_b_s, fy_b, n_b)

    shift = score_a - score_b
    table = pd.DataFrame(
        {
            "count_a": fxy_a_raw.astype("int64"),
            "count_b": fxy_b_raw.astype("int64"),
            "score_a": score_a,
            "score_b": score_b,
            "shift": shift,
        }
    )
    table.index.name = "collocate"
    sort_key = table["shift"].abs()
    return (
        table.assign(_sort_key=sort_key)
        .sort_values("_sort_key", ascending=False, kind="stable")
        .drop(columns="_sort_key")
    )
