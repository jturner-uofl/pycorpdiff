"""Effect-size measures for corpus comparison.

References
----------
Hardie, A. (2014). Log Ratio: An informal introduction. Centre for Corpus
Approaches to Social Science (CASS).

Gabrielatos, C. (2018). Keyness analysis: Nature, metrics and techniques.
In *Corpus Approaches to Discourse* (pp. 225-258). Routledge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_ratio(
    counts_a: pd.Series,
    counts_b: pd.Series,
    total_a: int,
    total_b: int,
    smoothing: float = 0.5,
) -> pd.Series:
    """Hardie's LogRatio: ``log2((a+α)/N_a) - log2((b+α)/N_b)``.

    The Laplace ``smoothing`` constant (default 0.5, per Hardie) is added
    to every count before normalisation so terms absent from one corpus
    yield a finite (rather than ``±inf``) effect size. Positive LogRatio
    means the term is more frequent in A; negative means more frequent in B.
    """
    if smoothing <= 0:
        raise ValueError(f"smoothing must be > 0; got {smoothing}")

    terms = counts_a.index.union(counts_b.index)
    a = counts_a.reindex(terms, fill_value=0).astype(float)
    b = counts_b.reindex(terms, fill_value=0).astype(float)
    rate_a = (a + smoothing) / total_a
    rate_b = (b + smoothing) / total_b
    return pd.Series(np.log2(rate_a / rate_b), index=terms, name="log_ratio")


def percent_diff(
    counts_a: pd.Series,
    counts_b: pd.Series,
    total_a: int,
    total_b: int,
) -> pd.Series:
    """%DIFF — Gabrielatos's normalised percentage difference.

    Computed as ``(rate_a - rate_b) / rate_b * 100`` where rates are per
    million tokens. The denominator's choice of per-million is convention
    only; it cancels out of the ratio. Returns ``+inf`` when ``b == 0``
    and ``a > 0`` (the term is novel in A) — the caller may wish to
    filter or replace those rows for plotting.
    """
    terms = counts_a.index.union(counts_b.index)
    a = counts_a.reindex(terms, fill_value=0).astype(float)
    b = counts_b.reindex(terms, fill_value=0).astype(float)

    rate_a = a / total_a * 1_000_000
    rate_b = b / total_b * 1_000_000
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = (rate_a - rate_b) / rate_b * 100.0
    return pd.Series(diff, index=terms, name="percent_diff")
