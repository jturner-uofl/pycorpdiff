"""Dispersion measures for corpus-comparison sanity checks.

A term can be "key" (significant + large effect) simply because one
document overuses it. Reporting dispersion alongside keyness lets the
caller filter out these spurious findings.

References
----------
Juilland, A., & Chang-Rodríguez, E. (1964). *Frequency Dictionary of
Spanish Words*. Mouton.

Gries, S. Th. (2008). Dispersions and adjusted frequencies in corpora.
*International Journal of Corpus Linguistics*, 13(4), 403-437.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def juilland_d(doc_term_matrix: pd.DataFrame) -> pd.Series:
    """Juilland's D — a 0..1 dispersion score; higher is more even.

    Assumes the rows of ``doc_term_matrix`` are equally weighted parts
    (i.e. treats each document as one "part"). For arbitrarily-sized
    parts, aggregate to fixed-size buckets first.

    Per-term D = ``1 - CV / sqrt(k - 1)``, where CV is the coefficient
    of variation of the term's per-document relative frequencies and
    ``k`` is the number of documents. D = 1 means perfectly even
    spread; D = 0 means concentrated in one document.

    Edge cases: when ``k == 1`` the formula is undefined and we return
    NaN. When a term's count is zero everywhere the per-document rates
    are all zero and we return 0 (no spread).
    """
    k = len(doc_term_matrix)
    if k <= 1:
        return pd.Series(np.nan, index=doc_term_matrix.columns, name="juilland_d")

    counts = doc_term_matrix.to_numpy(dtype=float)  # (k, V)
    doc_totals = counts.sum(axis=1)  # (k,)
    # Per-document relative frequencies. Empty documents contribute zero
    # rate everywhere (avoid divide-by-zero with a safe denominator).
    safe_totals = np.where(doc_totals > 0, doc_totals, 1.0)
    rates = counts / safe_totals[:, None]
    rates = np.where(doc_totals[:, None] > 0, rates, 0.0)

    mean = rates.mean(axis=0)
    std = rates.std(axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean > 0, std / mean, 0.0)
    d = np.where(mean > 0, 1.0 - cv / np.sqrt(k - 1), 0.0)
    return pd.Series(d, index=doc_term_matrix.columns, name="juilland_d")


def dispersion_dp(doc_term_matrix: pd.DataFrame) -> pd.Series:
    """Gries's DP (Deviation of Proportions) — 0..1; lower is more even.

    For each document ``i`` with size ``s_i`` (in tokens) and target-term
    count ``c_i``, let ``expected_i = s_i / S`` (the document's share of
    the corpus) and ``observed_i = c_i / C`` (the document's share of the
    target's occurrences). Then ``DP = 0.5 * Σ |observed_i - expected_i|``.

    DP = 0 means perfectly even spread; DP near 1 means total
    concentration. We return the unnormalised form (Gries 2008 §3); the
    normalised variant ``DPnorm = DP / (1 - min(expected_i))`` is a
    one-line transformation if needed.
    """
    if len(doc_term_matrix) == 0:
        return pd.Series(dtype=float, name="dispersion_dp")

    doc_sizes = doc_term_matrix.sum(axis=1).astype(float).to_numpy()
    s_total = doc_sizes.sum()
    if s_total == 0:
        return pd.Series(np.nan, index=doc_term_matrix.columns, name="dispersion_dp")
    expected = doc_sizes / s_total  # shape (k,)

    term_totals = doc_term_matrix.sum(axis=0).astype(float)
    counts = doc_term_matrix.astype(float).to_numpy()  # shape (k, V)

    with np.errstate(divide="ignore", invalid="ignore"):
        observed = counts / term_totals.to_numpy()  # shape (k, V), broadcasts
    # Terms with zero total get observed = NaN/Inf → replace with expected
    # so |observed - expected| = 0 (a uniformly-absent term has trivial DP).
    observed = np.where(term_totals.to_numpy() > 0, observed, expected[:, None])
    dp = 0.5 * np.abs(observed - expected[:, None]).sum(axis=0)
    return pd.Series(dp, index=doc_term_matrix.columns, name="dispersion_dp")
