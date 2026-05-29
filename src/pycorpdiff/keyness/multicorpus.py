"""N-way keyness — does a term's rate differ across more than two corpora?

The two-corpus :func:`log_likelihood` answers "is this term distinctive
in A vs B". When you have three or more corpora — three news outlets,
five parties, ten decades — the natural generalisation is a one-way
contingency test: for each term, does its observed rate vary across
corpora more than chance would explain?

The reported G² uses the same marginal-only form as the two-way path
(:func:`pycorpdiff.keyness.loglikelihood.log_likelihood`), so the N=2
result agrees exactly with the existing keyness pipeline. Asymptotic
distribution is χ²(df=N−1).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from ..corpus import Corpus, CorpusSlice


def keyness_multi(
    corpora: Sequence[Corpus | CorpusSlice],
    *,
    labels: Sequence[str] | None = None,
    min_count: int = 5,
    multiple_comparisons: str = "bh",
    stop_words: set[str] | list[str] | None = None,
) -> pd.DataFrame:
    """Test for each term whether its rate varies across ``corpora``.

    For ``N`` corpora and each shared-vocabulary term:

    .. math::

       G^2 = 2 \\sum_{i=1}^{N} O_i \\ln\\!\\left(\\frac{O_i}{E_i}\\right),
       \\quad E_i = \\frac{(\\sum_j O_j)\\, n_i}{\\sum_j n_j}

    where :math:`O_i` is the term's count in corpus *i* and :math:`n_i`
    is corpus *i*'s total tokens. Asymptotic distribution is
    :math:`\\chi^2(\\text{df}=N-1)`.

    Parameters
    ----------
    corpora
        Two or more corpora (or slices) to compare side-by-side.
    labels
        Optional labels for the corpora; default ``["corpus_0",
        "corpus_1", ...]``. Used to name the per-corpus count columns.
    min_count
        Drop terms whose total count across all corpora is below this.
    multiple_comparisons
        ``"bh"`` (default), ``"bonferroni"``, or ``"none"``.
    stop_words
        Iterable of terms to exclude before scoring; same semantics as
        the two-way :meth:`Comparison.keyness` parameter.

    Returns
    -------
    pandas.DataFrame
        Indexed by term, sorted by G² descending. Columns: per-corpus
        ``count_<label>``, ``g2``, ``p_value``, and
        ``p_adjusted`` (unless ``multiple_comparisons="none"``).
    """
    if len(corpora) < 2:
        raise ValueError(f"need at least 2 corpora; got {len(corpora)}")

    if labels is None:
        labels = [f"corpus_{i}" for i in range(len(corpora))]
    if len(labels) != len(corpora):
        raise ValueError(
            f"labels must have one entry per corpus; "
            f"got {len(labels)} labels for {len(corpora)} corpora"
        )

    vocabs = [c.vocab(min_count=1) for c in corpora]
    totals = np.array([int(v.sum()) for v in vocabs], dtype=np.int64)
    if (totals == 0).any():
        which = [labels[i] for i, n in enumerate(totals) if n == 0]
        raise ValueError(f"empty corpora: {which}")

    all_terms = pd.Index([])
    for v in vocabs:
        all_terms = all_terms.union(v.index)

    counts_matrix = np.zeros((len(all_terms), len(corpora)), dtype=np.int64)
    for j, v in enumerate(vocabs):
        aligned = v.reindex(all_terms, fill_value=0).astype(np.int64).to_numpy()
        counts_matrix[:, j] = aligned

    keep_mask = counts_matrix.sum(axis=1) >= min_count
    if stop_words is not None:
        stop_set = set(stop_words)
        keep_mask &= ~all_terms.isin(stop_set)
    kept_terms = all_terms[keep_mask]
    kept_counts = counts_matrix[keep_mask]

    if len(kept_terms) == 0:
        empty = pd.DataFrame(
            {f"count_{label}": pd.Series(dtype=np.int64) for label in labels}
        )
        empty["g2"] = pd.Series(dtype=float)
        empty["p_value"] = pd.Series(dtype=float)
        if multiple_comparisons != "none":
            empty["p_adjusted"] = pd.Series(dtype=float)
        return empty.rename_axis("term")

    total_per_term = kept_counts.sum(axis=1).astype(np.float64)
    grand_total = float(totals.sum())
    expected = total_per_term[:, None] * totals[None, :] / grand_total
    unsigned = 2.0 * (
        xlogy(kept_counts, kept_counts) - xlogy(kept_counts, expected)
    ).sum(axis=1)
    unsigned = np.maximum(unsigned, 0.0)
    p_value = chi2.sf(unsigned, df=len(corpora) - 1)

    table = pd.DataFrame(
        {f"count_{label}": kept_counts[:, j] for j, label in enumerate(labels)},
        index=kept_terms,
    )
    table["g2"] = unsigned
    table["p_value"] = p_value

    if multiple_comparisons == "bh":
        from .correction import benjamini_hochberg

        table["p_adjusted"] = benjamini_hochberg(p_value)
    elif multiple_comparisons == "bonferroni":
        from .correction import bonferroni

        table["p_adjusted"] = bonferroni(p_value)
    elif multiple_comparisons != "none":
        raise ValueError(
            f"multiple_comparisons must be 'bh', 'bonferroni', or 'none'; "
            f"got {multiple_comparisons!r}"
        )

    return table.sort_values("g2", ascending=False).rename_axis("term")
