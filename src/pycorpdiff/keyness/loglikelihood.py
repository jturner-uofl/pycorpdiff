"""Dunning's G² log-likelihood statistic.

Reference
---------
Dunning, T. (1993). Accurate methods for the statistics of surprise and
coincidence. *Computational Linguistics*, 19(1), 61-74.

Notes
-----
The G² returned by :func:`log_likelihood` is **signed**: positive when the
term is overused in corpus A relative to corpus B (i.e. ``a/N_a > b/N_b``)
and negative when overused in B. This is the convention CASS / Lancaster
tooling has gravitated toward — it carries direction information without
needing a separate column. The reported *p*-value uses ``|G²|`` as the
test statistic; the unsigned form is what's chi-squared distributed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2


def log_likelihood(
    counts_a: pd.Series,
    counts_b: pd.Series,
    total_a: int,
    total_b: int,
) -> pd.DataFrame:
    """Compute Dunning G² for every term in the union of input indices.

    ``counts_a`` and ``counts_b`` are aligned on their union; missing
    terms are imputed as zero. No min-count filtering is applied here —
    that is the caller's responsibility (see
    :meth:`pycorpdiff.Comparison.keyness`).

    Parameters
    ----------
    counts_a, counts_b
        Term-frequency series. Index entries are terms; values are
        non-negative integer counts.
    total_a, total_b
        Corpus totals (token counts before any min-count filter). Used
        for the contingency-table "not-term" cells.

    Returns
    -------
    pandas.DataFrame
        Indexed by term, with columns ``count_a``, ``count_b``,
        ``expected_a``, ``expected_b``, ``g2`` (signed), ``p_value``.
    """
    if total_a <= 0 or total_b <= 0:
        raise ValueError(f"total_a and total_b must be positive; got {total_a}, {total_b}")

    terms = counts_a.index.union(counts_b.index)
    a = counts_a.reindex(terms, fill_value=0).astype(np.int64).to_numpy()
    b = counts_b.reindex(terms, fill_value=0).astype(np.int64).to_numpy()

    obs_sum = a + b
    total = total_a + total_b
    expected_a = total_a * obs_sum / total
    expected_b = total_b * obs_sum / total

    # 2 * sum_i O_i * ln(O_i / E_i), with xlogy giving 0*log(0)=0.
    unsigned = 2.0 * (
        xlogy(a, a) - xlogy(a, expected_a) + xlogy(b, b) - xlogy(b, expected_b)
    )
    # Mathematically G² >= 0; clip away the tiny negative values that
    # surface from float roundoff when the two corpora have ~identical rates.
    unsigned = np.maximum(unsigned, 0.0)

    # Sign by direction of overuse: + when A's rate exceeds B's, else -.
    a_rate = a / total_a
    b_rate = b / total_b
    sign = np.where(a_rate >= b_rate, 1.0, -1.0)
    signed = sign * unsigned

    p_value = chi2.sf(unsigned, df=1)

    return pd.DataFrame(
        {
            "count_a": a,
            "count_b": b,
            "expected_a": expected_a,
            "expected_b": expected_b,
            "g2": signed,
            "p_value": p_value,
        },
        index=terms,
    )
