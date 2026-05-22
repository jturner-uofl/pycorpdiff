"""Pearson's χ² keyness statistic.

The historical alternative to Dunning's G² for 2×2 corpus-comparison
contingency tables. Both are asymptotically χ²(1)-distributed under
the null of identical relative frequencies; G² is more robust to
small expected counts (Dunning 1993), which is why pycorpdiff defaults
to it. χ² is exposed here for the *test* of the equivalence and for
researchers replicating older keyness-via-chi-squared studies.

Reference
---------
Pearson, K. (1900). On the criterion that a given system of deviations
from the probable in the case of a correlated system of variables is
such that it can be reasonably supposed to have arisen from random
sampling. *Philosophical Magazine*, 50(302), 157–175.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def chi_squared(
    counts_a: pd.Series,
    counts_b: pd.Series,
    total_a: int,
    total_b: int,
) -> pd.DataFrame:
    """Compute Pearson χ² for every term in ``counts_a ∪ counts_b``.

    Inputs and conventions mirror
    :func:`pycorpdiff.keyness.log_likelihood`: caller is responsible for
    min-count filtering; the returned ``chi_squared`` column is **signed**
    by the direction of overuse (positive when A's rate exceeds B's),
    while the *p*-value is computed from ``|χ²|``.

    Parameters
    ----------
    counts_a, counts_b
        Term-frequency series; missing terms imputed as zero on the union.
    total_a, total_b
        Corpus totals.

    Returns
    -------
    pandas.DataFrame
        Indexed by term, columns ``count_a``, ``count_b``, ``expected_a``,
        ``expected_b``, ``chi_squared`` (signed), ``p_value``.
    """
    if total_a <= 0 or total_b <= 0:
        raise ValueError(f"total_a and total_b must be positive; got {total_a}, {total_b}")

    terms = counts_a.index.union(counts_b.index)
    # Cast to float64 throughout: the 2×2 numerator below is
    # ``(ad − bc)² · N``, which overflows int64 for any realistic
    # corpus size (a few hundred occurrences against a million-token
    # corpus is enough).
    a = counts_a.reindex(terms, fill_value=0).astype(np.float64).to_numpy()
    b = counts_b.reindex(terms, fill_value=0).astype(np.float64).to_numpy()

    obs_sum = a + b
    total = float(total_a + total_b)
    expected_a = float(total_a) * obs_sum / total
    expected_b = float(total_b) * obs_sum / total

    # 2×2 closed form: χ² = ((ad − bc)² · N) / ((a+b)(c+d)(a+c)(b+d))
    # where c = N_a − a (non-term in A), d = N_b − b (non-term in B).
    c = float(total_a) - a
    d = float(total_b) - b
    numerator = (a * d - b * c) ** 2 * total
    denominator = obs_sum * (c + d) * float(total_a) * float(total_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        unsigned = np.where(denominator > 0, numerator / denominator, 0.0)
    unsigned = np.maximum(unsigned, 0.0)

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
            "chi_squared": signed,
            "p_value": p_value,
        },
        index=terms,
    )
