"""Dunning's G² log-likelihood statistic.

Reference
---------
Dunning, T. (1993). Accurate methods for the statistics of surprise and
coincidence. *Computational Linguistics*, 19(1), 61-74.
"""

from __future__ import annotations

import pandas as pd


def log_likelihood(
    counts_a: pd.Series,
    counts_b: pd.Series,
    total_a: int | None = None,
    total_b: int | None = None,
    min_count: int = 5,
) -> pd.DataFrame:
    """Compute Dunning G² for every term in ``counts_a ∪ counts_b``.

    Parameters
    ----------
    counts_a, counts_b
        Term-frequency series. The union of their indices defines the
        shared vocabulary; missing terms are imputed as zero.
    total_a, total_b
        Corpus totals. If ``None``, inferred as the sum of ``counts_a``
        / ``counts_b``. Pass explicit totals when ``counts_*`` has
        already been filtered (e.g. by a min-count threshold) so the
        relative-frequency normalisation remains correct.
    min_count
        Terms with summed-count below this threshold are dropped to
        avoid Dunning's well-documented small-cell unreliability.

    Returns
    -------
    pandas.DataFrame
        Columns: ``term``, ``count_a``, ``count_b``, ``g2``, ``p_value``.
    """
    raise NotImplementedError("log_likelihood() lands in Phase 1")
