"""Bayes factor keyness.

Reference
---------
Wilson, A. (2013). Embracing Bayes factors for key item analysis in
corpus linguistics. In *New Approaches to the Study of Linguistic
Variability* (pp. 3-11).
"""

from __future__ import annotations

import pandas as pd


def bayes_factor(
    counts_a: pd.Series,
    counts_b: pd.Series,
    total_a: int,
    total_b: int,
) -> pd.Series:
    """Compute the Bayes factor for each term's frequency difference.

    Returned values are in raw BF units (not log-BF); interpret with
    Kass & Raftery's (1995) thresholds: BF > 10 = "strong",
    BF > 100 = "decisive".
    """
    raise NotImplementedError("bayes_factor() lands in Phase 1")
