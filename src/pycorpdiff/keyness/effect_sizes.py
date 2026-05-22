"""Effect-size measures for corpus comparison.

References
----------
Hardie, A. (2014). Log Ratio: An informal introduction. Centre for Corpus
Approaches to Social Science (CASS).

Gabrielatos, C. (2018). Keyness analysis: Nature, metrics and techniques.
In *Corpus Approaches to Discourse* (pp. 225-258). Routledge.
"""

from __future__ import annotations

import pandas as pd


def log_ratio(
    counts_a: pd.Series,
    counts_b: pd.Series,
    total_a: int,
    total_b: int,
    smoothing: float = 0.5,
) -> pd.Series:
    """Hardie's LogRatio: ``log2( (a / N_a) / (b / N_b) )``.

    The ``smoothing`` constant is added to every count before
    normalisation to avoid ``log(0)`` for terms absent from one
    corpus; the default of 0.5 matches Hardie's recommendation.
    """
    raise NotImplementedError("log_ratio() lands in Phase 1")


def percent_diff(
    counts_a: pd.Series,
    counts_b: pd.Series,
    total_a: int,
    total_b: int,
) -> pd.Series:
    """%DIFF — the per-thousand percentage difference (Gabrielatos)."""
    raise NotImplementedError("percent_diff() lands in Phase 1")
