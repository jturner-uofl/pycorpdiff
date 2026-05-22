"""Collocation association measures.

Reference
---------
Rychlý, P. (2008). A lexicographer-friendly association score.
In *Proceedings of RASLAN 2008*.
"""

from __future__ import annotations

import pandas as pd


def logdice(
    f_xy: pd.Series,
    f_x: int,
    f_y: pd.Series,
) -> pd.Series:
    """logDice (Rychlý 2008) — range-bounded, robust to corpus size."""
    raise NotImplementedError("logdice() lands in Phase 2")


def pmi(
    f_xy: pd.Series,
    f_x: int,
    f_y: pd.Series,
    n: int,
) -> pd.Series:
    """Pointwise mutual information.

    Inflates rare collocations; pair with a min-count threshold.
    """
    raise NotImplementedError("pmi() lands in Phase 2")


def t_score(
    f_xy: pd.Series,
    f_x: int,
    f_y: pd.Series,
    n: int,
) -> pd.Series:
    """t-score (Church et al. 1991) — favours frequent collocates."""
    raise NotImplementedError("t_score() lands in Phase 2")


def mi_score(
    f_xy: pd.Series,
    f_x: int,
    f_y: pd.Series,
    n: int,
) -> pd.Series:
    """Mutual Information (log2 form, standard CL definition)."""
    raise NotImplementedError("mi_score() lands in Phase 2")


def mi_three(
    f_xy: pd.Series,
    f_x: int,
    f_y: pd.Series,
    n: int,
) -> pd.Series:
    """MI³ — MI cubed in the numerator; downweights rare collocates."""
    raise NotImplementedError("mi_three() lands in Phase 2")
