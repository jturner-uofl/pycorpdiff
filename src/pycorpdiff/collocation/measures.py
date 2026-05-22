"""Collocation association measures.

All four functions accept the same five-argument shape:

``f_xy``
    Joint count of (target, collocate) in a window.
``f_x``
    Total occurrences of the target in the corpus.
``f_y``
    Total occurrences of the collocate in the corpus.
``n``
    Total tokens in the corpus (required by every measure except logDice).

Counts may be passed as scalars or pandas Series; broadcasting follows
NumPy / pandas conventions.

References
----------
Rychlý, P. (2008). A lexicographer-friendly association score. In
*Proceedings of RASLAN 2008*.

Church, K., Gale, W., Hanks, P., & Hindle, D. (1991). Using statistics in
lexical analysis. In *Lexical Acquisition*, 115-164.

Daille, B. (1994). *Approche mixte pour l'extraction automatique de
terminologie*. PhD thesis, Université Paris 7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def logdice(
    f_xy: pd.Series,
    f_x: float,
    f_y: pd.Series,
) -> pd.Series:
    """Rychlý's logDice: ``14 + log2(2 · f_xy / (f_x + f_y))``.

    Range-bounded above at 14 (perfect co-occurrence). Robust to corpus
    size because it never references the total. Values below 0 are
    typically noise; the practical interesting band is roughly 7..14.

    Zero joint counts yield ``-inf``; pre-smooth ``f_xy`` upstream
    (e.g. via :func:`pycorpdiff.collocation.collocation_shift`) if you
    need finite scores across the union of vocabularies.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (2.0 * f_xy) / (f_x + f_y)
        return pd.Series(14.0 + np.log2(ratio), index=f_xy.index)


def pmi(
    f_xy: pd.Series,
    f_x: float,
    f_y: pd.Series,
    n: int,
) -> pd.Series:
    """Pointwise mutual information: ``log2(f_xy · N / (f_x · f_y))``.

    The "association ratio" of Church & Hanks (1990). PMI rewards rare
    pairs disproportionately — always pair with a frequency floor or
    use MI³ if rare-pair inflation is a concern.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(np.log2((f_xy * n) / (f_x * f_y)), index=f_xy.index)


def t_score(
    f_xy: pd.Series,
    f_x: float,
    f_y: pd.Series,
    n: int,
) -> pd.Series:
    """Welch-style t-score: ``(f_xy - E[f_xy]) / sqrt(f_xy)``.

    Where ``E[f_xy] = f_x · f_y / N`` is the count expected under
    independence. Favours frequent collocates — the inverse of PMI's
    sparsity bias.
    """
    expected = (f_x * f_y) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series((f_xy - expected) / np.sqrt(f_xy), index=f_xy.index)


def mi_three(
    f_xy: pd.Series,
    f_x: float,
    f_y: pd.Series,
    n: int,
) -> pd.Series:
    """Daille's MI³: ``log2(f_xy³ · N / (f_x · f_y))``.

    Cubes the joint count in the numerator, which empirically downweights
    PMI's rare-pair bias without t-score's frequency dominance.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(
            np.log2((np.power(f_xy, 3) * n) / (f_x * f_y)), index=f_xy.index
        )
