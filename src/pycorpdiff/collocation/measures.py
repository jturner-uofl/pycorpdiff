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
Church, K. W., & Hanks, P. (1990). Word association norms, mutual
information, and lexicography. *Computational Linguistics*, 16(1),
22-29. (Pointwise mutual information for collocation.)

Church, K., Gale, W., Hanks, P., & Hindle, D. (1991). Using statistics in
lexical analysis. In *Lexical Acquisition*, 115-164. (t-score.)

Daille, B. (1994). *Approche mixte pour l'extraction automatique de
terminologie*. PhD thesis, Université Paris 7. (MI³ — cube weighting
of PMI to correct rare-pair inflation.)

Rychlý, P. (2008). A lexicographer-friendly association score. In
*Proceedings of RASLAN 2008*. (logDice.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_float(s: pd.Series | float) -> Any:
    """Upcast counts to float64 so the intermediate products
    (``f_xy · n``, ``f_xy³``, ``f_x · f_y``) don't silently overflow
    int64 on large corpora — for a high-frequency target with
    ``f_xy ~ 10⁷``, ``f_xy³ ~ 10²¹`` overflows int64 (~9.2 × 10¹⁸).
    """
    if isinstance(s, pd.Series):
        return s.astype(np.float64)
    return np.float64(s)


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
    need finite scores across the union of vocabularies. Raises
    ``ValueError`` if the marginals sum to zero on every term — the
    formula is undefined there.
    """
    f_xy_f = _as_float(f_xy)
    f_x_f = _as_float(f_x)
    f_y_f = _as_float(f_y)
    denom = f_x_f + f_y_f
    # Guard the degenerate input where both marginals are zero everywhere.
    if isinstance(denom, pd.Series):
        if (denom == 0).all():
            raise ValueError(
                "logdice: f_x + f_y == 0 for every term; cannot compute "
                "the association measure on a vacuous contingency table."
            )
    elif denom == 0:
        raise ValueError(
            "logdice: f_x + f_y == 0; cannot compute the association "
            "measure on a vacuous contingency table."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (2.0 * f_xy_f) / denom
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

    Inputs are upcast to ``float64`` so the ``f_xy · N`` and
    ``f_x · f_y`` products don't silently overflow ``int64`` on
    large corpora (BNC, ukWaC, COCA scale).
    """
    f_xy_f = _as_float(f_xy)
    f_x_f = _as_float(f_x)
    f_y_f = _as_float(f_y)
    n_f = np.float64(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(
            np.log2((f_xy_f * n_f) / (f_x_f * f_y_f)), index=f_xy.index
        )


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

    Inputs are upcast to ``float64`` to avoid int64 overflow on the
    ``f_x · f_y`` product at large-corpus scale.
    """
    f_xy_f = _as_float(f_xy)
    f_x_f = _as_float(f_x)
    f_y_f = _as_float(f_y)
    n_f = np.float64(n)
    expected = (f_x_f * f_y_f) / n_f
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series((f_xy_f - expected) / np.sqrt(f_xy_f), index=f_xy.index)


def mi_three(
    f_xy: pd.Series,
    f_x: float,
    f_y: pd.Series,
    n: int,
) -> pd.Series:
    """Daille's MI³: ``log2(f_xy³ · N / (f_x · f_y))``.

    Cubes the joint count in the numerator, which empirically downweights
    PMI's rare-pair bias without t-score's frequency dominance.

    **Important:** inputs are upcast to ``float64`` because ``f_xy³``
    overflows ``int64`` for joint counts above roughly 2.1 × 10⁶ — a
    threshold real corpora cross routinely. Prior to this upcast the
    function returned silently-wrong negative values on plausible
    large-corpus inputs.
    """
    f_xy_f = _as_float(f_xy)
    f_x_f = _as_float(f_x)
    f_y_f = _as_float(f_y)
    n_f = np.float64(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(
            np.log2((np.power(f_xy_f, 3) * n_f) / (f_x_f * f_y_f)),
            index=f_xy.index,
        )
