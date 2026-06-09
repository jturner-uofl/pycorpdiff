"""Bayes factor keyness, BIC-based approximation.

References
----------
Wilson, A. (2013). Embracing Bayes factors for key item analysis in
corpus linguistics. In *New Approaches to the Study of Linguistic
Variability* (pp. 3-11).

Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the
American Statistical Association*, 90(430), 773-795.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loglikelihood import LLFormula, log_likelihood


def bayes_factor(
    counts_a: pd.Series,
    counts_b: pd.Series,
    total_a: int,
    total_b: int,
    *,
    formula: LLFormula = "rayson",
) -> pd.Series:
    """BIC-approximated Bayes factor for each term's frequency difference.

    The BIC approximation (Kass & Raftery 1995): ``BIC = |G²| - ln(N)``
    where ``N`` is the total tokens across both corpora and ``G²`` is
    the unsigned log-likelihood. The Bayes factor is then
    ``exp(BIC / 2)``. Wilson (2013) is the keyness application.

    ``formula`` selects which G² flavour feeds the BF: ``"rayson"`` (the
    2-cell shortcut, default; matches the LL Wizard) or ``"dunning"``
    (the full 4-cell G²; matches quanteda/NLTK). Use the same
    ``formula=`` as the ``keyness()`` call that produced the row so the
    G² and the Bayes factor in a single row describe the same statistic.

    Interpret with Kass & Raftery (1995):

    - ``BF > 2``  : positive evidence
    - ``BF > 6``  : strong evidence
    - ``BF > 10`` : very strong evidence
    - ``BF > 100``: decisive evidence

    Very large BF values overflow float64 and surface as ``inf``; that is
    semantically correct ("evidence is essentially conclusive") and pandas
    plots / sorts handle it.
    """
    terms = counts_a.index.union(counts_b.index)
    ll_table = log_likelihood(counts_a, counts_b, total_a, total_b, formula=formula)
    g2_abs = ll_table["g2"].abs()
    bic = g2_abs - np.log(total_a + total_b)
    with np.errstate(over="ignore"):
        bf = np.exp(bic / 2.0)
    return pd.Series(bf, index=terms, name="bayes_factor")
