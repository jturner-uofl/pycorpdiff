"""Log-likelihood-ratio keyness statistic (Dunning, Rayson formulations).

References
----------
Dunning, T. (1993). Accurate methods for the statistics of surprise and
coincidence. *Computational Linguistics*, 19(1), 61-74.

Rayson, P., & Garside, R. (2000). Comparing corpora using frequency
profiling. In *Proceedings of the Workshop on Comparing Corpora*,
ACL 2000, pp. 1-6.

Notes
-----
Two slightly different log-likelihood-ratio statistics circulate in the
corpus-linguistics literature for the 2-corpus / 2-state contingency
table:

- **Rayson** (``formula="rayson"``, default): the 2-cell shortcut
  ``LL = 2·(O₁·ln(O₁/E₁) + O₂·ln(O₂/E₂))``, summing only the
  *term-present* cells. This is the formulation behind the
  UCREL Lancaster LL Wizard (the standard online reference at
  http://ucrel.lancs.ac.uk/llwizard.html), originally published in
  Rayson & Garside (2000).

- **Dunning** (``formula="dunning"``): the full 4-cell G²,
  summing over *all four* cells of the 2×2 contingency table —
  ``LL = 2·Σᵢ₌₁..₄ Oᵢ·ln(Oᵢ/Eᵢ)``. This is the classical
  Dunning (1993) likelihood-ratio statistic and the formulation used
  by NLTK's :class:`BigramAssocMeasures` and R's
  :func:`quanteda::textstat_keyness(measure="lr")`.

For the canonical Rayson example (12000/1M vs 10000/1M), the two
formulations give 182.07 (Rayson) and 184.10 (Dunning). For the
near-symmetric, low-frequency cases that dominate corpus-linguistics
practice the two are typically within 1-2 % of each other; for highly
asymmetric corpora or high-frequency terms they diverge more.

The statistic returned by :func:`log_likelihood` is **signed**: positive
when the term is overused in corpus A relative to corpus B (i.e.
``a/N_a > b/N_b``) and negative when overused in B. This is the
convention CASS / Lancaster tooling has gravitated toward — it carries
direction information without needing a separate column. The reported
*p*-value uses the unsigned magnitude as the test statistic; both
formulations are asymptotically χ²₁-distributed under the null.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

LLFormula = Literal["rayson", "dunning"]


def log_likelihood(
    counts_a: pd.Series,
    counts_b: pd.Series,
    total_a: int,
    total_b: int,
    *,
    formula: LLFormula = "rayson",
) -> pd.DataFrame:
    """Compute the signed log-likelihood-ratio keyness statistic.

    Two formulations are available; see the module docstring for the
    references and the math.

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
        for the contingency-table "not-term" cells under
        ``formula="dunning"``.
    formula
        ``"rayson"`` (default; 2-cell shortcut, matches Rayson's LL
        Wizard) or ``"dunning"`` (full 4-cell G²).

    Returns
    -------
    pandas.DataFrame
        Indexed by term, with columns ``count_a``, ``count_b``,
        ``expected_a``, ``expected_b``, ``g2`` (signed), ``p_value``.
    """
    if total_a <= 0 or total_b <= 0:
        raise ValueError(f"total_a and total_b must be positive; got {total_a}, {total_b}")
    if formula not in ("rayson", "dunning"):
        raise ValueError(
            f"formula must be 'rayson' or 'dunning'; got {formula!r}"
        )

    terms = counts_a.index.union(counts_b.index)
    a = counts_a.reindex(terms, fill_value=0).astype(np.int64).to_numpy()
    b = counts_b.reindex(terms, fill_value=0).astype(np.int64).to_numpy()

    obs_sum = a + b
    total = total_a + total_b
    expected_a = total_a * obs_sum / total
    expected_b = total_b * obs_sum / total

    # Rayson 2-cell shortcut: only the term-present rows.
    # LL = 2 · (O₁·ln(O₁/E₁) + O₂·ln(O₂/E₂))
    contrib = (
        xlogy(a, a) - xlogy(a, expected_a) + xlogy(b, b) - xlogy(b, expected_b)
    )
    if formula == "dunning":
        # Add the term-absent cells for the full 4-cell G².
        c = total_a - a  # other tokens in A
        d = total_b - b  # other tokens in B
        expected_c = total_a - expected_a
        expected_d = total_b - expected_b
        contrib = contrib + (
            xlogy(c, c) - xlogy(c, expected_c) + xlogy(d, d) - xlogy(d, expected_d)
        )
    unsigned = 2.0 * contrib
    # Mathematically LL >= 0; clip away the tiny negative values that
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
