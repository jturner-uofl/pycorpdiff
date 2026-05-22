"""Dispersion measures for corpus-comparison sanity checks.

A term can be "key" (significant + large effect) simply because one
document overuses it. Reporting dispersion alongside keyness lets the
caller filter out these spurious findings.

References
----------
Juilland, A., & Chang-Rodríguez, E. (1964). *Frequency Dictionary of
Spanish Words*. Mouton.

Gries, S. Th. (2008). Dispersions and adjusted frequencies in corpora.
*International Journal of Corpus Linguistics*, 13(4), 403-437.
"""

from __future__ import annotations

import pandas as pd


def juilland_d(doc_term_matrix: pd.DataFrame) -> pd.Series:
    """Juilland's D — a 0..1 dispersion score; higher is more even."""
    raise NotImplementedError("juilland_d() lands in Phase 1")


def dispersion_dp(doc_term_matrix: pd.DataFrame) -> pd.Series:
    """Gries's DP (Deviation of Proportions) — 0..1; lower is more even."""
    raise NotImplementedError("dispersion_dp() lands in Phase 1")
