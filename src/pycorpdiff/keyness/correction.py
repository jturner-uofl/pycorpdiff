"""Multiple-comparison correction for keyness *p*-value vectors.

References
----------
Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
rate: A practical and powerful approach to multiple testing. *Journal
of the Royal Statistical Society: Series B*, 57(1), 289-300.
(BH-adjusted *p*-values; the FDR control used by default.)

Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo
delle probabilità. *Pubblicazioni del R Istituto Superiore di Scienze
Economiche e Commerciali di Firenze*, 8, 3-62. (Family-wise correction;
opt-in via ``multiple_comparisons="bonferroni"``.)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def benjamini_hochberg(pvals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return Benjamini–Hochberg–adjusted *p*-values.

    For each non-NaN input *p*, the adjusted value is the minimum over
    the rank-cumulative ``p_(k) * m / k`` from that rank rightward,
    clipped to ``[0, 1]``, where ``m`` is the count of non-NaN values.
    Order of the input is preserved.

    **NaN handling.** ``NaN`` p-values are excluded from the ranking and
    pass through to the output as ``NaN``. ``m`` is the number of
    non-NaN inputs. Previously (≤ 0.1.0a26) a single ``NaN`` silently
    propagated to *every* output position; surfaced as iter-3 audit
    finding H.1.
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    n = pvals.size
    if n == 0:
        return pvals
    valid_mask = ~np.isnan(pvals)
    m = int(valid_mask.sum())
    out = np.full(n, np.nan, dtype=np.float64)
    if m == 0:
        return out
    valid_pvals = pvals[valid_mask]
    order = np.argsort(valid_pvals)
    ranks = np.arange(1, m + 1)
    raw = valid_pvals[order] * m / ranks
    # Cumulative minimum from the right enforces monotonicity.
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = monotone
    out[valid_mask] = adjusted
    return out


def bonferroni(pvals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Bonferroni-corrected *p*-values: ``min(p * m, 1)`` elementwise.

    **NaN handling.** ``NaN`` inputs pass through to ``NaN`` outputs.
    ``m`` is the number of non-NaN inputs (the count of *actual* tests
    performed), matching the BH NaN-handling convention.
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    valid_mask = ~np.isnan(pvals)
    m = int(valid_mask.sum())
    out = np.full(pvals.shape, np.nan, dtype=np.float64)
    if m == 0:
        return out
    out[valid_mask] = np.clip(pvals[valid_mask] * m, 0.0, 1.0)
    return out
