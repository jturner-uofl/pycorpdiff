"""Multiple-comparison correction for keyness *p*-value vectors."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def benjamini_hochberg(pvals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return Benjamini–Hochberg–adjusted *p*-values.

    For each input *p*, the adjusted value is the minimum over the
    rank-cumulative ``p_(k) * n / k`` from that rank rightward, clipped
    to ``[0, 1]``. Order of the input is preserved.
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    n = pvals.size
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    raw = pvals[order] * n / ranks
    # Cumulative minimum from the right enforces monotonicity.
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    out = np.empty(n, dtype=np.float64)
    out[order] = monotone
    return out


def bonferroni(pvals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Bonferroni-corrected *p*-values: ``min(p * n, 1)`` elementwise."""
    pvals = np.asarray(pvals, dtype=np.float64)
    return np.clip(pvals * pvals.size, 0.0, 1.0)
