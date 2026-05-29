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
