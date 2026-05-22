"""Small statistical utilities used across the package.

Right now this is just the Wilson score interval — pycorpdiff prefers
Wilson over Wald for proportion CIs because Wald collapses badly near
``p = 0`` or ``p = 1``, which is exactly where temporal trajectories of
rare terms live.

Reference
---------
Wilson, E. B. (1927). Probable inference, the law of succession, and
statistical inference. *Journal of the American Statistical
Association*, 22(158), 209-212.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtri


def wilson_ci(
    x: int | npt.NDArray[np.int64],
    n: int | npt.NDArray[np.int64],
    confidence: float = 0.95,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Wilson score interval for a binomial proportion ``x / n``.

    Returns ``(lower, upper)`` as float arrays (or scalars cast to
    0-d arrays). When ``n == 0`` the interval is undefined and both
    bounds come back as NaN. ``confidence`` must be in ``(0, 1)``;
    defaults to 0.95 (z ≈ 1.96).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    # Two-sided z for the requested level: Φ^{-1}((1 + c) / 2).
    z = float(ndtri((1.0 + confidence) / 2.0))

    x_arr = np.asarray(x, dtype=np.float64)
    n_arr = np.asarray(n, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_arr > 0, x_arr / n_arr, np.nan)
        z2 = z * z
        denom = 1.0 + z2 / n_arr
        center = (p + z2 / (2.0 * n_arr)) / denom
        margin = z * np.sqrt(p * (1.0 - p) / n_arr + z2 / (4.0 * n_arr * n_arr)) / denom

    lower = np.where(n_arr > 0, center - margin, np.nan)
    upper = np.where(n_arr > 0, center + margin, np.nan)
    # Clip to [0, 1] — Wilson's interval is bounded mathematically, but
    # roundoff at p ≈ 0 / p ≈ 1 can spill a hair past.
    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)
