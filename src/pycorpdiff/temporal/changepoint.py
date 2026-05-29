"""Changepoint detection on temporal frequency / similarity series.

Wraps the ``ruptures`` library; importable only when the ``temporal``
extra is installed.

References
----------
Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of
offline change point detection methods. *Signal Processing*, 167,
107299.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

ChangepointMethod = Literal["pelt", "binseg", "window"]


def detect_changepoints(
    series: pd.Series,
    method: ChangepointMethod = "pelt",
    penalty: float | None = None,
    model: str = "rbf",
) -> pd.DataFrame:
    """Detect changepoints in a temporal value series.

    Parameters
    ----------
    series
        A 1-D series indexed by period (or any orderable). Index values
        propagate into the output so changepoints can be reported in
        their original time vocabulary.
    method
        ``"pelt"`` (default), ``"binseg"``, or ``"window"`` — the three
        offline algorithms ruptures exposes. PELT is exact and usually
        best; BinSeg is greedy but faster on long series; window scans
        a fixed window across the signal.
    penalty
        Penalty term for adding a changepoint. Larger = fewer
        changepoints. If ``None``, defaults to ``log(n)`` (BIC-style).
        Pass an explicit float to tune sensitivity.
    model
        Cost function: ``"rbf"`` (default), ``"l1"``, ``"l2"``,
        ``"normal"``. See ruptures documentation for the full list.

    Returns
    -------
    pandas.DataFrame
        Columns ``period`` (index value at the breakpoint), ``index``
        (positional index), and ``method`` (echo of the chosen method).
    """
    try:
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "detect_changepoints requires ruptures. "
            "Install with: pip install 'pycorpdiff[temporal]'"
        ) from exc

    values = np.asarray(series.to_numpy(dtype=float))
    if values.ndim != 1:
        raise ValueError("series must be 1-D")
    if len(values) < 4:
        raise ValueError(f"need at least 4 observations; got {len(values)}")
    if np.isnan(values).any():
        raise ValueError("series contains NaN values; impute or drop them first")

    pen = float(np.log(len(values))) if penalty is None else float(penalty)

    if method == "pelt":
        algo = rpt.Pelt(model=model)
    elif method == "binseg":
        algo = rpt.Binseg(model=model)
    elif method == "window":
        algo = rpt.Window(model=model)
    else:  # pragma: no cover - typing makes this unreachable
        raise ValueError(f"unknown method={method!r}")

    algo.fit(values.reshape(-1, 1))
    # ruptures returns endpoint indices including len(values) as the
    # final "boundary"; drop it to keep only interior changepoints.
    breakpoints = [bp for bp in algo.predict(pen=pen) if bp < len(values)]

    rows = [
        {"period": series.index[bp], "index": bp, "method": method}
        for bp in breakpoints
    ]
    return pd.DataFrame(rows, columns=["period", "index", "method"])
