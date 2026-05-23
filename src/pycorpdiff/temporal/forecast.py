"""Forward prediction for :class:`TemporalTrajectory`.

The other temporal modules answer *retrospective* questions —
"when did things change", "was there a step at this known event".
This one answers the *predictive* one: where is the trajectory
heading, and what's the uncertainty around that?

Method: state-space exponential smoothing (Hyndman et al. 2008) via
``statsmodels.tsa.exponential_smoothing.ets.ETSModel``. Default
``method="auto"`` selects ETS for series of length ≥ 8 and falls
back to a Holt linear-trend model for shorter histories where ETS
overfits. Rates are forecast on the logit scale and back-transformed
so prediction intervals are pinned to ``[0, 1]`` instead of admitting
nonsensical negative frequencies.

Prediction intervals are the analytical PIs that come out of the
state-space fit at the requested ``level``; for short series they
will be appropriately wide. Honest uncertainty beats false precision.

Reference
---------
Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
*Forecasting with Exponential Smoothing: The State Space Approach*.
Springer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ..results import _table_to_html, _table_to_json

if TYPE_CHECKING:
    import altair as alt

ForecastMethod = Literal["auto", "ets", "holt"]


@dataclass(frozen=True)
class ForecastResult:
    """Forward prediction of a :class:`TemporalTrajectory`.

    Carries both the original history (so :meth:`plot` can render the
    solid-then-dashed continuation in a single chart) and the forecast
    table with point estimates and prediction intervals.
    """

    history: pd.DataFrame
    forecast: pd.DataFrame
    targets: list[str]
    freq: str
    horizon: int
    level: float
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        """Return just the forecast table (point + PI per term × period)."""
        return self.forecast.copy()

    def to_combined(self) -> pd.DataFrame:
        """Return history + forecast stacked, with a ``kind`` column
        distinguishing the two."""
        h = self.history.copy()
        h["kind"] = "observed"
        h = h.rename(columns={"relfreq": "point"})[
            ["period", "term", "point", "ci_lower", "ci_upper", "kind"]
        ]
        f = self.forecast.copy()
        f["kind"] = "forecast"
        return pd.concat([h, f[h.columns]], ignore_index=True)

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the forecast table as HTML."""
        return _table_to_html(self.forecast, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the forecast table as JSON."""
        return _table_to_json(self.forecast, path, **kw)

    def plot(self, **kw: Any) -> alt.Chart:
        """Layered altair chart: solid observed + dashed forecast.

        The history portion shows the existing Wilson CI band; the
        forecast portion shows the prediction-interval band at the
        chosen ``level``.
        """
        from ..viz.forecast import forecast_plot

        return forecast_plot(
            history=self.history, forecast=self.forecast, **kw
        )

    def summary(self) -> str:
        return (
            f"ForecastResult(targets={self.targets!r}, freq={self.freq!r}, "
            f"horizon={self.horizon}, level={self.level}, method={self.method})"
        )


def forecast_trajectory(
    trajectory_table: pd.DataFrame,
    *,
    targets: Sequence[str] | None = None,
    horizon: int = 4,
    level: float = 0.95,
    method: ForecastMethod = "auto",
    logit_transform: bool = True,
) -> pd.DataFrame:
    """Forecast every (or selected) target term's trajectory ``horizon``
    periods forward.

    Parameters
    ----------
    trajectory_table
        The DataFrame attached to a :class:`TemporalTrajectory` —
        must carry ``period``, ``term``, ``relfreq`` columns at minimum.
    targets
        Which terms to forecast. ``None`` (default) forecasts every
        term present in the table.
    horizon
        Number of periods to extrapolate.
    level
        Prediction-interval level. ``0.95`` → 95% PI.
    method
        ``"auto"`` (default) picks ETS for series of length ≥ 8, Holt
        otherwise. ``"ets"`` or ``"holt"`` force a specific method.
    logit_transform
        If ``True`` (default), forecast on the logit scale and
        back-transform so the prediction intervals stay in ``[0, 1]``.
        Disable only if forecasting something other than a rate.

    Returns
    -------
    pandas.DataFrame
        Columns: ``period``, ``term``, ``point``, ``ci_lower``,
        ``ci_upper``. One row per (target, future period). Period
        values match the freq of the input trajectory.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1); got {level}")

    if targets is None:
        target_list = list(trajectory_table["term"].unique())
    else:
        target_list = list(targets)
        missing = set(target_list) - set(trajectory_table["term"].unique())
        if missing:
            raise ValueError(
                f"unknown targets: {sorted(missing)!r}; "
                f"trajectory carries {sorted(trajectory_table['term'].unique())!r}"
            )

    rows: list[pd.DataFrame] = []
    for term in target_list:
        sub = (
            trajectory_table[trajectory_table["term"] == term]
            .sort_values("period")
            .set_index("period")["relfreq"]
        )
        if sub.isna().any():
            sub = sub.dropna()
        if len(sub) < 4:
            raise ValueError(
                f"need at least 4 observations to forecast {term!r}; "
                f"got {len(sub)}"
            )
        fc = _forecast_one(
            sub,
            horizon=horizon,
            level=level,
            method=method,
            logit_transform=logit_transform,
        )
        fc.insert(1, "term", term)
        rows.append(fc)

    return pd.concat(rows, ignore_index=True)


def _forecast_one(
    series: pd.Series,
    *,
    horizon: int,
    level: float,
    method: ForecastMethod,
    logit_transform: bool,
) -> pd.DataFrame:
    """Forecast a single univariate rate series.

    Returns a DataFrame with columns ``period``, ``point``,
    ``ci_lower``, ``ci_upper``. Period values are constructed from the
    input series's PeriodIndex.
    """
    try:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        from statsmodels.tsa.holtwinters import Holt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "forecast() requires statsmodels. "
            "Install with: pip install 'pycorpdiff[temporal]'"
        ) from exc

    # statsmodels prefers a DatetimeIndex with a freq for ETS — convert.
    ts_series = series.copy()
    if isinstance(series.index, pd.PeriodIndex):
        period_freq = series.index.freq
        ts_series.index = series.index.to_timestamp()
    else:
        period_freq = None

    if logit_transform:
        eps = 1e-6
        y_arr = np.clip(ts_series.to_numpy(dtype=float), eps, 1.0 - eps)
        z = pd.Series(
            np.log(y_arr / (1.0 - y_arr)),
            index=ts_series.index,
            name=ts_series.name,
        )
    else:
        z = ts_series.astype(float)

    chosen = method
    if chosen == "auto":
        chosen = "ets" if len(z) >= 8 else "holt"

    alpha = 1.0 - level

    if chosen == "ets":
        model = ETSModel(
            z, error="add", trend="add", seasonal=None
        ).fit(disp=False)
        pred = model.get_prediction(start=len(z), end=len(z) + horizon - 1)
        summary = pred.summary_frame(alpha=alpha)
        z_point = summary["mean"].to_numpy(dtype=float)
        z_lower = summary["pi_lower"].to_numpy(dtype=float)
        z_upper = summary["pi_upper"].to_numpy(dtype=float)
    elif chosen == "holt":
        holt = Holt(z, initialization_method="estimated").fit()
        # Holt has no analytical PI in older statsmodels — use
        # forecast_int via simulate-based intervals.
        z_point = np.asarray(holt.forecast(steps=horizon), dtype=float)
        # Residual-based PI (Hyndman §6.4): std of fitted residuals
        # widens with sqrt(h) under the additive-error assumption.
        resid_std = float(np.std(holt.resid, ddof=1))
        from scipy.stats import norm

        crit = float(norm.ppf(1.0 - alpha / 2.0))
        h_steps = np.arange(1, horizon + 1, dtype=float)
        widening = crit * resid_std * np.sqrt(h_steps)
        z_lower = z_point - widening
        z_upper = z_point + widening
    else:
        raise ValueError(
            f"unknown method={chosen!r}; expected 'auto', 'ets', or 'holt'"
        )

    if logit_transform:
        point = _inv_logit(z_point)
        lower = _inv_logit(z_lower)
        upper = _inv_logit(z_upper)
    else:
        point = z_point
        lower = z_lower
        upper = z_upper

    # Future periods — match original index type.
    if period_freq is not None and isinstance(series.index, pd.PeriodIndex):
        last_period = series.index[-1]
        future_periods = pd.period_range(
            start=last_period + 1, periods=horizon, freq=period_freq
        )
        future_idx: pd.Index = pd.Index(future_periods)
    else:
        last_ts = series.index[-1]
        inferred = getattr(series.index, "inferred_freq", None) or "D"
        offset = pd.tseries.frequencies.to_offset(inferred)
        future_idx = pd.date_range(
            start=last_ts + offset, periods=horizon, freq=offset
        )

    return pd.DataFrame(
        {
            "period": future_idx,
            "point": point,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    )


def _inv_logit(x: np.ndarray) -> np.ndarray:
    """Numerically-stable inverse logit."""
    out: np.ndarray = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
    return out
