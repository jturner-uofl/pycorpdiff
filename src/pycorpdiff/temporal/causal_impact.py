"""Bayesian counterfactual causal impact (Brodersen et al. 2015).

The :func:`interrupted_time_series` module answers "is there a step
discontinuity at this known event?" via segmented OLS. The harder —
and more interesting — question is "what *would* the trajectory have
looked like *without* the event?" That's the counterfactual, and the
gap between observed reality and counterfactual prediction is the
causal effect of the event.

Method: Bayesian structural time series (BSTS) — a state-space model
with a local linear trend fit on the pre-event window, then projected
forward as the counterfactual for the post-event window. Implemented
via :class:`statsmodels.tsa.UnobservedComponents`. The credible
intervals on the pointwise and cumulative effects come from Monte
Carlo simulation against the joint posterior of the state-space
filter — anchored at the end of the pre-event training data and rolled
forward through the post-event horizon.

Reference
---------
Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L.
(2015). Inferring causal impact using Bayesian structural time-series
models. *Annals of Applied Statistics*, 9(1), 247-274.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..results import _table_to_html, _table_to_json

if TYPE_CHECKING:
    import altair as alt


@dataclass(frozen=True)
class CausalImpactResult:
    """Counterfactual causal-impact analysis around a known event.

    Tables (all period-indexed, columns ``period``, ``observed``,
    ``counterfactual``, ``counterfactual_lower``, ``counterfactual_upper``,
    plus the same for the pointwise and cumulative effects).

    Summary scalars live in :attr:`metrics` as a dict keyed by the
    standard CausalImpact reporting set: avg effect, absolute effect,
    relative effect, posterior probability of no effect.
    """

    target: str
    event_date: pd.Timestamp
    table: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    level: float = 0.95
    n_pre: int = 0
    n_post: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_html(self.table, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_json(self.table, path, **kw)

    def plot(self, **kw: Any) -> alt.Chart:
        from ..viz.causal_impact import causal_impact_plot

        return causal_impact_plot(self, **kw)

    def summary(self) -> str:
        import math

        m = self.metrics
        avg = m.get("avg_effect", float("nan"))
        cum = m.get("cumulative_effect", float("nan"))
        rel = m.get("relative_effect", float("nan"))
        ci_lo = m.get("avg_effect_lower", float("nan"))
        ci_hi = m.get("avg_effect_upper", float("nan"))
        p = m.get("posterior_prob_no_effect", float("nan"))
        rel_str = (
            "n/a (counterfactual ≈ 0)" if math.isnan(rel) else f"{rel * 100:+.1f}%"
        )
        return (
            f"CausalImpactResult(target={self.target!r}, "
            f"event={self.event_date.date()}, pre={self.n_pre}, post={self.n_post})\n"
            f"  avg effect:        {avg:+.4f} per period  "
            f"({int(self.level * 100)}% CrI [{ci_lo:+.4f}, {ci_hi:+.4f}])\n"
            f"  cumulative effect: {cum:+.4f}\n"
            f"  relative effect:   {rel_str} vs counterfactual mean\n"
            f"  P(no effect):      {p:.3f}"
        )


def causal_impact(
    series: pd.Series,
    event_date: str | pd.Period | pd.Timestamp,
    *,
    level: float = 0.95,
    n_samples: int = 1000,
    seed: int | None = 0,
    model: str = "local linear trend",
) -> CausalImpactResult:
    """Counterfactual causal impact of an event on a time series.

    Fits a Bayesian structural time-series model on the pre-event
    portion of ``series`` and projects it forward through the
    post-event window as the counterfactual "what would have happened
    without the event". Observed minus counterfactual is the causal
    effect, with credible intervals from Monte Carlo simulation
    against the joint state-space posterior.

    Parameters
    ----------
    series
        Period-indexed time series of values (e.g. relative
        frequencies from :meth:`TemporalTrajectory.over_time`).
    event_date
        Where to place the intervention. Anything pandas can compare
        to the series index.
    level
        Credible-interval level. ``0.95`` → 95% CrI on every band.
    n_samples
        Number of Monte Carlo paths to sample for the pointwise +
        cumulative CIs. 1000 is the conventional default and runs in
        well under a second on typical CL series.
    seed
        RNG seed for reproducibility.
    model
        Trend specification passed to
        :class:`statsmodels.tsa.UnobservedComponents` — usually
        ``"local linear trend"`` (level + slope) or ``"local level"``.

    Returns
    -------
    :class:`CausalImpactResult`
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "causal_impact requires statsmodels. "
            "Install with: pip install 'pycorpdiff[temporal]'"
        ) from exc

    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1); got {level}")
    if n_samples < 100:
        raise ValueError(f"n_samples must be >= 100; got {n_samples}")

    # Period → Timestamp index so the state-space model can use the
    # inferred frequency for forward extension.
    work = series.copy()
    if isinstance(work.index, pd.PeriodIndex):
        work.index = work.index.to_timestamp()
    inferred = getattr(work.index, "inferred_freq", None)
    if inferred is not None and isinstance(work.index, pd.DatetimeIndex):
        # Rebuild a frequency-tagged DatetimeIndex without trying to mutate
        # the read-only ``.freq`` attribute. statsmodels picks up the freq
        # from this for the forecast horizon.
        work.index = pd.DatetimeIndex(work.index, freq=inferred)

    event_ts = _coerce_event(event_date, work)

    pre_mask = np.asarray(work.index < event_ts)
    post_mask = ~pre_mask
    if int(pre_mask.sum()) < 4:
        raise ValueError(
            f"need at least 4 pre-event observations; got {int(pre_mask.sum())} "
            f"(event_date={event_date!r})"
        )
    if int(post_mask.sum()) < 1:
        raise ValueError(
            f"need at least 1 post-event observation; event_date={event_date!r} "
            "is at or after the last period"
        )

    pre = work[pre_mask]
    post = work[post_mask]

    # Fit BSTS on the pre-event window.
    uc = sm.tsa.UnobservedComponents(pre, level=model)
    fit = uc.fit(disp=False)

    # Counterfactual point forecast + marginal CI from get_forecast.
    h = len(post)
    fc = fit.get_forecast(steps=h)
    cf_mean = np.asarray(fc.predicted_mean, dtype=float)
    cf_ci = np.asarray(fc.conf_int(alpha=1.0 - level), dtype=float)
    cf_lower = cf_ci[:, 0]
    cf_upper = cf_ci[:, 1]

    # Monte Carlo against the joint posterior for the cumulative + pointwise
    # effect bands — anchored at the end of the pre-event filter state.
    sim = fit.simulate(
        nsimulations=h,
        repetitions=int(n_samples),
        anchor="end",
        random_state=seed,
    )
    paths = sim.to_numpy() if isinstance(sim, pd.DataFrame) else np.asarray(sim)
    # statsmodels returns shape (h, repetitions); transpose to (N, h).
    paths = paths.T if paths.shape == (h, n_samples) else paths

    obs_arr = post.to_numpy(dtype=float)
    pointwise_paths = obs_arr[None, :] - paths
    cumulative_paths = np.cumsum(pointwise_paths, axis=1)

    alpha = 1.0 - level
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    pw_lower = np.quantile(pointwise_paths, lo_q, axis=0)
    pw_upper = np.quantile(pointwise_paths, hi_q, axis=0)
    cum_mean = cumulative_paths.mean(axis=0)
    cum_lower = np.quantile(cumulative_paths, lo_q, axis=0)
    cum_upper = np.quantile(cumulative_paths, hi_q, axis=0)
    pw_mean = obs_arr - cf_mean

    # Summary scalars — Brodersen "average effect" is over the post window.
    avg_effect = float(pw_mean.mean())
    avg_lower = float(np.quantile(pointwise_paths.mean(axis=1), lo_q))
    avg_upper = float(np.quantile(pointwise_paths.mean(axis=1), hi_q))
    cf_mean_avg = float(cf_mean.mean())
    relative_effect = (
        avg_effect / cf_mean_avg if abs(cf_mean_avg) > 1e-12 else float("nan")
    )
    # Posterior probability of no effect: fraction of paths where the avg
    # effect crosses zero — two-tailed.
    avg_effect_per_path = pointwise_paths.mean(axis=1)
    if avg_effect >= 0:
        p_no_effect = float((avg_effect_per_path <= 0).mean()) * 2.0
    else:
        p_no_effect = float((avg_effect_per_path >= 0).mean()) * 2.0
    p_no_effect = min(1.0, max(0.0, p_no_effect))

    # Build the long table — one row per post-event period. Use the
    # original series' index so callers see Period objects if that's
    # what they passed in.
    if isinstance(series.index, pd.PeriodIndex):
        periods_out: pd.Index = series.index[post_mask]
    else:
        periods_out = work.index[post_mask]
    table = pd.DataFrame(
        {
            "period": periods_out,
            "observed": obs_arr,
            "counterfactual": cf_mean,
            "counterfactual_lower": cf_lower,
            "counterfactual_upper": cf_upper,
            "pointwise_effect": pw_mean,
            "pointwise_lower": pw_lower,
            "pointwise_upper": pw_upper,
            "cumulative_effect": cum_mean,
            "cumulative_lower": cum_lower,
            "cumulative_upper": cum_upper,
        }
    )

    metrics = {
        "avg_effect": avg_effect,
        "avg_effect_lower": avg_lower,
        "avg_effect_upper": avg_upper,
        "cumulative_effect": float(cum_mean[-1]),
        "cumulative_effect_lower": float(cum_lower[-1]),
        "cumulative_effect_upper": float(cum_upper[-1]),
        "relative_effect": relative_effect,
        "counterfactual_mean": cf_mean_avg,
        "posterior_prob_no_effect": p_no_effect,
    }
    return CausalImpactResult(
        target="",  # filled in by the caller — TemporalTrajectory knows the name
        event_date=event_ts,
        table=table,
        metrics=metrics,
        level=level,
        n_pre=int(pre_mask.sum()),
        n_post=int(post_mask.sum()),
        params={"model": model, "n_samples": n_samples, "seed": seed},
    )


def _coerce_event(event: object, series: pd.Series) -> pd.Timestamp:
    """Coerce an event-date argument to a Timestamp aligned with the series."""
    if isinstance(event, pd.Timestamp):
        return event
    if isinstance(event, pd.Period):
        return pd.Timestamp(str(event))
    return pd.Timestamp(str(event))
