"""Bayesian online changepoint detection (Adams & MacKay 2007).

Where :func:`detect_changepoints` runs PELT *offline* — needing the full
series and returning MAP locations after the fact — BOCPD runs *online*:
at each new observation it returns the posterior distribution over *run
length*, where the run length is the number of periods since the last
changepoint. The two actionable summaries:

- **MAP run length** at time *t*: the most-probable number of periods
  since the last changepoint. Drops sharply at changepoints, grows
  steadily during stable regimes. This is the field to monitor.
- **Posterior probability of recent changepoint**:
  P(*r_t* ≤ ``threshold`` | data through *t*) — sum the leftmost
  ``threshold + 1`` columns of ``run_length_posterior``. This IS
  data-driven and useful for live monitoring.

The :class:`BocpdResult` also exposes a ``cp_probability`` field
(``posterior[t, 0]``), but under constant hazard ``h`` and proper
predictive-probability conditioning that value collapses to ``h``
identically — the changepoint prior cancels in the normalisation.
Use the MAP run length or the threshold-mass formulation above for
data-driven monitoring; treat ``cp_probability`` as the hazard
hyperparameter rather than a posterior signal.

Observation model: Gaussian with a Normal-Inverse-Gamma conjugate
prior over (mean, variance). The predictive distribution is Student's
*t*, computed analytically. Note: NIG is *misspecified* for series
bounded in [0, 1] (relative frequencies); for proportion-trajectory
applications consider logit-transforming the input or using
:func:`changepoints` (offline PELT) on the rate column instead.

Reference
---------
Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint
detection. arXiv:0710.3742.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ..results import _table_to_html, _table_to_json

if TYPE_CHECKING:
    import altair as alt


@dataclass(frozen=True)
class BocpdResult:
    """Output of :func:`bocpd`.

    The full ``run_length_posterior`` matrix is retained for plotting
    (the canonical BOCPD diagnostic is a heatmap of this matrix). The
    derived summaries — ``map_run_length`` and ``cp_probability`` — are
    the actionable per-step signals.
    """

    series: pd.Series
    run_length_posterior: np.ndarray  # shape (T, max_r+1)
    map_run_length: pd.Series  # length T, indexed like series
    cp_probability: pd.Series  # length T; under constant hazard this collapses to the hazard hyperparameter — see module docstring
    hazard: float
    params: dict[str, Any] = field(default_factory=dict)

    def cp_probability_recent(self, threshold: int = 3) -> pd.Series:
        """Posterior probability of a changepoint in the last ``threshold`` periods.

        Sums the leftmost ``threshold + 1`` columns of
        :attr:`run_length_posterior` — i.e. P(*r_t* ≤ threshold | data).
        Unlike :attr:`cp_probability` (which collapses to the hazard
        hyperparameter under constant hazard), this signal responds to
        the data and is the right monitoring metric for live-detection
        applications.
        """
        cols = min(threshold + 1, self.run_length_posterior.shape[1])
        recent = self.run_length_posterior[:, :cols].sum(axis=1)
        return pd.Series(recent, index=self.series.index, name="cp_probability_recent")

    def to_df(self) -> pd.DataFrame:
        """Return a flat per-step table (period, value, map_run_length, cp_probability)."""
        return pd.DataFrame(
            {
                "period": self.series.index,
                "value": self.series.to_numpy(dtype=float),
                "map_run_length": self.map_run_length.to_numpy(dtype=int),
                "cp_probability": self.cp_probability.to_numpy(dtype=float),
            }
        )

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_html(self.to_df(), path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_json(self.to_df(), path, **kw)

    def detected_changepoints(self, *, threshold: int = 3) -> pd.DataFrame:
        """Periods where the MAP run length dropped below ``threshold``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``period``, ``map_run_length``, ``cp_probability``.
        """
        df = self.to_df()
        flagged = df[df["map_run_length"] <= threshold].copy()
        return flagged.reset_index(drop=True)

    def plot(self, **kw: Any) -> alt.Chart:
        from ..viz.bocpd import bocpd_plot

        return bocpd_plot(self, **kw)

    def summary(self) -> str:
        detected = self.detected_changepoints(threshold=3)
        return (
            f"BocpdResult(hazard={self.hazard}, T={len(self.series)}, "
            f"detected={len(detected)})"
        )


def bocpd(
    series: pd.Series,
    *,
    hazard: float = 0.01,
    mu_0: float | None = None,
    kappa_0: float = 1.0,
    alpha_0: float = 1.0,
    beta_0: float | None = None,
    max_run_length: int | None = None,
) -> BocpdResult:
    """Bayesian Online Changepoint Detection.

    Parameters
    ----------
    series
        Time-indexed series of scalar observations.
    hazard
        Constant hazard rate — probability of a changepoint at any
        given step. Equivalent to an expected mean run length of
        ``1/hazard``. ``0.01`` (default) → expect a regime change every
        ~100 periods. For monthly data, ``hazard=1/24`` → "every 2
        years".
    mu_0
        Prior mean. Defaults to the series mean, but for serious work
        pass a sensible scale-anchored prior (e.g. zero for centered
        rates).
    kappa_0
        Prior pseudo-count on the mean. ``1.0`` is weakly informative.
    alpha_0, beta_0
        Inverse-Gamma hyperparameters on the variance. ``beta_0``
        defaults to a small data-derived value; override if you have
        scale information from outside the data window.
    max_run_length
        Truncate the run-length state at this many steps. Defaults to
        the full series length. Setting it (e.g. ``200``) bounds the
        per-step cost at O(T·max_run_length) for very long series.

    Returns
    -------
    :class:`BocpdResult`
    """
    if hazard <= 0 or hazard >= 1:
        raise ValueError(f"hazard must be in (0, 1); got {hazard}")
    if kappa_0 <= 0 or alpha_0 <= 0:
        raise ValueError(
            f"kappa_0 and alpha_0 must be positive; got {kappa_0}, {alpha_0}"
        )

    y = series.to_numpy(dtype=float)
    n = len(y)
    if n < 2:
        raise ValueError(f"need at least 2 observations; got {n}")

    if mu_0 is None:
        mu_0 = float(np.mean(y))
    if beta_0 is None:
        # Weakly informative: encode the data's order-of-magnitude
        # variance so the t-distribution scale is sensible.
        beta_0 = max(float(np.var(y, ddof=1)) * (alpha_0 - 0.5) if alpha_0 > 0.5 else 0.01, 1e-6)
    if beta_0 <= 0:
        raise ValueError(f"beta_0 must be positive; got {beta_0}")

    max_r = n if max_run_length is None else min(n, int(max_run_length))

    # Posterior matrix and per-run-length sufficient stats.
    posterior = np.zeros((n + 1, max_r + 1))
    posterior[0, 0] = 1.0
    mu_vec = np.array([mu_0], dtype=float)
    kappa_vec = np.array([kappa_0], dtype=float)
    alpha_vec = np.array([alpha_0], dtype=float)
    beta_vec = np.array([beta_0], dtype=float)

    for t in range(n):
        x = y[t]

        # Truncate state at max_r entries.
        if len(mu_vec) > max_r:
            mu_vec = mu_vec[:max_r]
            kappa_vec = kappa_vec[:max_r]
            alpha_vec = alpha_vec[:max_r]
            beta_vec = beta_vec[:max_r]
        n_r = len(mu_vec)

        # 1. Predictive probability for x under each run length.
        scale_vec = np.sqrt(beta_vec * (kappa_vec + 1.0) / (alpha_vec * kappa_vec))
        df_vec = 2.0 * alpha_vec
        pi_vec = student_t.pdf(x, df=df_vec, loc=mu_vec, scale=scale_vec)

        # 2. Growth mass (no changepoint) — shifts up by one r.
        growth = posterior[t, :n_r] * pi_vec * (1.0 - hazard)
        # 3. Changepoint mass (everything collapses to r=0).
        cp_mass = float((posterior[t, :n_r] * pi_vec * hazard).sum())

        next_len = min(n_r + 1, max_r + 1)
        posterior[t + 1, 0] = cp_mass
        if next_len > 1:
            posterior[t + 1, 1:next_len] = growth[: next_len - 1]
        total = posterior[t + 1, :].sum()
        if total > 0:
            posterior[t + 1, :] /= total

        # 4. Update sufficient statistics — shift up with prior in slot 0.
        new_mu = (kappa_vec * mu_vec + x) / (kappa_vec + 1.0)
        new_kappa = kappa_vec + 1.0
        new_alpha = alpha_vec + 0.5
        new_beta = beta_vec + (kappa_vec * (x - mu_vec) ** 2) / (
            2.0 * (kappa_vec + 1.0)
        )
        mu_vec = np.concatenate([[mu_0], new_mu])
        kappa_vec = np.concatenate([[kappa_0], new_kappa])
        alpha_vec = np.concatenate([[alpha_0], new_alpha])
        beta_vec = np.concatenate([[beta_0], new_beta])

    posterior_obs = posterior[1:, : max_r + 1]
    map_r = posterior_obs.argmax(axis=1).astype(int)
    cp_prob = posterior_obs[:, 0]

    return BocpdResult(
        series=series,
        run_length_posterior=posterior_obs,
        map_run_length=pd.Series(map_r, index=series.index, name="map_run_length"),
        cp_probability=pd.Series(
            cp_prob, index=series.index, name="cp_probability"
        ),
        hazard=hazard,
        params={
            "mu_0": mu_0,
            "kappa_0": kappa_0,
            "alpha_0": alpha_0,
            "beta_0": beta_0,
            "max_run_length": max_r,
        },
    )
