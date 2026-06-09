"""Kleinberg burstiness detection on a temporal trajectory.

Where :func:`pycorpdiff.temporal.changepoint.detect_changepoints` and
:func:`pycorpdiff.temporal.bocpd.bocpd` answer "*when* did the rate
change?" (segmentation by changepoint), Kleinberg's 1999 algorithm
answers "*when was the rate elevated, and how strongly?*"
(state-by-state burst-intensity labelling). The two are
complementary; for a term whose discourse spikes around a known
event, both should agree on the spike location, but only Kleinberg
gives a per-period intensity score.

Method
------
The algorithm fits an infinite-state automaton where state *i* has
rate :math:`p_0 \\cdot s^i` (``s > 1`` is the *burst factor*; ``p_0``
is the overall base rate of the term across all periods). Per-period
observation cost is the negative log-Binomial likelihood under state
*i*'s rate; transition cost is :math:`(j - i) \\cdot \\gamma \\cdot
\\log T` when escalating to a higher state and zero when de-escalating
("burst-up is costly, burst-down is free"). A standard Viterbi pass
over the per-period × per-state cost matrix gives the
minimum-cost state sequence. We cap the state count at a finite
``n_states`` for tractability — five is plenty for corpus data, where
state 2 or 3 is already a very strong signal.

Reference
---------
Kleinberg, J. (2003). Bursty and hierarchical structure in streams.
*Data Mining and Knowledge Discovery*, 7(4), 373–397. (Originally
KDD 2002 / longer 1999 manuscript.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import altair as alt


@dataclass(frozen=True)
class BurstinessResult:
    """Per-period burst-state labels plus a per-burst summary table.

    Attributes
    ----------
    states
        Integer Series indexed by period; values in ``0 .. n_states-1``.
        Higher values = higher burst intensity. Zero = baseline.
    bursts
        DataFrame of detected bursts (maximal runs of ``state >= 1``)
        with columns ``start``, ``end``, ``n_periods``, ``max_state``,
        ``total_count``, ``mean_relfreq``. One row per burst.
    target
        The term whose rate was analysed.
    s
        Burst factor used.
    gamma
        Transition-cost multiplier used.
    n_states
        Number of states the DP considered.
    base_rate
        Overall :math:`p_0` (total counts / total tokens across all
        periods).
    trajectory_table
        The full period-level table used by ``plot()``. Carries the
        original ``period``, ``relfreq``, and joined ``state`` columns.
    """

    states: pd.Series
    bursts: pd.DataFrame
    target: str
    s: float
    gamma: float
    n_states: int
    base_rate: float
    trajectory_table: pd.DataFrame

    def to_df(self) -> pd.DataFrame:
        return self.bursts.copy()

    def summary(self) -> str:
        n_bursts = len(self.bursts)
        if n_bursts == 0:
            return (
                f"BurstinessResult(target={self.target!r}, no bursts detected; "
                f"base rate p_0 = {self.base_rate:.2e})"
            )
        max_state = int(self.bursts["max_state"].max())
        return (
            f"BurstinessResult(target={self.target!r}, "
            f"{n_bursts} burst(s), max intensity = state {max_state}, "
            f"base rate p_0 = {self.base_rate:.2e})"
        )

    def __repr__(self) -> str:
        return self.summary()

    def plot(self, **kw: Any) -> alt.Chart:
        """Plot the trajectory with burst-state bands overlaid."""
        from ..viz.burstiness import burstiness_plot

        return burstiness_plot(self.trajectory_table, target=self.target, **kw)


def kleinberg_bursts(
    counts: np.ndarray | pd.Series,
    totals: np.ndarray | pd.Series,
    *,
    s: float = 2.0,
    gamma: float = 1.0,
    n_states: int = 5,
) -> np.ndarray:
    """Run Kleinberg's two-parameter burst-detection algorithm.

    Returns the optimal state sequence (one state per period) under
    the cost model described in this module's docstring.

    Parameters
    ----------
    counts
        Per-period count of the target term. Non-negative integer.
    totals
        Per-period total token count (denominator). Strictly positive.
    s
        Burst factor. State *i* has rate :math:`p_0 \\cdot s^i`. Default
        ``2.0`` matches Kleinberg's recommendation. Must be ``> 1``.
    gamma
        Transition-cost multiplier. Higher ``gamma`` makes the algorithm
        more conservative (fewer / shorter bursts). Default ``1.0``.
    n_states
        Maximum number of states (including state 0). Default ``5`` is
        more than enough for corpus-linguistic data; rarely do real
        corpora populate states beyond 2 or 3.

    Returns
    -------
    numpy.ndarray
        Length-T array of integer state labels.

    Raises
    ------
    ValueError
        On obviously-degenerate inputs (mismatched lengths, empty
        series, all-zero totals, ``s <= 1``, ``n_states < 2``).
    """
    if s <= 1.0:
        raise ValueError(f"s must be > 1 (the burst factor); got {s}")
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2; got {n_states}")
    c = np.asarray(counts, dtype=np.float64).ravel()
    r = np.asarray(totals, dtype=np.float64).ravel()
    if c.shape != r.shape:
        raise ValueError(
            f"counts and totals must have the same length; got {c.shape} vs {r.shape}"
        )
    t_n = int(c.size)
    if t_n == 0:
        raise ValueError("counts/totals must be non-empty")
    if (r <= 0).any():
        raise ValueError(
            "every period must have a positive denominator (totals); "
            "drop empty periods before calling"
        )
    if (c < 0).any():
        raise ValueError("counts must be non-negative")
    if c.sum() == 0:
        # No events ever — every period is state 0 by construction.
        return np.zeros(t_n, dtype=np.int64)

    base_rate = c.sum() / r.sum()
    # State i has rate p_0 * s^i, capped just below 1 so the Binomial
    # log-likelihood stays finite for the highest state.
    state_rates = np.minimum(base_rate * (s ** np.arange(n_states)), 0.999_999)

    # Per-(period, state) observation cost = -log Binomial(c | r, p_i),
    # dropping the period-only log-binom-coefficient term (doesn't
    # affect the argmin over states).
    log_p = np.log(state_rates)[None, :]              # shape (1, n_states)
    log_1mp = np.log1p(-state_rates)[None, :]
    obs_cost = -(c[:, None] * log_p + (r - c)[:, None] * log_1mp)  # (T, n_states)

    # Transition cost τ(j → i) = (i - j) * γ * log(T) if i > j else 0.
    log_T = float(np.log(max(t_n, 2)))
    trans_up = np.maximum(
        np.arange(n_states)[None, :] - np.arange(n_states)[:, None], 0.0
    ) * gamma * log_T  # (n_states, n_states); rows = previous, cols = next

    # Viterbi DP.
    V = np.empty((t_n, n_states), dtype=np.float64)
    backptr = np.empty((t_n, n_states), dtype=np.int64)
    # Initial costs: starting in state i pays the i-step ramp from 0.
    init_ramp = np.arange(n_states, dtype=np.float64) * gamma * log_T
    V[0] = obs_cost[0] + init_ramp
    backptr[0] = -1
    for t in range(1, t_n):
        # For each next state i, find the previous state j minimising
        # V[t-1, j] + τ(j → i).
        cand = V[t - 1][:, None] + trans_up  # (n_states, n_states)
        best_prev = np.argmin(cand, axis=0)
        V[t] = obs_cost[t] + cand[best_prev, np.arange(n_states)]
        backptr[t] = best_prev

    # Backtrack the optimal state sequence.
    states = np.empty(t_n, dtype=np.int64)
    states[-1] = int(np.argmin(V[-1]))
    for t in range(t_n - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]
    return states


def _summarise_bursts(
    states: np.ndarray,
    periods: pd.Index,
    counts: np.ndarray,
    rates: np.ndarray,
) -> pd.DataFrame:
    """Collapse the per-period state sequence into one row per burst.

    A "burst" is a maximal contiguous run with state >= 1.
    """
    rows: list[dict[str, Any]] = []
    in_burst = False
    start_idx = 0
    for t in range(len(states)):
        if states[t] >= 1 and not in_burst:
            in_burst = True
            start_idx = t
        if in_burst and (t == len(states) - 1 or states[t + 1] == 0):
            # End-of-burst (or end-of-series with state still high).
            end_idx = t
            rows.append(
                {
                    "start": periods[start_idx],
                    "end": periods[end_idx],
                    "n_periods": int(end_idx - start_idx + 1),
                    "max_state": int(states[start_idx : end_idx + 1].max()),
                    "total_count": int(counts[start_idx : end_idx + 1].sum()),
                    "mean_relfreq": float(rates[start_idx : end_idx + 1].mean()),
                }
            )
            in_burst = False
    if not rows:
        return pd.DataFrame(
            columns=["start", "end", "n_periods", "max_state", "total_count", "mean_relfreq"]
        )
    return pd.DataFrame(rows)


def burstiness_from_table(
    table: pd.DataFrame,
    target: str,
    *,
    s: float = 2.0,
    gamma: float = 1.0,
    n_states: int = 5,
) -> BurstinessResult:
    """Compute burstiness from a :class:`TemporalTrajectory`-shape table.

    Expects the canonical columns ``period``, ``term``, ``count``, and
    ``total`` (the latter being the per-period token denominator).
    Used internally by :meth:`TemporalTrajectory.burstiness`; exposed
    here for advanced users who want to bypass the trajectory layer.
    """
    if "total" not in table.columns:
        raise ValueError(
            "table must have a 'total' column (per-period token total). "
            "Use TemporalTrajectory.burstiness() on a track().over_time() result."
        )
    sub = table[table["term"] == target].sort_values("period").reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"target {target!r} not present in trajectory")
    counts_arr = sub["count"].to_numpy(dtype=np.int64)
    totals_arr = sub["total"].to_numpy(dtype=np.int64)
    states = kleinberg_bursts(
        counts_arr, totals_arr, s=s, gamma=gamma, n_states=n_states
    )
    rates_arr = (
        sub["relfreq"].to_numpy(dtype=np.float64)
        if "relfreq" in sub.columns
        else counts_arr / np.maximum(totals_arr, 1)
    )
    periods = pd.Index(sub["period"].to_list(), name="period")
    bursts_df = _summarise_bursts(states, periods, counts_arr, rates_arr)
    state_series = pd.Series(states, index=periods, name="state")

    # Trajectory table with the state joined on; used by .plot().
    traj = sub.copy()
    traj["state"] = states
    return BurstinessResult(
        states=state_series,
        bursts=bursts_df,
        target=target,
        s=s,
        gamma=gamma,
        n_states=n_states,
        base_rate=(
            float(counts_arr.sum() / totals_arr.sum())
            if totals_arr.sum() > 0
            else 0.0
        ),
        trajectory_table=traj,
    )
