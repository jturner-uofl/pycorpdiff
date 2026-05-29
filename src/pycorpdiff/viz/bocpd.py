"""BOCPD diagnostic plot — series + run-length posterior heatmap + MAP line.

Three stacked panels sharing the time axis:

1. The input series with detected changepoints flagged.
2. Heatmap of the run-length posterior P(r_t | data so far) on a
   log colour scale — the canonical BOCPD diagnostic figure.
3. MAP run length over time. Visible drops mark changepoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import altair as alt

    from ..temporal.bocpd import BocpdResult


def bocpd_plot(
    result: BocpdResult,
    *,
    width: int = 660,
    height_per_panel: int = 180,
    max_run_length_shown: int = 40,
    threshold: int = 3,
) -> alt.Chart:
    """Three-panel BOCPD diagnostic chart.

    Parameters
    ----------
    result
        A :class:`BocpdResult`.
    width
        Width of each panel (panels are vertically concat'd).
    height_per_panel
        Height of each panel.
    max_run_length_shown
        Truncate the heatmap at this run length — most posterior mass
        lives in the first few dozen run lengths.
    threshold
        MAP-run-length threshold below which a step is flagged as a
        detected changepoint in panel 1.
    """
    import altair as alt

    # The heatmap panel has T × (max_run_length_shown + 1) cells which
    # can exceed altair's default 5000-row inline-data limit for longer
    # series. Disable the limit on the heatmap dataframe specifically —
    # the data is inline (no external host), so the rendering cost is
    # bounded by the local browser.
    alt.data_transformers.disable_max_rows()

    periods = result.series.index
    if isinstance(periods, pd.PeriodIndex):
        period_axis: pd.Index = pd.Index(periods.to_timestamp())
    else:
        period_axis = pd.Index(periods)

    series_df = pd.DataFrame(
        {"period": period_axis, "value": result.series.to_numpy(dtype=float)}
    )
    flagged_df = result.detected_changepoints(threshold=threshold).copy()
    if len(flagged_df) and isinstance(flagged_df["period"].iloc[0], pd.Period):
        flagged_df["period"] = flagged_df["period"].apply(lambda p: p.to_timestamp())

    # ---------- Panel 1: series + flagged changepoints ----------
    series_line = (
        alt.Chart(series_df)
        .mark_line(strokeWidth=2, color="#0b6e7c")
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("value:Q", title=None),
            tooltip=["period", alt.Tooltip("value:Q", format=".5f")],
        )
    )
    series_points = (
        alt.Chart(series_df)
        .mark_point(filled=True, size=40, color="#0b6e7c")
        .encode(x="period:T", y="value:Q")
    )
    cp_rules = (
        alt.Chart(flagged_df)
        .mark_rule(color="#e63946", strokeDash=[4, 3], opacity=0.7)
        .encode(x="period:T")
    )
    panel1 = (series_line + series_points + cp_rules).properties(
        width=width,
        height=height_per_panel,
        title=alt.TitleParams(
            text=f"Observed series -- {len(flagged_df)} flagged changepoint(s)",
            subtitle=f"red lines: MAP run length <= {threshold} (hazard = {result.hazard})",
        ),
    )

    # ---------- Panel 2: run-length posterior heatmap ----------
    R = np.asarray(result.run_length_posterior)
    R_shown = R[:, : max_run_length_shown + 1]
    # log10 + clip for colour mapping — most posterior values are
    # tiny floats, log compresses the dynamic range.
    with np.errstate(divide="ignore"):
        log_R = np.log10(np.clip(R_shown, 1e-12, 1.0))
    n_t, n_r = log_R.shape
    period_repeat = np.tile(period_axis.to_numpy(), n_r)
    runs = np.repeat(np.arange(n_r), n_t)
    heat_df = pd.DataFrame(
        {
            "period": period_repeat,
            "run_length": runs,
            "log_posterior": log_R.T.ravel(),
        }
    )
    heatmap = (
        alt.Chart(heat_df)
        .mark_rect()
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("run_length:O", title="run length r", sort="descending"),
            color=alt.Color(
                "log_posterior:Q",
                scale=alt.Scale(scheme="viridis", domain=[-6, 0]),
                title="log10 P(r | data)",
            ),
            tooltip=[
                "period",
                "run_length",
                alt.Tooltip("log_posterior:Q", format=".2f"),
            ],
        )
        .properties(
            width=width,
            height=height_per_panel,
            title="Run-length posterior P(r_t | data through t) -- log10 scale",
        )
    )

    # ---------- Panel 3: MAP run length over time ----------
    map_df = pd.DataFrame(
        {
            "period": period_axis,
            "map_run_length": result.map_run_length.to_numpy(dtype=int),
        }
    )
    map_line = (
        alt.Chart(map_df)
        .mark_line(strokeWidth=2, color="#1f7a3e")
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("map_run_length:Q", title="MAP run length"),
            tooltip=["period", "map_run_length"],
        )
    )
    map_points = (
        alt.Chart(map_df)
        .mark_point(filled=True, size=40, color="#1f7a3e")
        .encode(x="period:T", y="map_run_length:Q")
    )
    threshold_rule = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(color="#888", strokeDash=[3, 3], opacity=0.6)
        .encode(y="y:Q")
    )
    panel3 = (map_line + map_points + threshold_rule).properties(
        width=width,
        height=height_per_panel,
        title="MAP run length -- visible drops mark detected changepoints",
    )

    return alt.vconcat(panel1, heatmap, panel3).resolve_scale(x="shared")  # type: ignore[no-any-return]
