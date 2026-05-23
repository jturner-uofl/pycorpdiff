"""Forecast plot — solid history continues into dashed forecast.

Visual grammar: the same Wilson-CI band + line + points the trajectory
plot already uses for observed periods, then a *dashed* line + lighter
prediction-interval band for the forecast horizon. The visual handoff
between the two regions is what makes the chart read "this is the
observed history, this is what we project forward".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import altair as alt


def forecast_plot(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    width: int = 600,
    height: int = 320,
) -> alt.Chart:
    """Layered plot: observed Wilson-CI trajectory + dashed forecast band.

    Parameters
    ----------
    history
        The trajectory table — must carry ``period``, ``term``,
        ``relfreq``, ``ci_lower``, ``ci_upper``.
    forecast
        The forecast table — must carry ``period``, ``term``,
        ``point``, ``ci_lower``, ``ci_upper``.
    width, height
        Canvas dimensions.
    """
    import altair as alt

    h = history.copy()
    f = forecast.copy()
    if isinstance(h["period"].iloc[0], pd.Period):
        h["period"] = h["period"].apply(lambda p: p.to_timestamp())
    if len(f) and isinstance(f["period"].iloc[0], pd.Period):
        f["period"] = f["period"].apply(lambda p: p.to_timestamp())

    base_h = alt.Chart(h).encode(
        x=alt.X("period:T", title=None),
        color=alt.Color("term:N", title=None),
    )
    history_band = base_h.mark_area(opacity=0.18).encode(
        y=alt.Y("ci_lower:Q", title="Relative frequency"),
        y2="ci_upper:Q",
    )
    history_line = base_h.mark_line(strokeWidth=2).encode(
        y="relfreq:Q",
        tooltip=[
            "period",
            "term",
            "count",
            "total",
            alt.Tooltip("relfreq:Q", format=".5f"),
            alt.Tooltip("ci_lower:Q", format=".5f"),
            alt.Tooltip("ci_upper:Q", format=".5f"),
        ],
    )
    history_points = base_h.mark_point(filled=True, size=50).encode(
        y="relfreq:Q",
    )

    base_f = alt.Chart(f).encode(
        x=alt.X("period:T", title=None),
        color=alt.Color("term:N", title=None),
    )
    forecast_band = base_f.mark_area(opacity=0.12).encode(
        y=alt.Y("ci_lower:Q"),
        y2="ci_upper:Q",
    )
    forecast_line = base_f.mark_line(strokeDash=[6, 4], strokeWidth=2).encode(
        y="point:Q",
        tooltip=[
            "period",
            "term",
            alt.Tooltip("point:Q", format=".5f"),
            alt.Tooltip("ci_lower:Q", format=".5f"),
            alt.Tooltip("ci_upper:Q", format=".5f"),
        ],
    )
    forecast_points = base_f.mark_point(
        filled=False, strokeWidth=2, size=50
    ).encode(y="point:Q")

    # Stitch history + forecast at the seam: a connector line from the
    # last observed value to the first forecast point so the chart
    # reads as a single trajectory rather than two disconnected lines.
    if len(f) and len(h):
        last_h = h.sort_values("period").groupby("term").tail(1)
        first_f = f.sort_values("period").groupby("term").head(1)
        seam = pd.concat(
            [
                last_h.assign(point=last_h["relfreq"])[
                    ["period", "term", "point"]
                ],
                first_f[["period", "term", "point"]],
            ]
        )
        seam_line = (
            alt.Chart(seam)
            .mark_line(strokeDash=[6, 4], strokeWidth=2, opacity=0.55)
            .encode(
                x="period:T",
                y="point:Q",
                color=alt.Color("term:N", legend=None),
            )
        )
    else:
        seam_line = alt.Chart(pd.DataFrame({"period": [], "point": [], "term": []}))

    chart = (
        history_band
        + history_line
        + history_points
        + seam_line
        + forecast_band
        + forecast_line
        + forecast_points
    ).properties(width=width, height=height)
    return chart  # type: ignore[no-any-return]
