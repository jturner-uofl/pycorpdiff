"""Plot for :func:`pycorpdiff.forecast_semantic_drift` output.

Same dashed-extension grammar as :func:`pycorpdiff.viz.forecast_plot`
but operates on the cosine-distance scale (``distance_from_baseline``
column on the history side, ``point`` + PI on the forecast side).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import altair as alt


def semantic_forecast_plot(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    width: int = 600,
    height: int = 320,
) -> alt.Chart:
    """Layered plot of semantic drift with a forecast continuation.

    Parameters
    ----------
    history
        The DataFrame returned by :func:`pycorpdiff.semantic_trajectory`
        — must carry ``period``, ``target``, ``distance_from_baseline``.
    forecast
        The DataFrame returned by
        :func:`pycorpdiff.forecast_semantic_drift` — must carry
        ``period``, ``target``, ``point``, ``ci_lower``, ``ci_upper``.
    """
    import altair as alt

    h = history.copy()
    f = forecast.copy()
    if len(h) and isinstance(h["period"].iloc[0], pd.Period):
        h["period"] = h["period"].apply(lambda p: p.to_timestamp())
    if len(f) and isinstance(f["period"].iloc[0], pd.Period):
        f["period"] = f["period"].apply(lambda p: p.to_timestamp())

    base_h = alt.Chart(h).encode(
        x=alt.X("period:T", title=None),
        color=alt.Color("target:N", title=None),
    )
    history_line = base_h.mark_line(strokeWidth=2.5).encode(
        y=alt.Y("distance_from_baseline:Q", title="cosine distance from baseline"),
        tooltip=[
            "period",
            "target",
            alt.Tooltip("distance_from_baseline:Q", format=".4f"),
            alt.Tooltip("n_contexts:Q") if "n_contexts" in h.columns else "target:N",
        ],
    )
    history_points = base_h.mark_point(filled=True, size=55).encode(
        y="distance_from_baseline:Q",
    )

    base_f = alt.Chart(f).encode(
        x=alt.X("period:T", title=None),
        color=alt.Color("target:N", title=None),
    )
    forecast_band = base_f.mark_area(opacity=0.18).encode(
        y=alt.Y("ci_lower:Q"),
        y2="ci_upper:Q",
    )
    forecast_line = base_f.mark_line(strokeDash=[6, 4], strokeWidth=2).encode(
        y="point:Q",
        tooltip=[
            "period",
            "target",
            alt.Tooltip("point:Q", format=".4f"),
            alt.Tooltip("ci_lower:Q", format=".4f"),
            alt.Tooltip("ci_upper:Q", format=".4f"),
        ],
    )
    forecast_points = base_f.mark_point(filled=False, strokeWidth=2, size=55).encode(
        y="point:Q"
    )

    if len(h) and len(f):
        last_h = h.sort_values("period").groupby("target").tail(1)
        first_f = f.sort_values("period").groupby("target").head(1)
        seam = pd.concat(
            [
                last_h.assign(point=last_h["distance_from_baseline"])[
                    ["period", "target", "point"]
                ],
                first_f[["period", "target", "point"]],
            ]
        )
        seam_line = (
            alt.Chart(seam)
            .mark_line(strokeDash=[6, 4], strokeWidth=2, opacity=0.55)
            .encode(
                x="period:T",
                y="point:Q",
                color=alt.Color("target:N", legend=None),
            )
        )
    else:
        seam_line = alt.Chart(
            pd.DataFrame({"period": [], "point": [], "target": []})
        )

    chart = (
        history_line + history_points + seam_line + forecast_band
        + forecast_line + forecast_points
    ).properties(width=width, height=height)
    return chart  # type: ignore[no-any-return]
