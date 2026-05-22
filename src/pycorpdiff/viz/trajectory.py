"""Temporal trajectory plot — line + Wilson CI band."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import altair as alt


def trajectory_with_ci(
    df: pd.DataFrame,
    width: int = 600,
    height: int = 300,
) -> alt.Chart:
    """Time series of relative frequencies with a Wilson CI band.

    Expects the columns produced by :meth:`Tracker.over_time`:
    ``period``, ``term``, ``relfreq``, ``ci_lower``, ``ci_upper``.
    Multiple terms are layered with the standard altair colour scheme.

    The ``period`` column may contain :class:`pandas.Period` values —
    converted to timestamps internally so altair gets a temporal axis.
    """
    import altair as alt

    plot_df = df.copy()
    if isinstance(plot_df["period"].iloc[0], pd.Period):
        plot_df["period"] = plot_df["period"].apply(lambda p: p.to_timestamp())

    base = alt.Chart(plot_df).encode(
        x=alt.X("period:T", title=None),
        color=alt.Color("term:N", title=None),
    )
    band = base.mark_area(opacity=0.2).encode(
        y=alt.Y("ci_lower:Q", title="Relative frequency"),
        y2="ci_upper:Q",
    )
    line = base.mark_line(strokeWidth=2).encode(
        y="relfreq:Q",
        tooltip=["period", "term", "count", "total", "relfreq", "ci_lower", "ci_upper"],
    )
    points = base.mark_point(filled=True, size=50).encode(
        y="relfreq:Q",
    )
    return (band + line + points).properties(width=width, height=height)  # type: ignore[no-any-return]
