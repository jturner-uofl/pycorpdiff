"""Burstiness trajectory plot — line + per-period burst-state bands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import altair as alt


def burstiness_plot(
    df: pd.DataFrame,
    target: str,
    width: int = 600,
    height: int = 280,
) -> alt.Chart:
    """Trajectory line + coloured burst-state bands.

    Expects the columns produced by
    :func:`pycorpdiff.temporal.burstiness.burstiness_from_table`:
    ``period``, ``relfreq``, ``state``. ``period`` may be a pandas
    :class:`Period` and is converted to a timestamp for altair.
    """
    import altair as alt

    plot_df = df.copy()
    if isinstance(plot_df["period"].iloc[0], pd.Period):
        plot_df["period"] = plot_df["period"].apply(lambda p: p.to_timestamp())

    # Explicit, high-contrast palette for the burst states. The default
    # "reds" scheme maps low ordinal states (0 vs 1) to two near-identical
    # pale pinks that are impossible to tell apart. Instead, state 0
    # (baseline / no burst) is a neutral grey and states 1+ escalate
    # through warm reds, so a burst period is unmistakable against the
    # baseline. Domain/range are built from the states actually present
    # so the legend never lists unused states.
    _state_palette = ["#dfe3e6", "#fc8d59", "#e34a33", "#b30000", "#7f0000"]
    _states_present = sorted(int(s) for s in plot_df["state"].unique())
    _state_range = [
        _state_palette[min(s, len(_state_palette) - 1)] for s in _states_present
    ]

    # Bar chart of state intensity, sitting under the trajectory line.
    bars = (
        alt.Chart(plot_df)
        .mark_bar(opacity=0.55)
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y(
                "relfreq:Q",
                title="Relative frequency",
                axis=alt.Axis(format=".2%"),
            ),
            color=alt.Color(
                "state:O",
                title="Burst state",
                scale=alt.Scale(domain=_states_present, range=_state_range),
            ),
            tooltip=["period", "count", "total", "relfreq", "state"],
        )
    )
    line = (
        alt.Chart(plot_df)
        .mark_line(strokeWidth=2, color="#222")
        .encode(
            x="period:T",
            y="relfreq:Q",
        )
    )
    points = (
        alt.Chart(plot_df)
        .mark_point(filled=True, size=50, color="#222")
        .encode(
            x="period:T",
            y="relfreq:Q",
        )
    )
    return (bars + line + points).properties(  # type: ignore[no-any-return]
        width=width,
        height=height,
        title=f"Burstiness of '{target}'",
    )
