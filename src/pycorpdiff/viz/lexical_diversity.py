"""Lexical-diversity-over-time plot — one line per metric + optional CI bands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import altair as alt


def lexical_diversity_trajectory_plot(
    df: pd.DataFrame,
    width: int = 600,
    height: int = 320,
    metrics: list[str] | None = None,
    **_kw: Any,
) -> alt.Chart:
    """Per-metric trajectory with optional bootstrap-CI bands.

    Expects the long-form table from :class:`LexicalDiversityTrajectory`
    with columns ``period``, ``metric``, ``value`` and optional
    ``ci_lower`` / ``ci_upper``. Each metric is plotted as its own
    subplot (faceted vertically) so the very different value scales
    (TTR ~ 0.5, MATTR ~ 0.5, MTLD ~ 100, HD-D ~ 35) don't collapse
    into illegible noise on a single axis.
    """
    import altair as alt

    plot_df = df.copy()
    if isinstance(plot_df["period"].iloc[0], pd.Period):
        plot_df["period"] = plot_df["period"].apply(lambda p: p.to_timestamp())

    if metrics is not None:
        plot_df = plot_df[plot_df["metric"].isin(metrics)]

    has_ci = "ci_lower" in plot_df.columns and "ci_upper" in plot_df.columns

    base = alt.Chart(plot_df).encode(
        x=alt.X("period:T", title=None),
        color=alt.Color("metric:N", title=None, legend=None),
    )
    layers: list[alt.Chart] = []
    if has_ci:
        layers.append(
            base.mark_area(opacity=0.18).encode(
                y=alt.Y("ci_lower:Q", title=None),
                y2="ci_upper:Q",
            )
        )
    layers.append(
        base.mark_line(strokeWidth=2).encode(
            y=alt.Y("value:Q", title=None),
            tooltip=["period", "metric", "value", "n_tokens", "n_types"],
        )
    )
    layers.append(
        base.mark_point(filled=True, size=42).encode(
            y="value:Q",
        )
    )
    combined = alt.layer(*layers)
    return combined.facet(  # type: ignore[no-any-return]
        row=alt.Row("metric:N", title=None, sort=["TTR", "MATTR", "MTLD", "HD-D"]),
    ).resolve_scale(y="independent").properties(
        title="Lexical diversity over time",
    ).configure_view(
        continuousWidth=width,
        continuousHeight=max(60, height // 4),
    )
