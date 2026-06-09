"""Keyness visualisations — volcano plot and top-N bar chart."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import altair as alt


def keyness_volcano(
    df: pd.DataFrame,
    width: int = 600,
    height: int = 400,
    n_labels: int = 15,
) -> alt.Chart:
    """Volcano-style scatter: effect size (x) versus significance (y).

    Expects the columns produced by :meth:`Comparison.keyness`:
    ``term``, ``log_ratio`` (or fall back to ``g2``), and ``p_value``.
    The top ``n_labels`` rows by ``|log_ratio|`` get a text label;
    everything else is plotted as a circle only.
    """
    import altair as alt

    if "log_ratio" in df.columns:
        x_col = "log_ratio"
        x_title = "LogRatio (positive = overused in A)"
    else:
        x_col = "g2"
        x_title = "Signed G^2 (positive = overused in A)"

    # Significance axis: -log10(p), with infinities clipped at a sensible cap
    # so a single near-zero p doesn't crush the rest of the plot vertically.
    with np.errstate(divide="ignore"):
        neg_log_p = -np.log10(np.clip(df["p_value"].to_numpy(), 1e-300, 1.0))
    plot_df = df.assign(neg_log_p=neg_log_p)

    base = alt.Chart(plot_df).encode(
        x=alt.X(
            f"{x_col}:Q",
            title=x_title,
            axis=alt.Axis(labelExpr=r"replace(datum.label, '−', '-')"),
        ),
        y=alt.Y("neg_log_p:Q", title="-log10(p)"),
        tooltip=list(df.columns),
    )
    points = base.mark_circle(opacity=0.55, size=60).encode(
        color=alt.condition(
            alt.datum[x_col] >= 0,
            alt.value("#1f77b4"),  # A-leaning
            alt.value("#d62728"),  # B-leaning
        )
    )
    label_subset = plot_df.assign(_abs=plot_df[x_col].abs()).nlargest(n_labels, "_abs")
    labels = alt.Chart(label_subset).mark_text(align="left", dx=6, fontSize=10).encode(
        x=f"{x_col}:Q",
        y="neg_log_p:Q",
        text="term:N",
    )
    return (points + labels).properties(width=width, height=height)  # type: ignore[no-any-return]


def keyness_top_n_bar(
    df: pd.DataFrame,
    n: int = 20,
    width: int = 500,
    height: int | None = None,
) -> alt.Chart:
    """Top-N horizontal bar chart, sorted by ``|g2|`` (signed).

    Positive bars are A-leaning, negative are B-leaning. Useful when
    you want a clean publication-ready figure rather than the
    information-dense volcano.
    """
    import altair as alt

    subset = df.assign(_abs=df["g2"].abs()).nlargest(n, "_abs").drop(columns="_abs")
    if height is None:
        height = max(200, 18 * len(subset))

    chart = (
        alt.Chart(subset)
        .mark_bar()
        .encode(
            x=alt.X(
                "g2:Q",
                title="Signed G^2",
                axis=alt.Axis(labelExpr=r"replace(datum.label, '−', '-')"),
            ),
            y=alt.Y("term:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.g2 >= 0,
                alt.value("#1f77b4"),
                alt.value("#d62728"),
            ),
            tooltip=list(subset.columns),
        )
        .properties(width=width, height=height)
    )
    return chart  # type: ignore[no-any-return]
