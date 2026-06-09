"""Collocation-shift visualisation — diverging horizontal bar."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import altair as alt


def collocation_diverging_bar(
    df: pd.DataFrame,
    n: int = 20,
    width: int = 500,
    height: int | None = None,
) -> alt.Chart:
    """Diverging bar chart of the top ``n`` collocate shifts.

    Positive bars are A-leaning collocates (gained around the target in
    A); negative bars are B-leaning (lost from A, gained in B). Sorted
    by signed shift so the eye reads the divergence directly.
    """
    import altair as alt

    subset = (
        df.assign(_abs=df["shift"].abs()).nlargest(n, "_abs").drop(columns="_abs")
    ).sort_values("shift", ascending=False)
    if height is None:
        height = max(200, 18 * len(subset))

    chart = (
        alt.Chart(subset)
        .mark_bar()
        .encode(
            x=alt.X(
                "shift:Q",
                title="Shift (A - B)",
                axis=alt.Axis(labelExpr=r"replace(datum.label, '−', '-')"),
            ),
            y=alt.Y("collocate:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.shift >= 0,
                alt.value("#1f77b4"),
                alt.value("#d62728"),
            ),
            tooltip=list(subset.columns),
        )
        .properties(width=width, height=height)
    )
    return chart  # type: ignore[no-any-return]
