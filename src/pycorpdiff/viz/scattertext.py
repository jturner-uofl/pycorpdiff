"""Scattertext-style interactive scatter (Kessler 2017).

The signature Scattertext visualisation. Each term is a point whose
x-axis position is its rank-percentile in corpus A and whose y-axis
position is its rank-percentile in corpus B. Words common in both
land in the top-right corner; words common in only one side fall away
from the diagonal into one of the two off-diagonal "distinctiveness"
zones; words rare in both cluster near the origin.

The rank-based axes are the trick that makes Scattertext readable.
Plotting raw counts produces a plot dominated by stopwords in the
top-right corner with everything else crushed into the bottom-left;
rank-percentiles spread the whole vocabulary evenly across [0, 1].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import altair as alt


def scattertext_plot(
    df: pd.DataFrame,
    *,
    label_a: str = "a",
    label_b: str = "b",
    width: int = 600,
    height: int = 600,
    n_labels: int = 20,
) -> alt.Chart:
    """Scattertext-style interactive scatter of keyness terms.

    Expects the columns produced by :meth:`Comparison.keyness`:
    ``term``, ``count_a``, ``count_b``, and ``g2`` (the signed
    log-likelihood used as the colour channel). ``log_ratio`` is used
    in the tooltip when present.

    The chart is pan/zoom-able and every point has a hover tooltip
    listing the raw counts and effect sizes. The ``n_labels`` most
    A-leaning and ``n_labels`` most B-leaning terms (by ``|g2|``) get
    inline text labels — others remain dots.

    Parameters
    ----------
    df
        A KeynessResult-shaped DataFrame.
    label_a, label_b
        Axis titles — usually the corpus labels carried on the result.
    width, height
        Square by default (600×600); the rank-percentile axes deserve
        equal visual weight.
    n_labels
        Per-side label budget. ``n_labels=20`` produces up to 40 text
        labels total.
    """
    import altair as alt

    if df.empty:
        empty_chart = (
            alt.Chart(df).mark_point().properties(width=width, height=height)
        )
        return empty_chart  # type: ignore[no-any-return]

    plot = df.copy()
    # rank() with method="average" handles ties cleanly; pct=True scales to
    # (0, 1]. Higher percentile == more common in that corpus.
    plot["percentile_a"] = plot["count_a"].rank(pct=True, method="average")
    plot["percentile_b"] = plot["count_b"].rank(pct=True, method="average")
    plot["abs_g2"] = plot["g2"].abs()

    # Symmetric colour scale around zero so the colour midpoint corresponds
    # to "equally common in both", not to the median G² of the table.
    g2_max = float(plot["g2"].abs().max() or 1.0)

    tooltip_cols: list[str] = ["term", "count_a", "count_b", "g2"]
    for opt in ("log_ratio", "percent_diff", "p_value", "p_adjusted"):
        if opt in plot.columns:
            tooltip_cols.append(opt)

    base = alt.Chart(plot).encode(
        x=alt.X(
            "percentile_a:Q",
            title=f"Frequency rank in {label_a} (-> more common)",
            scale=alt.Scale(domain=[0, 1]),
        ),
        y=alt.Y(
            "percentile_b:Q",
            title=f"Frequency rank in {label_b} (-> more common)",
            scale=alt.Scale(domain=[0, 1]),
        ),
        tooltip=tooltip_cols,
    )

    # The diagonal x = y is the "equally distinctive" line — terms on it
    # have the same rank in both corpora. Drawing it as a reference rule
    # helps readers calibrate the distinctiveness zones.
    diag = (
        alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(strokeDash=[4, 4], color="#999", opacity=0.5)
        .encode(x="x:Q", y="y:Q")
    )

    points = base.mark_circle(opacity=0.55).encode(
        size=alt.Size(
            "abs_g2:Q",
            scale=alt.Scale(range=[20, 200]),
            legend=None,
        ),
        color=alt.Color(
            "g2:Q",
            scale=alt.Scale(
                scheme="redblue",
                domain=[-g2_max, 0.0, g2_max],
                reverse=True,  # blue = A-leaning, red = B-leaning
            ),
            title="Signed G^2",
        ),
    )

    # Pick the top-n_labels on each side by signed G².
    a_leaning = plot.nlargest(n_labels, "g2")
    b_leaning = plot.nsmallest(n_labels, "g2")
    labelled = pd.concat([a_leaning, b_leaning], ignore_index=True).drop_duplicates(
        subset="term"
    )
    labels = (
        alt.Chart(labelled)
        .mark_text(align="left", dx=5, dy=-2, fontSize=10)
        .encode(
            x="percentile_a:Q",
            y="percentile_b:Q",
            text="term:N",
            color=alt.Color(
                "g2:Q",
                scale=alt.Scale(
                    scheme="redblue",
                    domain=[-g2_max, 0.0, g2_max],
                    reverse=True,
                ),
                legend=None,
            ),
        )
    )

    chart = (
        (diag + points + labels)
        .properties(width=width, height=height)
        .interactive()
    )
    return chart  # type: ignore[no-any-return]


def _percentile_rank(series: pd.Series) -> np.ndarray:
    """Internal helper used by tests; kept here for stability."""
    return series.rank(pct=True, method="average").to_numpy()
