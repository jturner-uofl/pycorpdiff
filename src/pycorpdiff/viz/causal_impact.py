"""Three-panel causal-impact plot (Brodersen et al. 2015 style).

- Panel 1 — observed series vs counterfactual (dashed) with CrI band
- Panel 2 — pointwise effect (observed − counterfactual) with CrI band,
  zero reference line
- Panel 3 — cumulative effect with CrI band, zero reference line

Stacked vertically, sharing the time axis. The visual grammar matches
the figures from Brodersen, Gallusser, Koehler, Remy & Scott (2015).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import altair as alt

    from ..temporal.causal_impact import CausalImpactResult


def causal_impact_plot(
    result: CausalImpactResult,
    *,
    width: int = 640,
    height_per_panel: int = 200,
) -> alt.Chart:
    """Three-panel observed/counterfactual + effect plot.

    Parameters
    ----------
    result
        A :class:`CausalImpactResult` from
        :meth:`TemporalTrajectory.causal_impact`.
    width
        Width of each panel.
    height_per_panel
        Height of each panel. Total chart height ≈ 3× this.
    """
    import altair as alt

    df = result.table.copy()
    if len(df) and isinstance(df["period"].iloc[0], pd.Period):
        df["period"] = df["period"].apply(lambda p: p.to_timestamp())

    # The event marker: a vertical rule at result.event_date.
    event_df = pd.DataFrame({"event": [pd.Timestamp(result.event_date)]})
    event_rule = (
        alt.Chart(event_df)
        .mark_rule(color="#999", strokeDash=[3, 3], strokeWidth=1.5)
        .encode(x="event:T")
    )
    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0.0]}))
        .mark_rule(color="#666", strokeWidth=1)
        .encode(y="y:Q")
    )

    x_axis = alt.X("period:T", title=None)

    # ---------- Panel 1: observed vs counterfactual ----------
    base1 = alt.Chart(df).encode(x=x_axis)
    cf_band = base1.mark_area(opacity=0.18, color="#888").encode(
        y=alt.Y("counterfactual_lower:Q", title=f"{result.target} rate"),
        y2="counterfactual_upper:Q",
    )
    cf_line = base1.mark_line(
        color="#888", strokeDash=[5, 4], strokeWidth=2
    ).encode(y="counterfactual:Q")
    observed_line = base1.mark_line(color="#0b6e7c", strokeWidth=2.5).encode(
        y="observed:Q",
        tooltip=[
            "period",
            alt.Tooltip("observed:Q", format=".5f"),
            alt.Tooltip("counterfactual:Q", format=".5f"),
            alt.Tooltip("counterfactual_lower:Q", format=".5f"),
            alt.Tooltip("counterfactual_upper:Q", format=".5f"),
        ],
    )
    panel1 = (
        (cf_band + cf_line + observed_line + event_rule)
        .properties(
            width=width,
            height=height_per_panel,
            title=alt.TitleParams(
                text=f"Observed (teal) vs counterfactual (gray) -- {result.target!r}",
                subtitle=(
                    f"event = {pd.Timestamp(result.event_date).date()}, "
                    f"BSTS local linear trend on {result.n_pre} pre-event periods"
                ),
            ),
        )
    )

    # ---------- Panel 2: pointwise effect ----------
    base2 = alt.Chart(df).encode(x=x_axis)
    pw_band = base2.mark_area(opacity=0.22, color="#e63946").encode(
        y=alt.Y(
            "pointwise_lower:Q",
            title="pointwise effect",
            axis=alt.Axis(labelExpr=r"replace(datum.label, '−', '-')"),
        ),
        y2="pointwise_upper:Q",
    )
    pw_line = base2.mark_line(color="#e63946", strokeWidth=2).encode(
        y="pointwise_effect:Q",
        tooltip=[
            "period",
            alt.Tooltip("pointwise_effect:Q", format=".5f"),
            alt.Tooltip("pointwise_lower:Q", format=".5f"),
            alt.Tooltip("pointwise_upper:Q", format=".5f"),
        ],
    )
    panel2 = (pw_band + pw_line + zero_rule + event_rule).properties(
        width=width,
        height=height_per_panel,
        title=alt.TitleParams(
            text="Pointwise effect (observed - counterfactual)",
            subtitle=f"{int(result.level * 100)}% credible interval shaded",
        ),
    )

    # ---------- Panel 3: cumulative effect ----------
    base3 = alt.Chart(df).encode(x=x_axis)
    cum_band = base3.mark_area(opacity=0.22, color="#1f7a3e").encode(
        y=alt.Y(
            "cumulative_lower:Q",
            title="cumulative effect",
            axis=alt.Axis(labelExpr=r"replace(datum.label, '−', '-')"),
        ),
        y2="cumulative_upper:Q",
    )
    cum_line = base3.mark_line(color="#1f7a3e", strokeWidth=2).encode(
        y="cumulative_effect:Q",
        tooltip=[
            "period",
            alt.Tooltip("cumulative_effect:Q", format=".5f"),
            alt.Tooltip("cumulative_lower:Q", format=".5f"),
            alt.Tooltip("cumulative_upper:Q", format=".5f"),
        ],
    )
    panel3 = (cum_band + cum_line + zero_rule + event_rule).properties(
        width=width,
        height=height_per_panel,
        title="Cumulative effect",
    )

    return alt.vconcat(panel1, panel2, panel3).resolve_scale(x="shared")  # type: ignore[no-any-return]
