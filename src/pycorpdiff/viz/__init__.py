"""Visualisation helpers — altair-first, matplotlib for paper-grade figures.

Every Result type's ``.plot()`` method delegates here. Plot functions
also accept a bare DataFrame so users can call ``pcd.viz.keyness(df)``
directly without going through a Result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import altair as alt


def keyness(df: pd.DataFrame, **kw: Any) -> alt.Chart:
    """Volcano-style keyness plot (effect size × significance)."""
    raise NotImplementedError("viz.keyness() lands in Phase 4")


def trajectory(df: pd.DataFrame, **kw: Any) -> alt.Chart:
    """Time-series of relative frequencies with CI bands."""
    raise NotImplementedError("viz.trajectory() lands in Phase 4")


def collocation_shift(df: pd.DataFrame, **kw: Any) -> alt.Chart:
    """Diverging bar chart of gained / lost collocates."""
    raise NotImplementedError("viz.collocation_shift() lands in Phase 4")


def before_after(df: pd.DataFrame, **kw: Any) -> alt.Chart:
    """Side-by-side comparison plot for before/after analyses."""
    raise NotImplementedError("viz.before_after() lands in Phase 4")
