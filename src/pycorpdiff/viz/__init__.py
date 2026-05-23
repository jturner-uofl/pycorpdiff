"""Visualisation helpers — altair-first, matplotlib for paper-grade figures.

Every Result type's ``.plot()`` method delegates here. Plot functions
also accept a bare DataFrame so users can call
``pcd.viz.keyness_volcano(df)`` directly without going through a Result.

altair is an optional dependency declared in the ``viz`` extra. Each
plot function lazily imports altair on first call; the friendly
ImportError lives at that boundary.
"""

from __future__ import annotations

from .causal_impact import causal_impact_plot
from .collocation import collocation_diverging_bar
from .dispersion import dispersion_plot
from .forecast import forecast_plot
from .keyness import keyness_top_n_bar, keyness_volcano
from .network import network_plot
from .scattertext import scattertext_plot
from .trajectory import trajectory_with_ci

__all__ = [
    "causal_impact_plot",
    "collocation_diverging_bar",
    "dispersion_plot",
    "forecast_plot",
    "keyness_top_n_bar",
    "keyness_volcano",
    "network_plot",
    "scattertext_plot",
    "trajectory_with_ci",
]
