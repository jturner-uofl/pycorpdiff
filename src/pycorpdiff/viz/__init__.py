"""Visualisation helpers — altair-first, matplotlib for paper-grade figures.

Every Result type's ``.plot()`` method delegates here. Plot functions
also accept a bare DataFrame so users can call
``pcd.viz.keyness_volcano(df)`` directly without going through a Result.

altair is an optional dependency declared in the ``viz`` extra. Each
plot function lazily imports altair on first call; the friendly
ImportError lives at that boundary.
"""

from __future__ import annotations

from .collocation import collocation_diverging_bar
from .keyness import keyness_top_n_bar, keyness_volcano
from .trajectory import trajectory_with_ci

__all__ = [
    "collocation_diverging_bar",
    "keyness_top_n_bar",
    "keyness_volcano",
    "trajectory_with_ci",
]
