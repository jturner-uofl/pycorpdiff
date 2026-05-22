"""Bundled corpora for demonstrations, tutorials, and reproducible tests.

What ships with the package
---------------------------

- :func:`load_hansard_sample` — a 200-speech synthetic corpus designed
  to mimic UK Hansard's structure across two decades, four topics, and
  four parties, with topical language shifts around real-world events
  (Brexit referendum, COVID-19, the climate-emergency declarations).

The sample is **synthetic** but its structure is realistic enough to
demo every analytical surface in pycorpdiff. For an actual research
project users will want the real Hansard archive — see the docstring on
:func:`load_hansard_sample` for the canonical download paths.
"""

from __future__ import annotations

from .hansard import fetch_hansard, load_hansard_sample

__all__ = ["fetch_hansard", "load_hansard_sample"]
