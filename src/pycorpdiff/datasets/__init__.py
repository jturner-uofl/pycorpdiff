"""Bundled corpora for demonstrations, tutorials, and reproducible tests.

What ships with the package
---------------------------

- :func:`load_hansard_sample` — a 193-speech synthetic corpus designed
  to mimic UK Hansard's structure across two decades, four topics, and
  four parties, with topical language shifts around real-world events
  (Brexit referendum, COVID-19, the climate-emergency declarations).

The sample is **synthetic** but its structure is realistic enough to
demo every analytical surface in pycorpdiff. For an actual research
project users will want the real Hansard archive — see the docstring on
:func:`load_hansard_sample` for the canonical download paths.
"""

from __future__ import annotations

from .baselines import Baseline, baseline_from_corpus, list_baselines, load_baseline
from .hansard import fetch_hansard, load_hansard_sample
from .histwords import fetch_histwords_decade, histwords_cosine_shift

__all__ = [
    "Baseline",
    "baseline_from_corpus",
    "fetch_hansard",
    "fetch_histwords_decade",
    "histwords_cosine_shift",
    "list_baselines",
    "load_baseline",
    "load_hansard_sample",
]
