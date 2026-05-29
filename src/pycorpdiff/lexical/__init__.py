"""Lexical-diversity metrics — corpus-level vocabulary range, length-robust.

Lexical diversity sits alongside keyness (*which* words) and frequency
(*how many*) as the third foundational corpus statistic: *how varied
is the vocabulary*. The naive measure — type-token ratio (TTR) — is
strongly negatively correlated with text length, so it can't compare
texts or corpora of different sizes. The metrics in this module
correct that, each via a different strategy:

- :func:`ttr` — the uncorrected baseline (kept for backward
  compatibility / familiarity, but documented as length-dependent).
- :func:`mattr` — moving-average TTR over fixed-size windows
  (Covington & McFall 2010). Length-robust by construction.
- :func:`mtld` — mean factor length to a running-TTR threshold
  (McCarthy & Jarvis 2010). The most widely-cited modern metric.
- :func:`hdd` — hypergeometric-distribution-based expected vocabulary
  in a 42-token sample (McCarthy & Jarvis 2007). The most
  statistically principled.

Public entry points:

- :func:`lexical_diversity` — single function, returns either a
  pooled :class:`LexicalDiversityResult` or a per-period
  :class:`LexicalDiversityTrajectory` depending on whether ``freq``
  is supplied.
"""

from __future__ import annotations

from .diversity import (
    LexicalDiversityResult,
    LexicalDiversityTrajectory,
    hdd,
    lexical_diversity,
    mattr,
    mtld,
    ttr,
)

__all__ = [
    "LexicalDiversityResult",
    "LexicalDiversityTrajectory",
    "hdd",
    "lexical_diversity",
    "mattr",
    "mtld",
    "ttr",
]
