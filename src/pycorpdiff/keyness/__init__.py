"""Keyness measures — Dunning log-likelihood, LogRatio, Bayes factor.

Phase 1 will populate this subpackage. The intended public surface is:

- :func:`log_likelihood` — Dunning's G² with continuity correction.
- :func:`log_ratio` — Hardie's LogRatio effect size.
- :func:`bayes_factor` — Wilson / Gabrielatos Bayes factor.
- :func:`dispersion` — Juilland's D and DP for sanity-checking keyness.
"""

from __future__ import annotations

from .bayes import bayes_factor
from .dispersion import dispersion_dp, juilland_d
from .effect_sizes import log_ratio, percent_diff
from .loglikelihood import log_likelihood

__all__ = [
    "bayes_factor",
    "dispersion_dp",
    "juilland_d",
    "log_likelihood",
    "log_ratio",
    "percent_diff",
]
