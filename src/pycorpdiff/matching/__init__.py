"""Sub-corpus balancing via coarsened-exact matching.

When two corpora differ systematically on document-level covariates
beyond the variable of interest — e.g. a "criminalising-frame" set
that over-indexes on speeches from 2018-2020 — naive keyness picks up
the *confounded* signal. Matching pre-balances the two sides on
covariates so the lexical comparison is on like-for-like documents.

The module implements :func:`match`, a public entry point that returns
a :class:`MatchResult` with the matched slices plus a per-covariate
imbalance report. Downstream analyses (:func:`pycorpdiff.compare`,
:func:`pycorpdiff.against_baseline`, …) accept the matched slices
exactly like ordinary corpora.

Method
------
Coarsened Exact Matching (CEM; Iacus, King & Porro 2012). Each
covariate is coarsened (numeric → quantile bins, categorical →
as-is); documents are stratified on the joint coarsened key; strata
that contain documents from both sides are kept, strata that don't
are dropped. Within each kept stratum, the over-represented side is
subsampled with a seeded RNG so the matched-pair counts are equal —
the strongest CEM variant ("k-to-k" matching) and the one that
slots cleanly into our integer-count keyness pipeline.

CEM is the right default for corpus linguistics: it requires no
propensity model (which would be opaque to corpus linguists), it
handles the categorical metadata that corpus archives actually have
(party, year-bucket, topic, speaker-role) without contortion, and
it produces an interpretable diagnostic — L1 imbalance per
covariate, before and after.
"""

from __future__ import annotations

from .cem import MatchResult, match

__all__ = ["MatchResult", "match"]
