# Changelog

All notable changes to `pycorpdiff` are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0a3] — initial release

The initial public release of `pycorpdiff` — comparative corpus analysis
for modern Python workflows. Three public verbs (`compare`, `track`,
`compare.before_after`), nine `Result` dataclasses with a uniform
six-method contract (`.to_df / .plot / .explain / .summary / .to_html /
.to_json`), two `typing.Protocol` extension points (`Tokenizer`,
`Embedder`), and opt-in extras for visualisation, semantic embedding,
temporal modelling, polars interop, DuckDB ingestion, and 🤗 Datasets.

### Analytical surface

- **Keyness**: signed Dunning G², Pearson χ², Hardie LogRatio,
  Gabrielatos %DIFF, BIC-Bayes factor, Juilland D / Gries DP dispersion
  flagging, Benjamini–Hochberg correction, stop-word filtering,
  empirical permutation *p*-values, N-way contingency G² via
  `keyness_multi`.
- **Collocations**: logDice, PMI, t-score, MI³ with Laplace smoothing;
  cross-corpus `collocation_shift`; co-occurrence networks via
  `cooccurrence_network`.
- **Semantic shift**: averaged contextual embeddings, Procrustes
  alignment, multi-period `semantic_trajectory`, `neighborhood_drift`.
- **Temporal**: Wilson-CI trajectories, offline PELT changepoints,
  online Bayesian changepoint detection, segmented-OLS interrupted
  time series, Bayesian structural time-series causal impact,
  state-space exponential-smoothing forecasting.

### Cross-validated

Numerically agrees with Rayson's LL Wizard (15 reference triples),
NLTK's `BigramAssocMeasures` (≤ 1e-12 on PMI / t-score / MI³),
Scattertext on the 2012 US conventions, `quanteda` via `rpy2`, and
the HistWords COHA replication.

### Infrastructure

519 tests, `ruff` + `mypy --strict` clean across 55 source files,
matrix CI on three Python versions × two operating systems.
