# Changelog

All notable changes to `pycorpdiff` are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — Phase 1: lexical-comparative core
- `Corpus.doc_term_counts()`, `.vocab()`, `.tokens()`, `.total_tokens()`
  (and the same on `CorpusSlice`).
- `pycorpdiff.keyness.log_likelihood` — Dunning G² with signed convention,
  scipy chi-squared *p*-values, and the standard 0·log(0)=0 convention
  via `scipy.special.xlogy`.
- `pycorpdiff.keyness.log_ratio` — Hardie's LogRatio with Laplace smoothing
  (α=0.5 default).
- `pycorpdiff.keyness.percent_diff` — Gabrielatos's %DIFF (per-million
  normalisation, ±inf for novel terms).
- `pycorpdiff.keyness.bayes_factor` — Wilson's BIC-approximated BF with
  overflow → inf for decisive evidence.
- `pycorpdiff.keyness.juilland_d`, `dispersion_dp` — dispersion measures
  for keyness sanity-checking.
- `pycorpdiff.keyness.benjamini_hochberg`, `bonferroni` — multiple-comparison
  correction.
- `Comparison.keyness()` wired end-to-end: aligns vocabularies, applies
  `min_count`, computes the requested statistics, optionally adds dispersion,
  applies BH/Bonferroni correction, and returns a populated `KeynessResult`
  sorted by `|score|`.
- 58 new tests (74 total): known-answer tests against Rayson-style examples,
  hypothesis property tests on swap-symmetry / non-negativity / proportional
  invariance, and an end-to-end fixture corpus that produces interpretable
  keyness rankings.

### Phase 0
- Project scaffolding: `pyproject.toml`, MIT license, README, CITATION,
  layered src/ tree, tests/, docs/, examples/, paper/, benchmarks/, CI workflow.
- `Corpus` and `CorpusSlice` dataclasses (Layer 1 plumbing).
- `Tokenizer` Protocol with a `RegexTokenizer` default.
- I/O readers for CSV, parquet, and pandas DataFrames.
- Stub modules for collocation, semantic, temporal, viz subpackages
  (all raise `NotImplementedError` pending later phases).

## [0.1.0a0] — 2026-05-22

Initial scaffolding commit. No public functionality yet.
