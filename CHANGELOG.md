# Changelog

All notable changes to `pycorpdiff` are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project scaffolding: `pyproject.toml`, MIT license, README, CITATION,
  layered src/ tree, tests/, docs/, examples/, paper/, benchmarks/, CI workflow.
- `Corpus` and `CorpusSlice` dataclasses (Layer 1 plumbing).
- `Tokenizer` Protocol with a `RegexTokenizer` default.
- I/O readers for CSV, parquet, and pandas DataFrames.
- Stub modules for keyness, collocation, semantic, temporal, viz subpackages
  (all raise `NotImplementedError` pending Phase 1+).

## [0.1.0a0] — 2026-05-22

Initial scaffolding commit. No public functionality yet.
