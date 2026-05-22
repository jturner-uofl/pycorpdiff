# Changelog

All notable changes to `pycorpdiff` are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — Tutorial notebook (post-Phase-4 polish)
- `examples/pycorpdiff_tutorial.ipynb` rewritten as a real, executable
  guided tour. Builds a 144-document synthetic news corpus with two
  outlets across nine years, then walks the full working API: slicing,
  keyness with all four side-statistics, volcano + bar plots, KWIC
  explain, collocation shift with logDice, collocation explain,
  multi-term temporal trajectories with Wilson CIs, the trajectory plot,
  and `compare.before_after`.
- New `tutorial` CI job re-executes the notebook on every push so it
  can't drift from the public API silently.

### Added — Phase 4b: altair plots on every Result
- `KeynessResult.plot(kind="volcano" | "bar", **kw)` — volcano-style
  scatter of effect size vs −log₁₀(p) by default, with the top-N labelled;
  optional `kind="bar"` for the cleaner top-N horizontal bar.
- `CollocationShiftResult.plot(n=20, **kw)` — diverging horizontal bar
  of the top |shift| collocates, colour-coded by direction.
- `TemporalTrajectory.plot(**kw)` — line + Wilson CI band, multi-term
  ready, `pd.Period` automatically coerced to timestamps for altair's
  temporal axis.
- Module-level `pycorpdiff.viz.{keyness_volcano,keyness_top_n_bar,collocation_diverging_bar,trajectory_with_ci}`
  for users with a bare DataFrame.
- altair is lazy-imported inside each viz function so the base install
  remains lightweight; missing altair fails with a friendly install hint
  via the standard `ImportError`.
- CI workflow now installs `[dev,viz]` to cover the new tests.
- 7 new tests (151 total): spec-shape assertions for each Result.plot
  variant, period-to-timestamp coercion, and the bare-DataFrame entry
  points.

### Added — Phase 4a: temporal slicing and trajectories
- `pycorpdiff.stats.wilson_ci` — vectorised Wilson score interval for
  binomial proportions, the default CI for every relative-frequency
  surface in the package. Handles `n = 0` (returns NaN), clips
  roundoff at the unit-interval edges, and configurable confidence
  level via scipy's normal-quantile inverse.
- `TemporalCorpus.periods()` — sorted list of populated periods.
- `TemporalCorpus.slice(period)` — `CorpusSlice` for one period;
  accepts `pd.Period` or any string pandas can parse.
- `TemporalCorpus.iter_slices()` — chronological `(period, CorpusSlice)`
  iterator.
- `Tracker.over_time(freq, time_col, confidence)` — populated
  `TemporalTrajectory` with one row per (period, term), columns
  `period / term / count / total / relfreq / ci_lower / ci_upper`.
  Multi-term tracking supported. `Tracker.trajectory()` is the alias.
- 20 new tests (144 total): Wilson CI known-answer (Newcombe Table 1
  for x=10, n=100 → [0.0553, 0.1747]), CI bracketing of point
  estimates, multi-target sorting, quarter / month aliasing, period
  serialisation, and the alias parity between `over_time` and
  `trajectory`.

### Added — Phase 3: KWIC concordances and `explain()`
- `pycorpdiff.kwic(corpus, target, window, n, label)` — per-document
  windowed KWIC extraction; never crosses document boundaries.
- `pycorpdiff.representative_docs(corpus, target, n)` — top-n documents
  by target frequency, with stable doc-id tie-breaking.
- Internal `kwic_compare` — side-by-side KWIC tables from two corpora,
  with optional collocate filter (the engine behind
  `CollocationShiftResult.explain`).
- `KeynessResult.explain(term, n, window)` — pulls KWIC evidence from
  both source corpora (up to *n* lines per side).
- `CollocationShiftResult.explain(collocate, n)` — restricted to windows
  in which both the result's target and the collocate appear.
- `KeynessResult.corpus_a / corpus_b` and `CollocationShiftResult.corpus_a / corpus_b`
  — optional references populated by `Comparison.keyness/collocation_shift`
  so explain has evidence to work with. Constructing a Result without
  them raises a clear error if `.explain()` is called.
- 16 new tests (124 total): KWIC schema and window correctness,
  document-boundary isolation, n-cap behaviour, collocate-filtered
  windows, both-sides representation, and the explain-without-corpora
  error path.

### Added — Phase 2: collocations and collocation shift
- `pycorpdiff.collocation.collocate_counts` — window-based co-occurrence
  extractor, per-document isolation, configurable window size.
- `pycorpdiff.collocation.logdice` — Rychlý's logDice (14 + log2 form).
- `pycorpdiff.collocation.pmi` — Church-Hanks association ratio.
- `pycorpdiff.collocation.t_score` — Welch-style t for collocations.
- `pycorpdiff.collocation.mi_three` — Daille's MI³, downweighting rare pairs.
- `pycorpdiff.collocation.collocation_shift` — orchestrator that aligns
  collocate vocabularies, applies Laplace smoothing, computes the chosen
  measure on each side, and returns a tidy `shift = score_a - score_b`
  table sorted by `|shift|`.
- `Comparison.collocation_shift()` wired end-to-end, returning a populated
  `CollocationShiftResult`.
- 34 new tests (108 total): known-answer tests for each measure
  (sketchengine-style worked example), per-doc window-isolation tests,
  swap-symmetry tests, all-measures-agree-on-strong-signal integration
  test.
- Public `CollocationMeasure` literal: ``"logDice" | "PMI" | "t_score" | "MI3"``
  (`MI` dropped — it was an alias for PMI in CL practice and the duplicate
  was a misnomer).

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
