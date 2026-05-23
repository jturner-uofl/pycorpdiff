# Changelog

All notable changes to `pycorpdiff` are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

<!-- This section accumulates post-0.1.0a0 work. Move entries down to a
  new versioned heading when the next tag is cut. -->

## [0.1.0a0] — 2026-05-22

The complete pre-release feature set, frozen for the JSS submission.
519 default tests, ruff + mypy --strict clean across 55 source files,
three example notebooks rendered to self-contained HTML, paper
replication archive runnable in CI on every push.

### The temporal predictive stack
- `TemporalTrajectory.forecast(horizon)` — state-space exponential
  smoothing (Hyndman et al. 2008) with logit-pinned PIs.
- `TemporalTrajectory.causal_impact(event_date)` — Bayesian structural
  time-series counterfactual (Brodersen et al. 2015) with three-panel
  Brodersen-style plot.
- `TemporalTrajectory.changepoints_online(hazard)` — Bayesian online
  changepoint detection (Adams & MacKay 2007), three-panel diagnostic.
- `pcd.forecast_semantic_drift(traj_df)` — forecast cosine-distance
  trajectories with the same ETS machinery.

### N-way + network surfaces
- `pcd.keyness_multi([a, b, c, ...])` — one-way contingency G² with
  df=N−1; N=2 reduces exactly to pairwise.
- `pcd.cooccurrence_network(corpus)` — Kamada-Kawai term-as-vertex
  network with PMI / t-score / MI³ / logDice edge weights.

### Visualisation
- `viz.scattertext_plot` — Kessler 2017 rank-percentile scatter.
- `viz.dispersion_plot` — Mosteller-style tick chart.
- `viz.network_plot` — force-directed graph for `NetworkResult`.
- `viz.forecast_plot`, `viz.causal_impact_plot`, `viz.bocpd_plot`,
  `viz.semantic_forecast_plot` — predictive-stack visualisations.
- All seven surfaced at the package root for ergonomics
  (`pcd.dispersion_plot`, etc.) while remaining available via
  `pcd.viz.*`.

### Keyness
- Pearson χ² as an alternative method (`method="chi_squared"`).
- Stop-word filtering (`stop_words=`).
- Empirical permutation *p*-values (`permutation_n=B`) with
  Phipson–Smyth (2010) `+1/+1` correction.

### Tokenization + storage
- `NgramTokenizer` — bigrams / trigrams as first-class terms via any
  base tokenizer.
- `Corpus.doc_term_counts_sparse()` — scipy.sparse CSR escape hatch
  for large vocabularies.
- `Corpus.__hash__` — content-derived hashing for memoised analyses.

### I/O
- `pcd.from_huggingface(dataset_id, ...)` loader + `[huggingface]` extra.
- `read_duckdb(con, sql)` — out-of-core SQL ingestion.

### Cross-validation receipts
- 15 hand-derived Rayson LL Wizard reference triples.
- NLTK `BigramAssocMeasures` — PMI and t-score to ≤ 1e-12.
- Scattertext (Kessler 2017) on the 2012 US Conventions.
- quanteda (R) via rpy2 — byte-for-byte G² agreement (slow tier).
- HistWords (Hamilton et al. 2016) — diachronic cosine displacements
  on COHA (slow tier).

### Result surface
- `.to_html()` and `.to_json()` on every Result class.
- All Result types are frozen dataclasses; share the same six-method
  protocol.

### Tier-1 audit cleanups (incorporated into 0.1.0a0)
- `read_txt(one_doc_per="line")` now wired up. Each non-empty line
  becomes a separate document; the 1-based line number is preserved in
  the corpus's ``line`` column so KWIC results can point back to the
  original file.
- `Comparison.concordance(target, n, window)` implemented — returns a
  populated `ConcordanceResult` with KWIC lines from both corpora
  side-by-side. Documented in the README quick-start but had been a
  `NotImplementedError` stub through Phases 0-8.
- `SemanticShiftResult` now carries `corpus_a`, `corpus_b`, `embedder`,
  and `window` so `.neighbors_before(target, n)` and
  `.neighbors_after(target, n)` work on-demand. They filter
  `neighborhood_drift` output to the relevant side and sort by
  per-corpus similarity. Construction without corpus refs (bare
  DataFrame) raises a clear error when the methods are called.
- Hypothesis property tests for collocations (7 tests): logDice / PMI /
  t-score / MI³ symmetry in target↔collocate, monotonicity in joint
  count, logDice ≤ 14 in realistic regimes.
- Hypothesis property tests for Wilson CI (5 tests): CI brackets the
  point estimate, bounds in [0, 1], width monotonic in confidence,
  complement symmetry (CI for x/n is the reflection of CI for (n-x)/n
  through 0.5), and rel-freq monotonic in count.
- 29 new tests (222 total). ruff and mypy --strict still clean.

### Added — Phase 8b: JSS paper skeleton + replication
- `paper/paper.tex` — JSS-class manuscript skeleton with the full
  section structure (intro, related work, design, API, worked
  example, statistical defaults, reproducibility, conclusion).
  Sections carry `[TODO: ...]` placeholders with detailed prose
  scaffolding so the writing pass is straightforward to finish.
- `paper/references.bib` — 23 BibTeX entries covering every primary
  citation behind the package's statistical defaults: Dunning,
  Hardie, Gabrielatos, Wilson, Kass & Raftery, Juilland, Gries,
  Rychly, Church-Hanks, Daille, Wagner, Killick, Hamilton,
  Giulianelli, Newcombe, Wilson 1927, Schoenemann, Benoit
  (quanteda), Silge (tidytext), Reimers (SBERT), Truong (ruptures),
  Kessler (Scattertext), Grootendorst (BERTopic).
- `paper/replication/reproduce.py` — single-script regeneration of
  every figure and table referenced in the manuscript. Outputs four
  SVG figures (volcano, top-N bar, collocation diverging bar,
  trajectory with CI band) and a `paper_outputs.json` with the
  numeric tables (top-12 keyness rows, ITS coefficients,
  semantic-shift centroid distances, detected changepoints, corpus
  size). Uses the deterministic synthetic corpus so the outputs are
  byte-stable across runs.
- `paper/replication/README.md` — how to run the replication.
- `[paper]` extra in `pyproject.toml` for `vl-convert-python` +
  `jupyter`. Required only to render altair charts to SVG; not part
  of `[viz]` since the static-export driver is heavy (Rust binary).
- New `paper` CI job runs `reproduce.py` on every push, asserts the
  output JSON has the documented schema, and uploads the figures +
  JSON as an artefact.
- JSS template files (`jss.cls`, `jss.bst`, `jsslogo.jpg`) copied in
  from the official template for ease of local compilation.

### Added — Phase 8a: documentation site
- `docs/index.md` rewritten as a real landing page (what it's for, what
  it's not, design principles, quick taste).
- `docs/getting-started.md` expanded with installation, Corpus
  construction, slicing, compare verbs, track, before/after.
- `docs/design.md` — the three-layer architecture and the two-Protocol
  extension model. Includes the optional-extras table.
- `docs/statistical-methods.md` — what each metric in the package
  computes, why these defaults, and the full reference list (Dunning,
  Hardie, Gabrielatos, Wilson, Juilland, Gries, Rychly, Church-Hanks,
  Daille, Wagner, Killick, Hamilton, Giulianelli, Newcombe, Schonemann).
- `docs/multilingual.md` — concrete adapter snippets for spaCy, Stanza,
  jieba, fugashi. Covers tokenizer-swap idiom and multilingual SBERT.
- `docs/api/temporal.md`, `docs/api/semantic.md`, `docs/api/viz.md` —
  mkdocstrings-driven reference pages for the new modules.
- `mkdocs.yml` nav updated to surface the new pages.
- `mkdocs build --strict` passes (already running in CI's docs job).

### Added — Tutorial extension (covers Phases 6 + 7)
- Tutorial corpus redesigned: 19 years (2005–2023) with a 4× publication-
  volume swap at the engineered 2016 event, giving PELT and ITS real
  signal to find rather than a degenerate same-template-every-year
  pattern.
- New sections added: `### 6a Changepoint detection`,
  `### 6b Interrupted time series`, `## 8 Semantic shift via embeddings`,
  `### 8a Neighbourhood drift`. Uses `HashEmbedder` for reproducibility;
  documented escape hatch to swap in `SBERTEmbedder` for real semantics.
- 34 cells, executed end-to-end via nbconvert in CI on every push.

### Added — Phase 7: changepoint detection + interrupted time series
- `pycorpdiff.temporal.detect_changepoints(series, method, penalty, model)` —
  wraps `ruptures` (PELT / BinSeg / Window). Returns a tidy DataFrame
  of `(period, index, method)` triples; the index of the input series
  propagates so changepoints are reported in their original time
  vocabulary (`pd.Period` and friends). Default penalty is `log(n)`
  for BIC-style automatic selection.
- `pycorpdiff.temporal.interrupted_time_series(series, event_date)` —
  segmented-regression specification (`y_t = β₀ + β₁·t + β₂·post +
  β₃·time_after_event`) via `statsmodels.OLS`. Returns one row per
  coefficient with standard errors, *t*-stats, *p*-values, and 95%
  CIs; level_change (β₂) and slope_change (β₃) are the headline
  estimates.
- `TemporalTrajectory.changepoints(target=None, method, penalty)` and
  `TemporalTrajectory.interrupted_time_series(event_date, target=None)`
  — wired through the trajectory object; multi-target trajectories
  require explicit `target=`, single-target ones use it automatically.
- Both wrappers raise friendly `ImportError` pointing at the
  `[temporal]` extra when ruptures or statsmodels isn't installed.
- 17 new tests (193 total): changepoint detection on a synthetic step
  series lands within ±2 indices of the engineered breakpoint; ITS
  recovers a +5 level jump with p < 0.01 and a +0.5 slope change with
  p < 0.01; trajectory-level end-to-end test detects the engineered
  2010 discourse shift in a 40-year synthetic corpus.
- CI install line widens to `[dev,viz,temporal]`.

### Added — Phase 6: semantic shift via embeddings
- `pycorpdiff.semantic.semantic_shift(a, b, target, embedder, window, align)`
  — averaged contextual embeddings. For every occurrence of the target
  in each corpus, the surrounding window is encoded as a sentence and
  averaged into a corpus-specific centroid; cosine distance between
  centroids is the reported shift.
- `pycorpdiff.semantic.neighborhood_drift(a, b, target, k, embedder, window, min_count)`
  — top-k contextual neighbours in each corpus and a tidy `(neighbor,
  sim_a, sim_b, rank_a, rank_b, drift, status)` table where ``status``
  is one of ``"shared"`` / ``"gained_in_a"`` / ``"lost_in_a"``.
- `pycorpdiff.semantic.procrustes_align(source, target)` — Schönemann's
  closed-form orthogonal Procrustes (SVD-based). For Hamilton-style
  independently-trained diachronic embeddings.
- `Comparison.semantic_shift(target, embedder, window, align)` wired
  end-to-end, returning a populated `SemanticShiftResult`.
- `pycorpdiff.HashEmbedder(dim=32)` — deterministic seed-derived
  embedder for offline demos and reproducible tests. Maps each input
  string to a unit vector via SHA-256-seeded RNG. No semantic signal,
  but perfect for verifying that the orchestrators (averaging, cosine,
  Procrustes) wire up correctly without paying torch download time.
- `pycorpdiff.SBERTEmbedder(model_name="all-MiniLM-L6-v2")` — lazy
  sentence-transformers wrapper. The base install stays light;
  construction is free, the actual model loads on first `.encode()`.
  Friendly `ImportError` if the `[semantic]` extra isn't installed.
- 25 new tests (176 total): Procrustes identity / known-rotation /
  norm-preservation / Frobenius-minimisation; HashEmbedder
  determinism / shape / orthogonality; semantic_shift identity
  (distance≈0) / different-contexts (distance>0) / swap-symmetry /
  procrustes-end-to-end; integration through `Comparison.semantic_shift`.

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
