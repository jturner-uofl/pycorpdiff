# Changelog

All notable changes to `pycorpdiff` are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0a29]

Feature release: **explainable sense-drift detection** — the diachronic
counterpart to `induce_senses`. Where WSI partitions a single snapshot
and `semantic_trajectory` tracks one term's centroid, `sense_drift`
detects *when* a corpus's sense distribution drifts away from a
reference period and *what kind* of change it is. The design fuses
concept-drift detection (margin-density monitoring; Sethi & Kantardzic
2017) with lexical-semantic-change detection (cluster contextual
embeddings, track over time; Giulianelli et al. 2020, Montariol et al.
2021, Schlechtweg et al. 2020) and out-of-distribution scoring (Lee et
al. 2018), with two-sample-style flagging in the spirit of Rabanser et
al. (2019).

### Added

- **`sense_drift(items, embeddings, time_col, reference=..., k=...)`** —
  fit sense centroids on a reference period, then for every later period
  compute a **Mahalanobis** novelty score to the nearest sense (the
  out-of-distribution "uncertainty region"), a **margin density**
  (fraction outside every known sense), and the **Jensen–Shannon
  divergence** of the period's sense distribution from the reference.
  A period is flagged as drifting when its margin density *or* JSD
  exceeds a reference-calibrated control-chart threshold, confirmed by a
  sustained run (`min_run`) to suppress isolated spikes. Embeddings are
  bring-your-own; clustering/covariance use scikit-learn (`[semantic]`).
- **`SenseDriftResult`** — frozen-dataclass Result implementing the
  six-method contract (`to_df`/`plot`/`to_html`/`to_json`/`summary`),
  plus an **explanation layer**:
  - `.change_type` ∈ {`"emergence"`, `"frequency_shift"`,
    `"broadening"`} — a new coherent sense appearing, a re-weighting of
    known senses, or diffuse diversification.
  - `.drift_terms` — terms most distinctive of the novel material by
    log-ratio vs the reference (the *what* behind the *when*), for any
    change type.
  - `.flagged_records(period=None)` — the uncertainty-region records
    driving the drift, for inspection.
  - `.plot()` — margin density over time with drift-flagged periods.

### Notes

- `sense_drift` detects *temporal* sense change; it is complementary to
  `induce_senses`, which catches *static* classifier blind spots. They
  answer different questions and are not interchangeable.

## [0.1.0a28]

Feature release: embedding-based **word-sense induction** (WSI) and
reference-label auditing. Motivated by the need to defend a hand-built
sense classifier (regex / keyword buckets) with an independent,
unsupervised second opinion rather than trusting the buckets on faith.

### Added

- **`induce_senses(items, embeddings, ...)`** — cluster bring-your-own
  embeddings into senses and return a `SenseInductionResult`. Embeddings
  are always supplied by the caller (never embedded internally), keeping
  the base install light and model pinning / caching in the caller's
  hands. Clustering is deterministic by default (seeded k-means, or
  seed-free agglomerative); UMAP/HDBSCAN are deliberately avoided so the
  audit is byte-stable. `k` defaults to silhouette-selection over a
  range, or pass an explicit `k` (e.g. `k = n_reference_buckets` for a
  square cross-tab). Supports document-level and token-in-context
  (`unit="token"` + `item_to_doc`) granularity.
- **`SenseInductionResult`** — frozen-dataclass Result implementing the
  six-method contract (`to_df`/`plot`/`to_html`/`to_json`/`summary`),
  plus three audit operations:
  - `.agreement_with(reference_labels)` → `SenseAgreement` (adjusted
    Rand index, V-measure, homogeneity/completeness, contingency table)
    — quantifies how far an unsupervised partition agrees with a
    hand-built labelling.
  - `.leakage_audit(reference_labels, k)` → the records whose embedding
    geometry most disputes their reference label (deterministic,
    targeted counterpart to a random spot-check).
  - `.share_over_time(freq)` → induced-sense share per period (the
    computed counterpart to a hand-built sense-fraction trajectory).
- **`SenseAgreement`** — small Result for the agreement metrics above.
- `embedding_meta=` passthrough for reproducibility-manifest provenance
  (model, revision, vector hash) echoed onto the result.

`scikit-learn` (already in the `[semantic]` extra) is required and
imported lazily, so importing `pycorpdiff` never pulls it.

### Fixed

- **`fetch_histwords_decade` now loads the COHA subset.** The loader
  assumed a single per-decade filename layout (`{decade}.pkl` /
  `{decade}.npy`, as used by `eng-all` / `fiction`), but the COHA
  archive ships `{decade}-vocab.pkl` / `{decade}-w.npy`, so
  `source="coha"` raised `FileNotFoundError` even after a successful
  download. The loader now resolves either layout (bare form tried
  first, so the working sources are unaffected). This unblocks the
  package's own COHA cross-validation path against
  `HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990`. The unit-test `_fetch`
  mock previously reproduced the assumed naming, which is why CI never
  caught it; it now exercises both layouts.

## [0.1.0a27]

Iter-3 audit fix bundle. The package math is solid (G² Dunning/Rayson
match a hand-derivation to 1e-13; BH q-values match exactly; bootstrap
CIs replicate to 1e-14; the iter-2 four-column simultaneous-CI contract
holds). The *naming* and *interpretation* layers were where the iter-3
findings landed. All three MAJOR findings and both MINORs are
addressed here.

### Fixed (package surface) — BREAKING in alpha contract

- **`causal_impact` is now honestly named.** Iter-3 finding G.10
  determined that the engine is
  `statsmodels.tsa.UnobservedComponents` (Kalman-filter MLE) — a
  frequentist state-space model, **not** a Bayesian BSTS. All prose
  ("Bayesian state-space", "credible intervals", "joint posterior")
  has been rewritten:
  - module / function / class / parameter docstrings now describe an
    MLE-fit local-linear-trend state-space model with Wald-type
    asymptotic intervals and MLE-conditional MC paths;
  - the Brodersen (2015) citation is retained as a framework
    reference with an explicit disclaimer that pycorpdiff is the
    no-control, frequentist simplification (not the spike-and-slab
    Bayesian variant);
  - `metrics["posterior_prob_no_effect"]` renamed to
    `metrics["p_no_effect_mc"]` (the value semantics — a two-sided
    MC p-value-style summary — are unchanged; only the name);
  - `summary()` output: `95% CrI [...]` → `95% CI [...]`,
    `P(no effect): ...` line gains a clarifying suffix
    `(MC, MLE-conditional; not a Bayesian posterior)`.
  - Existing callers reading `metrics["posterior_prob_no_effect"]`
    need to rename to `metrics["p_no_effect_mc"]`. The `summary()`
    method falls back to the old key for backward compatibility.

- **BH and Bonferroni are now NaN-safe.** Iter-3 finding H.1: a
  single ``NaN`` p-value silently propagated to *every* adjusted
  output position. New contract: ``NaN`` inputs pass through to
  ``NaN`` outputs; ``m`` (the test count) is the number of non-NaN
  inputs. Two new tests in `test_correction.py`.

### Hardened (case studies)

- CBD case-study §7, §9.6, §9.6a, §9.6b, §9.6c prose rewritten to
  match the renamed semantics (BSTS → state-space, CrI → CI,
  credible → interval) — no analytical numbers change.
- CBD §9.1c (iter-3 finding G.12) section heading and prose
  rewritten to acknowledge the pool-heterogeneity caveat: the
  "known null" framing was incorrect because the pool concatenates
  2011-12 + 2019-20 cohorts; the test is now described as
  "approximate-null coverage under heterogeneous-pool re-split"
  with explicit interpretation of what coverage near 0.95 does and
  does not certify.
- Asylum §0c (iter-3 finding A.3) prose rewritten: typed
  references are precise to ~12 decimal digits (worst observed
  abs-error 1.77e-11), not 15. The `< 1e-10` assertion still
  passes comfortably.
- `examples/_cache/build_hansard_asylum.py` (iter-3 finding A.7):
  surviving "JSS narrative-audit case study" prose mention replaced
  with "asylum case study (examples/jss_case_study.ipynb)".

### Tests

611 unit tests pass (was 607; +4 NaN-safety tests for BH /
Bonferroni). All causal_impact tests updated to the new metric key.

## [0.1.0a26]

Bug-fix release surfaced by the CBD case-study iteration-2 external
audit. The simultaneous-CI return contract was wrong: under
`simultaneous_ci=True`, pycorpdiff replaced the per-term percentile
CI columns with the simultaneous bounds, returning only one pair
under the same column names. Code that asked for both perspectives
in one call (read top-ranked rows with simultaneous CIs *and* report
per-term CIs for pre-specified terms) silently lost the per-term
inference because the column names did not change.

### Changed (BREAKING — alpha contract)

- **`Comparison.keyness(simultaneous_ci=True)` now returns both
  per-term AND simultaneous CI columns.** New column contract:
  - `simultaneous_ci=False` (default): only `g2_ci_lower` /
    `g2_ci_upper` are produced — per-term percentile bounds, as
    before.
  - `simultaneous_ci=True`: per-term columns `g2_ci_lower` /
    `g2_ci_upper` remain populated, *and* new columns
    `g2_ci_lower_simultaneous` / `g2_ci_upper_simultaneous` carry
    the Westfall-Young studentized-max bounds.

  Old behaviour put the simultaneous bounds in `g2_ci_lower` /
  `g2_ci_upper` when `simultaneous_ci=True`, masking the per-term
  bounds. Downstream code that read `g2_ci_lower` after passing
  `simultaneous_ci=True` to get simultaneous bounds will need to
  rename to `g2_ci_lower_simultaneous`. The default
  (`simultaneous_ci=False`) is unchanged.

  Existing test `test_simultaneous_ci_widens_per_term_ci` updated
  to read both column pairs off a single `simultaneous_ci=True`
  call. New test `test_simultaneous_ci_returns_both_column_pairs`
  asserts the new contract.

## [0.1.0a25]

Chart-rendering polish on the visualisation layer. No analytical
results change.

### Fixed (package surface)

- **Signed-axis tick labels now render with an ASCII hyphen
  (`-`) instead of the Unicode minus (U+2212).** Vega/D3 formats
  negative numbers with the typographic minus `−` (U+2212) by
  default. That is technically correct UTF-8, but some notebook
  viewers / PDF pipelines mis-decode it (showing the raw bytes
  `\xe2\x88\x92`), making axes look corrupted. All signed-axis
  plots — `keyness` volcano + bar, `collocation_shift` diverging
  bar, and `causal_impact` pointwise/cumulative effect panels —
  now carry an axis ``labelExpr`` that rewrites U+2212 to ASCII
  `-`. The collocation diverging-bar axis title is also changed
  from "Shift (A − B)" to ASCII "Shift (A - B)". Underlying data
  values (tooltips, ARIA labels) are unaffected; only the rendered
  tick text changes.

### Hardened (asylum case study)

- **Uniform, larger chart sizing.** All case-study charts are sized
  to a consistent ~1100 px width (up from a mix of 600-820 px). An
  earlier blind 2× pass had inflated some charts to 1640 px (which
  broke the §5.5 diverging-bar layout) while leaving the four
  `.plot()`-default charts (§§ 5.7, 5.8, 5.9, 5.10) at their small
  ~600 px defaults. Every chart now passes an explicit width;
  the §5.11 heatmap and §5.12 network keep their natural aspect.
- **§5.11 heatmap colour legend** also rewrites the z-score
  negative labels to ASCII via ``labelExpr``.
- **Burstiness plot uses a contrasting baseline-vs-burst palette.**
  ``burstiness_plot`` previously coloured the ordinal burst-state
  bands with the ``"reds"`` scheme, which maps states 0 and 1 to two
  near-identical pale pinks — impossible to tell apart. State 0
  (baseline / no burst) is now a neutral grey and states 1+ escalate
  through warm reds, so burst periods are unmistakable; the legend
  lists only the states actually present. Bar opacity bumped 0.35 →
  0.55 for legibility.
- **§5.1 party bar uses a distinguishable UK-party palette.** The
  default 6-category altair scheme rendered the pale/grey
  categories too similarly to tell apart; the by-year-and-party
  bar now maps each party to a convention-aligned, high-contrast
  colour (Con blue, Lab red, LD orange, SNP yellow, crossbench
  dark grey, unattributed light grey).
- **Custom ASCII SVG renderer in the notebook.** §0 setup
  registers a vl-convert renderer with a d3 format locale whose
  ``minus`` is the ASCII hyphen, so *every* number in *every*
  chart — axis labels, legends, **and hover tooltips / ARIA
  labels** — renders with an ASCII `-` rather than U+2212. This
  is belt-and-suspenders over the per-axis ``labelExpr`` (which
  protects axes for all package users on any renderer); the
  notebook renderer additionally cleans the tooltip + ARIA data
  values that ``labelExpr`` cannot reach. Verified: zero U+2212
  anywhere in any embedded SVG.

## [0.1.0a24]

A package warning-noise fix plus two asylum case-study presentation
improvements. No analytical results change.

### Fixed (package surface)

- **`causal_impact` no longer emits a stack of statsmodels
  `ValueWarning`s on every fit.** Boolean-masking the pre-event window
  dropped the `DatetimeIndex` frequency, so statsmodels re-inferred it
  on each fit and warned ("date index has no associated frequency",
  "no supported index … integer index", "inferred frequency QS-OCT").
  The pre-event slice is now re-tagged with the inferred frequency
  before fitting. Purely cosmetic for the numbers — the MLE fit and
  forecast values are byte-identical; only the warning noise is
  removed (verified: P(no effect) on the §5.8 *illegal* series is
  0.100 before and after). Repeated `causal_impact` loops
  (leave-one-year-out, placebo sweeps) no longer produce a wall of
  red.

### Hardened (asylum case study)

- **§5.8 causal-impact cells suppress residual warning noise and
  report convergence honestly.** The cells now wrap fitting in
  `warnings.catch_warnings()` so the notebook output is clean, while
  §5.8e additionally **counts and reports how many leave-one-year-out
  BSTS fits did not fully converge** — cleaning the display without
  hiding information (non-convergence is itself relevant to the §5.8e
  instability finding).
- **§6 audit scoreboard split into two verdict axes.** The single
  ✓/✗ column conflated "did the software execute correctly?" with
  "did the substantive pre-registered prediction hold?". These are
  now separate columns: **Executed** (38/38 ✓ — zero software
  failures) and **Prediction held** (33/38). The five ✗ live only in
  the substantive-prediction axis, making explicit that they are the
  diagnostics *correctly surfacing real data/method limitations*, not
  package bugs. Addresses the reviewer perception that ✗ rows looked
  like the package failing.

## [0.1.0a23]

One substantive package feature (speaker-clustered bootstrap) plus
four asylum case-study rigour items, in response to a fourth-round
reviewer audit. The package feature corrects a real understatement
of CI width on hierarchical corpora.

### Added (package surface)

- **`compare.keyness(ci="bootstrap", cluster_col="speaker")`** — a
  cluster-robust bootstrap that resamples *clusters of non-independent
  documents* (e.g. all speeches by one speaker as a block) rather than
  individual documents. Hansard speeches are nested in speakers nested
  in parties; IID document resampling treats every speech as
  independent and **understates** the CI width — the effective sample
  size is closer to the speaker count than the speech count.
  Cluster-robust CIs are wider and more honest for hierarchical text
  corpora. The underlying :func:`pycorpdiff.keyness.bootstrap.bootstrap_g2_ci`
  gains the same ``cluster_col`` parameter. New tests
  ``test_cluster_col_widens_ci_vs_iid``,
  ``test_cluster_col_recorded_in_params``,
  ``test_cluster_col_missing_column_raises``.

### Hardened (asylum case study)

- **§5.3 demonstrates the speaker-clustered bootstrap.** The headline
  Con-vs-Lab keyness now reports both IID and speaker-clustered CIs so
  the reader sees how much the hierarchical dependence widens the
  intervals. Addresses the reviewer's "observations are not IID"
  objection at the package level.
- **§5.1b institutional-drift audit** (new). Sentence-length drift,
  contribution-length drift, metadata-missingness over time, and
  Commons/Lords balance over the 2010-2023 window — the
  infrastructure-drift checks any longitudinal corpus study needs.
- **§5.13 synthetic-signal injection + minimum-detectable-effect**
  (new). Plant a known lexical campaign into a copy of the corpus at
  known dates and known effect sizes; show keyness + burstiness
  recover it; estimate the minimum detectable effect and the empirical
  false-discovery rate. This is a pure software-validation move: does
  the tool detect what it claims to, and how small a signal can it
  find?
- **Tightened soft falsifiers.** The §5.6 "monotonic-ish increase"
  falsifier (reviewer-flagged as an analyst escape hatch) is now a
  quantified Kendall-τ threshold on the post-2020 trajectory; other
  qualitative falsifiers in §0b are labelled quantitative vs
  qualitative.
- **§5.10 forecasting reframed.** Explicitly labelled a *capability
  demonstration* of the package's forecast surface (with the §5.10b
  backtest showing it beats a naive baseline), not a substantive
  forecast of asylum discourse — addressing the "benchmarking theater"
  critique by stating the section's actual role.

### Reviewer-#4 items intentionally not implemented (with rationale)

These validate substantive claims about UK asylum discourse, not the
software, and belong to a separate computational-social-science paper:

- **Human annotation / construct validity** (blind coders, inter-rater
  agreement). Out of scope for a software paper; excluded in earlier
  rounds.
- **Parallel-corpus / conditioning expansion** (immigration, refugees,
  borders, small boats). Separate paper; the §5.0 caveat frames the
  conditioning honestly.
- **External predictive validation** (polling, asylum-application
  counts, court rulings). A substantive social-science question, not a
  software-correctness one.
- **Parliament ≠ population** (newspaper / manifesto / social-media
  comparison). Separate multi-corpus study.
- **Out-of-domain replication** (US / German / Canadian legislatures).
  Explicitly a separate methods paper.
- **Full cross-model embedding sweep** (SGNS vs SVD vs transformer) and
  **full preprocessing multiverse**. The HashEmbedder-vs-SBERT dual
  (§5.6) and the ``min_count`` sweep (§5.3e) are one-axis slices;
  the full grids are compute-heavy and documented as researcher-DoF
  limitations in §6 rather than executed.

## [0.1.0a22]

Focused tightening pass on the narrative-audit case study based
on reviewer-#3 feedback. Four notebook-level changes; no package
surface changes.

### Hardened (asylum case study)

- **Version-string consistency.** Pinned-version prose was lagging
  behind the executed manifest (0.1.0a19 prose / 0.1.0a21 output);
  now both align on 0.1.0a22.
- **§5.8c placebo-date sweep redesigned to use only eligible dates.**
  In 0.1.0a21 the safety rails (introduced this release) blocked
  3 of 5 placebo dates because of asymmetric pre/post windows, weakening
  the sweep. The redesigned sweep predefines 5 placebo dates all
  inside the eligible window (≥ 15 pre, ≥ 8 post, ratio ≤ 5) so the
  test now cleanly asks "does BSTS fire on well-conditioned non-event
  dates?" without being noisily dominated by safety-rail rejections.
- **§5.6d Leave-year-out SBERT trajectory.** Adds a new stability
  diagnostic: drop each year 2010-2023 in turn, refit the §5.6 SBERT
  trajectory, report the per-year envelope. This is the SBERT-grade
  version of §5.6c (which used HashEmbedder for tractability); the
  diagnostic now exists in both forms.
- **Explicit audit / substantive labels** on every §5.x sub-section.
  Each is now tagged as either *integration test of `pycorpdiff`* (audit
  cell), *substantive claim about UK asylum discourse* (research cell),
  or *both*. Reviewer #3 asked for this separation explicitly; the
  current version is a stronger reviewer-defence as a result.

### Reviewer-#3 items intentionally not implemented (with rationale)

- **Human inter-rater annotation.** Explicitly excluded from scope in
  earlier rounds. §6 "what this notebook is not" names this.
- **Broader-migration corpus replication** (*refugee*, *migrant*, *border*,
  *visa*, *Channel*). This is a separate paper; the §5.0 corpus-
  conditioning caveat already names the limitation, and §5.7b / §5.8b
  exercise some within-corpus generalisation on *migrant*.
- **Commons-only + Lords-only replication for every section.** §5.5b
  documents the structural chamber issue (Jaccard 0.00); replicating
  it across §§ 5.6-5.12 adds ~30 cells without new information once
  the §5.5b result is in.
- **Alternate tokenizers / embedders.** Tokenizer swap changes what
  counts as a token, not a sensitivity check. The HashEmbedder vs
  SBERT dual in §5.6 already exercises embedder-class sensitivity.

## [0.1.0a21]

Inferential-hardening pass on the narrative-audit case study, in
response to the second-round reviewer audit. Seven new sub-sections
address corpus-conditioning, placebo-date sweeps, embedding stability,
party-switcher leverage, chamber stratification, and leave-one-year-out
robustness on the causal-impact section. No breaking changes to the
package surface.

### Hardened (asylum case study)

- **§5.0 Scope and corpus conditioning.** New framing section
  immediately after §0c that names explicitly what the case study
  studies (asylum-*conditioned* parliamentary discourse) and what
  it does not claim (broader UK migration discourse). The conditioning
  is unavoidable for a focal-term case study; the framing makes the
  claim boundaries visible.
- **§5.3f Top-K-speaker leverage check.** Exclude the 10 most prolific
  Commons Con+Lab contributors and re-run §5.3 keyness. Report Jaccard
  overlap of the top-15 terms; if the keyness signal is small-set-
  leverage-driven (party-switchers, prolific ministers), the overlap
  collapses.
- **§5.5b Chamber-stratified collocation.** Replicate the §5.5
  collocation shift on *asylum* separately for Commons and Lords. If
  effects differ substantively by chamber, the Lords/Commons mixing
  caveat in §6 is structural rather than incidental.
- **§5.6c Bootstrap stability of semantic trajectory.** Resample
  documents within each year-period 20 times, refit the HashEmbedder
  trajectory on each resample, and report a year-by-year distance CI
  envelope. SBERT is too expensive for this loop; HashEmbedder is
  appropriate because the check is on the trajectory machinery, not
  the specific embedder.
- **§5.8c Placebo-date sweep.** §5.8c now sweeps five placebo dates
  (2011-03-15, 2013-06-01, 2017-09-01, 2019-04-15, 2022-08-01)
  alongside the real referendum date (2016-06-23). If only the real
  date returns low P(no effect), BSTS is not firing on arbitrary
  dates.
- **§5.8d Second control term.** *fisheries* (UK-procedural, not
  migration-policy-driven) joins *committee* as a donor-series
  control. Two non-migration controls strengthen the §5.8 specificity
  claim.
- **§5.8e Leave-one-year-out for causal_impact.** Drop each year in
  turn (2010-2023) and refit causal_impact at 2016-06-23. Report the
  distribution of P(no effect) across the 14 LOY runs.

### Notebook structure

- §0b pre-registered table grows from 22 → 28 rows (six new
  predictions for the new sub-sections).
- §6 audit scoreboard grows accordingly; tally remains at all-✔ if
  the new predictions hold (recorded honestly on first run).

### Reviewer-#2 items intentionally not implemented

- **Tokenizer / OCR-noise / punctuation-stripping perturbation tests.**
  Hansard is text, not OCR; tokenizer swaps fundamentally change what
  counts as a token rather than testing sensitivity to a parameter.
- **Human inter-rater annotation.** Standard CDA validation step;
  out of scope for a methodological-demonstration notebook. The
  notebook explicitly states "this is not a refereed CDA study" in
  §6 "what this notebook is *not*".
- **Global FDR across §5.1-§5.12.** Sections answer heterogeneous
  empirical questions; a global FDR across them is not methodologically
  meaningful. Local FDR (BH) is in §5.3 + §5.11; FWER simultaneous CI
  is in §5.3 via 0.1.0a20's ``simultaneous_ci=True``.
- **Hidden-event masked validation.** The case study is method-
  validation against documented events, not method-discovery of
  unknown structure. Discovery validation is a separate paper.

## [0.1.0a20]

One substantive analytical addition driven by an audit finding in the
asylum case study, plus the corresponding case-study fix. No breaking
changes.

### Added

- **`compare.keyness(ci="bootstrap", simultaneous_ci=True)`.** A
  Westfall-Young studentized-max simultaneous CI option on the keyness
  bootstrap. The existing per-term percentile CIs are well-calibrated
  for any single term named in advance, but anti-conservative when
  read off the top of a sorted keyness table (post-selection
  inference). The new option returns CIs with family-wise (1 − α)
  coverage across the whole vocabulary — slightly wider than per-term
  CIs, but valid to report on the top-ranked terms. Discovered via the
  §5.3d Monte-Carlo coverage check on the narrative-audit
  notebook, which showed per-term percentile CIs on top-ranked terms
  covered zero in only 63 % of known-null replicates (vs nominal
  95 %). With ``simultaneous_ci=True`` the coverage on the top term
  returns to nominal. New tests
  ``test_simultaneous_ci_widens_per_term_ci`` and
  ``test_simultaneous_ci_recorded_in_params``.

### Hardened (asylum case study)

- **§5.3 keyness now uses `simultaneous_ci=True`** so the headline CIs
  reported in the case-study table are valid for the top-ranked
  vocabulary the reader actually sees.
- **§5.3d Monte-Carlo coverage cell** updated to compare both modes
  side by side (per-term 63 %, simultaneous ~95 %), making the
  failure-mode and its fix visible in the audit scoreboard.
- **§6.5 "Where pycorpdiff fails"** updated: the post-selection
  caveat on per-term bootstrap CIs is now a *closed* failure case,
  cross-referenced to ``compare.keyness(simultaneous_ci=True)``.

### Docs

- ``docs/statistical-methods.md`` gains a "Simultaneous CIs" sub-
  section deriving the Westfall-Young max-T statistic and explaining
  when to use it.

## [0.1.0a19]

Two visual-quality fixes on the asylum case study, one of them backed by
a small package API addition. No breaking changes.

### Fixed (asylum case study)

- **§5.11 heatmap was rendering all-NaN** because the post-`reset_index`
  rates DataFrame was built with `pd.DataFrame({...}, index=term_series)`
  while the column Series carried int positional labels — pandas aligned
  by index, found zero overlap, and produced NaN throughout. The build
  script now keeps the term-indexed frame and builds rates from it
  directly, so the per-party z-scored heatmap renders correctly.
- **Chart sizes bumped throughout** the case-study notebook (most
  charts go from 560-600 wide to 820 wide; corresponding height bumps).

### Added

- **`cooccurrence_network(..., stop_words=...)`.** Without a filter,
  the function selected the top-N vocabulary by raw frequency — on
  any English corpus this is closed-class function words (`the`,
  `and`, `of`, ...). The case-study network was therefore a graph of
  stop-words with the focal `asylum` node buried in the middle.
  ``stop_words`` is applied before the top-N cut. New test
  ``test_stop_words_excludes_terms_from_vocabulary``.

## [0.1.0a18]

A data-quality bug fix on ``fetch_hansard`` and a substantially
hardened narrative-audit case study. No breaking changes.

### Fixed

- **``fetch_hansard`` was returning HTML-contaminated text.** The
  Hansard search API embeds ``<em>``, ``<span>``, ``<strong>``,
  ``<p>``, ``<td>``, ``<tr>``, ``<th>``, ``<TableWrapper ...>`` tags
  in contribution text. After naïve tokenisation, the tag names
  appeared as apparent English words (``em`` showed up ~11,000 times,
  ``span`` ~14,500 times in a moderate corpus), systematically
  polluting every keyness / collocation / network / lexical-diversity
  measurement. ``_default_parse_search_response`` now strips HTML
  tags via regex and decodes HTML entities before the text is stored
  as the canonical document body. Two new tests cover the cleanup
  (``test_default_parser_strips_html_markup_from_text``,
  ``test_default_parser_handles_hansard_table_wrapper``).

### Hardened (asylum case study)

The ``examples/jss_case_study.ipynb`` notebook is substantially
expanded for paper-grade rigour:

- **§ 0a Reproducibility manifest** — every package version, every
  seed, the data-fetch date are recorded at the top of the notebook
  so the run is byte-reproducible under matching versions.
- **§ 0b Pre-registered expectations** — a table of what each method
  is expected to surface, recorded before running, so the
  validation paragraphs in §§ 5.x test against an *a priori* prior
  rather than post-hoc rationalisation.
- **§ 0c Cross-package validation** — agreement of pycorpdiff's
  signed-G² against six canonical contingency-table references from
  Rayson & Garside (2000) to < 0.01 absolute error. Asserts hard;
  if regressed, every G² claim below is suspect.
- **§§ 5.3, 5.11 Commons-only filter** — removes the Lords-overflow
  confound (``noble``, ``lord``) that contaminated the previous
  Conservative-vs-Labour comparison.
- **§ 5.6 Paper-grade SBERT trajectory** alongside a HashEmbedder
  reproducibility check — the SBERT run is what goes in a publication;
  the HashEmbedder run regression-tests the trajectory machinery.
- **§ 5.7a Sensitivity to burst factor `s`** — sweep across
  {1.5, 2.0, 2.5, 3.0} to show burst stability.
- **§ 5.7b Multi-term replication** on *migrant* — guards against
  asylum-specific artefacts.
- **§ 5.8a Sensitivity to PELT penalty** — sweep across an order
  of magnitude to show changepoint stability.
- **§ 5.8b Multi-term replication** on *migrant* — repeats the
  causal-impact test on a second focal term.
- **Per-section "Falsifier" paragraphs** — explicit statements of
  what observation would invalidate each section's claim.
- **§ 6 Reproducibility receipts** — final summary of seeded results,
  acknowledged limitations (the ``enrich_party`` `latestParty` issue,
  the 10-quarter pre-event window for causal_impact, the Commons /
  Lords mixing in sections that don't filter), and an explicit
  scope statement.
- **31 / 31 code cells succeed** end-to-end on the hardened build.

## [0.1.0a17]

Bug fix + a substantive new example notebook. No breaking changes to
the public API; a few non-breaking additions to ``fetch_hansard``.

### Fixed

- **``fetch_hansard`` was hitting the wrong endpoint.** The default
  ``SEARCH_DEBATES_PATH = "/search/debates.json"`` returned debate
  *metadata* (titles, sitting dates) without the speech text itself,
  so every ``fetch_hansard`` call returned an empty :class:`Corpus`
  even when the API was reachable. The endpoint has been corrected to
  ``/search/contributions/Spoken.json``, which returns
  ``ContributionTextFull`` along with ``MemberId``, ``MemberName``,
  ``DebateSection``, ``SittingDate``, and ``House``.

### Added

- **``fetch_hansard`` automatic pagination.** ``max_results`` larger
  than the API's per-request ``page_size`` (default 50) now walks
  pages internally and stops on ``TotalResultCount`` exhaustion or
  on a short final page.
- **``fetch_hansard(..., enrich_party=True)``.** Hansard contributions
  carry ``MemberId`` but not party affiliation. When this flag is set,
  one Members-API call per unique ``MemberId`` populates ``party`` and
  ``party_abbrev`` columns on the returned corpus. ~0.3 s per unique
  member; required for downstream cross-party comparative work.
- **Returned columns.** ``fetch_hansard`` now returns ``text``,
  ``speaker``, ``member_id``, ``party``, ``date``, ``debate_title``,
  ``house``, ``hansard_id`` — plus ``party_abbrev`` when
  ``enrich_party=True``. (Previously: ``text``, ``speaker``,
  ``party``, ``date``, ``debate_title``, ``hansard_id``.) The two new
  metadata columns enable Commons-vs-Lords filtering and
  per-member-id enrichment without re-fetching.
- **narrative-audit case study.** New example notebook
  ``examples/jss_case_study.ipynb`` plus its data-prep helper
  ``examples/_cache/build_hansard_asylum.py`` and a cached parquet
  of 9,000 spoken contributions on *asylum* from UK Hansard
  2014-2023. The notebook walks twelve analytical sections
  end-to-end against the real data, each section closing with a
  *Validation* paragraph that compares the algorithm's blind output
  to historically attested events. 25 of 25 code cells execute
  successfully.

## [0.1.0a16]

One analytical addition: lexical-diversity metrics with a temporal arc.
No breaking changes.

### Added

- **Lexical diversity (`pcd.lexical_diversity`).** Four metrics
  reported side-by-side: TTR (uncorrected baseline), MATTR (Covington
  & McFall 2010, moving-average TTR), MTLD (McCarthy & Jarvis 2010,
  mean factor length to a running-TTR threshold), and HD-D (McCarthy
  & Jarvis 2007, hypergeometric expected vocabulary in a 42-token
  sample). Pure-math primitives :func:`pycorpdiff.lexical.ttr`,
  :func:`mattr`, :func:`mtld`, :func:`hdd` are individually
  importable.

  Without ``freq=``, returns a pooled :class:`LexicalDiversityResult`
  with corpus-level metric values and an optional per-document
  breakdown table. With ``freq=`` (e.g. ``"Y"``, ``"Q"``, ``"M"``),
  slices the corpus by period and returns a
  :class:`LexicalDiversityTrajectory` with per-period metric values
  plus a ``.plot()`` that facets one panel per metric so the very
  different scales (TTR ~ 0.5, MTLD ~ 100, HD-D ~ 35) stay legible.

  ``ci="bootstrap"`` adds ``ci_lower`` / ``ci_upper`` columns via
  document-level resampling within each period. *Caveat documented
  in the docstring:* MTLD and MATTR are order-sensitive walks, so
  document-level bootstrap can mildly bias their CIs; TTR and HD-D
  give clean percentile bands. New module
  :mod:`pycorpdiff.lexical`; new public exports
  ``lexical_diversity``, ``LexicalDiversityResult``,
  ``LexicalDiversityTrajectory``.

## [0.1.0a15]

Two analytical additions targeting causal-inference and temporal-pattern
use cases. No breaking changes.

### Added

- **Sub-corpus balancing via Coarsened Exact Matching.**
  :func:`pycorpdiff.match` pre-balances two corpora on document-level
  covariates (year, party, topic, speaker-role, …) before keyness,
  collocation, or trajectory analysis. Implements CEM (Iacus, King &
  Porro 2012): numeric covariates are quantile-binned, categorical
  covariates left as-is, strata are formed on the joint coarsened
  key, mismatched strata are dropped, and within each kept stratum
  the over-represented side is subsampled to match the minority
  count ("k-to-k" matching). Returns a :class:`MatchResult` with the
  matched slices plus a per-covariate L1-imbalance diagnostic before
  vs after. Fully reproducible under ``seed``. New module
  :mod:`pycorpdiff.matching`; new public exports ``match``,
  ``MatchResult``.

- **Kleinberg burstiness detection on temporal trajectories.**
  :meth:`TemporalTrajectory.burstiness` segments a target's per-period
  rate into burst-intensity states via Kleinberg's (1999) multi-state
  HMM. State *i* has rate ``p_0 * s ** i`` (``s`` is the burst factor,
  default ``2.0``); per-period observation cost is the negative
  log-Binomial likelihood; transition cost is
  ``(j - i) * gamma * log T`` when escalating to a higher state and
  zero when de-escalating. A standard Viterbi pass gives the
  minimum-cost state sequence. Returns a :class:`BurstinessResult`
  with per-period state labels, a per-burst summary table, and a
  ``.plot()`` view that overlays state intensity onto the
  trajectory. New module :mod:`pycorpdiff.temporal.burstiness`; new
  public exports ``kleinberg_bursts``, ``BurstinessResult``,
  ``burstiness_plot``.

## [0.1.0a14]

Two analytical additions and a small ergonomic fix on
:meth:`KeynessResult.explain`. No breaking changes.

### Added

- **Bootstrap confidence intervals on keyness G².**
  :meth:`pycorpdiff.compare.Comparison.keyness` accepts
  ``ci="bootstrap"``, optional ``n_boot`` (default 999), ``ci_level``
  (default 0.95), and ``bootstrap_seed``. When requested, the result
  table gains ``g2_ci_lower`` and ``g2_ci_upper`` columns from the
  percentile method (Efron & Tibshirani 1993), with documents as the
  unit of exchangeability. Honors the same ``formula="rayson"`` /
  ``"dunning"`` toggle as the point estimate. New module
  :mod:`pycorpdiff.keyness.bootstrap`.
- **Reference-corpus keyness via** :func:`pycorpdiff.against_baseline`.
  Compare a corpus against a pre-computed aggregated frequency
  baseline (term + count + corpus total), the canonical setup for
  lexicography and discourse analysis. Returns a standard
  :class:`KeynessResult`. Ships one bundled baseline,
  ``"gutenberg_fiction"`` (five Project Gutenberg English-fiction
  texts, 1813–1897, ~500K tokens, ~11K types). User-built baselines
  via :func:`pycorpdiff.baseline_from_corpus`. New modules
  :mod:`pycorpdiff.keyness.baseline` and
  :mod:`pycorpdiff.datasets.baselines`; new public exports
  ``against_baseline``, ``Baseline``, ``load_baseline``,
  ``list_baselines``, ``baseline_from_corpus``.

### Changed

- :meth:`KeynessResult.explain` now gracefully handles results built
  via :func:`against_baseline`, where ``corpus_b`` is ``None``
  (an aggregate frequency list, not a corpus). Falls back to KWIC
  lines from ``corpus_a`` only. The bare-DataFrame case still raises.

## [0.1.0a13] — first public release

The first public alpha of `pycorpdiff` — comparative corpus analysis
for modern Python workflows. Three public verbs (`compare`, `track`,
`compare.before_after`), nine `Result` dataclasses each implementing
the relevant subset of `.to_df / .plot / .explain / .summary /
.to_html / .to_json` (see `docs/design.md` for the per-Result method
matrix), two `typing.Protocol` extension points (`Tokenizer`,
`Embedder`), and opt-in extras for visualisation, semantic embedding,
temporal modelling, polars interop, DuckDB ingestion, 🤗 Datasets,
and notebook rendering.

### Analytical surface

- **Keyness**: signed log-likelihood G² with selectable formula
  (`formula="rayson"` 2-cell shortcut, default; matches the UCREL
  LL Wizard. `formula="dunning"` 4-cell G²; the canonical Dunning
  1993 form used by NLTK and R's `quanteda::textstat_keyness(measure="lr")`).
  Pearson χ², Hardie LogRatio, Gabrielatos %DIFF, BIC-approximated
  Bayes factor (also tracks the `formula=` choice), Juilland D /
  Gries DP dispersion flagging, Benjamini–Hochberg correction,
  stop-word filtering, empirical permutation *p*-values, N-way
  contingency G² via `keyness_multi`.
- **Collocations**: logDice, PMI, t-score, MI³ with Laplace smoothing;
  cross-corpus `collocation_shift`; co-occurrence networks via
  `cooccurrence_network`.
- **Semantic shift**: averaged contextual embeddings, Procrustes
  alignment, multi-period `semantic_trajectory`, `neighborhood_drift`.
  Embedder output shape is validated to catch silently-broken
  embedders before they produce nonsense.
- **Temporal**: Wilson-CI trajectories, offline PELT changepoints,
  online Bayesian changepoint detection, segmented-OLS interrupted
  time series, Bayesian structural time-series causal impact,
  state-space exponential-smoothing forecasting.

### Cross-validated

The package is checked against standard tools by automated test:

- **Rayson's LL Wizard** — hand-derived contingency-table reference
  triples (fast tier; runs on every push).
- **NLTK** `BigramAssocMeasures` — PMI + t-score agreement to ≤ 1e-12
  on every adjacent bigram (slow tier).
- **Scattertext (Kessler 2017)** — behavioural agreement on the 2012
  US Conventions corpus (slow tier).
- **HistWords (Hamilton et al. 2016)** — known-shifter / stable-word
  sanity check on Stanford SNAP COHA decade embeddings; skips
  gracefully when the archive isn't reachable (slow tier).

### Extras

`[viz]`, `[semantic]`, `[temporal]`, `[polars]`, `[duckdb]`, `[nlp]`,
`[huggingface]`, `[notebooks]`, `[all]` are MIT-compatible. A separate
`[showcase]` extra pulls in `pysofra` (GPL-3.0-or-later) for
JAMA-style table polish in the showcase notebook — opt in explicitly
if you accept that licence.

### Infrastructure

Hundreds of tests, `ruff` + `mypy --strict` clean across the source
tree, matrix CI on three Python versions × two operating systems,
plus a slow-tier CI job exercising the cross-validation receipts
against NLTK + Scattertext + HistWords on main pushes.
