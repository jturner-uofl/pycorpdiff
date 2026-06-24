# pycorpdiff

[![PyPI](https://img.shields.io/pypi/v/pycorpdiff.svg)](https://pypi.org/project/pycorpdiff/)
[![Python versions](https://img.shields.io/pypi/pyversions/pycorpdiff.svg)](https://pypi.org/project/pycorpdiff/)
[![CI](https://github.com/jturner-uofl/pycorpdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/jturner-uofl/pycorpdiff/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Comparative corpus analysis for modern Python workflows.**

`pycorpdiff` is the **missing comparative layer** between R's
[`quanteda`](https://quanteda.io/), the closed-source SketchEngine
platform, and the fragmented Python NLP stack
(`nltk`/`spaCy`/`gensim`/`sentence-transformers`). Three public verbs
— `compare(a, b)`, `track(c, term)`, `compare.before_after(c, event)` —
consolidate keyness, collocations, dispersion, temporal trajectories,
changepoint detection, interrupted time series, causal-impact analysis,
forecasting, online changepoint detection, embedding-based semantic
shift, and label-free drift & novelty detection under a single
notebook-native API. Keyness and collocation
results carry their own KWIC evidence: `.explain(term)` returns the
source-text concordances behind any ranked term.

The package answers the questions corpus linguistics, digital humanities,
and computational social science routinely have:

- *How does corpus A differ from corpus B?* — `compare(a, b).keyness()`
- *How has discourse around X evolved over time?* — `track(c, "x").over_time()`
- *What did "migrant" mean in 2005 vs 2023?* — `compare(...).semantic_shift("migrant", embedder=...)`
- *Did this event actually shift the conversation?* — `track(...).causal_impact(event_date=...)`
- *Where is the discourse heading?* — `track(...).forecast(horizon=4)`
- *What's emerging that no keyword list anticipated — and what is it?* — `pcd.knn_density_drift(...)` catches it label-free, `pcd.sense_drift(...)` names and types it

`pycorpdiff` is positioned as **orchestration**, not reinvention.
Tokenizers (`spaCy`, `Stanza`, `jieba`, `fugashi`) and embedders (any
`SBERT`-compatible model) plug in via two `typing.Protocol` extension
points — one-line adapters, no plugin registry. The base install's
direct runtime dependencies are `numpy`, `pandas`, `scipy`, and
`pyarrow`; everything else is opt-in via extras.

> **Status: alpha (0.1.0a33).** Public API is stable for the features
> described below; on PyPI as `pip install pycorpdiff`. Alpha releases
> are intentionally rapid (audit-driven), each shipping fixes and tests
> behind the published version; dependency pins will tighten at beta.

## The three-layer architecture

| Layer | Purpose | Key surface |
|---|---|---|
| **1 — Ingestion + `Corpus`** | get text in, slice it, hash it | `from_dataframe`, `read_csv`, `read_parquet`, `read_txt`, `read_duckdb`, `from_huggingface`, `fetch_hansard`, `Corpus.slice/by_time/__hash__/doc_term_counts(_sparse)/to_polars` |
| **2 — Pure math** | statistics with no I/O | `keyness.{log_likelihood,chi_squared,log_ratio,percent_diff,bayes_factor,permutation_pvalues,keyness_multi,juilland_d,benjamini_hochberg}`; `collocation.{logdice,pmi,t_score,mi_three,collocation_shift,cooccurrence_network}`; `semantic.{HashEmbedder,SBERTEmbedder,semantic_trajectory,neighborhood_drift,induce_senses,sense_drift,knn_density_drift}`; `temporal.{changepoints,interrupted_time_series,forecast,causal_impact,bocpd}` |
| **3 — Verbs + Results** | public API | `compare`, `track`, `compare.before_after`, `keyness_multi`, plus 9 frozen-dataclass Result types each implementing the relevant subset of `.to_df() / .plot() / .explain() / .summary() / .to_html() / .to_json()` |

## Quick start

```bash
pip install "pycorpdiff[viz]"
```

```python
import pycorpdiff as pcd

# Bundled synthetic Hansard-style sample — runs offline, no data download.
corpus = pcd.load_hansard_sample()
immigration = corpus.slice(topic="immigration")

# Which words separate the humanising and criminalising frames?
keyness = pcd.compare(
    immigration.slice(frame="humanising"),
    immigration.slice(frame="criminalising"),
).keyness(min_count=3)

keyness.plot()                # volcano plot — picture the result
# keyness.table.head(10)      # or look at the ranked table directly
# keyness.explain("criminal") # KWIC concordances showing the textual evidence
```

That's the entire surface in five lines: load a corpus, slice it,
compare two slices, plot the result. Every other analytical method —
collocation shifts, semantic drift, temporal trajectories, changepoint
detection, causal-impact analysis, forecasting, co-occurrence networks,
N-way keyness — follows the same shape. See
[the showcase notebook](https://github.com/jturner-uofl/pycorpdiff/blob/main/examples/pycorpdiff_showcase.ipynb)
for the full feature tour, or the cheat sheet below for one-line API previews.

### Cheat sheet — every analytical surface in one block

```python
# Compare verbs (returns Result objects; methods exposed vary by Result)
pcd.compare(a, b).keyness()                                                   # default formula="rayson" (LL Wizard)
pcd.compare(a, b).keyness(formula="dunning")                                  # full 4-cell G² (Dunning 1993; same family as quanteda / NLTK, edge-case tolerance not certified)
pcd.compare(a, b).keyness(ci="bootstrap", n_boot=999)                         # adds g2_ci_lower / g2_ci_upper columns
pcd.compare(a, b).collocation_shift("immigrant")
pcd.compare(a, b).semantic_shift("immigrant", embedder=pcd.SBERTEmbedder())   # [semantic]
# SBERTEmbedder downloads a sentence-transformers model on first call;
# use pcd.HashEmbedder() for offline / deterministic-test settings.

# Reference-baseline keyness (bundled or user-built)
pcd.against_baseline(corpus, "gutenberg_fiction")                             # vs bundled 19th-c. fiction baseline
pcd.against_baseline(corpus, pcd.baseline_from_corpus(reference_corpus))      # vs your own reference

# Sub-corpus balancing — Coarsened Exact Matching before keyness
m = pcd.match(a, b, on=["year", "party"], seed=0)                             # balances A and B on covariates
pcd.compare(m.a_matched, m.b_matched).keyness()                               # like-for-like comparison

# Lexical diversity (TTR, MATTR, MTLD, HD-D) — pooled and over time
pcd.lexical_diversity(corpus)                                                 # pooled corpus-level values
pcd.lexical_diversity(corpus, freq="Y", ci="bootstrap", n_boot=199)           # per-year trajectory + CIs

# Ecology diversity (Hill numbers + rarefaction) — size-fair comparison of
# corpora of different sizes (a bigger corpus has more types just by being bigger).
from collections import Counter
counts = list(Counter(tokens).values())                                       # any abundance vector
pcd.lexical.hill_numbers(counts, q=1)                                          # effective # types (q=0 richness, 1 exp-Shannon, 2 inv-Simpson)
pcd.lexical.rarefaction(counts, sample_size=10_000)                           # expected #types in a 10k-token subsample (Hurlbert 1971)

# Track over time (requires [temporal] for the changepoint + ITS + forecast + causal_impact methods).
# Note: ITS / causal_impact require sufficient pre/post-event periods to fit (min_pre_periods=15,
# min_post_periods=8 by default); the bundled Hansard sample is too small to exercise these
# lines literally -- they are shown here as API previews. See examples/jss_case_study.ipynb
# for a full-corpus run.
tr = pcd.track(corpus, "immigrant").over_time(freq="Y")
tr.changepoints()                                  # offline PELT
tr.changepoints_online(hazard=1/24)                # Bayesian online (Adams & MacKay 2007)
tr.burstiness()                                    # Kleinberg 1999 multi-state HMM — burst-intensity states
# tr.interrupted_time_series(event_date="2016")    # segmented OLS [needs >=15 pre-periods]
# tr.causal_impact(event_date="2016")              # Bayesian counterfactual (Brodersen 2015) [needs >=15 pre-periods]
tr.forecast(horizon=4)                             # 4 periods at the over_time freq (state-space ETS)

# Before / after a known event
pcd.compare.before_after(corpus, event_date="2016-06-23").keyness()

# N-way (≥ 2 corpora) — the four corpora `a, b, c, d` are illustrative placeholders
# (the cheat sheet's `a, b` from the keyness lines above; you supply `c, d`).
# pcd.keyness_multi([a, b, c, d], labels=["A", "B", "C", "D"])

# The discourse as a graph
pcd.cooccurrence_network(corpus, top_n=30).plot()

# Word-sense induction (BYO embeddings) — audit a hand-built sense classifier
# with an unsupervised second opinion. [semantic] extra for scikit-learn.
X = pcd.SBERTEmbedder().encode(df["text"].tolist())        # or any (n, d) matrix you own
senses = pcd.induce_senses(df, X, k=3)                     # k=None -> silhouette-selected
senses.agreement_with(df["regex_label"]).summary()         # ARI / V-measure vs your buckets
senses.leakage_audit(df["regex_label"], k=20)              # records whose geometry disputes the label
senses.share_over_time(freq="Y")                           # computed sense-fraction trajectory

# Sense drift over time — detect *and explain* when a corpus's sense
# distribution changes (concept-drift + lexical-semantic-change methods).
drift = pcd.sense_drift(df, X, time_col="year", reference=range(2000, 2010), k=3)
drift.summary()                                            # onset, change type, distinctive terms
drift.change_type                                          # "emergence" | "frequency_shift" | "broadening"
drift.drift_terms                                          # what drove the drift (log-ratio vs reference)
drift.plot()                                               # margin density + JSD over time, drift flagged

# For inference, calibrate the flag threshold against a label-shuffle null
# (removes the out-of-sample bias of the in-sample chart) + get a p-value:
drift = pcd.sense_drift(df, X, time_col="year", reference=range(2000, 2010),
                        k=3, n_permutations=50)
drift.p_value                                              # permutation p (max margin vs shuffled null)

# The fall-off hunt (mirror of emergence): which senses *decline*, and is it
# obsolescence (absolute count falls) or just dilution (share falls, count holds)?
drift.decline_report()                                     # per sense: verdict + early/late share & count + terms
drift.sense_trajectories()                                 # per-period per-sense count + share (plot-ready)

# Three story-carrying charts (the [viz] extra):
drift.plot()                                               # margin density + calibrated threshold + p-value
drift.plot_composition()                                   # stacked-area sense share over time — the takeover, seen
drift.plot_decline()                                       # slopegraph early→late, coloured obsolescence/dilution/rising

# The sense-free, streaming sibling of sense_drift: flag *new* content with no
# sense model — one time-filtered nearest-neighbour query (scales to a vector
# store; mode="cumulative" makes it an online monitor).
nov = pcd.knn_density_drift(df, X, time_col="year", reference=range(2000, 2010), k=10)
nov.summary()                                              # which periods drifted + novelty density
nov.exemplars(period=2023, top=8)                          # the most-novel records driving a flag
nov.plot()                                                 # novelty density + reference-calibrated threshold
# sense_drift *explains*, knn_density_drift *catches*; agreement across the two
# mechanically different views is a robustness check, not a coincidence.
```

See [`examples/pycorpdiff_showcase.ipynb`](https://github.com/jturner-uofl/pycorpdiff/blob/main/examples/pycorpdiff_showcase.ipynb)
for a walkthrough on the synthetic Hansard-style corpus exercising
every analytical surface.

## Installation

```bash
pip install pycorpdiff                       # lexical-comparative core (MIT)
pip install "pycorpdiff[viz]"                # + altair / matplotlib / networkx
pip install "pycorpdiff[semantic]"           # + sentence-transformers
pip install "pycorpdiff[temporal]"           # + ruptures / statsmodels
pip install "pycorpdiff[notebooks]"          # + jupyter / vl-convert
pip install "pycorpdiff[all]"                # everything MIT-compatible
pip install "pycorpdiff[all,showcase]"       # + pysofra (GPL-3.0-or-later) for the JAMA-style showcase
```

The base install's direct runtime dependencies are `numpy`, `pandas`,
`scipy`, and `pyarrow`; optional extras land per analytical layer so
you only pay for what you use. `[showcase]` is broken out separately
because `pysofra` is GPL-3.0-or-later — pure `pycorpdiff` use without
that extra remains MIT-only.

To work from source:

```bash
git clone https://github.com/jturner-uofl/pycorpdiff
cd pycorpdiff
pip install -e ".[dev]"
pytest -q
```

## Cross-validation receipts

The math is checked against standard tools by automated test. The
fast tier runs on every push (matrix CI); the slow tier needs heavy
optional dependencies (NLTK, Scattertext, Stanford SNAP downloads)
and runs on main pushes only.

Fast tier:

- **Rayson's LL Wizard** — hand-derived contingency-table reference
  triples ([`tests/integration/test_crossval_rayson.py`](https://github.com/jturner-uofl/pycorpdiff/blob/main/tests/integration/test_crossval_rayson.py))

Slow tier:

- **NLTK** `BigramAssocMeasures` — PMI + t-score agreement to ≤ 1e-12
  on every adjacent bigram
- **Scattertext (Kessler 2017)** — behavioural agreement on the 2012
  US Conventions corpus
- **HistWords (Hamilton et al. 2016)** — known-shifter / stable-word
  sanity check on Stanford SNAP COHA decade embeddings (skips
  gracefully if the archive isn't reachable)

## Citation

If you use `pycorpdiff` in academic work, please cite the software via
the `CITATION.cff` file in this repository — GitHub renders a "Cite this
repository" widget directly from it.

## License

MIT — see [LICENSE](https://github.com/jturner-uofl/pycorpdiff/blob/main/LICENSE).

## Case studies and demos (rendered)

GitHub's in-browser notebook renderer is unreliable on larger notebooks
with embedded SVG outputs. The links below point to the **pre-rendered
HTML artefacts** (the canonical read versions) and to nbviewer fallbacks
for the `.ipynb` source. Notebook sources still live under `examples/`
for re-execution.

- **asylum case study — lexicalising asylum in UK Parliament, 2010-2023.**
  [📊 rendered HTML](https://raw.githack.com/jturner-uofl/pycorpdiff/main/docs/rendered/jss_case_study.html)
  · [nbviewer](https://nbviewer.org/github/jturner-uofl/pycorpdiff/blob/main/examples/jss_case_study.ipynb)
  · [.ipynb source](https://github.com/jturner-uofl/pycorpdiff/blob/main/examples/jss_case_study.ipynb)
- **Full feature tour (showcase).**
  [📊 rendered HTML](https://raw.githack.com/jturner-uofl/pycorpdiff/main/docs/rendered/pycorpdiff_showcase.html)
  · [nbviewer](https://nbviewer.org/github/jturner-uofl/pycorpdiff/blob/main/examples/pycorpdiff_showcase.ipynb)
  · [.ipynb source](https://github.com/jturner-uofl/pycorpdiff/blob/main/examples/pycorpdiff_showcase.ipynb)
- **Tutorial.**
  [📊 rendered HTML](https://raw.githack.com/jturner-uofl/pycorpdiff/main/docs/rendered/pycorpdiff_tutorial.html)
  · [.ipynb source](https://github.com/jturner-uofl/pycorpdiff/blob/main/examples/pycorpdiff_tutorial.ipynb)
- **Hansard demo.**
  [📊 rendered HTML](https://raw.githack.com/jturner-uofl/pycorpdiff/main/docs/rendered/hansard_demo.html)
  · [.ipynb source](https://github.com/jturner-uofl/pycorpdiff/blob/main/examples/hansard_demo.ipynb)

## Further reading

- [`docs/design.md`](https://github.com/jturner-uofl/pycorpdiff/blob/main/docs/design.md) — three-layer architecture
- [`docs/statistical-methods.md`](https://github.com/jturner-uofl/pycorpdiff/blob/main/docs/statistical-methods.md) — every metric's formula + citation
- [`docs/rendered/`](https://github.com/jturner-uofl/pycorpdiff/tree/main/docs/rendered) — catalogue of static HTML renders for offline viewing
