# pycorpdiff

<!--
TODO post-publish (Phase 5 — once GitHub repo public + PyPI published + Zenodo DOI minted):

[![PyPI](https://img.shields.io/pypi/v/pycorpdiff.svg)](https://pypi.org/project/pycorpdiff/)
[![Python versions](https://img.shields.io/pypi/pyversions/pycorpdiff.svg)](https://pypi.org/project/pycorpdiff/)
[![CI](https://github.com/jturner-uofl/pycorpdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/jturner-uofl/pycorpdiff/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.<RECORD>.svg)](https://doi.org/10.5281/zenodo.<RECORD>)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
-->

**Comparative corpus analysis for modern Python workflows.**

`pycorpdiff` is the **missing comparative layer** between R's
[`quanteda`](https://quanteda.io/), the closed-source SketchEngine
platform, and the fragmented Python NLP stack
(`nltk`/`spaCy`/`gensim`/`sentence-transformers`). Three public verbs
— `compare(a, b)`, `track(c, term)`, `compare.before_after(c, event)` —
consolidate keyness, collocations, dispersion, temporal trajectories,
changepoint detection, interrupted time series, causal-impact analysis,
forecasting, online changepoint detection, and embedding-based semantic
shift under a single notebook-native API. Every result carries its own
KWIC evidence: `.explain(term)` returns the source-text concordances
behind any ranked term.

The package answers the questions corpus linguistics, digital humanities,
and computational social science routinely have:

- *How does corpus A differ from corpus B?* — `compare(a, b).keyness()`
- *How has discourse around X evolved over time?* — `track(c, "x").over_time()`
- *What did "migrant" mean in 2005 vs 2023?* — `compare(...).semantic_shift("migrant", embedder=...)`
- *Did this event actually shift the conversation?* — `track(...).causal_impact(event_date=...)`
- *Where is the discourse heading?* — `track(...).forecast(horizon=4)`

`pycorpdiff` is positioned as **orchestration**, not reinvention.
Tokenizers (`spaCy`, `Stanza`, `jieba`, `fugashi`) and embedders (any
`SBERT`-compatible model) plug in via two `typing.Protocol` extension
points — one-line adapters, no plugin registry. The base install pulls
only `numpy`, `pandas`, `scipy`, and `pyarrow`; everything else is opt-in
via extras.

> **Status: pre-release alpha (0.1.0a0).** Public API is stable for the
> features described below; PyPI publication is the next milestone.

## The three-layer architecture

| Layer | Purpose | Key surface |
|---|---|---|
| **1 — Ingestion + `Corpus`** | get text in, slice it, hash it | `from_dataframe`, `read_csv`, `read_parquet`, `read_txt`, `read_duckdb`, `from_huggingface`, `fetch_hansard`, `Corpus.slice/by_time/__hash__/doc_term_counts(_sparse)/to_polars` |
| **2 — Pure math** | statistics with no I/O | `keyness.{log_likelihood,chi_squared,log_ratio,percent_diff,bayes_factor,permutation_pvalues,keyness_multi,juilland_d,benjamini_hochberg}`; `collocation.{logdice,pmi,t_score,mi_three,collocation_shift,cooccurrence_network}`; `semantic.{HashEmbedder,SBERTEmbedder,semantic_trajectory,neighborhood_drift}`; `temporal.{changepoints,interrupted_time_series,forecast,causal_impact,bocpd}` |
| **3 — Verbs + Results** | public API | `compare`, `track`, `compare.before_after`, `keyness_multi`, plus 9 frozen-dataclass Result types each with `.to_df() / .plot() / .explain() / .summary() / .to_html() / .to_json()` |

## Quick start

```python
import pycorpdiff as pcd

news = pcd.from_dataframe(df, text_col="body", meta_cols=("outlet", "date"))

# Compare — three verbs
k = pcd.compare(news.slice(outlet="Guardian"), news.slice(outlet="Mail")).keyness()
c = pcd.compare(a, b).collocation_shift("migrant")
s = pcd.compare(a, b).semantic_shift("migrant", embedder=pcd.SBERTEmbedder())

# Track over time
tr = pcd.track(news, "migrant").over_time(freq="Y")
tr.changepoints()                                     # offline PELT
tr.changepoints_online(hazard=1/24)                   # Bayesian online (Adams & MacKay 2007)
tr.interrupted_time_series(event_date="2016-06-23")   # segmented OLS
tr.causal_impact(event_date="2016-06-23")             # Bayesian counterfactual (Brodersen 2015)
tr.forecast(horizon=4)                                # state-space ETS

# Before / after a known event
pcd.compare.before_after(news, event_date="2016-06-23").keyness()

# N-way (≥ 2 corpora)
pcd.keyness_multi([gu, ma, te, mi], labels=["Guardian", "Mail", "Telegraph", "Mirror"])

# The discourse as a graph
pcd.cooccurrence_network(news, top_n=50).plot()

# Every Result: .to_df() · .plot() · .explain() · .summary() · .to_html() · .to_json()
```

See [`examples/pycorpdiff_showcase.ipynb`](examples/pycorpdiff_showcase.ipynb)
([rendered HTML](docs/rendered/pycorpdiff_showcase.html)) for a
walkthrough on a synthetic UK Hansard corpus exercising every analytical
surface.

## Installation

<!-- TODO post-publish: replace this block with the PyPI install commands once published. -->

Currently a pre-release alpha. From a local clone:

```bash
git clone https://github.com/jturner-uofl/pycorpdiff
cd pycorpdiff
pip install -e ".[dev]"
pytest -q                          # 519 default tests, ~7s
```

Optional extras: `[viz]` (altair + matplotlib + networkx), `[semantic]`
(sentence-transformers + scikit-learn), `[temporal]` (ruptures +
statsmodels), `[polars]`, `[duckdb]`, `[huggingface]`, `[nlp]` (spaCy),
`[notebooks]` (jupyter + vl-convert + pysofra, for the showcase),
or `[all]`.

## Cross-validation receipts

The math agrees with the standard tools — by automated test:

- **Rayson's LL Wizard** — 15 hand-derived contingency-table reference triples
- **NLTK** `BigramAssocMeasures` — PMI + t-score to ≤ 1e-12 on every adjacent bigram
- **Scattertext (Kessler 2017)** — behavioural agreement on the 2012 US Conventions corpus
- **quanteda (R)** via `rpy2` — byte-for-byte G² agreement (slow tier)
- **HistWords (Hamilton et al. 2016)** — diachronic cosine displacements on COHA (slow tier)

## Citation

If you use `pycorpdiff` in academic work, please cite the software via
the `CITATION.cff` file in this repository — GitHub renders a "Cite this
repository" widget directly from it.

## License

MIT — see [LICENSE](LICENSE).

## Further reading

- [`docs/design.md`](docs/design.md) — three-layer architecture
- [`docs/statistical-methods.md`](docs/statistical-methods.md) — every metric's formula + citation
- [`examples/pycorpdiff_showcase.ipynb`](examples/pycorpdiff_showcase.ipynb) — full feature tour as a notebook
- [`docs/rendered/`](docs/rendered/) — self-contained HTML renders of the example notebooks
