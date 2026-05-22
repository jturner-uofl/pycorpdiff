# pycorpdiff

**Comparative corpus analysis for modern Python workflows.**

`pycorpdiff` is a Python package for *comparing* text corpora — across groups,
across time, and across discourse contexts. It unifies classical corpus
linguistics methods (keyness, collocations, dispersion) with modern
embedding-based semantic-shift analysis under a single, composable API.

> **Status: pre-alpha (0.1.0a0)** — scaffolding is in place; the lexical
> comparative core is the next milestone. Not yet on PyPI.

## What it's for

Researchers in corpus linguistics, digital humanities, and computational
social science routinely need to answer questions like:

- How does corpus A differ from corpus B?
- How has discourse around *X* changed between 1990 and 2020?
- What semantic shifts occurred around a given event?
- Which collocations gained or lost ground?

`pycorpdiff` provides a coherent comparative layer over the existing PyData
and NLP stacks (`pandas`, `polars`, `pyarrow`, `scipy`, `statsmodels`,
`sentence-transformers`, `spacy`) without reinventing tokenisation, embeddings,
or topic modelling.

## What it is not

- Not a general NLP framework. It does not replace `nltk`, `spaCy`, or `gensim`.
- Not a SketchEngine clone. It is comparative-first and notebook-native.
- Not a deep learning framework. Embeddings are treated as a pluggable
  interface, not a training substrate.
- Not a forecasting tool. Temporal analysis here is for *explanation*, not
  prediction.

## Design principles

- **Interoperability over reinvention** — adapters around existing libraries.
- **Comparative abstractions** — `compare(a, b)`, `track(c, "x")` as first-class verbs.
- **Temporal as first-class** — `before_after`, `over_time`, trajectories.
- **Explainability by default** — every result carries its evidence.
- **Statistically grounded defaults** — log-likelihood + effect sizes, Wilson
  CIs, dispersion sanity checks. No bare *p*-values.
- **Dataframe-first I/O** — `pandas` by default, `polars` opt-in.
- **Notebook-native** — `altair` plots, idiomatic Jupyter ergonomics.

## Quick start *(target API — Phase 1 in progress)*

```python
import pycorpdiff as pcd

news = pcd.read_parquet("uk_news.parquet", text_col="body", time_col="date")

# Lexical comparison
k = pcd.compare(
    news.slice(outlet=["Guardian", "Mirror"]),
    news.slice(outlet=["Mail", "Telegraph"]),
).keyness(method="log_likelihood", effect_size=True)
k.plot()
k.explain("migrant", n=5)

# Before / after an event
pcd.compare.before_after(news, event_date="2016-06-23").keyness()

# Diachronic trajectory of a single term
pcd.track(news, "sovereignty").over_time(freq="Q").plot()
```

## Installation *(once on PyPI)*

```bash
pip install pycorpdiff               # lexical-comparative core
pip install "pycorpdiff[viz]"        # + altair / matplotlib
pip install "pycorpdiff[semantic]"   # + sentence-transformers
pip install "pycorpdiff[temporal]"   # + ruptures / statsmodels
pip install "pycorpdiff[all]"        # everything
```

Until then, from a local clone:

```bash
git clone https://github.com/jasonsturner/pycorpdiff
cd pycorpdiff
pip install -e ".[dev]"
pytest
```

## Roadmap

| Phase | Milestone |
|-------|-----------|
| 0     | Scaffolding (this commit) |
| 1     | Corpus ingestion + frequency-based keyness (LL, LogRatio, BF, effect sizes) |
| 2     | Collocation measures + collocation shift |
| 3     | KWIC concordances + `explain()` plumbing |
| 4     | Temporal slicing, rolling frequencies, basic viz |
| 5     | **v0.1.0a1** — PyPI alpha release |
| 6     | Semantic shift via Procrustes-aligned embeddings |
| 7     | Changepoint detection + interrupted time series |
| 8     | Documentation site + JSS paper draft |
| 9     | **v0.2.0** — JSS submission |

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use `pycorpdiff` in academic work, see [CITATION.cff](CITATION.cff).
