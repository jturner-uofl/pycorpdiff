# pycorpdiff

**Comparative corpus analysis for modern Python workflows.**

`pycorpdiff` is a Python package for *comparing* text corpora — across groups,
across time, and across discourse contexts. It unifies classical corpus
linguistics methods (keyness, collocations, dispersion) with modern
embedding-based semantic-shift analysis under a single composable API.

## What it's for

Researchers in corpus linguistics, digital humanities, and computational
social science routinely need to answer questions like:

- How does corpus A differ from corpus B?
- How has discourse around *X* changed between 1990 and 2020?
- What semantic shifts occurred around a specific event?
- Which collocations gained or lost ground?

`pycorpdiff` provides a coherent comparative layer over the existing
PyData and NLP stacks (`pandas`, `polars`, `pyarrow`, `scipy`,
`statsmodels`, `sentence-transformers`, `spacy`) without reinventing
tokenisation, embeddings, or topic modelling.

## What it is not

- Not a general NLP framework. It does not replace `nltk`, `spaCy`, or `gensim`.
- Not a SketchEngine clone. It is comparative-first and notebook-native.
- Not a deep learning framework. Embeddings are treated as a pluggable
  interface, not a training substrate.
- Not a standalone time-series forecasting library. `tr.forecast()` exists
  for short-horizon trajectory continuation alongside changepoints and
  ITS — for serious forecasting reach for `sktime` / `prophet` / `Darts`.

## Design principles

- **Interoperability over reinvention** — adapters around existing libraries.
- **Comparative abstractions** — `compare(a, b)`, `track(c, "x")` as first-class verbs.
- **Temporal as first-class** — `before_after`, `over_time`, trajectories.
- **Explainability by default** — every result carries its evidence.
- **Statistically grounded defaults** — log-likelihood + effect sizes,
  Wilson CIs, dispersion sanity checks. No bare *p*-values.
- **Dataframe-first I/O** — `pandas` by default, `polars` opt-in.
- **Notebook-native** — `altair` plots, idiomatic Jupyter ergonomics.

## A quick taste

```python
import pycorpdiff as pcd
import pandas as pd

corpus = pcd.from_dataframe(
    pd.read_parquet("uk_news.parquet"),
    text_col="body",
    meta_cols=("outlet", "date"),
)

# Lexical contrast
k = pcd.compare(
    corpus.slice(outlet=["Guardian", "Mirror"]),
    corpus.slice(outlet=["Mail", "Telegraph"]),
).keyness()
k.plot()
k.explain("migrant", n=5)

# Diachronic trajectory
pcd.track(corpus, "sovereignty").over_time(freq="Q").plot()

# Around an event
pcd.compare.before_after(corpus, event_date="2016-06-23").keyness()
```

## Where next?

- **[Getting started](getting-started.md)** — install and first analysis.
- **[Design](design.md)** — the three-layer architecture and why.
- **[Statistical methods](statistical-methods.md)** — what each metric is
  computing and why we chose these defaults.
- **[Multilingual support](multilingual.md)** — plug in spaCy, Stanza,
  jieba, fugashi, or your own.
