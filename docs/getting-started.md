# Getting started

## Install

`pycorpdiff` is in pre-alpha and not yet on PyPI. Install from a local clone:

```bash
git clone https://github.com/jasonsturner/pycorpdiff
cd pycorpdiff
pip install -e ".[dev]"
pytest
```

Once on PyPI:

```bash
pip install pycorpdiff                # lexical-comparative core
pip install "pycorpdiff[viz]"         # + altair / matplotlib
pip install "pycorpdiff[semantic]"    # + sentence-transformers
pip install "pycorpdiff[temporal]"    # + ruptures / statsmodels
pip install "pycorpdiff[all]"         # everything
```

The base install keeps a small dependency footprint (numpy, pandas,
scipy, pyarrow). Optional extras land per analytical layer so you only
pay for what you use.

## Construct a Corpus

A `Corpus` wraps a `pandas.DataFrame` of documents plus metadata.

```python
import pandas as pd
import pycorpdiff as pcd

df = pd.DataFrame({
    "text": [
        "the migrant worker arrived and the family settled",
        "the migrant criminal threat increased",
        "the migrant community grew here",
    ],
    "outlet": ["Guardian", "Mail", "Guardian"],
    "date": ["2020-01-15", "2020-02-15", "2020-03-15"],
})
corpus = pcd.from_dataframe(df, text_col="text", meta_cols=("outlet", "date"))
print(f"{len(corpus)} docs · {corpus.total_tokens()} tokens")
```

For files on disk:

```python
corpus = pcd.read_csv("path/to/news.csv", text_col="body")
corpus = pcd.read_parquet("path/to/news.parquet", text_col="body")
corpus = pcd.read_txt("path/to/one_file.txt")
```

## Slice it

Slicing on metadata returns a `CorpusSlice` that propagates its filter
into result labels:

```python
a = corpus.slice(outlet="Guardian")
b = corpus.slice(outlet="Mail")
a.label   # "outlet='Guardian'"
```

## Compare

The package's headline verb is `compare(a, b)`:

```python
# Lexical: which words separate the two slices?
k = pcd.compare(a, b).keyness(min_count=3, dispersion=True)
k.table.head()              # tidy DataFrame
k.plot()                    # altair volcano
k.explain("migrant", n=5)   # KWIC contexts from both sides

# Collocational: what does each slice put next to "migrant"?
s = pcd.compare(a, b).collocation_shift("migrant", measure="logDice")
s.plot()                    # altair diverging bar
s.explain("criminal", n=3)  # contexts where "migrant criminal" co-occur

# Semantic: how is "migrant" used differently across slices?
m = pcd.compare(a, b).semantic_shift(
    "migrant",
    embedder=pcd.SBERTEmbedder(),  # or HashEmbedder() for offline / tests
)
m.table                     # cosine distance + context counts
```

## Track over time

`track(corpus, term).over_time()` returns a tidy diachronic trajectory
with Wilson confidence intervals:

```python
tr = pcd.track(corpus, ["worker", "criminal"]).over_time(
    freq="Y", time_col="date",
)
tr.plot()                                                   # CI band + line
tr.changepoints(target="criminal")                          # ruptures PELT
tr.interrupted_time_series(event_date="2016", target="x")   # statsmodels OLS
```

## Before / after an event

A specialised constructor for chronological comparison:

```python
ba = pcd.compare.before_after(
    corpus, event_date="2016-06-23", time_col="date",
).keyness()
ba.plot()
```

## Where next?

- **[Tutorial notebook](https://github.com/jasonsturner/pycorpdiff/blob/main/examples/pycorpdiff_tutorial.ipynb)** —
  end-to-end walkthrough on a synthetic two-frame corpus.
- **[Design](design.md)** — the architectural ideas behind the verbs.
- **[Statistical methods](statistical-methods.md)** — what each metric
  does and why we chose these defaults.
- **[Multilingual](multilingual.md)** — wire up spaCy / Stanza / jieba.
