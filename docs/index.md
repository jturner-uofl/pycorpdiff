---
title: pycorpdiff — Comparative Corpus Analysis for Python
description: >-
  Comparative corpus analysis for Python: keyness, collocations, KWIC
  concordances, semantic change, and temporal trajectories with changepoints
  and causal impact. The quanteda / SketchEngine comparative layer for
  Python — notebook-native, every result carrying its own evidence.
  MIT-licensed, with a small numpy / pandas / scipy / pyarrow core.
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# pycorpdiff { .hero-title }

**Comparative corpus analysis for Python.** Keyness, collocations,
semantic change, and temporal trajectories — with changepoints and
causal inference — unified behind three notebook-native verbs, every
result carrying its own evidence.

[Get started](getting-started.md){ .md-button .md-button--primary }
[CBD highlight reel](rendered/cbd_highlight_reel.html){ .md-button }
[Walkthrough](rendered/methods_highlight_reel.html){ .md-button }
[CBD case study](rendered/cbd_case_study.html){ .md-button }
[GitHub](https://github.com/jturner-uofl/pycorpdiff){ .md-button }

</div>

---

## At a glance

<div class="grid cards" markdown>

-   :material-vector-difference: __Three verbs, one API__

    ---

    `compare(a, b)`, `track(c, "x")`, and `compare.before_after(c, event)`
    consolidate keyness, collocations, dispersion, trajectories,
    changepoints, ITS, causal impact, and semantic shift.

    [Getting started →](getting-started.md)

-   :material-text-search: __Evidence by default__

    ---

    Every keyness and collocation term carries its KWIC concordances:
    `.explain(term)` returns the source-text evidence behind any ranked
    result. No bare *p*-values.

    [Design →](design.md)

-   :material-chart-timeline-variant: __Temporal as first-class__

    ---

    Changepoints, interrupted time series, online changepoint detection
    (BOCPD), causal-impact analysis, and short-horizon forecasting — on
    any time-stamped corpus.

    [Statistical methods →](statistical-methods.md)

-   :material-brain: __Semantic change, pluggable__

    ---

    `semantic_trajectory`, `neighborhood_drift`, `induce_senses`, and
    `sense_drift` over any SBERT-compatible embedder via a one-line
    adapter — no plugin registry.

    [Statistical methods →](statistical-methods.md)

-   :material-translate: __Multilingual by adapter__

    ---

    Tokenizers (`spaCy`, `Stanza`, `jieba`, `fugashi`) plug in through a
    single `Protocol`. The lexical-comparative core is language-agnostic.

    [Multilingual →](multilingual.md)

-   :material-language-python: __Small MIT core__

    ---

    Base install depends only on `numpy`, `pandas`, `scipy`, `pyarrow`.
    Visualisation, embeddings, temporal models, and big-data backends are
    opt-in extras. 735 tests.

    [Install ↓](#installation)

</div>

---

## Installation

```bash
pip install pycorpdiff
```

Optional extras (compose them — e.g. `pycorpdiff[viz,temporal,semantic]`):

```bash
pip install "pycorpdiff[viz]"        # altair + matplotlib + networkx
pip install "pycorpdiff[semantic]"   # sentence-transformers + scikit-learn
pip install "pycorpdiff[temporal]"   # ruptures + statsmodels
pip install "pycorpdiff[duckdb]"     # out-of-core querying for large corpora
```

Python 3.11 or later required.

---

## Usage

=== "Keyness + evidence"

    ```python
    import pycorpdiff as pcd

    k = pcd.compare(
        corpus.slice(outlet=["Guardian", "Mirror"]),
        corpus.slice(outlet=["Mail", "Telegraph"]),
    ).keyness()
    k.plot()
    k.explain("migrant", n=5)        # the KWIC concordances behind the term
    ```

=== "Over time"

    ```python
    pcd.track(corpus, "sovereignty").over_time(freq="Q").plot()
    ```

=== "Before vs after an event"

    ```python
    pcd.compare.before_after(corpus, event_date="2016-06-23").keyness()
    ```

=== "Semantic shift"

    ```python
    from pycorpdiff.semantic import SBERTEmbedder

    pcd.compare(corpus_2005, corpus_2023).semantic_shift(
        "migrant", embedder=SBERTEmbedder()
    )
    ```

=== "Causal impact"

    ```python
    pcd.track(corpus, "lockdown").causal_impact(event_date="2020-03-23")
    ```

Every Result is a frozen dataclass implementing the relevant subset of
`.to_df() / .plot() / .explain() / .summary() / .to_html() / .to_json()`.

---

## What pycorpdiff unifies

It is the **missing comparative layer** between R's `quanteda`, the
closed-source SketchEngine platform, and the fragmented Python NLP stack
(`nltk` / `spaCy` / `gensim` / `sentence-transformers`) — orchestration,
not reinvention.

| Question | Typical tooling | pycorpdiff |
|---|---|---|
| How does corpus A differ from B?        | `quanteda` (R), hand-rolled scipy | `compare(a, b).keyness()` |
| Which collocations gained/lost ground?  | AntConc, custom scripts           | `compare(a, b).collocations()` |
| How did *X*'s meaning shift over time?  | `gensim` + manual alignment       | `.semantic_shift(...)`, `sense_drift` |
| Did this event move the discourse?      | `CausalImpact` (R)                | `track(...).causal_impact(...)` |
| Where is the discourse heading?         | `statsmodels` + glue              | `track(...).forecast(horizon=4)` |
| Show me the evidence behind a term      | SketchEngine KWIC                 | `.explain(term)` |

---

## Design principles

- **Interoperability over reinvention** — thin adapters around existing
  libraries; tokenizers and embedders plug in via two `typing.Protocol`
  extension points.
- **Comparative abstractions** — `compare(a, b)` and `track(c, "x")` as
  first-class verbs; temporal comparison (`before_after`, `over_time`) is
  built in, not bolted on.
- **Explainability by default** — every result carries its evidence.
- **Statistically grounded defaults** — log-likelihood with effect sizes,
  Wilson confidence intervals, dispersion sanity checks.
- **Dataframe-first, notebook-native** — `pandas` by default, `polars`
  and `duckdb` opt-in; `altair` plots.

It is *not* a general NLP framework, a SketchEngine clone, or a
deep-learning substrate — embeddings are a pluggable interface, not a
training target.

---

## Cite this work

```bibtex
@software{pycorpdiff,
  author  = {Turner, Jason S.},
  title   = {pycorpdiff: Comparative Corpus Analysis for Python},
  version = {0.1.0a33},
  year    = {2026},
  url     = {https://github.com/jturner-uofl/pycorpdiff},
}
```

---

<small>
`pycorpdiff` is released under the MIT licence and is on PyPI as
`pip install pycorpdiff` (alpha). See the
[GitHub repository](https://github.com/jturner-uofl/pycorpdiff) for
source, releases, the issue tracker, and contribution guidelines.
</small>
