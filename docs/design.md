# Design

`pycorpdiff` is structured as three concentric layers, each with a
single responsibility. Understanding the layering makes the code easier
to read, easier to extend, and easier to test.

## The three layers

```text
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 3 — Comparison & Result objects (the public verbs)        │
│  compare(a, b) · track(c, "x") · compare.before_after(...)      │
│  → KeynessResult, CollocationShiftResult,                       │
│    SemanticShiftResult, TemporalTrajectory                      │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 2 — Methods (the math; pure, dataframe-in / dataframe-out)│
│  keyness · collocation · semantic · temporal · explain          │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 1 — Corpus & I/O (data plumbing; backend-agnostic)        │
│  Corpus, CorpusSlice, Tokenizer (Protocol), Embedder (Protocol) │
│  readers: txt / csv / parquet / pandas / polars / DuckDB        │
└─────────────────────────────────────────────────────────────────┘
```

Three invariants make the layering pay off:

- **Layer 2 carries the math, not the Result.** Methods take
  dataframes/arrays, return dataframes/arrays — the comparison verbs
  in Layer 3 wrap those outputs into Result dataclasses. A small set
  of serialisation helpers (`_table_to_html`, `_table_to_json`) is
  shared with Layer 3 via the `results` module today; if the boundary
  becomes load-bearing those helpers will move to a neutral module.
- **Layer 1 is pandas-internal with polars interop at the boundary.**
  `Corpus` operates on a `pandas.DataFrame`; `pcd.from_dataframe` accepts
  a `polars.DataFrame` (and converts at construction); `Corpus.to_polars()`
  round-trips back to polars. The `_backends` package is a placeholder
  for a deeper pandas/polars split that the package may or may not
  need — pandas is sufficient for the workloads we target today.
- **Tokenizer and Embedder are Protocols, not classes.** Multilingual
  support = bring your own spaCy/Stanza/jieba pipeline. The defaults are
  a regex tokenizer plus a `sentence-transformers` embedder loaded
  lazily on first use.

## Result objects are data, not god-objects

Every analytical Result is a `frozen=True` dataclass implementing
the relevant subset of an informal six-method contract:

| Method                | What it does                                       |
|-----------------------|----------------------------------------------------|
| `.to_df()`            | Returns the underlying tidy DataFrame              |
| `.plot(**kw)`         | Returns an `altair.Chart`                          |
| `.to_html(path=None)` | Renders the table as HTML (returns + optionally writes) |
| `.to_json(path=None)` | Renders the table as JSON (returns + optionally writes) |
| `.summary()`          | Returns a short human-readable string              |
| `.explain(term)`      | Returns a `ConcordanceResult` with KWIC evidence   |

Which methods apply varies by Result (✓ = implemented, — = not applicable
to this Result's shape):

| Result                    | `to_df` | `plot` | `to_html` | `to_json` | `summary` | `explain` |
|---------------------------|---------|--------|-----------|-----------|-----------|-----------|
| `KeynessResult`           | ✓       | ✓      | ✓         | ✓         | ✓         | ✓         |
| `CollocationShiftResult`  | ✓       | ✓      | ✓         | ✓         | ✓         | ✓         |
| `SemanticShiftResult`     | ✓       | ✓      | ✓         | ✓         | ✓         | —         |
| `TemporalTrajectory`      | ✓       | ✓      | ✓         | ✓         | ✓         | —         |
| `ForecastResult`          | ✓       | ✓      | ✓         | ✓         | ✓         | —         |
| `CausalImpactResult`      | ✓       | ✓      | ✓         | ✓         | ✓         | —         |
| `BocpdResult`             | ✓       | ✓      | ✓         | ✓         | ✓         | —         |
| `NetworkResult`           | ✓       | ✓      | ✓         | ✓         | ✓         | —         |
| `SenseInductionResult`    | ✓       | ✓      | ✓         | ✓         | ✓         | —         |
| `SenseDriftResult`        | ✓       | ✓      | ✓         | ✓         | ✓         | —         |
| `SenseNamingResult`       | ✓       | —      | ✓         | ✓         | ✓         | —         |
| `ConcordanceResult`       | ✓       | —      | ✓         | ✓         | ✓         | —         |

`.explain()` is meaningful only for term-ranked Results (keyness +
collocation shift); `ConcordanceResult` is *itself* the explained
output, so `.plot()` on it doesn't apply.

This is duck-typing rather than inheritance — it keeps Results
lightweight, lets them be built from a bare DataFrame, and avoids the
"god-object" trap where one class accretes everything.

## Three plug points, not a plugin system

The package exposes exactly three extension points, all as `typing.Protocol`:

```python
class Tokenizer(Protocol):
    def __call__(self, text: str) -> list[str]: ...

class Embedder(Protocol):
    def encode(self, terms: Sequence[str]) -> np.ndarray: ...   # (n, d)

class Annotator(Protocol):
    def __call__(self, prompt: str) -> str: ...                 # the interpretive layer
```

That's it. spaCy / Stanza / jieba / fugashi all satisfy `Tokenizer` with
a one-line adapter. SBERT / gensim / HuggingFace pipelines all satisfy
`Embedder`. A local Ollama model, a hosted API, or your own function satisfies
`Annotator`. There's no plugin registry, entry-points system, or DI
container — Python protocols **are** the plugin system.

`Annotator` was a **deliberate widening from the original two**, not a default
slide: an LLM may *name and gloss* a fitted sense, but the protocol is fenced by
an invariant. It consumes only the package's own *cited, measured* exemplars and
returns text, which lands in a **separate** `SenseNamingResult` and never in a
number, a flag, or a veracity verdict. Vectors and counts quantify; the LLM
interprets; never the reverse. The boundary is enforced by a unit test
(`test_annotator_output_never_enters_numeric_fields`). See
`SenseDriftResult.name_senses`.

## Optional extras

The base install does the lexical-comparative core with zero heavy
dependencies (numpy, pandas, scipy, pyarrow). Everything else is
opt-in:

| Extra         | Brings in                                  | Used for                              |
|---------------|--------------------------------------------|---------------------------------------|
| `viz`         | altair, matplotlib, networkx               | `.plot()` on every Result             |
| `semantic`    | sentence-transformers, scikit-learn        | `compare(a,b).semantic_shift`         |
| `temporal`    | ruptures, statsmodels                      | changepoints + ITS + forecast         |
| `polars`      | polars, pyarrow                            | polars DataFrame interop              |
| `duckdb`      | duckdb                                     | out-of-core querying                  |
| `nlp`         | spacy                                      | multilingual tokenisation             |
| `huggingface` | datasets                                   | `from_huggingface` corpus ingestion   |
| `notebooks`   | jupyter, vl-convert-python                 | running + rendering the example notebooks |
| `showcase`    | pysofra (**GPL-3.0-or-later**)             | JAMA-style table polish for the showcase notebook |
| `all`         | union of the MIT-compatible extras above   | everything except `showcase` (GPL)    |
| `dev`         | pytest, hypothesis, ruff, mypy, ...        | for contributors                      |

Each `.plot()` / `.semantic_shift()` / `.changepoints()` call lazy-imports
its dependency and raises a friendly `ImportError` pointing at the
right extras command if it's missing.

## Why two embedders by default?

`pycorpdiff` ships `SBERTEmbedder` for production use and `HashEmbedder`
for reproducibility:

- `SBERTEmbedder` wraps sentence-transformers (lazy-loaded). The real
  semantic content lives here.
- `HashEmbedder` maps strings to vectors via a SHA-256-seeded RNG. It
  has zero semantic signal but is fully deterministic and free of
  network dependencies. The package's own semantic tests use it; the
  tutorial uses it so its outputs reproduce byte-for-byte across CI
  runs and across machines.

Users can plug in their own embedder by implementing `encode()`.

## Scope discipline

The package deliberately *doesn't* do certain things:

- **No tokenisation reinvention.** spaCy, NLTK, and tokenizers from the
  HuggingFace ecosystem already do this well. We provide a default
  regex tokenizer for getting started and a Protocol for plugging in
  serious pipelines.
- **No embedding training.** sentence-transformers, gensim, and PyTorch
  all do this. We consume embeddings via the `Embedder` Protocol.
- **No standalone time-series forecasting library.** `tr.forecast()`
  wraps a state-space ETS for short-horizon trajectory continuation
  alongside changepoints and ITS — it's context for *explaining* a
  trajectory, not a Prophet / sktime / Darts replacement.
- **No distributed-systems plumbing.** The target workloads are
  medium-to-large corpora on a single machine. DuckDB and polars are
  opt-in for out-of-core querying when needed.

This is in service of one core idea: **the package is the comparative
layer**. Everything else is interoperability.
