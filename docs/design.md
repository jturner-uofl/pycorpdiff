# Design

`pycorpdiff` is structured as three concentric layers, each with a
single responsibility. Understanding the layering makes the code easier
to read, easier to extend, and (importantly for a JSS-track package)
easier to audit.

## The three layers

```
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

- **Layer 2 never imports Layer 3.** Methods take dataframes/arrays,
  return dataframes/arrays. This is what makes the math teachable,
  testable, and citable in a methods paper.
- **Layer 1 is backend-pluggable.** A `Corpus` wraps either a pandas or
  polars frame behind a thin `_backends` shim. pandas is the default;
  polars is opt-in.
- **Tokenizer and Embedder are Protocols, not classes.** Multilingual
  support = bring your own spaCy/Stanza/jieba pipeline. The defaults are
  a regex tokenizer plus a `sentence-transformers` embedder loaded
  lazily on first use.

## Result objects are data, not god-objects

Every analytical Result is a `frozen=True` dataclass with the same
informal contract:

| Method            | What it does                                       |
|-------------------|----------------------------------------------------|
| `.to_df()`        | Returns the underlying tidy DataFrame              |
| `.plot(**kw)`     | Returns an `altair.Chart`                          |
| `.explain(term)`  | Returns a `ConcordanceResult` with KWIC evidence   |
| `.summary()`      | Returns a short human-readable string              |

This is duck-typing rather than inheritance — it keeps Results
lightweight, lets them be built from a bare DataFrame, and avoids the
"god-object" trap where one class accretes everything.

## Two plug points, not a plugin system

The package exposes exactly two extension points, both as `typing.Protocol`:

```python
class Tokenizer(Protocol):
    def __call__(self, text: str) -> list[str]: ...

class Embedder(Protocol):
    def encode(self, terms: Sequence[str]) -> np.ndarray: ...   # (n, d)
```

That's it. spaCy / Stanza / jieba / fugashi all satisfy `Tokenizer` with
a one-line adapter. SBERT / gensim / HuggingFace pipelines all satisfy
`Embedder`. There's no plugin registry, entry-points system, or DI
container — Python protocols **are** the plugin system.

## Optional extras

The base install does the lexical-comparative core with zero heavy
dependencies (numpy, pandas, scipy, pyarrow). Everything else is
opt-in:

| Extra        | Brings in                                 | Used for                       |
|--------------|-------------------------------------------|--------------------------------|
| `viz`        | altair, matplotlib                        | `.plot()` on every Result      |
| `semantic`   | sentence-transformers, scikit-learn       | `compare(a,b).semantic_shift`  |
| `temporal`   | ruptures, statsmodels                     | changepoints + ITS             |
| `polars`     | polars, pyarrow                           | columnar backend (later phase) |
| `duckdb`     | duckdb                                    | out-of-core querying           |
| `nlp`        | spacy                                     | multilingual tokenisation      |
| `all`        | the union of the above                    | everything                     |
| `dev`        | pytest, hypothesis, ruff, mypy, ...       | for contributors               |

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
- **No forecasting.** The temporal layer is for *explaining* change,
  not predicting it. Forecasters (sktime, prophet, etc.) are a
  different problem space.
- **No distributed-systems plumbing.** The target workloads are
  medium-to-large corpora on a single machine. DuckDB and polars are
  opt-in for out-of-core querying when needed.

This is in service of one core idea: **the package is the comparative
layer**. Everything else is interoperability.
