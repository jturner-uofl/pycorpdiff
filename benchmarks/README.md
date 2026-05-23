# Benchmarks

Two kinds of benchmarks live here:

1. **Regression benchmarks** (asv): track pycorpdiff's own
   performance over time as the package evolves.
2. **Comparative benchmarks** (`external_comparison.py`): head-to-head
   timings against NLTK on a shared real-data fixture. The numbers
   feed §6 of the JSS paper.

## Regression suite

[airspeed-velocity (asv)](https://asv.readthedocs.io/) regression
benchmarks for pycorpdiff.

## What's measured

| Suite | What it covers | Sizes |
|---|---|---|
| `CorpusConstruction` | `doc_term_counts`, `vocab`, `total_tokens` | 100 / 1k / 10k docs |
| `Keyness` | `compare(a, b).keyness()` (with and without dispersion) | 100 / 1k / 10k docs |
| `CollocationShift` | `compare(a, b).collocation_shift(target)` | 500 / 5k docs |
| `TemporalTrack` | `track(c, term).over_time(freq="Y")` | 1k / 10k docs |
| `Tokenization` | The default `RegexTokenizer` at scale | 1k / 10k docs |

All benchmarks run against a deterministic synthetic corpus with a
500-word vocabulary; results should reproduce across machines modulo
CPU speed.

## Run locally

```bash
pip install asv

# Quick check against the current source (no git checkouts, no results saved)
asv dev

# Proper benchmark run against HEAD (single iteration per benchmark)
asv run --quick HEAD

# Render the HTML report
asv publish && asv preview
```

## In CI

The `benchmarks` job runs `asv dev` on every push to main as a smoke
check — it verifies the benchmark suite still imports and executes,
but does not save results or check for regressions. Persistent
regression tracking against a baseline is a future iteration.

## Comparative suite (`external_comparison.py`)

Head-to-head timings against external packages on the same workload —
specifically NLTK's `BigramAssocMeasures` for collocation measures.
Runs on the bundled 2012 US Conventions fixture (189 documents,
~138K tokens) from `paper/replication/data/conventions_2012.parquet`,
so the result is reproducible across machines modulo CPU speed.

```bash
pip install -e ".[viz,temporal]" nltk
python benchmarks/external_comparison.py
```

The script writes a markdown table to stdout (used in the JSS paper
§6) and the raw numbers to `external_comparison_results.json` for
LaTeX consumption.

Quanteda comparisons require R + the `quanteda` R package and live
separately under the `slow` test tier
(`tests/integration/test_crossval_quanteda.py`); they verify
numerical agreement, not performance.
