# Benchmarks

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
