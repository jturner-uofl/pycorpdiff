# Audit follow-ups

This page tracks features that were promised in the design doc or the
README but not delivered in Phases 0–8. It's the working list driving
the post-Phase-8 cleanup commits.

The numbering matches the audit table from the original session; items
are crossed off as commits close them.

## Status snapshot

| #  | Item                                                | Status        |
|----|-----------------------------------------------------|---------------|
| 6  | `Comparison.concordance(target)`                    | ✅ done (`c0cb437`) |
| 7  | `SemanticShiftResult.neighbors_before/after`        | ✅ done (`c0cb437`) |
| 8  | `read_txt(one_doc_per="line")`                      | ✅ done (`c0cb437`) |
| 14 | Property tests for collocations + temporal          | ✅ done (`c0cb437`) |
| 12 | Real SBERT integration test on a slow CI tier       | ✅ done       |
| 10 | DuckDB out-of-core reader                           | ✅ done       |
| 13 | LaTeX-compile CI job                                | ✅ done       |
| 11 | `benchmarks/` with `asv` regression suite           | ⏳ next       |
| 9  | polars backend                                      | 🔲 future    |
| 4  | Real benchmark corpora                              | 🔲 future    |
| 5  | The five demo analyses (depend on #4)               | 🔲 future    |
| 15 | Cross-validation against quanteda / Scattertext / histwords / Rayson | 🔲 deferred until 4–13 are done |

## Item #15 — cross-validation against open-source equivalents

Captured here so it doesn't get lost. **Not stupid** — this is the
"receipt" layer that turns a methods claim into a citeable one, and
it's exactly the pattern pysofra uses for its R cross-checks.

### Tier 1 — open source, exact number match possible

| Target | What to validate | How |
|---|---|---|
| **`quanteda` (R)** | keyness G², LogRatio, dispersion, collocations (`textstat_collocations`) | `rpy2` in CI, like pysofra. Same fixture corpus on both sides; assert numbers agree to 6+ decimals. *Highest-value validation we could add.* |
| **`NLTK`** | PMI, t-score, χ² for collocations | Pure Python; just `import nltk` in a test. Parameterise window size to match NLTK defaults. |
| **`Scattertext` (Kessler 2017)** | Scaled F-score on the 2012 US Conventions corpus — open data + published numbers | Run on the same corpus, check ranking ordinally matches their top-N. |
| **Rayson's LL Wizard** | Single-cell log-likelihood values | Already partially in `test_loglikelihood.py`. Add 5–10 more (counts, totals) → (LL, LogRatio) reference triples from his online calculator. |
| **`histwords` (Hamilton et al. 2016)** | Diachronic semantic-shift values for COHA on ~50 target words | Their code at github.com/williamleif/histwords; data published. Replicate a subset, compare cosine displacements. |

### Tier 2 — closed-source (SketchEngine)

SketchEngine's algorithms are documented (Rychlý 2008 for logDice,
Church–Hanks for MI). We already match the *formulas*. We can't match
their specific corpus outputs because of preprocessing differences
(tokenisation, lemmatisation). So:

- ✅ Validate the formula → done in `test_collocation_measures.py`
- ❌ Don't try to match SketchEngine's numbers on the BNC — apples to oranges.

### Recommended ordering (when we get here)

1. **More Rayson values** — 5-minute job.
2. **Scattertext worked example** — open data, ~30 lines of test, pure Python.
3. **`quanteda` cross-check via `rpy2`** — the big credibility win. Adds R to CI; probably ~50 lines of test code plus a new CI job.
4. **`histwords` partial replication** — bigger because COHA data isn't tiny.

### Open question for when we get here

Should the `quanteda` cross-check be a **required CI job** (paranoid,
slows every PR) or a **nightly job** (looser, weekly assurance)? My
default would be nightly + a smaller subset of checks on every PR.
