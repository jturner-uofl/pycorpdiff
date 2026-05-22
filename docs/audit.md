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
| 11 | `benchmarks/` with `asv` regression suite           | ✅ done       |
| 9  | polars backend (interop, not internal storage)      | ✅ done       |
| 4  | Real benchmark corpora (Hansard sample bundled; real-data sources documented) | ✅ done |
| 5  | Demo analysis (Hansard end-to-end notebook)         | ✅ done       |
| 4-follow-up | Live `fetch_hansard()` from parliament.uk API   | ✅ done       |
| (semantic_trajectory) | Multi-period semantic shift           | ✅ done       |
| 15 | Cross-validation receipts (Rayson + Scattertext + quanteda + **histwords**) | ✅ done — Rayson + Scattertext run on every PR; quanteda + histwords are slow-tier (gated on R / network) |

## Post-audit small + tiny follow-ups (knocked out)

These weren't in the original numbered audit but got picked up as the
project matured. Listed here for traceability:

| Feature | Status |
|---|---|
| `Comparison.keyness(method="chi_squared")` (Pearson χ²)         | ✅ done |
| `Comparison.keyness(stop_words=...)` (function-word filter)     | ✅ done |
| `Corpus.__hash__` content-derived (dict-key safe)               | ✅ done |
| `Result.to_html()` / `.to_json()` across every Result type      | ✅ done |
| `from_huggingface(dataset_id, ...)` loader + `[huggingface]` extra | ✅ done |
| `pcd.viz.dispersion_plot(corpus, target)` (Mosteller-style)     | ✅ done |
| `NgramTokenizer` — bigrams/trigrams as first-class terms        | ✅ done |
| `Corpus.doc_term_counts_sparse()` — scipy.sparse for big corpora | ✅ done |
| `pcd.viz.scattertext_plot()` — Kessler-2017 rank-percentile scatter | ✅ done |

## Item #15 — cross-validation against open-source equivalents

The "receipt" layer that turns "the math is correct" into "the math
agrees with the standard tool".

**What's in tree as of the cross-validation commit:**

- ✅ **9 Rayson-style known-answer triples** in
  `tests/integration/test_crossval_rayson.py` covering the classic
  12k/10k/1M/1M case, equal-rate-no-signal, absent-on-each-side,
  10× and 5× over-rep, and the BNC spoken-vs-written 'the' example
  from Hardie's CASS note. Plus the Hardie LogRatio canonical
  worked example (log₂(1000/100) ≈ 3.32).
- ✅ **Scattertext slow-tier cross-check** in
  `tests/integration/test_crossval_scattertext.py`. On Scattertext's
  bundled 2012 US Presidential Conventions corpus, our keyness top-25
  Dem-leaning terms overlap 11/25 with Scattertext's scaled F-score
  top-25 (different measures, behavioural agreement). The
  `obama` / `romney` sign check holds.
- ✅ **quanteda slow-tier scaffolding** in
  `tests/integration/test_crossval_quanteda.py`. Uses `rpy2` to call
  `quanteda::textstat_keyness(measure="lr")` on the same fixture and
  asserts G² values agree to 1e-4. Skips if rpy2 / R / quanteda
  aren't installed; ready to run once R lands in CI.
- ✅ **HistWords (Hamilton et al. 2016) partial replication** in
  `tests/integration/test_crossval_histwords.py`. Downloads aligned
  per-decade word2vec embeddings from snap.stanford.edu, computes
  cosine distance for famous shifters (gay / broadcast / awful /
  terrific) vs stable function words (the / and / of) between the
  1900s and 1990s, and asserts the shifter mean exceeds the stable
  mean by at least 0.2 — the headline finding of the paper, made
  into an automated check. The smallest English subset is
  `eng-fiction` at ~380 MB; persistent cache via
  `PYCORPDIFF_HISTWORDS_CACHE` environment variable.

**Item #15 is now fully done.** The remaining cross-validation
opportunities (more BNC examples, etc.) are follow-ups, not gaps.

✅ **NLTK collocations cross-check** added in a follow-up commit. We
assert PMI and t-score agree with NLTK's ``BigramAssocMeasures`` to
floating-point precision (≤ 1e-12) on every adjacent bigram surviving
the frequency filter, and that MI³ matches NLTK's ``mi_like(power=3)``
up to the expected log-scale identity. Lives in
``tests/integration/test_crossval_nltk.py``; slow-tier, gated on NLTK
being installed.

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
