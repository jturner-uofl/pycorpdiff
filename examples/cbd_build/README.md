# CBD case-study build pipeline

These scripts construct `examples/cbd_case_study.ipynb` from a local
archive of ~3.6 million Twitter CBD tweets (2011-2021).

## Scope

- `profile_cbd.py` — read-only profiler that streams the daily-CSV zip
  archive and reports raw row counts, schema variants, the 2014-07
  collection anomaly, and per-month volume. Used at design time only.
- `build_cbd_corpus.py` — streams the same zip, drops empties, dedups
  by tweet id, topic-filters to `cbd|cannabidiol`, cleans text, writes
  `data/cbd_tweets_2011_2021.parquet` (~300 MB).
- `build_cbd_notebook.py` — constructs `cbd_case_study.ipynb` cell-by-
  cell against the parquet. The notebook itself is committed under
  `examples/cbd_case_study.ipynb`; this script is the source of truth
  for its structure.

## Reproducibility caveats

- **The source zip is not redistributable.** The corpus was collected
  under the pre-2023 Twitter Developer Agreement, which restricts
  redistribution of raw tweet text. An external auditor must obtain
  their own corpus (or contact the author for restricted access) and
  point the scripts at it via the `CBD_DONE3_ZIP` environment variable.
- **The derived parquet is also not redistributed** for the same reason.
  `data/` is gitignored. Only aggregate-level outputs (the notebook's
  HTML, counts, rates, term lists) are published; raw text and
  usernames are never displayed.
- **Pre-registration in the notebook is procedural, not git-provenanced.**
  See `cbd_case_study.ipynb §0b` for the disclosure.

## Re-running

```bash
# 1. Point at your local copy of the source archive
export CBD_DONE3_ZIP=/path/to/your/DONE3-*.zip

# 2. Build the parquet (~5 min on a laptop)
python examples/cbd_build/build_cbd_corpus.py

# 3. Construct the notebook
python examples/cbd_build/build_cbd_notebook.py

# 4. Execute (~15-20 min — SBERT trajectory + neighborhood drift + BERTopic)
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 \
    examples/cbd_case_study.ipynb

# 5. Render HTML
jupyter nbconvert --to html --embed-images examples/cbd_case_study.ipynb
```

## Dependencies beyond pycorpdiff

The notebook uses `bertopic` (§10), `sentence-transformers` (§2/§3/§10),
`umap-learn` and `hdbscan` (§10 transitively). Install with:

```bash
pip install pycorpdiff==0.1.0a25 bertopic
```
