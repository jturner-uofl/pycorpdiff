"""Google Books Ngrams fetcher for cross-corpus validation.

Companion to the PubMed terminology case study: same 5 headline
shifts (mongolism→Down syndrome, shell shock→PTSD, MPD→DID,
mental retardation→intellectual disability, "committed"→"died by"
suicide), but queried against the Google Books English-2019 corpus
instead of PubMed.

The cross-corpus contrast lets us ask: does scientific-lit
terminology retirement match popular/published-books usage?

The Google Books Ngrams API is free, no auth, returns per-year
normalized frequency (relative to total tokens published that year
in the English-2019 corpus). Endpoint:

  https://books.google.com/ngrams/json
    ?content=<ngram1>,<ngram2>,...      # comma-separated, +-space-encoded
    &year_start=1950&year_end=2019      # year range
    &corpus=en-2019                     # English-2019 corpus (most comprehensive)
    &smoothing=0                        # 0 = raw counts; default 3 = moving average

Notes on the API:
 * URL encoding: multi-word phrases use `+` for spaces.
 * Up to ~12 ngrams per request; we batch by shift.
 * Response is JSON array; each element has `ngram`, `type`, `timeseries`.
 * `timeseries` is a list of per-year frequencies starting at year_start.
 * Coverage thins out after ~2019; we cap year_end at 2019 for the
   en-2019 corpus to avoid sparse-tail artefacts.

Public-domain data per Google's terms for non-commercial research.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

BOOKS_BASE = "https://books.google.com/ngrams/json"
USER_AGENT = (
    "pycorpdiff-books-ngrams/0.1 "
    "(https://github.com/jturner-uofl/pycorpdiff)"
)

_TRANSIENT_EXCS = (HTTPError, URLError, TimeoutError,
                   http.client.HTTPException, ConnectionError)


def _http_get_json(url: str, *, max_retries: int = 5) -> list:
    """GET a Books-Ngrams JSON endpoint with retry."""
    backoff = 1.0
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return json.loads(text, strict=False)
        except _TRANSIENT_EXCS as e:
            last_err = e
            if isinstance(e, HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"books-ngrams http_get failed: {last_err}")


def fetch_ngrams(
    ngrams: list[str],
    *,
    year_start: int = 1900,
    year_end: int = 2019,
    corpus: str = "en-2019",
    smoothing: int = 0,
    case_insensitive: bool = True,
    sleep: float = 1.0,
) -> pd.DataFrame:
    """Fetch a batch of ngrams from Google Books. Returns long-form DataFrame.

    Columns: ngram (str), year (int), frequency (float).

    The Google API caps queries at ~12 ngrams; we don't enforce here
    but the caller should batch accordingly.
    """
    if not ngrams:
        return pd.DataFrame(columns=["ngram", "year", "frequency"])
    params = {
        "content": ",".join(ngrams),
        "year_start": year_start,
        "year_end": year_end,
        "corpus": corpus,
        "smoothing": smoothing,
    }
    if case_insensitive:
        params["case_insensitive"] = "true"
    url = f"{BOOKS_BASE}?{urlencode(params)}"
    data = _http_get_json(url)
    time.sleep(sleep)
    # When case_insensitive=true, the API returns one entry per case-variant
    # (e.g. "mongolism", "Mongolism", "MONGOLISM") PLUS a combined entry
    # "mongolism (All)" that sums all variants. We want the combined entry —
    # it's the right denominator for a case-insensitive lookup.
    rows: list[dict] = []
    # First, build a map of base-ngram -> combined-entry timeseries if "(All)"
    # variants exist; otherwise fall back to the plain entry.
    canonical_by_base: dict[str, dict] = {}
    for entry in data:
        full = entry.get("ngram", "")
        if full.endswith(" (All)"):
            base = full[:-6]  # strip " (All)"
            canonical_by_base[base.lower()] = entry
    # Then iterate the input ngrams and emit one (ngram, year, frequency) per
    # year. Match case-insensitively.
    requested_lower = {n.lower(): n for n in ngrams}
    for req_lower, req_orig in requested_lower.items():
        if req_lower in canonical_by_base:
            entry = canonical_by_base[req_lower]
        else:
            # No "(All)" combined — find the first entry whose lowercased ngram
            # matches the request (single-case-variant fallback).
            entry = next(
                (e for e in data
                 if e.get("ngram", "").lower() == req_lower),
                None,
            )
        if entry is None:
            print(f"  WARNING: no timeseries returned for {req_orig!r}",
                  file=sys.stderr)
            continue
        ts = entry.get("timeseries", [])
        for i, freq in enumerate(ts):
            rows.append({
                "ngram": req_orig,
                "year": year_start + i,
                "frequency": float(freq),
            })
    return pd.DataFrame(rows)


# Five headline shifts — same as the PubMed case study. Each side is a list
# of ngrams to OR-equivalent (we sum their frequencies in post-processing).
BOOKS_SHIFTS: dict[str, dict[str, list[str]]] = {
    "1960s_down": {
        "old": ["mongolism"],
        "new": ["Down syndrome", "trisomy 21"],
    },
    "1980s_ptsd": {
        "old": ["shell shock", "war neurosis", "combat fatigue"],
        "new": ["PTSD", "posttraumatic stress disorder"],
    },
    "1990s_did": {
        "old": ["multiple personality disorder", "multiple personality"],
        "new": ["dissociative identity disorder"],
    },
    "2010s_id": {
        "old": ["mental retardation"],
        "new": ["intellectual disability"],
    },
    "neg_suicide_phrasing": {
        "old": ["committed suicide"],
        "new": ["died by suicide"],
    },
}


def run_books_sweep(
    out_csv: Path,
    *,
    year_start: int = 1900,
    year_end: int = 2019,
    corpus: str = "en-2019",
) -> pd.DataFrame:
    """Fetch all 5 shifts × old/new from Google Books Ngrams. Caches per shift."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = out_csv.parent / "books_ngrams_cache"
    cache_dir.mkdir(exist_ok=True)

    frames: list[pd.DataFrame] = []
    for shift, sides in BOOKS_SHIFTS.items():
        cache = cache_dir / f"{shift}.csv"
        if cache.exists():
            print(f"[cache] {shift}: reusing {cache.name}", file=sys.stderr)
            df = pd.read_csv(cache)
        else:
            # Query all ngrams for this shift in one batch (~5 ngrams per
            # shift max, well under the ~12 limit).
            all_ngrams = sides["old"] + sides["new"]
            print(f"[fetch] {shift}: {all_ngrams}", file=sys.stderr)
            df = fetch_ngrams(all_ngrams, year_start=year_start,
                              year_end=year_end, corpus=corpus)
            df["shift"] = shift
            # Annotate which side each ngram is on
            side_map = {n: "old" for n in sides["old"]}
            side_map.update({n: "new" for n in sides["new"]})
            df["side"] = df["ngram"].map(side_map)
            df.to_csv(cache, index=False)
        frames.append(df)
        # Show a quick crossover preview for this shift
        if "side" in df.columns:
            agg = df.groupby(["year", "side"])["frequency"].sum().unstack(
                "side", fill_value=0
            )
            if "old" in agg.columns and "new" in agg.columns:
                mask = (agg["new"] > agg["old"]) & ((agg["old"] + agg["new"]) > 0)
                cross = int(mask.idxmax()) if mask.any() else None
                old_peak_yr = int(agg["old"].idxmax()) if agg["old"].max() > 0 else None
                print(f"  {shift}: old-peak={agg['old'].max():.3e} "
                      f"in {old_peak_yr}; crossover={cross}", file=sys.stderr)

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out_csv, index=False)
    print(f"\nWrote {len(full):,} rows to {out_csv}", file=sys.stderr)
    return full


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year-start", type=int, default=1900)
    p.add_argument("--year-end", type=int, default=2019)
    p.add_argument("--corpus", default="en-2019")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "books_ngrams_counts.csv",
    )
    args = p.parse_args(argv)
    run_books_sweep(args.out, year_start=args.year_start,
                    year_end=args.year_end, corpus=args.corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
