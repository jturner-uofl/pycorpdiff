"""Build a cleaned, topic-filtered CBD tweet corpus from the daily-CSV zip.

Output (gitignored, LOCAL ONLY -- tweet text is not redistributable):
    data/cbd_tweets_2011_2021.parquet

Steps:
  1. Stream each DONE3/YYYY-MM-DD.csv from the zip (tolerant pandas parse).
  2. Keep date / tweet / username / id; drop the undocumented
     COMMERCIAL_CBD* fields and the pre-baked text variants.
  3. Drop empty-text rows; dedup by tweet id.
  4. Topical filter: tweet contains '\\bcbd\\b' or '\\bcannabidiol\\b'
     (word boundaries -- aligned with profile_cbd.py to avoid keeping
     substring matches like 'Cbdistillery'; audit v1 finding 5.4).
  5. Clean text: HTML-unescape, strip URLs, collapse whitespace.

Per-stage row-drop manifest is logged to stderr AND written to a
sidecar JSON next to the output parquet so an external auditor can
reproduce the cleaning pipeline's filter behaviour (audit v1 finding 5.4).
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import sys
import warnings
import zipfile
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Inputs / outputs are environment-overridable so this script is portable.
# Defaults reflect the author's local layout; an external auditor should
# set CBD_DONE3_ZIP (and optionally CBD_OUT_PARQUET) to their own paths.
ZIP = Path(os.environ.get(
    "CBD_DONE3_ZIP",
    "/Users/jasonturner/Downloads/DONE3-20260529T130022Z-3-001.zip",
))
OUT = Path(os.environ.get(
    "CBD_OUT_PARQUET",
    str(Path(__file__).resolve().parent.parent / "data" / "cbd_tweets_2011_2021.parquet"),
))

# Word-boundary topical filter -- matches "cbd" as a standalone token, not as
# a substring inside "Cbdistillery" or similar. Aligned with profile_cbd.py.
# Audit v1 finding 5.4: previous builder used `cbd|cannabidiol` (no \b),
# which kept substring matches; profile_cbd.py used `\bcbd\b`. Now identical.
_TOPICAL = re.compile(r"\b(?:cbd|cannabidiol)\b", re.IGNORECASE)
_URL = re.compile(r"https?://\S+|www\.\S+|\b\w+\.(?:com|net|org|ly|co)/\S*", re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = html.unescape(text)
    t = _URL.sub(" ", t)
    return _WS.sub(" ", t).strip()


def main() -> None:
    zf = zipfile.ZipFile(ZIP)
    names = sorted(
        n for n in zf.namelist() if re.search(r"DONE3/\d{4}-\d{2}-\d{2}\.csv$", n)
    )
    print(f"{len(names)} daily files in archive")

    # Per-stage drop counters (audit v1 finding 5.4: no silent drops).
    stats = {
        "n_files_total": len(names),
        "n_files_parse_failed": 0,
        "n_files_missing_tweet_col": 0,
        "n_files_kept": 0,
        "raw_rows_read": 0,
        "rows_empty_text_dropped": 0,
        "rows_topical_filter_kept": 0,
        "rows_topical_filter_dropped": 0,
        "rows_after_dedup_by_id": 0,
        "rows_dropped_in_dedup": 0,
        "rows_date_parse_failed_dropped": 0,
        "rows_after_clean_empty_dropped": 0,
        "rows_final": 0,
        "files_parse_failed_sample": [],
        "files_missing_tweet_col_sample": [],
    }

    parts: list[pd.DataFrame] = []
    for i, name in enumerate(names):
        raw = zf.read(name)
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                usecols=lambda c: c in ("date", "tweet", "username", "id"),
                dtype=str,
                on_bad_lines="skip",
                engine="c",
            )
        except Exception as e:
            stats["n_files_parse_failed"] += 1
            if len(stats["files_parse_failed_sample"]) < 5:
                stats["files_parse_failed_sample"].append(
                    {"file": name, "error": f"{type(e).__name__}: {str(e)[:100]}"}
                )
            continue
        if "tweet" not in df.columns:
            stats["n_files_missing_tweet_col"] += 1
            if len(stats["files_missing_tweet_col_sample"]) < 5:
                stats["files_missing_tweet_col_sample"].append(
                    {"file": name, "columns_seen": list(df.columns)[:8]}
                )
            continue
        stats["n_files_kept"] += 1
        n_raw = len(df)
        stats["raw_rows_read"] += n_raw

        non_empty = df["tweet"].notna() & (df["tweet"].str.strip() != "")
        stats["rows_empty_text_dropped"] += int((~non_empty).sum())
        df = df[non_empty]

        topical = df["tweet"].str.contains(_TOPICAL)
        stats["rows_topical_filter_kept"] += int(topical.sum())
        stats["rows_topical_filter_dropped"] += int((~topical).sum())
        df = df[topical]

        if len(df):
            parts.append(df)
        if (i + 1) % 500 == 0:
            kept = sum(len(p) for p in parts)
            print(f"  ...{i + 1}/{len(names)} files; kept {kept:,} rows so far", file=sys.stderr)

    corpus = pd.concat(parts, axis=0, ignore_index=True)
    print(f"\n[Stage 1] Raw rows read: {stats['raw_rows_read']:,}")
    print(f"[Stage 2] After empty-text drop: -{stats['rows_empty_text_dropped']:,}")
    print(f"[Stage 3] After topical filter ({_TOPICAL.pattern}): {stats['rows_topical_filter_kept']:,} kept, "
          f"{stats['rows_topical_filter_dropped']:,} dropped")
    print(f"  (Note: files that failed to parse or lacked a 'tweet' column were skipped: "
          f"{stats['n_files_parse_failed']} parse failures, "
          f"{stats['n_files_missing_tweet_col']} schema mismatches.)")

    if "id" in corpus.columns:
        before = len(corpus)
        corpus = corpus.drop_duplicates(subset="id", keep="first")
        stats["rows_dropped_in_dedup"] = before - len(corpus)
        stats["rows_after_dedup_by_id"] = len(corpus)
        print(f"[Stage 4] After dedup by id: {len(corpus):,} (-{stats['rows_dropped_in_dedup']:,})")

    before_date = len(corpus)
    corpus["date"] = pd.to_datetime(corpus["date"], errors="coerce", utc=True)
    corpus = corpus[corpus["date"].notna()]
    stats["rows_date_parse_failed_dropped"] = before_date - len(corpus)
    corpus["date"] = corpus["date"].dt.tz_localize(None)
    print(f"[Stage 5] After date-parse drop: {len(corpus):,} (-{stats['rows_date_parse_failed_dropped']:,})")

    before_clean = len(corpus)
    corpus["text"] = corpus["tweet"].map(clean)
    corpus = corpus[corpus["text"].str.len() > 0]
    stats["rows_after_clean_empty_dropped"] = before_clean - len(corpus)
    print(f"[Stage 6] After text-clean empty-drop: {len(corpus):,} "
          f"(-{stats['rows_after_clean_empty_dropped']:,})")

    corpus["year"] = corpus["date"].dt.year
    corpus["year_month"] = corpus["date"].dt.to_period("M").astype(str)

    out = corpus[["date", "year", "year_month", "text", "username"]].reset_index(drop=True)
    stats["rows_final"] = len(out)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False, compression="snappy")
    print(f"\nWrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB), {len(out):,} tweets")
    print(f"  date range: {out['date'].min()} -> {out['date'].max()}")

    # Write sidecar manifest so the cleaning pipeline is auditable
    # (audit v1 finding 5.4).
    manifest_path = OUT.with_suffix(".manifest.json")
    stats["topical_regex"] = _TOPICAL.pattern
    stats["url_regex"] = _URL.pattern
    stats["output_parquet"] = str(OUT)
    stats["output_parquet_bytes"] = OUT.stat().st_size
    stats["date_min"] = str(out["date"].min())
    stats["date_max"] = str(out["date"].max())
    manifest_path.write_text(json.dumps(stats, indent=2))
    print(f"  manifest written: {manifest_path}")

    # Clean monthly arc (every 6th month) + 2014-07 resolution.
    monthly = out.groupby("year_month").size()
    print("\nClean monthly volume (every 6th month):")
    for m in monthly.index[::6]:
        print(f"  {m}: {monthly[m]:>7,}")
    print(f"\n2014-07 after cleaning: {int(monthly.get('2014-07', 0)):,} CBD tweets")
    print("Tweets per year (clean):")
    for y, c in out.groupby("year").size().items():
        print(f"  {y}: {c:>8,}")


if __name__ == "__main__":
    main()
