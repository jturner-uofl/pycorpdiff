"""Build a cleaned, topic-filtered CBD tweet corpus from the daily-CSV zip.

Output (gitignored, LOCAL ONLY — tweet text is not redistributable):
    data/cbd_tweets_2011_2021.parquet

Steps:
  1. Stream each DONE3/YYYY-MM-DD.csv from the zip (tolerant pandas parse).
  2. Keep date / tweet / username / id; drop the undocumented
     COMMERCIAL_CBD* fields and the pre-baked text variants.
  3. Drop empty-text rows; dedup by tweet id.
  4. Topical filter: tweet contains 'cbd' or 'cannabidiol' (case-insensitive).
  5. Clean text: HTML-unescape, strip URLs, collapse whitespace.
"""
from __future__ import annotations

import html
import io
import os
import re
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

_TOPICAL = re.compile(r"cbd|cannabidiol", re.IGNORECASE)
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
    print(f"{len(names)} daily files")

    parts: list[pd.DataFrame] = []
    raw_total = 0
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
        except Exception:
            continue
        if "tweet" not in df.columns:
            continue
        raw_total += len(df)
        df = df[df["tweet"].notna() & (df["tweet"].str.strip() != "")]
        df = df[df["tweet"].str.contains(_TOPICAL)]
        if len(df):
            parts.append(df)
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(names)} files; kept {sum(len(p) for p in parts):,}")

    corpus = pd.concat(parts, axis=0, ignore_index=True)
    print(f"\nRaw rows read: {raw_total:,}")
    print(f"After empty-drop + topical filter: {len(corpus):,}")

    if "id" in corpus.columns:
        before = len(corpus)
        corpus = corpus.drop_duplicates(subset="id", keep="first")
        print(f"After dedup by id: {len(corpus):,} (-{before - len(corpus):,})")

    corpus["date"] = pd.to_datetime(corpus["date"], errors="coerce", utc=True)
    corpus = corpus[corpus["date"].notna()]
    corpus["date"] = corpus["date"].dt.tz_localize(None)
    corpus["text"] = corpus["tweet"].map(clean)
    corpus = corpus[corpus["text"].str.len() > 0]
    corpus["year"] = corpus["date"].dt.year
    corpus["year_month"] = corpus["date"].dt.to_period("M").astype(str)

    out = corpus[["date", "year", "year_month", "text", "username"]].reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False, compression="snappy")
    print(f"\nWrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB), {len(out):,} tweets")
    print(f"  date range: {out['date'].min()} -> {out['date'].max()}")

    # Clean monthly arc (every 6th month) + 2014-07 resolution
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
