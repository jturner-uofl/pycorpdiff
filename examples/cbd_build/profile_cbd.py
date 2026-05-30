"""Profile the CBD tweet archive directly from the zip (no full extract).

Read-only: streams each DONE3/YYYY-MM-DD.csv, counts rows, aggregates
monthly volume, and verifies schema consistency. Drops the
undocumented COMMERCIAL_CBD* fields entirely. Reports the numbers we
need to plan the case-study notebook (total tweets, monthly arc,
duplicate rate, 'cbd' literal-token rate over time).
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ZIP = Path(os.environ.get(
    "CBD_DONE3_ZIP",
    "/Users/jasonturner/Downloads/DONE3-20260529T130022Z-3-001.zip",  # author's local default
))
USE_COLS = ["date", "tweet", "username"]  # ignore COMMERCIAL_CBD*, preproc variants
_CBD_RE = re.compile(r"\bcbd\b", re.IGNORECASE)


def main() -> None:
    zf = zipfile.ZipFile(ZIP)
    names = [n for n in zf.namelist() if re.search(r"DONE3/\d{4}-\d{2}-\d{2}\.csv$", n)]
    names.sort()
    print(f"{len(names)} daily CSV files in archive")

    total = 0
    monthly: Counter[str] = Counter()
    monthly_cbd: Counter[str] = Counter()
    schema_variants: Counter[tuple] = Counter()
    id_seen: set[str] = set()
    dup = 0
    empties = 0

    for i, name in enumerate(names):
        with zf.open(name) as fh:
            raw = fh.read()
        try:
            df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in USE_COLS + ["id"],
                             dtype=str, on_bad_lines="skip")
        except Exception as e:
            print(f"  ! {name}: {e}")
            continue
        if "tweet" not in df.columns:
            # capture the actual header to diagnose
            hdr = pd.read_csv(io.BytesIO(raw), nrows=0)
            schema_variants[tuple(hdr.columns)[:6]] += 1
            continue
        n = len(df)
        total += n
        month = name.split("/")[-1][:7]  # YYYY-MM
        monthly[month] += n
        texts = df["tweet"].fillna("")
        empties += int((texts.str.len() == 0).sum())
        monthly_cbd[month] += int(texts.map(lambda t: bool(_CBD_RE.search(t))).sum())
        if "id" in df.columns:
            ids = df["id"].dropna().tolist()
            for tid in ids:
                if tid in id_seen:
                    dup += 1
                else:
                    id_seen.add(tid)
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(names)} files, {total:,} tweets so far")

    print(f"\nTOTAL tweets: {total:,}")
    print(f"Duplicate tweet-ids: {dup:,} ({100 * dup / max(1, total):.1f}%)")
    print(f"Empty-text rows: {empties:,}")
    print(f"Literal 'cbd' token present in: "
          f"{sum(monthly_cbd.values()):,} ({100 * sum(monthly_cbd.values()) / max(1, total):.1f}%)")

    # Yearly rollup
    yearly: Counter[str] = Counter()
    for m, c in monthly.items():
        yearly[m[:4]] += c
    print("\nTweets per year:")
    for y in sorted(yearly):
        print(f"  {y}: {yearly[y]:>9,}")

    # Monthly arc (compact): print every 6th month
    months = sorted(monthly)
    print(f"\nMonthly span: {months[0]} -> {months[-1]} ({len(months)} months)")
    print("Sampled monthly volume (every 6th month):")
    for m in months[::6]:
        cbd_rate = 100 * monthly_cbd[m] / max(1, monthly[m])
        print(f"  {m}: {monthly[m]:>7,} tweets | 'cbd' literal in {cbd_rate:4.0f}%")

    if schema_variants:
        print("\nNON-STANDARD schemas seen:")
        for cols, cnt in schema_variants.most_common(5):
            print(f"  {cnt}x: {cols}")


if __name__ == "__main__":
    main()
