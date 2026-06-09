"""Fetch UK Hansard contributions matching 'asylum', 2010-2013, with party
enrichment + HTML cleanup, then merge with the existing 2014-2023 cache.

Output: examples/_cache/hansard_asylum_2010_2023.parquet
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
if REPO.name == "tmp":
    REPO = Path("/Users/jasonturner/projects/pycorpdiff")

OLD_CACHE = REPO / "examples/_cache/hansard_asylum_2014_2023.parquet"
NEW_CACHE = REPO / "examples/_cache/hansard_asylum_2010_2023.parquet"
SEARCH_URL = "https://hansard-api.parliament.uk/search/contributions/Spoken.json"
MEMBER_URL = "https://members-api.parliament.uk/api/Members/{member_id}"
UA = "pycorpdiff-jss-case-study/0.1"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    no_tags = _HTML_TAG_RE.sub(" ", raw)
    decoded = html.unescape(no_tags)
    return _WS_RE.sub(" ", decoded).strip()


def get_json(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_contributions(query: str, start: str, end: str, page_size: int = 50) -> list[dict]:
    out: list[dict] = []
    skip = 0
    while True:
        params = {
            "queryParameters.searchTerm": query,
            "queryParameters.startDate": start,
            "queryParameters.endDate": end,
            "queryParameters.take": str(page_size),
            "queryParameters.skip": str(skip),
        }
        url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
        try:
            data = get_json(url)
        except Exception as e:
            print(f"  ! fetch error at skip={skip}: {e}; retrying once after 2s")
            time.sleep(2.0)
            data = get_json(url)
        results = data.get("Results", [])
        if not results:
            break
        out.extend(results)
        total = int(data.get("TotalResultCount", 0))
        skip += len(results)
        if skip % 200 == 0 or skip >= total:
            print(f"  fetched {skip}/{total}")
        if skip >= total:
            break
    return out


def enrich_members(member_ids: list[int], existing: dict[int, dict]) -> dict[int, dict]:
    party_map = dict(existing)
    missing = [m for m in member_ids if m not in party_map]
    print(f"  {len(member_ids)} unique members in extension fetch; "
          f"{len(missing)} not in existing cache, fetching...")
    for i, mid in enumerate(missing, 1):
        try:
            data = get_json(MEMBER_URL.format(member_id=mid))
            value = data.get("value", {})
            party = value.get("latestParty") or {}
            party_map[mid] = {
                "party": party.get("name", ""),
                "party_abbrev": party.get("abbreviation", ""),
            }
        except Exception as e:
            party_map[mid] = {"party": "", "party_abbrev": ""}
            print(f"    warn: member {mid}: {e}")
        if i % 50 == 0:
            print(f"  enriched {i}/{len(missing)}")
    return party_map


def main() -> None:
    if NEW_CACHE.exists():
        print(f"Already cached: {NEW_CACHE}")
        df = pd.read_parquet(NEW_CACHE)
        print(f"  {len(df):,} rows; date range {df['date'].min()} to {df['date'].max()}")
        return

    if not OLD_CACHE.exists():
        raise SystemExit(f"Missing old cache: {OLD_CACHE}")

    print(f"Loading existing 2014-2023 cache from {OLD_CACHE}")
    old_df = pd.read_parquet(OLD_CACHE)
    print(f"  {len(old_df):,} rows")

    # Build member_id → party map from existing cache (so we don't re-fetch known members)
    if "member_id" in old_df.columns:
        existing_member_party = (
            old_df.dropna(subset=["member_id"])
                  .groupby("member_id")[["party", "party_abbrev"]]
                  .first()
                  .to_dict(orient="index")
        )
    else:
        existing_member_party = {}
    print(f"  cached party for {len(existing_member_party)} members")

    t0 = time.time()
    print("Fetching 2010-01-01 to 2013-12-31...")
    rows = fetch_contributions("asylum", "2010-01-01", "2013-12-31")
    print(f"Fetched {len(rows):,} contributions in {time.time()-t0:.1f}s")
    if not rows:
        raise SystemExit("Empty fetch — API endpoint changed or rate-limited.")

    df = pd.DataFrame(rows)
    print(f"Raw columns: {list(df.columns)[:15]}...")

    # Enrich members
    unique_members = sorted(df["MemberId"].dropna().astype(int).unique().tolist())
    party_map = enrich_members(unique_members, existing_member_party)

    party_df = pd.DataFrame.from_dict(party_map, orient="index").reset_index()
    party_df.columns = ["MemberId", "party", "party_abbrev"]
    df = df.merge(party_df, on="MemberId", how="left")

    # Canonical columns + HTML cleanup
    new_part = pd.DataFrame({
        "text": (df["ContributionTextFull"].fillna(df["ContributionText"]).fillna(""))
                  .map(clean_text),
        "member": df["MemberName"].fillna(""),
        "member_id": df["MemberId"],
        "party": df["party"].fillna(""),
        "party_abbrev": df["party_abbrev"].fillna(""),
        "date": pd.to_datetime(df["SittingDate"]).dt.tz_localize(None),
        "house": df["House"].fillna(""),
        "debate_title": df["DebateSection"].fillna(""),
        "hansard_id": df["ContributionExtId"].fillna(""),
    })
    new_part = new_part[new_part["text"].str.len() > 0].reset_index(drop=True)
    print(f"\nNew part: {len(new_part):,} rows after dropping empty text")
    print(f"  date range: {new_part['date'].min()} to {new_part['date'].max()}")

    # Align columns + concat with old
    common_cols = [c for c in new_part.columns if c in old_df.columns]
    if "member_id" not in old_df.columns:
        common_cols = [c for c in common_cols if c != "member_id"]
        new_part = new_part[common_cols]
    print(f"  common columns: {common_cols}")
    merged = pd.concat([new_part[common_cols], old_df[common_cols]],
                       axis=0, ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"\nMerged: {len(merged):,} rows, {merged['date'].min()} to {merged['date'].max()}")
    print(f"  house split: {merged['house'].value_counts().to_dict()}")
    print(f"  top parties: {merged['party_abbrev'].value_counts().head(6).to_dict()}")

    merged.to_parquet(NEW_CACHE, index=False, compression="snappy")
    print(f"\nWrote {NEW_CACHE} ({NEW_CACHE.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
