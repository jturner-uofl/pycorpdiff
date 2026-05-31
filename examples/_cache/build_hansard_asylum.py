"""Fetch UK Hansard contributions matching 'asylum', 2014-2023, with party enrichment.

This is the data-prep step for the asylum case study
(``examples/jss_case_study.ipynb``).
The Hansard search API returns spoken contributions; the Members API
provides party affiliation per MemberId. Output is a parquet cached
alongside the case-study notebook so analytical iterations don't
re-hit the network.

Run with:
    python examples/_cache/build_hansard_asylum.py
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "hansard_asylum_2014_2023.parquet"
SEARCH_URL = "https://hansard-api.parliament.uk/search/contributions/Spoken.json"
MEMBER_URL = "https://members-api.parliament.uk/api/Members/{member_id}"
UA = "pycorpdiff-jss-case-study/0.1"


def get_json(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_contributions(
    query: str, start: str, end: str, page_size: int = 50, max_results: int | None = None
) -> list[dict]:
    """Paginate through all spoken contributions matching `query`."""
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
        data = get_json(url)
        results = data.get("Results", [])
        if not results:
            break
        out.extend(results)
        total = int(data.get("TotalResultCount", 0))
        skip += len(results)
        if max_results is not None and len(out) >= max_results:
            return out[:max_results]
        if skip >= total:
            break
        if skip % 500 == 0:
            print(f"  fetched {skip}/{total}")
    return out


def enrich_members(member_ids: list[int]) -> dict[int, dict]:
    """One Members-API call per unique MemberId; returns party info."""
    party_map: dict[int, dict] = {}
    for i, mid in enumerate(member_ids, 1):
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
            print(f"  enriched {i}/{len(member_ids)}")
    return party_map


def main() -> None:
    if OUT.exists():
        print(f"Already cached: {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
        df = pd.read_parquet(OUT)
        print(f"  {len(df):,} rows; columns: {list(df.columns)}")
        return

    t_start = time.time()
    print("Fetching Hansard contributions matching 'asylum', 2014-01-01 to 2023-12-31...")
    rows = fetch_contributions(
        query="asylum", start="2014-01-01", end="2023-12-31", page_size=50
    )
    print(f"Fetched {len(rows):,} contributions in {time.time()-t_start:.1f}s")

    df = pd.DataFrame(rows)
    print(f"\nRaw columns: {list(df.columns)}")

    # Enrich with party info
    unique_members = sorted(df["MemberId"].dropna().astype(int).unique().tolist())
    print(f"\nEnriching {len(unique_members)} unique members with party info...")
    party_map = enrich_members(unique_members)
    party_df = pd.DataFrame.from_dict(party_map, orient="index").reset_index()
    party_df.columns = ["MemberId", "party", "party_abbrev"]
    df = df.merge(party_df, on="MemberId", how="left")

    # Canonical columns for the case study
    out = pd.DataFrame({
        "text": df["ContributionTextFull"].fillna(df["ContributionText"]).fillna(""),
        "member": df["MemberName"].fillna(""),
        "party": df["party"].fillna(""),
        "party_abbrev": df["party_abbrev"].fillna(""),
        "date": pd.to_datetime(df["SittingDate"]).dt.tz_localize(None),
        "house": df["House"].fillna(""),
        "debate_title": df["DebateSection"].fillna(""),
        "hansard_id": df["ContributionExtId"].fillna(""),
    })
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    out.to_parquet(OUT, index=False, compression="snappy")
    print(f"\nWrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  {len(out):,} rows after dropping empty-text")
    print(f"  date range: {out['date'].min()} to {out['date'].max()}")
    print(f"  party value-counts:\n{out['party_abbrev'].value_counts().head(10)}")


if __name__ == "__main__":
    main()
