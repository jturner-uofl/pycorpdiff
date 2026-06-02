"""Fetch sepsis-related trials from ClinicalTrials.gov + classify by criteria framework.

The PubMed §5.5 case study established that the Sepsis-3 (2016)
operational-definition revision propagated into peer-reviewed
medical literature on the predicted timeline (first Sepsis-3 record
in PubMed 2016, within the 2015-2017 pre-registered window). §5.5b
extends that into a *second* corpus — ClinicalTrials.gov trial
registrations — to test whether the same shift propagated into
clinical-trial *design* + *registration* language.

The hypothesis: trials registered post-2016 that use sepsis in
their eligibility criteria should increasingly cite SOFA / qSOFA
(the Sepsis-3 framework) rather than SIRS (the Sepsis-2 / 1991
framework). The rate of post-2016 trials citing SOFA/qSOFA vs
SIRS is the cross-corpus check.

Why this matters as a cross-corpus validation:

  - PubMed measures what researchers *publish* (peer-reviewed
    medical-literature usage)
  - ClinicalTrials.gov measures what researchers *register*
    (operational study-design usage, pre-publication)
  - Trial registration typically occurs 6-12 months before
    publication; if Sepsis-3 propagated into trial design by
    2017-2018, the PubMed publication-level signal at 2016
    would correctly precede most peer-reviewed mentions

API: ClinicalTrials.gov v2 REST API (https://clinicaltrials.gov/api/v2)
Public-domain US-government data.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd


API_BASE = "https://clinicaltrials.gov/api/v2/studies"
CACHE = Path(__file__).resolve().parents[1] / "data" / "sepsis_clinicaltrials.parquet"
OUT_CSV = Path(__file__).resolve().parents[1] / "data" / "sepsis_clinicaltrials_by_year.csv"


def _fetch_page(params: dict, max_retries: int = 5, retry_delay: float = 3.0) -> dict:
    """Fetch one page of the v2 API with simple retry-on-transient-error."""
    url = f"{API_BASE}?{urlencode(params)}"
    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"Accept": "application/json",
                                          "User-Agent": "pycorpdiff-jss-cross-corpus/1.0"})
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt < max_retries - 1:
                print(f"  retry {attempt+1}/{max_retries}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise


def fetch_sepsis_trials(
    out_parquet: Path,
    *,
    query_term: str = "sepsis",
    page_size: int = 1000,
    max_pages: int = 50,
) -> pd.DataFrame:
    """Fetch sepsis-related trials from ClinicalTrials.gov v2 API.

    Strategy: a single broad query.term=sepsis fetch with all the
    framework-disambiguating fields. Classification happens in
    `classify_corpus` after the fact.
    """
    if out_parquet.exists():
        print(f"[cache] reusing {out_parquet}", file=sys.stderr)
        return pd.read_parquet(out_parquet)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Fields to retrieve (v2 API uses dot-path field selectors).
    # We need: NCT ID, brief title, brief summary, detailed description,
    # eligibility criteria text, first-posted date.
    fields = ",".join([
        "NCTId",
        "BriefTitle",
        "BriefSummary",
        "DetailedDescription",
        "EligibilityCriteria",
        "StudyFirstPostDate",
        "StudyType",
        "Phase",
    ])

    all_studies: list[dict] = []
    next_token: str | None = None
    page = 0
    while page < max_pages:
        page += 1
        params: dict = {
            "query.term": query_term,
            "pageSize": page_size,
            "fields": fields,
            "countTotal": "true",
        }
        if next_token:
            params["pageToken"] = next_token

        print(f"  page {page}: fetching up to {page_size} studies "
              f"(running total: {len(all_studies):,})...",
              file=sys.stderr)
        data = _fetch_page(params)

        studies = data.get("studies", [])
        if not studies:
            break

        for s in studies:
            ps = s.get("protocolSection", {})
            ident = ps.get("identificationModule", {})
            desc = ps.get("descriptionModule", {})
            elig = ps.get("eligibilityModule", {})
            status = ps.get("statusModule", {})
            design = ps.get("designModule", {})

            all_studies.append({
                "nct_id": ident.get("nctId", ""),
                "brief_title": ident.get("briefTitle", "") or "",
                "brief_summary": desc.get("briefSummary", "") or "",
                "detailed_description": desc.get("detailedDescription", "") or "",
                "eligibility_criteria": elig.get("eligibilityCriteria", "") or "",
                "first_posted_date": (status.get("studyFirstPostDateStruct", {})
                                       .get("date", "")),
                "study_type": design.get("studyType", ""),
                "phases": ",".join(design.get("phases", []) or []),
            })

        next_token = data.get("nextPageToken")
        if not next_token:
            break

    df = pd.DataFrame(all_studies)
    if not len(df):
        print("no studies returned", file=sys.stderr)
        return df

    # Parse first-posted date -> year
    df["first_posted_date"] = pd.to_datetime(df["first_posted_date"],
                                              errors="coerce")
    df["first_posted_year"] = df["first_posted_date"].dt.year

    # Build combined text field for sense classification
    df["combined_text"] = (df["brief_title"].fillna("") + " " +
                            df["brief_summary"].fillna("") + " " +
                            df["detailed_description"].fillna("") + " " +
                            df["eligibility_criteria"].fillna(""))

    df = df.dropna(subset=["first_posted_year"]).reset_index(drop=True)
    df["first_posted_year"] = df["first_posted_year"].astype(int)

    df.to_parquet(out_parquet, index=False)
    print(f"\nwrote {len(df):,} sepsis-related trials to {out_parquet}",
          file=sys.stderr)
    return df


# ----------------- Criteria-framework classification -----------------

# First-match-wins regex buckets. SIRS framework gets first dibs (so a
# trial mentioning both SIRS and SOFA counts as SIRS — the older
# framework — which is the conservative direction relative to the
# "Sepsis-3 propagated" claim).

FRAMEWORK_PATTERNS: list[tuple[str, str]] = [
    # --- Sepsis-3 / qSOFA framework (post-2016) ---
    ("sepsis3_qsofa",
     r"\b(sepsis.?3|sepsis\s*3|"
     r"quick\s+sofa|qsofa|"
     r"third\s+international\s+consensus.*(sepsis|septic\s+shock)|"
     r"singer\s+et\s+al.*2016|"
     r"jama\s+2016.*sepsis)\b"),

    # --- SOFA-based but not explicitly Sepsis-3 (transitional) ---
    ("sofa_score_based",
     r"\b(sofa\s+score|"
     r"sequential\s+organ\s+failure\s+assessment|"
     r"sofa\s+(criteria|threshold|cutoff)|"
     r"Δsofa|delta\s+sofa)\b"),

    # --- SIRS framework (pre-2016 / classic) ---
    ("sirs_framework",
     r"\b(systemic\s+inflammatory\s+response\s+syndrome|"
     r"\bsirs\s+criteria|"
     r"\bsirs\s+(positive|negative|score|response)|"
     r"two\s+(or\s+more|of\s+(the\s+)?four)\s+sirs|"
     r"meets?\s+sirs|"
     r"accp.?sccm\s+1991|"
     r"bone\s+(et\s+al\.?)?\s*1992|"
     r"sepsis.?2|sepsis\s*2)\b"),

    # --- Severe sepsis (Sepsis-2 era term, retired by Sepsis-3) ---
    ("severe_sepsis_only",
     r"\b(severe\s+sepsis|"
     r"sepsis\s+with\s+organ\s+dysfunction|"
     r"sepsis\s+syndrome)\b"),

    # --- Septic shock / general sepsis (no explicit framework) ---
    ("septic_shock_or_general_sepsis",
     r"\b(septic\s+shock|"
     r"sepsis\s+(patient|subject|management|treatment|patients|"
     r"with|in\s+adult|in\s+pediatric))\b"),
]


_FRAMEWORK_RE: list[tuple[str, re.Pattern]] = [
    (label, re.compile(pat, re.IGNORECASE | re.DOTALL))
    for label, pat in FRAMEWORK_PATTERNS
]


def classify_framework(text: str) -> str:
    for label, pat in _FRAMEWORK_RE:
        if pat.search(text):
            return label
    return "unknown"


def classify_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["framework"] = df["combined_text"].map(classify_framework)
    per_yr = (df.groupby(["first_posted_year", "framework"]).size()
              .unstack("framework", fill_value=0).astype(int))
    per_yr.index.name = "year"
    return per_yr


def summarise(per_yr: pd.DataFrame) -> None:
    print("\n=== ClinicalTrials.gov sepsis-related: per-framework totals ===")
    totals = per_yr.sum(axis=0).sort_values(ascending=False)
    print(totals.to_string())
    print(f"\nGrand total: {int(per_yr.sum().sum()):,}")
    print("\n=== Per-year decomposition (recent) ===")
    print(per_yr.loc[per_yr.index >= 2010].to_string())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--max-pages", type=int, default=30)
    p.add_argument("--abstracts", type=Path, default=CACHE)
    p.add_argument("--counts", type=Path, default=OUT_CSV)
    args = p.parse_args(argv)

    df = fetch_sepsis_trials(
        args.abstracts,
        page_size=args.page_size,
        max_pages=args.max_pages,
    )
    if not len(df):
        return 1

    per_yr = classify_corpus(df)
    per_yr.to_csv(args.counts)
    print(f"\nWrote per-(year, framework) counts to {args.counts}",
          file=sys.stderr)
    summarise(per_yr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
