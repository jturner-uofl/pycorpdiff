"""Fetch + sense-disambiguate the 'CBD' / 'cannabidiol' PubMed corpus.

The string 'CBD' in PubMed is polysemous — competing senses:

  cannabidiol      : hemp/cannabis pharmacology, the headline-relevant sense
  common bile duct : gastroenterology / hepatology / biliary surgery (CBD is
                     the standard abbreviation for this anatomical structure)
  other            : cubic blood density, congenital bronchial diverticulum,
                     rare niche uses

This is exactly the polysemy collision the §6.5.1 retard\\* WSI methodology
was designed for, in a completely different domain (anatomical-vs-
pharmacological abbreviation collision). The CBD notebook §12 uses these
sense-classified counts to:

  1. Quantify the sense-fraction shift (CBD = ~100% common bile duct in
     2010 → mixed by 2020+).
  2. Extract the cannabidiol-sense per-year rate.
  3. Run Kleinberg burstiness on the cannabidiol rate.
  4. Compare the cannabidiol-PubMed burst onset against the CBD-Twitter
     burst onset (§6 in the CBD notebook).

Public-domain (US-gov NLM E-utilities).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_pubmed_abstracts import (  # noqa: E402
    esearch_pmids,
    efetch_records,
)


CACHE = Path(__file__).resolve().parents[1] / "data" / "cbd_pubmed_abstracts.parquet"
OUT_CSV = Path(__file__).resolve().parents[1] / "data" / "cbd_pubmed_sense_counts_by_year.csv"


# First-match-wins sense regex buckets. Order matters — cannabidiol-clinical
# senses get first dibs, then common-bile-duct anatomy, then chemistry, then
# wellness. The conservative direction is to keep ambiguous records in
# `unknown` rather than misclassify into cannabidiol.

SENSE_PATTERNS: list[tuple[str, str]] = [
    # --- cannabidiol pharmacology / epilepsy / Epidiolex era ---
    ("cannabidiol_epilepsy_pharmacology",
     r"\b(epidiolex|charlotte\W{0,3}s\s+web|dravet|lennox.gastaut|"
     r"drug.resistant\s+epileps|refractory\s+epileps|epileptic\s+encephalopath|"
     r"tuberous\s+sclerosis\s+complex|tsc(?!\s+gene))\b"),

    # --- cannabidiol receptor / endocannabinoid system pharmacology ---
    ("cannabidiol_endocannabinoid_pharmacology",
     r"\b(endocannabinoid|cb1\s+receptor|cb2\s+receptor|cannabinoid\s+receptor|"
     r"cannabidiolic\s+acid|cbda|trpv1|fatty\s+acid\s+amide\s+hydrolase|faah|"
     r"tetrahydrocannabin|delta.?9.?thc|\bthc\b)\b"),

    # --- cannabidiol wellness / consumer (rising post-2018) ---
    ("cannabidiol_consumer_wellness",
     r"\b(hemp.derived\s+cbd|cbd\s+oil|cbd\s+gummies|cbd\s+(consumer|product|"
     r"vape|tincture|edible|topical|cream|gummy|capsule)|wellness\s+product|"
     r"dietary\s+supplement.*cannabidiol|cbd\s+industry|hemp\s+industry|"
     r"farm\s+bill.*cbd|2018\s+farm\s+bill)\b"),

    # --- cannabidiol chemistry / extraction / analytics ---
    ("cannabidiol_chemistry_analytics",
     r"\b(cannabidiol\s+(synthesis|isolation|extraction|purification|"
     r"crystal|polymorph|stability|degradation)|"
     r"hplc.*cannabidiol|cannabidiol.*hplc|gc.ms.*cannabidiol|cannabidiol.*gc.ms|"
     r"cannabis\s+extract|hemp\s+extract|cannabis\s+sativa|"
     r"cannabidiol\s+content|cannabidiol\s+concentration)\b"),

    # --- corticobasal degeneration neurology (third CBD sense, found via
    #     iter-1-style random-PMID spot check on the unknown bucket) ---
    ("corticobasal_degeneration_neurology",
     r"\b(corticobasal\s+(degeneration|syndrome)|"
     r"progressive\s+supranuclear\s+palsy|\bpsp\b|"
     r"4r.tauopath|tauopath.*corticobasal|corticobasal.*tauopath|"
     r"cbd\s+(syndrome|presentation|patient|case)|"
     r"\bcbs\b.*neurodegenerat|neurodegenerat.*\bcbs\b)\b"),

    # --- common bile duct anatomy / biliary surgery / gastroenterology ---
    ("common_bile_duct_anatomy",
     r"\b(common\s+bile\s+duct|biliary\s+(duct|tree|drainage|obstruction|"
     r"stricture|stent|cannulation|sphincter|stone|dilation|leak|injury|"
     r"reconstruction|atresia|epithelium|mucosa|wall)|"
     r"choledochol|choledocho|"
     r"erc\s*p|endoscopic\s+retrograde\s+cholangio|mrcp|"
     r"sphincterotomy|sphincter\s+of\s+oddi|"
     r"cholangiograph|cholangitis|cholangiocarcinoma|"
     r"cbd\s+(stone|stones|exploration|dilation|injury|leak|"
     r"stricture|obstruction|cannulation|drainage|stent|"
     r"reconstruction|repair|anastomosis|jejunostomy|imaging|"
     r"diameter|width|wall|mucosa|epithelium|duct))\b"),

    # --- cannabidiol clinical trial / safety / tolerability (post-Epidiolex) ---
    ("cannabidiol_clinical_trial",
     r"\b(cannabidiol\s+(safety|tolerability|adverse|placebo|"
     r"randomized|randomised|blind|crossover|efficacy|pharmacokinet|"
     r"dose.response|titration|monotherapy|adjunctive)|"
     r"placebo.*cannabidiol|cannabidiol.*placebo|"
     r"cannabidiol\s+(treatment|therapy|administration))\b"),

    # --- legal / regulatory / FDA scheduling on cannabidiol ---
    ("cannabidiol_regulatory_legal",
     r"\b(fda\s+(approval|guidance|warning).*cannabidiol|"
     r"cannabidiol.*fda|controlled\s+substance.*cannabidiol|"
     r"scheduling.*cannabidiol|cannabidiol.*scheduling|"
     r"deschedul.*cannabidiol|cannabidiol.*regulator|"
     r"hemp.*legal|farm\s+bill.*hemp)\b"),
]


_SENSE_RE: list[tuple[str, re.Pattern]] = [
    (label, re.compile(pat, re.IGNORECASE | re.DOTALL))
    for label, pat in SENSE_PATTERNS
]


def classify_text(text: str) -> str:
    """First-match-wins sense classifier. Returns 'unknown' for no match."""
    for label, pat in _SENSE_RE:
        if pat.search(text):
            return label
    return "unknown"


def fetch_cbd_pubmed_abstracts(
    out_parquet: Path,
    *,
    start_year: int = 2000,
    end_year: int = 2024,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Year-by-year fetch of PMIDs matching cannabidiol / CBD / Epidiolex."""
    if out_parquet.exists():
        print(f"[cache] reusing {out_parquet}", file=sys.stderr)
        return pd.read_parquet(out_parquet)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    # Per-term [Title/Abstract] qualifier discipline (suppresses MeSH
    # auto-mapping; see fetch_pubmed_abstracts.py §0c gotchas).
    pmids = esearch_pmids(
        ["cannabidiol", '"CBD"', "Epidiolex"],
        start_year=start_year,
        end_year=end_year,
        api_key=api_key,
    )
    print(f"\ntotal PMIDs (deduped): {len(pmids):,}", file=sys.stderr)
    records = efetch_records(pmids, api_key=api_key)
    df = pd.DataFrame(records)
    if not len(df):
        print("no records", file=sys.stderr)
        return df
    df["text"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")
                  ).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["year"] = df["year"].astype("Int64")
    df = df.dropna(subset=["year"]).reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df.to_parquet(out_parquet, index=False)
    print(f"wrote {len(df):,} records to {out_parquet}", file=sys.stderr)
    return df


def classify_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sense"] = df["text"].map(classify_text)
    per_yr = (df.groupby(["year", "sense"]).size()
              .unstack("sense", fill_value=0).astype(int))
    return per_yr


def summarise(per_yr: pd.DataFrame) -> None:
    print("\n=== CBD / cannabidiol sense decomposition (per-yr -> totals) ===")
    totals = per_yr.sum(axis=0).sort_values(ascending=False)
    print(totals.to_string())
    print(f"\nGrand total records: {int(per_yr.sum().sum()):,}")
    pct = (totals / totals.sum() * 100).round(1)
    print("\n=== Share of corpus by sense ===")
    print(pct.to_string())

    # Decade-level transparency
    per_dec = per_yr.copy()
    per_dec.index = (per_dec.index // 10) * 10
    print("\n=== Per-decade × sense totals ===")
    print(per_dec.groupby(per_dec.index).sum().to_string())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-key", default=None)
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--abstracts", type=Path, default=CACHE)
    p.add_argument("--counts", type=Path, default=OUT_CSV)
    args = p.parse_args(argv)

    df = fetch_cbd_pubmed_abstracts(
        args.abstracts,
        start_year=args.start_year,
        end_year=args.end_year,
        api_key=args.api_key,
    )
    if not len(df):
        return 1
    per_yr = classify_corpus(df)
    per_yr.to_csv(args.counts)
    print(f"\nWrote per-(year, sense) counts to {args.counts}", file=sys.stderr)
    summarise(per_yr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
