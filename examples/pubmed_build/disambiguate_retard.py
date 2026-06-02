"""Word-sense disambiguation for the morpheme `retard*` in PubMed.

Iter-1 audit of the PubMed case study REFUTED the §6.5.1 INVERSION
claim because the T3_retarded_slur query is polysemous: a 20-PMID
random sample from the 2021 peak returned 0/20 slur uses; all were
legitimate scientific senses (chemistry, materials, biology,
oncology, growth retardation, psychomotor retardation, etc.).

Stage 1 here: regex-bucket every `retard*` occurrence into known
senses. Stage 2 (deferred until we see how big the residual is):
SBERT clustering on the "unknown" remainder.

Output:
  data/retard_abstracts.parquet      title+abstract for each PMID 1990-2024
  data/retard_sense_counts_by_year.csv   per-(year, sense) record counts
  stdout summary table

Public domain (US-gov NLM E-utilities); free to redistribute.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Re-use the proven fetcher
sys.path.insert(0, str(Path(__file__).parent))
from fetch_pubmed_abstracts import (  # noqa: E402
    esearch_pmids,
    efetch_records,
)


# ----------------------- Stage 1: regex buckets -----------------------
#
# Order matters: first match wins. Put narrower patterns first; the
# residual "unknown" bucket is what Stage 2 SBERT would chew on.

# Each tuple: (sense_label, regex_pattern_string)
SENSE_PATTERNS: list[tuple[str, str]] = [
    # --- clinical intellectual-disability compound (pre-Rosa's-Law era) ---
    ("clinical_intellectual_disability",
     r"\b(mental(ly)?\s+retard\w+|mental\s+retardation|severe(ly)?\s+(mentally\s+)?retard\w+)\b"),

    # --- growth/developmental ---
    ("growth_developmental",
     r"\b(growth\s+retard\w+|fetal\s+growth\s+retard\w+|intrauterine\s+growth\s+retard\w+|"
     r"iugr|fgr|placental\s+retard\w+|developmental\s+retard\w+)\b"),

    # --- psychomotor / mood (depressive retardation) ---
    ("psychomotor_psychiatric",
     r"\b(psychomotor\s+retard\w+|motor\s+retardation|depressive\s+retard\w+|"
     r"mental\s+(state|status)\s+retard\w+|cognitive\s+retard\w+)\b"),

    # --- bone / skeletal ---
    ("bone_skeletal",
     r"\b(skeletal\s+retard\w+|bone\s+age\s+retard\w+|maturation\s+retard\w+|"
     r"ossification\s+retard\w+)\b"),

    # --- speech / language ---
    ("speech_language",
     r"\b(speech\s+retard\w+|language\s+retard\w+|reading\s+retard\w+)\b"),

    # --- physics: retarded potential, electron-lattice, retarded Green's fn ---
    ("physics_retarded_potential",
     r"\b(retarded\s+potential|retarded\s+field|retarded\s+green|"
     r"retarded\s+electron|electron-lattice\s+coupling|"
     r"non-retarded\s+regime|born-oppenheimer)\b"),

    # --- chemistry / materials: retarding rates of chemical processes ---
    # Allow up to 2 intervening modifier tokens after "retard*" before
    # the target noun ("the", "bacterial", "the rapid", etc.)
    ("chemistry_materials_process_verb",
     r"\bretard\w*\s+(\w+\s+){0,2}(corrosion|crystallization|crystallisation|"
     r"deposition|hydration|oxidation|reduction|conversion|decomposition|"
     r"polymerization|polymerisation|nucleation|degradation|aging|ageing|"
     r"reaction|kinetics|transition|transformation|formation|hydrolysis|"
     r"sintering|adsorption|leaching|denaturation|swelling|setting)\b"),

    # --- biology / oncology process-verb ---
    ("biology_oncology_process_verb",
     r"\bretard\w*\s+(\w+\s+){0,2}(growth|progression|proliferation|"
     r"development|onset|spread|invasion|metastasis|tumor|tumour|cell|"
     r"carcinoma|cancer|aging|ageing|senescence|differentiation|"
     r"regeneration|fibrosis|atrophy)\b"),

    # --- food science ---
    ("food_science",
     r"\bretard\w*\s+(\w+\s+){0,2}(staling|browning|spoilage|ripening|"
     r"acidity|fermentation|microbial|shelf)\b"),

    # --- environmental / agriculture ---
    ("environmental_agricultural",
     r"\bretard\w*\s+(\w+\s+){0,2}(infiltration|runoff|leaching|emergence|"
     r"germination|flowering|maturity)\b"),

    # --- passive-voice scientific process: "X is (significantly) retarded by Y" ---
    # Catches "crystallization is significantly retarded by the addition" form.
    # Aimed at chemistry / materials / biology process verbs that reverse the
    # active-voice ordering. Subject nouns enumerated to avoid picking up
    # passive-voice clinical statements like "the child was retarded".
    ("scientific_process_passive_voice",
     r"\b(crystallization|crystallisation|deposition|reaction|kinetics|"
     r"growth|nucleation|transition|transformation|degradation|"
     r"oxidation|polymerization|polymerisation|hydration|hydrolysis|"
     r"swelling|setting|sintering|absorption|adsorption|leaching|"
     r"diffusion|solubility|dissolution|conductivity|migration|"
     r"recovery|tumor|tumour|progression|spread|aging|ageing)\s+"
     r"(\w+\s+){0,3}retard\w+\b"),

    # --- slur-explicit framings (deliberately narrow, expecting near-zero) ---
    ("slur_explicit_mention",
     r"\b(the\s+r-word|the\s+slur\s+(ret|retard)|"
     r"called\s+(\w+\s+)?retarded|use\s+of\s+the\s+word\s+retard\w*|"
     r"(stigma\w*|derogat\w+|slur\w*)\s+(of\s+)?retard\w*|"
     r"r\*tard|ret\*\*d)\b"),
]


_SENSE_RE: list[tuple[str, re.Pattern]] = [
    (label, re.compile(pat, re.IGNORECASE))
    for label, pat in SENSE_PATTERNS
]


def classify_text(text: str) -> str:
    """Classify a (title + abstract) text by the *first* sense whose
    regex pattern fires anywhere in the text.

    Returns 'unknown' if no patterns match — these are the Stage-2
    SBERT candidates.

    A note on "first-match wins" ordering: clinical-ID compound
    patterns are first because they're the highest-stakes for the
    audit-refutation question. If a text matches both 'mentally
    retarded' AND 'retard tumor growth', it counts as clinical-ID —
    which is conservative against the slur narrative (it doesn't
    artificially inflate the process-verb bucket at the expense of
    the clinical bucket).
    """
    for label, pat in _SENSE_RE:
        if pat.search(text):
            return label
    return "unknown"


# ----------------------- Stage-0 fetch -----------------------

def fetch_retard_abstracts(
    out_parquet: Path,
    *,
    start_year: int = 1990,
    end_year: int = 2024,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Year-by-year fetch of PMIDs matching `retard*` in title/abstract,
    then efetch the records. Cached at out_parquet.

    Query: `retarded[Title/Abstract] OR "retards"[Title/Abstract] OR
            "retard"[Title/Abstract] OR "retardation"[Title/Abstract]`
    — same per-term qualification discipline the rest of the pipeline
    uses.

    iter-2 audit note: the original query omitted "retardation", which
    caused a ~95% undercount of the clinical-MR compound (PubMed
    `"mental retardation"[TIAB]` returns ~22.4K records, of which
    ~21.3K had no "retarded"/"retards"/"retard" verb/adj form and were
    therefore invisible to the WSI corpus). Adding "retardation" pulls
    those records in. The slur denominator is essentially unchanged
    because the slur form is overwhelmingly the adjective "retarded",
    not the noun "retardation"; including "retardation" therefore
    *strengthens* the audit-resolved verdict at §6.5.1 by enlarging
    the clinical-ID sense count without inflating the slur count.
    """
    if out_parquet.exists():
        print(f"[cache] reusing {out_parquet}", file=sys.stderr)
        return pd.read_parquet(out_parquet)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pmids = esearch_pmids(
        ["retarded", '"retards"', '"retard"', '"retardation"'],
        start_year=start_year,
        end_year=end_year,
        api_key=api_key,
    )
    print(f"\ntotal PMIDs returned by esearch (deduped): {len(pmids):,}",
          file=sys.stderr)
    records = efetch_records(pmids, api_key=api_key)
    df = pd.DataFrame(records)
    if not len(df):
        print("no records — bailing", file=sys.stderr)
        return df
    df["text"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["year"] = df["year"].astype("Int64")
    df = df.dropna(subset=["year"]).reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df.to_parquet(out_parquet, index=False)
    print(f"wrote {len(df):,} records to {out_parquet}", file=sys.stderr)
    return df


# ----------------------- Stage-1 classification -----------------------

def classify_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'sense' column to each record, then per-(year, sense) aggregate."""
    df = df.copy()
    df["sense"] = df["text"].map(classify_text)
    per_yr = (df.groupby(["year", "sense"]).size()
              .unstack("sense", fill_value=0).astype(int))
    return per_yr


def summarise(per_yr: pd.DataFrame) -> None:
    print("\n=== Total per sense (1990-2024) ===")
    totals = per_yr.sum(axis=0).sort_values(ascending=False)
    print(totals.to_string())
    print(f"\nGrand total records: {int(per_yr.sum().sum()):,}")
    pct = (totals / totals.sum() * 100).round(1)
    print("\n=== Share of corpus by sense ===")
    print(pct.to_string())

    print("\n=== Per-decade × sense totals ===")
    per_yr_dec = per_yr.copy()
    per_yr_dec.index = (per_yr_dec.index // 10) * 10
    decade = per_yr_dec.groupby(per_yr_dec.index).sum()
    print(decade.to_string())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-key", default=None)
    p.add_argument("--start-year", type=int, default=1990)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument(
        "--abstracts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "retard_abstracts.parquet",
    )
    p.add_argument(
        "--counts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "retard_sense_counts_by_year.csv",
    )
    args = p.parse_args(argv)

    df = fetch_retard_abstracts(args.abstracts,
                                start_year=args.start_year,
                                end_year=args.end_year,
                                api_key=args.api_key)
    if not len(df):
        return 1
    per_yr = classify_corpus(df)
    per_yr.to_csv(args.counts)
    print(f"\nWrote per-(year, sense) counts to {args.counts}", file=sys.stderr)
    summarise(per_yr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
