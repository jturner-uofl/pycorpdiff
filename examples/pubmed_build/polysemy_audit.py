"""Polysemy-discipline audit on Tier-2/3 single-word labels.

Iter-2 of the PubMed case study surfaced the same construct collision
that iter-1 caught for T3_retarded_slur in THREE more labels
(T3_dwarf_clinical, T3_lunatic, T3_midget). The pattern: a Tier-2/3
inventory label whose query is a single polysemous English word does
NOT measure clinical-deprecated-term usage — it measures whichever
non-clinical sense happens to dominate scientific publishing.

This script extends the iter-1/iter-2 random-20-PMID discipline to
every remaining single-word Tier-2/3 label. For each label we:

  1. Read its peak year from data/pubmed_{tier2,tier3}_counts.csv
  2. Pull up to 20 random PMIDs from that year via NCBI E-utilities
  3. efetch title + abstract
  4. Write to data/polysemy_audit_<label>.csv for human classification

The classification step (intended / different / about-the-term) is
captured by hand in data/polysemy_audit_classifications.csv and the
§6.5 rewrite reads that file.

Public domain (US-gov NLM E-utilities).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_pubmed_abstracts import (  # noqa: E402
    esearch_pmids_one_year,
    efetch_records,
)
from fetch_pubmed import FULL_INVENTORY, TIER2_INVENTORY, TIER3_INVENTORY  # noqa: E402


# Labels to audit this pass — focused on the single-word/morpheme queries
# that have HIGH polysemy risk. Already-audited labels and phrase-only
# queries are explicitly excluded with rationale.
#
# Iter-1 audited: T3_retarded_slur (REFUTED) — repaired to T3_retarded_morpheme
# Iter-2 audited: T3_dwarf_clinical (10%), T3_lunatic (15-25%), T3_freak (zero),
#                 T3_midget (0%), T3_imbecile_clinical (era-clinical 7/8),
#                 T2_spastic_clinical (100%); plus G.1 spot checks on extinct
#                 labels
#
# This pass: 8 high-polysemy single-word labels iter-1/2 did not probe.

POLYSEMY_AUDIT_LABELS: dict[str, tuple[str, str]] = {
    # label                     inventory  rationale
    "T2_moron":              ("TIER2", "single word; might be slur or clinical or informal"),
    "T2_imbecile":           ("TIER2", "single-word + plural; clinical-era vs slur"),
    "T2_idiocy_clinical":    ("TIER2", "single word; broad usage including phrases"),
    "T2_homosexuality":      ("TIER2", "broad topic term; DSM-II diagnosis vs population descriptor"),
    "T2_frigidity":          ("TIER2", "single word; clinical-women vs chemistry/physics"),
    "T2_illegitimate":       ("TIER2", "single word; legal/historical vs medical"),
    "T3_deformed":           ("TIER3", "single word + phrases; clinical vs metaphorical"),
    "T3_hottentot":          ("TIER3", "racial-anthropology vs modern Khoisan studies"),
    "T3_kaffir":             ("TIER3", "racial slur vs modern South Africa scholarship contexts"),
}


def peak_year_for_label(label: str) -> tuple[int, int]:
    """Return (peak_year, count_at_peak) by reading the appropriate CSV."""
    for tier_csv in ("data/pubmed_tier3_counts.csv",
                     "data/pubmed_tier2_counts.csv"):
        path = Path(__file__).resolve().parents[1] / tier_csv
        if not path.exists(): continue
        df = pd.read_csv(path)
        sub = df[df["label"] == label]
        if not len(sub): continue
        peak_row = sub.loc[sub["n_records"].idxmax()]
        return int(peak_row["year"]), int(peak_row["n_records"])
    raise ValueError(f"label {label!r} not found in either CSV")


def fetch_random_pmids_for_label(
    label: str,
    *,
    n_target: int = 20,
    rng_seed: int = 42,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch up to n_target random PMIDs from the peak year for `label`."""
    peak_yr, peak_n = peak_year_for_label(label)
    # Find query terms
    terms: list[str] | None = None
    for inv in (TIER3_INVENTORY, TIER2_INVENTORY, FULL_INVENTORY):
        if label in inv:
            terms = inv[label][0]
            break
    if terms is None:
        raise ValueError(f"no inventory entry for label {label!r}")

    pmids_all = esearch_pmids_one_year(terms, peak_yr, api_key=api_key)
    if len(pmids_all) == 0:
        print(f"  {label}: 0 PMIDs at peak {peak_yr}", file=sys.stderr)
        return pd.DataFrame()
    rng = random.Random(rng_seed)
    if len(pmids_all) > n_target:
        sample = rng.sample(pmids_all, n_target)
    else:
        sample = pmids_all
    print(f"  {label} peak {peak_yr} (n={peak_n}): sampled {len(sample)} of {len(pmids_all)} PMIDs",
          file=sys.stderr)
    records = efetch_records(sample, api_key=api_key)
    df = pd.DataFrame(records)
    if not len(df):
        return df
    df["label"] = label
    df["peak_year"] = peak_yr
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-key", default=None)
    p.add_argument("--n-target", type=int, default=20)
    p.add_argument("--out",
                   type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "polysemy_audit_samples.csv")
    p.add_argument("--only", nargs="*", default=None,
                   help="Restrict to subset of labels.")
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for label in POLYSEMY_AUDIT_LABELS:
        if args.only and label not in args.only:
            continue
        try:
            df = fetch_random_pmids_for_label(label, n_target=args.n_target,
                                              api_key=args.api_key)
        except Exception as e:
            print(f"  {label}: failed: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if len(df):
            # Truncate abstract to first 600 chars for human-classification
            df["abstract_short"] = df["abstract"].fillna("").str.slice(0, 600)
            frames.append(df[["label", "peak_year", "pmid", "year", "title",
                               "abstract_short", "journal"]])

    if not frames:
        print("no records fetched", file=sys.stderr)
        return 1
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(args.out, index=False)
    print(f"\nWrote {len(full)} sampled records across {full['label'].nunique()} "
          f"labels to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
