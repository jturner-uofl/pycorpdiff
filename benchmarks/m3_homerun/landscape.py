"""Movement B: the CBD medical-claim LANDSCAPE.

COVID is one sub-discourse among many. Here we map the major CBD medical-claim
conditions across the full 3.46M-tweet corpus: for each condition we MEASURE the
efficacy-claim volume + trajectory, and we TAG it with a veracity STATUS taken
from the external regulatory/scientific record (human-anchored, cited) -- never
asserted by the model. The epilepsy row is the signature reversal: it was widely
called unproven before 2018, then FDA-approved (Epidiolex).
"""
import re, json
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path("/Users/jasonturner/Projects/cbd-discourse/data")
HERE = Path(__file__).parent

# condition -> (regex, veracity status, note). Status is from the external record.
CONDITIONS = {
    "Epilepsy / seizures": (r"epilep|seizure|dravet|lennox|convuls",
        "FDA-approved", "Epidiolex approved 2018 (Dravet, Lennox-Gastaut) — was called unproven pre-2018"),
    "Anxiety":            (r"anxiet|anxious|panic attack", "Emerging evidence",
        "preclinical + small trials; not an approved indication"),
    "Chronic pain":       (r"chronic pain|\bpain relief|arthrit|\bpain\b|migraine",
        "Emerging evidence", "mixed trial evidence; not approved"),
    "Sleep / insomnia":   (r"insomnia|can'?t sleep|sleep better|sleepless|\bsleep aid",
        "Emerging evidence", "limited trial evidence"),
    "Inflammation":       (r"inflammat|anti-inflammat", "Emerging evidence",
        "preclinical; not an approved indication"),
    "Depression":         (r"depress", "Limited evidence", "preclinical signals; no approval"),
    "PTSD":               (r"ptsd|post-traumatic", "Under study", "early trials"),
    "Addiction / opioid": (r"addict|opioid|withdrawal|quit smoking|substance use",
        "Under study", "early trials for craving/withdrawal"),
    "COVID-19":           (r"covid|coronavirus|sars-cov|\bcorona\b|pandemic",
        "FDA-warned (unproven)", "FDA/FTC warning letters Mar 2020 — no evidence CBD treats COVID"),
    "Cancer":             (r"cancer|tumou?r|carcinoma|chemo|oncolog",
        "FDA-warned (unproven)", "numerous FDA warning letters 2015-2019 re: 'CBD cures cancer'"),
    "Alzheimer's / dementia": (r"alzheimer|dementia", "Unproven", "no clinical support for claims"),
    "Autism":             (r"\bautis", "Unproven / under study", "anecdotal; trials ongoing"),
    "Diabetes":           (r"diabet", "Unproven", "no clinical support for cure claims"),
}
HARD = r"cure|cures|cured|treat|treats|treated|prevent|prevents|heal|heals|kill|kills|fight|fights|eradicat"
SOFT = r"help|helps|reliev|relief|reduce|reduces|ease|eases|manage|soothe|calm|boost|protect|remed"
hard_re = re.compile(HARD, re.I); soft_re = re.compile(SOFT, re.I)

df = pd.read_parquet(DATA / "cbd_tweets_2011_2021.parquet", columns=["date", "text"])
df["date"] = pd.to_datetime(df.date, errors="coerce")
df = df.dropna(subset=["date", "text"]); df["text"] = df.text.astype(str)
df["yr"] = df.date.dt.year
txt = df.text.str.lower()
has_hard = txt.str.contains(hard_re); has_soft = txt.str.contains(soft_re)
print(f"corpus: {len(df):,} CBD tweets, {df.yr.min()}-{df.yr.max()}\n")

rows = []
for cond, (rgx, status, note) in CONDITIONS.items():
    m = txt.str.contains(rgx, regex=True)
    n = int(m.sum())
    eff_hard = int((m & has_hard).sum())
    eff_any = int((m & (has_hard | has_soft)).sum())
    sub = df[m]
    by_yr = sub.groupby("yr").size()
    peak_yr = int(by_yr.idxmax()) if len(by_yr) else None
    rows.append(dict(condition=cond, status=status, n_mentions=n,
                     eff_hard=eff_hard, eff_any=eff_any,
                     eff_hard_frac=round(eff_hard / max(n, 1), 3),
                     peak_year=peak_yr, note=note))
    print(f"  {cond:24s} | {status:22s} | mentions={n:7,} | hard-claim={eff_hard:6,} "
          f"({100*eff_hard/max(n,1):.1f}%) | peak {peak_yr}")

land = pd.DataFrame(rows).sort_values("eff_hard", ascending=False)
land.to_csv(HERE / "landscape.csv", index=False)
json.dump(rows, open(HERE / "landscape.json", "w"), indent=2)
print(f"\nsaved landscape.csv ({len(land)} conditions)")
print(f"total hard efficacy-claim tweets across conditions: {land.eff_hard.sum():,}")
