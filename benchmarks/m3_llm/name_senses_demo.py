"""M3 productized: name_senses() on real CBD-PubMed data via a local LLM.

The hand-rolled urllib version (m3_demo.py) is now a package capability:
    result = pcd.sense_drift(...)
    named  = result.name_senses(pcd.OllamaAnnotator(model="qwen3.6:35b"))
The LLM only NAMES the package's own cited exemplars; every number stays in the
SenseDriftResult, untouched.
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
from pathlib import Path

import numpy as np
import pandas as pd

import pycorpdiff as pcd

DATA = Path("/Users/jasonturner/Projects/cbd-discourse/data")
df = pd.read_parquet(DATA / "cbd_pubmed_abstracts.parquet")
X = np.load(DATA / "cbd_pubmed_embeddings.npy")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
keep = df["year"].between(2000, 2024)
df = df[keep].reset_index(drop=True)
X = X[keep.to_numpy()]
print(f"CBD-PubMed: {len(df)} abstracts, {int(df.year.min())}-{int(df.year.max())}")

res = pcd.sense_drift(df, X, "year", reference=list(range(2000, 2010)), k=4,
                      text_col="text", random_state=42)
print(f"\nsense_drift: change_type={res.change_type}  "
      f"drift {int(res.table.drift.sum())}/{len(res.table)} periods")
print("distinctive drift terms:", ", ".join(res.drift_terms[:8]))

# THE NEW CAPABILITY: name the senses with the local LLM (cited, never numeric).
named = res.name_senses(pcd.OllamaAnnotator(model="qwen3.6:35b"), n_examples=10)
print("\n" + "=" * 70)
print(named.summary())
print("=" * 70)
print("\nprovenance:", {k: v for k, v in named.provenance.items() if k != "senses"})

# Prove the invariant on real data: the numeric table is identical to a fresh run.
res2 = pcd.sense_drift(df, X, "year", reference=list(range(2000, 2010)), k=4,
                       text_col="text", random_state=42)
identical = res.table.equals(res2.table)
print(f"\nINVARIANT — sense_drift numeric table unchanged by naming: {identical}")
print("named columns:", list(named.table.columns),
      "| label dtype:", str(named.table.label.dtype))

named.to_json(Path(__file__).parent / "cbd_named_senses.json")
print("saved cbd_named_senses.json")
