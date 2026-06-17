"""The decisive M1 test: does background correction help on the register-stable
CBD core? Run sense_drift on CBD with vs without background=, compare. Honest
outcome either way (the docstring predicts a near no-op here)."""
import os; os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import numpy as np, pandas as pd, pycorpdiff as pcd
from pathlib import Path

DATA = Path("/Users/jasonturner/Projects/cbd-discourse/data")
df = pd.read_parquet(DATA / "cbd_pubmed_abstracts.parquet")
X = np.load(DATA / "cbd_pubmed_embeddings.npy")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
keep = df["year"].between(2000, 2024)
df = df[keep].reset_index(drop=True); X = X[keep.to_numpy()]
BG = np.load(DATA / "pubmed_background_embeddings.npy")
bper = np.load(DATA / "pubmed_background_periods.npy")
print(f"CBD {len(df)} records | background {BG.shape[0]} records over "
      f"{bper.min()}-{bper.max()}\n")
ref = list(range(2000, 2010))
for label, kw in [("WITHOUT background (current default)", {}),
                  ("WITH background correction (M1)",
                   dict(background_embeddings=BG, background_time=bper))]:
    r = pcd.sense_drift(df, X, "year", reference=ref, k=4, n_permutations=25,
                        text_col="text", random_state=42, **kw)
    t = r.table
    print(f"--- {label} ---")
    print(f"  change_type={r.change_type}  p={r.p_value:.3f}")
    print(f"  margin {t.margin_density.iloc[0]:.3f} -> {t.margin_density.iloc[-1]:.3f}"
          f"   jsd {t.jsd.iloc[0]:.3f} -> {t.jsd.iloc[-1]:.3f}"
          f"   drift {int(t.drift.sum())}/{len(t)}")
    print(f"  terms: {', '.join(list(getattr(r, 'drift_terms', []) or [])[:8])}\n")
print("Read: if WITH ~= WITHOUT -> M1 is a no-op on the core (drop it / "
      "cross-register-only). If WITH sharpens or shifts the terms -> M1 adds value.")
