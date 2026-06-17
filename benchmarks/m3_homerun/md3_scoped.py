"""MD3 at the right scope: does margin density detect the misinfo sub-discourse
EVOLVING internally? Fit claim-senses on the EARLY (Jan-Feb 2020) cannabidiol-
COVID discourse, then watch the margin density (Sethi & Kantardzic 2017) of each
later week -- a rise = NEW claim-types emerging within the misinfo, not just more
of the same. Honest test: could detect real internal drift, or show the claim
space was fully formed at onset (flat margin)."""
import os
os.environ.update(HF_HUB_OFFLINE="1")
from pathlib import Path
import numpy as np, pandas as pd, pycorpdiff as pcd
from sklearn.decomposition import PCA

DATA = Path("/Users/jasonturner/Projects/cbd-discourse/data")
meta = pd.read_parquet(DATA / "cbdcovid_meta.parquet").reset_index(drop=True)
E = np.load(DATA / "cbdcovid_emb.npy")
meta["date"] = pd.to_datetime(meta["date"])
meta["wk"] = meta["date"].dt.strftime("%G-W%V")

# scope: reference = early COVID-CBD discourse (before March), monitor = weekly
CUT = pd.Timestamp("2020-03-01")
meta["period"] = np.where(meta["date"] < CUT, "early", meta["wk"])
vc = meta["period"].value_counts()
keep = meta["period"].isin(vc[vc >= 40].index)
meta = meta[keep].reset_index(drop=True)
E = E[keep.to_numpy()]
print(f"cannabidiol-COVID subcorpus: {len(meta)} tweets | early(ref) n="
      f"{int((meta.period=='early').sum())} | monitor weeks="
      f"{meta[meta.period!='early'].period.nunique()}")

# PCA -> 50d so the Mahalanobis covariance is well-conditioned (384d from ~280
# reference tweets is rank-deficient -> the saturation failure mode).
Ep = PCA(n_components=50, random_state=42).fit_transform(E)

for k in (4, 5):
    res = pcd.sense_drift(meta, Ep, "period", reference=["early"], k=k,
                          n_permutations=0, text_col="text", random_state=42)
    tab = res.table[res.table.period != "early"].copy()
    tab = tab.sort_values("period")
    print(f"\n===== k={k} =====")
    print(res.summary())
    print(tab[["period", "n", "margin_density", "jsd", "drift"]].to_string(index=False))
    print("change_type:", res.change_type, "| drift_terms:", ", ".join(res.drift_terms[:10]))
    # timing vs FDA (2020-W13)
    early_ref = res.table.loc[res.table.period == "early", "margin_density"].iloc[0]
    rises = tab[tab.margin_density > early_ref * 1.5]
    print(f"early(ref) margin={early_ref:.3f} | weeks >1.5x ref: "
          f"{list(rises.period)} | FDA week = 2020-W13")
