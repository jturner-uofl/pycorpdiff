"""Decisive checks on the scoped-MD3 result: is the 0.73 margin REAL internal
drift or thin-reference inflation? (1) label-shuffle permutation null + p-value;
(2) reference-window sensitivity (widen the early window -> does the late-week
margin collapse?). Honest either way."""
import os
os.environ.update(HF_HUB_OFFLINE="1")
from pathlib import Path
import numpy as np, pandas as pd, pycorpdiff as pcd
from sklearn.decomposition import PCA

DATA = Path("/Users/jasonturner/Projects/cbd-discourse/data")
meta0 = pd.read_parquet(DATA / "cbdcovid_meta.parquet").reset_index(drop=True)
E0 = np.load(DATA / "cbdcovid_emb.npy")
meta0["date"] = pd.to_datetime(meta0["date"])
meta0["wk"] = meta0["date"].dt.strftime("%G-W%V")
Eall = PCA(n_components=50, random_state=42).fit_transform(E0)


def run(cut, n_perm, k=4):
    period = np.where(meta0["date"] < pd.Timestamp(cut), "early", meta0["wk"])
    m = meta0.assign(period=period)
    vc = m["period"].value_counts()
    keepmask = m["period"].isin(vc[vc >= 40].index).to_numpy()
    m2 = m[keepmask].reset_index(drop=True)
    Ep = Eall[keepmask]
    res = pcd.sense_drift(m2, Ep, "period", reference=["early"], k=k,
                          n_permutations=n_perm, text_col="text", random_state=42)
    tab = res.table[res.table.period != "early"].sort_values("period")
    ref_m = res.table.loc[res.table.period == "early", "margin_density"].iloc[0]
    return res, tab, ref_m, int((m2.period == "early").sum())


# (1) permutation-calibrated p-value at the pre-registered cut (March 1)
print("=== (1) permutation null, reference = Jan-Feb (cut 2020-03-01), k=4 ===")
res, tab, ref_m, nref = run("2020-03-01", n_perm=200)
print(f"early(ref) n={nref}  margin={ref_m:.3f}  | null-calibrated threshold={res.threshold:.3f}")
print(f"permutation p_value = {res.p_value}")
print(f"peak weekly margin = {tab.margin_density.max():.3f} at "
      f"{tab.loc[tab.margin_density.idxmax(), 'period']}  | drift weeks={int(tab.drift.sum())}/{len(tab)}")
print(f"first drift week = {tab[tab.drift].period.min() if tab.drift.any() else None}  (FDA = 2020-W13)")

# (2) reference-window sensitivity: widen early window into March
print("\n=== (2) reference-window sensitivity (does late-week margin hold?) ===")
print(f"{'reference cut':>16} | {'ref n':>6} | {'mean late-wk margin':>19} | {'change_type':>12}")
for cut in ["2020-02-25", "2020-03-01", "2020-03-15", "2020-03-25"]:
    res_s, tab_s, ref_s, nref_s = run(cut, n_perm=0)
    late = tab_s[tab_s.period >= "2020-W17"].margin_density.mean()
    print(f"{cut:>16} | {nref_s:>6} | {late:>19.3f} | {res_s.change_type:>12}")
