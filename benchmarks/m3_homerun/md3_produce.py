"""Produce the scoped-MD3 trigger artifacts for the §13c figure/notebook:
margin-density trajectory + permutation null + reference-window sensitivity.
MD3 = the timed TRIGGER (when/that); the targeted+LLM layer = the WHAT (§13a)."""
import os, json
os.environ.update(HF_HUB_OFFLINE="1")
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, pycorpdiff as pcd
from sklearn.decomposition import PCA

DATA = Path("/Users/jasonturner/Projects/cbd-discourse/data")
HERE = Path(__file__).parent
meta0 = pd.read_parquet(DATA / "cbdcovid_meta.parquet").reset_index(drop=True)
E0 = np.load(DATA / "cbdcovid_emb.npy")
meta0["date"] = pd.to_datetime(meta0["date"])
meta0["wk"] = meta0["date"].dt.strftime("%G-W%V")
Eall = PCA(n_components=50, random_state=42).fit_transform(E0)
FDA = datetime(2020, 3, 28)
def wk_date(w):
    y, k = w.split("-W"); return datetime.strptime(f"{y}-{k}-1", "%G-%V-%u")


def run(cut, n_perm, k=4):
    period = np.where(meta0["date"] < pd.Timestamp(cut), "early", meta0["wk"])
    m = meta0.assign(period=period)
    vc = m["period"].value_counts()
    keepmask = m["period"].isin(vc[vc >= 40].index).to_numpy()
    m2 = m[keepmask].reset_index(drop=True)
    res = pcd.sense_drift(m2, Eall[keepmask], "period", reference=["early"], k=k,
                          n_permutations=n_perm, text_col="text", random_state=42)
    ref_m = float(res.table.loc[res.table.period == "early", "margin_density"].iloc[0])
    return res, int((m2.period == "early").sum()), ref_m


# main: permutation-calibrated trajectory at the pre-registered Mar-1 cut
res, nref, ref_m = run("2020-03-01", n_perm=200)
tab = res.table[res.table.period != "early"].sort_values("period").copy()
tab["date"] = [wk_date(w) for w in tab.period]
onset = tab[tab.drift].period.min() if tab.drift.any() else None
tab.to_csv(HERE / "md3_trigger.csv", index=False)

# reference-window sensitivity (the honest caveat)
sens = []
for cut in ["2020-02-25", "2020-03-01", "2020-03-15", "2020-03-25"]:
    rs, nr, _ = run(cut, n_perm=0)
    ts = rs.table[rs.table.period != "early"]
    sens.append({"cut": cut, "ref_n": nr,
                 "late_margin": float(ts[ts.period >= "2020-W17"].margin_density.mean())})
pd.DataFrame(sens).to_csv(HERE / "md3_sensitivity.csv", index=False)

meta_out = {
    "reference": "early COVID-CBD discourse (Jan-Feb 2020)", "ref_n": nref,
    "ref_margin": round(ref_m, 3), "null_threshold": round(float(res.threshold), 3),
    "p_value": round(float(res.p_value), 4), "change_type": res.change_type,
    "drift_terms": list(res.drift_terms[:10]),
    "onset_week": onset, "onset_date": str(wk_date(onset).date()) if onset else None,
    "fda_week": "2020-W13", "fda_date": "2020-03-28",
    "peak_margin": round(float(tab.margin_density.max()), 3),
    "peak_week": tab.loc[tab.margin_density.idxmax(), "period"],
    "caveat": ("MD3 detects the regime shift (when/that); magnitude is inflated by a "
               "thin early reference (sensitivity: late margin 0.84->0.20 once the "
               "reference sees March) and the drift terms are commerce-led, not misinfo. "
               "The misinfo 'what' belongs to the targeted+LLM layer (§13a)."),
}
json.dump(meta_out, open(HERE / "md3_meta.json", "w"), indent=2, default=str)
print(json.dumps(meta_out, indent=2, default=str))
print("\nsaved md3_trigger.csv, md3_sensitivity.csv, md3_meta.json")
