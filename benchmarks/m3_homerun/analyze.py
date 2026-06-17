"""Home-run steps 2-4: unsupervised emergence detection of the CBD-COVID
sub-discourse, its diffusion, LLM characterization, and the FDA lead-time."""
import os, json, urllib.request; os.environ.update(HF_HUB_OFFLINE="1")
from datetime import datetime
import numpy as np, pandas as pd, pycorpdiff as pcd
from pathlib import Path
from sklearn.cluster import KMeans

DATA = Path("/Users/jasonturner/Projects/cbd-discourse/data")
meta = pd.read_parquet(DATA / "covid_corpus_meta.parquet").reset_index(drop=True)
E = np.load(DATA / "covid_corpus_emb.npy")
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
# POLYSEMY SAFEGUARD: keep the cannabidiol sense, drop central-business-district
from sentence_transformers import SentenceTransformer  # noqa: E402
_anc = SentenceTransformer("all-MiniLM-L6-v2").encode(
    ["cannabidiol CBD hemp oil cannabis wellness supplement health remedy tincture",
     "central business district downtown city centre crime lockdown office buildings"]).astype("float32")
_anc = _anc / np.linalg.norm(_anc, axis=1, keepdims=True)
_keep = (En @ _anc.T)[:, 0] > (En @ _anc.T)[:, 1]
print(f"cannabidiol-sense filter: kept {int(_keep.sum())}/{len(meta)} ({100 * _keep.mean():.0f}%)")
meta = meta[_keep].reset_index(drop=True); E = E[_keep]; En = En[_keep]
ref_mask = (meta.period == "2019").to_numpy()
COVID = r"covid|corona|pandemic|coronavirus|virus"
print(f"{len(meta)} cannabidiol records | weeks={meta[meta.period!='2019'].period.nunique()}\n")

# 1) UNSUPERVISED emergence: sense_drift weekly margin (senses fit on 2019)
res = pcd.sense_drift(meta, E, "period", reference=["2019"], k=6, n_permutations=0,
                      text_col="text", random_state=42)
tab = res.table[res.table.period != "2019"].copy().sort_values("period")
print("=== weekly margin (sense_drift, senses fit on 2019) ===")
print(tab[["period", "n", "margin_density", "jsd", "drift"]].to_string(index=False))
onset = tab[tab.drift].period.min() if tab.drift.any() else None
print(f"\nchange_type={res.change_type}  p={res.p_value}  | drift onset week = {onset}")

# 2) what IS the novel cluster? (re-derive novelty vs 2019 senses, cluster, label)
km = KMeans(6, random_state=42, n_init=10).fit(En[ref_mask])
d = ((En[:, None] - km.cluster_centers_[None]) ** 2).sum(2).min(1)
novel = (d > np.percentile(d[ref_mask], 95)) & (~ref_mask)
gidx = np.where(novel)[0]
nc = KMeans(5, random_state=42, n_init=10).fit(En[gidx]).labels_
covfrac = pd.Series(meta.iloc[gidx].text.str.contains(COVID, case=False).values).groupby(nc).mean()
cc = int(covfrac.idxmax())
print(f"\nnovel-cluster COVID fraction:\n{covfrac.round(2).to_string()}")
covid_g = gidx[nc == cc]
print(f"COVID cluster = {cc}: n={len(covid_g)}, covid-frac={covfrac.max():.2f}")
meta["is_covid"] = False; meta.loc[covid_g, "is_covid"] = True
share = meta[meta.period != "2019"].groupby("period").is_covid.mean()

# 3) LLM characterization (cited, interpretation only)
cen = En[covid_g].mean(0)
reps = meta.iloc[covid_g[np.argsort(((En[covid_g] - cen) ** 2).sum(1))[:12]]].text.tolist()
def llm(p):
    r = urllib.request.Request("http://localhost:11434/api/generate",
        data=json.dumps({"model": "qwen3.6:35b", "prompt": p, "stream": False,
                         "options": {"temperature": 0}}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=240).read())["response"].strip()
ex = "\n".join(f"- {t[:170]}" for t in reps)
prompt = (f'These "CBD" tweets emerged as a NEW cluster in early 2020, unlike prior CBD '
          f'discourse:\n\n{ex}\n\nStrict JSON only: {{"theme":"<5-8 words>",'
          f'"specific_claims":["..."],"is_health_claim":<bool>,"why":"<one sentence>"}}')
try:
    o = json.loads((lambda s: s[s.find("{"):s.rfind("}") + 1])(llm(prompt)))
except Exception as e:
    o = {"theme": "(parse error)", "why": str(e)[:80]}
print(f"\n=== LLM characterization (interpretation, from cited tweets) ===\n{json.dumps(o, indent=2)}")

# 4) lead-time vs FDA 2020-03-28
def wk_date(w):
    y, k = w.split("-W"); return datetime.strptime(f"{y}-{k}-1", "%G-%V-%u")
# emergence by COVID-share crossing (cleaner than raw margin): first week share > 2x the Jan baseline
base = share[share.index < "2020-W05"].mean()
em_weeks = share[(share > max(2 * base, 0.02))]
share_onset = em_weeks.index.min() if len(em_weeks) else None
for nm_, ow in [("drift (margin)", onset), ("COVID-share 2x baseline", share_onset)]:
    if ow:
        od = wk_date(ow); lead = (datetime(2020, 3, 28) - od).days
        print(f"\nLEAD-TIME via {nm_}: onset {ow} (~{od.date()}) vs FDA 2020-03-28 = "
              f"{lead} days ({lead/7:.1f} wk)")
print("\n=== weekly COVID-cluster share (diffusion) ===")
print(share.round(3).to_string())
out = pd.DataFrame({"period": share.index, "covid_share": share.values}).merge(
    tab[["period", "margin_density"]], on="period", how="left")
out.to_csv(Path(__file__).parent / "trajectory.csv", index=False)
print("\nsaved trajectory.csv + characterization")
json.dump({"theme": o, "onset": onset, "share_onset": share_onset}, open(Path(__file__).parent/"summary.json", "w"), default=str, indent=2)
