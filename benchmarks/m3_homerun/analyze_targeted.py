"""TARGETED CBD-COVID sub-discourse home run.

Subtopic is GIVEN (cannabidiol + COVID). We (1) track its diffusion as a SHARE
of all CBD discourse (gap-robust to the corpus's batch sampling), (2) decompose
it into claim-types via clustering + LLM naming (interpretation only, cited),
(3) flag which claim-types are efficacy/misinfo claims, and (4) measure the
lead-time of the discourse vs the FDA's first warning letters (2020-03-28).

Honest by construction: every number is from vectors/counts; the LLM only NAMES
clusters from cited tweets and never sources a number or a veracity verdict.
"""
import os, json, re, urllib.request
os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path("/Users/jasonturner/Projects/cbd-discourse/data")
HERE = Path(__file__).parent
FDA = datetime(2020, 3, 28)            # first CBD-COVID warning letters (FTC announced Mar 9)

cov = pd.read_parquet(DATA / "cbdcovid_meta.parquet").reset_index(drop=True)
E = np.load(DATA / "cbdcovid_emb.npy")
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
cov["wk"] = cov.date.dt.strftime("%G-W%V")
print(f"cannabidiol-COVID tweets (sense-filtered): {len(cov)}")

# ---- 1) DIFFUSION as share of all CBD discourse (gap-robust) -----------------
win = pd.read_parquet(DATA / "cbd_tweets_2011_2021.parquet", columns=["date"])
win["date"] = pd.to_datetime(win.date, errors="coerce")
win = win.dropna()
win = win[(win.date >= "2019-06-01") & (win.date < "2020-07-01")]
win["wk"] = win.date.dt.strftime("%G-W%V")
den = win.groupby("wk").size().rename("cbd_n")
num = cov.groupby("wk").size().rename("covid_n")
traj = pd.concat([num, den], axis=1).fillna(0)
traj["covid_n"] = traj.covid_n.astype(int); traj["cbd_n"] = traj.cbd_n.astype(int)
traj = traj[traj.cbd_n >= 200].copy()                 # keep well-sampled weeks
traj["share"] = traj.covid_n / traj.cbd_n
def wk_date(w):
    y, k = w.split("-W"); return datetime.strptime(f"{y}-{k}-1", "%G-%V-%u")
traj["date"] = [wk_date(w) for w in traj.index]
traj = traj.sort_values("date")

# onset = first week the share crosses 0.5% (clear take-off), and first detectable (>0)
detect = traj[traj.share > 0]
first_detect = detect.iloc[0] if len(detect) else None
takeoff = traj[traj.share >= 0.005]
first_takeoff = takeoff.iloc[0] if len(takeoff) else None
peak = traj.loc[traj.share.idxmax()]
print("\n=== diffusion: COVID share of CBD discourse ===")
print(traj[["covid_n", "cbd_n", "share"]].assign(share=lambda d: (100*d.share).round(2)).to_string())
for nm, row in [("first detectable", first_detect), ("take-off (>=0.5%)", first_takeoff), ("peak", peak)]:
    if row is not None:
        lead = (FDA - row.date).days
        print(f"  {nm:18s}: {row.name} (~{row.date.date()}) share={100*row.share:.2f}%  "
              f"lead vs FDA = {lead:+d} d ({lead/7:+.1f} wk)")

# ---- 2) CLAIM-TYPE decomposition (cluster -> distinctive terms -> LLM name) ---
K = 7
km = KMeans(K, random_state=42, n_init=10).fit(En)
cov["claim"] = km.labels_
# distinctive terms per cluster via tf-idf on the union, ranked by cluster mean
vec = TfidfVectorizer(max_features=4000, stop_words="english", token_pattern=r"[a-z][a-z]+",
                      ngram_range=(1, 2), min_df=5)
Xt = vec.fit_transform(cov.text.str.lower())
vocab = np.array(vec.get_feature_names_out())
def top_terms(mask, n=10):
    m = np.asarray(Xt[mask.values].mean(0)).ravel()
    return [vocab[i] for i in m.argsort()[::-1][:n]]

def llm(p):
    r = urllib.request.Request("http://localhost:11434/api/generate",
        data=json.dumps({"model": "qwen3.6:35b", "prompt": p, "stream": False,
                         "options": {"temperature": 0}}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())["response"].strip()

EFFICACY = re.compile(r"cure|prevent|treat|kill|fight|protect|boost.*immun|immun.*boost|"
                      r"ward off|defend|antiviral|anti-viral", re.I)
rows = []
print("\n=== claim-types (k=7) ===")
for c in range(K):
    mask = cov.claim == c
    sub = cov[mask]
    cen = En[mask.values].mean(0)
    order = np.argsort(((En[mask.values] - cen) ** 2).sum(1))
    reps = sub.iloc[order[:10]].text.tolist()
    terms = top_terms(mask)
    eff = sub.text.str.contains(EFFICACY).mean()
    ex = "\n".join(f"- {t[:160]}" for t in reps)
    prompt = (f'These tweets are one cluster of CBD-and-COVID discourse from early 2020.\n'
              f'Distinctive terms: {", ".join(terms)}\nExample tweets:\n{ex}\n\n'
              f'Return STRICT JSON only: {{"label":"<4-7 words>",'
              f'"is_efficacy_claim":<bool, true if it claims CBD treats/prevents/cures/'
              f'boosts-immunity-against COVID>,"gist":"<one sentence>"}}')
    try:
        o = json.loads((lambda s: s[s.find("{"):s.rfind("}") + 1])(llm(prompt)))
    except Exception as e:
        o = {"label": f"cluster {c}", "is_efficacy_claim": None, "gist": str(e)[:60]}
    # peak week of this cluster's share
    cw = sub.groupby("wk").size()
    cshare = (cw / den).dropna()
    cshare = cshare[traj.index.intersection(cshare.index)]
    pk = cshare.idxmax() if len(cshare) else None
    o.update(cluster=c, n=int(mask.sum()), pct=round(100*mask.mean(), 1),
             efficacy_term_frac=round(float(eff), 3), peak_week=pk, terms=terms[:8])
    rows.append(o)
    print(f"  [{c}] n={o['n']:5d} ({o['pct']:4.1f}%)  efficacy_terms={eff:.2f}  "
          f"peak={pk}  | {o['label']}  (efficacy_claim={o['is_efficacy_claim']})")
    print(f"        terms: {', '.join(terms[:8])}")

claims = pd.DataFrame(rows).sort_values("n", ascending=False)
eff_n = claims[claims.is_efficacy_claim == True].n.sum()
print(f"\nefficacy/misinfo-type claims: {eff_n} tweets "
      f"({100*eff_n/len(cov):.0f}% of cannabidiol-COVID discourse)")

# ---- 3) save artifacts for the hero figure -----------------------------------
traj_out = traj.reset_index().rename(columns={"index": "wk"})
traj_out[["wk", "date", "covid_n", "cbd_n", "share"]].to_csv(HERE / "diffusion.csv", index=False)
claims.to_json(HERE / "claim_types.json", orient="records", indent=2)
# per-week share for the top-3 efficacy clusters (for a stacked claim view)
top_eff = claims[claims.is_efficacy_claim == True].cluster.tolist()
stack = {}
for c in claims.cluster:
    cw = cov[cov.claim == c].groupby("wk").size()
    stack[c] = (cw / den).reindex(traj.index).fillna(0).values
pd.DataFrame(stack, index=traj.index).to_csv(HERE / "claim_share_by_week.csv")
summary = dict(
    n_cannabidiol_covid=int(len(cov)),
    first_detectable=dict(week=first_detect.name, date=str(first_detect.date.date()),
                          share=round(float(first_detect.share), 4),
                          lead_days=int((FDA - first_detect.date).days)) if first_detect is not None else None,
    take_off=dict(week=first_takeoff.name, date=str(first_takeoff.date.date()),
                  share=round(float(first_takeoff.share), 4),
                  lead_days=int((FDA - first_takeoff.date).days)) if first_takeoff is not None else None,
    peak=dict(week=peak.name, date=str(peak.date.date()), share=round(float(peak.share), 4)),
    fda_date="2020-03-28", fda_first_signal="2020-03-09",
    efficacy_claim_tweets=int(eff_n), efficacy_claim_frac=round(float(eff_n/len(cov)), 3),
    top_efficacy_clusters=[int(c) for c in top_eff],
)
json.dump(summary, open(HERE / "summary_targeted.json", "w"), indent=2, default=str)
print("\nsaved diffusion.csv, claim_types.json, claim_share_by_week.csv, summary_targeted.json")
