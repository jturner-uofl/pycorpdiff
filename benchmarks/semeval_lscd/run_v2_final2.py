"""Definitive isolation: identical row-unit embeddings (cap 300), compare the
NOVELTY METRIC head-to-head -- pcd.sense_drift's Mahalanobis margin (the default
that saturated) vs a bounded Euclidean-to-centroid margin -- plus the M1
mean-shift correction. Settles whether the SemEval saturation was a fundamental
design flaw or a metric artifact."""
import os; os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import re, random, json
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, pycorpdiff as pcd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

D = Path(__file__).parent/"data"/"semeval2020_ulscd_eng"
load = lambda p: Path(p).read_text(encoding="utf-8").splitlines()
c1l, c1t = load(D/"corpus1/lemma/ccoha1.txt"), load(D/"corpus1/token/ccoha1.txt")
c2l, c2t = load(D/"corpus2/lemma/ccoha2.txt"), load(D/"corpus2/token/ccoha2.txt")
targets = [t.strip() for t in load(D/"targets.txt") if t.strip()]
graded = {a: float(b) for a, b in (l.split("\t") for l in load(D/"truth/graded.txt"))}
CAP, SEED = 300, 42
rng = random.Random(SEED)
model = SentenceTransformer("all-MiniLM-L6-v2")
unit = lambda M: M/(np.linalg.norm(M, axis=1, keepdims=True)+1e-9)
def emb(t, lem, tokn):
    idx = [i for i, l in enumerate(lem) if t in l.split()]
    if len(idx) > CAP: idx = sorted(rng.sample(idx, CAP))
    return unit(model.encode([tokn[i] for i in idx], batch_size=128, show_progress_bar=False).astype("float64")) if len(idx) >= 10 else None
def freq(L):
    c = Counter()
    for l in L:
        for w in set(l.split()):
            if w.isalpha() and len(w) >= 4: c[w] += 1
    return c
f1, f2 = freq(c1l), freq(c2l); tset = set(targets)
controls = rng.sample(sorted(w for w in f1 if w not in tset and f1[w] >= 300 and f2.get(w, 0) >= 300), 60)
Sh = np.array([emb(w, c2l, c2t).mean(0)-emb(w, c1l, c1t).mean(0) for w in controls])
era = np.linalg.svd(Sh-Sh.mean(0), full_matrices=False)[2][:1]
pd2 = lambda A, B: np.maximum((A**2).sum(1)[:, None]+(B**2).sum(1)[None]-2*(A@B.T), 0.0)

rows = []
for t in targets:
    E1, E2 = emb(t, c1l, c1t), emb(t, c2l, c2t)
    X = np.vstack([E1, E2]); per = [0]*len(E1)+[1]*len(E2)
    df = pd.DataFrame({"text": ["x"]*len(X), "period": per})
    # (1) pcd.sense_drift DEFAULT margin (Mahalanobis)
    res = pcd.sense_drift(df, X.astype("float32"), "period", reference=[0], k=4, random_state=SEED)
    maha = float(res.table[res.table.period == 1].iloc[0].margin_density)
    # (2) bounded Euclidean-to-centroid margin (the fix)
    km = KMeans(4, random_state=SEED, n_init=10).fit(E1)
    d1 = pd2(E1, km.cluster_centers_).min(1); d2 = pd2(E2, km.cluster_centers_).min(1)
    eucl = float((d2 > np.percentile(d1, 95)).mean())
    # (3) mean-shift + M1 background correction
    d = E2.mean(0)-E1.mean(0); ms = float(np.linalg.norm(d - (d @ era.T) @ era))
    rows.append(dict(target=t, maha_margin=maha, eucl_margin=eucl, ms_corr=ms, gold=graded[t]))
R = pd.DataFrame(rows); R.to_csv(Path(__file__).parent/"results_metric.csv", index=False)
print("\n=== identical inputs (row-unit, cap 300) — rho vs gold ===")
out = {}
for c, name in [("maha_margin", "margin, MAHALANOBIS novelty (pcd default)"),
                ("eucl_margin", "margin, EUCLIDEAN novelty (the fix)"),
                ("ms_corr", "mean-shift + background-correction (M1)")]:
    rho = spearmanr(R[c], R.gold)[0]; out[name] = round(float(rho), 3)
    print(f"  {name:44} rho = {rho:+.3f}   var={R[c].std():.3f}")
json.dump(out, open(Path(__file__).parent/"metric_ladder.json", "w"), indent=2)
