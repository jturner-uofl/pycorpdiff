"""Consolidated, apples-to-apples: every variant, ONE preprocessing (row-unit
embeddings), cap 300/period, control-derived era subspace (k=1), seed 42. So the
progression threshold -> continuous -> +correction is a fair comparison.
Saves results_v2_final.csv + the rho ladder for the v2 narrative figure."""
import os; os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import re, random, json
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
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
rng = random.Random(SEED); np.random.seed(SEED)
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
print(f"{len(targets)} targets | {len(controls)} controls | cap={CAP}")
Sh = np.array([emb(w, c2l, c2t).mean(0)-emb(w, c1l, c1t).mean(0) for w in controls])
era = np.linalg.svd(Sh-Sh.mean(0), full_matrices=False)[2][:1]
strip = lambda M: M - (M @ era.T) @ era
def pdist2(A, B): return np.maximum((A**2).sum(1)[:, None]+(B**2).sum(1)[None]-2*(A@B.T), 0.0)
def mmd2(A, B):
    g = 1.0/(np.median(pdist2(A, A)[pdist2(A, A) > 0])+1e-9)
    return float(np.exp(-g*pdist2(A, A)).mean()+np.exp(-g*pdist2(B, B)).mean()-2*np.exp(-g*pdist2(A, B)).mean())

rows = []
for t in targets:
    E1, E2 = emb(t, c1l, c1t), emb(t, c2l, c2t)
    km = KMeans(4, random_state=SEED, n_init=10).fit(E1)
    d1 = pdist2(E1, km.cluster_centers_).min(1); d2 = pdist2(E2, km.cluster_centers_).min(1)
    margin = float((d2 > np.percentile(d1, 95)).mean())            # the OLD threshold signal
    d = E2.mean(0)-E1.mean(0); ms_raw = float(np.linalg.norm(d))    # continuous mean shift
    ms_corr = float(np.linalg.norm(d - (d @ era.T) @ era))         # + M1 correction
    mmd_raw = mmd2(E1, E2)                                          # continuous distributional
    mmd_corr = mmd2(strip(E1), strip(E2))                          # + M1 correction
    rows.append(dict(target=t, margin=margin, ms_raw=ms_raw, ms_corr=ms_corr,
                     mmd_raw=mmd_raw, mmd_corr=mmd_corr, gold=graded[t]))
R = pd.DataFrame(rows); R.to_csv(Path(__file__).parent/"results_v2_final.csv", index=False)
ladder = {}
print("\n=== Spearman rho vs gold (all cap=300, row-unit, k=1 era) ===")
for c, name in [("margin", "margin density (OLD threshold)"),
                ("mmd_raw", "MMD continuous (raw)"),
                ("mmd_corr", "MMD + background-correction"),
                ("ms_raw", "mean-shift (raw / cosine)"),
                ("ms_corr", "mean-shift + background-correction (M1)")]:
    rho = spearmanr(R[c], R.gold)[0]; ladder[name] = round(float(rho), 3)
    print(f"  {name:42} rho = {rho:+.3f}")
print(f"  margin variance (low=>saturated): {R.margin.std():.4f}")
json.dump(ladder, open(Path(__file__).parent/"ladder.json", "w"), indent=2)
