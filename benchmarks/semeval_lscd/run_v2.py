"""sense_drift v2 — M2 (continuous, calibrated statistic) composed with M1
(nuisance correction). Measured on the SemEval English ruler.

Hypothesis (the measure-3-times catch): a continuous two-sample statistic (RBF-
MMD) ALSO saturates on RAW embeddings (era gap separates the clouds), and only
de-saturates when computed in the M1 nuisance-removed space. We report:
  * MMD on raw embeddings            -> expect saturated / weak rho
  * MMD on era-corrected embeddings  -> expect signal
  * M1 mean-shift residual           -> the M1 baseline (~0.46)
  * permutation-calibrated p          -> the 'calibrated' part (sanity, not tuned)
All pre-registered: control-derived era subspace (k=1), median-heuristic
bandwidth, 200-perm null, seed 42, cap 300/period. No tuning to gold.
"""
import os; os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import re, random
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

D = Path(__file__).parent/"data"/"semeval2020_ulscd_eng"
load = lambda p: Path(p).read_text(encoding="utf-8").splitlines()
c1l, c1t = load(D/"corpus1/lemma/ccoha1.txt"), load(D/"corpus1/token/ccoha1.txt")
c2l, c2t = load(D/"corpus2/lemma/ccoha2.txt"), load(D/"corpus2/token/ccoha2.txt")
targets = [t.strip() for t in load(D/"targets.txt") if t.strip()]
graded = {a: float(b) for a, b in (l.split("\t") for l in load(D/"truth/graded.txt"))}
CAP, SEED, NPERM = 300, 42, 200
rng = random.Random(SEED); np.random.seed(SEED)
model = SentenceTransformer("all-MiniLM-L6-v2")
unit = lambda M: M/(np.linalg.norm(M, axis=1, keepdims=True)+1e-9)

def emb(t, lem, tokn):
    idx = [i for i, l in enumerate(lem) if t in l.split()]
    if len(idx) > CAP: idx = sorted(rng.sample(idx, CAP))
    return unit(model.encode([tokn[i] for i in idx], batch_size=128, show_progress_bar=False).astype("float64")) if len(idx) >= 10 else None

# control words -> era subspace
def freq(lemmas):
    c = Counter()
    for l in lemmas:
        for w in set(l.split()):
            if w.isalpha() and len(w) >= 4: c[w] += 1
    return c
f1, f2 = freq(c1l), freq(c2l); tset = set(targets)
controls = rng.sample(sorted(w for w in f1 if w not in tset and f1[w] >= 300 and f2.get(w, 0) >= 300), 60)
print(f"{len(targets)} targets | {len(controls)} controls | cap={CAP} perms={NPERM}")
print("embedding controls ...")
shifts = []
for w in controls:
    E1, E2 = emb(w, c1l, c1t), emb(w, c2l, c2t)
    if E1 is not None and E2 is not None: shifts.append(E2.mean(0) - E1.mean(0))
Sh = np.array(shifts); Sh -= Sh.mean(0)
era = np.linalg.svd(Sh, full_matrices=False)[2][:1]          # top-1 shared era direction
def strip(M): return M - (M @ era.T) @ era                    # project era out of rows

def pdist2(A, B):
    return np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2*(A @ B.T), 0.0)
def gamma(Z):
    s = Z[np.random.choice(len(Z), min(len(Z), 200), replace=False)]
    d = pdist2(s, s); return 1.0/(np.median(d[d > 0])+1e-9)
def rbf_K(Z, g): return np.exp(-g*pdist2(Z, Z))
def mmd2_K(K, lab):
    a, b = lab == 0, lab == 1
    return K[np.ix_(a, a)].mean() + K[np.ix_(b, b)].mean() - 2*K[np.ix_(a, b)].mean()

print("embedding targets + computing statistics ...")
rows = []
for t in targets:
    E1, E2 = emb(t, c1l, c1t), emb(t, c2l, c2t)
    # M1 mean-shift residual
    d = E2.mean(0)-E1.mean(0); r = d - (d @ era.T) @ era; m1 = float(np.linalg.norm(r))
    # M2 raw MMD (expect saturation)
    Zr = np.vstack([E1, E2]); labr = np.array([0]*len(E1)+[1]*len(E2))
    mmd_raw = float(mmd2_K(rbf_K(Zr, gamma(Zr)), labr))
    # M2 era-corrected MMD (M1+M2) + permutation null
    A, B = strip(E1), strip(E2); Zc = np.vstack([A, B]); lab = np.array([0]*len(A)+[1]*len(B))
    Kc = rbf_K(Zc, gamma(Zc)); mmd_corr = float(mmd2_K(Kc, lab))
    ge = sum(mmd2_K(Kc, np.random.permutation(lab)) >= mmd_corr for _ in range(NPERM))
    pval = (ge+1)/(NPERM+1)
    rows.append(dict(target=t, m1=m1, mmd_raw=mmd_raw, mmd_corr=mmd_corr, p=pval, gold=graded[t]))
    print(f"  {t:14} m1={m1:.3f} mmd_raw={mmd_raw:.3f} mmd_corr={mmd_corr:.3f} p={pval:.3f} gold={graded[t]:.3f}")

R = pd.DataFrame(rows); R.to_csv(Path(__file__).parent/"results_v2.csv", index=False)
print("\n=== Spearman rho vs graded gold ===")
for c, name in [("mmd_raw", "MMD, raw embeddings (expect saturation)"),
                ("mmd_corr", "MMD, era-corrected (M1+M2)"),
                ("m1", "M1 mean-shift residual (baseline)")]:
    rho = spearmanr(R[c], R.gold)[0]
    print(f"  {name:42} rho = {rho:+.3f}")
combo = spearmanr(R.mmd_corr.rank()+R.m1.rank(), R.gold)[0]
print(f"  {'M1 + M2 (rank-sum)':42} rho = {combo:+.3f}")
print(f"\n  raw-MMD variance (low => saturated): {R.mmd_raw.std():.4f} | "
      f"corr-MMD variance: {R.mmd_corr.std():.4f}")
print(f"  calibration: corrected-MMD permutation p in [{R.p.min():.3f}, {R.p.max():.3f}], "
      f"median {R.p.median():.3f}")
