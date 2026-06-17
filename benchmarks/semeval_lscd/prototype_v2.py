"""sense_drift v2 prototype — MOVE #1: background/nuisance-drift correction.

Diagnosis recap: raw novelty saturates because the whole embedding cloud shifts
by era. Fix: estimate the shared drift direction(s) from CONTROL words (frequent
non-targets, assumed ~stable), project it out of each target's drift vector, and
score the RESIDUAL magnitude = "how much this word moved BEYOND what the
vocabulary moved."

Measured on SemEval English ST2 purely as a ruler (de-saturation check), not a
goal. Reports raw vs corrected, and a transparent sweep over #era-dirs removed.
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
CAP, SEED = 300, 42
rng = random.Random(SEED)
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- control words: frequent alpha lemmas (len>=4) present in both periods, not targets
def freq(lemmas):
    c = Counter()
    for l in lemmas:
        for w in set(l.split()):
            if w.isalpha() and len(w) >= 4: c[w] += 1
    return c
f1, f2 = freq(c1l), freq(c2l)
tset = set(targets)
cands = [w for w in f1 if w not in tset and f1[w] >= 300 and f2.get(w, 0) >= 300]
controls = rng.sample(sorted(cands), 60)
print(f"{len(targets)} targets | {len(controls)} control words | cap={CAP}/period")

def lines_for(lem, t): return [i for i, l in enumerate(lem) if t in l.split()]
def drift_vec(t):
    i1, i2 = lines_for(c1l, t), lines_for(c2l, t)
    s1 = sorted(rng.sample(i1, CAP)) if len(i1) > CAP else i1
    s2 = sorted(rng.sample(i2, CAP)) if len(i2) > CAP else i2
    if len(s1) < 10 or len(s2) < 10: return None
    E1 = model.encode([c1t[i] for i in s1], batch_size=128, show_progress_bar=False)
    E2 = model.encode([c2t[i] for i in s2], batch_size=128, show_progress_bar=False)
    n = lambda M: M.mean(0) / (np.linalg.norm(M.mean(0)) + 1e-9)
    return n(E2) - n(E1)                      # drift in unit-mean space

print("embedding controls + targets ...")
Dc = np.array([v for v in (drift_vec(w) for w in controls) if v is not None], dtype="float64")
tvec = {t: drift_vec(t) for t in targets}
T = np.array([tvec[t] for t in targets], dtype="float64")
gold = np.array([graded[t] for t in targets])

# era subspace = top PCs of the CONTROL drift vectors (independent of targets -> no leakage)
Dc_c = Dc - Dc.mean(0)
U, S, Vt = np.linalg.svd(Dc_c, full_matrices=False)
era_dirs = Vt                                 # rows = principal era directions

def resid_scores(T, k):
    P = era_dirs[:k]                          # remove top-k shared directions
    R = T - (T @ P.T) @ P
    return np.linalg.norm(R, axis=1)

raw = np.linalg.norm(T, axis=1)               # == cosine-of-means ranking (~0.42)
print(f"\nraw drift magnitude (== cosine baseline):  rho = {spearmanr(raw, gold)[0]:+.3f}")
print("background-corrected (remove top-k shared era directions):")
best = (-9, 0)
for k in [1, 2, 3, 5, 8]:
    rho = spearmanr(resid_scores(T, k), gold)[0]
    flag = "  <-- best so far" if rho > best[0] else ""
    if rho > best[0]: best = (rho, k)
    print(f"   k={k:>2}  rho = {rho:+.3f}{flag}")
print(f"\nbest: rho = {best[0]:+.3f} at k={best[1]}  (raw/cosine = "
      f"{spearmanr(raw, gold)[0]:+.3f}; SemEval best system ~0.422)")
