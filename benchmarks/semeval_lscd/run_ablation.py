"""DECISIVE ABLATION (N-1a): does sense_drift's margin-density ADD over the
standard JSD-clustering baseline on SemEval-2020 Task 1 (English, Subtask-2
graded ranking)? All arms on IDENTICAL MiniLM target-token inputs (cap 300,
seed 42), Spearman rho vs gold. Pre-registered; every arm reported.

Arms:
  cosine-of-means     : continuous global-shift baseline (the trivial strong one)
  JSD-clustering      : Montariol/Giulianelli-style (joint k-means + JSD)  <- the FOIL
  margin-Mahalanobis  : pcd.sense_drift default (expected to saturate cross-era)
  margin-bounded      : bounded cosine/Euclidean novelty (the metric fix)
  COMBINED margin+JSD : rank-ensemble of margin-bounded + JSD  <- does margin ADD?
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import random
from pathlib import Path
import numpy as np, pandas as pd, pycorpdiff as pcd
from scipy.stats import spearmanr, rankdata
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

D = Path(__file__).parent / "data" / "semeval2020_ulscd_eng"
load = lambda p: Path(p).read_text(encoding="utf-8").splitlines()
c1l, c1t = load(D / "corpus1/lemma/ccoha1.txt"), load(D / "corpus1/token/ccoha1.txt")
c2l, c2t = load(D / "corpus2/lemma/ccoha2.txt"), load(D / "corpus2/token/ccoha2.txt")
targets = [t.strip() for t in load(D / "targets.txt") if t.strip()]
graded = {a: float(b) for a, b in (l.split("\t") for l in load(D / "truth/graded.txt"))}
CAP, SEED, K = 300, 42, 4
rng = random.Random(SEED)
model = SentenceTransformer("all-MiniLM-L6-v2")
unit = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def emb(t, lem, tokn):
    idx = [i for i, l in enumerate(lem) if t in l.split()]
    if len(idx) > CAP:
        idx = sorted(rng.sample(idx, CAP))
    if len(idx) < 10:
        return None
    return unit(model.encode([tokn[i] for i in idx], batch_size=128,
                             show_progress_bar=False).astype("float64"))


def jsd(p, q):
    eps = 1e-12; p = p + eps; q = q + eps; p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


pd2 = lambda A, B: np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * (A @ B.T), 0.0)
rows = []
for t in targets:
    E1, E2 = emb(t, c1l, c1t), emb(t, c2l, c2t)
    if E1 is None or E2 is None:
        continue
    X = np.vstack([E1, E2]); per = [0] * len(E1) + [1] * len(E2)
    # cosine-of-means (continuous baseline)
    com = 1.0 - float(unit(E1.mean(0)[None])[0] @ unit(E2.mean(0)[None])[0])
    # JSD-clustering (joint k-means on C1+C2, period distributions over clusters)
    lab = KMeans(K, random_state=SEED, n_init=10).fit_predict(X)
    p1 = np.bincount(lab[:len(E1)], minlength=K).astype(float)
    p2 = np.bincount(lab[len(E1):], minlength=K).astype(float)
    jsd_cl = jsd(p1, p2)
    # margin-Mahalanobis (pcd.sense_drift default)
    res = pcd.sense_drift(pd.DataFrame({"text": ["x"] * len(X), "period": per}),
                          X.astype("float32"), "period", reference=[0], k=K, random_state=SEED)
    maha = float(res.table[res.table.period == 1].iloc[0].margin_density)
    # margin-bounded (Euclidean-to-centroid on unit vectors == cosine novelty)
    km = KMeans(K, random_state=SEED, n_init=10).fit(E1)
    d1 = pd2(E1, km.cluster_centers_).min(1); d2 = pd2(E2, km.cluster_centers_).min(1)
    bounded = float((d2 > np.percentile(d1, 95)).mean())
    rows.append(dict(target=t, cos_means=com, jsd_cluster=jsd_cl, margin_maha=maha,
                     margin_bounded=bounded, gold=graded[t]))

R = pd.DataFrame(rows)
# COMBINED: rank-ensemble of margin-bounded + JSD (does margin add to JSD?)
R["combined_margin_jsd"] = (rankdata(R.margin_bounded) + rankdata(R.jsd_cluster)) / 2.0
R.to_csv(Path(__file__).parent / "results_ablation.csv", index=False)

print(f"\n=== DECISIVE ABLATION — SemEval-2020 English, n={len(R)} targets, k={K} ===")
print(f"{'arm':40} {'rho':>7} {'p':>7} {'score var':>10}")
arms = [("cos_means", "cosine-of-means (continuous baseline)"),
        ("jsd_cluster", "JSD-clustering (Montariol/Giulianelli) — FOIL"),
        ("margin_maha", "margin-density MAHALANOBIS (pcd default)"),
        ("margin_bounded", "margin-density BOUNDED (cosine/Euclidean)"),
        ("combined_margin_jsd", "COMBINED margin-bounded + JSD")]
res = {}
for col, name in arms:
    rho, p = spearmanr(R[col], R.gold)
    res[name] = (round(float(rho), 3), round(float(p), 3))
    print(f"{name:40} {rho:+7.3f} {p:7.3f} {R[col].std():10.3f}")
import json
json.dump({k: v[0] for k, v in res.items()}, open(Path(__file__).parent / "ablation.json", "w"), indent=2)
jsd_rho = res["JSD-clustering (Montariol/Giulianelli) — FOIL"][0]
mb_rho = res["margin-density BOUNDED (cosine/Euclidean)"][0]
cb_rho = res["COMBINED margin-bounded + JSD"][0]
print(f"\nVERDICT: margin-bounded rho={mb_rho:+.3f} vs JSD foil rho={jsd_rho:+.3f} "
      f"-> margin {'ADDS over' if mb_rho > jsd_rho else 'does NOT beat'} JSD "
      f"({mb_rho - jsd_rho:+.3f}); combined={cb_rho:+.3f}. "
      f"Mahalanobis saturates (var={R.margin_maha.std():.3f}).")
