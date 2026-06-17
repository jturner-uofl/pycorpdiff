"""Token-vector variants on SemEval-2020 Task 1 English. Reports ALL variants
(no cherry-pick): (a) sense_drift on target-token vectors, (b) joint k-means
clustering + JSD (the Montariol-style SemEval contextual method, = what
pcd.induce_senses wraps), (c) cosine-of-token-means baseline. Same
pre-registration as run_semeval.py (k=4, cap 500/period, seed 42)."""
import os; os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import re, random, numpy as np, pandas as pd, torch, pycorpdiff as pcd
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

D = Path(__file__).parent/"data"/"semeval2020_ulscd_eng"
load = lambda p: Path(p).read_text(encoding="utf-8").splitlines()
c1l, c1t = load(D/"corpus1/lemma/ccoha1.txt"), load(D/"corpus1/token/ccoha1.txt")
c2l, c2t = load(D/"corpus2/lemma/ccoha2.txt"), load(D/"corpus2/token/ccoha2.txt")
targets = [t.strip() for t in load(D/"targets.txt") if t.strip()]
graded = {a: float(b) for a, b in (l.split("\t") for l in load(D/"truth/graded.txt"))}
CAP, K, SEED = 500, 4, 42

ALPHA = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
def locate(lem, tokn, t):
    la = [w for w in lem.split() if w[:1].isalpha()]; ta = ALPHA.findall(tokn)
    if len(la) != len(ta) or t not in la: return None
    j = la.index(t); surf = ta[j]; return surf, sum(1 for w in ta[:j] if w == surf)
def span(s, surf, occ):
    sp = [m.span() for m in re.finditer(r'\b'+re.escape(surf)+r'\b', s)] or \
         [m.span() for m in re.finditer(re.escape(surf), s)]
    return sp[min(occ, len(sp)-1)] if sp else None

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
@torch.no_grad()
def token_vecs(sents, surfs, occs):
    out = []
    for s in range(0, len(sents), 64):
        bs = sents[s:s+64]
        enc = tok(bs, return_offsets_mapping=True, return_tensors="pt", padding=True, truncation=True, max_length=128)
        offs = enc.pop("offset_mapping"); hs = mdl(**enc).last_hidden_state
        for b, sent in enumerate(bs):
            sp = span(sent, surfs[s+b], occs[s+b])
            if sp is None: out.append(None); continue
            idx = [i for i, (a, e) in enumerate(offs[b].tolist()) if a < sp[1] and e > sp[0] and (a, e) != (0, 0)]
            out.append(hs[b, idx].mean(0).numpy() if idx else None)
    return out

rng = random.Random(SEED)
def vecs_for(lem, tokn, t):
    idx = [i for i, l in enumerate(lem) if t in l.split()]
    if len(idx) > CAP: idx = sorted(rng.sample(idx, CAP))
    sents, surfs, occs = [], [], []
    for i in idx:
        loc = locate(lem[i], tokn[i], t)
        if loc: sents.append(tokn[i]); surfs.append(loc[0]); occs.append(loc[1])
    return [v for v in token_vecs(sents, surfs, occs) if v is not None]

def jsd(p, q):
    p = p/p.sum(); q = q/q.sum(); m = 0.5*(p+q)
    kl = lambda a, b: np.sum(np.where(a > 0, a*np.log2(a/b), 0.0))
    return 0.5*kl(p, m) + 0.5*kl(q, m)

rows = []
for t in targets:
    v1, v2 = vecs_for(c1l, c1t, t), vecs_for(c2l, c2t, t)
    X = np.array(v1+v2, dtype="float32"); per = np.array([0]*len(v1)+[1]*len(v2))
    df = pd.DataFrame({"text": ["x"]*len(X), "period": per})
    # (a) sense_drift
    res = pcd.sense_drift(df, X, "period", reference=[0], k=K, cutoff_pctile=95, random_state=SEED)
    c2 = res.table[res.table.period == 1].iloc[0]; sd = float(c2.margin_density) + float(c2.jsd)
    # (b) joint k-means + JSD (Montariol-style)
    Xn = X/(np.linalg.norm(X, axis=1, keepdims=True)+1e-9)
    lab = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(Xn)
    jk = jsd(np.bincount(lab[per == 0], minlength=K)+1e-9, np.bincount(lab[per == 1], minlength=K)+1e-9)
    # (c) cosine of token-means
    A, B = X[:len(v1)].mean(0), X[len(v1):].mean(0)
    cos = 1 - float(A @ B/(np.linalg.norm(A)*np.linalg.norm(B)+1e-9))
    rows.append(dict(target=t, n1=len(v1), n2=len(v2), sense_drift=sd, joint_kmeans_jsd=jk,
                     cos_token=cos, gold=graded[t]))
    print(f"  {t:14} n=({len(v1)},{len(v2)}) sd={sd:.3f} jointJSD={jk:.3f} cos={cos:.3f} gold={graded[t]:.3f}")

R = pd.DataFrame(rows); R.to_csv(Path(__file__).parent/"results_token.csv", index=False)
print("\n=== Spearman rho vs graded gold (English ST2) ===")
for col, name in [("sense_drift", "sense_drift on token vectors"),
                  ("joint_kmeans_jsd", "joint k-means + JSD (Montariol-style)"),
                  ("cos_token", "cosine-of-token-means baseline")]:
    rho, p = spearmanr(R[col], R.gold)
    print(f"  {name:38} rho = {rho:+.3f}  (p={p:.3f})")
print("  reference: best SemEval-2020 English ~0.422 | SGNS+OP+cosine ~0.22")
