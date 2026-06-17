"""DIAGNOSTIC: does sense_drift de-saturate with target-TOKEN contextual vectors
(vs whole-sentence vectors)? Locate the target word via alpha-token alignment
between the lemma and token streams, extract its mean-pooled last-hidden-state
from all-MiniLM, run sense_drift on those. Probe on one word first."""
import os; os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import re, random, numpy as np, pandas as pd, torch, pycorpdiff as pcd
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

D = Path(__file__).parent/"data"/"semeval2020_ulscd_eng"
load = lambda p: Path(p).read_text(encoding="utf-8").splitlines()
c1l, c1t = load(D/"corpus1/lemma/ccoha1.txt"), load(D/"corpus1/token/ccoha1.txt")
c2l, c2t = load(D/"corpus2/lemma/ccoha2.txt"), load(D/"corpus2/token/ccoha2.txt")

ALPHA = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
def locate(lem, tokn, target):
    la = [w for w in lem.split() if w[:1].isalpha()]
    ta = ALPHA.findall(tokn)
    if len(la) != len(ta) or target not in la:
        return None
    j = la.index(target); surf = ta[j]; occ = sum(1 for w in ta[:j] if w == surf)
    return surf, occ
def span(sent, surf, occ):
    sp = [m.span() for m in re.finditer(r'\b'+re.escape(surf)+r'\b', sent)] \
         or [m.span() for m in re.finditer(re.escape(surf), sent)]
    return sp[min(occ, len(sp)-1)] if sp else None

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
@torch.no_grad()
def token_vecs(sents, surfs, occs):
    out = []
    for s in range(0, len(sents), 64):
        bs = sents[s:s+64]
        enc = tok(bs, return_offsets_mapping=True, return_tensors="pt", padding=True, truncation=True, max_length=128)
        offs = enc.pop("offset_mapping")
        hs = mdl(**enc).last_hidden_state
        for b, sent in enumerate(bs):
            sp = span(sent, surfs[s+b], occs[s+b])
            if sp is None: out.append(None); continue
            idx = [i for i, (a, e) in enumerate(offs[b].tolist()) if a < sp[1] and e > sp[0] and (a, e) != (0, 0)]
            out.append(hs[b, idx].mean(0).numpy() if idx else None)
    return out

t = "attack_nn"; rng = random.Random(42)
def collect(lem, tokn, cap=500):
    idx = [i for i, l in enumerate(lem) if t in l.split()]
    if len(idx) > cap: idx = sorted(rng.sample(idx, cap))
    sents, surfs, occs, aligned = [], [], [], 0
    for i in idx:
        loc = locate(lem[i], tokn[i], t)
        if loc: aligned += 1; sents.append(tokn[i]); surfs.append(loc[0]); occs.append(loc[1])
    return sents, surfs, occs, aligned, len(idx)

s1, sf1, o1, a1, n1 = collect(c1l, c1t); s2, sf2, o2, a2, n2 = collect(c2l, c2t)
print(f"{t}: alpha-alignment C1 {a1}/{n1} ({100*a1/n1:.0f}%)  C2 {a2}/{n2} ({100*a2/n2:.0f}%)")
v1 = [v for v in token_vecs(s1, sf1, o1) if v is not None]
v2 = [v for v in token_vecs(s2, sf2, o2) if v is not None]
print(f"valid target-token vecs: C1 {len(v1)}  C2 {len(v2)}")
X = np.array(v1+v2, dtype="float32")
df = pd.DataFrame({"text": ["x"]*len(X), "period": [0]*len(v1)+[1]*len(v2)})
res = pcd.sense_drift(df, X, "period", reference=[0], k=4, cutoff_pctile=95, random_state=42)
print("sense_drift on TOKEN vectors — table:\n", res.table)
print(">>> margin de-saturated?  (whole-sentence gave 1.000; <1.0 here = yes)")
