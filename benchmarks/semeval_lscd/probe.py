import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
from pathlib import Path
import random, numpy as np, pandas as pd, pycorpdiff as pcd
from sentence_transformers import SentenceTransformer

D = Path(__file__).parent/"data"/"semeval2020_ulscd_eng"
load = lambda p: Path(p).read_text(encoding="utf-8").splitlines()
c1l,c1t = load(D/"corpus1/lemma/ccoha1.txt"), load(D/"corpus1/token/ccoha1.txt")
c2l,c2t = load(D/"corpus2/lemma/ccoha2.txt"), load(D/"corpus2/token/ccoha2.txt")

def usages(lem,tok,t,cap=500,seed=42):
    idx=[i for i,l in enumerate(lem) if t in l.split()]
    r=random.Random(seed)
    if len(idx)>cap: idx=sorted(r.sample(idx,cap))
    return [tok[i] for i in idx]

t="attack_nn"
u1,u2 = usages(c1l,c1t,t), usages(c2l,c2t,t)
print(f"{t}: C1={len(u1)} C2={len(u2)} usages")
# confirm a real occurrence + alignment
i=[k for k,l in enumerate(c1l) if t in l.split()][0]
print("  lemma:", c1l[i][:90]); print("  token:", c1t[i][:90])

m=SentenceTransformer("all-MiniLM-L6-v2")
X=m.encode(u1+u2,batch_size=64,show_progress_bar=False).astype("float32")
df=pd.DataFrame({"text":u1+u2,"period":[0]*len(u1)+[1]*len(u2)})
res=pcd.sense_drift(df,X,"period",reference=[0],k=4,n_permutations=50,random_state=42)
print("RESULT type:",type(res).__name__)
print("public attrs:",[a for a in dir(res) if not a.startswith("_")])
print("table:\n",getattr(res,"table",None))
