import pickle, numpy as np, pandas as pd
from pathlib import Path
import pycorpdiff as pcd
from pycorpdiff.datasets.histwords import HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990 as REF
import scipy.stats as ss

D = Path.home()/".cache/pycorpdiff/histwords/coha"
def load(dec):
    vocab = pickle.load(open(D/f"{dec}-vocab.pkl","rb"))
    W = np.load(D/f"{dec}-w.npy")
    return {w: W[i] for i,w in enumerate(vocab) if np.any(W[i])}  # drop all-zero (OOV) rows

eras = {d: load(d) for d in (1900,1950,1990)}
for d,m in eras.items(): print(f"{d}: {len(m):,} non-zero words")

common = set(eras[1900]) & set(eras[1950]) & set(eras[1990])
focus = [w for w in REF if w in common]
sample = focus + [w for w in sorted(common) if w not in focus][:2000]
res = pcd.semantic_flow(eras, words=sample, knn=15)
t = res.table.set_index("word")

rows=[(w,REF[w],round(float(t.loc[w,'speed']),3),round(float(t.loc[w,'displacement']),3),
       round(float(t.loc[w,'divergence']),4)) for w in sorted(REF,key=lambda k:-REF[k]) if w in t.index]
vt=pd.DataFrame(rows,columns=['word','published_cos','speed','displacement','divergence'])
print("\n=== KNOWN SHIFTERS (published high) + stable (published ~0.10) ===")
print(vt.to_string(index=False))
print(f"\nSpearman(published_cos, displacement) = {ss.spearmanr(vt.published_cos,vt.displacement).statistic:.3f}")
print(f"Spearman(published_cos, speed)        = {ss.spearmanr(vt.published_cos,vt.speed).statistic:.3f}")
shifters=[w for w in REF if REF[w]>=0.4 and w in t.index]
stable=[w for w in REF if REF[w]<=0.1 and w in t.index]
print(f"\nmean displacement -- shifters {t.loc[shifters,'displacement'].mean():.3f}  vs  stable {t.loc[stable,'displacement'].mean():.3f}")
print(f"shifter speed rank (percentile in sample): "+
      ", ".join(f"{w}={100*(t['speed']<t.loc[w,'speed']).mean():.0f}%" for w in shifters))
print("\n=== fastest 12 in sample ===")
print(res.table.head(12)[['word','speed','displacement','divergence']].to_string(index=False))
print("\nbroadening(top div):", res.table.sort_values('divergence',ascending=False).head(8)['word'].tolist())
print("narrowing(bot div) :", res.table.sort_values('divergence').head(8)['word'].tolist())
