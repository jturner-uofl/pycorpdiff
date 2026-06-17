"""M3 prototype: LLM sense layer on the register-stable CBD-in-PubMed case.

Cluster CBD records into senses, have a local LLM (Ollama qwen3.6:35b) NAME and
DEFINE each sense from cited example titles, and surface the emergent one. Turns
sense_drift's term-list ("dravet, clobazam, ...") into interpretable named senses
grounded in evidence -- the genuine improvement on the tool's real job.
"""
import os; os.environ.update(HF_HUB_OFFLINE="1")
import json, urllib.request
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path

DATA = Path("/Users/jasonturner/Projects/cbd-discourse/data")
MODEL = "qwen3.6:35b"
df = pd.read_parquet(DATA/"cbd_pubmed_abstracts.parquet")
X = np.load(DATA/"cbd_pubmed_embeddings.npy")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
keep = df["year"].between(2000, 2024) & df["title"].astype(str).str.len().gt(10)
df = df[keep].reset_index(drop=True); X = X[keep.to_numpy()]
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

K = 6
km = KMeans(K, random_state=42, n_init=10).fit(Xn)
df["sense"] = km.labels_
early = df.year.between(2000, 2009); late = df.year.between(2018, 2024)

def llm(prompt, model=MODEL):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "options": {"temperature": 0}}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=240).read())["response"].strip()

print(f"{len(df)} CBD records | K={K} senses | model={MODEL}\n")
rows = []
for s in range(K):
    m = (df.sense == s).to_numpy()
    es = float((df[early].sense == s).mean()); ls = float((df[late].sense == s).mean())
    idx = np.where(m)[0]
    d = np.linalg.norm(Xn[idx] - km.cluster_centers_[s], axis=1)
    reps = df.iloc[idx[np.argsort(d)[:10]]].title.tolist()
    examples = "\n".join(f"- {t}" for t in reps)
    prompt = (f'These are the 10 most representative titles of biomedical abstracts that all '
              f'mention "CBD" and fall into ONE coherent sense/topic cluster:\n\n{examples}\n\n'
              f'Identify the single sense of "CBD" these share. Reply with ONLY strict JSON:\n'
              f'{{"name": "<3-6 word sense label>", "definition": "<one sentence>", '
              f'"meaning_of_CBD": "<what CBD stands for here>"}}')
    try:
        r = llm(prompt); obj = json.loads(r[r.find("{"):r.rfind("}") + 1])
    except Exception as e:
        obj = {"name": "(parse error)", "definition": str(e)[:60], "meaning_of_CBD": "?"}
    rows.append(dict(sense=s, n=int(m.sum()), early=es, late=ls,
                     growth=ls / (es + 1e-9), **obj))
    print(f"SENSE {s}:  n={int(m.sum()):4d}  share {es:5.1%} -> {ls:5.1%}  (x{ls/(es+1e-9):.1f})")
    print(f"   name: {obj.get('name')}")
    print(f"   def:  {obj.get('definition')}")
    print(f"   CBD = {obj.get('meaning_of_CBD')}\n")

rows.sort(key=lambda r: -r["growth"])
em = rows[0]
print("=" * 60)
print(f"EMERGENT SENSE (largest growth): sense {em['sense']}")
print(f"  \"{em['name']}\"  —  {em['definition']}")
print(f"  grew {em['early']:.1%} -> {em['late']:.1%}  (x{em['growth']:.1f})")
print("\n  OLD sense_drift output:  drivers = dravet, lennox-gastaut, clobazam, ...")
print("  NEW M3 output:           a NAMED, defined sense, grounded in cited abstracts.")
pd.DataFrame(rows).to_csv(Path(__file__).parent / "m3_senses.csv", index=False)
