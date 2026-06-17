"""Benchmark pycorpdiff.sense_drift on SemEval-2020 Task 1 (English LSCD).

PRE-REGISTERED DESIGN (fixed before computing any correlation with gold;
the task is unsupervised, so nothing is tuned to the answer key):

  * Data    : SemEval-2020 Task 1, English (CCOHA), 37 targets.
  * ST2 (primary)   : rank targets by semantic-change degree -> Spearman rho
                      vs the graded gold.
  * ST1 (secondary) : binary change/no-change -> accuracy vs the binary gold.
  * Encoder : all-MiniLM-L6-v2 (the sense_drift / SBERTEmbedder default; the
              same encoder used in the CBD case study).
  * Usages  : for each target `lemma_pos`, take corpus lines whose LEMMA stream
              contains that token; embed the ALIGNED raw-TOKEN sentence. Cap
              500 usages/period (random, seed 42) when more exist.
  * Method  : pcd.sense_drift(reference=[C1 period], k=4, cutoff_pctile=95,
              random_state=42).  k=4 fixed a priori (the CBD value).
  * Scores reported (ALL of them, no cherry-pick): margin_density(C2),
              JSD(C2), and margin_density+JSD(C2). Primary = the sum.
  * Baseline: cosine distance between the mean C1 and mean C2 sentence vectors
              (shares the era confound -> isolates whether any failure is the
              drift algorithm or the embedding).
  * ST1 rule: permutation p < 0.05 -> "changed".

  A-PRIORI CAVEAT: whole-sentence vectors encode era/register; across CCOHA's
  ~150-year gap this can conflate word-sense change with register change. We
  report whatever obtains and interpret honestly.
"""
import os, sys, json, random
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
from pathlib import Path
import numpy as np, pandas as pd, pycorpdiff as pcd
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
D = ROOT / "data" / "semeval2020_ulscd_eng"
CAP, K, SEED = 500, 4, 42
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = ST2-only (fast)

load = lambda p: Path(p).read_text(encoding="utf-8").splitlines()
c1l, c1t = load(D/"corpus1/lemma/ccoha1.txt"), load(D/"corpus1/token/ccoha1.txt")
c2l, c2t = load(D/"corpus2/lemma/ccoha2.txt"), load(D/"corpus2/token/ccoha2.txt")
targets = [t.strip() for t in load(D/"targets.txt") if t.strip()]
graded = {a: float(b) for a, b in (l.split("\t") for l in load(D/"truth/graded.txt"))}
binary = {a: int(b) for a, b in (l.split("\t") for l in load(D/"truth/binary.txt"))}
print(f"{len(targets)} targets | C1 {len(c1l)} lines | C2 {len(c2l)} lines | k={K} cap={CAP} perms={N_PERM}")

# index each target -> line numbers (whole-token match in the lemma stream)
def lines_for(lem, t):
    return [i for i, l in enumerate(lem) if t in l.split()]

model = SentenceTransformer("all-MiniLM-L6-v2")
rng = random.Random(SEED)

rows = []
for t in targets:
    i1, i2 = lines_for(c1l, t), lines_for(c2l, t)
    s1 = sorted(rng.sample(i1, CAP)) if len(i1) > CAP else i1
    s2 = sorted(rng.sample(i2, CAP)) if len(i2) > CAP else i2
    u1, u2 = [c1t[i] for i in s1], [c2t[i] for i in s2]
    X = model.encode(u1 + u2, batch_size=128, show_progress_bar=False).astype("float32")
    df = pd.DataFrame({"text": u1 + u2, "period": [0]*len(u1) + [1]*len(u2)})
    res = pcd.sense_drift(df, X, "period", reference=[0], k=K, cutoff_pctile=95,
                          n_permutations=N_PERM, random_state=SEED)
    tab = res.table
    c2row = tab[tab.period == 1].iloc[0]
    md, jsd = float(c2row.margin_density), float(c2row.jsd)
    # cosine-of-means baseline (normalized)
    A = X[:len(u1)].mean(0); B = X[len(u1):].mean(0)
    cos_dist = 1 - float(A @ B / (np.linalg.norm(A)*np.linalg.norm(B) + 1e-9))
    pv = getattr(res, "p_value", None)
    rows.append(dict(target=t, n1=len(u1), n2=len(u2), margin=md, jsd=jsd,
                     margin_plus_jsd=md+jsd, cos_baseline=cos_dist,
                     p=float(pv) if pv is not None else float("nan"),
                     change_type=getattr(res, "change_type", None),
                     gold_graded=graded[t], gold_binary=binary[t]))
    print(f"  {t:14} n=({len(u1)},{len(u2)})  margin={md:.3f} jsd={jsd:.3f} "
          f"cos={cos_dist:.3f}  gold={graded[t]:.3f}")

R = pd.DataFrame(rows)
R.to_csv(ROOT/"results_english.csv", index=False)

print("\n=== SUBTASK 2 — Spearman rho vs graded gold (higher = better) ===")
out = {}
for col in ["margin", "jsd", "margin_plus_jsd", "cos_baseline"]:
    rho, p = spearmanr(R[col], R.gold_graded)
    out[col] = rho
    tag = "  <- PRIMARY (sense_drift)" if col == "margin_plus_jsd" else ("  <- baseline" if col=="cos_baseline" else "")
    print(f"  {col:18} rho = {rho:+.3f}  (p={p:.3f}){tag}")

print("\n  reference points (English ST2): best SemEval system ~0.422 | "
      "SGNS+OP+cosine ~0.22 | freq baseline ~ -0.08 to 0.0")

if N_PERM > 0:
    print("\n=== SUBTASK 1 — accuracy vs binary gold (perm p<0.05 = changed) ===")
    pred = (R.p < 0.05).astype(int)
    acc = float((pred == R.gold_binary).mean())
    maj = float(max(R.gold_binary.mean(), 1 - R.gold_binary.mean()))
    print(f"  accuracy = {acc:.3f}   (majority-class baseline = {maj:.3f})")

json.dump(out, open(ROOT/"spearman_english.json", "w"), indent=2)
print(f"\nsaved results_english.csv + spearman_english.json")
