"""C/D VALIDATION (paper-grade): does sense_drift recover KNOWN change types and
the obsolescence/dilution decline split on synthetic data where we control the
ground truth? Multi-seed confusion matrices + honest failure modes + the
k-sensitivity disclosure. This is the load-bearing methods validation now that
the 2-period SemEval benchmark route is closed (it can't test multi-period
monitoring, change-typing, or the decline split).
"""
import numpy as np, pandas as pd
import pycorpdiff as pcd

D, SCALE, NOISE = 20, 5.0, 0.35
REF_CT = list(range(2000, 2010))


def _cent(n, seed): return np.random.default_rng(seed).standard_normal((n, D)) * SCALE


# ---- change-type generators (noise-parameterised) ----
def emergence(seed, nz):
    rng = np.random.default_rng(seed); C = _cent(4, seed); rows, e = [], []
    for y in range(2000, 2021):
        for _ in range(60):
            s = rng.integers(0, 3); e.append(C[s] + rng.standard_normal(D) * nz); rows.append({"year": y})
        if y >= 2012:
            for _ in range((y - 2011) * 5):
                e.append(C[3] + rng.standard_normal(D) * nz); rows.append({"year": y})
    return pd.DataFrame(rows), np.vstack(e)


def broadening(seed, nz):
    rng = np.random.default_rng(seed); C = _cent(3, seed); rows, e = [], []
    for y in range(2000, 2021):
        for _ in range(60):
            s = rng.integers(0, 3); e.append(C[s] + rng.standard_normal(D) * nz); rows.append({"year": y})
        if y >= 2012:
            for _ in range((y - 2011) * 5):
                v = rng.standard_normal(D); e.append(v / np.linalg.norm(v) * SCALE * 2.5); rows.append({"year": y})
    return pd.DataFrame(rows), np.vstack(e)


def frequency_shift(seed, nz):
    rng = np.random.default_rng(seed); C = _cent(3, seed); rows, e = [], []
    for y in range(2000, 2021):
        fc = 0.33 if y < 2012 else 0.85
        for _ in range(120):
            s = 2 if rng.random() < fc else rng.integers(0, 2)
            e.append(C[s] + rng.standard_normal(D) * nz); rows.append({"year": y})
    return pd.DataFrame(rows), np.vstack(e)


def stable(seed, nz):
    rng = np.random.default_rng(seed); C = _cent(3, seed); rows, e = [], []
    for y in range(2000, 2021):
        for _ in range(80):
            s = rng.integers(0, 3); e.append(C[s] + rng.standard_normal(D) * nz); rows.append({"year": y})
    return pd.DataFrame(rows), np.vstack(e)


# ---- decline generators ----
def dilution(seed):
    rng = np.random.default_rng(seed); C = _cent(3, seed); rows, e = [], []
    for y in range(2000, 2021):
        for _ in range(50): e.append(C[0] + rng.standard_normal(D) * NOISE); rows.append({"year": y})
        for _ in range(50): e.append(C[1] + rng.standard_normal(D) * NOISE); rows.append({"year": y})
        if y >= 2010:
            for _ in range((y - 2009) * 20): e.append(C[2] + rng.standard_normal(D) * NOISE); rows.append({"year": y})
    return pd.DataFrame(rows), np.vstack(e)


def obsolescence(seed):
    rng = np.random.default_rng(seed); C = _cent(2, seed); rows, e = [], []
    for y in range(2000, 2021):
        n0 = max(5, 60 - (y - 2000) * 3)
        for _ in range(n0): e.append(C[0] + rng.standard_normal(D) * NOISE); rows.append({"year": y})
        for _ in range(60): e.append(C[1] + rng.standard_normal(D) * NOISE); rows.append({"year": y})
    return pd.DataFrame(rows), np.vstack(e)


M = 25
print("=" * 64)
print("PART C — change-type recovery (k=3, 25 seeds/class, noise=0.35)")
gens = {"emergence": emergence, "broadening": broadening,
        "frequency_shift": frequency_shift, "stable": stable}
conf = {t: {p: 0 for p in ["emergence", "broadening", "frequency_shift", "stable"]} for t in gens}
for true, g in gens.items():
    for s in range(M):
        df, X = g(1000 + s, NOISE)
        r = pcd.sense_drift(df, X, "year", reference=REF_CT, k=3, random_state=42)
        pred = r.change_type if (r.change_type and r.table.drift.any()) else "stable"
        conf[true][pred] += 1
cm = pd.DataFrame(conf).T
print(cm.to_string())
acc = sum(conf[t][t] for t in gens) / (len(gens) * M)
print(f"change-type accuracy: {acc:.2f}  (diagonal / total)")

print("\n" + "=" * 64)
print("PART D — obsolescence vs dilution (the under-precedented novel split)")
for k in (2, 3, 4):
    dconf = {"obsolescence": {"obsolescence": 0, "dilution": 0, "other": 0},
             "dilution": {"obsolescence": 0, "dilution": 0, "other": 0}}
    for true, g in [("obsolescence", obsolescence), ("dilution", dilution)]:
        ref = list(range(2000, 2005))
        for s in range(M):
            df, X = g(2000 + s)
            rep = pcd.sense_drift(df, X, "year", reference=ref, k=k, random_state=42).decline_report()
            decl = rep.sort_values("rel_share_change").iloc[0]  # most-declining sense
            v = decl["verdict"] if decl["verdict"] in ("obsolescence", "dilution") else "other"
            dconf[true][v] += 1
    dm = pd.DataFrame(dconf).T
    dacc = (dconf["obsolescence"]["obsolescence"] + dconf["dilution"]["dilution"]) / (2 * M)
    print(f"\n  k={k}:  accuracy={dacc:.2f}")
    print("   " + dm.to_string().replace("\n", "\n   "))
print("\n(memory flagged D fragments at k>=4 — disclosed, not hidden)")
