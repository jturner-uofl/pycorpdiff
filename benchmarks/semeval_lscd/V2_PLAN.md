# sense_drift → 2026: the plan

**Vision:** a nuisance-corrected, continuously-calibrated, LLM-explained,
streaming change monitor that says not just *"drift, p≈0.024"* but *"above the
corpus background (σ=3.1), the sense of X emerged ~2014 — '[definition]' — driven
by [these sources]; here are five cited usages; control term Y did NOT move."*

**What we're fixing (measured, not guessed):**
- D1 — can't separate nuisance drift (register/era/topic) from sense change.
- D2 — the core statistic (binary novelty threshold → margin density) saturates
  under any global shift.
- O1 — the "explain" half (term lists) is the weakest output and the biggest
  opportunity.

---

## Milestones

### M0 — Bounded novelty metric (the cheapest fix) ✅ VALIDATED
**Correction (measure-3-times caught this):** the SemEval saturation was a
*Mahalanobis* artifact, NOT a fundamental design flaw. On identical inputs the
Mahalanobis margin is constant (saturated, ρ undefined) while a **bounded
Euclidean/cosine** novelty ranks at **ρ 0.33**. Mahalanobis is *designed* to
amplify distribution shift (its MD3 job) — which backfires when the whole cloud
shifts by era. Fix = swap `sense_drift`'s default novelty metric. This revises
our earlier overstated "sense_drift can't do cross-era LSCD."

### M1 — Background / nuisance-drift correction ✅ VALIDATED
Estimate the shared drift direction(s) from **control terms** (assumed stable),
project them out of each target's drift, score the **residual** = "moved beyond
what the vocabulary moved."
- **Result:** SemEval English ST2 ρ **0.26 → 0.33** (cap-300 apples-to-apples;
  up to 0.47 with mean-then-normalize — magnitude is preprocessing-sensitive,
  direction robust; control-derived, **no gold leakage**). Fixes D1.
- **Ship:** add `background=` to `sense_drift`.

### M2 — Continuous, calibrated change statistic ⚠️ PARTIAL
Replace the saturating threshold with continuous two-sample tests (**MMD /
Wasserstein / C2ST**).
- **Result:** continuous MMD does NOT saturate like the threshold (ρ 0.22 → 0.27
  with M1) — confirms the direction — but **RBF-MMD does not beat the mean-shift
  or the metric swap** here, and its permutation p saturates (all words
  significant). Its real value is catching distributional *sense-splits* the
  mean misses, not winning SemEval. Keep as a complementary signal, not the core.

### M3 — LLM sense layer (the leap)
Off-the-shelf encoders bake era/style into the vector; an LLM reads *through* it.
- **LLM-as-WiC-judge** → usage-similarity graph → correlation-cluster (WUG/DWUG
  method) instead of raw-embedding k-means (kills the "era clusters" failure).
- **Definition generation** → each sense named in plain English with citations.
- Fixes D1 at the root **and** delivers O1. Needs an LLM (API key or local).

### M4 — Explanation + attribution
LLM-written *"what changed and why"* grounded in **cited KWIC exemplars**;
decompose drift by metadata (**which sources/authors/regions/subreddits** drove
it); flag "this is just background drift, ignore."

### M5 — Streaming / online
Online clustering + **sequential change detection (ADWIN / online BOCPD)** with
bounded memory; monitor many terms at once under **FDR** control. (Closes the
loop back to MD3's streaming roots.)

### M6 — Register-stable benchmark suite
Stop testing on CCOHA's era gap. Curate/evaluate on **register-stable** diachronic
tasks (news-over-years, scientific literature, social streams) where this class
of method should win — and pre-register the evals.

---

## How it composes (sense_drift v2 API sketch)
```
sense_drift(items, embeddings, time_col,
            reference=...,                 # M-base
            background="controls"|"corpus",# M1  nuisance correction
            statistic="mmd"|"wasserstein", # M2  continuous + calibrated
            sense_backend="llm-wic",       # M3  register-invariant senses
            explain="llm",                 # M4  narrative + attribution
            online=True, fdr=0.05)         # M5  streaming + multiplicity
```

## Status
M1 done and measured. M2 next (cheapest, no new deps). M3 is the headline leap
(needs an LLM). Everything validated on the SemEval ruler + demonstrated on the
register-stable CBD corpora (the real use case), pre-registered each step.
