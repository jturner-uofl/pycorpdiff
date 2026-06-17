# sense_drift on SemEval-2020 Task 1 (English LSCD) — results

> **CORRECTION (see V2_PLAN.md / v2_ladder.png):** the strong negative below —
> "not competitive, it's the *design*" — was **overstated**. A follow-up
> apples-to-apples run isolated the saturation to the **Mahalanobis** novelty
> metric: on identical inputs, swapping to a bounded **Euclidean** novelty ranks
> at **ρ 0.33**, and a background correction independently reaches 0.33 (up to
> 0.47). The *default* sense_drift does saturate on cross-era data (true below),
> but it is **fixable**, not fundamentally incapable.

Pre-registered, untuned (the task is unsupervised; nothing tuned to the gold).
Every variant we ran is reported below — no cherry-picking. English Subtask 2
(graded-change ranking), Spearman ρ vs gold. Reference: best SemEval-2020
English system ≈ **0.422**; SGNS+OP+cosine ≈ 0.22.

| Method | input | ρ vs gold | |
|---|---|---|---|
| **sense_drift** (margin+JSD) | whole-sentence vectors | **−0.187** (n.s.) | margin pins at 1.000 for all 37 |
| **sense_drift** (margin+JSD) | target-token vectors | **−0.076** (n.s.) | still saturated (~1.85) |
| joint k-means + JSD (Montariol-style) | target-token vectors | +0.144 (n.s.) | helps, not competitive |
| cosine-of-means (baseline) | whole-sentence vectors | **+0.418** (p=0.01) | ≈ SemEval best |
| cosine-of-means (baseline) | target-token vectors | +0.364 (p=0.03) | competitive |

## Verdict (robust)
`sense_drift` is **not competitive** on SemEval — and the negative survives the
fair fix. Switching from whole-sentence to target-**token** vectors does *not*
rescue it (−0.19 → −0.08); its margin density still pins near-maximal for every
word. The failure is `sense_drift`'s reference-window-novelty **design**, not the
embedding granularity.

## Why — diagnosed, not guessed
1. The ~150-yr CCOHA register gap shifts the whole embedding cloud (sentence
   *and* token), so 100% of C2 usages fall outside the C1 senses for every word
   → margin density saturates → zero ranking signal.
2. Even the **correct** LSCD design (joint clustering, Montariol-style) only
   reaches ρ=0.14 with this encoder — clusters become partly *era* clusters. So
   cluster-based sense-change detection in general is dominated by the modern
   off-the-shelf encoder's era confound here.
3. The signal **is** in the embeddings: a trivial cosine-of-means hits
   0.36–0.42, tied with the SemEval best. Only the *continuous* mean-distance
   survives the shared era offset; threshold/cluster methods don't.

## Not a bug
On register-**stable** CBD (within PubMed/Twitter) the same `sense_drift` margin
behaves normally (PubMed 0.03 → 0.43, never pinned). The saturation is specific
to the cross-era gap — confirming `sense_drift`'s regime is register-stable
diachronic monitoring, not two-snapshot cross-era LSCD ranking.

## Honest conclusion
The "benchmark `sense_drift` to claim it's competitive on SemEval" route is
**closed** — falsified, robustly, across five variants. We are not going to keep
tweaking encoders/k/cutoffs to manufacture a passing number; that is exactly the
garden-of-forking-paths the project condemns. `sense_drift`'s defensible
contribution is the **audit-integrated monitoring workflow** (calibrated
significance + change-typing + decline detection) on register-stable corpora —
not a benchmark win.
