# Pre-registration: eponym-retirement terminology shifts in PubMed

**This document is committed BEFORE any data retrieval or inspection.**
The commit hash and timestamp of this file in the public repository
constitute the pre-registration. At the time of this commit, the
author has not retrieved, viewed, or computed any PubMed counts,
trajectories, or records for the terms below. The analysis querying
these terms will be added in a later, separately-timestamped commit.

Associated study: the narrative-audit methods paper ("Auditing the
narrative…"). This pre-registration supplies the one genuinely *prior*,
git-verifiable predicted shift for that paper; the paper's other
demonstrations are retrospective and are reported transparently as
pre-specified-within-workflow rather than prior-committed.

## Background (from documented nomenclature changes only)

In 2011, a joint recommendation (American College of Rheumatology,
American Society of Nephrology, and the European League Against
Rheumatism) proposed replacing the eponym *Wegener's granulomatosis*
with the descriptive term *granulomatosis with polyangiitis*. The 2012
Revised International Chapel Hill Consensus Conference Nomenclature of
Vasculitides formalised this, together with the parallel replacement of
*Churg–Strauss syndrome* by *eosinophilic granulomatosis with
polyangiitis*.

These are **eponym-retirement** shifts — a distinct archetype from
diagnostic-reclassification renames (e.g. DSM-5). Medical eponyms are
known to be linguistically sticky, so the direction and especially the
timing of their replacement in the literature are genuinely uncertain.
That uncertainty is what makes these legitimate falsifiable tests
rather than foregone conclusions.

## Query terms (fixed in advance; phrase-anchored)

Consistent with the polysemy safeguard of the narrative-audit pattern,
we deliberately query phrase-anchored terms and **not** the ambiguous
abbreviations *GPA* / *EGPA*, which collide with unrelated senses
(e.g. grade-point average). All terms are title/abstract-qualified.
Window: 1990–2024.

**Shift 1 — GPA**
- Old (eponym): `"Wegener's granulomatosis"`, `"Wegener granulomatosis"`, `"Wegeners granulomatosis"`
- New (descriptive): `"granulomatosis with polyangiitis"`
- Anchor year: **2011**

**Shift 2 — EGPA**
- Old (eponym): `"Churg-Strauss syndrome"`, `"Churg Strauss syndrome"`
- New (descriptive): `"eosinophilic granulomatosis with polyangiitis"`
- Anchor year: **2012**

## Predictions and falsifiers

For each shift, annual PubMed title/abstract record counts are computed
for the old and new term sets across 1990–2024.

**P1 (direction + durable success).** The new descriptive term overtakes
the eponym — annual count(new) > annual count(old) — in at least one
year by 2024, and is at or above the eponym in the final observed year
(2024).
- **Falsifier F1:** the eponym's annual count exceeds the new term's in
  *every* year through 2024 (no crossover), **or** the new term crosses
  over but the eponym regains the lead in 2024.

**P2 (timing).** The crossover year falls within **±7 years** of the
anchor (i.e. by 2018 for GPA, by 2019 for EGPA).
- **Falsifier F2 (for "on-time"):** crossover occurs but later than the
  ±7-year window.

**Scoreboard mapping (fixed in advance):**
- **PASS** — P1 holds *and* crossover within ±7 of anchor.
- **PARTIAL** — P1 holds (durable crossover by 2024) but outside the
  ±7-year window.
- **FAIL** — F1 (no durable crossover by 2024).

## Explicit non-prediction

We do **not** predict that the eponym disappears. Clinical, historical,
and review literature is expected to retain it. The test concerns
*dominance crossover in current annual output*, not extinction.

## Attestation

At the time of this commit the author has not retrieved, viewed, or
computed any counts for the terms above. Any deviation from the terms,
anchors, windows, or thresholds specified here will be disclosed
explicitly and prominently in the final report.
