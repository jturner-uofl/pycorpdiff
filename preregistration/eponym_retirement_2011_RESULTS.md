# Results: pre-registered eponym-retirement shifts

**Temporal trail (verifiable):**
- Pre-registration committed: `8c0b577` — 2026-06-06T09:55:28-04:00
  (`preregistration/eponym_retirement_2011.md`), BEFORE any data.
- This results file + the underlying counts are committed AFTER, in a
  separate, later-timestamped commit. PubMed counts were retrieved only
  after the pre-registration commit.

Counts are annual PubMed title/abstract record counts (1990–2024) for
the verbatim pre-registered term sets. Verdicts apply the locked
PASS/PARTIAL/FAIL mapping (durable crossover by 2024; timing within
±7 years of anchor) without modification.

## Shift 1 — GPA (anchor 2011)

| | old (Wegener's) | new (granulomatosis with polyangiitis) |
|---|---|---|
| total 1990–2024 | 5,103 | 4,820 |
| 2011 (anchor) | 222 | 49 |
| 2013 (crossover) | 159 | 219 |
| 2024 | 41 | 494 |

- First durable crossover (new > old): **2013** (within ±7 of 2011).
- New term at/above eponym in 2024: **yes** (494 vs 41).
- **VERDICT: PASS.**

## Shift 2 — EGPA (anchor 2012)

| | old (Churg-Strauss) | new (eosinophilic granulomatosis with polyangiitis) |
|---|---|---|
| total 1990–2024 | 2,226 | 1,802 |
| 2012 (anchor) | 139 | 11 |
| 2015 (crossover) | 54 | 90 |
| 2024 | 33 | 221 |

- First durable crossover (new > old): **2015** (within ±7 of 2012).
- New term at/above eponym in 2024: **yes** (221 vs 33).
- **VERDICT: PASS.**

## Honest note on the registered tolerance

We registered a wide ±7-year window because we expected medical eponyms
to be linguistically sticky and slow to retire. The observed adoption
was in fact *fast* — crossover within 2–3 years of the anchor, quicker
than the DSM-5 diagnostic-reclassification renames in the same study.
The predictions therefore passed more comfortably than anticipated;
this is reported as observed, not as expected.

## Deviations from pre-registration

None. The query terms, anchors, window (±7 years), and PASS/PARTIAL/FAIL
mapping are exactly as committed in `eponym_retirement_2011.md`.
