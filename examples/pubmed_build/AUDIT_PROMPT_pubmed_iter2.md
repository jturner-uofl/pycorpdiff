# Independent Methodological Audit — PubMed case study, iteration 2

## Iteration metadata

This is **iteration 2** of the PubMed case-study audit. **GitHub at commit `24a8d82`** (in pycorpdiff-public).

Iter-1 found a **BLOCKING** refutation of the §6.5.1 INVERSION claim: 20 random PMIDs from the alleged `T3_retarded_slur` 2021 peak were 0/20 slur uses — all 20 were legitimate scientific senses (chemistry retardation, biology retarding growth, etc.). The authors then ran a word-sense induction (WSI) pipeline (`disambiguate_retard.py`) on 31,479 records, regex-bucketed them into 11 senses + an `unknown` residual, and rewrote §6.5.1 as an **AUDIT-RESOLVED** finding: slur sense is 4/31,479 = 0.013% (essentially absent); clinical-ID compound declines 96% 1990s→2020s (corroborates §5).

**Your two jobs**:

1. **Verify the audit-resolution holds** — the new §6.5.1 numbers, the WSI pipeline construct validity, the sense-bucket boundaries.
2. **Apply the same 20-PMID polysemy discipline** to the 5 other Tier-3 labels iter-1 flagged but did not check: `T3_dwarf_clinical`, `T3_lunatic`, `T3_freak`, `T3_midget`, `T3_imbecile_slur`, `T2_spastic_clinical`. The 1-of-15 sense-collision rate observed in iter-1 makes broader screening mandatory.

## What you are auditing

1. **Software**: `pycorpdiff` on PyPI (`pip install pycorpdiff==0.1.0a27`). Source: <https://github.com/jturner-uofl/pycorpdiff>.
2. **PubMed case study notebook**: <https://raw.githack.com/jturner-uofl/pycorpdiff/main/docs/rendered/pubmed_case_study.html> · [.ipynb](https://github.com/jturner-uofl/pycorpdiff/blob/main/examples/pubmed_case_study.ipynb) — commit `24a8d82`.
3. **WSI pipeline**: `examples/pubmed_build/disambiguate_retard.py`.
4. **Aux data** (all in clone): `data/retard_sense_counts_by_year.csv`, `data/pubmed_full_counts.csv`, `data/pubmed_tier2_counts.csv`, `data/pubmed_tier3_counts.csv`, `data/books_ngrams_counts.csv`.
5. **iter-1 audit report**: `examples/pubmed_build/AUDIT_PROMPT_pubmed_iter1.md` (the prompt + the audit's findings are in the previous commit history).

## Carry-forward state from iter-1

| Finding | iter-1 status | iter-2 verification |
|---|---|---|
| §6.5.1 INVERSION construct | REFUTED (0/20 slur in 20 random PMIDs) | Verify the WSI rewrite numbers + sense-bucket validity |
| `T3_retarded_slur` label name | recommended rename to `T3_retarded_morpheme` | Confirm rename in `fetch_pubmed.py`, CSV, and notebook |
| Other Tier-3 polysemy risk: `T3_dwarf_clinical`, `T3_lunatic`, `T3_freak`, `T3_midget`, `T3_imbecile_slur`, `T2_spastic_clinical` | NOT-AUDITED | Apply 20-PMID sense check to each |
| §8.4 BH-vs-CI threshold weakening (0.30 vs CBD's 0.20) | flagged, verdict robust | If unchanged, log as NOT-RESOLVED; if justified, verify justification |
| §3b burstiness PARTIAL | structural not tuning | Re-verify |
| §8.3 shuffled-label null re-run | NOT-AUDITED (parquet + 400s budget) | Attempt re-run if abstract parquets are available |

## Banned phrases

| BANNED | REPLACE WITH |
|---|---|
| "looks reasonable" / "appears reasonable" | VERIFIED + receipt, or NOT-AUDITED |
| "seems consistent" / "appears consistent" | VERIFIED or REFUTED |
| "seems robust" | VERIFIED-ACROSS-{N}-VARIATIONS or NOT-VARIED |
| "generally agrees" | EXACT-MATCH or DIFFERS-BY-{X} |
| "mostly correct" | name the wrong part |
| "trust the author" | irrelevant — verify the code |
| "documented in the prose" | irrelevant — verify the execution |
| "looks like the intent was" | irrelevant — describe what the code does |

## Output labels

- **VERIFIED** — executed yourself, observed claimed result. Receipt required.
- **CONFIRMED-BY-INSPECTION** — read carefully, did not execute. State why.
- **NOT-AUDITED** — did not attempt. State why.
- **REFUTED** — checked and observed a different result.

## Checklist — iter-2

### A. Verify the iter-1 audit-resolution (§6.5.1)

- **A.1** Open `data/retard_sense_counts_by_year.csv`. Verify these claims:
  - Total records 1990–2024: 31,479 (sum across all senses and years)
  - `slur_explicit_mention` total: 4
  - `clinical_intellectual_disability` total: 2,968
  - Per-decade clinical compound: 1990s = 1,679; 2020s = 73 (96% decline)
  - Per-decade growth_developmental: 1990s = 652; 2020s = 90 (86% decline)
  - `unknown` total: 16,521 (52.5%)
- **A.2** Read `disambiguate_retard.py` lines 30–150 (the `SENSE_PATTERNS` regex list). For 5 random PMIDs from the `clinical_intellectual_disability` bucket and 5 random PMIDs from `slur_explicit_mention`, verify the classifier's verdict by reading the abstract text:
  - For clinical-ID: should be a clinical/research paper using "mentally retarded" or "mental retardation" compound. If any is actually a history-of-medicine paper *about* the term, that's a CONSTRUCT FAILURE.
  - For slur (only 4 records exist; check all 4): should be papers explicitly discussing the slur (e.g., "the R-word," "the slur 'retard'," "stigma of being called retarded"). If they're false-positive regex matches, the 4 number could be lower (even more impressive null result).
- **A.3** Sample 10 random PMIDs from the `unknown` (16,521-record) bucket. Are they ALL scientific process-verb uses (as iter-1's spot-check of 15 found)? Or do any contain hidden slur uses that the regex missed?

### B. **20-PMID polysemy discipline on other Tier-3 labels** (PRIMARY)

The iter-1 audit recommended applying the same random-20-PMID sense-check to 6 other labels with potential polysemy. Do each.

For each of the following labels, query NCBI E-utilities directly using the per-term `[Title/Abstract]` qualified syntax (same as iter-1 verified). Pull 20 random PMIDs from the year with the highest record count for that label. Read each title + abstract; classify into:
- (a) the intended sense (clinical/medical use of the deprecated term)
- (b) a different sense (e.g., astrophysics "dwarf star," lay informal use of "lunatic," etc.)
- (c) history-of-medicine / stigma-research *about* the term (not deprecating use)

Report the fraction of intended-sense matches for each.

- **B.1** `T3_dwarf_clinical` queries: `dwarfism`, `dwarf`, `"primordial dwarf"`. Possible cross-sense: dwarf stars (astrophysics), botanical dwarf species. *In medical-journal PubMed this should be high signal, but verify.*
- **B.2** `T3_lunatic` queries: `lunatic`, `lunatics`, `"lunatic asylum"`, `lunacy`. Possible cross-sense: lay informal use, history-of-medicine retrospective scholarship, legal references.
- **B.3** `T3_freak` queries: `"freak of nature"`, `"medical freak"`, `"freaks of nature"`. iter-1 noted this returns zero records. Re-verify via direct esearch.
- **B.4** `T3_midget` queries: `midget`, `midgets`. Possible cross-sense: scientific use ("midget cell" in retinal biology, "midget penguin" in zoology, etc.).
- **B.5** `T3_imbecile_slur` queries: `imbeciles`, `imbecility`. Possible cross-sense: historical/legal references to eugenic-era statutory language.
- **B.6** `T2_spastic_clinical` queries: `"spastic child"`, `"spastic children"`, `"the spastics"`, `"spastic diplegic"`. Should be high-clinical (cerebral-palsy literature). But: is "the spastics" historically a UK-charity-name reference (Spastics Society renamed to Scope in 1994)?

**Required output**: for each B.1–B.6 label, a number "X / 20 records are the intended sense" + a 2-sentence interpretation. If any label has < 15/20 intended-sense, flag as polysemy collision requiring §6.5 rewrite.

### C. WSI pipeline construct probes

- **C.1** Are the §6.5.1 sense buckets MUTUALLY EXCLUSIVE? Read `classify_text()` in `disambiguate_retard.py`. The function uses first-match-wins ordering. Check: if a single abstract contains BOTH "mental retardation" (clinical) AND "retard tumor growth" (biology process-verb), which sense wins? Is the ordering defensible? (clinical-ID is listed first, so it should win — verify.)
- **C.2** Does the WSI pipeline UNDERCOUNT the clinical sense by missing "retardation" (noun, no -ed suffix)? The fetcher query is `retarded OR "retards" OR "retard"` — does NOT match "retardation". Pull a sample and check: does PubMed report records with "mental retardation" in the title but NO instance of "retard*" without -ation suffix? If so, the §6.5.1 clinical-ID-compound count is undercounted.
- **C.3** The slur-explicit_mention regex pattern is `r"\b(the\s+r-word|the\s+slur\s+(ret|retard)|called\s+(\w+\s+)?retarded|use\s+of\s+the\s+word\s+retard\w*|(stigma\w*|derogat\w+|slur\w*)\s+(of\s+)?retard\w*|r\*tard|ret\*\*d)\b"` — try alternative slur-detection patterns: `"called retarded"`, `"used as a slur"`, `r-word`. Does broader pattern catch more records? Even if it catches 20 instead of 4, is the headline "essentially absent" still defensible (vs. the corpus total 31,479)?

### D. Audit-layer carry-forward

- **D.1** §3b PARTIAL: iter-1 found this is structural (PTSD is sustained growth, not burst-shaped). The notebook prose currently reports PARTIAL without further explanation. Should it be relabeled "EXPECTED-PARTIAL" or stay PARTIAL? Make a recommendation.
- **D.2** §8.4 disagreement threshold `TH_BH_CI_DISAGREE = 0.30` vs CBD's `0.20`. iter-1 flagged this as needing justification but observed disagreement 0.129 is robust to either threshold. Read the cell around the threshold definition. Is there a justification comment? If not, log as iter-3 polish item.
- **D.3** §8.3 shuffled null re-run with new seed: the cached output is 30,028 vs 239 = 126×. iter-1 deferred this due to the gitignored abstract parquets. If you can locally fetch the §5 (mental retardation + intellectual disability) abstracts via `fetch_pubmed_abstracts.py --only 2010s_id`, attempt a fresh-seed replicate. ~400s wall-time.

### E. Survey: what's in the published HTML vs the notebook?

- **E.1** Open <https://raw.githack.com/jturner-uofl/pycorpdiff/main/docs/rendered/pubmed_case_study.html>. Confirm the §6.5.1 prose is the AUDIT-RESOLVED version (not the old refuted INVERSION narrative). Confirm the scoreboard row 11 is "AUDIT-RESOLVED" not "INVERSION."

### F. Honest-disclosure carry-forward

- **F.1** Grep notebook + build scripts for `# was:` / `# tried:` / commented-out code. iter-1 found zero; re-verify.
- **F.2** Inspect git log of `build_pubmed_notebook.py` between `4bf929f` and `24a8d82` (the audit-resolution commit). Are the §6.5.1 changes consistent with the audit-mandated rewrite, or did anything else change unannounced?

### G. New surface area

- **G.1** **Apply 20-PMID discipline to §6.5.2 extinctions.** Iter-1 verified the *arithmetic* (4 labels with peak ≤ 1990 and zero records in 2020s) but did NOT verify the construct. Sample 5 PMIDs from each of: `T2_mongoloid_idiot`, `T2_dope_fiend`, `T3_bastard`, `T3_imbecile_slur`. Are they CLINICAL USE of the deprecated term, or are they history-of-medicine retrospective scholarship *about* the term? The "extinction" label should reflect clinical use disappearing, not all scholarly mention.
- **G.2** Re-check the §6.5.4 persistent labels with the 20-PMID protocol — iter-1 only spot-checked `T3_dwarf_clinical`. Apply to the other 6 (`T3_deformed`, `T3_lunatic`, `T3_midget`, `T2_spastic_clinical`, etc.).
- **G.3** Does §7's "PubMed leads Books" claim hold for the AUDIT-RESOLVED §6.5.1 (now that the morpheme is correctly interpreted as a multi-sense scientific verb-form)? The cross-corpus comparison was originally framed against the §2–§5 shifts, which are unaffected by the §6.5.1 fix — but verify.

### H. Resumption / iteration tracking

- **H.1** Every NOT-AUDITED with blocker.
- **H.2** Three priorities for iter-3.
- **H.3** Honest reflection: iter-1 found 1 BLOCKING; iter-2 should be doing carry-forward verification on that fix + finding new issues. If iter-2 finds zero NEW issues, voice the suspicion.

---

## Output format — mandatory

```
# PUBMED CASE STUDY — AUDIT ITERATION 2

## 1. Bottom line
[One sentence + ≤3 supporting.]

## 2. Verified-and-reproduced
[A.x / B.x / C.x / etc. — VERIFIED + receipt.]

## 3. Confirmed-by-inspection
[A.x — CONFIRMED-BY-INSPECTION + why not executed.]

## 4. Refuted
[A.x REFUTED + observed vs claim.]

## 5. Failures and corrections required
[A.x — severity BLOCKING / MAJOR / MINOR — specific file/line — corrected text/computation.]

## 6. Methodological concerns short of failures
[Concern + suggested correction.]

## 7. Not-audited and resumption notes
[H.1 / H.2 / H.3.]

## 8. Math + methods receipt log
[For every executed claim:

X — <claim>
  observed:  <value>
  expected:  <value>
  verdict:   EXACT-MATCH / DIFFERS-BY-<X> / REFUTED

Spine. Thin = incomplete audit. Sections B.1–B.6 receipts are mandatory.]

## 9. Checklist
[A.1-H.3, each [D] / [X] / [ ].]
```

**Time budget: 60 minutes.** First 30 → Section A + B (carry-forward verification + 6 new polysemy probes). Remaining 30 → C / D / E / G / H. Cut F if pressured. Section 8 mandatory for B.1–B.6.

Begin.
