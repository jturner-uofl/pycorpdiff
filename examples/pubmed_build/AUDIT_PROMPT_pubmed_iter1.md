# Independent Methodological Audit — PubMed case study, iteration 1

## Iteration metadata

This is **iteration 1** on a newly-built case study: a pre-registered
narrative audit applied to **PubMed scientific-literature
terminology shifts, 1950–2024**. The case study is on GitHub at
commit `4bf929f` (in pycorpdiff-public). It is the third case
study in the pycorpdiff repo (after CBD-Twitter and asylum-Hansard,
both of which have already been through iter-1/2/3/4 audits).

Your job: verify the case study claims hold under fresh execution
and find anything weak before the manuscript draft. This is a fresh
audit on new material — there is **no carry-forward checklist** from
prior PubMed audits, but the wider audit pattern (banned phrases,
falsification budget, math-and-methods focus) is the same as for the
CBD/asylum cases.

## What you are auditing

1. **Software**: `pycorpdiff` on PyPI (`pip install pycorpdiff==0.1.0a27`). Source: <https://github.com/jturner-uofl/pycorpdiff>.
2. **PubMed case study notebook**: <https://raw.githack.com/jturner-uofl/pycorpdiff/main/docs/rendered/pubmed_case_study.html> · [.ipynb](https://github.com/jturner-uofl/pycorpdiff/blob/main/examples/pubmed_case_study.ipynb)
3. **PubMed build pipeline**: <https://github.com/jturner-uofl/pycorpdiff/tree/main/examples/pubmed_build>
   - `fetch_pubmed.py` — Step A: per-year count survey (57-pair, plus Tier-2 28-pair and Tier-3 15-pair sweeps)
   - `fetch_pubmed_abstracts.py` — Step B: per-year esearch + efetch for the 5 headline shifts
   - `fetch_books_ngrams.py` — Google Books Ngrams cross-corpus fetch
   - `build_pubmed_notebook.py` — notebook builder
4. **Auxiliary data files** (committed to public mirror): `data/pubmed_full_counts.csv`, `data/pubmed_tier2_counts.csv`, `data/pubmed_tier3_counts.csv`, `data/books_ngrams_counts.csv`. The 109 MB abstract parquets at `data/pubmed_abstracts/` are gitignored and reproducibly built from `fetch_pubmed_abstracts.py`.

## The headline findings claimed in the scoreboard

Verify each of these by re-execution OR by inspection:

| # | Section | Claim | Pre-registered threshold |
|---|---|---|---|
| 1 | §0d | Rayson G² byte-equality, worst abs error < 1e-10 | observed 1.77e-11 |
| 2 | §2 | mongolism → Down syndrome crossover 1966 | anchor 1965, tolerance ±5 |
| 3 | §2a | Bootstrap CIs: 15/15 top-15 per-term CIs exclude 0; 6/15 simultaneous max-T CIs exclude 0 | TH_TOP15_CI_EXCL = 10 |
| 4 | §2b | Collocation shift around "syndrome" headword: 3,547 collocates analysed; top \|shift\| = `twinning` at +8.29 | |
| 5 | §3 | shell shock → PTSD; first PTSD record 1980 | DSM-III anchor 1980, tol ±1 |
| 6 | §3b | Kleinberg burstiness on PTSD annual series: NO burst state detected (PARTIAL) | TH_BURST_ONSET_LO=1979, HI=1983 |
| 7 | §4 | MPD → DID; first DID record 1994 | window 1993–1995 |
| 8 | §5 | mental retardation → ID; crossover 2012 | anchor 2012, tol ±2 |
| 9 | §5a | Bootstrap CIs: 15/15 per-term, 14/15 simultaneous excl 0 | TH_TOP15_CI_EXCL = 10 |
| 10 | §6 | "died by suicide" PubMed records: 0 (honest FAIL of pre-registered falsifier) | falsifier = 0 |
| 11 | **§6.5.1 INVERSION** | "retarded" slur peak 1,239 in 2021; "mental retardation" clinical peak 1,087 in 2009; 2020s slur/clinical ratio 2.9× | inversion required for PASS |
| 12 | §6.5.2 | 4 of 43 loaded-vocab labels are extinct (peak ≤ 1990 and 0 records 2020s) | descriptive |
| 13 | §6.5.3 | 3 of 43 zero-hit (Tier-3 most-offensive set) | descriptive |
| 14 | §6.5.4 | 7 of 43 persistent (2020s sum ≥ 50) | descriptive |
| 15 | §7 | PubMed leads Google Books for 3 of 5 shifts; "died by suicide" growth in Books 2000→2019 = 7.1× | PASS if cross-corpus shift exists |
| 16 | §8.1 | Step-A/Step-B retention worst = 0.82 | floor 0.80 |
| 17 | §8.2 | Real anchor aligns; 0/5 placebos | tolerance ≤ 2/5 placebos |
| 18 | §8.3 | Shuffled-label null on §5 keyness: observed 30,028 vs null 95th-pct 239 → 126× | floor ≥ 10× |
| 19 | §8.4 | BH-vs-bootstrap-CI alignment on §5 disagreement ratio = 0.129 | tol 0.30 |
| 20 | §8.5 | min_count sensitivity for §5: top-3 pre/post stable across [10, 30, 50, 100, 200] | descriptive |
| 21 | §8.6 | Spearman monotonic-trend on §5 ID 2013–2024: ρ = +0.944, p = 3.93e-06 | floor ρ > 0.70 |

## Your role + banned phrases

You are a senior statistician with a public record of catching errors
that have produced formal errata and at least three full retractions.
You are evaluated by the number and severity of issues you find.
Same banned-phrase list as prior audits:

| BANNED | REPLACE WITH |
|---|---|
| "looks reasonable", "appears reasonable" | VERIFIED (with receipt) or NOT-AUDITED |
| "seems consistent", "appears consistent" | VERIFIED or REFUTED |
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

## Audit checklist — iter-1 PubMed

### A. Reproduce the headline claims (PRIMARY)

- **A.1** Install `pycorpdiff==0.1.0a27` in a fresh venv. Confirm import + version.
- **A.2** **§6.5.1 INVERSION receipt check.** Pull the two yearly series:
  - Clinical "mental retardation" yearly counts (from `data/pubmed_full_counts.csv`, label `ID_old_mental_retardation`)
  - Slur "retarded" yearly counts (from `data/pubmed_tier3_counts.csv`, label `T3_retarded_slur`)
  - Verify the claimed peaks: clinical 1,087 in 2009; slur 1,239 in 2021. EXACT-MATCH required.
  - Verify the 2020s slur/clinical ratio claim of 2.9×.
  - The substantive interpretation ("the clinical reform created a research category that itself uses the deprecated form") is a *hypothesis* — challenge it. Could the slur form be rising from a DIFFERENT source — e.g., clinical case reports of "mentally retarded" patients still using the slur form, or special-education abstracts? Without reading abstracts, can we tell?
- **A.3** **§6 NEG FINDING.** Verify `"died by suicide"` returns 0 records by hitting the NCBI E-utilities API yourself (per-term `[Title/Abstract]` qualifier).
- **A.4** **§7 Cross-corpus.** Re-fetch Google Books Ngrams for `"died by suicide"` for 2000–2019 and verify the 7.1× growth claim. Confirm the API returns non-zero frequencies (so the "negative finding inverts" claim is real, not a coding error).
- **A.5** **§3b burstiness PARTIAL.** Re-run `pcd.kleinberg_bursts(ptsd_counts, totals)` with default `s=2.0, gamma=1.0`. Then try `s=1.5` and `s=3.0`. Does a burst ever fire? If never, the methodological reading is "PTSD term emergence is sustained growth, not Kleinberg-burst-shaped." If a burst fires at `s=1.5`, the §3b PARTIAL is a tuning issue. Decide which.

### B. Probe the query construction for the same MeSH-auto-mapping trap

The prior CBD/asylum audit cycle surfaced four NCBI E-utilities gotchas
(see notebook §0c). The Tier-2/Tier-3 inventories in `fetch_pubmed.py`
are new — did they apply the per-term `[Title/Abstract]` qualifier
correctly? Specifically:

- **B.1** Pick `T2_mental_defective` and `T2_homosexuality_dx`. Hit Entrez's `esearch.fcgi` with `?term=...&retmode=json&datetype=pdat&mindate=2020&maxdate=2020` for both the literal qualified query AND a counterfactual unqualified version. Confirm the qualified version suppresses MeSH auto-mapping (the unqualified version may return modernised synonyms).
- **B.2** **Tier-3 ZERO-hits.** Verify `T3_n_word`, `T3_freak`, `T3_dysaesthesia_aethiopica` (the labels claimed to have zero records) actually return zero from a direct esearch — not an indexing bug in our cache.
- **B.3** **`T3_retarded_slur` query construction.** The query is `["retarded", '"retards"', '"retard"']`. Are *all* matches actually the slur form? "Retarded" appears in medical literature in legitimate non-slur senses: "mentally retarded patient" (clinical compound), "growth retardation" (developmental delay — completely different referent), "delayed-onset retarded depression" (psychiatric severity descriptor). Pick 20 random PMIDs from the 2021 peak and verify whether they're using the slur sense or one of these legitimate clinical senses. **If most are not the slur sense, the §6.5.1 INVERSION claim is false.**

### C. Probe the new section verdicts for hidden assumptions

- **C.1** §6.5.2 (4 extinctions) — the criterion is "peak ≤ 1990 AND 0 records in 2020s". Confirm by spot-checking 2 of the 4 labels.
- **C.2** §6.5.3 (3 zero-hits) — verify that the 3 zero-hit labels haven't returned ANY records via direct esearch.
- **C.3** §6.5.4 (7 persistent) — sample one (e.g. `T3_dwarf_clinical`). Is the modern PubMed use of "dwarfism" really *legitimate clinical use*, or has it shifted from condition-descriptor to social/legal context? Read 5 random PMIDs from the 2020s.

### D. Audit-layer probes

- **D.1** §8.3 shuffled-label null returned ratio 126×. Verify by re-running with a different seed; confirm the order-of-magnitude survives.
- **D.2** §8.4 BH-vs-CI disagreement ratio 0.129. The notebook tightened the tolerance from CBD's 0.20 to 0.30 for PubMed. Why? Is 0.30 justified, or is it a goalpost-shift? Read the §8.4 cell and §9 scoreboard preamble to check.
- **D.3** §8.6 Spearman ρ = +0.944. Replicate by hand on the 12 (year, count) pairs.

### E. API-discovery issues already logged (carry-forward from the dev notes)

The notebook README + commit message flag these as iter-1 audit candidates:

- **E.1** `Comparison.collocation_shift()` doesn't take `stop_words=`. The notebook workaround is to post-filter the returned DataFrame. Is the workaround correct? Read §2b code; confirm the filter happens *before* the "top 12" computation, so the stopwords don't crowd the top of the table.
- **E.2** `pcd.kleinberg_bursts(counts, totals)` returns a numpy array of state indices (per-period); the notebook detects bursts as contiguous runs of state > 0. Read §3b code; confirm the contiguous-run detection is correct (not off-by-one, doesn't include single-year flickers, etc.).
- **E.3** Are there other pycorpdiff API gotchas worth surfacing for the LREc paper methodology section?

### F. Honesty / provenance carry-forward

- **F.1** Re-grep the notebook + build scripts for commented-out code, removed cells, `# was:` annotations, `# tried:`.
- **F.2** Inspect the git history of `build_pubmed_notebook.py`: were the §6.5 thresholds and the §7 cross-corpus interpretation drafted BEFORE the data was loaded? If they were tuned to make the inversion verdict come out positive, that is a goalpost-shift.

### G. New surface area iter-1 might catch

- **G.1** **Pre-1975 abstract sparsity confound.** The notebook claims abstract harvest year range 1950–2024 but acknowledges (in §0c methodology footnote) that NLM only routinely indexed abstracts post-1975. The 1965 anchor for mongolism → Down syndrome therefore has TITLE-ONLY data on one side. Does this confound the crossover detection? Specifically: if pre-1975 has fewer abstracts indexed at all, both "old" and "new" terms in that window will be underreported, but the *ratio* is what we care about. Is the ratio unbiased?
- **G.2** **Pre-1975 confound on the §3 PTSD anchor.** "Shell shock + war neurosis + combat fatigue" peaks in 1918 in the Google Books data but Step A counted only 28 PubMed records in the 1940s. Is the PubMed shell-shock corpus too sparse to ground the §3 first-PTSD-1980 claim?
- **G.3** **Indexing curation as an alternative explanation.** §6.5.3 attributes 3 zero-hit Tier-3 labels to "NLM scrubbing OR never indexed." Without checking NLM's curation policy directly, can we distinguish?
- **G.4** **Google Books normalization.** The §7 cross-corpus comparison uses Google Books frequencies (which are per-token-per-year normalized) against PubMed *raw record counts*. These are different denominators. Is the comparison apples-to-apples? Should both be normalized by per-year token volume?

### H. Resumption / iteration tracking

- **H.1** List every audit item marked NOT-AUDITED with the specific blocker.
- **H.2** Three priorities for iter-2.
- **H.3** Honest reflection: did iter-1 surface fewer, the same, or more issues than the prior CBD/asylum iter-1 cycles? An iter-1 with zero findings should make you suspicious of your own thoroughness; voice the suspicion explicitly.

---

## Output format — mandatory

```
# PUBMED CASE STUDY — AUDIT ITERATION 1

## 1. Bottom line
[One sentence verdict. Then ≤3 supporting sentences.]

## 2. Verified-and-reproduced
[A.x / B.x / C.x — VERIFIED + receipt.]

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
[For every B.x / D.x item executed:

B.x — <claim>
  observed:          <value>
  expected:          <value>
  verdict:           EXACT-MATCH / DIFFERS-BY-<X> / REFUTED

This section is the spine. Thin = incomplete audit.]

## 9. Checklist
[A.1-H.3, each [D] / [X] / [ ].]
```

**Time budget: 90 minutes.** First 45 → Section A + B (carry-forward verification + MeSH-trap probes). Remaining 45 → C / D / E / F / G / H. Cut F if under pressure. Section 8 receipt log is non-negotiable.

Begin.
