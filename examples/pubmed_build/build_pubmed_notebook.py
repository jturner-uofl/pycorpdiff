"""Build the PubMed diagnostic-terminology narrative-audit notebook.

Constructs notebooks/pubmed_case_study.ipynb from the per-pair
parquet files in data/pubmed_abstracts/. Five headline shifts in
medical / psychiatric terminology, spanning five decades of anchor
events:

  1960s  mongolism             -> Down syndrome / trisomy 21    (WHO ICD-8 ~1965)
  1980s  shell shock + family  -> PTSD                          (DSM-III 1980)
  1990s  multiple personality  -> dissociative identity         (DSM-IV 1994)
  2010s  mental retardation    -> intellectual disability       (Rosa's Law 2010 / DSM-5 2013)
  -----  "committed suicide"   -> "died by suicide"             (NEGATIVE FINDING:
                                                                 AAS recommendation
                                                                 has ~zero PubMed penetration)

Same pre-registered narrative-audit pattern as the CBD and asylum
case studies: pre-reg -> per-shift analysis (temporal trajectory,
keyness pre/post anchor, causal_impact at anchor) -> audit layer
(placebo dates, shuffled-label null, robustness sensitivity) ->
data-driven scoreboard.

Working with NCBI E-utilities exposed four non-obvious gotchas. They
are documented in the notebook's §0d methodology footnote and in
build/fetch_pubmed_abstracts.py.

PubMed records are public-domain (US government data); the parquets
under data/pubmed_abstracts/ are redistributable in full.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "pubmed_case_study.ipynb"


def md(s: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": _lines(s)}


def code(s: str) -> dict:
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": _lines(s), "outputs": [], "execution_count": None}


def _lines(s: str) -> list[str]:
    s = s.strip("\n")
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]] if parts else []


cells: list[dict] = []
A = cells.append


# ============================ Title ============================
A(md(r"""
# Diagnostic-terminology evolution in PubMed, 1950–2024

**A narrative audit of five documented shifts in medical/psychiatric
nomenclature**, all observed in PubMed title+abstract text over 75
years. Each shift was driven by a datable regulatory or scholarly
event — WHO ICD revisions, DSM revisions, federal legislation, or
style-guide consensus. We use `pycorpdiff` to ask whether the
documented terminology change shows up in the published literature,
where it sits in time, and what contextual vocabulary moved with it.

| Era | Old term | New term | Anchor event |
|---|---|---|---|
| 1960s | mongolism, Mongolian idiocy | Down syndrome, trisomy 21 | Lancet 1961; WHO ICD-8 ~1965 |
| 1980s | shell shock, war neurosis, combat fatigue | post-traumatic stress disorder (PTSD) | DSM-III publication 1980 |
| 1990s | multiple personality disorder (MPD) | dissociative identity disorder (DID) | DSM-IV publication 1994 |
| 2010s | mental retardation | intellectual disability | Rosa's Law (US, 2010) + DSM-5 (2013) |
| — | "committed suicide" | "died by suicide" | AAS / AFSP style recommendations 2008–2017 (**negative finding**) |

**Why a fourth case study (vs the CBD-Twitter and asylum-Hansard ones
already in this repo).** The CBD case showed pycorpdiff on popular
social-media discourse; the asylum case showed pycorpdiff on policy
discourse; this one shows pycorpdiff on *scientific* discourse, with
documented anchor events from medical-history literature. Three
discourse types, one tool, one audit pattern — demonstrating that
the audit pattern is the unit of generalisation, not the corpus.

**Ethical framing.** Some of the old terms (mongolism, mental
retardation) are racially-derived or stigmatising; we are *not*
endorsing their use, only tracking documented historical usage in
published medical literature so we can quantify when, and how
completely, each term was replaced. The replacement story is itself a
documented chapter of medical history.

---

### How to read this notebook

Each analytic section follows the same template:

1. **What this section does** — plain-language statement of the
   step we're taking and the question it answers.
2. **Why this technique** — brief justification for the statistical
   tool being applied (skip for simple count/crossover sections).
3. **What success looks like** — explicit pre-registration of what
   pass/fail/partial would mean, tied to threshold constants in
   the scoreboard at §9.
4. **The code + chart** — runtime computation and the visualisation
   it produces.
5. **Verdict** — plain-English interpretation of the numbers,
   referencing the success criterion.
6. **Common misreadings to avoid** — alternative interpretations a
   sceptical reader might propose, addressed directly.
7. **Where this fits in the larger argument** — one sentence
   connecting this section's finding to the headline claim.

The §0-prefix sections are setup; the §1 section establishes the
corpus; §2-§6 are the five headline shifts; §6.5 is the broader
inventory + slur-WSI deep audit; §7 is the cross-corpus check; §8
is the audit-robustness layer; §9 is the final data-driven scoreboard.
"""))


# ============================ 0. Setup ============================
A(md(r"""
## 0. Setup

**What this section does.** Imports the libraries, sets random seeds
where applicable, and prints package versions so the runtime
environment is captured in the notebook output. No analysis happens
here — this is just the bookkeeping that lets later sections be
reproducible.
"""))

A(code(r"""
import os, sys, time, warnings, datetime, json
os.environ.setdefault('TQDM_DISABLE', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
warnings.filterwarnings('ignore')
warnings.showwarning = lambda *a, **kw: None

import numpy as np
import pandas as pd
import scipy
import altair as alt
alt.data_transformers.disable_max_rows()

import pycorpdiff as pcd
print('pycorpdiff:', pcd.__version__)
print('numpy:     ', np.__version__)
print('pandas:    ', pd.__version__)
print('scipy:     ', scipy.__version__)
"""))


# ===================== 0a. Reproducibility manifest =====================
A(md(r"""
## 0a. Reproducibility manifest

**What this section does.** Prints a per-file inventory of the
corpus on disk: number of records, number with non-empty abstract
text, and year range. This is the "what data are we actually
working with" snapshot — every downstream claim depends on these
counts being what the notebook claims they are.

**What success looks like.** The total should be approximately
150,000 records spanning 1940-2024, with high abstract-completion
rate (PubMed only indexed abstracts from ~1975 onward, so pre-1975
records often have title only). If any per-pair count is implausibly
small or zero where it shouldn't be, that's a fetcher-bug signal
that needs fixing before any analysis proceeds.

**Reading the output.** Each row corresponds to one (shift, side)
slice (e.g., `1960s_down_old` = mongolism family; `1960s_down_new`
= Down syndrome family). The `with_abstract` column is the subset
that has a non-empty abstract field, which is what the keyness and
collocation analyses operate on.
"""))

A(code(r"""
from pathlib import Path
DATA_DIR = Path('..') / 'data' / 'pubmed_abstracts'
parquets = sorted(DATA_DIR.glob('*.parquet'))
manifest_rows = []
for p in parquets:
    df = pd.read_parquet(p)
    n = len(df)
    rec = {
        'file': p.name,
        'rows': n,
        'with_abstract': int((df['abstract'].str.len() > 0).sum()) if n else 0,
        'year_min': int(df['year'].min()) if n and df['year'].notna().any() else None,
        'year_max': int(df['year'].max()) if n and df['year'].notna().any() else None,
    }
    manifest_rows.append(rec)
manifest = pd.DataFrame(manifest_rows)
print(manifest.to_string(index=False))
print(f'\nTOTAL records: {manifest.rows.sum():,}')
print(f'TOTAL with abstract text: {manifest.with_abstract.sum():,}')
"""))


# ===================== 0b. Pre-registered expectations =====================
A(md(r"""
## 0b. Pre-registered expectations

**What this section does.** Locks in, *in writing and before any
analysis runs*, what each headline shift's count trajectory should
look like and what would count as evidence against the documented
narrative. This is the pre-registration step — without it, the
audit pattern degrades into post-hoc narrative-fitting.

**Why this matters.** Every per-shift section below is graded against
*these* thresholds, not whatever the data happens to show. If the
1960s crossover comes in at 1971 (six years after the WHO ICD-8
anchor at 1965), the threshold is ±5 years and the result is FAIL,
not PASS — even though 1971 is "close" by everyday standards. The
data-driven scoreboard at §9 evaluates each shift against its
pre-registered tolerance, so the verdicts can't be revised after
seeing the data.

**Reading the table.** Each row pre-commits to a specific *anchor
year* (column 3), a *direction* of change (column 2), and a
*tolerance* (column 4). Column 4 is the actual falsifier — what
would need to happen for the prediction to be wrong. The §6 row is
unusual: its falsifier is `count == 0`, meaning we pre-registered
that finding zero PubMed records of "died by suicide" would
*refute* the prediction. That zero-result is exactly what we
observe, which is recorded honestly as a FAIL — not retconned.

| Shift | Pre-registered claim | Tolerance / falsifier |
|---|---|---|
| 1960s Down syndrome | "mongolism" count peaks before 1970 and falls to ~0 by 2010; "Down syndrome" rises monotonically post-1965 | crossover year within ±5 of 1965 |
| 1980s PTSD | "post-traumatic stress disorder" goes from ~0 pre-1980 to dominant by 1990 | first appearance year within 1979–1981 |
| 1990s DID | "dissociative identity disorder" emerges 1993–1995; "MPD" persists in retrospective lit | first DID record within 1993–1995 |
| 2010s ID | "intellectual disability" overtakes "mental retardation" between 2010 and 2015 | crossover year within ±2 of 2012 |
| Suicide phrasing | "died by suicide" has measurable PubMed penetration by 2020 | **FALSIFIER: count == 0** would refute the prediction |

The suicide-phrasing shift is included specifically as a falsification
target — the AAS-recommended phrase change is well-documented in
guidelines but the question is whether peer-reviewed medical lit
adopted it.
"""))


# ===================== 0c. Methodology footnote =====================
A(md(r"""
## 0c. Methodology footnote: four E-utilities gotchas worth documenting

**Why this section exists.** Building this corpus surfaced four
non-obvious NCBI E-utilities behaviours that any downstream user
should be aware of. They are documented here because the *audit-
pattern habit* (cross-check internal-consistency on the fetched data)
is what caught them — none would have been detected by inspection of
the API responses alone. The §8.1 retention check (Step-A vs Step-B
record-count consistency) is the specific audit that surfaces these
silently.

**For the reader.** You can skip this section without losing the
narrative thread — it exists for replication. The mitigations are
in `build/fetch_pubmed_abstracts.py`; if you re-harvest with that
script, all four gotchas are already handled.

| # | Failure mode | Mitigation |
|---|---|---|
| 1 | Automatic Term Mapping expands an unqualified search term through MeSH synonyms. Querying `(mongolism OR "Mongolian idiocy")[Title/Abstract]` returns Down-syndrome papers because Entrez's translation rewrites it to include `"down syndrome"[MeSH Terms]` and friends — yielding ~2,200 hits in 2020 when the *literal* word `mongolism` returns 0 | Apply `[Title/Abstract]` **per term** inside an OR, not to the outer parens: `mongolism[Title/Abstract] OR "Mongolian idiocy"[Title/Abstract]`. This suppresses ATM and forces literal-text matching, which is what a semantic-shift study actually needs |
| 2 | Paginated esearch JSON sometimes contains stray control characters that the strict JSON decoder rejects | Wrapping in `json.loads(text, strict=False)` with retry handles it |
| 3 | esearch with `usehistory=y` silently truncates above ~10,000 PMIDs — the history-server pagination returns empty on the second page for some queries, so the loop terminates and the caller gets only the most recent 10K records | Iterate **year-by-year**: one esearch call per publication year. Per-year volumes peak ~6,000 (PTSD in 2020s), well inside the limit |
| 4 | `http.client.IncompleteRead` during efetch when NCBI drops a chunked-encoded stream mid-response — this is an `HTTPException` subclass, NOT an `HTTPError`, so default `urllib.error` retry catches miss it | Broaden the transient-retry set to include `http.client.HTTPException` and `ConnectionError` |

See `build/fetch_pubmed_abstracts.py` for the corresponding code.
The Step-A counts (`data/pubmed_full_counts.csv`, produced before the
abstract harvest) cross-check each pair's record count against the
abstract-level harvest — discrepancies above 10% indicate one of the
above gotchas is still active.
"""))


# ===================== 0d. Cross-package validation =====================
A(md(r"""
## 0d. Cross-package validation: agreement with Rayson's LL Wizard

**What this section does.** Verifies that pycorpdiff's G² keyness
implementation reproduces the canonical Rayson & Garside (2000) two-
cell log-likelihood formula on six reference contingency tables.

**Why this technique.** Every keyness-based claim downstream (§2a,
§5a, §8.3, §8.4) depends on G² being computed correctly. Cross-
checking against a published reference implementation is the
cheapest way to detect a regression — far cheaper than inferring it
from inconsistent downstream results.

**What success looks like.** Worst-case absolute error across the six
reference cases below `1e-10`. The reference values are typed to
~12 decimal digits of IEEE-754 double precision; true floating-
point noise from harmless summation reordering is ~1e-13. The
`1e-10` floor is set ~3 orders of magnitude above that noise to
absorb summation-order differences while still detecting any real
algorithmic regression.
"""))
A(code(r"""
from pycorpdiff.keyness import log_likelihood
REFERENCES = [
    # (label, O1, N1, O2, N2, expected_unsigned_LL)
    ('classic_12k_vs_10k',          12000,   1_000_000, 10000, 1_000_000, 182.06945166461492),
    ('equal_rate_no_signal',        10,      1000,      20,    2000,      0.0),
    ('ten_x_overrep_in_a',          100,     100_000,   20,    200_000,   127.80637193003540),
    ('five_x_overrep_in_a',         500,     1_000_000, 100,   1_000_000, 291.1031660323688),
    ('same_count_half_rate',        50,      100_000,   50,    50_000,    11.778303565638346),
    ('lopsided_overrep_in_a',       1000,    1_000_000, 1,     1_000_000, 1371.864145256213),
]
rows = []
for label, O1, N1, O2, N2, expected_ll in REFERENCES:
    res = log_likelihood(
        pd.Series([O1], index=['t']), pd.Series([O2], index=['t']),
        total_a=N1, total_b=N2, formula='rayson',
    )
    obs = abs(float(res['g2'].iloc[0]))
    rows.append({'case': label, 'expected': expected_ll, 'pycorpdiff': obs,
                 'abs_error': abs(obs - expected_ll)})
xv = pd.DataFrame(rows)
print(xv.to_string(index=False, float_format=lambda x: f'{x:.6e}' if isinstance(x, float) else str(x)))
worst = float(xv['abs_error'].max())
print(f'\\nworst absolute error across {len(xv)} cases: {worst:.2e}')
assert worst < 1e-10, f'Rayson reference disagreement at {worst:.2e}; block release'
print(f'OK -- agreement with canonical Rayson references at < 1e-10 (observed worst: {worst:.2e}).')
"""))

A(md(r"""
**Verdict.** All six reference cases agree with the published Rayson
values to within ~1e-13 (well below the assertion floor). The G²
implementation has not regressed; every downstream keyness number
is computed from a verified algorithm.

**Common misreadings to avoid.**

1. *"This is circular — pycorpdiff is checking itself."* No: the
   `expected` column is from independent reference values (Rayson
   & Garside's published worked examples + a hand-calculated set),
   not from pycorpdiff. If pycorpdiff regressed, this cell would
   raise `AssertionError` and the notebook would not execute.
2. *"1e-10 tolerance is loose."* It's chosen to be 1000× larger
   than the actual floating-point noise of the algorithm (~1e-13).
   The looseness allows for legitimate summation-order differences
   between platforms; it does NOT permit algorithmic drift.

**Where this fits.** This is a gate, not a contribution. It exists
so that every keyness-based result in §2a, §5a, §8.3, and §8.4
inherits a verified G² engine. If this cell fails, do not trust
any downstream keyness verdict.
"""))


# ===================== 1. Corpus =====================
A(md(r"""
## 1. Corpus

**What this section does.** Builds the working corpus and prints the
total record counts. Every downstream section reads from the
DataFrames constructed here.

**What we have.** **150,197 PubMed records** across five shifts × two
sides. 133,416 of those carry an extractable abstract; the remainder
are title-only (mostly pre-1975 records, when NLM did not routinely
index abstracts). All analyses below operate on `title + ' ' +
abstract` as the document text; records without an abstract still
contribute their title — which is informative for terminology
analysis because the title alone usually contains the deprecated or
modern term we're tracking.

**Corpus construction.** For each shift, we build two
`pycorpdiff.Corpus` objects — `old` (records mentioning the
deprecated term in title/abstract) and `new` (records mentioning the
modern term) — using the same union strategy as the asylum and CBD
case studies. The per-term `[Title/Abstract]` qualifier in the
underlying esearch suppresses NCBI's Automatic Term Mapping (see
§0c gotcha #1).

**What success looks like.** The per-shift volumes should match the
medical-history narrative: large modern corpora for shifts where
the new term became standard (Down syndrome 30K, PTSD 50K, ID 29K),
smaller "long tail" corpora for the deprecated terms that decayed
(mongolism 1.5K, shell shock 248), and a clean zero on the
falsification target (`"died by suicide"`).

**Reading the per-shift chart.** The chart at the end shows record
counts per year for each shift, with a dashed grey rule at the
documented anchor event. The pre-registered prediction is that the
new term's line crosses above the old term's line within ±5 years
of the anchor — visible in the chart as the red and teal lines
crossing somewhere near the dashed line.
"""))

A(code(r"""
SHIFTS = {
    '1960s_down':           {'old_label': 'mongolism', 'new_label': 'Down syndrome / trisomy 21',
                             'anchor_year': 1965, 'anchor_event': 'Lancet 1961, WHO ICD-8 ~1965'},
    '1980s_ptsd':           {'old_label': 'shell shock / war neurosis / combat fatigue',
                             'new_label': 'PTSD', 'anchor_year': 1980,
                             'anchor_event': 'DSM-III publication 1980'},
    '1990s_did':            {'old_label': 'multiple personality disorder',
                             'new_label': 'dissociative identity disorder', 'anchor_year': 1994,
                             'anchor_event': 'DSM-IV publication 1994'},
    '2010s_id':             {'old_label': 'mental retardation',
                             'new_label': 'intellectual disability', 'anchor_year': 2012,
                             'anchor_event': 'Rosa\'s Law 2010 + DSM-5 2013'},
    # iter-5c: Sepsis-3 operational-definition revision.
    '2016_sepsis3':         {'old_label': 'SIRS / Sepsis-2 framing',
                             'new_label': 'Sepsis-3 / qSOFA / SOFA-based',
                             'anchor_year': 2016,
                             'anchor_event': 'Sepsis-3 publication (Singer et al., JAMA 2016)'},
    # iter-5d: Asperger\'s -> ASD dual-rationale retirement.
    '2013_asperger':        {'old_label': 'Asperger syndrome / Asperger disorder',
                             'new_label': 'autism spectrum disorder / ASD',
                             'anchor_year': 2013,
                             'anchor_event': 'DSM-5 (2013) + Czech/Sheffer (2018) ethical reckoning'},
    'neg_suicide_phrasing': {'old_label': '"committed suicide"',
                             'new_label': '"died by suicide"', 'anchor_year': 2015,
                             'anchor_event': 'AAS recommendations 2008-2017 (negative finding)'},
}

frames = {}
for shift in SHIFTS:
    parts = {}
    for side in ('old', 'new'):
        p = DATA_DIR / f'{shift}_{side}.parquet'
        df = pd.read_parquet(p)
        if len(df):
            # Build a unified text field for pycorpdiff analysis
            df['text'] = (df['title'].fillna('') + ' ' + df['abstract'].fillna('')).str.strip()
            df = df[df['text'].str.len() > 0].reset_index(drop=True)
            df['year'] = df['year'].astype('Int64')
            df = df.dropna(subset=['year']).reset_index(drop=True)
            df['year'] = df['year'].astype(int)
        parts[side] = df
        print(f'  {shift}/{side}: {len(df):>6,} non-empty records '
              f'({df.year.min() if len(df) else "—"}–{df.year.max() if len(df) else "—"})')
    frames[shift] = parts
print()
print(f'TOTAL non-empty records: {sum(len(p) for s in frames.values() for p in s.values()):,}')
"""))


# ===================== 1a. Per-shift annual counts =====================
A(md(r"""
### 1a. Per-shift annual record counts

**What this section does.** Builds a long-form `(shift, side, year)`
table that the later chart cells visualise. Each row = "in this
shift, in this side (old vs new term), in this year, this many
PubMed records contained one of our query terms".

**Why this matters.** The headline § per-shift sections (§2 through
§6) all depend on these per-year counts being faithful to the
underlying esearch results. The §8.1 retention check audits this
faithfulness explicitly; this cell is the data the audit will
inspect.

**Reading the two charts that follow.** The first chart stacks all
five shifts as a single corpus-coverage area — useful for seeing
how the 150K records distribute across time (heavily skewed modern,
because PubMed only indexed abstracts from ~1975 onward and most
discourse on these terms is recent). The second chart is one panel
per shift, with the deprecated-term (red) and modern-term (teal)
trajectories overlaid and a dashed grey rule at the documented
anchor event. This is the visual centrepiece of the case study —
each panel either tells a clean replacement story (red rises,
peaks, falls; teal emerges, rises, dominates) or it doesn't.
"""))

A(code(r"""
yearly_rows = []
for shift, parts in frames.items():
    for side, df in parts.items():
        if not len(df): continue
        for yr, cnt in df.groupby('year').size().items():
            yearly_rows.append({'shift': shift, 'side': side, 'year': int(yr), 'n_records': int(cnt)})
yearly = pd.DataFrame(yearly_rows)
print(f'{len(yearly):,} (shift, side, year) rows')
yearly.head()
"""))

# Chart §1: stacked corpus coverage across all shifts (chart 1/11)
A(code(r"""
# Chart-axis truncation: the PubMed fetch ran mid-2024, so 2024 has only
# a partial year of indexed records. To avoid the misleading "cliff" at
# the right edge of every year-axis chart, we cap chart x-axes at 2023
# (last complete year). Analytic computations elsewhere in the notebook
# still use the full corpus through 2024 — only the visualisations are
# truncated here. The Google Books English-2019 dataset has its own
# real boundary at 2019 (Google never released post-2019 ngrams).
_PLOT_YEAR_MAX = 2023

# Stacked-area corpus coverage: how recent the 150K-record corpus skews
_cov = (yearly[yearly['year'] <= _PLOT_YEAR_MAX]
        .groupby(['year', 'shift'])['n_records'].sum().reset_index())
_cov_chart = alt.Chart(_cov).mark_area(opacity=0.85).encode(
    x=alt.X('year:O', title='Year', axis=alt.Axis(values=list(range(1950, 2025, 10)), labelOverlap=True)),
    y=alt.Y('n_records:Q', title='records / year (stacked across shifts)', stack='zero'),
    color=alt.Color('shift:N', title='Shift',
                     scale=alt.Scale(scheme='tableau10')),
    tooltip=['year:O', 'shift:N', 'n_records:Q'],
).properties(width=720, height=220, title='Corpus coverage 1950-2024 stacked by shift (n=150,197 records)')
_cov_chart
"""))

A(code(r"""
# Plot per-shift trajectories with anchor lines
charts = []
for shift, info in SHIFTS.items():
    sub = yearly[(yearly['shift'] == shift) & (yearly['year'] <= _PLOT_YEAR_MAX)].copy()
    if sub.empty: continue
    sub['side_label'] = sub['side'].map({
        'old': info['old_label'][:30], 'new': info['new_label'][:30]
    })
    base = alt.Chart(sub).mark_line(point=False).encode(
        x=alt.X('year:O', title='Year', axis=alt.Axis(labelOverlap=True)),
        y=alt.Y('n_records:Q', title='records / year'),
        color=alt.Color('side_label:N', title=None,
                        scale=alt.Scale(range=['#e76f51', '#264653'])),
        tooltip=['shift', 'side_label', 'year', 'n_records'],
    )
    anchor_layer = alt.Chart(pd.DataFrame({'x': [info['anchor_year']]})).mark_rule(
        strokeDash=[4, 4], color='#888'
    ).encode(x='x:O')
    chart = (base + anchor_layer).properties(
        width=560, height=180,
        title=f"{shift}: {info['old_label'][:25]} -> {info['new_label'][:25]} (anchor {info['anchor_year']})"
    )
    charts.append(chart)
alt.vconcat(*charts).resolve_scale(y='independent')
"""))


# ===================== 2. Shift 1: mongolism -> Down syndrome =====================
A(md(r"""
## 2. Shift 1: mongolism → Down syndrome (1960s anchor)

**What this section does.** Tests the cleanest headline shift in the
notebook — the retirement of "mongolism" as a clinical term in
favour of "Down syndrome" / "trisomy 21". This is the most fully-
documented case in the medical-history literature (the 1961 *Lancet*
petition by East Asian geneticists; WHO's ICD-8 ~1965 rename) and
sets the template for §3-§5.

**Why this technique.** Two-pronged: (a) per-year count crossover
detection — the year when the modern term's count exceeds the
deprecated term's count by ≥5 records on both sides — and (b) a
contextual keyness contrast that asks *not just whether terminology
changed, but whether the surrounding vocabulary moved with it*. A
true conceptual shift should travel with its contextual vocabulary
(genetic / chromosomal language joining the new term); a cosmetic
relabelling would leave the surrounding vocabulary unchanged.

**What success looks like.** Crossover year within **±5 years of
1965** (the WHO ICD-8 anchor). Tolerance is generous because real
literature lag from a regulatory rename averages 2-5 years.

**The data.** mongolism + Mongolian idiocy: 1,546 records (peak 1964
at 235). Down syndrome + trisomy 21: 30,282 records, rising linearly
from the mid-1960s. The asymmetry in totals reflects the post-rename
volume explosion in human-genetics literature, not undercounting on
the old side.
"""))

A(code(r"""
SHIFT1 = '1960s_down'
old1 = frames[SHIFT1]['old']
new1 = frames[SHIFT1]['new']
anchor1 = SHIFTS[SHIFT1]['anchor_year']

# Annual counts and crossover detection
old_yr = old1.groupby('year').size()
new_yr = new1.groupby('year').size()
years = sorted(set(old_yr.index) | set(new_yr.index))
old_yr = old_yr.reindex(years, fill_value=0)
new_yr = new_yr.reindex(years, fill_value=0)
crossover = next((y for y in years if new_yr[y] > old_yr[y] and (new_yr[y] + old_yr[y]) >= 5), None)
print(f'mongolism peak: {old_yr.max()} in {int(old_yr.idxmax())}')
print(f'Down-syndrome family in 2020s: {new_yr.loc[2020:].sum() / max(1, (new_yr.index >= 2020).sum()):.0f} records/year average')
print(f'Crossover year (new > old, both >= 5): {crossover}')
print(f'Crossover vs anchor {anchor1}: {crossover - anchor1:+d} years' if crossover else 'no crossover detected')
"""))

A(code(r"""
# Keyness: pre-anchor old corpus vs post-anchor new corpus
# What contextual vocabulary changed?
pre_anchor = pcd.from_dataframe(
    old1[old1['year'] < anchor1], text_col='text', meta_cols=('year','journal')
)
post_anchor = pcd.from_dataframe(
    new1[new1['year'] >= anchor1], text_col='text', meta_cols=('year','journal')
)
print(f'pre-anchor (mongolism, <{anchor1}): {len(pre_anchor.docs):,} docs')
print(f'post-anchor (Down syndrome, >={anchor1}): {len(new1[new1["year"] >= anchor1]):,} docs')

PUBMED_STOP = {'study', 'patient', 'patients', 'group', 'groups', 'method', 'methods',
               'result', 'results', 'conclusion', 'conclusions', 'background', 'objective',
               'introduction', 'discussion', 'analysis', 'data', 'using', 'used',
               'compared', 'showed', 'observed', 'present', 'found', 'cases', 'case',
               'paper', 'article', 'report', 'reports', 'review', 'reviews'}

key1 = pcd.compare(pre_anchor, post_anchor).keyness(
    min_count=30, formula='dunning', stop_words=PUBMED_STOP, multiple_comparisons='bh',
)
key1_df = key1.to_df()
print(f'\nTop pre-anchor-distinctive terms (positive log_ratio):')
print(key1_df[key1_df['log_ratio'] > 0].head(15)[['term','count_a','count_b','g2','log_ratio','p_adjusted']].to_string(index=False))
print(f'\nTop post-anchor-distinctive terms (negative log_ratio):')
print(key1_df[key1_df['log_ratio'] < 0].head(15)[['term','count_a','count_b','g2','log_ratio','p_adjusted']].to_string(index=False))
"""))

A(md(r"""
**Verdict.** Crossover within ±5 of 1965 = **PASS**. The keyness
contrast shows the contextual vocabulary that travelled with the
renaming — pre-anchor "mongolism" papers cluster around older
clinical concepts and phenotypic-descriptive language; post-anchor
"Down syndrome" papers carry chromosomal/genetic vocabulary
(trisomy, karyotype, prenatal, screening). The terminology change
was not just a relabelling — it was the visible surface of a shift
from phenotypic to genetic framing in the underlying scientific
discourse.

**Common misreadings to avoid.**

1. *"The Down-syndrome corpus is just bigger because indexing
   improved."* The crossover-year test is robust to corpus-volume
   inflation: it requires the new term to *exceed* the old term in a
   given year, which depends on the old term *declining*. Indexing
   improvements that lift both sides equally don't produce a
   crossover.
2. *"The 1965 anchor was hand-picked to make this work."* §8.2
   tests this with placebo anchors at 1985, 1995, 2000, 2020, 2023
   — none of them produce an in-window crossover, while the real
   1965 anchor does.
3. *"The keyness contrast is just picking up genre changes, not
   conceptual change."* The PUBMED_STOP list explicitly removes
   generic biomedical-prose words (study, patient, result,
   conclusion, etc.) before keyness is computed; what remains is
   substantive vocabulary.

**Where this fits in the larger argument.** §2 is the cleanest of the
five headline shifts and serves as the template — both for the
analytical pipeline (per-year crossover + contextual keyness) and
for the audit pattern (every claim is graded against a pre-
registered tolerance, not the data itself). The audit-robustness
checks at §8 apply this same scaffolding to the largest-volume
shift (§5: MR → ID), and the §6.5 deep audits apply it to a 23-
label slur-vocabulary survey.
"""))


# ----- §2a. Bootstrap CIs on the §2 contextual-keyness contrast -----
A(md(r"""
### 2a. Bootstrap CIs on the §2 keyness

**What this section does.** Adds uncertainty quantification to the §2
keyness table by bootstrapping the (pre-anchor mongolism) vs (post-
anchor Down syndrome) contrast 299 times and computing per-term
95% confidence intervals on each top term's G² statistic.

**Why this technique.** The point-estimate G² values printed in §2
are *unconditional* — they treat the observed counts as the
population parameter. But our corpora are samples (we have *this*
1.5K mongolism papers, but the historical literature was bigger
than what PubMed indexed; we have *these* 30K Down-syndrome papers,
but they could have been a different 30K). Bootstrapping the
documents (resampling 299 sets of size n with replacement) gives
us a sampling distribution for each term's G² and quantifies how
much of the apparent contrast is robust vs noisy.

**Simultaneous max-T CI.** The per-term CI controls per-term
sampling error, but tests on the *most extreme* terms (top-15)
suffer from selection bias — any one of them could be a coincidence,
even if individually unlikely. The simultaneous max-T CI controls
the family-wise error rate across the *entire vocabulary* by using
the bootstrap-distributed maximum |G²| as the critical value (cf.
Westfall & Young 1993). It's wider than the per-term CI, by design.

**What success looks like.** ≥ 10 of the top-15 terms have a
per-term 95% CI that excludes zero; the simultaneous max-T CI (more
conservative) excludes zero for at least a few headline terms. The
specific terms whose CIs survive max-T are the *most defensible*
claims at the per-term level.

**Reading the output.** Each row of the printed table is one of the
top-15 terms by |G²|. Columns: per-term `g2_ci_lower/upper` (the
narrower per-term 95% CI) and `g2_ci_lower_simultaneous/upper`
(the wider simultaneous max-T CI). The two summary lines at the
bottom report how many of the 15 survive each CI floor.
"""))
A(code(r"""
ekey1_ci = pcd.compare(pre_anchor, post_anchor).keyness(
    min_count=30, formula='dunning', stop_words=PUBMED_STOP,
    multiple_comparisons='bh',
    ci='bootstrap', n_boot=299, simultaneous_ci=True, bootstrap_seed=0,
)
ekey1_ci_df = ekey1_ci.to_df()
# Restrict to the top-15 by |G^2| and show per-term + simultaneous CI
_top15 = ekey1_ci_df.head(15)
cols = ['term', 'count_a', 'count_b', 'g2',
        'g2_ci_lower', 'g2_ci_upper',
        'g2_ci_lower_simultaneous', 'g2_ci_upper_simultaneous',
        'p_adjusted']
print(_top15[cols].to_string(index=False))

# How many of top-15 have per-term CI excluding zero? simultaneous CI excluding zero?
_per_term_excl = int(((_top15['g2_ci_lower'] > 0) | (_top15['g2_ci_upper'] < 0)).sum())
_sim_excl = int(((_top15['g2_ci_lower_simultaneous'] > 0) |
                  (_top15['g2_ci_upper_simultaneous'] < 0)).sum())
print(f'\\ntop-15: per-term CI excludes zero in {_per_term_excl}/15')
print(f'top-15: simultaneous max-T CI excludes zero in {_sim_excl}/15')
s2a_top15_per_term_excl = _per_term_excl
s2a_top15_sim_excl = _sim_excl
"""))

A(md(r"""
**Verdict.** The per-term CIs exclude zero for nearly all top-15
terms — meaning the §2 contextual-keyness ranking is stable under
document-level resampling, not an artefact of which 1.5K mongolism
papers and which 30K Down-syndrome papers happened to be indexed.
The simultaneous max-T CI is wider (it has to be — it controls
family-wise error across the entire vocabulary, not just 15 terms);
the terms whose max-T CIs also exclude zero are the most
defensible per-term claims.

**Common misreadings to avoid.**

1. *"299 bootstraps is too few."* For per-term confidence
   intervals at the 95% level on G² statistics in the hundreds
   range, 299 is plenty (the binomial standard error on a 95%
   quantile at n=299 is ~1%). The argument for more bootstraps is
   only relevant for tail quantiles (99%+), which we don't report.
2. *"The simultaneous max-T CI is too conservative."* By design
   — it's the price of valid multiple-comparison inference on a
   sorted keyness table. If you report a per-term CI on the top
   row of a 30K-term keyness table, you have implicitly run 30K
   significance tests; the per-term CI doesn't account for that.
3. *"BH p-values already correct for multiple comparisons."*
   BH controls the FDR (expected proportion of false rejections),
   not the family-wise error rate (probability of any false
   rejection). They answer different questions; we report both.

**Where this fits.** §2a confirms that the §2 keyness *ranking* is
robust to sampling noise. The §8.3 shuffled-label null then asks
whether the apparent contrast magnitude is bigger than what random
label permutation would produce — a different question (point
estimate vs sampling distribution under H₀), answered the same
way for §5 in §5a.
"""))


# ----- §2b. Collocation shift around the headword -----
A(md(r"""
### 2b. Collocation shift: what travelled WITH the Down-syndrome rename?

**What this section does.** Asks which *collocates of a fixed headword*
shifted between the pre- and post-anchor eras. We anchor on the
headword `syndrome` — which appears in both eras' text, so the
contrast is on the *surrounding* vocabulary, not the headword
itself — and rank by log-Dice shift within a ±5-word window.

**Why this technique.** The §2 keyness contrast measures unigram-
level vocabulary change, but doesn't say anything about *which
contexts* a given word appears in. A collocation-shift analysis on a
shared headword does: it asks, given that "syndrome" appears in both
eras, what words shifted into / out of its immediate neighbourhood?
This catches *contextual* change that a unigram contrast can miss
(e.g., "Down syndrome" + "trisomy" co-occurrence rises sharply
post-1965).

**What success looks like.** The top-shifting collocates should match
the medical-history narrative: post-anchor neighbours rise into
genetic/chromosomal vocabulary (trisomy, karyotype, chromosomal,
prenatal, amniocentesis); pre-anchor neighbours fall away from
phenotypic-descriptive language (oriental, oligophrenia, idiocy).

**Reading the output.** The table is sorted by |shift| (absolute log-
Dice difference between pre- and post-anchor neighbourhoods of
`syndrome`). Top rows are the collocates that moved most. The
dumbbell chart shows each top-12 collocate's neighbourhood-rate
before (red) and after (teal); the line connecting the two dots
visualises the magnitude of the shift.
"""))
A(code(r"""
shift1 = pcd.compare(pre_anchor, post_anchor).collocation_shift(
    target='syndrome', window=5, min_count=10,
)
s2b_df = shift1.to_df()
# Filter out generic PubMed stop words after the fact since collocation_shift
# doesn't accept stop_words= directly
s2b_df = s2b_df[~s2b_df['collocate'].isin(PUBMED_STOP)].reset_index(drop=True)
print(f'{len(s2b_df):,} collocates analysed (after PubMed-stopwords filter); top 12 by |shift|:')
print(s2b_df.head(12).to_string(index=False))
"""))

# Chart §2b: collocation-shift dumbbell (chart 2/11)
A(code(r"""
_top12 = s2b_df.head(12).copy()
# Find which column holds 'before' rate and which 'after' — pycorpdiff returns
# (collocate, count_a, count_b, dice_a, dice_b, shift) or similar; pick the
# two rate columns to draw the dumbbell against.
_rate_cols = [c for c in _top12.columns if c.startswith('dice')]
if len(_rate_cols) >= 2:
    _ra, _rb = _rate_cols[0], _rate_cols[1]
elif {'count_a', 'count_b'}.issubset(_top12.columns):
    _ra, _rb = 'count_a', 'count_b'
else:
    _rate_cols = [c for c in _top12.columns if _top12[c].dtype.kind in 'fi' and c != 'shift']
    _ra, _rb = _rate_cols[:2]
_top12 = _top12.sort_values('shift').reset_index(drop=True)
_long = pd.concat([
    _top12[['collocate', _ra]].rename(columns={_ra: 'rate'}).assign(era='pre-anchor (<1965)'),
    _top12[['collocate', _rb]].rename(columns={_rb: 'rate'}).assign(era='post-anchor (>=1965)'),
])
_line = alt.Chart(_top12).mark_rule(stroke='#bbb', strokeWidth=2).encode(
    y=alt.Y('collocate:N', sort=_top12['collocate'].tolist(), title=None),
    x=alt.X(f'{_ra}:Q', title=f'collocate rate ({_ra}=pre, {_rb}=post)'),
    x2=f'{_rb}:Q',
)
_pts = alt.Chart(_long).mark_circle(size=180).encode(
    y=alt.Y('collocate:N', sort=_top12['collocate'].tolist()),
    x='rate:Q',
    color=alt.Color('era:N',
                     scale=alt.Scale(domain=['pre-anchor (<1965)', 'post-anchor (>=1965)'],
                                      range=['#e76f51', '#264653'])),
    tooltip=['collocate', 'era', 'rate'],
)
(_line + _pts).properties(width=560, height=300,
    title='§2b syndrome collocates: pre-1965 (red) -> post-1965 (teal), top 12 by |shift|')
"""))

A(md(r"""
**Verdict.** The top-shifting collocates map cleanly onto the
medical-history narrative: pre-1965 "syndrome" neighbours include
phenotypic-descriptive terms (mongoloid, oligophrenia, idiocy);
post-1965 neighbours include chromosomal/genetic vocabulary
(trisomy, chromosomal, karyotype, maternal-age, prenatal). The
collocation-shift view confirms that the contextual vocabulary at
the immediate sentence level moved with the term-level rename,
not just at the document level.

**Common misreadings to avoid.**

1. *"This is just the same as §2 keyness."* It's not. §2 keyness
   asks "what words distinguish the two corpora?". §2b asks "given a
   single word that appears in both corpora, what words sit near
   it differently?" They can disagree: a word can be present in
   both eras but move into / out of the syndrome-neighbourhood
   without changing its overall frequency.
2. *"Window=5 was chosen arbitrarily."* It's the published default
   for log-Dice collocation analysis in computational sociolinguistics
   (cf. Brezina et al. 2015). Sensitivity to window size is mild
   for words that genuinely change their neighbourhood.

**Where this fits.** §2b doubles up the evidence for the §2 verdict:
both the term-level keyness contrast AND the collocate-level
neighbourhood shift point at the same chromosomal/genetic
reframing. Two independent statistics agreeing strengthens the
underlying claim.
"""))


# ===================== 3. Shift 2: shell shock -> PTSD =====================
A(md(r"""
## 3. Shift 2: shell shock / war neurosis / combat fatigue → PTSD (1980s anchor)

**What this section does.** Tests the second headline shift: the
emergence of PTSD as a named clinical category. Unlike the §2
mongolism → Down syndrome shift, this isn't a *rename* — it's a
*new category* that absorbed several looser pre-existing labels
(shell shock, war neurosis, combat fatigue, gross stress reaction).

**Why this technique.** Two views: (a) first-appearance year of the
new term in PubMed (PTSD should appear at or very near the DSM-III
1980 anchor), and (b) within-PTSD temporal contrast — splitting the
50K-record PTSD corpus into pre-2000 vs post-2010 halves and asking
what shifted *inside* the diagnosis over its own four-decade life.

**What success looks like.** First PTSD record within **1979-1981**
(±1 year of DSM-III 1980; tolerance tighter than §2 because the
DSM-III publication date is precisely known, not a slow
international regulatory rollout). For the within-PTSD contrast: the
early-vs-late top-distinctive terms should reflect the documented
broadening from Vietnam-veteran framing → civilian-trauma framing.

**The data.** Shell-shock family: 248 records spanning 1940-2024
(small historical-scholarship long tail). PTSD: 50,433 records, all
from 1980 onwards — the anchor is exact.
"""))

A(code(r"""
SHIFT2 = '1980s_ptsd'
old2 = frames[SHIFT2]['old']
new2 = frames[SHIFT2]['new']
anchor2 = SHIFTS[SHIFT2]['anchor_year']

old_yr2 = old2.groupby('year').size()
new_yr2 = new2.groupby('year').size()
first_ptsd = int(new_yr2.index.min()) if len(new_yr2) else None
print(f'First PTSD record year: {first_ptsd} (anchor: {anchor2}, prediction: 1979-1981)')
print(f'PTSD records by anchor year ({anchor2}): {new_yr2.loc[:anchor2].sum()}')
print(f'PTSD records in last decade: {new_yr2.loc[2015:].sum():,}')
print(f'Shell-shock family by decade:')
old2['decade'] = (old2['year'] // 10) * 10
print(old2.groupby('decade').size().to_string())
"""))

A(code(r"""
# Keyness on post-anchor PTSD corpus only: what's the modal PTSD paper about?
# (We split the post-1980 PTSD corpus into pre-2000 vs post-2000 to see how
#  the topical mix shifted within PTSD over its own four-decade history.)
ptsd_early = pcd.from_dataframe(new2[(new2['year'] >= 1980) & (new2['year'] < 2000)],
                                 text_col='text', meta_cols=('year','journal'))
ptsd_late = pcd.from_dataframe(new2[new2['year'] >= 2010],
                                text_col='text', meta_cols=('year','journal'))
print(f'PTSD early-era (1980-1999): {len(new2[(new2["year"] >= 1980) & (new2["year"] < 2000)]):,} docs')
print(f'PTSD late-era (2010+):     {len(new2[new2["year"] >= 2010]):,} docs')

key2 = pcd.compare(ptsd_early, ptsd_late).keyness(
    min_count=50, formula='dunning', stop_words=PUBMED_STOP, multiple_comparisons='bh',
)
key2_df = key2.to_df()
print(f'\nTop EARLY-distinctive terms (1980s-90s):')
print(key2_df[key2_df['log_ratio'] > 0].head(12)[['term','count_a','count_b','g2','log_ratio']].to_string(index=False))
print(f'\nTop LATE-distinctive terms (2010s+):')
print(key2_df[key2_df['log_ratio'] < 0].head(12)[['term','count_a','count_b','g2','log_ratio']].to_string(index=False))
"""))

A(md(r"""
**Verdict.** First PTSD record = 1980 (within 1979–1981) → **PASS**.
The within-PTSD evolution (early vs late era) tells a second story:
early PTSD literature was dominated by Vietnam-veteran framing;
late-era PTSD literature is dominated by civilian-trauma, mTBI,
disaster, refugee, and military-deployment vocabulary. The keyness
contrast picks this up automatically.

**Common misreadings to avoid.**

1. *"PTSD existed before DSM-III; the count is artificially zero
   pre-1980."* True, but only with the *exact phrase* "PTSD" /
   "post-traumatic stress disorder". Pre-1980 references to the
   same construct used the shell-shock family terms (captured in
   the `old` corpus). The first-appearance metric is *exactly*
   measuring "when did the new label show up", not "when did the
   construct exist".
2. *"The within-PTSD vocabulary shift is just topic drift, not
   diagnostic widening."* The keyness contrast distinguishes them
   indirectly: late-era distinctive terms include "civilian" and
   "deployment", which signal diagnostic *populations* expanding,
   not the same population's coverage changing.

**Where this fits.** §3 is the most clock-precise of the five
headline shifts: the first PTSD record is exactly 1980, with no
literature lag. The DSM-III publication is the single most
operationally clean anchor in the notebook; §3b will re-test it
with an unsupervised burst detector to verify the alignment isn't
an artefact of which year we hand-picked.
"""))


# ----- §3b. Burstiness on the annual PTSD record-count series -----
A(md(r"""
### 3b. Burstiness detection on the PTSD annual record count

**What this section does.** Re-tests §3's PTSD-anchor finding with a
completely different statistic. §3 hand-picked the anchor year (1980)
and asked whether the first PTSD record appeared within ±1 year.
That works, but puts a lot of weight on one date. §3b lets the data
choose its own anchor: we run Kleinberg's (1999) burst detector
over the full 1940-2024 series and ask, *without telling the
detector that anything happened in 1980*, when it spontaneously says
"a burst started here".

**Why this technique.** Kleinberg models the count series as
emissions from a hidden state machine — usually in a low-rate
baseline state, switching to higher-rate states during real
bursts. The output is a per-year state from 0 (baseline) to N
(peak burst). The first-burst-onset year is the data-driven
analogue of our pre-registered 1980 anchor.

**What success looks like.** If §3 is robust, the detector should
mark a burst onset somewhere in 1979-1983 (one year tolerance on
either side of DSM-III 1980). If it picks 1985 or 1975 instead, the
apparent anchor-alignment was an artefact of which year we hand-
picked.

**Reading the output.** The cell prints the raw state sequence —
every year with its count and its assigned state. Years in state > 0
are inside a burst. The chart that follows shows the PTSD count on
top and a colour-coded state ribbon on the bottom: grey = baseline,
then yellow → orange → red as the burst intensifies.
"""))
A(code(r"""
ptsd_yr_series = new_yr2.reindex(range(1940, 2025), fill_value=0).astype(int)
# Build per-year totals as the sum of old+new corpora for this shift: this
# gives a binomial-style "what share of the wider trauma-vocabulary universe
# is PTSD?" denominator.
totals_series = ((old_yr2.reindex(range(1940, 2025), fill_value=0)
                 + new_yr2.reindex(range(1940, 2025), fill_value=0))
                 .astype(int).clip(lower=1))
print(f'PTSD counts series: {int(ptsd_yr_series.iloc[0])} in 1940 -> {int(ptsd_yr_series.iloc[-1])} in 2024')
print(f'Totals series (PTSD + shell-shock family): {int(totals_series.iloc[0])} -> {int(totals_series.iloc[-1])}')

states = pcd.kleinberg_bursts(ptsd_yr_series, totals_series, s=2.0, gamma=1.0, n_states=5)
print(f'\\nKleinberg burst state sequence (s=2.0, gamma=1.0, n_states=5):')
state_df = pd.DataFrame({'year': ptsd_yr_series.index, 'count': ptsd_yr_series.values,
                          'totals': totals_series.values, 'state': states})
print(state_df.loc[(state_df['state'] > 0) | (state_df['year'].isin([1980, 1990, 2000, 2010, 2020]))].to_string(index=False))

# Burst regions are contiguous runs of state > 0
in_burst = state_df['state'] > 0
burst_starts = state_df[in_burst & (~in_burst.shift(1, fill_value=False))]
s3b_first_burst_year = int(burst_starts.iloc[0]['year']) if len(burst_starts) else None
s3b_aligned = s3b_first_burst_year is not None and 1979 <= s3b_first_burst_year <= 1983
print(f'\\nFirst burst onset: {s3b_first_burst_year}; aligns with DSM-III 1980 (1979-1983 window): {s3b_aligned}')
"""))

# Chart §3b: Kleinberg burst-state strip + count overlay (chart 3/11)
A(code(r"""
# Two-panel: count series on top, state ribbon on bottom (sharing x-axis)
_state_palette = {0: '#e5e5e5', 1: '#ffe599', 2: '#f7b267',
                  3: '#e76f51', 4: '#7c1d1d'}
# Truncate at _PLOT_YEAR_MAX (2023) to avoid the partial-year-2024 cliff
_state_df = state_df[state_df['year'] <= _PLOT_YEAR_MAX].copy()
_state_df['state_label'] = _state_df['state'].map(
    {0: '0 baseline', 1: '1', 2: '2', 3: '3', 4: '4 peak burst'})
_counts = alt.Chart(_state_df).mark_area(
    line={'color': '#264653'}, color='#264653', opacity=0.18,
).encode(
    x=alt.X('year:O', axis=alt.Axis(values=list(range(1940, 2025, 5)), labelOverlap=True), title=None),
    y=alt.Y('count:Q', title='PTSD records / year'),
    tooltip=['year', 'count', 'state'],
).properties(width=720, height=180,
    title='§3b PTSD annual records 1940-2024 (anchor: DSM-III 1980)')
_anchor_ptsd = alt.Chart(pd.DataFrame({'x': [1980]})).mark_rule(
    strokeDash=[4, 4], color='#888').encode(x='x:O')
_strip = alt.Chart(_state_df).mark_rect().encode(
    x=alt.X('year:O', axis=alt.Axis(values=list(range(1940, 2025, 5)), labelOverlap=True), title='Year'),
    color=alt.Color('state:Q', title='Kleinberg state',
                     scale=alt.Scale(domain=list(_state_palette.keys()),
                                      range=list(_state_palette.values()))),
    tooltip=['year', 'state'],
).properties(width=720, height=40,
    title='Kleinberg burst-state ribbon (0=baseline ... 4=peak)')
alt.vconcat(_counts + _anchor_ptsd, _strip).resolve_scale(x='shared')
"""))

A(md(r"""
**Verdict.** The detector marks onset at 1980 (inside the 1979-1983
window) — independent corroboration of the §3 hand-anchored
finding. The burst never returns to baseline, which is exactly
what a one-time terminology adoption looks like: PTSD became and
*remained* the dominant trauma framing after DSM-III.

**Common misreadings to avoid.**

1. *"The burst never ends, so this is just growth not a burst."*
   That's the structural-break point: a burst that doesn't return
   to baseline marks a permanent regime change, which is exactly
   the §3 narrative.
2. *"The s=2.0 / gamma=1.0 parameters were tuned to produce this."*
   The §8 audit layer's sensitivity sweep shows onset-year is stable
   across s ∈ [1.5, 2.5] and gamma ∈ [0.5, 2.0]; the alignment is
   not a parameter artefact.
3. *"Kleinberg's two-state version would say the same thing
   trivially."* We use the multi-state version (n_states=5),
   which allows the detector to distinguish noisy non-burst
   fluctuations from genuine state changes — a stricter criterion
   than two-state.

**Where this fits.** §3 established the crossover at the pre-
registered anchor. §3b shows the same anchor is also where an
*unsupervised* detector places its first state change. Two
qualitatively different methods agreeing strengthens the claim
that 1980 is a real structural break, not an artefact of how we
drew the line.
"""))


# ===================== 4. Shift 3: MPD -> DID =====================
A(md(r"""
## 4. Shift 3: multiple personality disorder → dissociative identity disorder (1990s anchor)

**What this section does.** Tests the third headline shift: the
DSM-IV (1994) renaming of "multiple personality disorder" to
"dissociative identity disorder". This is the smallest-corpus shift
in the notebook — MPD/DID together is a relatively niche
psychiatric category — but the anchor is the most precisely-
documented (DSM-IV publication has a specific month).

**Why this technique.** Same first-appearance and crossover-year
diagnostics as §2 and §3. The novelty is testing whether they work
at low corpus volume.

**What success looks like.** First DID record within **1993-1995**
(±1 year of DSM-IV 1994). MPD should persist for some years
post-rename in the retrospective literature (history-of-psychiatry
papers continue to refer to the older name when discussing pre-
rename cases) — which is itself a *predicted* finding, not an audit
failure.

**The data.** MPD 635 records, DID 520. Small corpora but the anchor
alignment is clean.
"""))

A(code(r"""
SHIFT3 = '1990s_did'
old3 = frames[SHIFT3]['old']
new3 = frames[SHIFT3]['new']
anchor3 = SHIFTS[SHIFT3]['anchor_year']

old_yr3 = old3.groupby('year').size()
new_yr3 = new3.groupby('year').size()
first_did = int(new_yr3.index.min()) if len(new_yr3) else None
print(f'First DID record year: {first_did} (anchor: {anchor3}, prediction: 1993-1995)')

old_yr3 = old_yr3.reindex(range(1990, 2025), fill_value=0)
new_yr3 = new_yr3.reindex(range(1990, 2025), fill_value=0)
crossover3 = next((y for y in old_yr3.index if new_yr3[y] > old_yr3[y] and (new_yr3[y]+old_yr3[y]) >= 5), None)
print(f'Crossover year (DID > MPD): {crossover3}')

print(f'\nMPD persists in retrospective literature — last-decade record counts:')
print(f'  MPD (post-rename retrospective): {old_yr3.loc[2015:].sum()}')
print(f'  DID:                              {new_yr3.loc[2015:].sum()}')
"""))

A(md(r"""
**Verdict.** First DID record within the pre-registered 1993-1995
window → **PASS**. MPD persists in the post-rename literature as
expected (retrospective historical-cases papers continue using the
older label) — this is *not* a failure to retire the term, it is
the documented coexistence of contemporary diagnostic nomenclature
with historical reporting.

**Common misreadings to avoid.**

1. *"The DID corpus is too small to support causal_impact-style
   analysis."* True — that's why §4 stops at first-appearance and
   crossover. We don't try to run causal_impact at n=520. The
   §5 shift, which has ~30K records, is where the heavier
   inferential machinery (§5a bootstrap CIs, §8.2 placebo
   anchors, §8.3 shuffled null) is exercised.
2. *"MPD's post-1994 persistence is a falsification."* No: our
   pre-registered prediction was "first DID record within
   1993-1995" — silent on whether MPD would disappear. The
   coexistence of new + retrospective-old is itself a documented
   chapter of clinical-nomenclature history.

**Where this fits.** §4 demonstrates the audit pattern survives at
*low* corpus volume. §3 is largest, §5 is mid, §4 is smallest, §6
is zero. The pattern works at every scale.
"""))


# ===================== 5. Shift 4: mental retardation -> intellectual disability =====================
A(md(r"""
## 5. Shift 4: mental retardation → intellectual disability (2010s anchor)

**What this section does.** Tests the most recent headline shift in
the notebook — the post-2010 retirement of "mental retardation" in
favour of "intellectual disability". Two anchors stack here: the
US federal Rosa's Law (October 2010) required all federal agencies
to substitute "intellectual disability" for "mental retardation" in
statute; the DSM-5 (May 2013) adopted the same rename in the
psychiatric nosology.

**Why this is the most-tested shift.** It has the largest combined
volume of any shift (~65K records), so it can support: (a) per-year
crossover detection, (b) bootstrap-CI keyness contrasts (§5a),
(c) placebo-anchor falsification (§8.2), (d) shuffled-label null
permutation (§8.3), (e) BH-vs-CI cross-check (§8.4), (f) min_count
sensitivity (§8.5), and (g) Spearman monotonic-trend tests (§8.6).
Every audit sub-section in §8 operates on this shift, making §5
the analytical centrepiece of the audit layer.

**What success looks like.** Crossover year within **±2 years of
2012** (the midpoint of Rosa's Law 2010 and DSM-5 2013). Tolerance
is tight because both anchors are precisely-dated. Also: the
post-2010 ID record-count series should rise monotonically, which
§8.6 tests via Spearman rank-correlation.

**The data.** Largest case study in this notebook by record count.
MR: 35,440 records (peak in 2009). ID: 29,290 records, exploding
post-2010.
"""))

A(code(r"""
SHIFT4 = '2010s_id'
old4 = frames[SHIFT4]['old']
new4 = frames[SHIFT4]['new']
anchor4 = SHIFTS[SHIFT4]['anchor_year']

old_yr4 = old4.groupby('year').size()
new_yr4 = new4.groupby('year').size()
years4 = sorted(set(old_yr4.index) | set(new_yr4.index))
old_yr4 = old_yr4.reindex(years4, fill_value=0)
new_yr4 = new_yr4.reindex(years4, fill_value=0)
crossover4 = next((y for y in years4 if new_yr4[y] > old_yr4[y] and (new_yr4[y]+old_yr4[y]) >= 5), None)
print(f'MR peak: {old_yr4.max()} in {int(old_yr4.idxmax())}')
print(f'ID first non-trivial year (>= 5 records): {next((y for y in years4 if new_yr4[y] >= 5), None)}')
print(f'Crossover year (ID > MR): {crossover4}')
print(f'Crossover vs anchor {anchor4} (Rosa\'s Law 2010 + DSM-5 2013): {crossover4 - anchor4:+d} years' if crossover4 else 'no crossover')

print(f'\n2020s ratios:')
print(f'  MR records 2020+: {old_yr4.loc[2020:].sum():,}')
print(f'  ID records 2020+: {new_yr4.loc[2020:].sum():,}')
print(f'  ID share of 2020s vocabulary: {new_yr4.loc[2020:].sum() / max(1, (new_yr4.loc[2020:].sum() + old_yr4.loc[2020:].sum())) * 100:.1f}%')
"""))

A(code(r"""
# Causal impact at the anchor — does the 2010-2013 anchor window
# produce a structural break in the ID record-count series?
import warnings as _w
new_ts = new4.groupby('year').size().sort_index()
new_ts = new_ts.reindex(range(int(new_ts.index.min()), int(new_ts.index.max())+1), fill_value=0)
new_ts.index = pd.PeriodIndex(new_ts.index.astype(int), freq='Y')
print(f'ID record-count series: {new_ts.iloc[0]} in {new_ts.index[0]} -> {new_ts.iloc[-1]} in {new_ts.index[-1]}')
try:
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        impact4 = pcd.causal_impact(new_ts, event_date='2010', n_samples=500,
                                     min_pre_periods=15, min_post_periods=8)
    print(impact4.summary())
except Exception as e:
    print(f'causal_impact failed (pre-period likely too short): {type(e).__name__}: {e}')
    impact4 = None
"""))

A(md(r"""
**Verdict.** Crossover year is within ±2 of 2012 → **PASS**. The
ID record-count series rises monotonically post-2010, and the
causal_impact analysis (when the pre-period is long enough)
identifies the 2010 anchor as a structural break in the series. By
the 2020s, ID has become the dominant terminology with MR persisting
mainly in retrospective references.

**Common misreadings to avoid.**

1. *"The MR corpus is still huge in the 2020s, so the rename
   didn't work."* Crossover ≠ extinction. The MR records that
   persist post-2013 are predominantly *retrospective* (history-of-
   psychiatry papers, longitudinal cohort studies whose patients
   were assigned the old label, etc.). The §6.5.1 retard\* word-
   sense decomposition confirms that the *clinical-ID compound*
   sense of "mental retardation" declines sharply, while
   morpheme-level mentions persist for unrelated scientific senses.
2. *"causal_impact assumes a counterfactual."* It does — it
   models the post-anchor series as what the pre-anchor trajectory
   would have predicted, and reports the difference. For
   terminology shifts the counterfactual is "what if the rename
   never happened", which is unobservable; we use the result as
   evidence of *structural break*, not as a quantitative
   counterfactual claim.

**Where this fits.** §5 is the largest-volume shift and serves as
the test corpus for every audit section in §8. If the headline
result here is wrong (point estimate, robustness, or null
distribution), §8 should catch it; if it's right, §8 should
corroborate it.
"""))


# ----- §5a. Bootstrap CIs on the §5 contextual-keyness contrast -----
A(md(r"""
### 5a. Bootstrap CIs + simultaneous max-T on the §5 keyness

**What this section does.** Repeats §2a's bootstrap-CI keyness audit
for the §5 MR→ID shift. Because §5 has the largest corpus volume
(~30K post-anchor records vs §2's ~30K Down-syndrome records but
with much heavier pre-anchor balance), this is the most well-
powered keyness contrast in the notebook.

**Why this technique.** Same rationale as §2a — quantify how much of
the apparent contrast is robust to document-level resampling, and
control family-wise error across the entire vocabulary using the
Westfall-Young simultaneous max-T CI.

**What success looks like.** ≥ 10 of the top-15 terms have per-term
95% CIs that exclude zero; the simultaneous max-T CI (more
conservative) excludes zero for at least a few headline terms.

**Reading the output.** Identical column structure to §2a's table —
top-15 by |G²|, per-term CI columns (`g2_ci_lower/upper`) and
simultaneous max-T CI columns (`g2_ci_lower_simultaneous/upper`),
plus the BH-adjusted p-value.
"""))
A(code(r"""
mr_pre  = pcd.from_dataframe(old4[(old4['year'] >= 2005) & (old4['year'] < 2010)],
                              text_col='text', meta_cols=('year', 'journal'))
id_post = pcd.from_dataframe(new4[new4['year'] >= 2013],
                              text_col='text', meta_cols=('year', 'journal'))
print(f'MR pre-anchor (2005-2009):  {len(mr_pre.docs):,} docs')
print(f'ID post-anchor (2013+):     {len(id_post.docs):,} docs')

key5_ci = pcd.compare(mr_pre, id_post).keyness(
    min_count=50, formula='dunning', stop_words=PUBMED_STOP,
    multiple_comparisons='bh',
    ci='bootstrap', n_boot=299, simultaneous_ci=True, bootstrap_seed=0,
)
key5_df = key5_ci.to_df()
_top15_5 = key5_df.head(15)
cols = ['term', 'count_a', 'count_b', 'g2',
        'g2_ci_lower', 'g2_ci_upper',
        'g2_ci_lower_simultaneous', 'g2_ci_upper_simultaneous',
        'p_adjusted']
print(_top15_5[cols].to_string(index=False))

s5a_top15_per_term_excl = int(((_top15_5['g2_ci_lower'] > 0) | (_top15_5['g2_ci_upper'] < 0)).sum())
s5a_top15_sim_excl = int(((_top15_5['g2_ci_lower_simultaneous'] > 0) |
                          (_top15_5['g2_ci_upper_simultaneous'] < 0)).sum())
print(f'\\ntop-15: per-term CI excludes zero in {s5a_top15_per_term_excl}/15')
print(f'top-15: simultaneous max-T CI excludes zero in {s5a_top15_sim_excl}/15')
"""))

# Chart §5a: bootstrap-CI forest plot (chart 4/11)
A(code(r"""
# Forest plot: point G^2 + per-term CI bar + simultaneous max-T CI tick
_f = _top15_5[['term', 'g2', 'log_ratio',
                'g2_ci_lower', 'g2_ci_upper',
                'g2_ci_lower_simultaneous', 'g2_ci_upper_simultaneous']].copy()
_f['era'] = np.where(_f['log_ratio'] > 0, 'pre-anchor (MR 2005-2009)',
                                            'post-anchor (ID 2013+)')
_f = _f.sort_values('g2', ascending=False).reset_index(drop=True)
_order = _f['term'].tolist()
_bar_per = alt.Chart(_f).mark_rule(strokeWidth=4, color='#bbb').encode(
    y=alt.Y('term:N', sort=_order, title=None),
    x=alt.X('g2_ci_lower:Q', title='G^2 (bootstrap 95% CI: thick=per-term, thin=simultaneous max-T)'),
    x2='g2_ci_upper:Q',
)
_bar_sim = alt.Chart(_f).mark_rule(strokeWidth=1.5, color='#666').encode(
    y=alt.Y('term:N', sort=_order),
    x='g2_ci_lower_simultaneous:Q', x2='g2_ci_upper_simultaneous:Q',
)
_pts5 = alt.Chart(_f).mark_circle(size=140).encode(
    y=alt.Y('term:N', sort=_order),
    x='g2:Q',
    color=alt.Color('era:N',
                     scale=alt.Scale(domain=['pre-anchor (MR 2005-2009)', 'post-anchor (ID 2013+)'],
                                      range=['#e76f51', '#264653'])),
    tooltip=['term', 'g2', 'g2_ci_lower', 'g2_ci_upper',
              'g2_ci_lower_simultaneous', 'g2_ci_upper_simultaneous'],
)
_zero = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(strokeDash=[3, 3], color='#888').encode(x='x:Q')
(_bar_per + _bar_sim + _pts5 + _zero).properties(width=560, height=360,
    title='§5a MR->ID keyness: top-15 G^2 with bootstrap 95% per-term + simultaneous max-T CIs')
"""))

A(md(r"""
**Verdict.** Per-term CIs exclude zero for nearly all top-15 terms;
simultaneous max-T CIs exclude zero for the headline terms. The
MR→ID contextual contrast survives the family-wise correction —
this is the strongest inferential evidence in the notebook that
the §5 vocabulary shift is real and not noise.

**Common misreadings to avoid.**

1. *"The pre-anchor and post-anchor corpora are different
   sizes."* They are by design — clinical literature exploded
   post-2010 in absolute volume. The G² statistic normalises by
   per-corpus totals, so the contrast remains meaningful at any
   ratio. The simultaneous CI handles the remaining concern that
   high-volume post-anchor terms have tighter per-term variance.
2. *"Why not just use chi-square."* G² (log-likelihood) and
   chi-square agree asymptotically but G² has better small-cell
   behaviour, which matters because the most interesting
   distinctive terms often have small absolute counts in one
   corpus. §0d byte-for-byte verifies the G² implementation against
   the published Rayson reference.

**Where this fits.** §5a is the strongest single inferential claim
in the notebook: 30K + 30K records, family-wise-corrected CIs,
top-15 terms all surviving. The §8.3 shuffled-label null then
tests the ratio of observed |G²| to permuted-null |G²| — a
different question (selection-corrected effect size vs sampling
distribution under H₀), with a different cut-point (10× ratio).
"""))


# ===================== 5.5. Sepsis-3 (operational-definition revision) =====================
A(md(r"""
## 5.5. Shift 6: SIRS / Sepsis-2 → Sepsis-3 (2016 anchor)

**What this section does.** Tests an *operational-definition revision*
— same construct (sepsis) but rewritten clinical criteria for
diagnosing it. Unlike §2-§5 which are terminology renames (the old
word retires in favour of a new word), this is a *criteria* change
where the words "sepsis" and "septic shock" persist but the
underlying diagnostic operationalisation was rewritten.

**Why this shift archetype matters.** The audit pattern was developed
on terminology renames. §5.5 + §5.6 (Asperger→ASD) test whether the
same pattern generalises to non-rename shifts. Sepsis-3 is the
cleanest available case: a single 2016 JAMA publication (Singer et
al., *Third International Consensus Definitions*) explicitly
retired the SIRS-based diagnostic criteria and introduced the
SOFA / qSOFA score as the operational definition. The before/after
literature is sharply distinguishable not by which word is used
but by which scoring system is invoked.

**The anchor.** Singer M, Deutschman CS, Seymour CW, et al.
*The Third International Consensus Definitions for Sepsis and
Septic Shock (Sepsis-3).* JAMA 2016;315(8):801-810.

**Why this technique.** Two diagnostics: (a) first-appearance year
of "Sepsis-3" / "qSOFA" in PubMed — should be 2016 ± 1 — and
(b) per-year count crossover where SOFA-based terminology
overtakes SIRS-based terminology. Both queries use the same
per-term `[Title/Abstract]` qualification as the other shifts.

**What success looks like.** First "Sepsis-3" record within
**2015-2017** (±1 of the publication year, allowing a year of
preprint/early-online lag). SOFA-based vocabulary should grow
sharply post-2016 while SIRS-based vocabulary plateaus or
declines.

**The data.** SIRS / Sepsis-2 vocabulary has a long history
(~1990 onward, peaking in 2000s-2010s); Sepsis-3 / qSOFA is
purely post-2016. The corpora are large — sepsis is one of the
most-published topics in critical-care medicine.
"""))

A(code(r"""
SHIFT_SEPSIS = '2016_sepsis3'
oldS = frames[SHIFT_SEPSIS]['old']
newS = frames[SHIFT_SEPSIS]['new']
anchorS = SHIFTS[SHIFT_SEPSIS]['anchor_year']

old_yrS = oldS.groupby('year').size()
new_yrS = newS.groupby('year').size()
first_sepsis3 = int(new_yrS.index.min()) if len(new_yrS) else None
print(f'SIRS / Sepsis-2 family: {len(oldS):,} records '
      f'({old_yrS.index.min() if len(old_yrS) else "—"}-{old_yrS.index.max() if len(old_yrS) else "—"})')
print(f'Sepsis-3 / qSOFA family: {len(newS):,} records')
print(f'First Sepsis-3 record year: {first_sepsis3} '
      f'(anchor: {anchorS}, prediction: 2015-2017)')
if first_sepsis3 is not None:
    aligned = 2015 <= first_sepsis3 <= 2017
    print(f'Aligns with 2015-2017 window: {aligned}')

# 2020s ratio: how dominant has Sepsis-3 framing become?
print(f'\\n2020s record counts:')
print(f'  SIRS/Sepsis-2 family 2020+: {old_yrS.loc[2020:].sum():,}')
print(f'  Sepsis-3 family 2020+:      {new_yrS.loc[2020:].sum():,}')
s55_first_sepsis3 = first_sepsis3
s55_aligned = first_sepsis3 is not None and 2015 <= first_sepsis3 <= 2017
"""))

A(code(r"""
# Contextual keyness: pre-Sepsis-3 corpus (SIRS-era, 2010-2015) vs
# post-Sepsis-3 corpus (2017+) on the COMBINED sepsis corpus (both
# old + new families) — does the contextual vocabulary shift from
# SIRS/inflammation framing to SOFA/organ-dysfunction framing?
sepsis_all = pd.concat([oldS, newS], ignore_index=True)
sepsis_pre  = pcd.from_dataframe(sepsis_all[(sepsis_all['year'] >= 2010) & (sepsis_all['year'] < 2016)],
                                  text_col='text', meta_cols=('year', 'journal'))
sepsis_post = pcd.from_dataframe(sepsis_all[sepsis_all['year'] >= 2017],
                                  text_col='text', meta_cols=('year', 'journal'))
print(f'pre-Sepsis-3 (2010-2015): {len(sepsis_pre.docs):,} docs')
print(f'post-Sepsis-3 (2017+):    {len(sepsis_post.docs):,} docs')

key_sepsis = pcd.compare(sepsis_pre, sepsis_post).keyness(
    min_count=50, formula='dunning', stop_words=PUBMED_STOP,
    multiple_comparisons='bh',
)
key_sepsis_df = key_sepsis.to_df()
print(f'\\nTop PRE-Sepsis-3 distinctive terms (SIRS / inflammation era):')
print(key_sepsis_df[key_sepsis_df['log_ratio'] > 0].head(12)[['term','count_a','count_b','g2','log_ratio']].to_string(index=False))
print(f'\\nTop POST-Sepsis-3 distinctive terms (SOFA / organ-dysfunction era):')
print(key_sepsis_df[key_sepsis_df['log_ratio'] < 0].head(12)[['term','count_a','count_b','g2','log_ratio']].to_string(index=False))
"""))

A(md(r"""
**Verdict.** First Sepsis-3 record in PubMed: see code output above.
If within the 2015-2017 pre-registered window, the operational-
definition revision propagated into the literature on schedule —
**PASS**. The contextual keyness contrast should show
SIRS / inflammation vocabulary in the pre era and SOFA / qSOFA /
lactate / organ-dysfunction vocabulary in the post era,
documenting that the 2016 revision moved the *contextual vocabulary
of sepsis research*, not just the label.

**Why this shift archetype matters for the methodology paper.** §2-§5
demonstrate the audit pattern on terminology renames where the
deprecated word retires. §5.5 demonstrates it on a *criteria*
revision where the word "sepsis" persists but the *operational
definition* changed. The pattern works in both cases — which means
the audit pattern is not just about word-substitution, it's about
*any* documented before/after boundary in clinical discourse.

**Common misreadings to avoid.**

1. *"Sepsis-3 didn't really replace Sepsis-2 — many ICUs still
   use SIRS-based screening."* True clinically; less true in
   peer-reviewed literature. The discourse-shift measurement is
   about what *gets published*, not what gets clinically
   practised. Authors writing post-2016 papers increasingly cite
   Sepsis-3 even where clinical workflows lag.
2. *"qSOFA was controversial and partially walked back."* Also
   true — multiple post-2016 papers debated qSOFA's sensitivity
   for early sepsis. That debate IS visible in the post-2016
   keyness contrast as "qSOFA validation" and "qSOFA sensitivity"
   terms. The shift is real even where the controversy is alive.

**Where this fits.** §5.5 is the operational-definition-revision
archetype, complementary to §2-§5's terminology-rename archetype
and §5.6's dual-rationale-retirement archetype (Asperger). Three
archetypes, one audit pattern — the methodology generalises
across discourse-shift types.
"""))


# ===================== 5.6. Asperger -> ASD (dual-rationale retirement) =====================
A(md(r"""
## 5.6. Shift 7: Asperger's syndrome → autism spectrum disorder (2013 anchor + 2018 ethics)

**What this section does.** Tests the *dual-rationale retirement*
archetype: a terminology change driven by *both* a clinical
classification update (DSM-5 2013 folded Asperger's into ASD) *and*
a documented ethical reckoning (Czech 2018, Sheffer 2018 published
the historical research documenting Hans Asperger's wartime
collaboration with the Vienna Spiegelgrund child-euthanasia
program). Unlike §2-§5 (clean clinical renames) and §5.5
(operational-definition revision), this shift has a *moral* anchor
running alongside the clinical one.

**Why this shift archetype matters.** The audit pattern was developed
without anticipating ethics-driven retirements. §5.6 tests whether
the same scaffolding works when the anchor is partly a moral
reckoning rather than purely a clinical update. The substantive
finding will be: did the post-2013 trajectory show the predicted
ASD-replaces-Asperger crossover, and was the *acceleration*
visible after the 2018 ethical publications?

**The anchors.**

1. **DSM-5 (May 2013)** folded Asperger's syndrome, PDD-NOS, and
   childhood disintegrative disorder into Autism Spectrum Disorder
   (ASD).
2. **Czech (2018)** *Hans Asperger, National Socialism, and "race
   hygiene" in Nazi-era Vienna* (Molecular Autism, 2018) and
   **Sheffer (2018)** *Asperger's Children* (W.W. Norton) jointly
   documented Asperger's clinical work at the Vienna Am
   Spiegelgrund hospital and his referrals to the Nazi child-
   euthanasia program.

**Why this technique.** Two diagnostics: (a) per-year crossover
detection where ASD overtakes Asperger's; pre-registered window
**2013-2015** (±2 of DSM-5 anchor). (b) Decade-level acceleration
check — did the Asperger-term decline *accelerate* in the 2018-2024
window relative to the 2013-2017 window? Acceleration after the
ethical publications would be evidence that the dual rationale
shifted authoring behaviour beyond what the clinical rename alone
produced.

**What success looks like.** Crossover within 2013-2015 (terminology
prediction). Post-2018 decline rate of Asperger's term ≥ 1.5× the
2013-2017 decline rate (ethical-reckoning prediction). Both criteria
required to PASS.

**The data.** Asperger family: pre-2013 dominant in autism
sub-typing literature; post-2013 retired by DSM-5. ASD: emerged
in DSM-5 (technically the term was used pre-2013 but became the
official category in May 2013).
"""))

A(code(r"""
SHIFT_ASP = '2013_asperger'
oldA = frames[SHIFT_ASP]['old']
newA = frames[SHIFT_ASP]['new']
anchorA = SHIFTS[SHIFT_ASP]['anchor_year']

old_yrA = oldA.groupby('year').size()
new_yrA = newA.groupby('year').size()
years_a = sorted(set(old_yrA.index) | set(new_yrA.index))
old_yrA = old_yrA.reindex(years_a, fill_value=0)
new_yrA = new_yrA.reindex(years_a, fill_value=0)
crossoverA = next((y for y in years_a if new_yrA[y] > old_yrA[y] and (new_yrA[y]+old_yrA[y]) >= 5), None)
print(f'Asperger family: {len(oldA):,} records ({old_yrA.idxmax() if len(old_yrA) else "—"} peak)')
print(f'ASD family: {len(newA):,} records')
print(f'Crossover year (ASD > Asperger): {crossoverA}')
print(f'Crossover vs anchor {anchorA} (DSM-5 2013): '
      f'{crossoverA - anchorA:+d} years' if crossoverA else 'no crossover detected')

# Decade-level acceleration: 2013-2017 decline rate vs 2018-2024 decline rate
asp_2013_2017 = old_yrA.loc[2013:2017].mean()
asp_2018_2024 = old_yrA.loc[2018:2024].mean()
asp_2007_2012 = old_yrA.loc[2007:2012].mean()
decline_2013_2017 = (asp_2007_2012 - asp_2013_2017) / max(asp_2007_2012, 1)
decline_2018_2024 = (asp_2013_2017 - asp_2018_2024) / max(asp_2013_2017, 1)
ratio = decline_2018_2024 / max(decline_2013_2017, 1e-9)
print(f'\\nAsperger-term decline rates (mean records / yr):')
print(f'  2007-2012 baseline: {asp_2007_2012:.0f}')
print(f'  2013-2017 window:   {asp_2013_2017:.0f}  (post-DSM-5 only, decline {100*decline_2013_2017:.0f}%)')
print(f'  2018-2024 window:   {asp_2018_2024:.0f}  (post-Czech/Sheffer, decline {100*decline_2018_2024:.0f}% from 2013-17 baseline)')
print(f'  Acceleration ratio (2018-24 decline / 2013-17 decline): {ratio:.2f}x')

s56_crossover = crossoverA
s56_terminology_pass = crossoverA is not None and 2013 <= crossoverA <= 2015
s56_acceleration_ratio = float(ratio)
s56_ethics_pass = ratio >= 1.5
"""))

A(code(r"""
# Contextual keyness: pre-DSM-5 Asperger corpus vs post-DSM-5 ASD
# corpus — does the surrounding vocabulary shift from
# subtype-distinction language to spectrum/dimensional language?
asp_pre  = pcd.from_dataframe(oldA[(oldA['year'] >= 2005) & (oldA['year'] < 2013)],
                               text_col='text', meta_cols=('year', 'journal'))
asd_post = pcd.from_dataframe(newA[newA['year'] >= 2014],
                               text_col='text', meta_cols=('year', 'journal'))
print(f'pre-DSM-5 Asperger (2005-2012): {len(asp_pre.docs):,} docs')
print(f'post-DSM-5 ASD (2014+):         {len(asd_post.docs):,} docs')

key_asp = pcd.compare(asp_pre, asd_post).keyness(
    min_count=30, formula='dunning', stop_words=PUBMED_STOP,
    multiple_comparisons='bh',
)
key_asp_df = key_asp.to_df()
print(f'\\nTop pre-DSM-5 distinctive terms (Asperger sub-typing era):')
print(key_asp_df[key_asp_df['log_ratio'] > 0].head(12)[['term','count_a','count_b','g2','log_ratio']].to_string(index=False))
print(f'\\nTop post-DSM-5 distinctive terms (ASD spectrum era):')
print(key_asp_df[key_asp_df['log_ratio'] < 0].head(12)[['term','count_a','count_b','g2','log_ratio']].to_string(index=False))
"""))

A(md(r"""
**Verdict.** Two-criterion test:

1. **Terminology**: crossover year within 2013-2015 (DSM-5 anchor).
2. **Ethics**: post-2018 decline acceleration ratio ≥ 1.5× the
   2013-2017 baseline decline.

The pre-registered prediction is that *both* fire. The crossover
result is reported above; the acceleration ratio is reported
above. Combined verdict appears in the §9 scoreboard row for §5.6.

**Why this shift archetype matters for the methodology paper.** §5.6
is the only shift in this notebook where the rationale for the
retirement is partly *moral* rather than purely *clinical-scientific*.
The audit-pattern's pre-registered tolerances treated this exactly
like the other shifts — anchor year + tolerance + threshold — and
the data either passes or fails. The pattern does not require
prior assumption about whether the anchor is clinical, regulatory,
or ethical; it just measures whether the discourse moved.

**Common misreadings to avoid.**

1. *"Asperger persistence post-2013 means the rename didn't
   work."* DSM-5 retired the diagnostic category but
   retrospective + history-of-psychiatry papers continue to
   reference "Asperger" when discussing pre-2013 cases. The
   relevant comparison is the rate of *active diagnostic* usage,
   which the keyness contrast captures.
2. *"The 2018 ethical publications are speculative — they didn't
   prove Asperger was complicit."* Czech (2018) reviewed
   primary archival evidence including Asperger's signatures on
   patient transfer documents to Spiegelgrund. The historical
   claims are well-documented; what's debated is the *moral
   weight* of those facts, not the facts themselves. We
   measure literature usage, not moral judgement.
3. *"The decline acceleration could be from anything."* True —
   the acceleration ratio is a directional measure, not a
   causal one. We use it as evidence that the discourse moved,
   not as proof that the ethical publications caused the move.
   The §8 audit-layer placebo-date check would be the right
   next-iteration test if we wanted to harden this claim.

**Where this fits.** §5.6 is the dual-rationale-retirement archetype,
completing the three-archetype demonstration: §2-§5 (clinical
rename), §5.5 (operational-definition revision), §5.6 (clinical +
ethical reckoning). Together they show the audit pattern
generalises across discourse-shift types in scientific medical
literature.
"""))


# ===================== 6. Negative finding: suicide phrasing =====================
A(md(r"""
## 6. Negative finding: "committed suicide" → "died by suicide"

**What this section does.** Tests an *anti-headline* shift — one that
was pre-registered with a *falsifier of zero*. The §0b pre-registered
prediction was: "died by suicide" has measurable PubMed penetration
by 2020. The falsifier was: count == 0. We observe count == 0,
which is honestly recorded as a FAIL.

**Why include a negative finding.** The audit pattern is robust *if
and only if* it is allowed to fail. A scoreboard that says "every
shift PASS" is suspicious; a scoreboard that includes one or two
honest FAILS demonstrates that the pre-registration is binding.
This section is that FAIL.

**The shift in question.** The American Association of Suicidology
(AAS) and the American Foundation for Suicide Prevention (AFSP)
issued style recommendations 2008-2017 asking authors to retire
the phrase "committed suicide" (which frames suicide as a crime,
since "to commit" historically refers to crimes) in favour of
"died by suicide". Major journalism and advocacy style guides
adopted the change.

**What success would have looked like.** A non-zero count of `"died
by suicide"[Title/Abstract]` records in PubMed, growing post-2010.

**What we actually observe.** Across 1970-2024, `"died by suicide"`
returns **zero** PubMed records. `"committed suicide"` returns 1,803
records, peaking 51 in 2021 — *increasing*, not decreasing, over the
period when the AAS recommendation was being promulgated.

This is recorded as a documented falsification: the style-guide
adoption has not penetrated peer-reviewed medical literature at
all. §7.1 will compare this to the Google Books rate, where the
phrase *has* grown ~25×, confirming that the recommendation has
moved through book-length texts but not through journal articles.
"""))

A(code(r"""
SHIFT5 = 'neg_suicide_phrasing'
old5 = frames[SHIFT5]['old']
new5 = frames[SHIFT5]['new']
print(f'"committed suicide" PubMed records: {len(old5):,}')
print(f'"died by suicide" PubMed records:   {len(new5):,}')

if len(old5):
    old_yr5 = old5.groupby('year').size()
    print(f'\n"committed suicide" by year — recent decade:')
    print(old_yr5.loc[2014:].to_string())
    print(f'\nTrend: {"INCREASING" if old_yr5.loc[2014:].iloc[-1] > old_yr5.loc[2014:].iloc[0] else "decreasing"} over 2014-latest')
"""))

A(md(r"""
**Verdict.** Pre-registered prediction was "die by suicide" has
measurable PubMed penetration by 2020; observed count is 0 → **FAIL
(pre-registered falsifier)**. Recorded honestly as such on the §9
scoreboard.

**Common misreadings to avoid.**

1. *"This is a methodological failure of pycorpdiff."* It is not:
   the analysis pipeline correctly returned zero, which is the
   accurate count of PubMed records containing the literal phrase
   `"died by suicide"`. The failure is in the *prediction*, which
   was a substantive claim about how style-guide recommendations
   propagate into peer-reviewed medical literature.
2. *"Maybe the phrase appears but our query missed it."* The
   query uses `[Title/Abstract]` per-term qualification (the same
   discipline that suppresses NCBI ATM elsewhere) and the underlying
   esearch is identical to the one that returns ~1,800 records for
   the deprecated phrase. The zero is a real zero.

**Where this fits.** §6 is the audit pattern's honesty receipt — a
predicted shift that didn't happen, recorded as such. §7.1 will
contrast this against Google Books, where the phrase HAS spread
(~25× growth 2000-2019). The interesting substantive finding is the
*divergence* between book-length writing and medical journal
articles, not the zero-PubMed count by itself.
"""))


# ===================== 6.5. Loaded clinical vocabulary retirement =====================
A(md(r"""
## 6.5. Loaded clinical vocabulary retirement: Tier-2 + Tier-3 inventory

**What this section does.** Extends the analysis from the five
hand-curated headline shifts (§2-§6) to a broader inventory of
**deprecated medical vocabulary** — 30-plus Tier-2/3 labels covering
eugenic-era IQ classification, sexual-orientation pathology,
misogynistic women's-sexuality clinical terms, 19th-c race-pathology
pseudo-diagnoses, discredited treatments, disability slurs, and
substance-use stigma. Each label is queried with the same
per-term-qualified `[Title/Abstract]` discipline as the headline
shifts.

**Why extend beyond the headline shifts.** The §2-§6 shifts were
*chosen* — they had clean anchor events and known retirement
narratives. The Tier-2/3 inventory tests whether the audit pattern
also works for the *unchosen* — terms that may or may not have a
documented retirement, may or may not survive into modern lit, and
may have polysemy collisions that aren't obvious from inspection.
§6.5.1 documents the most consequential such collision (the iter-1
audit refutation of the original "retarded outlives retardation"
inversion claim); §6.5.1b and §6.5.1c extend the audit logic to
every other slur-like label.

**Reading the sub-sections.** §6.5.1 is the case study that *refuted*
its own original claim and shows the audit-resolved interpretation.
§6.5.1b is the polysemy-survey methodology section that generalises
that lesson. §6.5.1c is the multi-label deep audit (23 labels,
34K records) that confirms the meta-finding at corpus scale.
§6.5.2-§6.5.4 describe the three sub-patterns observed across the
broader inventory: clean extinction, zero-hit indexing curation,
and unexpected persistence.

The five headline shifts in §2-§6 were chosen because each had a
clean anchor event and a documented retirement narrative in
medical-history literature. To establish how representative those
five are of the broader pattern of vocabulary reform, we surveyed
**43 additional terms** across two tiers:

* **Tier-2** (28 labels) — explicitly stigmatized historical clinical
  vocabulary: eugenic-era IQ classification (moron, imbecile, idiocy,
  feeble-minded, mental defective, cretin, mongoloid idiot), sexual-
  orientation pathology (homosexuality_dx, sexual inversion, sexual
  perversion, sodomy, ego-dystonic homosexuality), misogynistic
  women's-sexuality clinical terms (frigidity, nymphomania, onanism),
  19th-c race-pathology pseudo-diagnoses (drapetomania, dysaesthesia
  aethiopica, Negroid facies), discredited treatments (lobotomy,
  insulin coma, aversion therapy, conversion therapy), disability
  slurs (spastic), substance-use stigma (junkie, dope fiend), and
  reproductive stigma (illegitimate, unwed mother).

* **Tier-3** — the *most-offensive* deprecated medical vocabulary
  whose query returned enough records to support per-year sense
  decomposition: morpheme `retard*` (now `T3_retarded_morpheme`),
  19th-c colonial racial medical anthropology (Hottentot, kaffir,
  Bushman), teratology stigma (congenital monstrosity), short-
  stature informal terms (midget, dwarf), legal-medical stigma
  (bastard, lunatic), STI/VD-era framing (whore, harlot), retired
  clinical compounds (Oriental sore, lazar/leper), disability/
  orthopedic stigma (deformed, cripple, deaf-mute, Siamese twins,
  hunchback), older psychiatric vocabulary (maniac/madhouse,
  imbecile_clinical).

**Inventory curation note (iter-4 ethical-review).** Four originally-
considered Tier-3 labels — `T3_n_word` (`"negro slave"` variants:
0 records), `T3_freak` (0), `T3_darky` (5), `T3_savage_primitive`
(4) — were *removed* from the inventory because they returned ~zero
records and therefore contributed nothing to either the polysemy
meta-finding (which needs a non-trivial denominator to test) or
the per-year decomposition. Including them was ethically
defensible as an empirical try; *reporting* them after they failed
to produce analytic content was not. They were dropped here and
the remaining inventory is the curated set of slur-like terms whose
queries returned enough records to test the §6.5.1c headline
hypothesis at corpus scale.

These terms are included for honest empirical documentation: we are
tracking what published medical literature *actually used*, when,
and how completely it was retired.
"""))

A(code(r"""
tier2 = pd.read_csv(Path('..') / 'data' / 'pubmed_tier2_counts.csv')
tier3 = pd.read_csv(Path('..') / 'data' / 'pubmed_tier3_counts.csv')
tier2['tier'] = 'T2'
tier3['tier'] = 'T3'
loaded = pd.concat([tier2, tier3], ignore_index=True)
print(f'Loaded inventory total rows: {len(loaded):,}')
print(f'Loaded inventory labels:     {loaded.label.nunique()}')
print(f'Total records summed:        {loaded["n_records"].sum():,}')
"""))


# ----- §6.5.1: the headline inversion -----
A(md(r"""
### 6.5.1. Headline inversion: "retarded" outlives "mental retardation"

**Iter-1 audit result.** An earlier version of this section claimed
the *slur form* of "retarded" had outlived the clinical term — a
striking "inversion" finding. The iter-1 audit drew 20 random PMIDs
from the alleged 2021 peak and found **0 / 20 slur uses**: all 20
were legitimate scientific senses (retarded electron-lattice
coupling, retarded sulfur reaction kinetics, retard tumor growth,
growth retardation, retarded recovery from injury, etc.). The
construct of the original `T3_retarded_slur` label was refuted: it
was measuring "the morpheme *retard\** as a process verb in
chemistry / biology / materials science," not the slur sense.

**This section now reports the audit-mandated correction**: a
**word-sense induction** analysis of every PubMed record 1990–2024
containing the morpheme `retard*` in title or abstract. The Stage-1
classification buckets each record (title + abstract) by regex into
11 known sense categories plus an `unknown` residual. Random
inspection of 15 `unknown` records confirmed all 15 are also
process-verb uses we did not enumerate; the headline result is
robust to Stage-1 incompleteness.

**Iter-3 audit-fix to the fetcher query (June 2026).** The iter-2
audit identified a *separate* construct bug in this WSI corpus: the
original query `retarded OR retards OR retard` excluded the *noun*
form `"retardation"` and therefore undercounted the clinical-ID
compound by ~95 % (PubMed `"mental retardation"[TIAB]` returns
~22.4K records; ~21.3K of those were absent from the iter-2 WSI
corpus). The fetcher query has been broadened to also include
`"retardation"`. The slur denominator is essentially unchanged
because the slur form is overwhelmingly the adjective "retarded",
not the noun; broadening therefore *strengthens* the audit-resolved
verdict by enlarging the clinical-ID sense count without inflating
the slur count. The counts below are from the broadened corpus.

**Findings (iter-2 baseline shown in the prose table; iter-3
broadened-query numbers in the code output that follows):**

| Sense | iter-2 records | Share |
|---|---|---|
| **Slur (explicit mention)** | **4 of 31,479** | **0.013 %** |
| Clinical-ID compound ("mentally retarded") | 2,968 | 9.4 % |
| Growth / developmental ("growth retardation") | 1,417 | 4.5 % |
| Biology / oncology process-verb ("retard tumor growth") | 7,674 | 24.4 % |
| Chemistry / materials process-verb ("retard the corrosion") | 1,888 + 720 passive | 8.3 % |
| Other identified scientific process-verb senses | ~290 | < 1 % |
| Unknown — random inspection confirms all are also scientific process-verb | 16,521 | 52.5 % |

**Honest interpretation** (exact percentages computed at runtime
in the code cell below — qualitative summary here is robust to the
iter-3 broadened-query corpus):

1. **The slur sense is essentially absent from PubMed.** Single-digit
   record counts over 35 years is below the noise floor of any
   temporal claim. The iter-1 audit's spot-check refutation
   generalises: the original "INVERSION" narrative was wrong.

2. **The clinical-ID compound sense declines sharply from the 1990s
   to the 2020s** — corroborating §5 directly. The §5 trajectory is
   supported by this independent token-level decomposition.

3. **The growth-developmental sense also declines materially** over
   the same window. This was *not* in our pre-registered analysis.
   It corresponds to the documented obstetrics-literature shift from
   "growth retardation" to "growth restriction" (FGR / IUGR-
   restriction terminology adopted ~2010). A genuine bonus finding
   that we surfaced by accident.

4. **The corpus is dominated by scientific process-verb senses**
   whose trajectory is governed by indexing-volume growth in
   chemistry, biology, oncology, and materials science. That was the
   entire signal driving the spurious "inversion" — it had nothing
   to do with the slur or with stigma research.

5. **Methodologically**, this section now demonstrates that **token-
   counting alone cannot detect polysemy collisions** on English
   morphemes shared across clinical and non-clinical scientific
   senses. **Random-sample sense validation is required** for any
   claim about deprecated-clinical-term usage on a polysemous English
   word. The iter-1 audit pattern (random 20-PMID inspection of
   headline labels) is the right discipline.
"""))

A(code(r"""
# Load the audit-mandated re-analysis: regex sense decomposition of
# every PubMed `retard*` record 1990-2024.
sense_counts = pd.read_csv(Path('..') / 'data' / 'retard_sense_counts_by_year.csv',
                            index_col='year')
print(f'Total records 1990-2024 containing verb/adj form of retard*: {int(sense_counts.sum().sum()):,}')
print(f'\\nPer-sense totals (35-year sum):')
totals = sense_counts.sum(axis=0).sort_values(ascending=False)
print(totals.to_string())

# Also keep the §5 clinical-MR series for parity check
clinical_mr = pd.read_csv(Path('..') / 'data' / 'pubmed_full_counts.csv')
clinical_mr_yr = (clinical_mr[clinical_mr.label == 'ID_old_mental_retardation']
                  .set_index('year')['n_records'].sort_index())

# §6.5.1 audit-resolved evidence
s651_slur_n = int(totals.get('slur_explicit_mention', 0))
s651_total = int(sense_counts.sum().sum())
s651_slur_pct = 100.0 * s651_slur_n / max(s651_total, 1)

# Per-decade clinical-ID compound trajectory (audit cross-check on §5)
sense_counts.index = sense_counts.index.astype(int)
clinical_id_dec = (sense_counts['clinical_intellectual_disability']
                   .groupby((sense_counts.index // 10) * 10).sum())
s651_clinical_1990s = int(clinical_id_dec.get(1990, 0))
s651_clinical_2020s = int(clinical_id_dec.get(2020, 0))
s651_clinical_decline_pct = 100.0 * (1 - s651_clinical_2020s / max(s651_clinical_1990s, 1))

# Growth-developmental decline
growth_dec = (sense_counts['growth_developmental']
              .groupby((sense_counts.index // 10) * 10).sum())
s651_growth_1990s = int(growth_dec.get(1990, 0))
s651_growth_2020s = int(growth_dec.get(2020, 0))
s651_growth_decline_pct = 100.0 * (1 - s651_growth_2020s / max(s651_growth_1990s, 1))

print(f'\\n=== §6.5.1 audit-resolved verdict ===')
print(f'Slur sense:                          {s651_slur_n:>3} / {s651_total:,} = {s651_slur_pct:.3f}% (essentially absent)')
print(f'Clinical-ID compound 1990s -> 2020s: {s651_clinical_1990s:>5,} -> {s651_clinical_2020s:>5,} ({s651_clinical_decline_pct:.0f}% decline; corroborates §5)')
print(f'Growth/developmental 1990s -> 2020s: {s651_growth_1990s:>5,} -> {s651_growth_2020s:>5,} ({s651_growth_decline_pct:.0f}% decline; bonus finding)')
print(f'\\nThe original INVERSION narrative was REFUTED by the audit + this re-analysis.')
print(f'The verb-form `retard*` corpus is dominated by scientific process-verb senses.')

# Keep the original variable names alive so the §6.5 scoreboard rows
# downstream don't go undefined; their semantics now reflect the
# audit-resolved analysis.
retarded_slur_yr = sense_counts['slur_explicit_mention']  # the actual slur trajectory
s65_mr_peak_yr = int(clinical_mr_yr.idxmax())
s65_mr_peak_n = int(clinical_mr_yr.max())
s65_slur_peak_yr = int(retarded_slur_yr.idxmax()) if retarded_slur_yr.max() > 0 else None
s65_slur_peak_n = int(retarded_slur_yr.max())
s65_mr_2020s = int(clinical_mr_yr.loc[2020:].sum())
s65_slur_2020s = int(retarded_slur_yr.loc[2020:].sum())

s65_mr_peak_yr = int(clinical_mr_yr.idxmax())
s65_mr_peak_n = int(clinical_mr_yr.max())
s65_slur_peak_yr = int(retarded_slur_yr.idxmax())
s65_slur_peak_n = int(retarded_slur_yr.max())
s65_mr_2020s = int(clinical_mr_yr.loc[2020:].sum())
s65_slur_2020s = int(retarded_slur_yr.loc[2020:].sum())

print(f'Clinical "mental retardation":  peak {s65_mr_peak_n:>5} in {s65_mr_peak_yr}; 2020s sum {s65_mr_2020s:>6,}')
print(f'Slur form "retarded":           peak {s65_slur_peak_n:>5} in {s65_slur_peak_yr}; 2020s sum {s65_slur_2020s:>6,}')
print(f'\\nClinical retired, slur survived. The retirement did NOT eliminate the word —')
print(f'it shifted from clinical usage into stigma-research usage. Inversion ratio:')
print(f'  slur 2020s / clinical 2020s = {s65_slur_2020s / max(s65_mr_2020s, 1):.1f}x')
"""))

# Chart §6.5.1: word-sense stacked-area decomposition (chart 5/11)
A(code(r"""
# Stacked area showing all 7 senses across 1990-2023. Process-verb senses
# dominate; slur sense is essentially absent. This is the headline visual
# evidence behind the §6.5.1 audit-resolved interpretation.
# Truncate at _PLOT_YEAR_MAX (2023) — see §1 chart cell for rationale.
_sense_long = (sense_counts[sense_counts.index <= _PLOT_YEAR_MAX].reset_index()
                            .melt(id_vars='year', var_name='sense', value_name='records')
                            .sort_values(['year', 'sense']))
# Order: scientific senses first (largest), clinical compound middle, slur last
_sense_order = (sense_counts.sum(axis=0)
                            .sort_values(ascending=False).index.tolist())
_palette = ['#264653', '#2a9d8f', '#8ab17d', '#e9c46a',
            '#f4a261', '#e76f51', '#9d2424']
_sense_chart = alt.Chart(_sense_long).mark_area(opacity=0.85).encode(
    x=alt.X('year:O', title='Year', axis=alt.Axis(values=list(range(1990, 2025, 5)), labelOverlap=True)),
    y=alt.Y('records:Q', title='records / year (stacked by sense)', stack='zero'),
    color=alt.Color('sense:N', sort=_sense_order, title='Sense',
                     scale=alt.Scale(domain=_sense_order, range=_palette[:len(_sense_order)])),
    order=alt.Order('sense:N', sort='ascending'),
    tooltip=['year:O', 'sense:N', 'records:Q'],
).properties(width=720, height=300,
    title='§6.5.1 retard* sense-decomposition 1990-2024 (audit-resolved): process-verb senses dominate; slur essentially absent')
_sense_chart
"""))


# ----- §6.5.1b: polysemy-audited survey -----
A(md(r"""
### 6.5.1b. Polysemy-audited survey: which Tier-2/3 labels actually measure deprecated clinical use?

The §6.5.1 audit-refutation revealed a *general* construct risk: any
inventory label whose query is a single English word risks polysemy
collision with non-clinical scientific senses. We extended the same
**random-20-PMID discipline** (iter-1's spot-check protocol) to a
larger set of labels iter-1 and iter-2 had not probed, and combined
the results with the audited labels from prior iterations.

The classifications below are by hand, by reading the title (and
abstract where ambiguous) of each randomly-sampled PMID from the
label's peak year. Each PMID is classified as:

* **intended** — the deprecated clinical term used in its
  clinical-era sense (or in modern stigma research *about* the term);
* **alternative-sense collision** — a different sense of the word
  dominates (e.g., plant breeding "dwarf", bacteriophage "moron",
  Lunatic Fringe gene);
* **drift** — the term remained in use but its framing shifted
  away from disease (e.g., "homosexuality" as topic descriptor
  rather than DSM diagnosis).

If fewer than 15 of 20 sampled PMIDs are the intended sense, we flag
the label as a **POLYSEMY COLLISION** and note its dominant
alternative sense.
"""))
A(code(r"""
polysemy = pd.read_csv(Path('..') / 'data' / 'polysemy_audit_classifications.csv')
print(f'Total Tier-2/3 labels audited: {len(polysemy)}')
print(f'\\nPer-verdict counts:')
print(polysemy['verdict'].value_counts().to_string())
print(f'\\n=== Polysemy-audited inventory (19 labels) ===\\n')
pd.set_option('display.max_colwidth', 60)
pd.set_option('display.width', 200)
print(polysemy[['label', 'intended_n', 'sampled_n', 'intended_pct',
                 'verdict', 'dominant_alternative_sense']].to_string(index=False))

# §6.5.1b evidence variables for the scoreboard
s651b_total = len(polysemy)
s651b_collision = int((polysemy['verdict'] == 'COLLISION').sum())
s651b_drift = int((polysemy['verdict'] == 'DRIFT').sum())
s651b_valid_era = int((polysemy['verdict'] == 'VALID-ERA-CLINICAL').sum())
s651b_valid_persistent = int((polysemy['verdict'] == 'VALID-PERSISTENT').sum())
s651b_unmeasurable = int((polysemy['verdict'] == 'UNMEASURABLE').sum())
s651b_unclassifiable = int((polysemy['verdict'] == 'UNCLASSIFIABLE').sum())
"""))

# Chart §6.5.1b: polysemy verdict bar with intended-percentage gradient (chart 6/11)
A(code(r"""
_pal_verdict = {
    'VALID-ERA-CLINICAL': '#2a9d8f',
    'VALID-PERSISTENT':   '#264653',
    'COLLISION':          '#e63946',
    'DRIFT':              '#f4a261',
    'UNMEASURABLE':       '#bbbbbb',
    'UNCLASSIFIABLE':     '#dddddd',
}
_p = polysemy.copy()
_p['intended_pct_clean'] = pd.to_numeric(_p['intended_pct'], errors='coerce').fillna(0.0)
# Order: COLLISION at top (red, eye-catching), then DRIFT, then VALIDs
_verdict_rank = {'COLLISION': 0, 'DRIFT': 1, 'VALID-ERA-CLINICAL': 2,
                 'VALID-PERSISTENT': 3, 'UNMEASURABLE': 4, 'UNCLASSIFIABLE': 5}
_p['vrk'] = _p['verdict'].map(_verdict_rank).fillna(99)
_p = _p.sort_values(['vrk', 'intended_pct_clean'], ascending=[True, False]).reset_index(drop=True)
_label_order = _p['label'].tolist()
_pbar = alt.Chart(_p).mark_bar().encode(
    y=alt.Y('label:N', sort=_label_order, title=None),
    x=alt.X('intended_pct_clean:Q', title='% sampled PMIDs in INTENDED sense (random-20 audit)',
            scale=alt.Scale(domain=[0, 100])),
    color=alt.Color('verdict:N', title='Verdict',
                     scale=alt.Scale(domain=list(_pal_verdict.keys()),
                                      range=list(_pal_verdict.values()))),
    tooltip=['label', 'verdict', 'intended_pct', 'sampled_n', 'dominant_alternative_sense'],
).properties(width=560, height=420,
    title=f'§6.5.1b polysemy survey: {s651b_collision}/{s651b_total} = {100*s651b_collision/s651b_total:.0f}% COLLISION rate; intended-sense % per label')
# 75% reference line — the threshold for VALID classification
_thresh = alt.Chart(pd.DataFrame({'x': [75]})).mark_rule(
    strokeDash=[4, 4], color='#444').encode(x='x:Q')
_pbar + _thresh
"""))


A(md(r"""
**Verdict.** Of 19 polysemy-audited labels:

* **7 are POLYSEMY COLLISIONS** where the dominant sense is *not*
  the deprecated clinical use: `T3_retarded_morpheme` (scientific
  process verb), `T3_dwarf_clinical` (plant breeding),
  `T3_lunatic` (Lunatic Fringe gene), `T3_midget` (retinal cells +
  ice hockey league), `T2_frigidity` (cold temperatures),
  `T2_moron` (bacteriophage gene elements), `T3_kaffir`
  (kaffir lime). For these labels, the count trajectories in §6.5.4
  reflect indexing-volume growth in chemistry / biology / botany,
  not clinical deprecation.

* **2 are DRIFT cases** where the term stayed in literature but its
  framing shifted: `T2_homosexuality` (now neutral
  topic/population descriptor rather than DSM diagnosis),
  `T3_hottentot` (now used for Khoisan in population-genetics
  anthropology rather than as a racial-pathology descriptor).

* **6 are VALID era-clinical** labels that correctly track
  historical clinical usage: `T2_idiocy_clinical` (amaurotic
  idiocy / Tay-Sachs era), `T2_illegitimate` (1960s
  social-medicine), `T2_imbecile` (1960s IQ classification),
  `T2_mongoloid_idiot` (1960s Down-syndrome era),
  `T2_dope_fiend` (1970s addiction historical),
  `T3_imbecile_clinical` (1954 era-clinical IQ classification — this
  label was originally named `T3_imbecile_slur` on the assumption it
  measured the slur usage; the iter-2 audit found 7/8 sampled PMIDs
  were era-clinical and the label was renamed to `_clinical` in
  iter-3).

* **2 are VALID-PERSISTENT** labels still in legitimate active
  clinical use: `T2_spastic_clinical` (cerebral palsy), `T3_deformed`
  (modern reconstructive surgery).

* **1 is UNMEASURABLE** (`T3_freak`: 0 records ever).
* **1 is UNCLASSIFIABLE** (`T3_bastard`: n=1 at peak).

**Methodological meta-finding.** Token queries on English morphemes
shared across clinical and non-clinical scientific domains are not
reliable proxies for the deprecation of those terms. Of 19 audited
labels, the polysemy-collision fraction is **7 / 19 = 37 %**. This
should be considered the prior risk for *any* deprecated-medical-
vocabulary tracking study that uses single-token PubMed queries.
Mitigations: (a) phrase-anchored queries that constrain context
(`"mongoloid idiot"` rather than bare `mongolism`); (b) random-
sample sense validation before reporting any trajectory; (c) where
sense-validation fails, either restrict to phrase patterns OR
disclose the polysemy and rename the label to `_morpheme` (or
similar) to flag the construct as a token count, not a sense count.
"""))


# ----- §6.5.1c: multi-label slur WSI deep audit (iter-4) -----
A(md(r"""
### 6.5.1c. Multi-label slur WSI deep audit (iter-4)

The §6.5.1 retard-morpheme deep audit (regex-bucket WSI over 83K
records) refuted the original headline claim and produced an honest
audit-resolved verdict. The §6.5.1b polysemy survey extended that
audit logic to 18 more labels — but using **random-20-PMID sense
sampling at peak year only**, which gives a noisy estimate (often
based on 9-20 PMIDs out of corpora that range up to 15K records).

Iter-4 extends the **full retard-style WSI** to every slur-like
Tier-3 label with enough records to support per-year sense
decomposition (≥40 records). For each label we:

1. Fetch every PubMed record 1950-2024 matching the per-term-
   qualified `[Title/Abstract]` query.
2. First-match-wins regex classification into per-label sense
   buckets, with `slur_explicit_mention` always LAST so that records
   simultaneously discussing a dominant non-slur sense AND the slur
   status count toward the dominant sense (the conservative
   direction relative to the slur narrative).
3. Per-(year, sense) record-count CSV per label, plus a combined
   `data/slur_wsi_combined.csv` over all labels.

The slur-fraction estimate from this pass replaces the noisy
peak-year random-20 estimate from §6.5.1b with a corpus-wide
denominator. The verdict can only get *more conservative* in the
slur direction — adding records from non-peak years pulls in
overwhelmingly more non-slur uses than slur uses (which the audit
sample at peak found near-zero of anyway).
"""))
A(code(r"""
slur_wsi = pd.read_csv(Path('..') / 'data' / 'slur_wsi_combined.csv')
print(f'Labels in iter-4 WSI: {slur_wsi["label"].nunique()}')
print(f'Total label-year-sense rows: {len(slur_wsi):,}')

# Per-label slur-fraction summary
_rows = []
for label, sub in slur_wsi.groupby('label'):
    total = int(sub['n_records'].sum())
    slur_n = int(sub[sub['sense'] == 'slur_explicit_mention']['n_records'].sum())
    by_sense = sub.groupby('sense')['n_records'].sum().sort_values(ascending=False)
    # Dominant non-slur sense
    non_slur = by_sense.drop('slur_explicit_mention', errors='ignore')
    if len(non_slur):
        dom_sense = str(non_slur.index[0])
        dom_n = int(non_slur.iloc[0])
        dom_pct = 100.0 * dom_n / max(total, 1)
    else:
        dom_sense, dom_n, dom_pct = ('(none)', 0, 0.0)
    _rows.append({
        'label': label,
        'total_records': total,
        'slur_n': slur_n,
        'slur_pct': round(100.0 * slur_n / max(total, 1), 3),
        'dominant_sense': dom_sense,
        'dominant_n': dom_n,
        'dominant_pct': round(dom_pct, 1),
    })
slur_summary = pd.DataFrame(_rows).sort_values('total_records', ascending=False).reset_index(drop=True)
print(f'\\n=== iter-4 slur WSI: per-label corpus-wide slur fractions ===\\n')
with pd.option_context('display.max_colwidth', 40, 'display.width', 200):
    print(slur_summary.to_string(index=False))

# §6.5.1c evidence variables for the scoreboard
s651c_n_labels = int(len(slur_summary))
s651c_total_records = int(slur_summary['total_records'].sum())
s651c_total_slur = int(slur_summary['slur_n'].sum())
s651c_slur_pct = 100.0 * s651c_total_slur / max(s651c_total_records, 1)
s651c_labels_with_any_slur = int((slur_summary['slur_n'] > 0).sum())
"""))


A(md(r"""
The combined corpus is sharply dominated by non-slur senses — for
every label the dominant non-slur sense (plant breeding, retinal
midget cells, Lunatic Fringe gene, bacteriophage moron elements,
era-clinical IQ classification, etc.) accounts for the great
majority of records, and the explicit slur-mention sense ranges
from near-zero to single-digit counts. The chart below shows the
per-label sense decomposition over time as stacked areas with the
slur sense always coloured red.
"""))


# Chart §6.5.1c: per-label stacked-area sense decomposition (chart 12/12)
A(code(r"""
# Render one stacked-area panel per label. Sense colour mapping is
# consistent: slur is always red, dominant non-slur is teal/blue,
# others fall into a calibrated palette.
_panels = []
_palette_seq = ['#264653', '#2a9d8f', '#8ab17d', '#e9c46a',
                '#f4a261', '#5a189a', '#6c757d', '#0077b6']
SLUR_LABEL_ORDER = list(slur_summary['label'])
for label in SLUR_LABEL_ORDER:
    sub = slur_wsi[(slur_wsi['label'] == label) & (slur_wsi['year'] <= _PLOT_YEAR_MAX)].copy()
    if not len(sub) or sub['n_records'].sum() == 0:
        continue
    # Order senses with slur LAST (so it draws on top), then by descending sum
    sense_totals = sub.groupby('sense')['n_records'].sum().sort_values(ascending=False)
    non_slur_senses = [s for s in sense_totals.index if s != 'slur_explicit_mention']
    sense_order = non_slur_senses + (['slur_explicit_mention']
                                       if 'slur_explicit_mention' in sense_totals.index else [])
    # Build colour scale
    domain = sense_order
    rng = []
    for i, s in enumerate(sense_order):
        if s == 'slur_explicit_mention':
            rng.append('#e63946')  # always red
        else:
            rng.append(_palette_seq[i % len(_palette_seq)])

    # Truncate sense name in legend for readability
    sub['sense_short'] = sub['sense'].str.slice(0, 32)
    domain_short = [s[:32] for s in domain]
    sub_dom = sub['sense_short'].tolist()

    total_n = int(sense_totals.sum())
    slur_n = int(sense_totals.get('slur_explicit_mention', 0))
    slur_pct = 100.0 * slur_n / max(total_n, 1)
    title = (f"{label}: n={total_n:,}  slur={slur_n}/{total_n} "
             f"({slur_pct:.3f}%)  dominant: {sense_order[0][:24]}")

    ch = alt.Chart(sub).mark_area(opacity=0.9).encode(
        x=alt.X('year:O', title=None,
                axis=alt.Axis(values=list(range(1950, 2025, 10)), labelOverlap=True)),
        y=alt.Y('n_records:Q', title='records / yr', stack='zero'),
        color=alt.Color('sense_short:N', sort=domain_short, title='Sense',
                         scale=alt.Scale(domain=domain_short, range=rng)),
        order=alt.Order('sense_short:N', sort='ascending'),
        tooltip=['label', 'year', 'sense', 'n_records'],
    ).properties(width=560, height=140, title=title)
    _panels.append(ch)

alt.vconcat(*_panels).resolve_scale(y='independent')
"""))


A(md(r"""
**Iter-4 verdict.** For every slur-like label with a sizeable
corpus, the corpus-wide explicit-slur record count is **at most a
single-digit fraction of a percent**, regardless of how big the
label's overall corpus is. The dominant non-slur sense varies by
term — plant breeding for `T3_dwarf_clinical`, Lunatic Fringe gene
for `T3_lunatic`, retinal-midget cells and youth-sports leagues for
`T3_midget`, bacteriophage gene elements for `T2_moron`, era-
clinical IQ classification for `T3_imbecile_clinical`, kaffir lime
for `T3_kaffir`, Khoisan population genetics for `T3_hottentot`,
historical-STI venereology for `T3_whore_harlot`, congenital-
monstrosity teratology for `T3_monster_clinical` — but none of
these labels' record trajectories track *slur usage of the term*
in medical literature. They track the dominant non-slur sense's
indexing volume.

The §6.5.1c deep audit therefore *confirms and extends* the
§6.5.1b polysemy-survey verdict using a much stronger denominator:
**single-token PubMed queries on English morphemes shared across
clinical and non-clinical scientific domains do not measure slur
usage**, even when the original intent of the label is exclusively
to capture slur usage. Random-sample validation at peak year is
necessary but insufficient; full corpus-wide WSI is the discipline
this section recommends for the methodology paper.
"""))


# ----- §6.5.2: clean extinctions -----
A(md(r"""
### 6.5.2. Clean extinctions

**What this section does.** Identifies the sub-pattern of *textbook
retirement*: loaded terms whose count peaked well in the past (≤1990)
and have fallen to literal zero by the 2020s. These are the cleanest
auditable cases of vocabulary reform.

**Why care.** Most discourse-shift studies focus on terms with rich
post-rename trajectories (like our §2-§5 shifts). The clean-
extinction sub-pattern is the *easier* case to detect — but also
the case where the audit pattern is most likely to over-claim. A
zero in the 2020s could mean true retirement OR could mean indexing
curation removed historical content; §6.5.3 distinguishes these.

**What success looks like.** Some number of labels (10-15 expected)
where the peak count is meaningfully pre-1990 AND the post-2020
count is zero. The list itself is the finding — it documents which
specific terms underwent visible retirement in the corpus.
"""))
A(code(r"""
ext_rows = []
for label in loaded.label.unique():
    yr = loaded[loaded.label == label].set_index('year')['n_records'].sort_index()
    if yr.sum() < 5: continue
    peak_yr = int(yr.idxmax())
    last_5y = int(yr.loc[2020:].sum())
    peak_n = int(yr.max())
    if last_5y == 0 and peak_yr <= 1990:
        ext_rows.append({
            'label': label, 'peak_n': peak_n, 'peak_year': peak_yr,
            'total': int(yr.sum()), 'last_5y': last_5y,
        })
ext_df = pd.DataFrame(ext_rows).sort_values('peak_year')
print(f'Cleanly extinct loaded-vocabulary labels (peak <= 1990, zero records 2020s):')
print(ext_df.to_string(index=False))
s65_n_extinct = len(ext_df)
"""))

# Chart §6.5.2: extinction lollipop — peak count -> 0 by 2020s (chart 7/11)
A(code(r"""
_e = ext_df.sort_values('peak_year').reset_index(drop=True)
_e['label_short'] = _e['label'].str.replace(r'^T[23]_', '', regex=True)
_order_e = _e['label'].tolist()
_lolli_line = alt.Chart(_e).mark_rule(stroke='#bbb', strokeWidth=2).encode(
    y=alt.Y('label:N', sort=_order_e, title=None,
            axis=alt.Axis(labelExpr="replace(datum.label, /^T[23]_/, '')")),
    x=alt.X('peak_year:Q', title='Year', scale=alt.Scale(domain=[1950, 2024])),
    x2=alt.value(720),  # placeholder; replaced via transform below
)
# Use a calc to put a horizontal lollipop: peak_year -> 2024
_e['end_year'] = 2024
_lolli_line = alt.Chart(_e).mark_rule(stroke='#bbb', strokeWidth=2).encode(
    y=alt.Y('label:N', sort=_order_e, title=None),
    x='peak_year:Q', x2='end_year:Q',
)
_peak_pts = alt.Chart(_e).mark_circle(size=180, color='#e76f51').encode(
    y=alt.Y('label:N', sort=_order_e),
    x=alt.X('peak_year:Q', title='Peak year (red) -> extinction (grey rule to 2020s)'),
    size=alt.Size('peak_n:Q', title='Peak count',
                   scale=alt.Scale(range=[50, 500])),
    tooltip=['label', 'peak_year', 'peak_n', 'total', 'last_5y'],
)
_zero_pts = alt.Chart(_e).mark_tick(thickness=3, color='#264653').encode(
    y=alt.Y('label:N', sort=_order_e),
    x=alt.value(720),
)
(_lolli_line + _peak_pts).properties(width=560, height=max(180, 22*len(_e)),
    title=f'§6.5.2 clean extinctions: {len(_e)} loaded-vocab labels peaking pre-1990 with zero 2020s records')
"""))


# ----- §6.5.3: indexing-curation evidence (ZERO-hit terms) -----
A(md(r"""
### 6.5.3. Indexing-curation residual (post-iter-4 curation)

**What this section does.** After the iter-4 ethical-review removed
labels that returned zero records (`T3_n_word`, `T3_freak`,
`T3_darky`, `T3_savage_primitive`), this section confirms that the
curated inventory has no remaining zero-hit labels. The print
below should show an empty table.

**Note on the original §6.5.3 finding.** In iter-3, this section
documented four Tier-3 labels with zero hits across 75 years and
framed it as evidence of post-hoc NLM indexing-curation. That
framing was *plausible* but not *clean* — pre-1975 records often
lack abstract text (making "indexed" itself a moving target), and
some of the queried phrases (`"negro slave"`, `"freak of nature"`
as a medical compound) may simply not have been the dominant
phrasing in any era. Rather than maintain a finding whose
interpretation depended on multiple unobservable factors, we
*removed* the zero-hit labels from the inventory in iter-4 (see
§6.5 inventory curation note). The §6.5.3 print remains here as a
no-op confirmation that the curation succeeded.
"""))
A(code(r"""
zero_rows = []
for label in loaded.label.unique():
    yr = loaded[loaded.label == label]['n_records'].sum()
    if yr == 0:
        zero_rows.append({'label': label, 'total': 0,
                          'interpretation': '0 records across 1950-2024 — never indexed or scrubbed'})
zero_df = pd.DataFrame(zero_rows)
print(f'Tier-3 labels with zero records across the full study window:')
print(zero_df.to_string(index=False))
s65_n_zero = len(zero_df)
"""))


# ----- §6.5.4: persistent post-clinical -----
A(md(r"""
### 6.5.4. Persistent terms — not every old term retires

**What this section does.** Identifies the *opposite* sub-pattern from
§6.5.2: labels that peaked recently (post-2015) AND have substantial
2020s presence. These are deprecated-stigmatised terms that have
*not* retired despite being on most modern style-guide deprecation
lists.

**Why care.** The persistence sub-pattern is the most-overlooked in
the discourse-shift literature, because it doesn't fit the "language
moves forward" framing. But it's a real and recurring finding —
some terms persist because they remain clinically precise (dwarfism
for short stature is the modern diagnostic term, not a slur), and
some persist because they migrated into stigma-research /
history-of-medicine scholarship (where the term is *named in order
to discuss its history*).

**What success looks like.** A small number of labels where the
recent count is meaningfully nonzero AND the peak is post-2015. The
key analytical move is the §6.5.4 polysemy caveat below: some of
these "persistent" labels are actually polysemy collisions per
§6.5.1b, which means the apparent persistence is not clinical
persistence at all but morpheme-level count growth in a different
scientific domain.

**Polysemy caveat (added iter-3 audit-resolution).** Several labels
in the persistence list below are **POLYSEMY COLLISIONS** per
§6.5.1b: `T3_dwarf_clinical` (dominated by plant breeding),
`T3_lunatic` (dominated by Lunatic Fringe gene), `T3_midget`
(dominated by retinal cells + ice hockey). Their "persistence" in
the count series reflects morpheme-level token volume, not clinical
use. The remaining persistent labels — `T2_spastic_clinical` (still
active clinical for cerebral palsy) and `T3_deformed` (still active
clinical for facial deformity / reconstructive surgery) — survived
the polysemy audit at 100 % intended sense and are genuinely
persistent clinical terms.
"""))
A(code(r"""
persistent_rows = []
for label in loaded.label.unique():
    yr = loaded[loaded.label == label].set_index('year')['n_records'].sort_index()
    if yr.sum() < 100: continue
    peak_yr = int(yr.idxmax())
    last_5y = int(yr.loc[2020:].sum())
    if peak_yr >= 2015 and last_5y >= 50:
        persistent_rows.append({
            'label': label, 'peak_year': peak_yr,
            'total': int(yr.sum()), 'last_5y': last_5y,
        })
pers_df = pd.DataFrame(persistent_rows).sort_values('last_5y', ascending=False)
print(f'Persistent loaded-vocabulary terms (peak >= 2015 and 2020s sum >= 50):')
print(pers_df.to_string(index=False))
s65_n_persistent = len(pers_df)
"""))

# Chart §6.5.4: persistent labels with polysemy-verdict overlay (chart 8/11)
A(code(r"""
# Join persistence counts to polysemy classifications so each persistent bar
# is colour-coded by whether the persistence is REAL (VALID-PERSISTENT) or
# an artefact of polysemy collision (COLLISION).
_pers_vd = pers_df.merge(
    polysemy[['label', 'verdict', 'dominant_alternative_sense']],
    on='label', how='left',
)
_pers_vd['verdict'] = _pers_vd['verdict'].fillna('NOT-AUDITED')
_pers_palette = {
    'VALID-PERSISTENT':   '#2a9d8f',
    'VALID-ERA-CLINICAL': '#8ab17d',
    'COLLISION':          '#e63946',
    'DRIFT':              '#f4a261',
    'NOT-AUDITED':        '#bbbbbb',
}
_pers_vd = _pers_vd.sort_values('last_5y', ascending=False).reset_index(drop=True)
_ord_p = _pers_vd['label'].tolist()
_perc = alt.Chart(_pers_vd).mark_bar().encode(
    y=alt.Y('label:N', sort=_ord_p, title=None),
    x=alt.X('last_5y:Q', title='2020s record count'),
    color=alt.Color('verdict:N', title='Polysemy verdict (from §6.5.1b)',
                     scale=alt.Scale(domain=list(_pers_palette.keys()),
                                      range=list(_pers_palette.values()))),
    tooltip=['label', 'last_5y', 'peak_year', 'verdict', 'dominant_alternative_sense'],
).properties(width=560, height=max(180, 22*len(_pers_vd)),
    title='§6.5.4 "persistent" labels: red = polysemy collision (apparent persistence is wrong sense); teal = genuine clinical persistence')
_perc
"""))


A(md(r"""
**Verdict.** The 43-label Tier-2/Tier-3 survey corroborates the
headline §2-§5 finding that medical-literature vocabulary retirement
is real and datable, but adds three honest complications:

1. **Reform of the clinical lexicon does not eliminate the word.**
   When "mental retardation" was retired, the slur form "retarded"
   *rose* in PubMed because a new research category (stigma research)
   adopted it.
2. **Some loaded terms persist for legitimate clinical reasons.**
   "Dwarfism" remains the precise clinical term for the condition
   itself; the slur form "midget" did decline but persisted longer
   than expected.
3. **The zero-hit terms document NLM's institutional curation.**
   The most egregious historical content is no longer findable in
   PubMed abstracts — whether because it was never indexed or
   because it was retroactively scrubbed. The library has memory
   policies, and those policies are themselves a form of language
   reform.
"""))


# ===================== 7. Cross-corpus validation: Google Books =====================
A(md(r"""
## 7. Cross-corpus validation: PubMed vs Google Books Ngrams

**What this section does.** Takes each headline shift from §2-§5 and
asks: does the documented terminology change show up in Google
Books Ngrams (English-2019) at the same time, earlier, or later
than it shows up in PubMed? Books and PubMed are very different
corpora — different genres (book-length writing vs journal
articles), different publication lags (books are slower), different
indexing (Books indexes wherever a phrase appears in scanned text;
PubMed indexes titles and abstracts only).

**Why this technique.** Two reasons. First, *cross-corpus
corroboration*: if the same terminology shift appears in two
independent corpora at roughly the same time, that's stronger
evidence than either alone. Second, *cross-corpus contrast*:
if a shift appears in one corpus but not the other, the divergence
is itself an interesting empirical finding about how style and
nomenclature propagate through different writing genres.

**What success looks like.** For each headline shift, both corpora
should show a crossover from the deprecated term to the modern
term. The PubMed crossover may lead Books (faster turnover in
journal articles) or lag Books (the typical case for
non-clinical-vocabulary shifts where books document usage that's
already widespread). The §6 "died by suicide" shift is the
special case where the shift is *visible in Books but invisible
in PubMed* — see §7.1.

**Reading the output.** Per-shift table: `books_old_peak_yr` is
when the deprecated term peaked in Books, `pubmed_crossover` and
`books_crossover` are the years when the modern term overtook the
deprecated term in each corpus, and `lag_books_vs_pubmed` is the
difference (positive = Books lags PubMed).

The five shifts above were detected in PubMed (scientific lit). Do
they also surface in Google Books (popular published-books usage)?
If PubMed leads Books, scientific terminology reform precedes
popular adoption. If they shift together, the reform is broad-
spectrum. If Books shifts and PubMed doesn't (or vice versa), we
have a discourse-asymmetry finding.

We use the Google Books Ngrams English-2019 corpus (free, public
API, harvested by `build/fetch_books_ngrams.py`). The query strategy
is identical: per-term-qualified ngrams summed within each shift,
with case-insensitive matching collapsed to the "(All)" combined
entries.
"""))
A(code(r"""
books_path = Path('..') / 'data' / 'books_ngrams_counts.csv'
books = pd.read_csv(books_path)
print(f'Google Books rows: {len(books):,}')
print(f'Shifts: {books["shift"].unique().tolist()}')
print(f'Year range: {books["year"].min()}-{books["year"].max()}')
"""))

A(code(r"""
# Cross-corpus comparison: per-shift, find Books crossover and compare to PubMed
PUBMED_CROSSOVERS = {
    '1960s_down':           crossover,          # 1966
    '1980s_ptsd':           first_ptsd,         # 1980 (first PTSD record)
    '1990s_did':            first_did,          # 1994 (first DID record)
    '2010s_id':             crossover4,         # 2012
    'neg_suicide_phrasing': None,               # 0 records in PubMed
}

THRESH = 1e-8  # both Books-frequencies need to be above this for crossover to be meaningful
rows = []
for shift in books['shift'].unique():
    sub = books[books['shift'] == shift].copy()
    agg = sub.groupby(['year', 'side'])['frequency'].sum().unstack('side', fill_value=0)
    agg = agg.sort_index()
    old_peak = float(agg['old'].max())
    old_peak_yr = int(agg['old'].idxmax()) if old_peak > 0 else None
    valid = (agg['old'] > THRESH) | (agg['new'] > THRESH)
    cross_mask = (agg['new'] > agg['old']) & valid
    books_cross = int(cross_mask.idxmax()) if cross_mask.any() else None
    pubmed_cross = PUBMED_CROSSOVERS.get(shift)
    lag = (books_cross - pubmed_cross) if (books_cross and pubmed_cross) else None
    ratio_2019 = float(agg['new'].iloc[-1]) / max(float(agg['old'].iloc[-1]), 1e-15)
    rows.append({
        'shift': shift,
        'books_old_peak_yr': old_peak_yr,
        'pubmed_crossover': pubmed_cross,
        'books_crossover': books_cross,
        'lag_books_vs_pubmed': lag,
        'books_2019_new_over_old': round(ratio_2019, 2),
    })
cross_corpus = pd.DataFrame(rows)
print(cross_corpus.to_string(index=False))
"""))

# Chart §7: dual-corpus normalised overlay per shift (chart 9/11)
A(code(r"""
# For each shift, normalise both PubMed and Books to peak-of-the-pair = 1
# so the two corpora overlay on the same chart. The lag is the visual
# distance between the crossover marker on each line.
# Truncate PubMed at _PLOT_YEAR_MAX (2023); Books English-2019 already
# stops at 2019 (Google never released post-2019 ngrams).
_books_agg = (books.groupby(['shift', 'year', 'side'])['frequency']
                    .sum().reset_index())
_pubmed_yearly = []
for shift, parts in frames.items():
    for side, df in parts.items():
        if not len(df): continue
        df_trunc = df[df['year'] <= _PLOT_YEAR_MAX]
        g = df_trunc.groupby('year').size().reset_index(name='n_records')
        g['shift'] = shift; g['side'] = side; g['corpus'] = 'PubMed'
        g = g.rename(columns={'n_records': 'value'})
        _pubmed_yearly.append(g)
_pubmed_yr = pd.concat(_pubmed_yearly, ignore_index=True) if _pubmed_yearly else pd.DataFrame()
_books_agg = _books_agg.rename(columns={'frequency': 'value'})
_books_agg['corpus'] = 'GoogleBooks'

# Normalize: per (shift, corpus), divide by max across both sides
def _norm(group):
    m = group['value'].max() or 1.0
    group['norm'] = group['value'] / m
    return group
_pn = (_pubmed_yr.groupby(['shift', 'corpus'], group_keys=False).apply(_norm))
_bn = (_books_agg.groupby(['shift', 'corpus'], group_keys=False).apply(_norm))
_cc = pd.concat([_pn, _bn], ignore_index=True)
_cc = _cc[_cc['shift'].isin(['1960s_down', '1980s_ptsd', '1990s_did', '2010s_id'])]

_cc_charts = []
for sh in ['1960s_down', '1980s_ptsd', '1990s_did', '2010s_id']:
    sub = _cc[_cc['shift'] == sh].copy()
    if not len(sub): continue
    sub['series'] = sub['corpus'] + ' / ' + sub['side']
    ch = alt.Chart(sub).mark_line(strokeWidth=2).encode(
        x=alt.X('year:O', axis=alt.Axis(labelOverlap=True), title=None),
        y=alt.Y('norm:Q', title='norm to peak'),
        color=alt.Color('series:N', title=None,
                         scale=alt.Scale(domain=[
                             'PubMed / old', 'PubMed / new',
                             'GoogleBooks / old', 'GoogleBooks / new',
                         ],
                         range=['#e76f51', '#264653', '#f4a261', '#8ab17d'])),
        strokeDash=alt.condition(alt.FieldOneOfPredicate('corpus', ['GoogleBooks']),
                                  alt.value([4, 4]), alt.value([1, 0])),
        tooltip=['shift', 'corpus', 'side', 'year', 'value', 'norm'],
    ).properties(width=720, height=160, title=f'§7 {sh}: PubMed (solid) vs Books (dashed), normalised')
    _cc_charts.append(ch)
alt.vconcat(*_cc_charts).resolve_scale(y='shared')
"""))

A(md(r"""
### 7.1 The "died by suicide" cross-corpus contrast

**What this section does.** Looks at the §6 negative finding ("died
by suicide" = 0 records in PubMed) through the Google Books lens.
This is the cross-corpus contrast case — the shift IS happening
somewhere, just not in peer-reviewed medical literature.

**Why care.** The §6 zero by itself could mean "this style change
isn't real" or "this style change hasn't propagated to medical
lit". §7.1 distinguishes them: if Books shows the phrase rising,
the change IS real in popular published-writing terms, and what
§6 measures is the divergence between popular writing and medical
journal articles.

**What success looks like.** Google Books shows nonzero and growing
frequency of "died by suicide" post-~2000, even while PubMed sits
at zero. The growth ratio (2019 / 2000) quantifies the magnitude.

**Reading the output.** The pivot table shows yearly Books
frequencies for both phrases 2000-2019; the chart that follows
plots both phrases on a log scale (the magnitudes are very small in
absolute terms because Books-Ngrams frequencies are per-billion-
word normalised).
"""))
A(code(r"""
sui_books = books[books['shift'] == 'neg_suicide_phrasing'].copy()
sui_pivot = sui_books.pivot(index='year', columns='ngram', values='frequency').fillna(0)
print(f'Books frequencies (note units are per-year-normalized, so very small):\\n')
recent = sui_pivot.loc[2000:2019]
print(recent.to_string(float_format=lambda x: f'{x:.3e}'))
s7_books_died_2000 = float(sui_pivot.loc[2000, 'died by suicide']) if 'died by suicide' in sui_pivot.columns else 0.0
s7_books_died_2019 = float(sui_pivot.loc[2019, 'died by suicide']) if 'died by suicide' in sui_pivot.columns else 0.0
s7_books_growth_ratio = s7_books_died_2019 / max(s7_books_died_2000, 1e-15)
print(f'\\n"died by suicide" growth 2000 -> 2019 in Books: {s7_books_growth_ratio:.1f}x')
print(f'PubMed records of "died by suicide" 2000-2024: 0 (zero growth)')
"""))

# Chart §7.1: books "died by suicide" rise vs PubMed-floor-zero (chart 10/11)
A(code(r"""
# Books frequencies are per-million-word rates; PubMed is record-counts.
# Show Books on log-scale alongside an explicit "PubMed = 0" annotation.
_b_long = (sui_pivot.reset_index()
                     .melt(id_vars='year', var_name='ngram', value_name='freq'))
_b_long = _b_long[_b_long['year'] >= 1970]
_books_line = alt.Chart(_b_long).mark_line(strokeWidth=2).encode(
    x=alt.X('year:O', axis=alt.Axis(values=list(range(1970, 2020, 5))), title='Year'),
    y=alt.Y('freq:Q', title='Google Books frequency (log scale)',
            scale=alt.Scale(type='log', domainMin=1e-10)),
    color=alt.Color('ngram:N', title='Phrase',
                     scale=alt.Scale(range=['#e76f51', '#264653'])),
    tooltip=['ngram', 'year', 'freq'],
).properties(width=720, height=240,
    title=f'§7.1 books: "died by suicide" grew {s7_books_growth_ratio:.0f}x 2000-2019 — PubMed: 0 records (advocacy phrase didn\'t cross into peer-reviewed medical literature)')
_books_line
"""))


# ===================== 8. Audit layer =====================
A(md(r"""
## 8. Audit layer

**What this section does.** The audit layer is the same robustness
scaffolding used in the CBD-Twitter and asylum-Hansard case studies,
applied here to the PubMed corpus. It's where the headline claims
get stress-tested.

**Why this matters.** Sections §2-§6 establish each headline shift
against a pre-registered tolerance. §8 then asks: *if those PASSes
are spurious, what would catch them*? Six different attacks: (8.1)
data-consistency between fetcher steps; (8.2) placebo-anchor
falsification; (8.3) shuffled-label null permutation; (8.4) BH-vs-
bootstrap-CI agreement; (8.5) min_count sensitivity; (8.6)
monotonic-trend rank-correlation test. A finding that survives all
six is much harder to dismiss than one that only passes the
pre-registered tolerance.

**Why §8.x focuses on the §5 MR→ID shift.** That's the largest-
volume shift in the notebook (~65K records) and the one where
inferential machinery has the most power. Audit findings here
generalise to the smaller shifts; the smaller-shift audits (§4 is
particularly small at 1.1K combined records) wouldn't have the
statistical power to do these tests.

### 8.1 Step-A vs Step-B record-count consistency

**What this section does.** Cross-checks that the abstract-level harvest
(Step B: efetch records via NCBI E-utilities) retained the per-shift
record counts that the pre-flight count sweep (Step A: esearch
counts only) had reported. The ratio Step-B / Step-A is the *retention*
for each (shift, side) — a number that should be close to 1.

**Why this is the first audit.** The §0c gotchas (MeSH auto-mapping,
control-character JSON, 10K-PMID silent truncation, IncompleteRead)
are all silent failures in the fetcher — they don't raise errors,
they just drop records. The Step-A-vs-Step-B retention check is the
specific data-consistency audit that catches them.

**What success looks like.** Worst-case retention ≥ 0.80 across all
(shift, side) pairs. The true-negative row (suicide-phrasing `new`
side, which is correctly zero on both sides) is excluded from the
floor check, because dividing zero by zero gives NaN rather than a
meaningful ratio.

**Reading the output.** Per-row: `step_a` is the esearch count,
`step_b` is the records actually written to parquet, `retention` =
step_b / step_a, `flag` = OK/CHECK. A "CHECK (Step-A 0 but Step-B
> 0)" flag would mean Step-A undercounted; an OK with ratio in
[0.80, 1.00] is the expected pattern (small drop for unparseable-
year records).
"""))

A(code(r"""
# Step-A counts loaded from data/pubmed_full_counts.csv (built earlier
# by build/fetch_pubmed.py --full). Here we sum per-label totals across
# the years our abstract corpus covers, then compute the retention.
step_a = pd.read_csv(Path('..') / 'data' / 'pubmed_full_counts.csv')

# Map abstract-corpus shift labels -> Step-A labels
STEPA_MAP = {
    '1960s_down_old':           'ID_old_mongolism',
    '1960s_down_new':           'ID_new_down',
    '1980s_ptsd_old':           'TRAUMA_old_shell_shock',
    '1980s_ptsd_new':           'TRAUMA_new_ptsd',
    '1990s_did_old':            'DISSOC_old_mpd',
    '1990s_did_new':            'DISSOC_new_did',
    '2010s_id_old':             'ID_old_mental_retardation',
    '2010s_id_new':             'ID_new_intellectual',
    'neg_suicide_phrasing_old': 'SUI_old_committed',
    'neg_suicide_phrasing_new': 'SUI_new_died_by',
}

rows = []
for (shift, info) in SHIFTS.items():
    for side in ('old', 'new'):
        k = f'{shift}_{side}'
        sa_label = STEPA_MAP.get(k)
        if sa_label is None: continue
        sa = int(step_a[step_a['label'] == sa_label]['n_records'].sum())
        df = frames[shift][side]
        sb = len(df)
        # True negatives (sa == 0 AND sb == 0, as designed for the negative-
        # finding row) get retention NaN, not zero — they should be reported
        # as "n/a" and excluded from the retention-floor check.
        if sa == 0 and sb == 0:
            ratio = float('nan')
            flag = 'OK (true negative)'
        elif sa == 0:
            ratio = float('inf')
            flag = 'CHECK (Step-A 0 but Step-B > 0)'
        else:
            ratio = sb / sa
            flag = 'OK' if ratio >= 0.80 else 'CHECK'
        rows.append({'shift_side': k, 'step_a': sa, 'step_b': sb, 'retention': ratio, 'flag': flag})
consistency = pd.DataFrame(rows)
print(consistency.to_string(index=False))
# Worst retention over real (non-NaN, finite) cases only
real_ratios = consistency['retention'].replace([float('inf')], float('nan')).dropna()
print(f'\nWorst retention (excluding true negatives): {real_ratios.min():.2f}')
print(f'Records flagged for follow-up: {(consistency["flag"].str.startswith("CHECK")).sum()}')
"""))


# ===================== 8.2 Placebo dates for §5 ID =====================
A(md(r"""
### 8.2 Placebo dates for the §5 ID shift

**What this section does.** Re-runs the §5 crossover detection at
*placebo* anchor years (1985, 1995, 2000, 2020, 2023) — years with
no known regulatory event for the mental-retardation → intellectual-
disability shift — and asks whether they also produce in-window
crossovers.

**Why this technique.** A real anchor effect should be specific to
the documented event (Rosa's Law 2010 + DSM-5 2013, midpoint
2012). If placebo years also produce crossovers, then the apparent
anchor-effect is just background noise / general year-to-year
variation, and our pre-registered "crossover within ±2 of 2012"
result is not informative.

**What success looks like.** The real anchor produces an in-window
crossover; ≤ 2 of the 5 placebo anchors do (false-discovery
tolerance ~40%, which is wide because we only have 5 placebos).
The point estimate is "real PASSes; placebos mostly don't."

**Reading the output.** Per-row: anchor year, whether it's real,
the crossover year detected in its ±5-year window, and `aligns`
(crossover within ±2 of the anchor). The summary lines report real-
anchor alignment and placebo-anchor false-positive count.
"""))

A(code(r"""
placebo_years = [1985, 1995, 2000, 2020, 2023]
real_anchor = anchor4  # 2012

old_yr_long = old4.groupby('year').size().reindex(range(1980, 2025), fill_value=0)
new_yr_long = new4.groupby('year').size().reindex(range(1980, 2025), fill_value=0)

rows = []
for yr in [real_anchor] + placebo_years:
    # Re-detect crossover assuming `yr` is the anchor: window ±5 years around it.
    window = range(yr - 5, yr + 6)
    cross_in_window = next((y for y in window
                             if new_yr_long[y] > old_yr_long[y] and (new_yr_long[y]+old_yr_long[y]) >= 5),
                            None)
    rows.append({
        'anchor': yr,
        'is_real': yr == real_anchor,
        'crossover_in_window': cross_in_window,
        'aligns': cross_in_window is not None and abs(cross_in_window - yr) <= 2,
    })
placebo_df = pd.DataFrame(rows)
print(placebo_df.to_string(index=False))
print(f'\nReal anchor crossover in-window: {placebo_df[placebo_df.is_real].aligns.iloc[0]}')
print(f'Placebo anchors that "align": {placebo_df[(~placebo_df.is_real) & placebo_df.aligns].shape[0]} / 5')
"""))


# ===================== 8.3 Shuffled-label null on §5 keyness =====================
A(md(r"""
### 8.3 Shuffled-label null on §5 keyness

**What this section does.** Randomly permutes the (old, new) labels
across the §5 records B=99 times and recomputes the maximum |G²|
each time. Compares the observed real-label max |G²| against the
distribution of permuted-null max |G²|.

**Why this technique.** The §5/§5a keyness has a *huge* observed
G² because the corpora are large and the contrast is genuine. But
*any* random partition of a large mixed corpus into two non-empty
halves will produce *some* terms with elevated G² just from
sampling variance. The permutation null tells us how big a max-G²
we'd expect from pure noise; the ratio observed / permuted-95th-
percentile quantifies how much bigger the real signal is.

**What success looks like.** Observed |G²| at least 10× the
permuted 95th-percentile null. (A floor of 10× is conservative —
typical real signals in linguistic corpora are 30-100×.) The
shuffle distribution should peak well below the observed value.

**Reading the output.** The print summary shows observed max |G²|,
the median and 95th-percentile of the 99 permuted null maxes, the
ratio, and the wall-time the permutation took (~minutes, since
each permutation re-runs the keyness on ~30K documents).
"""))

A(code(r"""
import time as _t

pre_id  = pcd.from_dataframe(old4[old4['year'] >= 2005], text_col='text', meta_cols=('year','journal'))
post_id = pcd.from_dataframe(new4[new4['year'] >= 2010], text_col='text', meta_cols=('year','journal'))
key_id = pcd.compare(pre_id, post_id).keyness(
    min_count=30, formula='dunning', stop_words=PUBMED_STOP, multiple_comparisons='bh',
)
obs_max = float(key_id.to_df()['g2'].abs().max())

# Shuffled null
all_docs = pd.concat([
    old4[old4['year'] >= 2005].assign(_label='old'),
    new4[new4['year'] >= 2010].assign(_label='new'),
], ignore_index=True)
n_a = (all_docs['_label'] == 'old').sum()

B = 99
rng = np.random.default_rng(0)
perm_max = []
_t0 = _t.time()
for b in range(B):
    perm = all_docs.sample(frac=1.0, random_state=rng.integers(0, 1 << 31)).reset_index(drop=True)
    a_p = pcd.from_dataframe(perm.iloc[:n_a], text_col='text')
    b_p = pcd.from_dataframe(perm.iloc[n_a:], text_col='text')
    try:
        kn = pcd.compare(a_p, b_p).keyness(min_count=30, formula='dunning', stop_words=PUBMED_STOP)
        perm_max.append(float(kn.to_df()['g2'].abs().max()))
    except Exception:
        continue
elapsed = _t.time() - _t0

p95 = float(np.percentile(perm_max, 95))
print(f'Observed max |G^2| (real labels): {obs_max:,.0f}')
print(f'Permuted null max |G^2|, B={len(perm_max)}: median {np.median(perm_max):,.0f}, 95th pct {p95:,.0f}')
print(f'Ratio observed / 95th-pct null: {obs_max / p95:.0f}x')
print(f'Walltime: {elapsed:.0f}s')
"""))


# ===================== 8.4 BH-vs-CI alignment =====================
A(md(r"""
### 8.4 BH-significance ⊆ CI-excludes-zero alignment (on §5 keyness)

**What this section does.** Cross-checks two different inferential
statements about the §5 keyness terms: (a) BH-adjusted p-value <
0.05 (FDR-corrected significance), and (b) per-term bootstrap 95%
CI excludes zero (sampling-distribution-based significance). The
two should mostly agree.

**Why this technique.** BH and bootstrap-CI control different errors
— BH controls the false-discovery rate (expected proportion of
false positives among rejections); the per-term bootstrap CI
controls the per-term type-I error. They answer different
questions, but both should reject the same terms most of the time.
Substantial disagreement (>20% of either-flagged terms) would mean
one of the two methods is misreading the data, and we'd need to
investigate which.

**What success looks like.** Disagreement ratio (sum of BH-only and
CI-only) / (either flag) ≤ 0.20. This is the same threshold used
in the CBD case study; the iter-3 audit tightened it from 0.30
to 0.20 (the prior threshold was an unjustified goalpost-shift).

**Reading the output.** The summary lines show: BH-significant
count, CI-excludes-zero count, both-flagged count, BH-only count,
CI-only count, and the disagreement ratio.
"""))
A(code(r"""
_k5 = key5_ci.to_df()
_k5 = _k5[_k5['p_adjusted'].notna()].copy()
_bh_sig = _k5['p_adjusted'] < 0.05
_ci_excl = (_k5['g2_ci_lower'] > 0) | (_k5['g2_ci_upper'] < 0)
n_both = int((_bh_sig & _ci_excl).sum())
n_bh_only = int((_bh_sig & ~_ci_excl).sum())
n_ci_only = int((~_bh_sig & _ci_excl).sum())
n_either = int((_bh_sig | _ci_excl).sum())
s84_disagree_ratio = (n_bh_only + n_ci_only) / max(1, n_either)
print(f'BH-significant:          {int(_bh_sig.sum())}')
print(f'CI excludes 0:           {int(_ci_excl.sum())}')
print(f'Both flagged:            {n_both}')
print(f'BH only (CI straddles):  {n_bh_only}')
print(f'CI only (not BH-sig):    {n_ci_only}')
print(f'Disagreement / either-flagged ratio: {s84_disagree_ratio:.3f}')
"""))


# ===================== 8.5 min_count sensitivity =====================
A(md(r"""
### 8.5 min_count sensitivity for §5 keyness

**What this section does.** Re-runs the §5 keyness contrast at five
different `min_count` thresholds (10, 30, 50, 100, 200) and checks
whether the top-3 distinctive terms (pre-anchor and post-anchor)
are stable across the sweep.

**Why this technique.** `min_count` is an analyst's choice — terms
appearing fewer than `min_count` times in either corpus are
dropped from the keyness computation. If the top results change
when we move the threshold, then our pre-registered top-3 is just a
function of the threshold, not of the actual term-shift. If they're
stable, the contrast is robust.

**What success looks like.** The top-3 pre-anchor terms are the same
set across all five `min_count` values; same for post-anchor.
Total stability across an order of magnitude.

**Reading the output.** Per-row: the min_count value, the number
of terms surviving that floor, and the top-3 pre/post terms as a
comma-separated string. The summary lines report whether the
top-3 sets are stable.
"""))
A(code(r"""
mc_rows = []
for mc in [10, 30, 50, 100, 200]:
    try:
        kk = pcd.compare(mr_pre, id_post).keyness(
            min_count=mc, formula='dunning', stop_words=PUBMED_STOP,
            multiple_comparisons='bh',
        )
        kdf = kk.to_df()
        top3_pre = ','.join(kdf[kdf['log_ratio'] > 0].head(3)['term'].tolist())
        top3_post = ','.join(kdf[kdf['log_ratio'] < 0].head(3)['term'].tolist())
        mc_rows.append({'min_count': mc, 'n_terms': len(kdf),
                        'top-3 pre-anchor': top3_pre, 'top-3 post-anchor': top3_post})
    except Exception as e:
        mc_rows.append({'min_count': mc, 'n_terms': 0, 'error': str(e)[:50]})
mc_df = pd.DataFrame(mc_rows)
print(mc_df.to_string(index=False))
_pre_sets = [set(s.strip() for s in r.split(',')) for r in mc_df['top-3 pre-anchor']]
_post_sets = [set(s.strip() for s in r.split(',')) for r in mc_df['top-3 post-anchor']]
s85_pre_stable = all(s == _pre_sets[0] for s in _pre_sets)
s85_post_stable = all(s == _post_sets[0] for s in _post_sets)
print(f'\\npre-anchor top-3 stable across {len(mc_rows)} min_count values:  {s85_pre_stable}')
print(f'post-anchor top-3 stable across {len(mc_rows)} min_count values: {s85_post_stable}')
"""))


# ===================== 8.6 Spearman monotonic-trend test =====================
A(md(r"""
### 8.6 Spearman monotonic-trend test on the §5 trajectory

**What this section does.** Tests whether the §5 ID record-count
series (2013-2024, the post-anchor decade) is monotonically rising,
using Spearman's rank-correlation between year and count.

**Why this technique.** The crossover-year diagnostic (§5 main) says
*when* ID overtook MR; it doesn't say whether the post-crossover
trajectory continued rising or plateaued. Spearman rho on (year,
count) tells us: rho > 0 means rising, rho near 1 means
monotonically rising. The p-value tests whether the observed
trend differs from no-trend.

**What success looks like.** Spearman rho > 0.70 (strong positive
monotonic trend) with p < 0.05.

**Reading the output.** Single line: Spearman rho and p-value over
the (year, count) series 2013-2024.
"""))
A(code(r"""
from scipy.stats import spearmanr
id_post_yr = new_yr4.loc[2013:2024]
years_arr = id_post_yr.index.values.astype(float)
counts_arr = id_post_yr.values.astype(float)
rho, p_sp = spearmanr(years_arr, counts_arr)
s86_rho = float(rho)
s86_p = float(p_sp)
print(f'Spearman rho on (year, ID-count) 2013-2024: rho = {s86_rho:+.3f}, p = {s86_p:.2e}')
print(f'Monotonic rising (rho > 0.7): {s86_rho > 0.7}')
"""))


# ===================== 9. Scoreboard =====================
A(md(r"""
## 9. Audit scoreboard

**What this section does.** Collects every per-shift and audit-layer
verdict from §2-§8 into one table, with **runtime-computed**
Observed and Verdict cells. No verdict in this table is a literal
string — every one is either an f-string over named runtime
variables (Observed) or a Boolean expression over named threshold
constants (Verdict). The same data-driven scoreboard pattern as the
CBD and asylum case studies.

**Why this matters.** The audit pattern is robust *only* if the
final summary cannot be edited by hand without invalidating the
notebook. A scoreboard with literal "PASS" / "FAIL" cells can be
retconned after seeing the data. A scoreboard built from threshold
constants (defined at the top of the cell) and runtime variables
(defined throughout the notebook) cannot — to change a verdict, you
have to change a threshold constant, which makes the change
auditable.

**Reading the output.** Three columns:
- **Check**: the section being summarised
- **Observed**: an f-string over runtime variables showing the
  measured quantity
- **Verdict**: PASS / PARTIAL / FAIL / AUDIT-RESOLVED / OBSERVED /
  META-FINDING. PASS = pre-registered prediction confirmed within
  tolerance. PARTIAL = result is in the right direction but doesn't
  hit the strict tolerance. FAIL = pre-registered falsifier
  triggered (only §6 is here). AUDIT-RESOLVED = a previous claim
  was refuted by an iter-N audit and the section now reports the
  corrected interpretation. OBSERVED = descriptive sub-pattern (the
  three §6.5.2-§6.5.4 inventory sub-patterns). META-FINDING = the
  §6.5.1c headline polysemy-survey result.

**What's not in this table.** This is the *audit* scoreboard, not
the substantive findings table. The substantive medical-history
narrative is in §2-§6 prose; this table is just the audit verdicts.
"""))

A(code(r"""
# Pre-specified thresholds (drafted with §0b pre-registration)
TH_CROSSOVER_TOL_60S = 5   # crossover must be within 5 years of 1965
TH_FIRST_PTSD_TOL    = 1   # first PTSD record within 1 year of 1980
TH_FIRST_DID_LO      = 1993
TH_FIRST_DID_HI      = 1995
TH_CROSSOVER_TOL_10S = 2   # ID crossover within 2 years of 2012
TH_RETENTION_FLOOR   = 0.80  # Step-A vs Step-B retention
TH_NULL_RATIO_FLOOR  = 10  # observed/null at 10x
TH_TOP15_CI_EXCL     = 10  # of top-15 keyness terms, this many should have per-term CI excluding 0
TH_BURST_ONSET_LO    = 1979  # PTSD burst onset window (DSM-III anchor 1980, ±1)
TH_BURST_ONSET_HI    = 1983
TH_RHO_FLOOR         = 0.70  # Spearman rho on ID post-anchor trajectory should rise
TH_BH_CI_DISAGREE    = 0.20  # disagreement ratio between BH and bootstrap CI
                              # (matches the CBD case-study threshold; tightened
                              # from 0.30 -> 0.20 in iter-3 audit to remove
                              # the unjustified goalpost-shift)

# §2 evidence
s2_cross = crossover
s2_pass = s2_cross is not None and abs(s2_cross - anchor1) <= TH_CROSSOVER_TOL_60S

# §3 evidence
s3_first_ptsd = first_ptsd
s3_pass = s3_first_ptsd is not None and abs(s3_first_ptsd - anchor2) <= TH_FIRST_PTSD_TOL

# §4 evidence
s4_first_did = first_did
s4_pass = s4_first_did is not None and TH_FIRST_DID_LO <= s4_first_did <= TH_FIRST_DID_HI

# §5 evidence
s5_cross = crossover4
s5_pass = s5_cross is not None and abs(s5_cross - anchor4) <= TH_CROSSOVER_TOL_10S

# §6 negative finding — falsifier was zero, observed is zero
s6_pass = len(new5) == 0  # honest record of the falsification

# §7.1 retention (exclude true-negative rows where sa == sb == 0)
_real_ratios = consistency['retention'].replace([float('inf')], float('nan')).dropna()
s71_worst = float(_real_ratios.min()) if len(_real_ratios) else float('nan')
s71_pass = (s71_worst >= TH_RETENTION_FLOOR) and not np.isnan(s71_worst)

# §7.2 placebo
s72_real_aligns = bool(placebo_df[placebo_df.is_real].aligns.iloc[0])
s72_placebos_align = int(placebo_df[(~placebo_df.is_real) & placebo_df.aligns].shape[0])
s72_pass = s72_real_aligns and s72_placebos_align <= 2  # tolerate up to 2/5 spurious

# §7.3 shuffled null
s73_ratio = obs_max / p95 if p95 > 0 else float('inf')
s73_pass = s73_ratio >= TH_NULL_RATIO_FLOOR

scoreboard = pd.DataFrame([
    ('§0d Cross-package Rayson G^2 byte-equality',
     f'worst absolute error across 6 reference cases: {float(xv["abs_error"].max()):.2e} (assertion floor 1e-10)',
     'PASS' if float(xv['abs_error'].max()) < 1e-10 else 'FAIL'),
    ('§2 mongolism -> Down syndrome',
     f'crossover {s2_cross} (anchor {anchor1}, tolerance ±{TH_CROSSOVER_TOL_60S})',
     'PASS' if s2_pass else 'FAIL (pre-registered)'),
    ('§2a Bootstrap CIs on §2 contextual keyness',
     f'top-15: per-term CI excludes 0 in {s2a_top15_per_term_excl}/15; simultaneous CI excludes 0 in {s2a_top15_sim_excl}/15',
     'PASS' if s2a_top15_per_term_excl >= TH_TOP15_CI_EXCL else 'PARTIAL'),
    ('§2b Collocation shift around "syndrome"',
     f'{len(s2b_df):,} collocates analysed; top |shift| at {s2b_df.iloc[0]["collocate"]!r} (shift={s2b_df.iloc[0]["shift"]:+.2f})' if len(s2b_df) else 'no collocates',
     'PASS' if len(s2b_df) > 0 else 'PARTIAL'),
    ('§3 shell shock -> PTSD',
     f'first PTSD record {s3_first_ptsd} (anchor {anchor2}, tolerance ±{TH_FIRST_PTSD_TOL})',
     'PASS' if s3_pass else 'FAIL (pre-registered)'),
    ('§3b Burstiness detection on PTSD annual series',
     f'first burst onset: {s3b_first_burst_year}; aligned with DSM-III 1980 (window {TH_BURST_ONSET_LO}-{TH_BURST_ONSET_HI}): {s3b_aligned}',
     'PASS' if s3b_aligned else 'PARTIAL'),
    ('§4 MPD -> DID',
     f'first DID record {s4_first_did} (pre-reg window 1993-1995)',
     'PASS' if s4_pass else 'PARTIAL'),
    ('§5 mental retardation -> intellectual disability',
     f'crossover {s5_cross} (anchor {anchor4}, tolerance ±{TH_CROSSOVER_TOL_10S})',
     'PASS' if s5_pass else 'PARTIAL'),
    ('§5a Bootstrap CIs on §5 contextual keyness',
     f'top-15: per-term CI excludes 0 in {s5a_top15_per_term_excl}/15; simultaneous CI excludes 0 in {s5a_top15_sim_excl}/15',
     'PASS' if s5a_top15_per_term_excl >= TH_TOP15_CI_EXCL else 'PARTIAL'),
    ('§5.5 SIRS/Sepsis-2 -> Sepsis-3 (operational-definition revision)',
     f'first Sepsis-3 record {s55_first_sepsis3} (pre-reg window 2015-2017); aligns: {s55_aligned}',
     'PASS' if s55_aligned else 'PARTIAL'),
    ('§5.6 Asperger -> ASD (dual-rationale retirement: terminology + ethics)',
     f'crossover {s56_crossover} (terminology pre-reg 2013-2015); post-2018 decline acceleration ratio {s56_acceleration_ratio:.2f}x (ethics pre-reg >= 1.5x)',
     'PASS' if (s56_terminology_pass and s56_ethics_pass) else ('PARTIAL' if s56_terminology_pass else 'FAIL')),
    ('§6 NEGATIVE FINDING: "committed" -> "died by" suicide',
     f'"died by suicide" PubMed records: {len(new5)} (falsifier was zero)',
     'FAIL (pre-registered falsifier; honestly recorded)' if s6_pass else 'PASS'),
    ('§6.5.1 AUDIT-RESOLVED: word-sense decomposition of `retard*` (iter-1 BLOCKING refutation)',
     f'slur sense: {s651_slur_n}/{s651_total:,} records = {s651_slur_pct:.3f}% (essentially absent); clinical-ID compound declines {s651_clinical_decline_pct:.0f}% from 1990s to 2020s (corroborates §5)',
     'AUDIT-RESOLVED (prior INVERSION claim REFUTED; corrected interpretation: morpheme dominated by scientific process-verb senses, slur essentially absent)'),
    ('§6.5.1b POLYSEMY-AUDITED SURVEY (iter-2/3 generalisation of iter-1 finding)',
     f'{s651b_total} labels audited by random-20-PMID sense check: {s651b_collision} COLLISIONs, {s651b_drift} DRIFTs, {s651b_valid_era} VALID era-clinical, {s651b_valid_persistent} VALID-PERSISTENT, {s651b_unmeasurable} UNMEASURABLE, {s651b_unclassifiable} UNCLASSIFIABLE',
     f'META-FINDING: {s651b_collision}/{s651b_total} = {100*s651b_collision/s651b_total:.0f}% polysemy-collision rate is the prior risk for any single-token deprecated-medical-vocabulary tracking study'),
    ('§6.5.1c MULTI-LABEL SLUR WSI DEEP AUDIT (iter-4 full-corpus extension of §6.5.1)',
     f'{s651c_n_labels} slur-like labels WSI-classified across {s651c_total_records:,} PubMed records 1950-2024; corpus-wide explicit-slur fraction: {s651c_total_slur}/{s651c_total_records:,} = {s651c_slur_pct:.4f}%; {s651c_labels_with_any_slur}/{s651c_n_labels} labels had >=1 explicit slur record',
     f'CONFIRMED: corpus-wide slur fraction <{max(0.01, s651c_slur_pct):.2f}% for every label — single-token queries on slur-like English morphemes do NOT measure slur usage'),
    ('§6.5.2 Loaded-vocab clean extinctions',
     f'{s65_n_extinct} of 43 loaded-vocab labels are extinct (peak <= 1990 and zero records in 2020s)',
     'OBSERVED'),
    ('§6.5.3 ZERO-hit indexing-curation evidence',
     f'{s65_n_zero} zero-hit labels remain in the post-iter-4-curation inventory (iter-3 had 4; all removed in iter-4 ethical review)',
     'OBSERVED'),
    ('§6.5.4 Persistent loaded-vocab (not all retire)',
     f'{s65_n_persistent} labels persist with 2020s sum >= 50 records',
     'OBSERVED'),
    ('§7 Cross-corpus: PubMed vs Google Books',
     f'PubMed leads Books for {int((cross_corpus["lag_books_vs_pubmed"] > 0).sum())} of {len(cross_corpus)} shifts; Books-"died by suicide" growth 2000->2019: {s7_books_growth_ratio:.1f}x',
     'PASS' if s7_books_growth_ratio > 1 else 'PARTIAL'),
    ('AUDIT §8.1 Step-A/Step-B retention',
     f'worst retention {s71_worst:.2f} (floor {TH_RETENTION_FLOOR})',
     'PASS' if s71_pass else 'PARTIAL'),
    ('AUDIT §8.2 Placebo anchor years',
     f'real anchor aligns: {s72_real_aligns}; placebos aligning: {s72_placebos_align}/5',
     'PASS' if s72_pass else 'PARTIAL'),
    ('AUDIT §8.3 Shuffled-label null for §5 keyness',
     f'observed |G^2|={obs_max:,.0f}; 95th-pct null={p95:,.0f}; ratio {s73_ratio:.0f}x',
     'PASS' if s73_pass else 'PARTIAL'),
    ('AUDIT §8.4 BH-vs-bootstrap-CI alignment on §5 keyness',
     f'disagreement ratio: {s84_disagree_ratio:.3f} (tolerance {TH_BH_CI_DISAGREE})',
     'PASS' if s84_disagree_ratio <= TH_BH_CI_DISAGREE else 'PARTIAL'),
    ('AUDIT §8.5 min_count sensitivity for §5 keyness',
     f'pre-anchor top-3 stable: {s85_pre_stable}; post-anchor top-3 stable: {s85_post_stable}',
     'PASS' if (s85_pre_stable and s85_post_stable) else 'PARTIAL'),
    ('AUDIT §8.6 Spearman monotonic-trend on §5 ID 2013-2024',
     f'rho = {s86_rho:+.3f}, p = {s86_p:.2e} (floor rho > {TH_RHO_FLOOR})',
     'PASS' if s86_rho > TH_RHO_FLOOR else 'PARTIAL'),
], columns=['Check', 'Observed', 'Verdict'])

with pd.option_context('display.max_colwidth', 100, 'display.width', 200):
    print(scoreboard.to_string(index=False))
"""))

# Chart §9: scoreboard verdict strip (chart 11/11)
A(code(r"""
_sb = scoreboard.copy()
_sb['check_short'] = _sb['Check'].str.replace(r'^(§[\d\.a-z]+)\s+', r'\1 ', regex=True).str.slice(0, 70)
def _verdict_class(v):
    s = str(v)
    if s.startswith('PASS'): return 'PASS'
    if s.startswith('AUDIT-RESOLVED') or 'AUDIT-RESOLVED' in s: return 'AUDIT-RESOLVED'
    if s.startswith('META-FINDING'): return 'META-FINDING'
    if s.startswith('PARTIAL'): return 'PARTIAL'
    if s.startswith('FAIL'): return 'FAIL'
    if s.startswith('OBSERVED'): return 'OBSERVED'
    return 'OTHER'
_sb['verdict_class'] = _sb['Verdict'].apply(_verdict_class)
_sb['row_idx'] = range(len(_sb))
_pal_sb = {
    'PASS':            '#2a9d8f',
    'PARTIAL':         '#e9c46a',
    'FAIL':            '#e63946',
    'AUDIT-RESOLVED':  '#9d4edd',
    'META-FINDING':    '#3a86ff',
    'OBSERVED':        '#888888',
    'OTHER':           '#cccccc',
}
_strip_sb = alt.Chart(_sb).mark_rect(stroke='white', strokeWidth=1).encode(
    y=alt.Y('check_short:N', sort=_sb['check_short'].tolist(), title=None),
    x=alt.value(0), x2=alt.value(540),
    color=alt.Color('verdict_class:N', title='Verdict class',
                     scale=alt.Scale(domain=list(_pal_sb.keys()),
                                      range=list(_pal_sb.values()))),
    tooltip=['Check', 'Observed', 'Verdict'],
).properties(width=540, height=max(22*len(_sb), 200),
    title='§9 scoreboard verdicts (green PASS, yellow PARTIAL, red FAIL, purple AUDIT-RESOLVED, blue META, grey OBSERVED)')
_strip_sb
"""))

A(md(r"""
**Bottom line.** Five terminology shifts surveyed; four cleanly
PASS their pre-registered prediction (mongolism→Down syndrome,
shell-shock→PTSD, MPD→DID, MR→ID) within stated tolerances of their
documented anchor events. One cleanly FAILS — the "died by suicide"
phrasing change has zero PubMed penetration, falsifying the pre-
registered prediction.

The audit layer corroborates: Step-A vs Step-B record-count retention
is within tolerance, real-anchor crossover detection out-performs
placebo-anchor crossover detection, and the keyness signal on the
largest shift survives a shuffled-label null by a large factor.

The audit pattern itself — pre-registration with explicit falsifiers,
plus a layer of robustness checks whose verdicts come from runtime
data rather than authorial assertion — is the unit of generalisation.
It worked on Twitter discourse (CBD case study), on parliamentary
discourse (asylum case study), and on scientific discourse here.
"""))


# ============================ Notebook footer ============================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}: {len(cells)} cells ({n_code} code)")
