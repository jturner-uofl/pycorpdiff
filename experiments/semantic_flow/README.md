# `semantic_flow` — shelved experiment (negative result)

**Status: shelved.** Built, validated on real data, and *not* shipped — the
novel measurement it proposed does not survive contact with real diachronic
embeddings. Kept here as a documented lesson rather than deleted.

## The idea

Treat a whole vocabulary's diachronic embeddings as a **time-varying vector
field** and borrow the analysis vocabulary that fluid dynamics and single-cell
genomics (RNA velocity) built for exactly that object:

- **velocity / speed** — per-step displacement → rate of semantic change.
- **acceleration** — change in velocity → "semantic shocks".
- **divergence** — local flux of the velocity field → **broadening (+) vs.
  narrowing (−)**, giving the 19th-century historical-semantics typology a
  computable, sign-carrying instrument.
- plot it as an RNA-velocity-style flow field (arrows on a 2-D projection,
  coloured by divergence).

The divergence → broadening/narrowing bridge was the genuinely novel part: a
*new measurement*, not a restatement of existing cosine-shift metrics.

## What the data said (COHA 1900/1950/1990, Hamilton et al. reference shifts)

Run `validate_coha.py` (requires the COHA HistWords archive). Findings:

| Claim | Result |
|-------|--------|
| speed/displacement recovers known shifts | **✅ validated** — Spearman ρ = 0.87 vs Hamilton's published cosine shifts; `gay` 99th pctile, `terrific` 86th, function words `the`/`of` at 0th |
| raw "fastest words" are meaningful | **⚠️ confounded** — top of the ranking is dominated by proper nouns (barry, abraham, cameron…): referent drift, not meaning change. A known diachronic-embedding artifact a content-word filter only partly removes |
| divergence measures broadening/narrowing | **❌ failed** — the field-divergence operator came out uniformly negative and merely inversely tracked speed |
| neighbourhood-dispersion measures broadening/narrowing | **❌ failed** — also noise: `gay` +0.002 (≈0), function word `the` showed the *largest* "broadening" (+0.017). Signal swamped by the global geometry of how word2vec spaces evolve over time |

## Why it was shelved

- The part that **works** (speed/displacement) is essentially a vocabulary-wide,
  batched version of `histwords_cosine_shift` + a visualization — modest new value
  over existing package surface.
- The part that's **novel** (divergence → broadening/narrowing) **does not measure
  the intended construct** on real embeddings. Two independent operationalizations
  both failed.
- Net: the new part fails, the working part is redundant. Not worth a shipped,
  maintained public API.

## Lessons

1. **Physics-envy check, validated.** The fluid-dynamics analogy was seductive
   (the warning was raised *before* building it) and the empirics cashed the
   warning. Borrowing impressive machinery from another field is only worth it
   when the borrowed quantity lands on a phenomenon the data actually exhibits.
2. **Synthetic tests can give false confidence.** The unit tests passed because
   the synthetic fixture had *planted* clean source/sink structure. Real word2vec
   has no such structure, so the green test suite said nothing about whether
   divergence measures broadening. Validate novel *measurements* against real
   data / known ground truth, not just against a fixture you designed to pass.
3. **Proper-noun / referent drift dominates raw diachronic speed.** Any
   vocabulary-wide drift ranking needs content-word filtering to be interpretable.

## Side-find (real bug — now fixed)

While validating against COHA, `pycorpdiff.datasets.histwords.fetch_histwords_decade`
was found to be broken for the COHA source: it expected `{decade}.pkl` /
`{decade}.npy` but the COHA archive ships `{decade}-vocab.pkl` / `{decade}-w.npy`,
raising `FileNotFoundError` even after a successful download. **Fixed** — the loader
now resolves either layout (see the CHANGELOG 0.1.0a28 "Fixed" entry and the COHA
regression tests in `tests/unit/test_histwords_loader.py`). `validate_coha.py` here
still loads the files directly (it predates the fix); the public loader now works
with `source="coha"` directly.
