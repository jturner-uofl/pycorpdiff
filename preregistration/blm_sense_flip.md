# Pre-registration: the "BLM" abbreviation sense-flip in news text

**This document is committed BEFORE any data retrieval or inspection.**
The commit hash and timestamp of this file in the public repository
constitute the pre-registration. At the time of this commit, the author
has **not** retrieved, viewed, or computed any New York Times (or other
news) articles, counts, embeddings, or sense assignments for the token
*BLM* or its expansions. The analysis will be added in a later,
separately-timestamped commit (and the fetched data are not
redistributed; only aggregate results are reported).

Associated study: the narrative-audit methods paper. This is a second
**abbreviation-polysemy** case alongside *CBD* (cannabidiol vs central
business district): it tests the polysemy safeguard and the
`sense_drift` detector on a token whose dominant sense is documented to
have changed over time.

## Background (from documented public events only)

*BLM* is a polysemous abbreviation with two dominant senses in U.S.
English:

- **Bureau of Land Management** — a federal agency (est. 1946) managing
  public lands. It was the salient *BLM* of the **April 2014** armed
  standoff in Bunkerville, Nevada (rancher Cliven Bundy vs the agency
  over grazing fees).
- **Black Lives Matter** — a social movement; the hashtag dates to
  **2013** (after the acquittal in the Trayvon Martin case), reached
  mass salience in **August 2014** (Ferguson, Missouri), and peaked in
  **2020** (George Floyd).

Critically, **both** senses spiked in **2014** — the agency standoff in
April, the movement in August — making 2014 a documented year of maximal
ambiguity for the bare token *BLM*.

## Predictions (directional; fixed before any data)

Source corpus: New York Times articles **containing the token "BLM"**,
2008–2022 (Article Search API), classified into the two senses by
context. `sense_drift` is fit with a **reference window of 2008–2013**
(pre-movement), then run forward.

- **P1 — sense flip.** Among "BLM"-token articles, the dominant sense
  flips from *Bureau of Land Management* (pre-2014) to *Black Lives
  Matter*, with the crossover in **2014–2016**; by **2020** the movement
  sense is the overwhelming majority.
- **P2 — the 2014 collision.** 2014 is bimodal — both senses materially
  present in the same year — rather than a clean single-sense year.
- **P3 — change type.** `sense_drift` classifies the movement sense as an
  **emergence** (a coherent new sense entering a space that did not
  previously hold it), not merely a re-weighting of pre-existing senses.
- **P4 — the agency sense's fate (the key, non-obvious fork).** For the
  *token* "BLM", the *Bureau of Land Management* sense undergoes genuine
  **obsolescence**, not mere dilution: as the movement claims the
  abbreviation, writers increasingly spell out "Bureau of Land
  Management" / "the agency" to avoid confusion, so the agency sense of
  the **abbreviation** declines in **absolute** count, not only in share.

## Falsifiers (what would prove each prediction wrong)

- **F1 (vs P1):** *Bureau of Land Management* remains the dominant
  "BLM"-token sense through 2018.
- **F2 (vs P2):** 2014 resolves to a single dominant sense (not bimodal).
- **F3 (vs P3):** the change is classified `frequency_shift` /
  `broadening`, i.e. no coherent emergent sense.
- **F4 (vs P4):** the agency's "BLM"-token absolute count holds steady or
  grows (i.e. **dilution**, not obsolescence). This outcome is reported
  honestly if observed — it would mirror the bile-duct dilution case and
  is a live possibility, which is what makes P4 a real test.

## Negative controls

- **N1 — no spurious flip.** A non-polysemous control abbreviation with a
  single stable referent (e.g. *NASA*) shows **no** sense flip and no
  significant drift over the same window.
- **N2 — agency vs abbreviation.** Articles using the **spelled-out**
  "Bureau of Land Management" do **not** collapse in absolute count: the
  agency did not stop being covered. This separates the real claim
  (writers stopped using the *abbreviation* for the agency) from a false
  one (the agency vanished from the news). N2 is the decisive check on
  P4: obsolescence of the abbreviation-sense, not of the referent.

## Two-wolves framing predictions (for the optional framing layer)

Pre-registered so neither can be fit after the fact. The corpus
adjudicates via keyness/collocation; we report which prediction the data
support and take no side.

- **🐺A (left-leaning hypothesis):** *Black Lives Matter* coverage is
  framed predominantly as movement/justice, but a measurable
  "riot/violence/looting" sub-framing is present and concentrates in
  2020.
- **🐺B (right-leaning hypothesis):** the Bundy/*Bureau of Land
  Management* standoff is framed with harsher delegitimising labels
  ("extremist", "domestic terrorist", "armed") than a comparable
  left-coded standoff would receive.

## Commitment

This file is committed and pushed **before** any news data is fetched.
Results will be added as `blm_sense_flip_RESULTS.md` in a later,
separately-timestamped commit. Source articles are not redistributed
(copyright); only aggregate counts, sense shares, distinctive terms, and
the `sense_drift` outputs are reported.
