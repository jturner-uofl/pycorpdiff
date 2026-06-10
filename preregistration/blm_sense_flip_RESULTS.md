# Results: the "BLM" abbreviation sense-flip in news text

Companion to the pre-registration `blm_sense_flip.md` (committed first, before
any data). Source: New York Times Article Search API, token query `"BLM"`,
classified by context (LAND = Bureau of Land Management; MOVE = Black Lives
Matter). Aggregate counts only; no article text is redistributed.

## What we found

Yearly NYT articles whose text contains the token *BLM*, with the top
relevance-ranked results classified by sense:

| Year | "BLM" hits | dominant sense | example headline (paraphrased) |
|---|--:|---|---|
| 2012 | 2 | **LAND** | drillers and a U.S. land agency |
| 2014 | 3 | **LAND** | the Bundy rancher standoff |
| 2016 | 3 | **LAND** | Bundy verdict; wild-horse program |
| 2018 | 2 | **LAND** | fracking on public lands |
| 2020 | 51 | **MOVE** | federal employees and Black Lives Matter |

## Verdict on each prediction

- **P1 (flip Bureau of Land Management → Black Lives Matter): CONFIRMED,
  directionally.** Pre-2020 the token *BLM* in the NYT is the land agency;
  by 2020 it is the movement, overwhelmingly. ✅
- **P2 (2014 bimodal at the token level): FALSIFIED.** ❌ In the NYT, 2014
  *BLM* is *entirely* the land agency (the Bundy standoff). The movement used
  the **spelled-out** "Black Lives Matter," not the abbreviation. The
  collision was real at the *event* level (Bundy in April, Ferguson in
  August) but did **not** surface in the bare token in edited prose.
- **P3 (emergence) / P4 (agency obsolescence vs dilution): NOT FORMALLY
  TESTABLE here.** The token *BLM* is too **sparse** in the NYT before 2020
  (2–3 articles/year) to fit a `sense_drift` trajectory or a `decline_report`.
  Qualitatively, the agency sense persists in land/environment reporting
  (Bundy follow-ups, public-lands coverage) through 2024 — i.e. closer to
  dilution than obsolescence — but the counts are too small to call.

## The finding the pre-registration produced

The headline result is **the falsification of P2, and what it reveals**: in
the paper of record, **the abbreviation lagged the movement by roughly six
years.** "BLM" stayed the *land agency* in edited NYT prose until ~2020, then
flipped abruptly when the movement's shorthand finally entered the paper —
*the abbreviation's sense lags the concept's salience in edited text.* This
is the same institutional-adoption lag we measured directly in the
terminology successions (e.g. the AP-Stylebook-2013 → NYT-crossover-2016 lag
for "undocumented"), and it reinforces the polysemy-safeguard thesis: a bare
abbreviation's dominant sense is a property of the *corpus and era*, not the
token.

## Honest limitations

- A bare-token query in a single, edited corpus understates the movement's
  real-world dominance, which lived first in vernacular, social media, and
  spelled-out headlines. To see the *gradual* flip one would need a corpus
  that prints the abbreviation colloquially (the same register lesson as
  Twitter vs PubMed for CBD).
- `sense_drift` was the wrong instrument for *this* corpus/token: it needs a
  dense, consistently-used token, which the NYT's editorial style denies. The
  pre-registration correctly surfaced this by failing P2 rather than being
  quietly fit to the data.
