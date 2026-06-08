# Pre-registration: cross-register polysemy (PubMed vs Wikipedia)

**This document is committed BEFORE any Wikipedia data is retrieved or
inspected.** The commit hash and timestamp constitute the
pre-registration. The PubMed sense compositions referenced below are
already public (in the accompanying methods paper / executed notebook);
the *Wikipedia* sense compositions have not been fetched or viewed at
the time of this commit. The Wikipedia analysis will be added in a
later, separately-timestamped commit.

Purpose: test whether the single-token polysemy problem demonstrated on
PubMed is register-dependent — i.e. whether the sense composition of a
single-token query, and hence the validity of that query, changes when
the corpus changes. A confirmed register shift strengthens the paper's
central claim from "single-token queries mislead in PubMed" to
"single-token queries mislead in a corpus-specific way, so the dominant
sense cannot be assumed."

## Tokens and the known PubMed baseline

Four polysemous tokens, with their already-published PubMed sense shares
(samples of ~3,000 title/abstract records each):

| Token | PubMed intended/drug sense | PubMed share | PubMed modal sense |
|-------|---------------------------|--------------|--------------------|
| CBD   | cannabidiol               | (medical mix: cannabidiol / common bile duct / corticobasal degeneration) | cannabidiol-pharmacology (among classified) |
| weed  | cannabis                  | 1.5%         | agricultural (64.4%) |
| horse | heroin slang              | 0.0%         | equine / unknown   |
| AAS   | anabolic-androgenic       | 9.5%         | unknown (66.6%); astronomy/spectroscopy in residual |

## Data source and method (fixed in advance)

- **Register B:** English Wikipedia, retrieved via the public MediaWiki
  API (`action=query&list=search`, full-text `srsearch`) on the date
  recorded in the results commit. Results reflect that snapshot;
  Wikipedia is versioned, so a reader can reconstruct the nearest dump.
- **Retrieval:** up to **N = 300** articles per token returned by
  full-text search for the bare token, classified by sense from each
  article's lead/extract. This mirrors the PubMed procedure (records
  *mentioning* the token, classified by sense), and measures what a
  naive single-token search returns in each register.
- **Sense inventory (pre-specified now; categories locked):**
  - **CBD:** cannabidiol; central business district; common bile duct;
    corticobasal degeneration; other/unknown.
  - **weed:** cannabis (drug); agricultural/plant weed; place name
    (e.g. Weed, California); media/music; other/unknown.
  - **horse:** equine (animal); heroin slang (drug); seahorse;
    other/metaphor (Trojan horse, sawhorse, etc.); other/unknown.
  - **AAS:** anabolic--androgenic steroids; American Astronomical
    Society (astronomy); atomic absorption spectroscopy; other acronym;
    other/unknown.
  - First-match-wins regex buckets with a conservative `other/unknown`
    residual, published with the results. We commit not to iterate the
    bucket rules after seeing the Wikipedia data; any deviation is
    disclosed.

## Predictions and falsifiers

**P1 (primary — register-dependence).** For **at least 3 of the 4
tokens**, a sense that is rare or absent in PubMed becomes substantial
in Wikipedia, by the following pre-committed per-token thresholds:

- **CBD:** the *central business district* sense is $\geq 10\%$ of
  Wikipedia results (it is essentially absent in PubMed).
- **weed:** the *cannabis* sense is $\geq 10\%$ of Wikipedia results
  (vs 1.5% in PubMed).
- **horse:** the *heroin-slang* sense is $> 0\%$ of Wikipedia results
  (vs 0.0% in PubMed) — i.e. it appears at all.
- **AAS:** the *anabolic--androgenic-steroid* sense is $\geq 20\%$ of
  Wikipedia results (vs 9.5% in PubMed).

**Falsifier F1:** fewer than 3 of the 4 per-token thresholds are met —
i.e. the sense composition does **not** shift meaningfully by register.

**Scoreboard mapping (fixed in advance):**
- **PASS** — $\geq 3$ of 4 per-token thresholds met.
- **PARTIAL** — exactly 2 met.
- **FAIL** — $\leq 1$ met (no register-dependence).

## Explicit non-prediction

We do **not** predict which single sense will be modal on Wikipedia for
every token, nor that the agricultural/equine senses disappear. The
claim is narrower and falsifiable: that at least one sense rare or
absent in PubMed is substantial in Wikipedia for most tokens, so the
dominant sense of a single-token query is corpus-specific.

## Attestation

At the time of this commit the author has not retrieved, viewed, or
computed any Wikipedia sense data for these tokens. Any deviation from
the tokens, retrieval method, sense inventory, thresholds, or scoreboard
mapping above will be disclosed explicitly in the results.
