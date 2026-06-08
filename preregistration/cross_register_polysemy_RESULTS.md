# Results: cross-register polysemy (PubMed vs Wikipedia)

**Temporal trail:** pre-registration `d99f7ed` (2026-06-08T14:48:16-04:00),
before any fetch. Wikipedia retrieved 2026-06-08 via MediaWiki full-text
search (N=300/token), classified by the pre-specified sense buckets.
Verdicts apply the locked thresholds unchanged.

| Token | Pre-registered threshold | Wikipedia | PubMed baseline | Met |
|-------|--------------------------|-----------|-----------------|-----|
| CBD   | central-business-district >= 10% | 34.3% | ~0% | YES |
| weed  | cannabis >= 10%                  | 13.7% | 1.5% | YES |
| horse | heroin-slang > 0%                | 0.0%  | 0.0% | NO  |
| AAS   | anabolic-androgenic >= 20%       | 18.7% | 9.5% | NO  |

**2 of 4 thresholds met -> PRE-REGISTERED VERDICT: PARTIAL.**

## Honest reading (reported as observed, not as hoped)

The substantive claim -- that single-token sense composition is
register-dependent -- is clearly supported for CBD and weed, and the
formal PARTIAL reflects two pre-registered thresholds that were
miscalibrated, not an absence of the effect:

- **CBD (strong flip):** the modal *classified* sense flips from medical
  (PubMed: cannabidiol / common-bile-duct / corticobasal) to
  **central business district** (34.3%) on Wikipedia; the anatomical and
  neurological senses that are substantial in PubMed nearly vanish (2.3%).
- **weed (clear shift):** cannabis is ~9x its PubMed share (13.7% vs
  1.5%), and *place* and *media* senses appear (9.7%, 11.7%) that do not
  exist in the PubMed records.
- **AAS (rose, missed the bar):** the anabolic-androgenic sense doubled
  (9.5% -> 18.7%) and the sense mix differs (an "other-acronym" sense at
  18.7%), but it fell 1.3 points short of the pre-registered 20% bar.
- **horse (a genuine null):** the heroin-slang sense is absent in *both*
  registers (0.0%); equine dominates everywhere (94%). Not every
  polysemy is register-dependent -- some senses are simply rare in
  formal writing of any kind. This is recorded as observed.

## Deviations from pre-registration

None. Tokens, retrieval method, sense inventory, thresholds, and the
PASS/PARTIAL/FAIL mapping are exactly as committed in
`cross_register_polysemy.md`.
