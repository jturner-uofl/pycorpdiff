"""PubMed (NLM E-utilities) fetcher for the diagnostic-terminology case study.

Two modes:

  * **count** — per-year hit counts for one or more terms. Fast (1 request
    per (term, year) pair). Used for the pilot to confirm that the
    documented terminology shifts are visible in PubMed abstract text
    before we commit to the bulk-abstract fetch.

  * **abstracts** (TODO; not in pilot) — full title + abstract harvest
    for a term over a date range, paginated via the WebEnv/history
    server. Writes parquet.

Why title/abstract text instead of MeSH descriptor? MeSH gets re-tagged
retroactively (e.g. records originally indexed as "Mongolism" got
retagged "Down Syndrome" around 1965), but the original abstract prose
preserves the term-as-published — which is exactly what a semantic-
shift analysis needs.

Rate limits: NLM allows ~3 req/sec without an API key, ~10 req/sec
with one. We default to a 0.4 s inter-request sleep; with --api-key
the script drops to 0.12 s and goes ~3x faster.

Public domain (US government data); free to redistribute.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "pycorpdiff-pubmed-pilot/0.1 (https://github.com/jturner-uofl/pycorpdiff)"


def _http_get_json(url: str, *, max_retries: int = 4) -> dict:
    """GET a JSON endpoint with simple exponential backoff."""
    backoff = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=30) as r:
                return json.load(r)
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            # 429 (rate-limit) and 5xx → retry; 4xx → bubble up immediately
            if isinstance(e, HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"esearch failed after {max_retries} retries: {last_err}")


def esearch_count(
    terms: str | list[str],
    year: int,
    *,
    api_key: str | None = None,
    sleep: float = 0.4,
) -> int:
    """Number of PubMed records matching any of ``terms`` in title/abstract for ``year``.

    ``terms`` is a list of search expressions (each may itself be a single
    word or a multi-word quoted phrase). Each expression is qualified
    *individually* with ``[Title/Abstract]`` before being OR-joined.

    **Why per-term qualification.** Entrez automatic term mapping
    expands unqualified terms via the MeSH synonym set; that expansion
    can drag in *modernised* synonyms which defeats the whole point of
    a deprecated-terminology study. Verified failure mode (June 2026):

        ``(mongolism OR "Mongolian idiocy")[Title/Abstract]``

    is translated by Entrez to a 13-alternation expression that
    includes ``"down syndrome"[MeSH Terms]`` and ``"down syndrome"[All Fields]``,
    so the count for 2020 comes back as ~2,000 — every Down-syndrome
    paper. Wrapping each term individually:

        ``mongolism[Title/Abstract] OR "Mongolian idiocy"[Title/Abstract]``

    suppresses the auto-mapping and returns the literal-prose count
    (0 in 2020 — the term is genuinely extinct in modern medical lit).

    For backward compatibility, passing a single ``str`` still works
    but is wrapped into a one-element list; users are encouraged to
    pass lists so each search expression is properly qualified.
    """
    if api_key:
        sleep = min(sleep, 0.12)
    if isinstance(terms, str):
        terms = [terms]
    if not terms:
        raise ValueError("esearch_count requires at least one search term")
    # Per-term [Title/Abstract] qualification (see docstring on why).
    qualified = " OR ".join(f"{t}[Title/Abstract]" for t in terms)
    params = {
        "db": "pubmed",
        "term": qualified,
        "datetype": "pdat",
        "mindate": str(year),
        "maxdate": str(year),
        "retmax": "0",
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS}/esearch.fcgi?{urlencode(params)}"
    data = _http_get_json(url)
    time.sleep(sleep)
    return int(data["esearchresult"]["count"])


def count_range(
    terms: str | list[str],
    start_year: int,
    end_year: int,
    *,
    api_key: str | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame with columns ``term, year, n_records``."""
    if isinstance(terms, str):
        terms = [terms]
    term_label = " OR ".join(terms)
    rows: list[dict] = []
    n = end_year - start_year + 1
    t0 = time.time()
    for i, year in enumerate(range(start_year, end_year + 1)):
        c = esearch_count(terms, year, api_key=api_key)
        rows.append({"term": term_label, "year": year, "n_records": c})
        if progress:
            elapsed = time.time() - t0
            print(
                f"  [{i + 1:>3}/{n:>3}] {term_label[:40]:<40s} {year}: {c:>6,}  "
                f"({elapsed:.0f}s elapsed)",
                file=sys.stderr,
            )
    return pd.DataFrame(rows)


# ----- Pilot configuration --------------------------------------------

# Three anchor shifts, one per anchor decade. Each pair (old, new) gives a
# clean before-after signal we expect to see in PubMed abstract text.
#
#   1. mongolism / mongoloid     -> Down syndrome / trisomy 21
#      Anchor: Lancet 1961 letter + WHO ICD ~1965.
#   2. shell shock / war neurosis -> post-traumatic stress disorder, PTSD
#      Anchor: DSM-III 1980.
#   3. Asperger / Asperger's syndrome -> autism spectrum disorder, ASD
#      Anchor: DSM-5 May 2013.
#
# For the pilot we sweep yearly counts across the full visible window of
# each pair so the crossover is unambiguous if it exists.

PILOT_TERMS: dict[str, tuple[list[str], int, int]] = {
    # Three-pair confirmation pilot (1960s anchor, 1980s anchor, 2010s anchor).
    # Per-term [Title/Abstract] qualification is applied automatically inside
    # esearch_count() to suppress Entrez auto-mapping into modernised
    # MeSH synonyms.
    #
    # "mongoloid" deliberately dropped — modern uses are "Mongolian spot"
    # (dermatology), a different referent.
    "old_mongolism":     (["mongolism", '"Mongolian idiocy"'],                              1950, 2024),
    "new_down_syndrome": (['"Down syndrome"', '"Down\'s syndrome"', '"trisomy 21"'],         1950, 2024),
    "old_shell_shock":   (['"shell shock"', '"war neurosis"', '"combat fatigue"'],           1940, 2024),
    "new_ptsd":          (['"post-traumatic stress disorder"',
                           '"posttraumatic stress disorder"',
                           "PTSD"],                                                          1940, 2024),
    "old_asperger":      (["Asperger", "Asperger's"],                                         1980, 2024),
    "new_asd":           (['"autism spectrum disorder"', '"autism spectrum disorders"'],    1980, 2024),
}


# Full inventory — ~50 deprecated→modern terminology pairs in psychiatry,
# developmental disability, mood, ADHD, personality, substance, sexual/gender,
# aging, race-in-medical-contexts, and defunct-diagnosis categories. Each is
# given a category prefix in the label so post-hoc grouping is trivial.
#
# A few notes on design:
#   * Race terms (Negro, Oriental, Eskimo) and disability-framing terms
#     (handicapped) are *messier* than the diagnostic-rename pairs because
#     the "modern" replacement words have huge non-replacement use (Asian,
#     Black, disabled are everywhere in non-substitutive contexts). They're
#     included anyway — Step B will pick the cleanest-signal pairs for the
#     deep-dive analysis.
#   * Sensitive items (homosexuality-as-diagnosis, gender-dysphoria, etc.)
#     are included with neutral labels; the paper framing is descriptive,
#     not normative, per the case-study discussion.

FULL_INVENTORY: dict[str, tuple[list[str], int, int]] = {
    # --- A. Intellectual disability / developmental ---
    "ID_old_mongolism":         (["mongolism", '"Mongolian idiocy"'],                              1950, 2024),
    "ID_new_down":              (['"Down syndrome"', '"Down\'s syndrome"', '"trisomy 21"'],         1950, 2024),
    "ID_old_idiot_imbecile":    (['"feeble-minded"', '"feeble minded"', "imbecile", "moron",
                                  "cretin"],                                                        1950, 2024),
    "ID_old_mental_retardation": (['"mental retardation"', '"mentally retarded"',
                                   '"mental retardate"'],                                            1950, 2024),
    "ID_new_intellectual":      (['"intellectual disability"', '"intellectually disabled"',
                                   '"intellectual disabilities"'],                                  1950, 2024),

    # --- B. Autism spectrum ---
    "AUT_old_asperger":         (["Asperger", "Asperger's"],                                         1980, 2024),
    "AUT_new_asd":              (['"autism spectrum disorder"', '"autism spectrum disorders"'],    1980, 2024),
    "AUT_old_pdd":              (['"pervasive developmental disorder"', '"PDD-NOS"',
                                  '"PDD NOS"'],                                                      1980, 2024),
    "AUT_old_infantile":        (['"infantile autism"', '"early infantile autism"'],               1950, 2024),

    # --- C. Mood / psychotic ---
    "MOOD_old_manic_depressive": (['"manic-depressive"', '"manic depressive"',
                                   '"manic depression"', '"manic-depression"'],                     1950, 2024),
    "MOOD_new_bipolar":         (['"bipolar disorder"', '"bipolar I"', '"bipolar II"',
                                   '"bipolar affective"'],                                            1950, 2024),
    "MOOD_old_involutional":    (['"involutional melancholia"', '"involutional depression"'],      1950, 2024),
    "MOOD_old_neurosis":        (["neurosis", "neuroses", "neurotic"],                              1950, 2024),
    "DISSOC_old_mpd":           (['"multiple personality disorder"', '"multiple personality"'],     1950, 2024),
    "DISSOC_new_did":           (['"dissociative identity disorder"'],                              1980, 2024),
    "TRAUMA_old_shell_shock":   (['"shell shock"', '"war neurosis"', '"combat fatigue"'],           1940, 2024),
    "TRAUMA_new_ptsd":          (['"post-traumatic stress disorder"',
                                   '"posttraumatic stress disorder"', "PTSD"],                       1940, 2024),
    "SOMA_old_hysteria":        (["hysteria", '"hysterical neurosis"'],                              1950, 2024),
    "SOMA_new_conversion":      (['"conversion disorder"', '"functional neurological disorder"',
                                  '"somatic symptom disorder"'],                                     1950, 2024),

    # --- D. ADHD / attention ---
    "ADHD_old_mbd":             (['"minimal brain dysfunction"', '"minimal brain damage"',
                                  '"hyperkinetic reaction"'],                                        1950, 2024),
    "ADHD_new":                 (["ADHD", '"attention deficit hyperactivity"',
                                  '"attention-deficit"'],                                            1970, 2024),

    # --- E. Personality / antisocial ---
    "PERS_old_psychopathy":     (["psychopath", "psychopathy", "psychopathic",
                                  '"moral insanity"'],                                                1950, 2024),
    "PERS_new_aspd":            (['"antisocial personality disorder"', "ASPD"],                     1970, 2024),

    # --- F. Substance use ---
    "SUD_old_alcoholism":       (["alcoholism", "alcoholic"],                                        1950, 2024),
    "SUD_new_aud":              (['"alcohol use disorder"', '"alcohol use disorders"'],            1990, 2024),
    "SUD_old_addict":           (['"drug addict"', '"drug addicts"', '"drug addiction"'],          1950, 2024),
    "SUD_new_sud":              (['"substance use disorder"', '"substance use disorders"'],        1990, 2024),
    "SUD_old_crack_baby":       (['"crack baby"', '"crack babies"', '"cocaine baby"',
                                  '"cocaine babies"'],                                               1980, 2024),
    "SUD_new_nas":              (['"neonatal abstinence syndrome"'],                                 1950, 2024),

    # --- G. Sexual / gender (descriptive labels only) ---
    "GEN_old_transvestism":     (["transvestism", "transsexualism", "transsexual"],                 1950, 2024),
    "GEN_new_dysphoria":        (['"gender dysphoria"', '"gender identity disorder"'],              1970, 2024),
    "GEN_old_hermaphrodite":    (["hermaphrodite", "hermaphroditism", "hermaphroditic"],            1950, 2024),
    "GEN_new_dsd_intersex":     (['"disorder of sex development"',
                                  '"disorders of sex development"', "intersex"],                     1990, 2024),
    "SEX_old_frigidity":        (["frigidity", "frigid"],                                            1950, 2024),
    "SEX_old_impotence":        (["impotence", "impotent"],                                          1950, 2024),
    "SEX_new_ed":               (['"erectile dysfunction"'],                                         1970, 2024),
    "SEX_old_nymphomania":      (["nymphomania", "nymphomaniac"],                                    1950, 2024),

    # --- H. Aging / cognitive ---
    "DEM_old_senility":         (["senility", '"senile dementia"', "senile"],                       1950, 2024),
    "DEM_new_alzheimer":        (["Alzheimer", '"Alzheimer\'s disease"', '"Alzheimer disease"'],   1950, 2024),
    "DEM_old_obs":              (['"chronic brain syndrome"', '"organic brain syndrome"'],         1950, 2024),

    # --- I. Disability framing ---
    "DIS_old_handicapped":      (["handicapped", '"physically handicapped"',
                                  '"mentally handicapped"'],                                          1950, 2024),
    "DIS_old_crippled":         (["crippled", "cripple"],                                            1950, 2024),
    "DIS_old_deaf_mute":        (['"deaf-mute"', '"deaf mute"', '"deaf and dumb"'],                 1950, 2024),

    # --- J. Race in medical contexts (descriptive use, not diagnosis) ---
    "RACE_old_negro":           (["Negro", "Negroid"],                                                1950, 2024),
    "RACE_new_african_amer":    (['"African American"', '"African-American"',
                                  '"African Americans"'],                                             1950, 2024),
    "RACE_old_oriental":        (["Oriental"],                                                        1950, 2024),
    "RACE_old_eskimo":          (["Eskimo"],                                                          1950, 2024),
    "RACE_new_inuit":           (["Inuit"],                                                           1950, 2024),
    "RACE_old_caucasian":       (["Caucasian", "Caucasians"],                                         1950, 2024),
    "RACE_new_white_european":  (['"European descent"', '"non-Hispanic white"'],                    1950, 2024),

    # --- K. Suicide framing ---
    "SUI_old_committed":        (['"committed suicide"', '"commits suicide"'],                       1970, 2024),
    "SUI_new_died_by":          (['"died by suicide"'],                                              1990, 2024),

    # --- L. Defunct diagnoses (negative-control candidates) ---
    "DEFUNCT_drapetomania":     (["drapetomania"],                                                    1950, 2024),
    "DEFUNCT_neurasthenia":     (["neurasthenia", "neurasthenic"],                                    1950, 2024),

    # --- M. Child welfare ---
    "CHILD_old_battered":       (['"battered child syndrome"', '"battered child"'],                  1960, 2024),
    "CHILD_old_shaken":         (['"shaken baby syndrome"', '"shaken infant syndrome"'],            1970, 2024),
    "CHILD_new_aht":            (['"abusive head trauma"'],                                          1990, 2024),
}


# Tier 2: explicitly stigmatized / loaded historical clinical vocabulary.
# Same per-term [Title/Abstract] discipline as the headline inventory; the
# point is to empirically document *when* and *how completely* the medical
# literature retired each loaded term. The analysis is descriptive; we are
# not endorsing or rehabilitating any of these terms.

TIER2_INVENTORY: dict[str, tuple[list[str], int, int]] = {
    # --- Eugenic-era IQ classification (Goddard 1914 scale; became slurs) ---
    "T2_moron":              (["moron", "morons", "moronic"],                                          1950, 2024),
    "T2_imbecile":           (["imbecile", "imbeciles", "imbecility"],                                  1950, 2024),
    "T2_idiocy_clinical":    (["idiocy", '"Mongolian idiocy"', '"amaurotic idiocy"'],                  1950, 2024),
    "T2_feeble_minded":      (['"feeble-minded"', '"feeble minded"', "feeblemindedness",
                               '"feeble-mindedness"'],                                                  1950, 2024),
    "T2_mental_defective":   (['"mental defective"', '"mental defectives"', '"mental deficiency"'],   1950, 2024),
    "T2_cretin":             (["cretin", "cretins", "cretinism", "cretinoid"],                          1950, 2024),
    "T2_mongoloid_idiot":    (['"mongoloid idiot"', '"mongoloid idiocy"', '"mongol idiocy"'],          1950, 2024),

    # --- Sexual orientation as pathology ---
    # Overall topic volume (will show post-1973 reframing as the COUNT staying
    # large but the keyness CONTEXT shifting):
    "T2_homosexuality":      (["homosexuality"],                                                        1950, 2024),
    # Explicitly pathology-framed phrases (much smaller volume, clear retirement):
    "T2_homosexuality_dx":   (['"homosexual disorder"', '"homosexual deviation"',
                               '"homosexual neurosis"', '"latent homosexuality"',
                               '"treatment of homosexuality"', '"cure of homosexuality"',
                               '"etiology of homosexuality"'],                                          1950, 2024),
    "T2_ego_dystonic":       (['"ego-dystonic homosexuality"', '"ego dystonic homosexuality"'],       1970, 2024),
    "T2_sexual_inversion":   (['"sexual inversion"', '"sexual invert"', '"sexual inverts"',
                               '"congenital invert"'],                                                  1950, 2024),
    "T2_sexual_perversion":  (['"sexual perversion"', '"sexual pervert"', '"sexual perversions"'],    1950, 2024),
    "T2_sodomy_clinical":    (["sodomy", "sodomite", '"sodomitical"'],                                  1950, 2024),

    # --- Misogynistic women's-sexuality clinical terms ---
    "T2_frigidity":          (["frigidity", "frigid"],                                                  1950, 2024),
    "T2_nymphomania":        (["nymphomania", "nymphomaniac", "nymphomaniacal"],                        1950, 2024),
    "T2_onanism":            (["onanism", '"self-abuse"', '"self abuse"', '"self-pollution"'],         1950, 2024),

    # --- 19th-century race-pathology pseudo-diagnoses ---
    "T2_drapetomania":       (["drapetomania"],                                                          1950, 2024),
    "T2_dysaesthesia_aeth":  (['"dysaesthesia aethiopica"', '"dysesthesia aethiopica"'],               1950, 2024),
    "T2_negroid_facies":     (['"Negroid facies"', '"Negroid skull"', '"Negroid features"'],          1950, 2024),

    # --- Discredited / contested treatments ---
    "T2_lobotomy":           (["lobotomy", "leukotomy", '"prefrontal lobotomy"',
                               '"transorbital lobotomy"'],                                              1940, 2024),
    "T2_insulin_coma":       (['"insulin coma"', '"insulin shock therapy"',
                               '"insulin coma therapy"'],                                                1940, 2024),
    "T2_aversion_therapy":   (['"aversion therapy"', '"aversion conditioning"',
                               '"electrical aversion"'],                                                1950, 2024),
    "T2_conversion_therapy": (['"conversion therapy"', '"reparative therapy"',
                               '"sexual orientation change"', '"sexual reorientation"'],                1950, 2024),

    # --- Disability slurs from clinical origin ---
    "T2_spastic_clinical":   (['"spastic child"', '"spastic children"', '"the spastics"',
                               '"spastic diplegic"'],                                                    1950, 2024),

    # --- Substance-use stigma ---
    "T2_junkie":             (["junkie", "junkies"],                                                    1950, 2024),
    "T2_dope_fiend":         (['"dope fiend"', '"dope fiends"', '"dope addict"', '"dope addicts"'],  1950, 2024),

    # --- Reproductive stigma ---
    "T2_illegitimate":       (['"illegitimate child"', '"illegitimate children"',
                               "illegitimacy"],                                                          1950, 2024),
    "T2_unwed_mother":       (['"unwed mother"', '"unwed mothers"', '"out of wedlock"'],              1950, 2024),

    # === iter-4 additions (Jun 2026) — labels missed in the original
    # iter-1/2/3 inventory, added after the iter-4 brainstorm pass.

    # --- Era-clinical psychiatric categories that drifted/retired ---
    "T2_hysteria":           (["hysteria", "hysterical", "hysterics",
                               '"hysterical neurosis"'],                                                 1950, 2024),
    # iter-5b: add plural form.
    "T2_neurasthenia":       (["neurasthenia", "neurasthenic", "neurasthenics"],                         1950, 2024),
    "T2_moral_insanity":     (['"moral insanity"', '"moral imbecile"',
                               '"moral imbecility"'],                                                    1950, 2024),
    "T2_puerperal_insanity": (['"puerperal insanity"', '"puerperal psychosis"',
                               '"puerperal mania"'],                                                     1950, 2024),
    # iter-5b: add plurals + derivational forms.
    "T2_psychopath_socio":   (["psychopath", "psychopaths", "psychopathy", "psychopathic",
                               "sociopath", "sociopaths", "sociopathy", "sociopathic",
                               '"psychopathic personality"'],                                            1950, 2024),

    # --- Intersex / trans clinical terms (DSM-IV->DSM-5 + 2006 Chicago intersex consensus) ---
    # iter-5b morphological-completeness pass: add singular/plural/adjective forms.
    "T2_hermaphrodite":      (["hermaphrodite", "hermaphrodites", "hermaphroditism",
                               "hermaphroditic",
                               '"pseudohermaphrodite"', '"pseudohermaphrodites"',
                               '"pseudohermaphroditism"',
                               '"true hermaphrodite"'],                                                  1950, 2024),
    "T2_transsexual_xvest":  (["transsexual", "transsexuals", "transsexualism", "transsexuality",
                               "transvestite", "transvestites", "transvestism", "transvestic",
                               '"gender identity disorder"'],                                            1950, 2024),

    # --- Substance-use historical ---
    # iter-5b: add plurals + -ness/-y derivational forms.
    "T2_drunkard_inebriate": (["drunkard", "drunkards", "drunkenness",
                               "inebriate", "inebriates", "inebriety",
                               "dipsomania", "dipsomaniac", "dipsomaniacs"],                             1950, 2024),

    # --- Race-medicine descriptors that have measurable clinical-literature footprint ---
    "T2_mongolian_spot":     (['"Mongolian spot"', '"Mongolian spots"',
                               '"Mongolian blue spot"'],                                                 1950, 2024),
    # iter-5b: add bare "Negroid" case-descriptor in addition to phrase forms.
    "T2_negroid_descriptor": (["Negroid",
                               '"Negroid race"', '"Negroid descent"',
                               '"Negroid ancestry"', '"Negroid patient"',
                               '"Negroid subject"'],                                                     1950, 2024),

    # --- Retired-treatment vocabulary ---
    "T2_electroshock":       (["electroshock", '"electric shock therapy"',
                               '"shock therapy"'],                                                       1940, 2024),
    "T2_metrazol":           (["metrazol", "cardiazol", '"convulsive therapy"',
                               '"pentylenetetrazol convulsive"'],                                        1940, 2024),
    "T2_deep_sleep_therapy": (['"deep sleep therapy"', '"continuous narcosis"',
                               '"sleep cure"'],                                                          1940, 2024),
    # iter-5b: add adjective + plural derivational forms.
    "T2_psychosurgery":      (["psychosurgery", "psychosurgical", "psychosurgeries",
                               '"prefrontal leucotomy"',
                               '"prefrontal leukotomy"'],                                                1940, 2024),
}


# Tier 3: the most-offensive deprecated medical vocabulary. Included for
# completeness and honest empirical documentation: we are tracking *what
# the medical literature actually published* and *when those terms were
# retired*. Modern PubMed indexing may have scrubbed some of the most
# egregious historical content, so several of these will return ~zero
# even where the original publications used the term — which is itself a
# data point about post-hoc indexing curation.
#
# This is descriptive empirical history. No endorsement of these terms.

TIER3_INVENTORY: dict[str, tuple[list[str], int, int]] = {
    # --- Morpheme `retard*` (NOT the slur — see notebook §6.5.1 audit-resolved
    # disambiguation; this label measures the verb/adjective morpheme across
    # all senses, dominated by scientific process-verb uses). Renamed from
    # T3_retarded_slur after the iter-1 audit refutation. ---
    # iter-5b morphological-completeness pass: include all common inflectional
    # and derivational forms of `retard*` so the WSI corpus captures the full
    # morpheme rather than just the verb + noun + past-participle.
    "T3_retarded_morpheme": (["retarded", "retards", "retard", "retardation",
                              "retarding", "retardations",
                              "retardant", "retardants"],                                                1950, 2024),

    # --- 19th-c racial anthropology / colonial tropical medicine ---
    "T3_hottentot":         (["Hottentot", "Hottentots", '"Hottentot Venus"', '"Hottentot apron"'],   1950, 2024),
    "T3_kaffir":            (["kaffir", "kaffirs"],                                                     1950, 2024),
    # Iter-4 ethical-review removal: T3_savage_primitive (4 records),
    # T3_darky (5), T3_n_word (0), T3_freak (0) were dropped from the
    # inventory because they returned ~zero records — the inclusion was
    # ethically defensible to attempt but did not earn its place in the
    # final reporting once the per-label counts were known. The labels
    # that DO earn inclusion are those whose query returned enough records
    # to support per-year sense decomposition.

    # --- Teratology / pediatric / "freak" historical ---
    "T3_monster_clinical":  (['"congenital monster"', '"congenital monstrosity"',
                              '"human monster"', '"monstrous birth"',
                              '"acardiac monster"'],                                                    1950, 2024),

    # --- Short-stature informal terms ---
    "T3_midget":            (["midget", "midgets"],                                                     1950, 2024),
    # iter-5b: add plurals + verb forms.
    "T3_dwarf_clinical":    (["dwarfism", "dwarf", "dwarfs", "dwarves",
                              "dwarfed", "dwarfing",
                              '"primordial dwarf"'],                                                    1950, 2024),

    # --- Legal-medical with stigma ---
    "T3_bastard":           (['"bastard child"', '"bastard children"', '"bastardy"'],                  1950, 2024),
    "T3_lunatic":           (["lunatic", "lunatics", '"lunatic asylum"', "lunacy"],                    1950, 2024),

    # --- STI / venereal disease era stigma ---
    "T3_whore_harlot":      (["whore", "whores", "harlot", "harlots", '"common prostitute"'],          1950, 2024),

    # --- Disability slurs (more explicit than DIS_old_*) ---
    "T3_deformed":          (['"deformed child"', '"deformed children"', '"hideously deformed"',
                              '"facial deformity"'],                                                    1950, 2024),
    # NOTE: iter-2 audit (random 20-PMID inspection of peak year) found this
    # label's records were 7/8 era-clinical IQ classification, NOT slur usage.
    # Renamed _slur -> _clinical to reflect the validated construct.
    # iter-5b: add singular form (previously missing!) and adjective.
    "T3_imbecile_clinical": (["imbecile", "imbeciles", "imbecility",
                              "imbecilic"],                                                             1950, 2024),

    # === iter-4 additions (Jun 2026) — Tier-3 entries missed in the
    # original brainstorm.

    # --- Race-medicine: colonial anthropology + retired clinical compounds ---
    "T3_bushman":            (["Bushman", "Bushmen"],                                                    1950, 2024),
    "T3_oriental_disease":   (['"Oriental sore"', '"Oriental cholera"',
                               '"Oriental schistosomiasis"',
                               '"Oriental boil"'],                                                       1950, 2024),
    # iter-5b: add bare "lazar" + "leprosy" (the disease name).
    "T3_lazar_leper":        (["leper", "lepers", "leprosy", "lazar", "lazars",
                               '"leper colony"',
                               "lazaretto", "leprous"],                                                  1950, 2024),

    # --- Disability/orthopedic clinical-era stigma ---
    # iter-5b: add plurals + -ing form.
    "T3_cripple":            (["cripple", "cripples", "crippled", "crippling",
                               '"crippled child"', '"crippled children"',
                               '"Crippled Children\'s Services"'],                                       1950, 2024),
    "T3_deaf_mute":          (['"deaf-mute"', '"deaf mute"', '"deaf-mutes"',
                               '"deaf and dumb"'],                                                       1950, 2024),
    "T3_siamese_twins":      (['"Siamese twins"', '"Siamese twin"'],                                     1950, 2024),
    "T3_hunchback":          (["hunchback", "hunchbacked", "hunchbacks"],                                1950, 2024),

    # --- Older psychiatric / mental-illness vocabulary ---
    # iter-5b: add plurals + adjective + feminine forms.
    "T3_maniac_madhouse":    (["maniac", "maniacs", "maniacal",
                               "madman", "madmen", "madwoman", "madwomen",
                               "madhouse", "madhouses"],                                                 1950, 2024),
}


def run_inventory(
    inventory: dict[str, tuple[list[str], int, int]],
    out_csv: Path,
    *,
    api_key: str | None = None,
    only: list[str] | None = None,
    cache_dir_name: str = "pubmed_pilot_cache",
) -> pd.DataFrame:
    """Run a (label -> (terms, start, end)) inventory. Caches per-label CSV."""
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    cache_dir = out_csv.parent / cache_dir_name
    cache_dir.mkdir(exist_ok=True)

    all_frames = []
    for label, (terms, start, end) in inventory.items():
        if only and label not in only:
            continue
        cache = cache_dir / f"{label}.csv"
        if cache.exists():
            print(f"[cache] {label}: reusing {cache}", file=sys.stderr)
            df = pd.read_csv(cache)
        else:
            term_label = " OR ".join(terms)
            print(f"[fetch] {label}: {term_label[:60]}... [{start}-{end}]", file=sys.stderr)
            df = count_range(terms, start, end, api_key=api_key)
            df["label"] = label
            df.to_csv(cache, index=False)
        if "label" not in df.columns:
            df["label"] = label
        all_frames.append(df)

    full = pd.concat(all_frames, ignore_index=True)
    full = full[["label", "term", "year", "n_records"]]
    full.to_csv(out_csv, index=False)
    print(f"\nWrote {len(full):,} rows to {out_csv}", file=sys.stderr)
    return full


def run_pilot(
    out_csv: Path,
    *,
    api_key: str | None = None,
    only: list[str] | None = None,
) -> pd.DataFrame:
    """Run the three-shift confirmation pilot."""
    return run_inventory(PILOT_TERMS, out_csv, api_key=api_key, only=only,
                         cache_dir_name="pubmed_pilot_cache")


def run_full(
    out_csv: Path,
    *,
    api_key: str | None = None,
    only: list[str] | None = None,
) -> pd.DataFrame:
    """Run the full ~50-pair deprecated-terminology inventory."""
    return run_inventory(FULL_INVENTORY, out_csv, api_key=api_key, only=only,
                         cache_dir_name="pubmed_full_cache")


def run_tier2(
    out_csv: Path,
    *,
    api_key: str | None = None,
    only: list[str] | None = None,
) -> pd.DataFrame:
    """Run the Tier-2 (explicitly stigmatized historical vocabulary) inventory."""
    return run_inventory(TIER2_INVENTORY, out_csv, api_key=api_key, only=only,
                         cache_dir_name="pubmed_tier2_cache")


def run_tier3(
    out_csv: Path,
    *,
    api_key: str | None = None,
    only: list[str] | None = None,
) -> pd.DataFrame:
    """Run the Tier-3 (most-offensive deprecated medical vocabulary) inventory."""
    return run_inventory(TIER3_INVENTORY, out_csv, api_key=api_key, only=only,
                         cache_dir_name="pubmed_tier3_cache")


def summarise(df: pd.DataFrame) -> str:
    """Tabular summary of the pilot — counts by label × decade."""
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10
    pivot = (
        df.groupby(["label", "decade"])["n_records"]
        .sum()
        .unstack(fill_value=0)
        .astype(int)
    )
    return pivot.to_string()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--api-key",
        default=None,
        help="NCBI E-utilities API key (raises rate limit 3/s -> 10/s).",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict run to a subset of labels.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Run the full ~50-pair deprecated-terminology inventory instead of "
             "the three-shift pilot.",
    )
    p.add_argument(
        "--tier2",
        action="store_true",
        help="Run the Tier-2 (explicitly stigmatized historical vocabulary) "
             "inventory: eugenic-era IQ classification, sexual-orientation "
             "pathology, race-pathology pseudo-diagnoses, discredited "
             "treatments, women's-sexuality clinical terms, etc.",
    )
    p.add_argument(
        "--tier3",
        action="store_true",
        help="Run the Tier-3 (most-offensive deprecated medical vocabulary) "
             "inventory: explicit slur forms, 19th-century colonial racial "
             "medical anthropology, teratology, legal-medical stigma, etc. "
             "Some may return ~zero from modern PubMed indexing; that is "
             "itself a data point.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: data/pubmed_pilot_counts.csv, "
             "_full_counts.csv, or _tier2_counts.csv depending on mode).",
    )
    args = p.parse_args(argv)

    if sum([args.full, args.tier2, args.tier3]) > 1:
        p.error("--full / --tier2 / --tier3 are mutually exclusive")

    if args.tier3:
        default_csv = "pubmed_tier3_counts.csv"
    elif args.tier2:
        default_csv = "pubmed_tier2_counts.csv"
    elif args.full:
        default_csv = "pubmed_full_counts.csv"
    else:
        default_csv = "pubmed_pilot_counts.csv"
    default_out = Path(__file__).resolve().parents[1] / "data" / default_csv
    out = args.out or default_out

    if args.tier3:
        print(f"Running Tier-3 inventory ({len(TIER3_INVENTORY)} labels)...",
              file=sys.stderr)
        df = run_tier3(out, api_key=args.api_key, only=args.only)
    elif args.tier2:
        print(f"Running Tier-2 inventory ({len(TIER2_INVENTORY)} labels)...",
              file=sys.stderr)
        df = run_tier2(out, api_key=args.api_key, only=args.only)
    elif args.full:
        print(f"Running full inventory ({len(FULL_INVENTORY)} labels)...",
              file=sys.stderr)
        df = run_full(out, api_key=args.api_key, only=args.only)
    else:
        df = run_pilot(out, api_key=args.api_key, only=args.only)
    print("\n=== Per-decade summary (n_records) ===\n", file=sys.stderr)
    print(summarise(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
