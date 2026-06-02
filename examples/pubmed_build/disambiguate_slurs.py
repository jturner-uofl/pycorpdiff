"""Multi-label word-sense disambiguation for slur-like Tier-3 labels.

Iter-4 extension of the iter-2 audit-resolved retard* WSI in
disambiguate_retard.py. Applies the same random-PMID + regex-bucket
discipline to every slur-like Tier-3 label with enough records to
support per-year sense decomposition.

Labels processed (intent):

  T3_dwarf_clinical (15,464)  plant breeding / pituitary / skeletal / slur
  T3_lunatic         (639)    Lunatic Fringe gene / asylum / slur
  T3_midget          (379)    retinal midget cells / hockey / pituitary / slur
  T2_moron           (144)    bacteriophage gene / moronic acid / IQ / slur
  T3_imbecile        (111)    era-clinical IQ / slur
  T3_whore_harlot     (67)    historical STI / slur
  T3_hottentot        (61)    Khoisan anthropology / racial-medical / slur
  T3_kaffir           (53)    kaffir lime / racial / slur
  T3_monster_clinical (39)    congenital monstrosity / slur

For each label we:
  1. esearch the per-label query (using the same per-term-qualified
     `[Title/Abstract]` discipline as fetch_pubmed_abstracts.py)
  2. efetch all matching records 1950-2024
  3. first-match-wins regex classification into per-label sense
     buckets. Slur-explicit-mention is LAST so that any text matching
     a dominant non-slur sense first counts there — that's the
     conservative direction relative to the slur narrative.
  4. write per-(year, sense) CSV per label
  5. write a combined CSV with one row per (label, year, sense)

The §6.5 notebook reads these CSVs and renders per-label sense
decomposition tables + stacked-area charts.

Public domain (US-gov NLM E-utilities) — same as fetch_pubmed_abstracts.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_pubmed_abstracts import (  # noqa: E402
    esearch_pmids,
    efetch_records,
)


# ----------------------- Per-label sense regexes -----------------------

# Each entry: (sense_label, regex_pattern). Order matters: first-match
# wins. Dominant non-slur senses come first; slur_explicit_mention is
# LAST (a record discussing the slur AND its clinical history will
# usually classify clinical, biasing toward the conservative reading
# that the slur sense is overstated, not understated).


@dataclass
class SlurWSIConfig:
    label: str
    esearch_terms: list[str]
    sense_patterns: list[tuple[str, str]]   # (sense_name, regex)
    start_year: int = 1950
    end_year: int = 2024

    # Compiled regexes — populated lazily.
    compiled: list[tuple[str, re.Pattern]] = field(default_factory=list)

    def compile(self) -> None:
        self.compiled = [(s, re.compile(p, re.IGNORECASE))
                         for s, p in self.sense_patterns]


# --- Sense pattern library. Tuned by hand from inspection of
#     iter-2 audit samples + biomed/clinical lit knowledge.
#     "slur_explicit_mention" is always the LAST entry.

SLUR_WSI_CONFIGS: dict[str, SlurWSIConfig] = {

    "T3_dwarf_clinical": SlurWSIConfig(
        label="T3_dwarf_clinical",
        esearch_terms=["dwarfism", "dwarf", "dwarfs", "dwarves",
                       "dwarfed", "dwarfing",
                       '"primordial dwarf"'],
        sense_patterns=[
            ("plant_breeding",
             r"\b(semi[\-\s]?dwarf|dwarf\s+(wheat|rice|sorghum|maize|barley|rye|tomato|"
             r"pea|bean|cotton|mutant|cultivar|variety|line|hybrid|allele|plant|crop|tree|"
             r"fruit|seedling|grass|root\s?stock|fig|apple)|"
             r"(dwarfing|dwarfism)\s+(gene|trait|allele|locus|qtl)|"
             r"breeding\s+for\s+dwarf|green\s+revolution|norin\s*10|"
             r"\b(rht|gai|sd1|d1|d8)\s+(gene|locus|allele|mutant)|gibberellin\s+(insensitiv|deficien))\b"),
            ("clinical_pituitary_endocrine",
             r"\b(pituitary\s+dwarf|growth\s+hormone\s+(deficien|insensitiv|resist)|"
             r"laron\s+syndrome|hypopituitar|isolated\s+gh\s+deficien|"
             r"primordial\s+dwarf|seckel|russell.silver|mulibrey|"
             r"dwarfism\s+(syndrome|disorder|associated\s+with|in\s+(humans|children|patients|infants)|"
             r"due\s+to|caused\s+by)|congenital\s+(growth\s+hormone\s+deficien|hypothyroid).*dwarf)\b"),
            ("clinical_skeletal_dysplasia",
             r"\b(skeletal\s+dysplasia|achondropla|hypochondropla|spondyloepiphyseal|"
             r"metaphyseal\s+dysplasia|chondrodysplasia|fgfr3\s+mutation|"
             r"osteogenesis\s+imperfect|disproportionate\s+(short\s+)?stature|"
             r"thanatophoric\s+dysplasia|diastrophic\s+dysplasia|kniest)\b"),
            ("biology_animal_model",
             r"\b(dwarf\s+(mouse|mice|rat|chicken|zebrafish|drosophila|c\.\s*elegans|nematode|"
             r"hamster|cattle|goat|horse|pig|sheep)|snell\s+dwarf|ames\s+dwarf|little\s+mouse|"
             r"lit/lit|df/df)\b"),
            ("historical_circus_performer",
             r"\b(circus\s+(dwarf|midget|performer)|carnival\s+(dwarf|midget)|"
             r"dwarf\s+(entertainer|actor|performer)|sideshow|p\.\s*t\.?\s*barnum|"
             r"general\s+tom\s+thumb|"
             r"(\"|')little\s+people(\"|'))\b"),
            ("slur_explicit_mention",
             r"\b((dwarf|midget)\s+(slur|epithet|insult|derogat|offensive|pejorative|stigmatiz)|"
             r"(\"|')(dwarf|midget)(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as)\s+(offensive|pejorative|derogat|stigmatiz))|"
             r"m\-word|"
             r"reclaim(ed|ing)?\s+(the\s+term\s+)?(dwarf|midget)|"
             r"(little\s+person|person\s+with\s+dwarfism)\s+(rather\s+than|instead\s+of|not)\s+(dwarf|midget))\b"),
        ],
    ),

    "T3_lunatic": SlurWSIConfig(
        label="T3_lunatic",
        esearch_terms=["lunatic", "lunatics", '"lunatic asylum"', "lunacy"],
        sense_patterns=[
            ("lunatic_fringe_gene",
             r"\b(lunatic\s+fringe|\blfng\b|fringe\s+(family|paralog|homolog)|"
             r"\blfng[\-\s/]?(deficien|knockout|null|mutant|express|gene|promoter|protein|/?\-?)|"
             r"notch\s+signal.*(lunatic|fringe)|o\-fucosyltransfer)\b"),
            ("historical_asylum_lunacy_law",
             r"\b(lunatic\s+asylum|lunacy\s+(act|commission|board|reform|inquiry|law|hearings?)|"
             r"board\s+of\s+lunacy|county\s+(asylum|lunatic)|state\s+lunatic|"
             r"asylum\s+for\s+(the\s+)?(lunatic|insane)|criminal\s+lunatic|certified\s+lunatic|"
             r"lunatic\s+ward|(insane|lunatic)\s+pauper|lunatic\s+hospital)\b"),
            ("history_of_psychiatry",
             r"\b(history\s+of\s+(lunacy|psychiatry|the\s+(insane|lunatic|asylum))|"
             r"alienist|moral\s+treatment|asylum\s+(physician|superintendent|reform)|"
             r"victorian\s+(psychiatry|asylum)|nineteenth\-?century\s+(psychiatry|lunatic|asylum))\b"),
            ("informal_extreme_behavior",
             r"\b(lunatic\s+(behavior|behaviour|ravings?|ideas|notions)|"
             r"complete(ly)?\s+lunatic|absolute\s+lunatic|drove\s+(him|her|them|me)\s+lunatic|"
             r"lunatic\s+ideas)\b"),
            ("slur_explicit_mention",
             r"\b(lunatic\s+(slur|epithet|insult|derogat|offensive|pejorative|stigmatiz)|"
             r"(\"|')lunatic(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed)\s+(offensive|pejorative|derogat|stigmatiz))|"
             r"reclaim.*lunatic|use\s+of\s+(the\s+term\s+)?(\"|')?lunatic(\"|')?\s+(is|has\s+been)\s+(offensive|pejorative|derogat))\b"),
        ],
    ),

    "T3_midget": SlurWSIConfig(
        label="T3_midget",
        esearch_terms=["midget", "midgets"],
        sense_patterns=[
            ("retinal_midget_cells",
             r"\b(midget\s+(bipolar|ganglion|cell|pathway|circuit|system|cone|response|receptive\s+field)|"
             r"on\-?midget|off\-?midget|parvocellular.*midget|retinal\s+midget|"
             r"primate\s+(retina|fovea).*midget|"
             r"\bmidgets?\s+(form|receive|relay|encode|project|carry|provide))\b"),
            ("youth_sports_league",
             r"\b(midget\s+(hockey|league|tournament|championship|cup|category|division|level|player|"
             r"aged?|class|team|trial|competition|football|baseball|basketball|soccer|sport)|"
             r"aged?\s+1[3-9].*midget|youth\s+ice\s+hockey|peewee.*midget|under\-?(14|15|16|17)|"
             r"minor\s+hockey)\b"),
            ("historical_pituitary_clinical",
             r"\b(pituitary\s+midget|midget\s+patient|true\s+midget|primordial\s+midget|"
             r"midget\s+stature|midget\s+race|midget\s+with\s+(growth|hypopituitar|gh)|"
             r"hypopituitar.*midget)\b"),
            ("optical_engineering",
             r"\b(midget\s+(lens|microscope|telescope|electrode|chamber|detector|sensor|antenna|"
             r"valve|mount|coil|implant|cup|dose|fixation|amplifier|tube|battery|capsule))\b"),
            ("informal_small_object",
             r"\b(midget\s+(submarine|car|race|version|sized?|model|specimen|portion|edition|format))\b"),
            ("slur_explicit_mention",
             r"\b(midget\s+(slur|epithet|insult|derogat|offensive|pejorative|stigmatiz)|"
             r"(\"|')midget(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as)\s+(offensive|pejorative|derogat|stigmatiz))|"
             r"m\-word|reclaim.*midget|"
             r"(little\s+person|person\s+with\s+dwarfism)\s+(rather\s+than|instead\s+of|not)\s+midget|"
             r"disability\s+(activist|community|advocate).*midget)\b"),
        ],
    ),

    "T2_moron": SlurWSIConfig(
        label="T2_moron",
        esearch_terms=["moron", "morons"],
        sense_patterns=[
            ("bacteriophage_moron_element",
             r"\b(moron\s+(gene|element|sequence|cassette|insertion|content|cluster|region|orf)|"
             r"prophage\s+moron|moron\s+protein|phage\s+morons?|bacteriophage\s+moron|"
             r"\bmorons?\s+(encode|harbor|carry|insert|integrate))\b"),
            ("moronic_acid_chemistry",
             r"\b(moronic\s+acid|moronic\s+(ester|salt|analog|derivativ)|"
             r"3\-?oxo[\-\s]?moron|olean\-?(1[28])\-?en\-?3\-?on|triterpenoid\s+moron|"
             r"oleanolic.*moron|moronic.*triterpen)\b"),
            ("historical_iq_classification",
             r"\b(moron\s+(grade|class|level|category|range)|grade\s+of\s+moron|"
             r"iq\s+(of|range|score).*moron|feeble[\-\s]?minded.*moron|"
             r"kallikak|terman|stanford[\-\s]?binet.*moron|moron\s+intelligence|"
             r"institution\s+for\s+morons?|colony\s+for\s+morons?|"
             r"vineland|goddard|certif.*moron|moral\s+imbecile.*moron)\b"),
            ("slur_explicit_mention",
             r"\b(moron\s+(slur|epithet|insult|derogat|offensive|pejorative|stigmatiz)|"
             r"(\"|')moron(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as)\s+(offensive|pejorative|derogat|stigmatiz))|"
             r"reclaim.*moron|"
             r"disability\s+(activist|community|advocate).*moron)\b"),
        ],
    ),

    "T3_imbecile_clinical": SlurWSIConfig(
        label="T3_imbecile_clinical",
        esearch_terms=["imbecile", "imbeciles", "imbecility", "imbecilic"],
        sense_patterns=[
            ("historical_iq_classification",
             r"\b(imbecile\s+(grade|class|level|category|range|child(ren)?|patient|adult)|"
             r"grade\s+of\s+imbecile|iq\s+(of|range|score).*imbecile|"
             r"feeble[\-\s]?minded.*imbecile|moron.*imbecile|moral\s+imbecile|"
             r"institution\s+for\s+imbeciles?|colony\s+for\s+imbeciles?|"
             r"asylum.*imbecile|"
             r"imbecility\s+(due\s+to|caused\s+by|associated\s+with|in\s+(children|patients))|"
             r"certified\s+(as\s+)?(an\s+)?imbecile|congenital\s+imbecility)\b"),
            ("slur_explicit_mention",
             r"\b(imbecile\s+(slur|epithet|insult|derogat|offensive|pejorative|stigmatiz)|"
             r"(\"|')imbecile(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed)\s+(offensive|pejorative|derogat|stigmatiz))|"
             r"reclaim.*imbecile|stigma.*(\"|')?imbecile(\"|')?)\b"),
        ],
    ),

    "T3_whore_harlot": SlurWSIConfig(
        label="T3_whore_harlot",
        esearch_terms=["whore", "whores", "harlot", "harlots", '"common prostitute"'],
        sense_patterns=[
            ("historical_sti_venereology",
             r"\b((venereal|venereologic|gonorrhea|gonorrhoea|syphil|chancroid)[\s\S]{0,60}(whore|harlot)|"
             r"(whore|harlot)[\s\S]{0,60}(venereal|venereologic|gonorrhea|syphil|chancroid)|"
             r"common\s+prostitute|registered\s+prostitute|"
             r"brothel\s+(inspection|examination|registration|medical)|"
             r"contagious\s+diseases\s+acts?|cd\s+acts?)\b"),
            ("history_of_sex_work_archive",
             r"\b(use\s+of\s+(the\s+term\s+)?(whore|harlot|(\"|')common\s+prostitute(\"|'))|"
             r"whore\s+stigma|harlot\s+stigma|stigmatiz.*(whore|harlot)|"
             r"reclaim.*(whore|harlot)|"
             r"sex\s+work(er)?.*(whore|harlot)|street\s+walker)\b"),
            ("slur_explicit_mention",
             r"\b((whore|harlot)\s+(slur|epithet|insult|derogat|offensive|pejorative)|"
             r"(\"|')(whore|harlot)(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as)\s+(offensive|pejorative|derogat)))\b"),
        ],
    ),

    "T3_hottentot": SlurWSIConfig(
        label="T3_hottentot",
        esearch_terms=["Hottentot", "Hottentots",
                       '"Hottentot Venus"', '"Hottentot apron"'],
        sense_patterns=[
            ("khoisan_population_genetics",
             r"\b(khoi[\-\s]?(san|khoi)|khoisan|khoekhoe|nama\s+(people|language|population)|"
             r"south\s+african\s+(genome|population|ancestry|genetic)|"
             r"sub[\-\s]?saharan\s+(population\s+)?genetics|click\s+(consonant|language)|"
             r"hottentot.*(now|today)\s+(known|called|referred)|"
             r"(\"|')hottentot(\"|')\s+(is|has\s+been)\s+(now\s+)?(known|considered|recognized|used|replaced|reclassified))\b"),
            ("historical_racial_pathology",
             r"\b((\"|')hottentot\s+venus(\"|')|(\"|')hottentot\s+apron(\"|')|"
             r"(\"|')hottentot\s+bustle(\"|')|steatopygia|venus\s+of\s+south\s+africa|"
             r"saa?rtjie\s+baartman|sarah\s+baartman|"
             r"colonial\s+(pathology|medicine).*hottentot|"
             r"racial\s+(anthropology|typology|science).*hottentot|"
             r"physical\s+anthropology.*hottentot)\b"),
            ("slur_explicit_mention",
             r"\b(hottentot\s+(slur|epithet|insult|derogat|offensive|pejorative)|"
             r"(\"|')hottentot(\"|')\s+(was|is)\s+(used\s+as\s+)?(a\s+)?(slur|insult|pejorative|offensive\s+term)|"
             r"reclaim.*hottentot)\b"),
        ],
    ),

    "T3_kaffir": SlurWSIConfig(
        label="T3_kaffir",
        esearch_terms=["kaffir", "kaffirs"],
        sense_patterns=[
            ("botanical_kaffir_lime",
             r"\b(kaffir\s+(lime|leaf|leaves|tree|orange|plum|melon|corn|bean|pea|cherry|"
             r"rose|aloe|wattle|boom|fig)|"
             r"citrus\s+hystrix|makrut|"
             r"kaffir(?:\s+lime)?\s+(essential\s+oil|extract|peel|oil|aroma|flavor|flavour))\b"),
            ("south_african_history",
             r"\b(kaffir\s+(beer|wars?|tribe|land|chief|labourer|labor|worker|miner|patient|man|women)|"
             r"(\"|')kaffir(\"|')\s+(is|was)\s+(considered|used\s+as|now\s+considered)|"
             r"south\s+african\s+(apartheid|colonial|gold\s+mine)|cape\s+colony)\b"),
            ("slur_explicit_mention",
             r"\b(kaffir\s+(slur|epithet|insult|derogat|offensive|pejorative|hate\s+speech)|"
             r"(\"|')kaffir(\"|')\s+(is|was|has\s+been)\s+(used\s+as\s+)?(a\s+)?(slur|insult|pejorative|offensive\s+term|hate\s+speech)|"
             r"k[\-\s]?word|crimen\s+injuria|reclaim.*kaffir|"
             r"south\s+african.*hate\s+speech.*kaffir)\b"),
        ],
    ),

    "T3_monster_clinical": SlurWSIConfig(
        label="T3_monster_clinical",
        esearch_terms=['"congenital monster"', '"congenital monstrosity"',
                       '"medical monster"'],
        sense_patterns=[
            ("historical_congenital_monstrosity",
             r"\b((\"|')congenital\s+monster(\"|')|(\"|')congenital\s+monstrosity(\"|')|"
             r"monstrosity\s+(of|in|associated\s+with|due\s+to|caused\s+by)|"
             r"teratology[\s\S]{0,60}monstrosit|"
             r"monstrosities\s+(observed|described|reported|noted)|"
             r"case\s+of\s+monstrosity|"
             r"(\"|')medical\s+monster(\"|'))\b"),
            ("slur_explicit_mention",
             r"\b(monster\s+(slur|epithet|insult|derogat|offensive|pejorative|stigmatiz)|"
             r"(\"|')(monster|monstrosity)(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed)\s+(offensive|pejorative|derogat|stigmatiz)|harm)|"
             r"reclaim.*monster|"
             r"stigma.*(monster|monstrosity))\b"),
        ],
    ),

    # ===== iter-4 wave 2: additional slur-like labels with large clinical-history footprint =====

    "T2_hysteria": SlurWSIConfig(
        label="T2_hysteria",
        esearch_terms=["hysteria", "hysterical", "hysterics",
                       '"hysterical neurosis"'],
        sense_patterns=[
            ("pre_freudian_clinical",
             r"\b(hysterical\s+(paralysis|seizure|fugue|amnesia|fit|attack|"
             r"hemianesth|aphasia|aphonia|blindness|conversion)|"
             r"charcot|salp(e|\xea|\xe9|\xea)tri(e|\xe8|\xe9|\xea)re|"
             r"pierre\s+janet|grand\s+hysterie|petite\s+hysterie|arc\s+de\s+cercle|"
             r"hysteria\s+(major|minor)|hysteroepilepsy)\b"),
            ("conversion_disorder_modern",
             r"\b(conversion\s+disorder|functional\s+neurological\s+(symptom|disorder)|"
             r"\bfnd\b|somatoform|somatic\s+symptom\s+disorder|"
             r"psychogenic\s+(non[\-\s]?epileptic|paralysis|seizure|movement)|"
             r"\bpnes\b|dissociative\s+disorder)\b"),
            ("psychoanalytic_freud",
             r"\b(freud|breuer|psychoanaly[st].*hysteri|"
             r"(\"|')studies\s+on\s+hysteria(\"|')|anna\s+o\.|"
             r"hysterical\s+neurosis|psychoanalytic\s+(theory|interpret).*hysteri)\b"),
            ("mass_collective",
             r"\b(mass\s+hysteria|collective\s+hysteria|epidemic\s+hysteria|"
             r"mass\s+(psychogenic|sociogenic)\s+illness|outbreak\s+of\s+hysteria)\b"),
            ("informal_emotional",
             r"\b(hysterical\s+(behavior|behaviour|patient|reaction|outburst|crying|laughter)|"
             r"became\s+hysterical|hysterical\s+about)\b"),
            ("slur_explicit_mention",
             r"\b(hysteri\w*\s+(slur|epithet|insult|derogat|offensive|pejorative|sexist|misogyn|stigmatiz)|"
             r"(\"|')hysterical?(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as)\s+(offensive|pejorative|derogat|sexist|misogyn|stigmatiz))|"
             r"(sexist|misogyn).*hysteri|reclaim.*hysteri)\b"),
        ],
    ),

    "T2_hermaphrodite": SlurWSIConfig(
        label="T2_hermaphrodite",
        esearch_terms=["hermaphrodite", "hermaphrodites", "hermaphroditism",
                       "hermaphroditic",
                       '"pseudohermaphrodite"', '"pseudohermaphrodites"',
                       '"pseudohermaphroditism"',
                       '"true hermaphrodite"'],
        sense_patterns=[
            ("dsd_intersex_modern",
             r"\b(disorder\s+of\s+sex\s+development|\bdsd\b|"
             r"intersex|differences?\s+of\s+sex\s+development|"
             r"chicago\s+consensus|"
             r"(\"|')dsd(\"|')\s+(terminology|nomenclature))\b"),
            ("endocrine_genetic_clinical",
             r"\b(androgen\s+insensitivity|complete\s+androgen|partial\s+androgen|\bcais\b|\bpais\b|"
             r"5(\-?alpha|\-?\xce\xb1)?[\-\s]?reductase|21\-hydroxylase|"
             r"congenital\s+adrenal\s+hyperplasia|\bcah\b|"
             r"ovotestis|ovotesticular|"
             r"klinefelter|turner\s+syndrome|swyer|gonadal\s+dysgenesis|"
             r"(45|46|47),?\s*\bxx?[xy]?\b)\b"),
            ("historical_clinical_taxonomy",
             r"\b(true\s+hermaphrodite|true\s+hermaphroditism|"
             r"(male|female)\s+pseudohermaphrod|"
             r"pseudohermaphroditism\s+(of|in|due\s+to)|"
             r"hermaphroditism\s+(verus|spurious))\b"),
            ("biology_animal_plant",
             r"\b(hermaphrodit\w+\s+(snail|worm|fish|earthworm|nematode|trematode|"
             r"flatworm|plant|species|population|individual|protandr|protogyn)|"
             r"sequential\s+hermaphrod|simultaneous\s+hermaphrod|"
             r"self[\-\s]?fertilization|monoecious|\bc\.\s*elegans\b|"
             r"sex\s+determination\s+in\s+(c\.\s*elegans|caenorhabditis))\b"),
            ("slur_explicit_mention",
             r"\b(hermaphrodit\w+\s+(slur|epithet|insult|derogat|offensive|pejorative|stigmatiz)|"
             r"(\"|')hermaphrodite(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as)\s+(offensive|pejorative|stigmatiz|outdated))|"
             r"(intersex|dsd)\s+(rather\s+than|instead\s+of|preferred\s+over|replac).*hermaphrod|"
             r"reclaim.*hermaphrod)\b"),
        ],
    ),

    "T2_transsexual_xvest": SlurWSIConfig(
        label="T2_transsexual_xvest",
        esearch_terms=["transsexual", "transsexuals", "transsexualism", "transsexuality",
                       "transvestite", "transvestites", "transvestism", "transvestic",
                       '"gender identity disorder"'],
        sense_patterns=[
            ("gender_affirming_clinical_modern",
             r"\b(gender[\-\s]?affirming|transgender\s+(health|medicine|care|patient|population)|"
             r"sex\s+reassign|gender\s+reassign|"
             r"vaginoplasty|phalloplasty|metoidioplasty|orchiectomy|"
             r"(testosterone|estrogen|estradiol)\s+(therapy|treatment).*trans|"
             r"hormone\s+replac.*trans|\bgaht\b|"
             r"top\s+surgery|bottom\s+surgery)\b"),
            ("gender_dysphoria_diagnosis",
             r"\b(gender\s+dysphoria|gender\s+identity\s+disorder|\bgid\b|"
             r"transsexualism\s+(diagnosis|criteria)|"
             r"\bdsm[\-\s]?(iv|5|v|iii)\b.*(transsex|gender|transvest)|"
             r"\bicd[\-\s]?(10|11|9)\b.*(transsex|gender|transvest)|\bf64\b)\b"),
            ("psychiatric_paraphilia",
             r"\b(transvestic\s+(disorder|fetishism)|transvestic\s+fetish|"
             r"paraphilia.*transvest|fetishistic\s+transvestism|"
             r"autogynephilia|transvestic\s+behavior)\b"),
            ("epidemiology_advocacy",
             r"\b(transgender\s+(population|community|adults?|adolescents?|youth)|"
             r"trans(?:gender)?\s+(rights|advocacy|health\s+disparit|stigma)|"
             r"\blgbtq\+?\b|gender\s+minority\s+(stress|population)|"
             r"transgender\s+(women|men|persons|individuals))\b"),
            ("slur_explicit_mention",
             r"\b((transsexual|transvestite)\s+(slur|epithet|insult|derogat|offensive|pejorative|outdated|stigmatiz)|"
             r"(\"|')(transsexual|transvestite)(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as|now\s+considered)\s+(offensive|pejorative|outdated|stigmatiz))|"
             r"(transgender|trans)\s+(rather\s+than|instead\s+of|preferred\s+over|replac).*(transsexual|transvestite)|"
             r"reclaim.*(transsexual|transvestite))\b"),
        ],
    ),

    "T3_cripple": SlurWSIConfig(
        label="T3_cripple",
        esearch_terms=["cripple", "cripples", "crippled", "crippling",
                       '"crippled child"', '"crippled children"'],
        sense_patterns=[
            ("crippled_childrens_services_program",
             r"\b((\"|')crippled\s+children('s)?\s+(services?|program|act|fund|hospital|clinic|society|aid|commission)(\"|')?|"
             r"title\s+v\s+(crippled|maternal\s+and\s+child)|march\s+of\s+dimes|"
             r"infantile\s+paralysis|poliomyeli.*(crippled|cripple)|"
             r"national\s+foundation.*infantile)\b"),
            ("polio_orthopedic_historical",
             r"\b(polio\s+(rehabilitation|patient|cripple|epidemic)|poliomyelitis\s+(cripple|patient|sequela)|"
             r"post[\-\s]?polio|iron\s+lung|salk\s+vaccine|sabin\s+vaccine|"
             r"crippled\s+by\s+polio|polio\s+(survivor|victim))\b"),
            ("orthopedic_skeletal_clinical",
             r"\b(crippled\s+by\s+(arthritis|cerebral\s+palsy|rheumatoid|fracture|tuberculosis|tb)|"
             r"crippling\s+(arthritis|disease|condition|disability|injury)|"
             r"rheumatoid\s+(cripple|crippling)|orthopedic\s+cripple|"
             r"spinal\s+(cripple|crippling)|cerebral\s+palsy.*cripple)\b"),
            ("metaphor_figurative",
             r"\b(crippling\s+(effect|impact|cost|burden|fear|anxiety|debt|stigma|inflation|loss|defeat)|"
             r"crippled\s+(by\s+(fear|anxiety|debt|cost|inflation)|economy|industry|company|infrastructure|system|network))\b"),
            ("slur_explicit_mention",
             r"\b(cripple\w*\s+(slur|epithet|insult|derogat|offensive|pejorative|outdated|stigmatiz|ableis)|"
             r"(\"|')cripple(d)?(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as|now\s+considered)\s+(offensive|pejorative|outdated|stigmatiz|ableis))|"
             r"(disabled|person\s+with\s+disab).*\b(rather\s+than|instead\s+of|preferred\s+over).*cripple|"
             r"reclaim.*cripple|crip\s+(theory|studies|culture))\b"),
        ],
    ),

    "T3_lazar_leper": SlurWSIConfig(
        label="T3_lazar_leper",
        esearch_terms=["leper", "lepers", "leprosy", "lazar", "lazars",
                       '"leper colony"',
                       "lazaretto", "leprous"],
        sense_patterns=[
            ("hansens_disease_modern_clinical",
             r"\b(hansen'?s?\s+disease|mycobacterium\s+leprae|m\.\s*leprae|"
             r"leprosy\s+(treatment|control|elimination|chemotherapy)|"
             r"dapsone|rifampin|rifampicin|clofazimine|"
             r"multibacillary|paucibacillary|tuberculoid|lepromatous|"
             r"borderline\s+leprosy|leprosy\s+reaction|type\s+1\s+leprosy|type\s+2\s+leprosy|"
             r"erythema\s+nodosum\s+leprosum|\benl\b)\b"),
            ("historical_leprosy_archive",
             r"\b(leper\s+colony|leper\s+colonies|leprosarium|leprosaria|lazaretto|lazar\s+house|"
             r"medieval\s+lepros|history\s+of\s+lepros|"
             r"molokai|kalaupapa|robben\s+island|father\s+damien|"
             r"san\s+lazaro|carville|culion|sungei\s+buloh|"
             r"leprosy\s+stigma|leprosy\s+segregation|leprosy\s+exile)\b"),
            ("terminology_advocacy_modern",
             r"\b((\"|')person\s+(affected\s+by|with)\s+lepros(\"|')|"
             r"\bidea\s+lepros|(\"|')leprosy(\"|')\s+(rather\s+than|preferred\s+over)\s+(\"|')leper(\"|')|"
             r"(\"|')leper(\"|')\s+(is|has\s+been)\s+(considered|deemed|seen\s+as)\s+(offensive|stigmatiz|derogat|outdated)|"
             r"who\s+(leprosy\s+)?terminolog|leprosy\s+language)\b"),
            ("biology_animal_plant",
             r"\b(armadillo.*lepra?osy|leprosy\s+in\s+(armadillo|mouse|squirrel|nine[\-\s]banded)|"
             r"animal\s+model.*lepros)\b"),
            ("slur_explicit_mention",
             r"\b((leper|lazar)\s+(slur|epithet|insult|derogat|offensive|pejorative|outdated|stigmatiz)|"
             r"(\"|')(leper|lazar)(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as|now\s+considered)\s+(offensive|pejorative|outdated|stigmatiz))|"
             r"reclaim.*leper|like\s+a\s+leper)\b"),
        ],
    ),

    "T3_maniac_madhouse": SlurWSIConfig(
        label="T3_maniac_madhouse",
        esearch_terms=["maniac", "maniacs", "maniacal",
                       "madman", "madmen", "madwoman", "madwomen",
                       "madhouse", "madhouses"],
        sense_patterns=[
            ("historical_asylum_madhouse",
             r"\b(madhouse|madhouses|bedlam|bethlem|"
             r"asylum\s+(physician|history|reform|inmate)|history\s+of\s+the\s+asylum|"
             r"madhouses\s+act|county\s+(asylum|madhouse)|private\s+madhouse)\b"),
            ("manic_depressive_clinical",
             r"\b(manic[\-\s]?(depressive|depression)|mania\s+(diagnosis|criteria)|"
             r"kraepelin|hypomania|circular\s+insanity|mixed\s+state|"
             r"bipolar\s+(disorder|i|ii)|lithium.*mania)\b"),
            ("informal_extreme_behavior",
             r"\b(maniac\s+(behavior|behaviour|patient|driver|killer)|"
             r"maniacal\s+(laugh|behaviour|behavior)|"
             r"drove\s+(him|her|them|me)\s+(mad|crazy|maniacal)|"
             r"absolute\s+maniac|complete\s+maniac)\b"),
            ("slur_explicit_mention",
             r"\b((maniac|madman|madhouse)\s+(slur|epithet|insult|derogat|offensive|pejorative|outdated|stigmatiz)|"
             r"(\"|')(maniac|madman)(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as|now\s+considered)\s+(offensive|pejorative|outdated|stigmatiz))|"
             r"reclaim.*(mad|maniac)|mad\s+(pride|studies|movement))\b"),
        ],
    ),

    "T2_drunkard_inebriate": SlurWSIConfig(
        label="T2_drunkard_inebriate",
        esearch_terms=["drunkard", "drunkards", "inebriate", "inebriates",
                       "dipsomania", "dipsomaniac"],
        sense_patterns=[
            ("pre_aa_temperance_movement",
             r"\b(temperance\s+(movement|society|society|league|union)|"
             r"\bw\s*c\s*t\s*u\b|salvation\s+army.*drunkard|"
             r"inebriates\s+act|habitual\s+drunkards?\s+act|"
             r"keeley\s+(cure|institute)|washingtonian\s+(movement|society)|"
             r"gold\s+cure)\b"),
            ("pre_dsm_clinical",
             r"\b(inebriate\s+(hospital|asylum|home|reformatory)|"
             r"habitual\s+drunkard|chronic\s+alcoholic|"
             r"delirium\s+tremens|\bdt'?s\b|korsakoff|wernicke|"
             r"dipsomania)\b"),
            ("modern_alcohol_use_disorder",
             r"\b(alcohol\s+use\s+disorder|\baud\b|alcoholism|"
             r"\baa\b|alcoholics\s+anonymous|alcohol\s+dependence|"
             r"alcohol[\-\s]?related\s+(disorder|harm)|hazardous\s+drinking)\b"),
            ("slur_explicit_mention",
             r"\b((drunkard|inebriate)\s+(slur|epithet|insult|derogat|offensive|pejorative|outdated|stigmatiz)|"
             r"(\"|')(drunkard|inebriate)(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as|now\s+considered)\s+(offensive|pejorative|outdated|stigmatiz))|"
             r"(person\s+with\s+(alcohol|aud)).*(rather\s+than|instead\s+of).*(drunkard|inebriate))\b"),
        ],
    ),

    "T2_neurasthenia": SlurWSIConfig(
        label="T2_neurasthenia",
        esearch_terms=["neurasthenia", "neurasthenic"],
        sense_patterns=[
            ("beard_19c_clinical",
             r"\b(george\s+beard|american\s+nervousness|"
             r"weir\s+mitchell|rest\s+cure|"
             r"19th\s+(century|c)|nineteenth[\-\s]?century|victorian|"
             r"nervous\s+exhaustion|exhausted\s+nerves)\b"),
            ("chronic_fatigue_modern",
             r"\b(chronic\s+fatigue\s+syndrome|\bcfs\b|me/cfs|"
             r"myalgic\s+encephalomyel|fibromyalgia|"
             r"burnout|adrenal\s+fatigue|long\s+covid)\b"),
            ("asian_modern_clinical",
             r"\b(neurasthenia\s+in\s+(china|japan|korea|taiwan)|"
             r"chinese\s+neurasthenia|shenjingshuairuo|"
             r"asian\s+(culture|culture[\-\s]?bound)\s+(syndrome|neurasthen))\b"),
            ("slur_explicit_mention",
             r"\b(neurasthenia\s+(slur|epithet|insult|derogat|offensive|pejorative|outdated|stigmatiz)|"
             r"(\"|')neurasthenia(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as|now\s+considered)\s+(offensive|pejorative|outdated))|"
             r"(chronic\s+fatigue|cfs).*rather\s+than\s+neurasthen)\b"),
        ],
    ),

    "T2_psychopath_socio": SlurWSIConfig(
        label="T2_psychopath_socio",
        esearch_terms=["psychopath", "sociopath",
                       '"psychopathic personality"'],
        sense_patterns=[
            ("antisocial_personality_disorder",
             r"\b(antisocial\s+personality\s+disorder|\baspd\b|"
             r"\bdsm[\-\s]?(iv|5|v|iii)\b.*(antisocial|psychopath|sociopath)|"
             r"\bicd[\-\s]?(10|11)\b.*(antisocial|psychopath)|"
             r"conduct\s+disorder.*adult)\b"),
            ("forensic_psychopathy_research",
             r"\b(hare\s+(psychopathy|checklist)|\bpcl[\-\s]?(r|sv)\b|"
             r"cleckley|mask\s+of\s+sanity|psychopathy\s+(checklist|construct|trait)|"
             r"factor\s+(1|2|i|ii).*psychopathy|forensic\s+psychopath|"
             r"interpersonal\s+affective|callous[\-\s]?unemotional|"
             r"\bcu\s+trait)\b"),
            ("pop_psychology_corporate",
             r"\b(corporate\s+psychopath|workplace\s+psychopath|"
             r"snakes\s+in\s+suits|babiak|everyday\s+psychopath|"
             r"successful\s+psychopath|psychopath\s+next\s+door)\b"),
            ("slur_explicit_mention",
             r"\b((psychopath|sociopath)\s+(slur|epithet|insult|derogat|offensive|pejorative|stigmatiz)|"
             r"(\"|')(psychopath|sociopath)(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as)\s+(offensive|pejorative|stigmatiz))|"
             r"reclaim.*psychopath)\b"),
        ],
    ),

    "T3_deaf_mute": SlurWSIConfig(
        label="T3_deaf_mute",
        esearch_terms=['"deaf-mute"', '"deaf mute"', '"deaf-mutes"',
                       '"deaf and dumb"'],
        sense_patterns=[
            ("historical_deafness_clinical",
             r"\b(deaf[\-\s]?(and[\-\s]?)?dumb|deaf[\-\s]?mute|"
             r"institution\s+for\s+the\s+deaf|school\s+for\s+(the\s+)?deaf|"
             r"oralism|manualism|sign\s+language\s+history|gallaudet)\b"),
            ("terminology_advocacy",
             r"\b(deaf\s+community|deafness\s+terminology|deaf\s+identity|"
             r"deaf\s+culture|capital\s+d\s+deaf|big\s+d\s+deaf|"
             r"hard\s+of\s+hearing|hearing\s+loss\s+terminology)\b"),
            ("slur_explicit_mention",
             r"\b(deaf[\-\s]?(mute|and[\-\s]?dumb)\s+(slur|epithet|insult|derogat|offensive|pejorative|outdated|stigmatiz)|"
             r"(\"|')deaf[\-\s]?(mute|and[\-\s]?dumb)(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as|now\s+considered)\s+(offensive|pejorative|outdated|stigmatiz))|"
             r"(deaf|hard\s+of\s+hearing)\s+(rather\s+than|instead\s+of|preferred\s+over).*(deaf[\-\s]?mute|deaf[\-\s]?and[\-\s]?dumb))\b"),
        ],
    ),

    "T3_siamese_twins": SlurWSIConfig(
        label="T3_siamese_twins",
        esearch_terms=['"Siamese twins"', '"Siamese twin"'],
        sense_patterns=[
            ("clinical_conjoined_twins",
             r"\b(conjoined\s+twins?|conjoined\s+twin\s+(surgery|separation|case)|"
             r"thoracopagus|omphalopagus|pygopagus|craniopagus|ischiopagus|"
             r"parapagus|cephalopagus|rachipagus|"
             r"twin\s+separation\s+surgery|fetus\s+in\s+fetu)\b"),
            ("historical_chang_eng",
             r"\b(chang\s+and\s+eng|chang\s+eng|bunker\s+(brothers|twins)|"
             r"siam.*conjoined|18(19|20|30|40|50|60|70)s?.*siamese)\b"),
            ("terminology_advocacy",
             r"\b((\"|')conjoined\s+twins?(\"|')\s+(rather\s+than|preferred\s+over|replac)|"
             r"(\"|')siamese\s+twins?(\"|')\s+(is|has\s+been)\s+(considered|deemed|seen\s+as|now\s+considered)\s+(offensive|outdated|inappropriate))\b"),
            ("slur_explicit_mention",
             r"\b(siamese\s+twins?\s+(slur|epithet|insult|derogat|offensive|outdated|stigmatiz)|"
             r"reclaim.*siamese\s+twin)\b"),
        ],
    ),

    "T3_bushman": SlurWSIConfig(
        label="T3_bushman",
        esearch_terms=["Bushman", "Bushmen"],
        sense_patterns=[
            ("khoisan_san_modern_anthropology",
             r"\b(\bsan\b\s+(people|population|community|hunter[\-\s]?gatherer|society)|"
             r"khoisan|khoekhoe|nama\s+(people|community)|"
             r"hai//om|ju/'?hoansi|!kung|kalahari\s+(san|hunter[\-\s]?gatherer)|"
             r"southern\s+african\s+(forager|hunter[\-\s]?gather))\b"),
            ("population_genetics_studies",
             r"\b(bushman.*(genome|genetic|mtdna|y[\-\s]?chromosome|allele|haplogroup|haplotype)|"
             r"(\"|')bushmen?(\"|')\s+(now|today)\s+(known|called|referred)|"
             r"(\"|')bushmen?(\"|')\s+(is|has\s+been)\s+(replaced|reclassified))\b"),
            ("colonial_historical_anthropology",
             r"\b(colonial.*bushman|bushman.*colonial|"
             r"(\"|')bushman\s+race(\"|')|racial\s+anthropology.*bushman|"
             r"physical\s+anthropology.*bushman|hottentot.*bushman|"
             r"bushman.*hottentot)\b"),
            ("slur_explicit_mention",
             r"\b(bushman\s+(slur|epithet|insult|derogat|offensive|pejorative|outdated|stigmatiz)|"
             r"(\"|')bushm(a|e)n(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as|now\s+considered)\s+(offensive|pejorative|outdated|stigmatiz))|"
             r"reclaim.*bushm(a|e)n)\b"),
        ],
    ),

    "T3_hunchback": SlurWSIConfig(
        label="T3_hunchback",
        esearch_terms=["hunchback", "hunchbacked", "hunchbacks"],
        sense_patterns=[
            ("clinical_kyphosis",
             r"\b(kyphosis|kyphotic|gibbus|pott'?s?\s+disease|"
             r"vertebral\s+(collaps|wedg|compress)|severe\s+kyphos|"
             r"scoliokyphos|hyperkyphos|thoracic\s+(kyphos|deformity)|"
             r"sengupta|spinal\s+(deformity|curvature))\b"),
            ("drosophila_hunchback_gene",
             r"\b(hunchback\s+(gene|mrna|protein|locus|expression|transcription|"
             r"regulation|target|domain|repressor|gradient)|"
             r"\bhb\b\s+(gene|mrna|protein)|"
             r"drosophila.*hunchback|hunchback.*drosophila|"
             r"bicoid.*hunchback|hunchback.*bicoid|"
             r"gap\s+gene.*hunchback|segmentation.*hunchback)\b"),
            ("vertebrate_zebrafish_hunchback",
             r"\b(hunchback[\-\s]?like|hbl\d|"
             r"zebrafish.*hunchback|c\.\s*elegans.*hunchback|"
             r"hunchback.*zebrafish)\b"),
            ("historical_literary_quasimodo",
             r"\b(notre[\-\s]?dame|quasimodo|hunchback\s+of\s+notre|"
             r"victor\s+hugo|richard\s+iii|shakespeare.*hunchback)\b"),
            ("slur_explicit_mention",
             r"\b(hunchback\s+(slur|epithet|insult|derogat|offensive|pejorative|outdated|stigmatiz|ableis)|"
             r"(\"|')hunchback(\"|')\s+(as\s+a\s+slur|is\s+(considered|deemed|seen\s+as|now\s+considered)\s+(offensive|pejorative|outdated|stigmatiz|ableis))|"
             r"(person\s+with\s+kyphosis)\s+(rather\s+than|instead\s+of).*hunchback|"
             r"reclaim.*hunchback)\b"),
        ],
    ),

    "T3_oriental_disease": SlurWSIConfig(
        label="T3_oriental_disease",
        esearch_terms=['"Oriental sore"', '"Oriental cholera"',
                       '"Oriental schistosomiasis"', '"Oriental boil"'],
        sense_patterns=[
            ("historical_clinical_compound",
             r"\b(oriental\s+(sore|cholera|schistosomiasis|boil|plague|spotted\s+fever|"
             r"trypanosomiasis|liver\s+fluke))\b"),
            ("modern_replacement_terminology",
             r"\b(cutaneous\s+leishmaniasis|leishmania\s+tropica|leishmania\s+major|"
             r"l\.\s*tropica|l\.\s*major|"
             r"(vibrio\s+cholerae|cholerae\s+serogroup).*((?<!oriental\s)cholera)|"
             r"schistosoma\s+japonicum|s\.\s*japonicum|"
             r"clonorchis|opisthorchis)\b"),
            ("colonial_tropical_medicine_history",
             r"\b((\"|')oriental(\"|')\s+(is|was|has\s+been)\s+(considered|now\s+considered|seen\s+as|now\s+seen\s+as)\s+(offensive|pejorative|outdated|stigmatiz|inappropriate)|"
             r"colonial\s+tropical\s+medicine|imperial\s+tropical\s+medicine|"
             r"history\s+of\s+tropical\s+medicine|orientalism.*medicine)\b"),
            ("slur_explicit_mention",
             r"\b(oriental\s+(slur|epithet|insult|derogat|offensive|pejorative|outdated|stigmatiz)|"
             r"(asian|east\s+asian|chinese|south\s+asian)\s+(rather\s+than|instead\s+of|preferred\s+over).*oriental)\b"),
        ],
    ),
}


# ----------------------- Generic fetch + classify -----------------------

def _fetch_one_label(
    cfg: SlurWSIConfig,
    *,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Year-by-year esearch + efetch for all PMIDs of one label."""
    pmids = esearch_pmids(
        cfg.esearch_terms,
        start_year=cfg.start_year,
        end_year=cfg.end_year,
        api_key=api_key,
    )
    print(f"  {cfg.label}: {len(pmids):,} PMIDs from esearch",
          file=sys.stderr)
    if not len(pmids):
        return pd.DataFrame()
    records = efetch_records(pmids, api_key=api_key)
    df = pd.DataFrame(records)
    if not len(df):
        return df
    df["text"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")
                  ).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["year"] = df["year"].astype("Int64")
    df = df.dropna(subset=["year"]).reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["label"] = cfg.label
    return df


def _classify_one_label(
    cfg: SlurWSIConfig,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Classify a label's record corpus by first-match-wins sense regex."""
    if not cfg.compiled:
        cfg.compile()

    def _classify(text: str) -> str:
        for s, pat in cfg.compiled:
            if pat.search(text):
                return s
        return "unknown"

    df = df.copy()
    df["sense"] = df["text"].map(_classify)
    return df


def run_one_label(
    cfg: SlurWSIConfig,
    *,
    out_dir: Path,
    api_key: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch, classify, save per-(year, sense) CSV for one label.

    Returns the classified per-record DataFrame.
    """
    label_dir = out_dir / cfg.label
    label_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = label_dir / "abstracts.parquet"
    per_yr_path = label_dir / "sense_counts_by_year.csv"

    if parquet_path.exists() and not force:
        print(f"[cache] reusing {parquet_path}", file=sys.stderr)
        df = pd.read_parquet(parquet_path)
    else:
        df = _fetch_one_label(cfg, api_key=api_key)
        if len(df):
            df.to_parquet(parquet_path, index=False)
            print(f"  {cfg.label}: wrote {len(df):,} records to "
                  f"{parquet_path}", file=sys.stderr)

    if not len(df):
        return df

    df = _classify_one_label(cfg, df)
    per_yr = (df.groupby(["year", "sense"]).size()
                .unstack("sense", fill_value=0).astype(int))
    per_yr.to_csv(per_yr_path)
    print(f"  {cfg.label}: per-(year, sense) CSV at {per_yr_path}",
          file=sys.stderr)

    # Summary print to stderr — useful when running interactively.
    totals = per_yr.sum(axis=0).sort_values(ascending=False)
    print(f"  {cfg.label}: per-sense totals:", file=sys.stderr)
    for s, n in totals.items():
        pct = 100.0 * n / max(1, int(per_yr.sum().sum()))
        print(f"    {s:<40} {n:>6,}  {pct:>5.1f}%", file=sys.stderr)
    return df


# ----------------------- Combined output -----------------------

def write_combined_csv(out_dir: Path, combined_path: Path) -> None:
    """Read each per-label sense_counts_by_year.csv and stack to one CSV
    with rows (label, year, sense, n_records)."""
    rows: list[pd.DataFrame] = []
    for cfg in SLUR_WSI_CONFIGS.values():
        per_yr_path = out_dir / cfg.label / "sense_counts_by_year.csv"
        if not per_yr_path.exists():
            continue
        per_yr = pd.read_csv(per_yr_path, index_col="year")
        long_df = (per_yr.reset_index()
                          .melt(id_vars="year", var_name="sense",
                                value_name="n_records"))
        long_df["label"] = cfg.label
        rows.append(long_df[["label", "year", "sense", "n_records"]])
    if not rows:
        print("no per-label CSVs to combine", file=sys.stderr)
        return
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    print(f"wrote combined CSV ({len(combined):,} rows, "
          f"{combined['label'].nunique()} labels) to {combined_path}",
          file=sys.stderr)


# ----------------------- CLI -----------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-key", default=None)
    p.add_argument("--out",
                   type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "slur_wsi")
    p.add_argument("--combined",
                   type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "slur_wsi_combined.csv")
    p.add_argument("--only", nargs="*", default=None,
                   help="Restrict to subset of labels.")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch even if parquet cache exists.")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for label, cfg in SLUR_WSI_CONFIGS.items():
        if args.only and label not in args.only:
            continue
        print(f"\n=== {label} ===", file=sys.stderr)
        try:
            run_one_label(cfg, out_dir=args.out,
                          api_key=args.api_key, force=args.force)
        except Exception as e:
            print(f"  {label}: FAILED {type(e).__name__}: {e}",
                  file=sys.stderr)

    write_combined_csv(args.out, args.combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
