"""Deterministically generate the synthetic Hansard-sample parquet.

This script is the source-of-truth for ``hansard_sample.parquet``. It is
*not* run at import time — the parquet is committed and shipped with the
package, and :func:`load_hansard_sample` just reads it. The script is
here so reviewers can verify the sample is reproducible and so we can
regenerate it when the templates change.

Run with::

    python -m pycorpdiff.datasets._generate_hansard

The output is written to ``src/pycorpdiff/datasets/_data/hansard_sample.parquet``.
The generator is seeded so output is byte-identical across machines and
Python versions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OPENINGS = [
    "I rise to address the House on",
    "I beg leave to bring to the attention of this House the question of",
    "Mr Speaker, I wish to make a statement concerning",
    "I am grateful for the opportunity to debate",
    "The Honourable Members of this House should consider",
    "Madam Deputy Speaker, I wish to address",
    "I am pleased to speak on the matter of",
]

CLOSINGS = [
    "I commend this motion to the House.",
    "I urge my Honourable colleagues to support this.",
    "The Government must take immediate action.",
    "We owe this much to our constituents.",
    "This matter brooks no further delay.",
    "The time for inaction has long passed.",
    "I beg to move.",
]

# (topic, period_label) -> list of body templates.
# The period_label structure encodes the temporal-frame shifts:
#   - immigration shifts from humanising (pre-2016) to criminalising (post-2016)
#   - brexit goes through emerging → peak → aftermath
#   - nhs has a steady frame + crisis spikes in 2010 (austerity) and 2020 (covid)
#   - climate sharpens scientific → policy → crisis post-2019
TOPIC_BODIES: dict[tuple[str, str], list[str]] = {
    ("immigration", "humanising"): [
        "the immigrant worker arrived with hope and the immigrant family settled with dignity",
        "the immigrant community contributes to our shared prosperity and shared future",
        "the immigrant family deserves protection refuge and a clear path to citizenship",
        "the immigrant worker rights advance through union solidarity and our labour movement",
        "the immigrant community organised with strength and the immigrant worker spoke with pride",
        "the immigrant family brings cultural richness and economic vitality to our towns",
        "the immigrant worker contributes to public services education and our national life",
        "the immigrant community has thrived and the immigrant family has flourished here",
    ],
    ("immigration", "criminalising"): [
        "the immigrant criminal threat grows and the immigrant invasion of gangs spreads",
        "the immigrant criminal element alarms residents and our border control has failed",
        "the immigrant invasion narrative dominates news and immigrant criminal gangs persist",
        "the immigrant threat has increased and the immigrant crime narrative grew with concern",
        "the immigrant gangs threaten the border and the immigrant criminal risk grows daily",
        "the immigrant criminal threat must be confronted and the immigrant invasion halted",
        "the immigrant criminal gangs operate freely and immigrant invasion routes remain open",
    ],
    ("brexit", "emerging"): [
        "the european question must be addressed and the referendum we promised must be delivered",
        "our relationship with europe requires renegotiation and reform from this government",
        "the european union framework no longer serves british interests and british sovereignty",
        "the european treaties demand renegotiation before the public can give their consent",
        "the european question divides this house but the people must have their say",
    ],
    ("brexit", "peak"): [
        "the brexit referendum result must be respected and delivered without further delay",
        "the brexit deal must respect the democratic will of seventeen million leave voters",
        "the brexit transition must protect british businesses and british workers from harm",
        "the brexit negotiations require firm leadership and a clear vision for our nation",
        "the brexit outcome will define this generation and the brexit deal must be honoured",
        "the brexit settlement requires patience but the brexit mandate is unambiguous",
    ],
    ("brexit", "aftermath"): [
        "the brexit deal has delivered for british sovereignty and british democratic accountability",
        "the brexit aftermath reveals supply chain disruption and significant economic adjustment",
        "the brexit transition continues with new opportunities for global trade and partnership",
        "the brexit dividend has yet to materialise for working families across our nation",
        "the brexit settlement requires further work on northern ireland and our customs arrangements",
    ],
    ("nhs", "normal"): [
        "the national health service requires sustained investment for our nurses and doctors",
        "the nhs workforce must be supported with proper funding and training programmes",
        "the patient care standards must be maintained across all hospitals and trusts",
        "the nhs provides universal care and the nhs principles remain our foundation",
        "the national health service belongs to all of us and to our future generations",
    ],
    ("nhs", "austerity"): [
        "the nhs austerity cuts threaten patient care and waiting times grow alarming",
        "the nhs underfunding creates crises in accident and emergency departments nationwide",
        "the nhs austerity decisions cost lives and the nhs underfunding harms our communities",
        "the nhs cuts must be reversed and the nhs funding settlement must be honoured",
    ],
    ("nhs", "covid"): [
        "the nhs response to the pandemic deserves our gratitude and continued support",
        "the nhs covid crisis demands emergency funding and ventilator capacity now",
        "the nhs frontline workers face unprecedented pressure and the nhs covid response saves lives",
        "the nhs covid surge requires us to clap and to legislate for fair pay",
    ],
    ("climate", "scientific"): [
        "the scientific consensus on climate change requires policy response from this government",
        "the climate models indicate warming trends that demand emissions reduction targets",
        "the climate science is settled and the climate research is unambiguous in its conclusions",
        "the climate evidence accumulates and the climate scientists call for action",
    ],
    ("climate", "policy"): [
        "the climate policy framework must align with our paris agreement obligations",
        "the climate change committee recommends carbon budget reductions for the coming decade",
        "the climate policy must include just transition for fossil fuel workers and communities",
        "the climate framework needs revision and the climate policy targets need strengthening",
    ],
    ("climate", "crisis"): [
        "the climate crisis is here now and the climate emergency demands urgent action",
        "the climate breakdown threatens our coastlines and the climate emergency cannot wait",
        "the climate crisis requires immediate emissions cuts and the climate emergency is upon us",
        "the climate disaster unfolds and the climate emergency response must accelerate",
        "the climate emergency declaration must be backed by the climate action this house owes",
    ],
}

# Period predicates for each topic. Each yields a (year) -> period_label
# mapping that decides which template bucket a speech in that year uses.
def _immigration_period(year: int) -> str:
    return "humanising" if year < 2016 else "criminalising"


def _brexit_period(year: int) -> str:
    if year < 2016:
        return "emerging"
    if year < 2020:
        return "peak"
    return "aftermath"


def _nhs_period(year: int) -> str:
    if 2010 <= year <= 2014:
        return "austerity"
    if 2020 <= year <= 2022:
        return "covid"
    return "normal"


def _climate_period(year: int) -> str:
    if year < 2011:
        return "scientific"
    if year < 2019:
        return "policy"
    return "crisis"


PERIOD_FOR = {
    "immigration": _immigration_period,
    "brexit": _brexit_period,
    "nhs": _nhs_period,
    "climate": _climate_period,
}

PARTIES = ["Labour", "Conservative", "Liberal Democrat", "SNP"]
TOPICS = ["immigration", "brexit", "nhs", "climate"]


def generate(seed: int = 20260522) -> pd.DataFrame:
    """Return a deterministic 200-speech synthetic Hansard sample."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    speech_id = 0
    for year in range(2005, 2024):
        # Roughly 10-11 speeches per year. Brexit and immigration get
        # more airtime in years close to the referendum.
        n_speeches = 10 + (1 if year in {2016, 2017, 2019} else 0)
        for _ in range(n_speeches):
            topic = TOPICS[int(rng.integers(0, len(TOPICS)))]
            period = PERIOD_FOR[topic](year)
            body_pool = TOPIC_BODIES[(topic, period)]
            body = body_pool[int(rng.integers(0, len(body_pool)))]
            opening = OPENINGS[int(rng.integers(0, len(OPENINGS)))]
            closing = CLOSINGS[int(rng.integers(0, len(CLOSINGS)))]
            party = PARTIES[int(rng.integers(0, len(PARTIES)))]
            month = int(rng.integers(1, 13))
            day = int(rng.integers(1, 28))
            rows.append(
                {
                    "speech_id": speech_id,
                    "text": f"{opening} {topic}. {body}. {closing}",
                    "topic": topic,
                    "frame": period,
                    "party": party,
                    "date": f"{year}-{month:02d}-{day:02d}",
                    "year": year,
                }
            )
            speech_id += 1
    return pd.DataFrame(rows)


def main() -> None:
    df = generate()
    out_path = Path(__file__).parent / "_data" / "hansard_sample.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} speeches to {out_path}")
    print(f"topic distribution: {df['topic'].value_counts().to_dict()}")
    print(f"frame distribution: {df['frame'].value_counts().to_dict()}")
    print(f"party distribution: {df['party'].value_counts().to_dict()}")
    print(f"year range: {df['year'].min()}–{df['year'].max()}")


if __name__ == "__main__":
    main()
