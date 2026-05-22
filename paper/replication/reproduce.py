"""Regenerate every figure and table in paper.tex.

Run from the repository root:

    python paper/replication/reproduce.py

Writes JSON outputs to paper/replication/paper_outputs.json and PNG /
SVG figures to paper/figures/. The CI workflow re-runs this script on
every push and fails if any of the recorded outputs change — that's
how the paper stays in sync with the package.

Outputs (all keyed in paper_outputs.json):

  figure_1_volcano      — KeynessResult.plot() volcano (saved as SVG + PNG)
  figure_2_collocations — collocation_shift diverging bar
  figure_3_trajectory   — trajectory line + Wilson CI band
  table_1_keyness_top12 — top-12 keyness rows as a small JSON
  table_2_its_results   — ITS coefficients for the synthetic 2016 event
  table_3_semantic      — semantic_shift cosine distance + context counts
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import pycorpdiff as pcd

# Reproducible-by-construction corpus — same one used in the tutorial.
HUMANISING_TEMPLATES = [
    "the migrant worker arrived and the migrant family settled here",
    "the migrant community grew and migrant workers thrived together",
    "the migrant family settled and the migrant community welcomed them",
    "the migrant worker and migrant rights advanced in the workplace",
    "the migrant family arrived seeking refuge and dignity",
    "the migrant community organised and the migrant worker spoke out",
    "the migrant family settled and migrant children attended school",
    "the migrant worker contributed and the migrant family flourished",
]
CRIMINALISING_TEMPLATES = [
    "the migrant criminal threat and the migrant invasion grew worse",
    "the migrant threat and the migrant crime increased again",
    "the migrant invasion of migrant criminal gangs spread further",
    "the migrant criminal gangs and migrant invasion stayed dangerous",
    "the migrant threat persisted and the migrant criminal risk grew",
    "the migrant invasion and migrant criminal gangs threatened the border",
    "the migrant criminal element and the migrant threat alarmed residents",
    "the migrant gangs and migrant invasion narrative dominated coverage",
]
EVENT_YEAR = 2016
YEAR_RANGE = range(2005, 2024)


def build_corpus() -> pcd.Corpus:
    rows: list[dict[str, object]] = []
    for year in YEAR_RANGE:
        n_h = 8 if year < EVENT_YEAR else 2
        n_c = 2 if year < EVENT_YEAR else 8
        for i in range(n_h):
            rows.append(
                {
                    "text": HUMANISING_TEMPLATES[i % len(HUMANISING_TEMPLATES)],
                    "outlet": "humanising",
                    "date": f"{year}-{(i % 12) + 1:02d}-15",
                }
            )
        for i in range(n_c):
            rows.append(
                {
                    "text": CRIMINALISING_TEMPLATES[i % len(CRIMINALISING_TEMPLATES)],
                    "outlet": "criminalising",
                    "date": f"{year}-{(i % 12) + 1:02d}-15",
                }
            )
    return pcd.from_dataframe(
        pd.DataFrame(rows), text_col="text", meta_cols=("outlet", "date")
    )


def main() -> None:
    here = Path(__file__).parent
    figures = here.parent / "figures"
    figures.mkdir(exist_ok=True)

    corpus = build_corpus()
    humanising = corpus.slice(outlet="humanising")
    criminalising = corpus.slice(outlet="criminalising")

    # --- Section §5: keyness ---------------------------------------------------
    keyness = pcd.compare(humanising, criminalising).keyness(
        min_count=3, dispersion=True
    )
    keyness.plot().save(str(figures / "figure_1_volcano.svg"))
    keyness.plot(kind="bar", n=12).save(str(figures / "figure_1_topn.svg"))

    # --- collocations ----------------------------------------------------------
    shift = pcd.compare(humanising, criminalising).collocation_shift(
        "migrant", window=3, min_count=3, measure="logDice"
    )
    shift.plot(n=15).save(str(figures / "figure_2_collocations.svg"))

    # --- temporal trajectory + ITS --------------------------------------------
    trajectory = pcd.track(corpus, ["worker", "criminal", "family"]).over_time(
        freq="Y", time_col="date"
    )
    trajectory.plot().save(str(figures / "figure_3_trajectory.svg"))
    its_results = trajectory.interrupted_time_series(
        event_date="2016", target="criminal"
    )
    cps = trajectory.changepoints(target="criminal")

    # --- semantic (HashEmbedder for byte-reproducibility) ---------------------
    sem = pcd.compare(humanising, criminalising).semantic_shift(
        target="migrant", embedder=pcd.HashEmbedder(dim=64), window=3
    )

    # --- record everything --------------------------------------------------
    outputs = {
        "table_1_keyness_top12": keyness.table.head(12).to_dict(orient="records"),
        "table_2_its_results": its_results.to_dict(orient="records"),
        "table_3_semantic": sem.table.to_dict(orient="records"),
        "changepoints": cps.astype(str).to_dict(orient="records"),
        "corpus_size": {
            "n_documents": len(corpus),
            "n_tokens": corpus.total_tokens(),
        },
    }
    out_path = here / "paper_outputs.json"
    out_path.write_text(json.dumps(outputs, indent=2, default=str))
    print(f"wrote {out_path}")
    for figname in (
        "figure_1_volcano.svg",
        "figure_1_topn.svg",
        "figure_2_collocations.svg",
        "figure_3_trajectory.svg",
    ):
        print(f"wrote {figures / figname}")


if __name__ == "__main__":
    main()
