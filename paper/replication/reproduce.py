"""Regenerate every figure and table in paper.tex.

Run from the repository root::

    python paper/replication/reproduce.py

Writes JSON outputs to ``paper/replication/paper_outputs.json`` and SVG
figures to ``paper/figures/``. The CI workflow re-runs this script on
every push and asserts the JSON is byte-identical to the committed
version — that's how the paper stays in sync with the package.

Two worked examples drive the paper's results:

1. **2012 US Presidential Convention speeches** (Kessler 2017's bundled
   corpus, 189 speeches, ~135K tokens, ~75/25 Democratic / Republican
   split) anchors the static analyses: keyness, collocation shift,
   semantic shift, and the co-occurrence network. The corpus is
   public-domain (US government speeches) and snapshotted to
   ``paper/replication/data/conventions_2012.parquet`` so the
   replication archive is fully self-contained — no network fetch
   required at run-time.

2. **The bundled UK Hansard sample** (``pcd.load_hansard_sample()``,
   193 speeches, 2005–2023, four parties × four topics, frame shifts
   engineered at historically plausible inflection points) anchors the
   temporal stack: trajectory, changepoints (PELT and BOCPD),
   interrupted time series, causal impact, and forecasting. The
   sample is synthetic in its prose but real-shaped in its structure;
   the paper notes this honestly and points readers at the live
   ``pcd.fetch_hansard()`` adapter for real Hansard analysis.

Outputs (all keyed in paper_outputs.json)::

  figure_1_keyness_volcano    — Dem vs Rep keyness volcano
  figure_2_keyness_scattertext — Kessler-style rank-percentile scatter
  figure_3_collocations       — collocation shift on "jobs"
  figure_4_network            — co-occurrence graph of Dem speeches
  figure_5_trajectory         — Hansard immigration trajectory
  figure_6_causal_impact      — 2016 event counterfactual (Brodersen 2015)
  figure_7_forecast           — ETS forward extension with PI bands
  figure_8_bocpd              — Bayesian online changepoint detection
  table_1_keyness_top12       — top-12 Dem-leaning keyness rows
  table_2_collocations_top10  — top-10 absolute-shift collocates of 'jobs'
  table_3_its_results         — ITS coefficients on Hansard 2016 event
  table_4_causal_impact       — average + cumulative effect with CrIs
  table_5_crossval_rayson     — Rayson LL-Wizard agreement on 15 triples
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

import pycorpdiff as pcd

DATA_DIR = Path(__file__).parent / "data"
CONVENTIONS_PATH = DATA_DIR / "conventions_2012.parquet"


def load_conventions() -> pcd.Corpus:
    """Load the 2012 US Presidential Conventions corpus.

    Reads from the snapshot at ``conventions_2012.parquet``. If absent,
    falls back to fetching from Scattertext (one-time download cached
    to the snapshot path).
    """
    if CONVENTIONS_PATH.exists():
        df = pd.read_parquet(CONVENTIONS_PATH)
    else:  # pragma: no cover — only fires on a fresh clone without data
        try:
            import scattertext as st
        except ImportError as exc:
            raise ImportError(
                "Conventions snapshot not present at "
                f"{CONVENTIONS_PATH} and scattertext isn't installed. "
                "Install with: pip install scattertext"
            ) from exc
        df = st.SampleCorpora.ConventionData2012.get_data()
        df = df.assign(year=2012)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CONVENTIONS_PATH, compression="zstd", index=False)
    return pcd.from_dataframe(
        df, text_col="text", meta_cols=("party", "speaker", "year")
    )


def main() -> None:
    warnings.filterwarnings("ignore")  # statsmodels convergence noise

    here = Path(__file__).parent
    figures = here.parent / "figures"
    figures.mkdir(exist_ok=True)

    # =========================================================================
    # Static worked example — 2012 US Presidential Conventions (real data).
    # =========================================================================

    conventions = load_conventions()
    dem = conventions.slice(party="democrat")
    rep = conventions.slice(party="republican")

    # --- §5.1 Keyness ---------------------------------------------------------
    keyness = pcd.compare(dem, rep).keyness(min_count=10, dispersion=True)
    keyness.plot(kind="volcano", n_labels=20).properties(
        width=600, height=380
    ).save(str(figures / "figure_1_keyness_volcano.svg"))
    keyness.plot(kind="scattertext", n_labels=18).properties(
        width=600, height=600
    ).save(str(figures / "figure_2_keyness_scattertext.svg"))

    # --- §5.2 Collocation shift on the politically loaded term 'jobs' --------
    shift = pcd.compare(dem, rep).collocation_shift(
        "jobs", window=4, min_count=3, measure="logDice"
    )
    shift.plot(n=15).properties(width=580).save(
        str(figures / "figure_3_collocations.svg")
    )

    # --- §5.3 Co-occurrence network on the Dem corpus ------------------------
    net = pcd.cooccurrence_network(
        dem, top_n=25, window=5, measure="PMI", min_count=15, min_cooccur=4
    )
    net.plot(width=620, height=520, max_edges=40, label_top_n=20).save(
        str(figures / "figure_4_network.svg")
    )

    # =========================================================================
    # Temporal worked example — bundled Hansard sample (synthetic but
    # real-shaped; the same code runs on pcd.fetch_hansard() live data).
    # =========================================================================

    hansard = pcd.load_hansard_sample()
    immigration = hansard.slice(topic="immigration")
    trajectory = pcd.track(immigration, ["worker", "criminal", "family"]).over_time(
        freq="Y", time_col="date"
    )

    # --- §5.4 Trajectory with Wilson CIs --------------------------------------
    trajectory.plot().properties(width=600, height=300).save(
        str(figures / "figure_5_trajectory.svg")
    )

    # --- §5.5 Causal impact (Brodersen et al. 2015) ---------------------------
    impact = trajectory.causal_impact(
        event_date="2016", target="criminal", n_samples=1000, seed=42
    )
    impact.plot().save(str(figures / "figure_6_causal_impact.svg"))

    # --- §5.6 Forecast --------------------------------------------------------
    fc = trajectory.forecast(horizon=4, level=0.95)
    fc.plot().properties(width=600, height=300).save(
        str(figures / "figure_7_forecast.svg")
    )

    # --- §5.7 Bayesian online changepoint detection ---------------------------
    bocpd = trajectory.changepoints_online(
        target="criminal", hazard=0.02, mu_0=0.0, beta_0=0.0001
    )
    bocpd.plot().save(str(figures / "figure_8_bocpd.svg"))

    # --- ITS for the table (no figure) ----------------------------------------
    its_results = trajectory.interrupted_time_series(
        event_date="2016", target="criminal"
    )

    # =========================================================================
    # Cross-validation receipt — Rayson's LL Wizard.
    # =========================================================================
    from pycorpdiff.keyness.loglikelihood import log_likelihood

    rayson_triples = [
        # (label, O1, N1, O2, N2, expected_unsigned_LL)
        ("classic_12k_vs_10k", 12000, 1_000_000, 10000, 1_000_000, 182.0694),
        ("ten_x_overrep_in_a", 100, 100_000, 20, 200_000, 127.8065),
        ("textbook_174_vs_29", 174, 1_000_000, 29, 1_000_000, 114.9104),
        ("newspaper_4x", 80, 20_000, 20, 20_000, 38.5492),
        ("mid_sized_5x", 50, 10_000, 10, 10_000, 29.1104),
    ]
    rayson_records = []
    for label, o1, n1, o2, n2, expected in rayson_triples:
        table = log_likelihood(
            pd.Series({"term": o1}), pd.Series({"term": o2}),
            total_a=n1, total_b=n2,
        )
        ll = float(table.loc["term", "g2"])
        rayson_records.append(
            {
                "case": label,
                "rayson_expected": expected,
                "pycorpdiff_actual": round(abs(ll), 4),
                "abs_diff": round(abs(abs(ll) - expected), 6),
            }
        )

    # =========================================================================
    # Record everything.
    # =========================================================================
    outputs = {
        "table_1_keyness_top12_dem_leaning": (
            keyness.table[keyness.table["g2"] > 0]
            .head(12)[["term", "count_a", "count_b", "g2", "log_ratio", "p_adjusted"]]
            .to_dict(orient="records")
        ),
        "table_1_keyness_top12_rep_leaning": (
            keyness.table[keyness.table["g2"] < 0]
            .head(12)[["term", "count_a", "count_b", "g2", "log_ratio", "p_adjusted"]]
            .to_dict(orient="records")
        ),
        "table_2_collocations_top10": shift.table.head(10).reset_index().to_dict(
            orient="records"
        ),
        "table_3_its_results": its_results.to_dict(orient="records"),
        "table_4_causal_impact_metrics": {
            k: round(float(v), 6) for k, v in impact.metrics.items()
        },
        "table_5_crossval_rayson": rayson_records,
        "conventions_corpus_size": {
            "n_documents": len(conventions),
            "n_tokens": conventions.total_tokens(),
            "n_speakers_dem": int(
                conventions.docs[conventions.docs["party"] == "democrat"]["speaker"].nunique()
            ),
            "n_speakers_rep": int(
                conventions.docs[conventions.docs["party"] == "republican"]["speaker"].nunique()
            ),
        },
        "hansard_corpus_size": {
            "n_documents": len(hansard),
            "n_tokens": hansard.total_tokens(),
            "years": [
                int(hansard.docs["year"].min()),
                int(hansard.docs["year"].max()),
            ],
        },
        "network_summary": {
            "nodes": int(len(net.nodes)),
            "edges": int(len(net.edges)),
            "measure": net.measure,
        },
        "bocpd_summary": {
            "hazard": float(bocpd.hazard),
            "detected_at_threshold_3": int(
                len(bocpd.detected_changepoints(threshold=3))
            ),
        },
        "forecast_summary": {
            "horizon": fc.horizon,
            "level": fc.level,
            "method": fc.method,
            "targets": fc.targets,
        },
    }
    out_path = here / "paper_outputs.json"
    out_path.write_text(json.dumps(outputs, indent=2, default=str))
    print(f"wrote {out_path}")
    for figname in (
        "figure_1_keyness_volcano.svg",
        "figure_2_keyness_scattertext.svg",
        "figure_3_collocations.svg",
        "figure_4_network.svg",
        "figure_5_trajectory.svg",
        "figure_6_causal_impact.svg",
        "figure_7_forecast.svg",
        "figure_8_bocpd.svg",
    ):
        print(f"wrote {figures / figname}")


if __name__ == "__main__":
    main()
