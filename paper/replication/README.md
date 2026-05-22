# Paper replication

This directory regenerates every figure and table in `paper/paper.tex`
from the published package code.

## Run

From the repository root:

```bash
pip install -e ".[viz,temporal]"
python paper/replication/reproduce.py
```

The script writes:

- `paper/figures/figure_1_volcano.svg` — keyness volcano
- `paper/figures/figure_1_topn.svg` — top-N keyness bar
- `paper/figures/figure_2_collocations.svg` — collocation diverging bar
- `paper/figures/figure_3_trajectory.svg` — temporal trajectory + CI band
- `paper/replication/paper_outputs.json` — every numeric table cited
  in the paper (top-12 keyness rows, ITS coefficients, semantic-shift
  centroid distances, detected changepoints, corpus size).

## What's reproducible

The synthetic corpus is deterministic by construction — same templates,
fixed publication-volume ratios, no randomness. Every metric in
`paper_outputs.json` should be byte-identical across runs. The
`HashEmbedder` used in §8 (semantic shift) is deterministic too —
SHA-256-seeded RNG per input string.

For the real-world worked examples referenced in §5 of the paper draft
(UK Hansard, CORD-19), see `paper/replication/realdata/` (to be added).

## CI

A future `paper` job in `.github/workflows/ci.yml` will run this script
and assert the recorded JSON matches a frozen snapshot. That is the
backstop that keeps the paper's numerical claims aligned with the
package code.
