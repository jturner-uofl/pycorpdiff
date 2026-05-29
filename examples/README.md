# Examples

This directory hosts the executable example notebooks. All three are
re-executed in CI on every push so they can't silently drift from the
public API.

Self-contained HTML renders of each notebook (with every chart
pre-rendered as inline SVG — no CDN, no JS) live in
[`docs/rendered/`](../docs/rendered/). Regenerate with
[`scripts/render_notebooks_to_html.py`](../scripts/render_notebooks_to_html.py).

- **🌟 [`pycorpdiff_showcase.ipynb`](pycorpdiff_showcase.ipynb)** —
  *the* showcase. One coherent research narrative (UK parliamentary
  discourse on migration) driving every analytical surface in
  pycorpdiff: keyness with volcano + bar + KWIC explain, collocation
  shift, temporal trajectories with Wilson CIs, changepoint
  detection, interrupted time series, semantic shift via averaged
  contextual embeddings, neighbourhood drift, cross-party and
  cross-topic fanouts, plus live cross-validation against
  Scattertext on the 2012 US Conventions corpus and reference values
  from Rayson + HistWords. Also exercises polars / DuckDB / custom
  tokenizer interop. **Start here if you want the full tour.**
- [`pycorpdiff_tutorial.ipynb`](pycorpdiff_tutorial.ipynb) — the
  introductory guided tour on a synthetic two-frame fixture. Smaller
  scope, gentler ramp.
- [`hansard_demo.ipynb`](hansard_demo.ipynb) — a focused worked
  example on the bundled Hansard sample. Less infrastructure, more
  research narrative.

## Conventions

**The notebooks must stay current.** Whenever a public API changes,
update the relevant notebook and its rendered HTML in the same commit.
CI re-executes all three notebooks on every push and will fail loud
if any one drifts from the live API surface.

Run the notebook locally with:

```bash
jupyter lab examples/pycorpdiff_tutorial.ipynb
```
