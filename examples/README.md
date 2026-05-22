# Examples

This directory hosts the executable example notebooks.

- [`pycorpdiff_tutorial.ipynb`](pycorpdiff_tutorial.ipynb) — the full
  guided tour on a synthetic two-frame fixture. Covers every analytical
  surface in pycorpdiff.
- [`hansard_demo.ipynb`](hansard_demo.ipynb) — a worked example on the
  bundled Hansard-style sample corpus (`pcd.load_hansard_sample()`).
  Drives keyness, collocation shift, temporal trajectories, changepoint
  detection, ITS, and semantic shift on one realistic-shape corpus,
  with cross-party and cross-topic comparisons.

## Conventions

Mirroring the rule that's served `pysofra` well: **the tutorial must stay
current.** Whenever a public API changes, update the notebook and its
rendered HTML in the same change. CI executes the notebook in the
documentation job and will fail loud if the notebook drifts.

Run the notebook locally with:

```bash
jupyter lab examples/pycorpdiff_tutorial.ipynb
```
