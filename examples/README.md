# Examples

This directory hosts the canonical end-to-end tutorial notebook.

- [`pycorpdiff_tutorial.ipynb`](pycorpdiff_tutorial.ipynb) — the full guided
  tour. Currently a Phase-0 stub; sections are filled in as each phase lands.

## Conventions

Mirroring the rule that's served `pysofra` well: **the tutorial must stay
current.** Whenever a public API changes, update the notebook and its
rendered HTML in the same change. CI executes the notebook in the
documentation job and will fail loud if the notebook drifts.

Run the notebook locally with:

```bash
jupyter lab examples/pycorpdiff_tutorial.ipynb
```
