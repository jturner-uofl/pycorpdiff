# Rendered notebooks

Self-contained HTML exports of the example notebooks, with every
altair chart pre-rendered to an inline SVG via
[`vl-convert`](https://github.com/vega/vl-convert). No CDN, no
JavaScript — just open in any browser.

| File | Source notebook | Charts |
|---|---|---|
| [`pycorpdiff_showcase.html`](pycorpdiff_showcase.html) | [`examples/pycorpdiff_showcase.ipynb`](../../examples/pycorpdiff_showcase.ipynb) | 8 |
| [`pycorpdiff_tutorial.html`](pycorpdiff_tutorial.html) | [`examples/pycorpdiff_tutorial.ipynb`](../../examples/pycorpdiff_tutorial.ipynb) | 4 |
| [`hansard_demo.html`](hansard_demo.html) | [`examples/hansard_demo.ipynb`](../../examples/hansard_demo.ipynb) | 3 |

## Regenerating

```bash
pip install -e ".[paper]"   # for vl-convert
python scripts/render_notebooks_to_html.py
```

The script does three things per notebook:

1. Loads the executed `.ipynb`.
2. For every output cell carrying an `application/vnd.vegalite.v6+json`
   spec, calls `vl_convert.vegalite_to_svg` and replaces the spec with
   the SVG (also drops the `text/plain` placeholder).
3. Exports to HTML via nbconvert's classic template; the SVG ends up
   embedded as a base64 `<img>` tag.

The native `.ipynb` files still emit Vega-Lite JSON for in-browser
rendering on GitHub / JupyterLab / VS Code / nbviewer; this static
HTML export is for places that don't speak Vega.
