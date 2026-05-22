"""Render the example notebooks to self-contained HTML.

altair charts emit ``application/vnd.vegalite.v6+json`` mimetypes, which
JupyterLab / VS Code / nbviewer / GitHub all render natively — but
nbconvert's HTML template doesn't know how to render that mimetype and
falls back to text/plain placeholders ("alt.Chart(...)").

This script bridges that gap: for each notebook, we use ``vl-convert``
to pre-render every Vega-Lite spec into an SVG, replace the spec with
the SVG in the cell outputs, then export to HTML via nbconvert's
classic template. The result is a fully self-contained HTML file
(no CDN, no JS) with every chart embedded as a base64 ``<img>``.

Run with:

    python scripts/render_notebooks_to_html.py

Outputs land in ``docs/rendered/``. Re-run after re-executing the
notebooks (e.g. via ``jupyter nbconvert --to notebook --execute``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import vl_convert as vlc

NOTEBOOKS = [
    "examples/pycorpdiff_showcase.ipynb",
    "examples/pycorpdiff_tutorial.ipynb",
    "examples/hansard_demo.ipynb",
]
OUTPUT_DIR = Path("docs/rendered")


def patch_notebook(nb_path: Path) -> Path:
    """Replace every Vega-Lite output with a pre-rendered SVG.

    Returns the path to the temporary patched notebook.
    """
    with nb_path.open() as f:
        nb = json.load(f)

    converted = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            vegalite_mimes = [m for m in data if "vegalite" in m]
            if not vegalite_mimes:
                continue
            spec = data[vegalite_mimes[0]]
            svg = vlc.vegalite_to_svg(json.dumps(spec))
            # nbconvert prefers vegalite > svg > text/plain; drop both
            # non-renderable types so the SVG wins the priority lookup.
            for m in vegalite_mimes:
                del data[m]
            data.pop("text/plain", None)
            data["image/svg+xml"] = svg
            converted += 1

    patched = nb_path.with_suffix(".patched.ipynb")
    with patched.open("w") as f:
        json.dump(nb, f, indent=1)
    print(f"  patched {converted:>2} chart(s) → {patched}")
    return patched


def render_to_html(patched: Path, output_dir: Path) -> Path:
    """Run nbconvert on the patched notebook with the classic template."""
    output_name = patched.name.replace(".patched.ipynb", ".html")
    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "html",
            "--template", "classic",
            "--output-dir", str(output_dir),
            "--output", output_name,
            str(patched),
        ],
        check=True,
    )
    return output_dir / output_name


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for nb_path_str in NOTEBOOKS:
        nb_path = Path(nb_path_str)
        if not nb_path.exists():
            print(f"skipping missing notebook {nb_path}")
            continue
        print(f"Processing {nb_path}...")
        patched = patch_notebook(nb_path)
        try:
            html = render_to_html(patched, OUTPUT_DIR)
            with html.open() as f:
                content = f.read()
            n_charts = content.count("data:image/svg+xml")
            size_kb = html.stat().st_size // 1024
            print(f"  → {html}  ({size_kb} KB, {n_charts} embedded charts)")
        finally:
            patched.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
