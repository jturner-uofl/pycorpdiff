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

The showcase notebook additionally gets a *polish pass* that injects a
modern theme — hero banner, sticky table-of-contents sidebar, hidden
In[]/Out[] prompts, banded tables, framed charts, system-font
typography. Tutorial and Hansard notebooks keep the classic template
since their narrative scope is smaller.

Run with:

    python scripts/render_notebooks_to_html.py

Outputs land in ``docs/rendered/``. Re-run after re-executing the
notebooks (e.g. via ``jupyter nbconvert --to notebook --execute``).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import vl_convert as vlc

NOTEBOOKS = [
    "examples/pycorpdiff_showcase.ipynb",
    "examples/pycorpdiff_tutorial.ipynb",
    "examples/hansard_demo.ipynb",
]
# Only this notebook gets the polish pass.
POLISH_NOTEBOOKS = {"examples/pycorpdiff_showcase.ipynb"}
OUTPUT_DIR = Path("docs/rendered")


POLISH_CSS = """
:root {
  --pcd-bg: #fafaf9;
  --pcd-fg: #1a1a1a;
  --pcd-muted: #555;
  --pcd-accent: #0b6e7c;
  --pcd-accent-dark: #074d57;
  --pcd-accent-soft: #e6f1f3;
  --pcd-card: #ffffff;
  --pcd-border: #e8e8e6;
  --pcd-code-bg: #1e1e2e;
  --pcd-code-fg: #cdd6f4;
  --pcd-table-band: #f5f5f4;
  --pcd-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.06);
  --pcd-radius: 10px;
  --pcd-font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                   system-ui, 'Helvetica Neue', Arial, sans-serif;
  --pcd-font-display: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                      system-ui, sans-serif;
  --pcd-font-mono: 'JetBrains Mono', 'SF Mono', SFMono-Regular, Menlo,
                   Consolas, monospace;
}

* { box-sizing: border-box; }
html, body { background: var(--pcd-bg); color: var(--pcd-fg); }
body {
  font-family: var(--pcd-font-body);
  font-size: 16px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
  margin: 0;
  padding: 0;
}

/* Hide nbconvert prompts and the anchor-link pilcrows */
.prompt, .input_prompt, .output_prompt { display: none !important; }
.anchor-link { display: none !important; }

/* Hero banner */
.pcd-hero {
  background: linear-gradient(135deg, #062a32 0%, #0b6e7c 55%, #129d96 100%);
  color: white;
  padding: 88px 48px 96px;
  margin-bottom: 0;
  position: relative;
  overflow: hidden;
}
.pcd-hero::after {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse at 80% 20%, rgba(255,255,255,0.10), transparent 60%),
    radial-gradient(ellipse at 10% 90%, rgba(255,255,255,0.06), transparent 50%);
  pointer-events: none;
}
.pcd-hero-inner {
  max-width: 1180px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
}
.pcd-hero h1 {
  font-family: var(--pcd-font-display);
  font-size: 64px;
  font-weight: 800;
  letter-spacing: -0.035em;
  margin: 0 0 16px;
  line-height: 1.0;
  color: white;
  background: none;
}
.pcd-hero .tagline {
  font-size: 22px;
  font-weight: 400;
  opacity: 0.92;
  margin: 0 0 32px;
  max-width: 720px;
  line-height: 1.5;
}
.pcd-hero .chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.pcd-hero .chip {
  display: inline-block;
  padding: 6px 14px;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 999px;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.02em;
  color: white;
  text-decoration: none;
  transition: background 0.15s;
}
.pcd-hero .chip:hover { background: rgba(255,255,255,0.25); }
.pcd-hero .meta {
  margin-top: 32px;
  font-size: 14px;
  opacity: 0.78;
  font-family: var(--pcd-font-mono);
}

/* Layout: sticky TOC + main */
.pcd-page {
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 48px;
  max-width: 1280px;
  margin: 0 auto;
  padding: 48px 32px 96px;
}
@media (max-width: 900px) {
  .pcd-page { grid-template-columns: 1fr; padding: 32px 20px; }
  .pcd-toc { display: none; }
}
.pcd-toc {
  position: sticky;
  top: 24px;
  align-self: start;
  font-size: 13px;
  max-height: calc(100vh - 48px);
  overflow-y: auto;
}
.pcd-toc-title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--pcd-muted);
  margin: 0 0 14px;
  padding-left: 12px;
}
.pcd-toc ul { list-style: none; padding: 0; margin: 0; }
.pcd-toc li.h2 { margin-bottom: 4px; }
.pcd-toc li.h3 { margin-left: 16px; }
.pcd-toc a {
  display: block;
  padding: 6px 12px;
  color: var(--pcd-muted);
  text-decoration: none;
  border-left: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
  border-radius: 0 6px 6px 0;
  line-height: 1.4;
}
.pcd-toc a:hover { color: var(--pcd-accent); background: var(--pcd-accent-soft); }
.pcd-toc a.active {
  color: var(--pcd-accent-dark);
  border-left-color: var(--pcd-accent);
  font-weight: 600;
  background: var(--pcd-accent-soft);
}
.pcd-toc li.h3 a { font-size: 12px; padding: 4px 12px; }

/* Main content column */
#notebook, #notebook-container {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
}
.container { width: 100% !important; max-width: 100% !important; padding: 0 !important; }
.cell { padding: 0 !important; margin: 0 0 28px !important; background: transparent !important; }
.text_cell { background: transparent !important; }
.input_area, .output_area, .inner_cell { background: transparent !important; }

/* Headings */
h1, .text_cell h1 {
  font-family: var(--pcd-font-display);
  font-weight: 800;
  letter-spacing: -0.02em;
  font-size: 36px;
  line-height: 1.15;
  margin: 56px 0 12px;
  border: 0;
  padding: 0;
  color: var(--pcd-fg);
}
h2, .text_cell h2 {
  font-family: var(--pcd-font-display);
  font-weight: 700;
  letter-spacing: -0.02em;
  font-size: 30px;
  margin: 72px 0 18px;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--pcd-accent);
  color: var(--pcd-fg);
  position: relative;
}
h2::before {
  content: '';
  display: block;
  width: 48px;
  height: 4px;
  background: var(--pcd-accent);
  border-radius: 2px;
  margin-bottom: 16px;
}
h3, .text_cell h3 {
  font-family: var(--pcd-font-display);
  font-weight: 700;
  letter-spacing: -0.01em;
  font-size: 22px;
  margin: 40px 0 14px;
  color: var(--pcd-accent-dark);
}

/* Body text */
.text_cell p, .text_cell li {
  font-size: 16.5px;
  line-height: 1.7;
  color: #2a2a2a;
}
.text_cell strong { color: var(--pcd-fg); font-weight: 700; }
.text_cell em { font-style: italic; }
.text_cell hr { border: 0; border-top: 1px dashed var(--pcd-border); margin: 56px 0; }

/* Inline code */
.text_cell code, p code, li code, td code {
  font-family: var(--pcd-font-mono);
  font-size: 13.5px;
  background: var(--pcd-accent-soft);
  color: var(--pcd-accent-dark);
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 500;
}
.text_cell code { white-space: nowrap; }

/* Code blocks — dark themed (catches BOTH Jupyter code cells and
   markdown fenced code, which render through the same .highlight class). */
div.input { margin-bottom: 0 !important; }
.input_area {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
}
.highlight {
  background: var(--pcd-code-bg) !important;
  border-radius: var(--pcd-radius) !important;
  border: 0 !important;
  padding: 16px 18px !important;
  margin: 8px 0 !important;
  box-shadow: var(--pcd-shadow);
  overflow-x: auto;
}
.highlight pre, .highlight code {
  background: transparent !important;
  color: var(--pcd-code-fg) !important;
  font-family: var(--pcd-font-mono) !important;
  font-size: 13.5px !important;
  line-height: 1.6 !important;
  margin: 0 !important;
  padding: 0 !important;
  white-space: pre !important;
  border: 0 !important;
}
/* Inline code inside a .highlight block should NOT get the inline-chip pill. */
.highlight code { display: inline; white-space: pre; }

/* Syntax-token overrides — Catppuccin Mocha palette, high contrast on
   the dark code background. Applied globally because every code block
   now has the dark background. */
.highlight .k, .highlight .kn, .highlight .kc, .highlight .kd,
.highlight .kr, .highlight .kp, .highlight .ow {
  color: #cba6f7 !important; font-weight: 500;
}
.highlight .s, .highlight .s1, .highlight .s2, .highlight .sb,
.highlight .sx, .highlight .sd { color: #a6e3a1 !important; }
.highlight .c, .highlight .c1, .highlight .cm, .highlight .ch,
.highlight .cp { color: #9399b2 !important; font-style: italic; }
.highlight .nf, .highlight .nc, .highlight .nn { color: #89b4fa !important; }
.highlight .nb { color: #f9e2af !important; }
.highlight .mi, .highlight .mf, .highlight .m, .highlight .il {
  color: #fab387 !important;
}
.highlight .o, .highlight .p { color: #bac2de !important; }
.highlight .n, .highlight .na, .highlight .nx { color: #cdd6f4 !important; }
.highlight .kc { color: #fab387 !important; }  /* True/False/None */
.highlight .se { color: #f5c2e7 !important; }  /* String escapes */

/* Output area */
.output_area { padding: 0 !important; margin-top: 10px !important; }
.output_text { font-family: var(--pcd-font-mono); font-size: 13px; }
.output_subarea { max-width: 100% !important; padding: 0 !important; }
.output_stream, .output_text pre {
  background: #f6f5f3 !important;
  border: 1px solid var(--pcd-border) !important;
  border-radius: var(--pcd-radius) !important;
  padding: 14px 18px !important;
  font-family: var(--pcd-font-mono);
  font-size: 13px;
  color: #3a3a3a;
  margin: 6px 0 !important;
  overflow-x: auto;
}

/* DataFrames — banded, modern, breathing */
.dataframe, table.dataframe {
  border-collapse: separate !important;
  border-spacing: 0;
  width: auto !important;
  max-width: 100%;
  margin: 12px 0 !important;
  background: var(--pcd-card);
  border-radius: var(--pcd-radius);
  overflow: hidden;
  box-shadow: var(--pcd-shadow);
  font-family: var(--pcd-font-body);
  font-size: 13.5px;
}
.dataframe thead th {
  background: var(--pcd-accent-soft) !important;
  color: var(--pcd-accent-dark) !important;
  font-weight: 600 !important;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 12px 16px !important;
  border: 0 !important;
  text-align: left !important;
  border-bottom: 2px solid var(--pcd-accent) !important;
}
.dataframe tbody td, .dataframe tbody th {
  padding: 10px 16px !important;
  border: 0 !important;
  border-bottom: 1px solid var(--pcd-border) !important;
  font-family: var(--pcd-font-mono);
  font-size: 12.5px;
  color: #2a2a2a;
}
.dataframe tbody tr:nth-child(even) { background: var(--pcd-table-band); }
.dataframe tbody tr:hover { background: var(--pcd-accent-soft); }
.dataframe tbody tr:last-child td, .dataframe tbody tr:last-child th { border-bottom: 0 !important; }

/* Charts — framed, breathing */
.output_svg, .output_png, .output_jpeg {
  display: block;
  margin: 12px 0 !important;
  padding: 18px !important;
  background: var(--pcd-card);
  border-radius: var(--pcd-radius);
  border: 1px solid var(--pcd-border);
  box-shadow: var(--pcd-shadow);
  overflow-x: auto;
  text-align: center;
}
.output_svg img, .output_svg svg, .output_png img {
  max-width: 100%;
  height: auto;
  display: inline-block;
  margin: 0 auto;
}

/* Anchor scroll offset (don't tuck headings under nothing — this matches body padding) */
h2, h3 { scroll-margin-top: 24px; }

/* Subtle "kbd"-like styling on chip text inside body if used */
kbd {
  font-family: var(--pcd-font-mono);
  background: var(--pcd-accent-soft);
  border: 1px solid var(--pcd-border);
  border-radius: 4px;
  padding: 1px 6px;
  font-size: 12px;
}

/* Footer */
.pcd-footer {
  border-top: 1px solid var(--pcd-border);
  padding: 32px 0 12px;
  margin-top: 80px;
  color: var(--pcd-muted);
  font-size: 13px;
  text-align: center;
}
.pcd-footer a { color: var(--pcd-accent); text-decoration: none; }
.pcd-footer a:hover { text-decoration: underline; }
"""


POLISH_JS = """
(function () {
  // Build a TOC from h2/h3 in the rendered content.
  var main = document.querySelector('#pcd-main');
  if (!main) return;
  var headings = main.querySelectorAll('h2, h3');
  var tocList = document.querySelector('#pcd-toc-list');
  if (!tocList || !headings.length) return;
  var items = [];
  headings.forEach(function (h) {
    if (!h.id) return;
    var li = document.createElement('li');
    li.className = h.tagName.toLowerCase();
    var a = document.createElement('a');
    a.href = '#' + h.id;
    // Strip trailing pilcrow text if any.
    var label = h.textContent.replace(/¶.*$/, '').trim();
    // Compact "Part X — Foo" labels in the sidebar.
    a.textContent = label;
    a.dataset.target = h.id;
    li.appendChild(a);
    tocList.appendChild(li);
    items.push({ id: h.id, link: a, el: h });
  });

  // Scrollspy: highlight whichever heading is currently in view.
  var current = null;
  function onScroll() {
    var y = window.scrollY + 80;
    var active = items[0];
    for (var i = 0; i < items.length; i++) {
      if (items[i].el.offsetTop <= y) active = items[i];
    }
    if (active && active !== current) {
      if (current) current.link.classList.remove('active');
      active.link.classList.add('active');
      current = active;
    }
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
"""


HERO_HTML = """\
<div class="pcd-hero">
  <div class="pcd-hero-inner">
    <h1>pycorpdiff</h1>
    <p class="tagline">
      Comparative corpus analysis for modern Python workflows — keyness,
      collocations, semantic drift, temporal trajectories, and the cluster
      structure of discourse itself.
    </p>
    <div class="chips">
      <a class="chip" href="#Part-I-%E2%80%94-The-corpus">I · the corpus</a>
      <a class="chip" href="#Part-II-%E2%80%94-Lexical-fingerprints">II · keyness</a>
      <a class="chip" href="#Part-III-%E2%80%94-Collocational-fingerprints">III · collocations</a>
      <a class="chip" href="#Part-IV-%E2%80%94-The-temporal-arc">IV · time</a>
      <a class="chip" href="#Part-V-%E2%80%94-Beneath-frequency:-semantic-shift">V · semantic shift</a>
      <a class="chip" href="#Part-VI-%E2%80%94-The-fanout">VI · the fanout</a>
      <a class="chip" href="#Part-VII-%E2%80%94-Cross-validation-receipts">VII · receipts</a>
      <a class="chip" href="#Part-VIII-%E2%80%94-The-plumbing">VIII · plumbing</a>
    </div>
    <div class="meta">
      synthetic UK Hansard fixture · 193 speeches · 4 parties · 4 topics · 19 years
    </div>
  </div>
</div>
"""


FOOTER_HTML = """\
<footer class="pcd-footer">
  <p>
    Generated from
    <a href="../examples/pycorpdiff_showcase.ipynb"><code>examples/pycorpdiff_showcase.ipynb</code></a>
    via <code>scripts/render_notebooks_to_html.py</code>.
    Charts pre-rendered as inline SVG — no CDN, no JavaScript runtime.
  </p>
</footer>
"""


def polish_html(html_path: Path) -> None:
    """Apply the modern theme polish in-place.

    Mutates the file at ``html_path`` to:
    1. Replace ``<title>`` with a proper one.
    2. Strip the dead CDN scripts (jquery / require.js / mermaid).
    3. Append our CSS bundle in ``<head>``.
    4. Insert the hero ``<div>`` immediately after ``<body>``.
    5. Wrap ``#notebook`` in a TOC + main layout.
    6. Inject the TOC-building JS just before ``</body>``.
    """
    html = html_path.read_text()

    # 1. Title
    html = re.sub(
        r"<title>[^<]*</title>",
        "<title>pycorpdiff — the showcase</title>",
        html,
        count=1,
    )

    # 2. Drop the CDN script tags + mermaid block (charts are inline SVG, no JS needed).
    html = re.sub(
        r'<script[^>]*src="https://cdnjs\.cloudflare\.com[^"]+"[^>]*>\s*</script>',
        "",
        html,
    )
    html = re.sub(
        r'<script type="module">\s*import mermaid.*?</script>',
        "",
        html,
        flags=re.DOTALL,
    )

    # 3. Inject CSS just before </head>.
    polish_block = f"<style>{POLISH_CSS}</style>"
    html = html.replace("</head>", polish_block + "</head>", 1)

    # 4 + 5. Wrap the notebook container with hero + layout shell.
    # nbconvert wraps the body content in `<div id="notebook">` — we
    # insert the hero before it and a TOC sidebar next to it.
    toc_block = (
        '<aside class="pcd-toc">'
        '<div class="pcd-toc-title">Contents</div>'
        '<ul id="pcd-toc-list"></ul>'
        "</aside>"
    )
    # Open the new layout shell.
    new_open = (
        HERO_HTML
        + '<div class="pcd-page">'
        + toc_block
        + '<main id="pcd-main">'
    )
    # Close it before </body>.
    new_close = "</main></div>" + FOOTER_HTML

    # nbconvert's classic template emits `<body class="...">...content...</body>`.
    # We don't want to capture the whole body; just inject right after <body...>
    # and right before </body>.
    body_open_match = re.search(r"<body[^>]*>", html)
    if body_open_match is None:
        raise RuntimeError("no <body> tag found in HTML")
    insert_after_idx = body_open_match.end()
    html = html[:insert_after_idx] + new_open + html[insert_after_idx:]
    html = html.replace("</body>", new_close + f"<script>{POLISH_JS}</script></body>", 1)

    html_path.write_text(html)


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
            if nb_path_str in POLISH_NOTEBOOKS:
                polish_html(html)
                print("  applied polish pass (hero + TOC + modern theme)")
            with html.open() as f:
                content = f.read()
            n_charts = content.count("data:image/svg+xml")
            size_kb = html.stat().st_size // 1024
            print(f"  → {html}  ({size_kb} KB, {n_charts} embedded charts)")
        finally:
            patched.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
