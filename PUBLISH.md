# Publication checklist

The repository is **99% complete for JSS submission.** This document
collects every placeholder that needs a real value once the repository
is public on GitHub, the package is on PyPI, and a Zenodo DOI is
minted for the tagged release. Hit these in order and the paper +
package will be 100% submission-ready.

## 1 · Decide the GitHub URL

The placeholder throughout is `https://github.com/jasonsturner/pycorpdiff`.
If you publish under a different organization or rename the repo,
search-and-replace across:

```bash
grep -rn "github.com/jasonsturner/pycorpdiff" \
  README.md CITATION.cff pyproject.toml paper/paper.tex \
  paper/replication/README.md docs/
```

Files that reference the GitHub URL:
- `README.md` — clone instructions (line ~110) + the `Documentation`
  link block at the bottom
- `CITATION.cff` — `repository-code:` field
- `pyproject.toml` — `[project.urls]` Homepage / Repository / Issues
  / Documentation
- `paper/paper.tex` — `\Address{...}` URL line + the `\section*{Computational details}` section
- `docs/` — any internal links

## 2 · Publish to PyPI

When ready to tag `v0.1.0` (drop the `a0` suffix for the JSS submission
or keep it as `0.1.0a0` — JSS accepts pre-1.0 versions):

```bash
# Bump version + classifier (pyproject.toml)
#   version = "0.1.0"
#   classifiers: "Development Status :: 4 - Beta"  (or 5 - Production/Stable)
# Cut the tag locally
git tag -a v0.1.0 -m "v0.1.0 — JSS submission"
git push origin main --tags

# Build + publish
uv build
uv publish --token "$PYPI_TOKEN"
```

After publication, update:

- **`README.md`** — uncomment the badge block at the top
  (PyPI / Python versions / CI / DOI / License). Five HTML-commented
  shields lines.
- **`README.md`** — replace the `## Installation` section's clone
  instructions with the PyPI install commands. The standard block:

  ```bash
  pip install pycorpdiff                # base
  pip install 'pycorpdiff[viz]'         # + altair / matplotlib / networkx
  pip install 'pycorpdiff[temporal]'    # + ruptures / statsmodels
  pip install 'pycorpdiff[semantic]'    # + sentence-transformers
  pip install 'pycorpdiff[all]'         # everything
  ```

- **`pyproject.toml`** — bump `Development Status` classifier:

  ```toml
  classifiers = [
      "Development Status :: 4 - Beta",   # was: Pre-Alpha
      # ... rest unchanged ...
  ]
  ```

- **`paper/paper.tex`** — in the `\section*{Computational details}`,
  replace the `% TODO post-publish: PyPI install command + Zenodo DOI`
  comment with one prose sentence covering the install path:

  > The released version of `pycorpdiff` is available on PyPI; install
  > via `pip install pycorpdiff` for the base, or with the relevant
  > extras (`[viz]`, `[temporal]`, `[semantic]`) for full functionality.

## 3 · Mint a Zenodo DOI

The simplest path: enable the Zenodo–GitHub integration on the public
repository, then push a tagged release. Zenodo mints a DOI
automatically for the tag and gives you both a record-specific DOI
(e.g. `10.5281/zenodo.1234567`) and a concept DOI that always
resolves to the latest release.

Once the DOI exists:

- **`CITATION.cff`** — add:

  ```yaml
  doi: 10.5281/zenodo.<RECORD>
  identifiers:
    - type: doi
      value: 10.5281/zenodo.<RECORD>
      description: This specific release
  ```

  GitHub's "Cite this repository" widget will then surface a proper
  DOI in the BibTeX it generates.

- **`README.md`** — uncomment the DOI shield in the badge block:

  ```html
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.<RECORD>.svg)](https://doi.org/10.5281/zenodo.<RECORD>)
  ```

- **`paper/paper.tex`** — in the `\section*{Computational details}`,
  cite the Zenodo DOI for byte-identical reproducibility:

  > The exact source tree used to generate this paper's figures and
  > tables is archived at
  > [10.5281/zenodo.<RECORD>](https://doi.org/10.5281/zenodo.<RECORD>).

## 4 · Acknowledgments

In `paper/paper.tex`, replace the placeholder comment:

```latex
% TODO post-publish: acknowledgements.
```

with the real acknowledgments paragraph(s). The existing prose already
acknowledges the dependency maintainers; if you want to thank specific
collaborators, reviewers, or institutions, add those before submission.

## 5 · arXiv preprint (optional but recommended)

JSS review cycles are 12-18 months. A parallel arXiv preprint is the
standard mechanism to start accruing citations during review. The
process:

```bash
# Use the compiled PDF from `tectonic --outdir /tmp/jss-build paper/paper.tex`
# Submit at https://arxiv.org/submit, category cs.CL, cross-list to stat.AP
```

Once arXiv assigns an ID:

- Add the arXiv ID to `CITATION.cff` under `references:`
- Update the BibTeX block in `README.md`'s `## Citation` section with
  the arXiv reference alongside the JSS entry
- Optionally add an `arXiv` badge to the README badge block

## 6 · Final sanity checks before submission

Run these once before sending:

```bash
# All gates clean?
uv run --extra dev pytest -q
uv run --extra dev ruff check src/ tests/
uv run --extra dev mypy --strict src/

# Replication archive deterministic?
uv run --extra dev --extra viz --extra temporal --extra semantic --extra paper \
  python paper/replication/reproduce.py
git diff --stat paper/replication/paper_outputs.json  # should be empty

# Paper compiles cleanly?
mkdir -p /tmp/jss-build
tectonic --outdir /tmp/jss-build paper/paper.tex
# Expect: 21 pages, no LaTeX errors, no "undefined reference" warnings

# Notebook executes cleanly?
uv run --extra dev --extra viz --extra semantic --extra temporal \
  --extra polars --extra duckdb --extra paper \
  jupyter nbconvert --to notebook --execute \
  examples/pycorpdiff_showcase.ipynb --output pycorpdiff_showcase.ipynb
```

## Files this checklist touches

When all six items above are done, the diff against the current `main`
should be confined to:

| File | Section / lines |
|---|---|
| `README.md` | Top badge block (uncomment 5 lines); install block (rewrite); citation block (update BibTeX) |
| `CITATION.cff` | Add `doi:` and `identifiers:` |
| `pyproject.toml` | Bump `Development Status` classifier |
| `paper/paper.tex` | Computational details (PyPI sentence + Zenodo citation); Acknowledgments (replace placeholder) |
| `PUBLISH.md` | Delete this file once executed |

Everything else — code, tests, paper prose, figures, bibliography,
worked examples, benchmarks, replication, CI, documentation — is
ready as-is.
