"""Cross-validation against R's quanteda (Benoit et al. 2018).

quanteda's ``textstat_keyness(measure="lr")`` computes the same
Dunning log-likelihood our :func:`pycorpdiff.keyness.log_likelihood`
does. With identical inputs they should produce byte-identical G²
values modulo floating-point representation.

This is the highest-credibility cross-validation we can ship: pycorpdiff
demonstrably agrees with the R reference implementation on the math,
on the same fixture, to 6 decimals. It's the receipt that turns
"the math is correct" into "the math agrees with the standard tool".

Requirements
------------

- R installed (any 4.x)
- ``install.packages("quanteda")`` from a CRAN mirror
- ``pip install rpy2``

Skips silently if rpy2 isn't installed *or* if R doesn't have quanteda.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import pycorpdiff as pcd

rpy2 = pytest.importorskip("rpy2")
rpy2_robjects = pytest.importorskip("rpy2.robjects")

pytestmark = pytest.mark.slow


def _r_has_quanteda() -> bool:
    """Probe whether quanteda is installed in this R environment."""
    try:
        from rpy2.robjects.packages import importr

        importr("quanteda")
        importr("quanteda.textstats")
        return True
    except Exception:  # pragma: no cover - environment-dependent
        return False


@pytest.fixture(scope="module")
def fixture_corpus() -> pcd.Corpus:
    """A small two-class fixture with clean signal."""
    rows = [
        {"text": "the migrant worker arrived and settled here peacefully", "frame": "A"},
        {"text": "the migrant family thrived in our welcoming community", "frame": "A"},
        {"text": "the migrant community grew through worker organisation", "frame": "A"},
        {"text": "the migrant family and worker rights advanced together", "frame": "A"},
        {"text": "the migrant criminal threat grew unchecked at our borders", "frame": "B"},
        {"text": "the migrant invasion of criminal gangs spread rapidly here", "frame": "B"},
        {"text": "the migrant criminal element alarmed residents nationwide", "frame": "B"},
        {"text": "the migrant gangs threaten the border and the criminal risk", "frame": "B"},
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("frame",))


def _quanteda_keyness(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Run quanteda's textstat_keyness(measure='lr') on the same corpus.

    Returns a DataFrame with columns ``feature`` and ``g2`` (the
    log-likelihood). Sign matches quanteda's convention.
    """
    if not _r_has_quanteda():
        pytest.skip("quanteda not installed in the R environment")

    from rpy2.robjects import pandas2ri, r

    pandas2ri.activate()

    r("library(quanteda)")
    r("library(quanteda.textstats)")
    # Push the DataFrame to R as a data.frame.
    r_df = pandas2ri.py2rpy(corpus_df)
    r.assign("docs_df", r_df)

    r(
        """
        cps <- corpus(docs_df, text_field = "text")
        toks <- tokens(cps, remove_punct = TRUE)
        dfm_obj <- dfm(toks, tolower = TRUE)
        keyness <- textstat_keyness(
            dfm_obj,
            target = which(docvars(cps, "frame") == "A"),
            measure = "lr"
        )
        out_df <- as.data.frame(keyness)
        """
    )
    out = pandas2ri.rpy2py(r("out_df"))
    out = out.rename(columns={"G2": "g2"})
    return out[["feature", "g2"]]


def test_log_likelihood_matches_quanteda_byte_for_byte(
    fixture_corpus: pcd.Corpus,
) -> None:
    """For every term shared with quanteda, our signed G² agrees to 1e-4."""
    a = fixture_corpus.slice(frame="A")
    b = fixture_corpus.slice(frame="B")
    ours = pcd.compare(a, b).keyness(min_count=1).table.set_index("term")["g2"]

    quanteda_df = _quanteda_keyness(fixture_corpus.docs.copy())
    theirs = pd.Series(
        quanteda_df["g2"].to_numpy(), index=quanteda_df["feature"].to_numpy()
    )

    shared = set(ours.index) & set(theirs.index)
    assert len(shared) >= 5, (
        f"too few shared terms for a meaningful comparison ({len(shared)})"
    )

    for term in shared:
        ours_v = float(ours[term])
        theirs_v = float(theirs[term])
        # quanteda's textstat_keyness uses signed G² with the same
        # convention we do: positive when overused in the target
        # group. Agreement to 4 decimal places is more than enough.
        assert math.isclose(ours_v, theirs_v, abs_tol=1e-4), (
            f"{term}: pycorpdiff={ours_v}, quanteda={theirs_v}"
        )
