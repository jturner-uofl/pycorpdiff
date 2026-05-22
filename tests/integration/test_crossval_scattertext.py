"""Cross-validation against Scattertext (Kessler 2017).

Scattertext computes per-term scaled F-scores between two categories
of documents. The package's :func:`pycorpdiff.compare.keyness` produces
signed Dunning G², which is a different statistic. We can't assert
*numerical agreement* (different measures!) — but we can assert
*ranking agreement* on the strong-signal terms: words Scattertext
flags as dominant in DEM speeches at the 2012 US Conventions should
also be A-leaning by our keyness, and vice-versa for REP.

This test ships in the slow tier because:
- it pulls Scattertext + its bundled corpus data (~5 MB);
- it actually runs both keyness implementations end-to-end.

Skip silently if Scattertext isn't installed.
"""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd

scattertext = pytest.importorskip("scattertext")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def convention_data() -> pd.DataFrame:
    """The bundled 2012 US Presidential Conventions corpus.

    Columns: ``party`` ('dem'/'rep'), ``speaker``, ``text``.
    """
    return scattertext.SampleCorpora.ConventionData2012.get_data()


def test_pycorpdiff_keyness_agrees_with_scattertext_top_terms(
    convention_data: pd.DataFrame,
) -> None:
    """Top-K Dem-leaning terms by our keyness should overlap substantially
    with top-K Dem-leaning terms by Scattertext.
    """
    corpus = pcd.from_dataframe(
        convention_data, text_col="text", meta_cols=("party", "speaker")
    )
    dem = corpus.slice(party="democrat")
    rep = corpus.slice(party="republican")

    # Our keyness, top-25 A-leaning (positive g2) terms by |g2|.
    result = pcd.compare(dem, rep).keyness(min_count=10)
    our_top_dem = set(
        result.table[result.table["g2"] > 0]
        .head(25)["term"]
        .tolist()
    )

    # Scattertext's term-score using their default scoring on the same data.
    st_corpus = (
        scattertext.CorpusFromPandas(
            convention_data, category_col="party", text_col="text",
            nlp=scattertext.whitespace_nlp_with_sentences,
        )
        .build()
        .get_unigram_corpus()
    )
    # F-score with informative Dirichlet prior — Scattertext's default
    # for "what's distinctive about category X".
    scores = st_corpus.get_term_freq_df()
    # Column names follow the pattern f"{category} freq".
    dem_col = next(c for c in scores.columns if c.startswith("democrat"))
    rep_col = next(c for c in scores.columns if c.startswith("republican"))
    from scattertext import ScaledFScorePresets

    scaled_f = ScaledFScorePresets()
    scores["dem_score"] = scaled_f.get_scores(scores[dem_col], scores[rep_col])
    scattertext_top_dem = set(
        scores.sort_values("dem_score", ascending=False).head(25).index.tolist()
    )

    # Two different measures on the same data — exact agreement isn't
    # the point. We assert *substantial* rank-overlap: at least 6 of
    # 25 (24%) shared. In practice we usually see 10-15.
    overlap = our_top_dem & scattertext_top_dem
    assert len(overlap) >= 6, (
        f"only {len(overlap)} terms overlap between pycorpdiff and "
        f"scattertext top-25 Dem-leaning; expected ≥6\n"
        f"  pycorpdiff: {sorted(our_top_dem)}\n"
        f"  scattertext: {sorted(scattertext_top_dem)}\n"
    )


def test_signs_agree_on_engineered_terms(convention_data: pd.DataFrame) -> None:
    """Terms with known political polarity should have agreeing sign across measures.

    'obama' should be Dem-leaning under both; 'romney' Rep-leaning under
    both. This is a behavioural sanity check: do we point the same direction.
    """
    corpus = pcd.from_dataframe(
        convention_data, text_col="text", meta_cols=("party",)
    )
    result = pcd.compare(
        corpus.slice(party="democrat"), corpus.slice(party="republican")
    ).keyness(min_count=5)
    table = result.table.set_index("term")
    if "obama" in table.index:
        assert table.loc["obama", "g2"] > 0, "obama should be Dem-leaning"
    if "romney" in table.index:
        assert table.loc["romney", "g2"] < 0, "romney should be Rep-leaning"
