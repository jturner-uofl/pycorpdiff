"""Explainability helpers — KWIC concordances, representative documents.

Every public analytical Result delegates its ``.explain()`` method here
so the concordance machinery lives in one place. KWIC lines are
returned as a tidy DataFrame on :class:`ConcordanceResult` with the
columns ``corpus``, ``doc_id``, ``position``, ``left``, ``keyword``,
``right``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .corpus import Corpus, CorpusSlice
from .results import ConcordanceResult


@dataclass(frozen=True)
class _KwicLine:
    corpus: str
    doc_id: int
    position: int
    left: str
    keyword: str
    right: str


def _kwic_lines_from_corpus(
    corpus: Corpus | CorpusSlice,
    target: str,
    label: str,
    window: int,
    collocate: str | None = None,
) -> list[_KwicLine]:
    """Extract KWIC lines for ``target`` from one corpus.

    When ``collocate`` is given, only windows that *also* contain
    ``collocate`` are kept — this is what powers the collocation
    explainer ("show me the contexts that drive this shift").
    """
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    docs = corpus.docs[corpus.text_col].tolist()
    tokenizer = corpus.tokenizer
    lines: list[_KwicLine] = []
    for doc_id, text in enumerate(docs):
        tokens = tokenizer(text)
        n_tokens = len(tokens)
        for pos in range(n_tokens):
            if tokens[pos] != target:
                continue
            lo = max(0, pos - window)
            hi = min(n_tokens, pos + window + 1)
            if collocate is not None:
                context = [tokens[j] for j in range(lo, hi) if j != pos]
                if collocate not in context:
                    continue
            left = " ".join(tokens[lo:pos])
            right = " ".join(tokens[pos + 1 : hi])
            lines.append(
                _KwicLine(
                    corpus=label,
                    doc_id=doc_id,
                    position=pos,
                    left=left,
                    keyword=target,
                    right=right,
                )
            )
    return lines


def _lines_to_concordance(
    lines: Sequence[_KwicLine], target: str, window: int, n: int | None
) -> ConcordanceResult:
    if not lines:
        empty = pd.DataFrame(
            columns=["corpus", "doc_id", "position", "left", "keyword", "right"]
        )
        return ConcordanceResult(target=target, table=empty, window=window)
    table = pd.DataFrame([line.__dict__ for line in lines])
    if n is not None and len(table) > n:
        table = table.head(n)
    return ConcordanceResult(
        target=target, table=table.reset_index(drop=True), window=window
    )


def kwic(
    corpus: Corpus | CorpusSlice,
    target: str,
    window: int = 5,
    n: int | None = None,
    label: str = "corpus",
) -> ConcordanceResult:
    """Return KWIC (keyword-in-context) concordance lines for ``target``.

    Walks each document, finds every occurrence of ``target``, and emits
    one row per occurrence with the ``window`` tokens of left context,
    the keyword itself, and the ``window`` tokens of right context.
    Document boundaries are respected — context never crosses them.

    Parameters
    ----------
    corpus
        Source corpus or slice.
    target
        Term to find. Compared against tokenized output, so case
        sensitivity follows the corpus's tokenizer.
    window
        Tokens of context on each side.
    n
        Cap on the number of lines returned (the first ``n``). Use
        ``None`` for "all matches".
    label
        Value to fill in the ``corpus`` column — useful when stitching
        KWIC tables from two corpora together for comparative explain.
    """
    lines = _kwic_lines_from_corpus(corpus, target, label=label, window=window)
    return _lines_to_concordance(lines, target=target, window=window, n=n)


def kwic_compare(
    a: Corpus | CorpusSlice,
    b: Corpus | CorpusSlice,
    target: str,
    window: int = 5,
    n_per_side: int = 5,
    collocate: str | None = None,
    label_a: str = "a",
    label_b: str = "b",
) -> ConcordanceResult:
    """Side-by-side KWIC lines for ``target`` from two corpora.

    Returns up to ``n_per_side`` lines from each corpus, concatenated
    with a ``corpus`` column distinguishing them. If ``collocate`` is
    given, only windows that also contain that collocate are kept —
    this is the engine behind
    :meth:`CollocationShiftResult.explain`.
    """
    lines_a = _kwic_lines_from_corpus(
        a, target, label=label_a, window=window, collocate=collocate
    )[:n_per_side]
    lines_b = _kwic_lines_from_corpus(
        b, target, label=label_b, window=window, collocate=collocate
    )[:n_per_side]
    return _lines_to_concordance(
        [*lines_a, *lines_b], target=target, window=window, n=None
    )


def representative_docs(
    corpus: Corpus | CorpusSlice,
    target: str,
    n: int = 5,
) -> pd.DataFrame:
    """Return up to ``n`` documents ranked by frequency of ``target``.

    Ties are broken by document index (earlier first). Documents without
    ``target`` are excluded.
    """
    tokenizer = corpus.tokenizer
    text_col = corpus.text_col
    rows: list[dict[str, object]] = []
    for doc_id, text in enumerate(corpus.docs[text_col].tolist()):
        count = tokenizer(text).count(target)
        if count > 0:
            rows.append({"doc_id": doc_id, "count": count, "text": text})
    if not rows:
        return pd.DataFrame(columns=["doc_id", "count", "text"])
    df = pd.DataFrame(rows).sort_values(
        ["count", "doc_id"], ascending=[False, True], kind="stable"
    )
    return df.head(n).reset_index(drop=True)
