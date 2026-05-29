"""Window-based co-occurrence extraction.

Given a tokenized corpus and a target term, walks each document and
accumulates the count of every other token that appears within ``window``
positions of the target. Windows never cross document boundaries — each
document's contexts are isolated, which matches the SketchEngine / NLTK
convention and avoids spurious cross-doc associations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def collocate_counts(
    docs_tokens: Sequence[Sequence[str]],
    target: str,
    window: int = 5,
) -> tuple[Counter[str], int]:
    """Return ``(collocate_counter, target_occurrences)`` for ``target``.

    Each occurrence of ``target`` contributes to up to ``2 * window``
    collocate counts (fewer near document boundaries). The target itself
    is excluded from its own context window — if the target appears
    twice within ``window`` of itself, each occurrence still contributes
    one count to the other.

    Parameters
    ----------
    docs_tokens
        One sequence of tokens per document, in original order.
    target
        The pivot term whose context we are accumulating.
    window
        Tokens on each side of the target. ``window=5`` gives a
        ten-token context (five left, five right).

    Returns
    -------
    counter
        Mapping ``collocate -> joint count``.
    target_occurrences
        Number of times the target appears across all documents. Used
        downstream as ``f_x`` (the marginal target count) for
        association measures.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")

    counter: Counter[str] = Counter()
    target_n = 0
    for tokens in docs_tokens:
        n = len(tokens)
        for i in range(n):
            if tokens[i] != target:
                continue
            target_n += 1
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                counter[tokens[j]] += 1
    return counter, target_n
