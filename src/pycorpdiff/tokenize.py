"""Tokenizer protocol and the default regex tokenizer.

The :class:`Tokenizer` protocol is the package's only extension point for
language-specific preprocessing. Adapters around spaCy, Stanza, jieba,
fugashi, etc. need to satisfy a single ``__call__(text: str) -> list[str]``
contract — no inheritance, no registration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Anything callable that maps a string to a list of token strings."""

    def __call__(self, text: str) -> list[str]: ...


@dataclass(frozen=True)
class RegexTokenizer:
    """A minimal Unicode-aware regex tokenizer used as the default.

    The default pattern matches sequences of word characters (``\\w+``),
    which under Python's default regex engine is Unicode-aware and
    therefore safe for non-Latin scripts at the granularity of "word-like
    runs of letters / digits / underscores". Researchers needing
    language-specific behaviour (lemmatisation, segmentation of CJK
    scripts, MWE handling, etc.) should plug in a spaCy/Stanza/jieba
    adapter instead.
    """

    pattern: str = r"\w+"
    lowercase: bool = True
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", re.compile(self.pattern, re.UNICODE))

    def __call__(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return self._compiled.findall(text)


@dataclass(frozen=True)
class NgramTokenizer:
    """A tokenizer wrapper that emits joined n-grams from a base tokenizer.

    Wraps any :class:`Tokenizer` and yields ``n``-token sequences joined
    by ``sep``. The output is a flat ``list[str]`` of joined n-grams,
    which means every downstream analytical surface — keyness,
    dispersion, collocation, semantic shift — treats them as ordinary
    terms with no special-casing needed.

    Parameters
    ----------
    base
        The underlying tokenizer producing unigrams. Defaults to
        :class:`RegexTokenizer`.
    n
        N-gram order. ``2`` = bigrams, ``3`` = trigrams. Must be ``>=1``.
    sep
        Joiner string. Default ``"_"`` matches gensim's convention and
        sidesteps the usual ambiguity of whitespace-joined n-grams when
        the base tokenizer itself strips whitespace.
    include_lower
        If ``True``, the output also includes every ``k``-gram for
        ``k < n`` (so ``n=3`` emits unigrams + bigrams + trigrams).
        Useful when you want a single Comparison to keyness-rank
        single words *and* their multi-word collocations side-by-side.

    Examples
    --------
    >>> from pycorpdiff.tokenize import NgramTokenizer
    >>> tok = NgramTokenizer(n=2)
    >>> tok("the cat sat on the mat")
    ['the_cat', 'cat_sat', 'sat_on', 'on_the', 'the_mat']
    >>> NgramTokenizer(n=2, include_lower=True)("a b c")
    ['a', 'b', 'c', 'a_b', 'b_c']
    """

    base: Tokenizer = field(default_factory=RegexTokenizer)
    n: int = 2
    sep: str = "_"
    include_lower: bool = False

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1; got {self.n}")

    def __call__(self, text: str) -> list[str]:
        unigrams = self.base(text)
        if self.n == 1:
            return unigrams
        out: list[str] = []
        start = 1 if self.include_lower else self.n
        for k in range(start, self.n + 1):
            if k == 1:
                out.extend(unigrams)
                continue
            sep = self.sep
            out.extend(
                sep.join(unigrams[i : i + k])
                for i in range(len(unigrams) - k + 1)
            )
        return out
