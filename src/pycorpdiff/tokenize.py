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
