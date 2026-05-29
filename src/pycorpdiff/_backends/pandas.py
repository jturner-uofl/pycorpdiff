"""Pandas-backed internals for :class:`pycorpdiff.Corpus`.

Corpus operations route through this module so backend-specific code
stays out of the public API. The pandas backend is the default and is
exercised on every install; polars is opt-in via the ``polars`` extra
and lives in the sibling :mod:`pycorpdiff._backends.polars`.
"""

from __future__ import annotations
