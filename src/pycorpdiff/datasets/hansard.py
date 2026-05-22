"""UK Hansard sample loader.

The bundled sample is a deterministic synthetic corpus of 200 speeches
spanning 2005-2023, designed to mimic real Hansard's structure (four
topics, four parties, frame shifts at the right historical moments) so
tutorials and demos can run offline with byte-stable outputs.

For an actual research project you want the real archive:

- **TheyWorkForYou** publishes Hansard as XML / JSON at
  https://www.theyworkforyou.com/api/ (free, no auth).
- **Parliament's own data portal**: https://hansard.parliament.uk/
  (search UI) and https://hansard-api.parliament.uk/ (API).
- **HuggingFace datasets**: search for ``hansard`` — several pre-cleaned
  variants are published with permissive licences.

Real Hansard is in the public domain under the UK Open Government
Licence. Drop it into a DataFrame with the same columns the synthetic
sample uses (``text``, ``date``, ``party``, optionally ``topic``,
``frame``) and you can use it through every pycorpdiff verb.
"""

from __future__ import annotations

from pathlib import Path

from ..corpus import Corpus
from ..io.readers import read_parquet


def load_hansard_sample() -> Corpus:
    """Return the bundled 200-speech synthetic Hansard sample as a :class:`Corpus`.

    The corpus has columns ``speech_id``, ``text``, ``topic``,
    ``frame``, ``party``, ``date``, ``year``. Frames shift over time
    to mimic real discourse: immigration goes humanising → criminalising
    around 2016 (Brexit referendum), Brexit moves emerging → peak →
    aftermath, NHS has austerity (2010-14) and COVID (2020-22)
    pressure points, climate sharpens scientific → policy → crisis.

    Use this for tutorials, demos, and reproducible package tests. For
    actual research, fetch real Hansard from one of the sources noted in
    the module docstring.
    """
    data_path = Path(__file__).parent / "_data" / "hansard_sample.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Hansard sample not found at {data_path}. The package may have "
            "been installed without its bundled data; re-run "
            "`python -m pycorpdiff.datasets._generate_hansard` to regenerate."
        )
    return read_parquet(
        data_path,
        text_col="text",
        id_col="speech_id",
        meta_cols=("topic", "frame", "party", "date", "year"),
    )
