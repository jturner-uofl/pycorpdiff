"""LLM annotation layer: the interpretive surface, kept honest by construction.

This module adds the package's *third* extension point, alongside
:class:`~pycorpdiff.Tokenizer` and :class:`~pycorpdiff.Embedder`: an
:class:`Annotator` --- anything callable that maps a ``str`` prompt to a ``str``
response (a local LLM, a hosted API, your own function). It is opt-in, pulls no
base dependency (the bundled :class:`OllamaAnnotator` uses only the standard
library), and a deterministic :class:`EchoAnnotator` makes the layer testable
and demoable offline.

The design enforces the project's *honest division of labour*:

- **Vectors and counts QUANTIFY.** Every number in the package comes from the
  statistical layer and is falsifiable.
- **The LLM only INTERPRETS.** An :class:`Annotator` is handed the package's own
  *measured, cited* exemplars (the most central texts per sense plus their
  distinctive terms) and returns a label and a one-line gloss. Its output is
  returned in a *separate* :class:`SenseNamingResult` and never flows back into a
  numeric field, a statistic, a flag, or a veracity verdict. **The LLM reads
  numbers; it never writes them.**

This invariant is structural, not aspirational: the source
:class:`~pycorpdiff.SenseDriftResult` is frozen, ``name_senses`` returns a new
object, and the annotation columns (``label``, ``gloss``) are held as string
dtype while every measured column stays integer/float. A unit test asserts that
no annotator output can reach a numeric column.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

from .results import _table_to_html, _table_to_json

if TYPE_CHECKING:
    from .semantic.drift import SenseDriftResult


@runtime_checkable
class Annotator(Protocol):
    """Anything callable that maps a prompt string to a response string.

    A local Ollama model (:class:`OllamaAnnotator`), a hosted API wrapped in a
    one-line function, or your own callable all satisfy this. An optional
    ``model_id`` attribute (str) is recorded in provenance; if absent, the class
    name is used.
    """

    def __call__(self, prompt: str) -> str: ...


@dataclass
class OllamaAnnotator:
    """Default :class:`Annotator` backed by a local `Ollama <https://ollama.com>`_
    server. Standard-library only (``urllib``) --- no added dependency.

    Parameters
    ----------
    model
        Ollama model tag (e.g. ``"llama3.2"``, ``"qwen2.5"``). Bring your own;
        the naming task is light, so a small instruct model is plenty.
    host
        Base URL of the Ollama server.
    temperature
        Sampling temperature; ``0.0`` (default) for reproducible labels.
    timeout
        Per-request timeout in seconds.
    options
        Extra Ollama ``options`` merged into the request.
    """

    model: str = "llama3.2"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout: float = 120.0
    options: dict[str, Any] = field(default_factory=dict)

    def __call__(self, prompt: str) -> str:
        import urllib.request

        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, **self.options},
        }
        req = urllib.request.Request(
            self.host.rstrip("/") + "/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("response", "")).strip()

    @property
    def model_id(self) -> str:
        return f"ollama:{self.model}"


@dataclass
class EchoAnnotator:
    """Deterministic, network-free :class:`Annotator` for tests and offline demos.

    Returns canned JSON whose label is a stable short hash of the prompt, so
    distinct senses receive distinct labels without any model call. It performs
    **no interpretation** --- use it to exercise the plumbing, not to read meaning.
    """

    gloss: str = "offline stub — no interpretation performed"

    def __call__(self, prompt: str) -> str:
        tag = "sense-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:6]
        return json.dumps({"label": tag, "gloss": self.gloss})

    @property
    def model_id(self) -> str:
        return "echo-stub"


@dataclass(frozen=True)
class SenseNamingResult:
    """Human-readable labels and glosses for fitted senses --- the interpretive
    companion to :class:`~pycorpdiff.SenseDriftResult`.

    ``table`` columns: ``sense`` (int), ``kind`` (``"reference"`` /
    ``"emergent"``), ``label`` and ``gloss`` (**the only LLM-authored fields**,
    string dtype), ``terms`` (measured distinctive terms), ``n`` (measured record
    count), ``n_cited`` (number of exemplars shown to the model). ``provenance``
    records the model id, per-sense prompt hashes, and call/cache counts so every
    label is auditable and reproducible.
    """

    table: pd.DataFrame
    provenance: dict[str, Any]

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_html(self.table, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_json(self.table, path, **kw)

    def summary(self) -> str:
        p = self.provenance
        head = (f"Named {len(self.table)} sense(s) via {p.get('model_id')} "
                f"({p.get('calls')} call(s), {p.get('cache_hits')} cached). "
                f"Labels are interpretation of cited exemplars, never measurements.")
        lines = [head]
        for _, r in self.table.iterrows():
            lines.append(f"  [{r['sense']}] {r['label']} — {r['gloss']}")
        return "\n".join(lines)


def _build_sense_prompt(kind_desc: str, terms: Sequence[str], examples: Sequence[str]) -> str:
    ex = "\n".join(f"- {str(e)[:200]}" for e in examples) or "(no exemplars available)"
    term_str = ", ".join(terms) if terms else "(none)"
    return (
        f"Below are representative text examples of {kind_desc} of a target term in a "
        f"corpus, together with the terms most distinctive of it.\n\n"
        f"Distinctive terms: {term_str}\n\n"
        f"Examples:\n{ex}\n\n"
        "Give a SHORT human-readable label (2-5 words) and a one-sentence gloss describing "
        "this usage. Base it ONLY on the examples above; do not invent facts, numbers, "
        "statistics, or claims, and do not judge whether anything is true. "
        'Return strict JSON only: {"label": "...", "gloss": "..."}'
    )


def _extract_label_gloss(raw: str) -> tuple[str, str]:
    """Parse the model's reply into (label, gloss); robust to extra prose."""
    try:
        snippet = raw[raw.find("{"): raw.rfind("}") + 1]
        obj = json.loads(snippet)
        label = str(obj.get("label", "")).strip()[:80]
        gloss = str(obj.get("gloss", "")).strip()[:240]
        return (label or "(unlabeled)", gloss)
    except Exception:
        first = raw.strip().splitlines()[0][:80] if raw.strip() else "(unlabeled)"
        return (first, "")


def _name_senses(
    result: SenseDriftResult,
    annotator: Annotator,
    *,
    n_examples: int,
    include_novel: bool,
    cache: dict[str, str] | None,
) -> SenseNamingResult:
    """Implementation of :meth:`SenseDriftResult.name_senses` (duck-typed on the
    result's ``_records`` / ``k`` / ``drift_terms`` / ``_cluster_terms``)."""
    recs = result._records
    model_id = str(getattr(annotator, "model_id", type(annotator).__name__))
    if cache is None:
        cache = {}
    calls = 0
    cache_hits = 0
    prov_senses: list[dict[str, Any]] = []

    def ask(prompt: str) -> tuple[str, str]:
        nonlocal calls, cache_hits
        key = hashlib.sha256((model_id + "\x00" + prompt).encode("utf-8")).hexdigest()
        if key in cache:
            cache_hits += 1
            return cache[key], key
        out = str(annotator(prompt))
        cache[key] = out
        calls += 1
        return out, key

    rows: list[dict[str, Any]] = []

    # --- reference senses: most-central (lowest-novelty) non-novel exemplars ---
    for c in range(result.k):
        mask = (recs["nearest_sense"] == c) & (~recs["novel"])
        central = recs[mask].sort_values("novelty")
        examples = central["text"].head(n_examples).tolist()
        terms = result._cluster_terms(c)
        out, key = ask(_build_sense_prompt("an existing usage sense", terms, examples))
        label, gloss = _extract_label_gloss(out)
        rows.append({"sense": c, "kind": "reference", "label": label, "gloss": gloss,
                     "terms": ", ".join(terms), "n": int(mask.sum()),
                     "n_cited": len(examples)})
        prov_senses.append({"sense": c, "prompt_sha256": key, "n_cited": len(examples)})

    # --- the emergent / novel bin: most-novel exemplars, distinctive terms ---
    if include_novel:
        nmask = recs["novel"].to_numpy()
        novel = recs[recs["novel"]].sort_values("novelty", ascending=False)
        examples = novel["text"].head(n_examples).tolist()
        if examples:
            terms = list(result.drift_terms)
            out, key = ask(_build_sense_prompt("a newly emerging usage", terms, examples))
            label, gloss = _extract_label_gloss(out)
            rows.append({"sense": result.k, "kind": "emergent", "label": label,
                         "gloss": gloss, "terms": ", ".join(terms[:8]),
                         "n": int(nmask.sum()), "n_cited": len(examples)})
            prov_senses.append({"sense": result.k, "prompt_sha256": key,
                                "n_cited": len(examples)})

    table = pd.DataFrame(rows, columns=["sense", "kind", "label", "gloss",
                                        "terms", "n", "n_cited"])
    # Invariant guard: LLM-authored columns are held as string dtype; measured
    # columns stay integer. No annotator output can land in a numeric field.
    table["label"] = table["label"].astype("string")
    table["gloss"] = table["gloss"].astype("string")
    table["kind"] = table["kind"].astype("string")
    table["terms"] = table["terms"].astype("string")
    table["sense"] = table["sense"].astype("int64")
    table["n"] = table["n"].astype("int64")
    table["n_cited"] = table["n_cited"].astype("int64")

    provenance = {
        "model_id": model_id,
        "n_examples": n_examples,
        "calls": calls,
        "cache_hits": cache_hits,
        "senses": prov_senses,
        "contract": "LLM names/glosses cited exemplars only; never writes a number "
                    "or a veracity verdict.",
    }
    return SenseNamingResult(table=table, provenance=provenance)
