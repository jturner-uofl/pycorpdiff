"""Lightweight smoke check for the benchmark suite.

Runs one method per benchmark class at the smallest configured param,
printing wall-clock times. The point isn't to *measure* performance —
that's `asv run`'s job — but to verify the benchmarks themselves still
parse and execute after every change to the underlying API. CI runs
this on every push.
"""

from __future__ import annotations

import time

from benchmarks.benchmarks import (
    CollocationShift,
    CorpusConstruction,
    Keyness,
    TemporalTrack,
    Tokenization,
)

SPECS = [
    (CorpusConstruction, "time_vocab", 100),
    (Keyness, "time_keyness_default", 100),
    (CollocationShift, "time_collocation_shift", 500),
    (TemporalTrack, "time_track_over_time", 1_000),
    (Tokenization, "time_tokens", 1_000),
]


def main() -> None:
    print(f"{'benchmark':<55} {'wall (s)':>10}")
    print("-" * 66)
    for cls, method_name, param in SPECS:
        instance = cls()
        instance.setup(param)
        t0 = time.perf_counter()
        getattr(instance, method_name)(param)
        dt = time.perf_counter() - t0
        print(f"  {cls.__name__}.{method_name}(n={param}):  {dt:>8.4f}")


if __name__ == "__main__":
    main()
