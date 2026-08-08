"""Throughput accounting.

The metric that matters is not tokens per second.  It is **useful loss-bearing
tokens per second at the target mixture**.  A loader can report a large raw
number while most of what it delivers is padding, context-only tokens, or
candidates OPUS is about to reject - none of which teach the model anything.

So four throughput figures are tracked, and the gaps between them are the
report:

    raw tokens/sec       every position moved, padding included
    useful tokens/sec    positions the loss mask actually grades
    accepted tokens/sec  useful tokens that survived OPUS
    effective tokens/sec useful tokens per second of *wall clock*, including
                         the time spent loading, packing and scoring

Every figure is a ratio of two recorded counters, so the evidence verifier can
recompute all of them from the ledgers.  A number that cannot be reconstructed
does not get credit, and rightly so.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Counters:
    raw_positions: int = 0
    useful_tokens: int = 0
    accepted_tokens: int = 0
    pad_tokens: int = 0
    context_only_tokens: int = 0
    rejected_candidate_tokens: int = 0
    sequences: int = 0
    microbatches: int = 0
    steps: int = 0
    shard_reads: int = 0
    cache_hits: int = 0


class PerfTracker:
    def __init__(self):
        self.counters = Counters()
        self.timers: Dict[str, float] = defaultdict(float)
        self.timer_calls: Dict[str, int] = defaultdict(int)
        self.rejections_by_lane: Dict[str, int] = defaultdict(int)
        self.candidates_by_lane: Dict[str, int] = defaultdict(int)
        self.events: List[dict] = []
        self._wall_start = time.perf_counter()

    # -- timing -----------------------------------------------------------

    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timers[name] += time.perf_counter() - start
            self.timer_calls[name] += 1

    def mark(self, name: str, seconds: float) -> None:
        self.timers[name] += seconds
        self.timer_calls[name] += 1

    @property
    def wall_seconds(self) -> float:
        return time.perf_counter() - self._wall_start

    # -- counting ---------------------------------------------------------

    def count_microbatch(self, samples) -> None:
        self.counters.microbatches += 1
        for sample in samples:
            self.counters.sequences += 1
            self.counters.raw_positions += sample.sequence_length
            self.counters.useful_tokens += sample.loss_bearing_count
            self.counters.accepted_tokens += sample.loss_bearing_count
            self.counters.pad_tokens += sample.pad_count
            self.counters.context_only_tokens += sample.context_only_count

    def count_rejection(self, lane: str, tokens: int) -> None:
        self.rejections_by_lane[lane] += 1
        self.counters.rejected_candidate_tokens += tokens

    def count_candidate(self, lane: str) -> None:
        self.candidates_by_lane[lane] += 1

    # -- reporting --------------------------------------------------------

    def report(self, extra: Dict[str, object] = None) -> dict:
        c = self.counters
        compute = self.timers.get("compute", 0.0) or 1e-9
        loader = self.timers.get("loader", 0.0)
        opus = self.timers.get("opus", 0.0)
        wall = self.wall_seconds or 1e-9

        total_candidates = sum(self.candidates_by_lane.values())
        total_rejected = sum(self.rejections_by_lane.values())

        report = {
            "counters": {
                "raw_positions": c.raw_positions,
                "useful_loss_bearing_tokens": c.useful_tokens,
                "accepted_tokens_after_opus": c.accepted_tokens,
                "pad_tokens": c.pad_tokens,
                "context_only_tokens": c.context_only_tokens,
                "rejected_candidate_tokens": c.rejected_candidate_tokens,
                "sequences": c.sequences,
                "microbatches": c.microbatches,
                "steps": c.steps,
                "shard_reads": c.shard_reads,
                "cache_hits": c.cache_hits,
            },
            "timings_seconds": {k: round(v, 6) for k, v in sorted(self.timers.items())},
            "timer_calls": dict(sorted(self.timer_calls.items())),
            "wall_seconds": round(wall, 6),
            "throughput": {
                "raw_tokens_per_sec_compute": round(c.raw_positions / compute, 2),
                "useful_tokens_per_sec_compute": round(c.useful_tokens / compute, 2),
                "accepted_tokens_per_sec_compute": round(c.accepted_tokens / compute, 2),
                "useful_tokens_per_sec_wall": round(c.useful_tokens / wall, 2),
                "sequences_per_sec_wall": round(c.sequences / wall, 4),
            },
            "efficiency": {
                # utilisation: how much of the window held a real token
                "packing_utilisation": _ratio(
                    c.raw_positions - c.pad_tokens, c.raw_positions
                ),
                # loss density: how much of the window was actually graded.
                # Always <= utilisation, and the gap is the context-only cost.
                "loss_density": _ratio(c.useful_tokens, c.raw_positions),
                "padding_waste": _ratio(c.pad_tokens, c.raw_positions),
                "context_only_share": _ratio(c.context_only_tokens, c.raw_positions),
                "loader_wait_share_of_wall": _ratio(loader, wall),
                "opus_share_of_wall": _ratio(opus, wall),
                "compute_share_of_wall": _ratio(compute, wall),
                "idle_share_of_wall": _ratio(
                    max(0.0, wall - compute - loader - opus), wall
                ),
                "cache_hit_rate": _ratio(c.cache_hits, c.cache_hits + c.shard_reads),
            },
            "opus": {
                "candidates_scored": total_candidates,
                "candidates_rejected": total_rejected,
                "rejection_rate": _ratio(total_rejected, total_candidates),
                "rejections_by_lane": dict(sorted(self.rejections_by_lane.items())),
                "candidates_by_lane": dict(sorted(self.candidates_by_lane.items())),
            },
            "how_to_reconstruct": {
                "packing_utilisation": "(raw_positions - pad_tokens) / raw_positions",
                "loss_density": "useful_loss_bearing_tokens / raw_positions",
                "useful_tokens_per_sec_compute": "useful_loss_bearing_tokens / timings.compute",
                "note": (
                    "raw_positions, pad_tokens and useful_loss_bearing_tokens are also "
                    "recoverable by summing total_positions, pad_tokens and "
                    "loss_bearing_tokens over the consumption ledger, which is what "
                    "verify_evidence.py does rather than trusting this file."
                ),
            },
        }
        if extra:
            report.update(extra)
        return report


def _ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return round(numerator / denominator, 6)
