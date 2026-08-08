"""Apportionment and protected floors.

Two small pieces of arithmetic that decide what the model actually sees.

**Apportionment.**  A mixture is a set of fractions; a step serves a whole
number of sequences.  Rounding each fraction independently loses or gains a
sequence depending on the remainders, and because the function runs once per
step the error accumulates into a visible drift away from the planned mixture.
Largest-remainder allocation always sums to exactly the target, and ties break
on the lane name so a replay reproduces the same split.

**Floors.**  A protected floor is not a preference, it is a hard minimum.  The
session's argument is about practice rather than capacity: a language the model
stops seeing for a long stretch is one it has effectively lost, so the indic and
agentic lanes get a guaranteed share of *every* step rather than a large share
of a few.  When a floor and the mixture disagree, the floor wins and the
overshoot is taken from the largest unprotected lane.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def apportion(weights: Dict[str, float], total: int) -> Dict[str, int]:
    """Split `total` whole units across weights, summing to exactly `total`."""
    if total <= 0:
        return {lane: 0 for lane in weights}
    mass = sum(max(0.0, w) for w in weights.values())
    if mass <= 0:
        return {lane: 0 for lane in weights}

    exact = {lane: total * max(0.0, w) / mass for lane, w in weights.items()}
    floors = {lane: int(value) for lane, value in exact.items()}
    leftover = total - sum(floors.values())

    order = sorted(exact, key=lambda lane: (-(exact[lane] - floors[lane]), lane))
    for lane in order[:leftover]:
        floors[lane] += 1
    return floors


def enforce_floors(
    counts: Dict[str, int],
    floors: Dict[str, float],
    total: int,
    available: Dict[str, int] = None,
) -> Tuple[Dict[str, int], List[dict]]:
    """Raise any lane below its floor, paying for it from the largest lane.

    Returns the adjusted counts and a list of adjustment records, so a floor
    that had to intervene is visible in the schedule rather than silent.

    `available` caps a lane at what actually exists; a floor cannot be met from
    an empty lane, and pretending otherwise would just produce empty batches.
    """
    counts = dict(counts)
    adjustments: List[dict] = []
    available = available or {}

    for lane in sorted(floors):
        floor_fraction = floors[lane]
        required = math.ceil(floor_fraction * total - 1e-9)
        cap = available.get(lane, required)
        required = min(required, cap)
        if required <= counts.get(lane, 0):
            continue

        deficit = required - counts.get(lane, 0)
        donors = sorted(
            (l for l in counts if l != lane and l not in floors and counts[l] > 0),
            key=lambda l: (-counts[l], l),
        )
        if not donors:
            # every other lane is itself protected - take from the largest one
            # that stays above its own floor after donating
            donors = sorted(
                (
                    l for l in counts
                    if l != lane
                    and counts[l] - 1 >= math.ceil(floors.get(l, 0.0) * total - 1e-9)
                ),
                key=lambda l: (-counts[l], l),
            )

        taken = 0
        for donor in donors:
            while taken < deficit and counts[donor] > 0:
                counts[donor] -= 1
                counts[lane] = counts.get(lane, 0) + 1
                taken += 1
            if taken >= deficit:
                break

        if taken:
            adjustments.append(
                {
                    "lane": lane,
                    "floor": floor_fraction,
                    "required_sequences": required,
                    "added": taken,
                    "donors": donors[: max(1, len(donors))][:3],
                }
            )
    return counts, adjustments


def cap_to_available(
    counts: Dict[str, int], available: Dict[str, int], total: int,
    floors: Dict[str, float] = None,
) -> Tuple[Dict[str, int], List[dict]]:
    """Never schedule more distinct sequences from a lane than it can supply.

    Overflow is redistributed to lanes that still have room.  A lane that is
    short is recorded, because that shortfall is the input to the scarcity
    decision - repeat, synthesise, reduce the share or move it to a later stage.
    """
    counts = dict(counts)
    floors = floors or {}
    shortfalls: List[dict] = []
    spare = 0

    for lane in sorted(counts):
        limit = available.get(lane, 0)
        if counts[lane] > limit:
            shortfalls.append(
                {"lane": lane, "wanted": counts[lane], "available": limit,
                 "shortfall": counts[lane] - limit}
            )
            spare += counts[lane] - limit
            counts[lane] = limit

    while spare > 0:
        candidates = sorted(
            (l for l in counts if counts[l] < available.get(l, 0)),
            key=lambda l: (-(available[l] - counts[l]), l),
        )
        if not candidates:
            break
        for lane in candidates:
            if spare == 0:
                break
            counts[lane] += 1
            spare -= 1

    return counts, shortfalls


def shares(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {lane: 0.0 for lane in counts}
    return {lane: counts[lane] / total for lane in counts}


def blend(a: Dict[str, float], b: Dict[str, float], t: float) -> Dict[str, float]:
    """Linear interpolation between two mixtures, used for stage warmup.

    A stage boundary is not a switch.  The design's point is that a bit of the
    next stage should start early and a bit of the previous stage should
    continue - so `t` walks from 0 to 1 across the warmup window instead of
    jumping.
    """
    lanes = sorted(set(a) | set(b))
    t = min(1.0, max(0.0, t))
    return {lane: (1.0 - t) * a.get(lane, 0.0) + t * b.get(lane, 0.0) for lane in lanes}
