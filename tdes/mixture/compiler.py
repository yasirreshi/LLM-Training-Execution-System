"""Compile the curriculum into executable per-step quotas.

A mixture described in human terms has to become a number of sequences per lane per optimizer step, and it has to do so before
the run starts, so the plan can be checked against what is actually available.

The compiler answers four questions for every stage:

1. What share does each lane get, including the warmup ramp from the previous
   stage rather than a hard switch at the boundary.
2. Which floors are protected, and what does enforcing them cost the other lanes.
3. Can the lane be satisfied from the shards that exist.
4. If not, what is the resolution - repeat, synthesise, reduce the share, or
   move it to a later stage - recorded explicitly rather than decided implicitly
   by whatever the loader happened to have on hand.

Question 4 is the one that is easy to skip and expensive to skip.  A schedule
that quietly runs a lane dry produces a run whose real mixture diverges from
its stated mixture, and nothing reports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import CONFIG, LANES, PROTECTED_LANES, STAGES, Stage
from ..hashing import hash_obj
from .floors import apportion, blend, enforce_floors, shares

# Scarcity resolutions, in the order the compiler prefers them.
RESOLUTION_OK = "satisfied"
RESOLUTION_REPEAT = "repeat_existing_data"
RESOLUTION_REDUCE = "reduce_lane_share"
RESOLUTION_DEFER = "defer_to_later_stage"
RESOLUTION_SYNTHESISE = "generate_synthetic"
RESOLUTION_IMPOSSIBLE = "lane_empty_at_this_stage"


@dataclass
class StepQuota:
    step: int
    stage: str
    sequence_length: int
    reserved_unlocked: bool
    counts: Dict[str, int]
    planned_shares: Dict[str, float]
    warmup_t: float

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "stage": self.stage,
            "sequence_length": self.sequence_length,
            "reserved_unlocked": self.reserved_unlocked,
            "counts": dict(sorted(self.counts.items())),
            "planned_shares": {k: round(v, 5) for k, v in sorted(self.planned_shares.items())},
            "warmup_t": round(self.warmup_t, 4),
        }


@dataclass
class LaneFeasibility:
    stage: str
    lane: str
    sequences_required: int
    distinct_samples_available: int
    tokens_available: int
    resolution: str
    repeat_factor: float = 1.0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "lane": self.lane,
            "sequences_required": self.sequences_required,
            "distinct_samples_available": self.distinct_samples_available,
            "tokens_available": self.tokens_available,
            "resolution": self.resolution,
            "repeat_factor": round(self.repeat_factor, 3),
            "note": self.note,
        }


@dataclass
class MixtureSchedule:
    steps: List[StepQuota]
    feasibility: List[LaneFeasibility]
    floor_adjustments: List[dict]
    shortfalls: List[dict]
    anneal_reserve: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def quota(self, step: int) -> StepQuota:
        return self.steps[step]

    def planned_lane_sequences(self) -> Dict[str, int]:
        out: Dict[str, int] = {lane: 0 for lane in LANES}
        for quota in self.steps:
            for lane, count in quota.counts.items():
                out[lane] = out.get(lane, 0) + count
        return out

    def planned_lane_tokens(self) -> Dict[str, int]:
        out: Dict[str, int] = {lane: 0 for lane in LANES}
        for quota in self.steps:
            for lane, count in quota.counts.items():
                out[lane] = out.get(lane, 0) + count * quota.sequence_length
        return out

    @property
    def schedule_hash(self) -> str:
        return hash_obj([q.as_dict() for q in self.steps])

    def stage_intent_vs_compiled(self) -> List[dict]:
        """Where the compiled quotas differ from the stage's stated mixture.

        Worth reporting rather than hiding.  A step serves a whole number of
        sequences, so at N sequences per step the finest share expressible is
        1/N - here 12.5%.  A stated 4% agentic share cannot be served as 4%; it
        is served as 0% or 12.5%, and the protected floor forces the latter.
        The compiled plan is therefore the honest statement of what the run will
        actually do, and mixture compliance is measured against it rather than
        against an intent the batch geometry cannot express.
        """
        out: List[dict] = []
        for stage, start, end in _stage_bounds():
            quotas = self.steps[start:end]
            if not quotas:
                continue
            served: Dict[str, int] = {lane: 0 for lane in LANES}
            for quota in quotas:
                for lane, count in quota.counts.items():
                    served[lane] += count
            total = sum(served.values()) or 1
            for lane in LANES:
                intended = stage.mixture.get(lane, 0.0)
                compiled = served[lane] / total
                out.append(
                    {
                        "stage": stage.name,
                        "lane": lane,
                        "intended_share": round(intended, 5),
                        "compiled_share": round(compiled, 5),
                        "delta": round(compiled - intended, 5),
                        "floor": stage.protected_floors.get(lane),
                        "limited_by": _granularity_reason(
                            intended, compiled, stage.protected_floors.get(lane),
                            CONFIG.sequences_per_step,
                        ),
                    }
                )
        return out

    def as_dict(self) -> dict:
        planned_tokens = self.planned_lane_tokens()
        total_tokens = sum(planned_tokens.values())
        return {
            "schedule_hash": self.schedule_hash,
            "total_steps": self.total_steps,
            "sequences_per_step": CONFIG.sequences_per_step,
            "stages": [
                {
                    "name": stage.name,
                    "steps": stage.steps,
                    "sequence_length": stage.sequence_length,
                    "mixture": stage.mixture,
                    "protected_floors": stage.protected_floors,
                    "warmup_steps": stage.warmup_steps,
                    "anneal": stage.anneal,
                }
                for stage in STAGES
            ],
            "planned_lane_sequences": self.planned_lane_sequences(),
            "planned_lane_tokens": planned_tokens,
            "planned_lane_shares": {
                lane: round(count / total_tokens, 5) if total_tokens else 0.0
                for lane, count in sorted(planned_tokens.items())
            },
            "protected_lanes": list(PROTECTED_LANES),
            "quota_granularity": round(1.0 / CONFIG.sequences_per_step, 5),
            "stage_intent_vs_compiled": self.stage_intent_vs_compiled(),
            "anneal_reserve": self.anneal_reserve,
            "feasibility": [f.as_dict() for f in self.feasibility],
            "floor_adjustments": self.floor_adjustments,
            "shortfalls": self.shortfalls,
            "per_step": [q.as_dict() for q in self.steps],
        }


# --------------------------------------------------------------------------


def _granularity_reason(
    intended: float, compiled: float, floor: Optional[float], sequences_per_step: int
) -> str:
    """Name the constraint that moved a lane off its intended share."""
    step = 1.0 / sequences_per_step
    if abs(compiled - intended) < 1e-9:
        return ""
    if floor is not None and compiled + 1e-9 >= floor > intended - 1e-9 and compiled > intended:
        return "protected_floor"
    if intended > 0 and intended < step:
        return "below_quota_granularity"
    if abs(compiled - intended) <= step:
        return "integer_rounding"
    return "reallocated_from_empty_or_short_lanes"


def _stage_bounds() -> List[Tuple[Stage, int, int]]:
    bounds, cursor = [], 0
    for stage in STAGES:
        bounds.append((stage, cursor, cursor + stage.steps))
        cursor += stage.steps
    return bounds


def compile_schedule(availability, reserved_sample_ids=None) -> MixtureSchedule:
    """Build the per-step quota table.

    `availability(sequence_length, reserved_unlocked, lane) -> (windows, tokens)`
    reports what each lane could supply.  It is passed in rather than imported
    so the compiler can run against window *counts* - which are known before any
    token is materialised - and so it can be tested against a stub.
    """
    reserved_sample_ids = reserved_sample_ids or {}
    steps: List[StepQuota] = []
    feasibility: List[LaneFeasibility] = []
    floor_adjustments: List[dict] = []
    shortfalls: List[dict] = []

    bounds = _stage_bounds()

    for stage_index, (stage, start, end) in enumerate(bounds):
        previous = bounds[stage_index - 1][0].mixture if stage_index > 0 else stage.mixture
        reserved_unlocked = bool(stage.unlocks_reserved) or stage.anneal

        supply = {
            lane: availability(stage.sequence_length, reserved_unlocked, lane)
            for lane in LANES
        }
        availability_counts = {lane: supply[lane][0] for lane in LANES}
        tokens_available = {lane: supply[lane][1] for lane in LANES}

        stage_counts_total: Dict[str, int] = {lane: 0 for lane in LANES}

        for step in range(start, end):
            local = step - start
            t = 1.0 if stage.warmup_steps <= 0 else min(1.0, (local + 1) / stage.warmup_steps)
            effective = blend(previous, stage.mixture, t)

            # Lanes with nothing available at this stage get no share; their
            # weight goes to the lanes that can actually be served.
            usable = {
                lane: (weight if availability_counts.get(lane, 0) > 0 else 0.0)
                for lane, weight in effective.items()
            }

            counts = apportion(usable, CONFIG.sequences_per_step)
            counts, adjustments = enforce_floors(
                counts, stage.protected_floors, CONFIG.sequences_per_step,
                availability_counts,
            )
            for adjustment in adjustments:
                floor_adjustments.append({"step": step, "stage": stage.name, **adjustment})

            # Repetition is allowed within a step only when a lane has fewer
            # distinct samples than its quota; the planner handles the actual
            # wraparound, so no capping is applied here beyond "lane is empty".
            for lane in list(counts):
                if availability_counts.get(lane, 0) == 0:
                    counts[lane] = 0
            deficit = CONFIG.sequences_per_step - sum(counts.values())
            if deficit > 0:
                donors = sorted(
                    (l for l in counts if availability_counts.get(l, 0) > 0),
                    key=lambda l: (-usable.get(l, 0.0), l),
                )
                for i in range(deficit):
                    if not donors:
                        break
                    counts[donors[i % len(donors)]] += 1

            steps.append(
                StepQuota(
                    step=step,
                    stage=stage.name,
                    sequence_length=stage.sequence_length,
                    reserved_unlocked=reserved_unlocked,
                    counts=counts,
                    planned_shares=shares(counts),
                    warmup_t=t,
                )
            )
            for lane, count in counts.items():
                stage_counts_total[lane] += count

        # -- stage-level feasibility and scarcity resolution ---------------
        for lane in LANES:
            required = stage_counts_total[lane]
            have = availability_counts.get(lane, 0)
            if required == 0:
                resolution, factor, note = RESOLUTION_OK, 1.0, "lane not scheduled in this stage"
                if have == 0:
                    resolution = RESOLUTION_IMPOSSIBLE
                    note = "no packed samples exist for this lane at this window length"
            elif have == 0:
                resolution, factor = RESOLUTION_IMPOSSIBLE, 0.0
                note = "quota requested but the lane is empty; share reallocated"
            elif required <= have:
                resolution, factor, note = RESOLUTION_OK, 1.0, ""
            else:
                factor = required / have
                if lane in PROTECTED_LANES:
                    resolution = RESOLUTION_REPEAT
                    note = (
                        f"protected lane is short {required - have} sequences; "
                        f"repeating existing data {factor:.2f}x rather than "
                        f"dropping below the floor"
                    )
                elif factor <= 2.5:
                    resolution = RESOLUTION_REPEAT
                    note = f"repeating existing data {factor:.2f}x"
                else:
                    resolution = RESOLUTION_SYNTHESISE
                    note = (
                        f"would need {factor:.2f} passes over the same data; at that "
                        f"point synthetic generation or a schedule change is cheaper "
                        f"than further repetition"
                    )
            feasibility.append(
                LaneFeasibility(
                    stage=stage.name,
                    lane=lane,
                    sequences_required=required,
                    distinct_samples_available=have,
                    tokens_available=tokens_available.get(lane, 0),
                    resolution=resolution,
                    repeat_factor=factor,
                    note=note,
                )
            )

    return MixtureSchedule(
        steps=steps,
        feasibility=feasibility,
        floor_adjustments=floor_adjustments,
        shortfalls=shortfalls,
        anneal_reserve={k: sorted(v) for k, v in sorted(reserved_sample_ids.items())},
    )


# --------------------------------------------------------------------------
# Compliance: planned versus actual
# --------------------------------------------------------------------------


def compliance_report(
    schedule: MixtureSchedule,
    actual_lane_tokens: Dict[str, int],
    tolerance: float = None,
    floors_by_stage: Optional[Dict[str, Dict[str, float]]] = None,
    actual_by_stage: Optional[Dict[str, Dict[str, int]]] = None,
) -> dict:
    """Compare what was planned against what the ledger says was consumed.

    Two different questions, reported separately:

    *   Did the overall mixture land within tolerance of the plan?  A small
        drift is expected - quotas are integers and lanes run short.
    *   Was any protected floor ever breached?  That is not a tolerance
        question. A floor is a hard minimum and a breach is a failure.
    """
    tolerance = CONFIG.mixture_tolerance if tolerance is None else tolerance
    planned_tokens = schedule.planned_lane_tokens()
    planned_total = sum(planned_tokens.values()) or 1
    actual_total = sum(actual_lane_tokens.values()) or 1

    lanes: List[dict] = []
    worst = 0.0
    for lane in LANES:
        planned_share = planned_tokens.get(lane, 0) / planned_total
        actual_share = actual_lane_tokens.get(lane, 0) / actual_total
        delta = actual_share - planned_share
        worst = max(worst, abs(delta))
        lanes.append(
            {
                "lane": lane,
                "planned_tokens": planned_tokens.get(lane, 0),
                "actual_tokens": actual_lane_tokens.get(lane, 0),
                "planned_share": round(planned_share, 5),
                "actual_share": round(actual_share, 5),
                "delta": round(delta, 5),
                "within_tolerance": abs(delta) <= tolerance,
            }
        )

    floor_checks: List[dict] = []
    if floors_by_stage and actual_by_stage:
        for stage_name, floors in sorted(floors_by_stage.items()):
            stage_actual = actual_by_stage.get(stage_name, {})
            stage_total = sum(stage_actual.values()) or 1
            for lane, floor in sorted(floors.items()):
                achieved = stage_actual.get(lane, 0) / stage_total
                floor_checks.append(
                    {
                        "stage": stage_name,
                        "lane": lane,
                        "floor": floor,
                        "achieved_share": round(achieved, 5),
                        "respected": achieved + 1e-9 >= floor,
                    }
                )

    return {
        "tolerance": tolerance,
        "max_abs_delta": round(worst, 5),
        "all_lanes_within_tolerance": all(entry["within_tolerance"] for entry in lanes),
        "lanes": lanes,
        "protected_floor_checks": floor_checks,
        "all_floors_respected": all(check["respected"] for check in floor_checks)
        if floor_checks else None,
        "planned_total_tokens": planned_total,
        "actual_total_tokens": actual_total,
    }
