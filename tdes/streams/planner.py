"""The batch plan: a pure function from step number to what that step may consume.

This is the determinism spine.  Resume, replay, fork and audit all reduce to
questions about this function, so it is deliberately kept free of I/O, free of
global RNG and free of any dependence on how far the run has already got.

Two layers, and the distinction matters:

*   **The plan** - which candidate samples step N is offered, and how many of
    each lane it must accept.  Depends only on (seed, branch, step) and the
    packed sample index.  Fully recomputable by anyone, at any time, without a
    model.  `plan_hash` names it.

*   **The batch** - which candidates OPUS actually accepted, laid out into
    ranks and accumulation slots.  Depends on the model state as well, so it is
    reproducible only from a checkpoint.  `batch_id` names it.

Verification uses both.  Replay reads the batch from the ledger, reconstructs
it, and then recomputes the plan independently; agreement across all three is
what makes the replay claim mean something.  A plan that matched only itself
would prove nothing.

Structure follows Megatron's document/sample/shuffle index idea: build the
ordering once, hash it, and index into it - rather than drawing from a
generator whose state has to be checkpointed and restored correctly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import CONFIG, LANES
from ..hashing import derive_seed, hash_obj, short_hash
from ..mixture.compiler import MixtureSchedule

ContextKey = Tuple[int, bool]          # (sequence_length, reserved_unlocked)


@dataclass
class MicrobatchSpec:
    rank: int
    accum_index: int
    microbatch_id: str
    sample_ids: List[str]

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "accum_index": self.accum_index,
            "microbatch_id": self.microbatch_id,
            "sample_ids": list(self.sample_ids),
        }


@dataclass
class StepPlan:
    """Everything about step N that does not depend on the model."""

    step: int
    branch_id: str
    stage: str
    sequence_length: int
    reserved_unlocked: bool
    quotas: Dict[str, int]
    candidate_pools: Dict[str, List[str]]
    pass_numbers: Dict[str, int]           # sample_id -> which pass over its lane

    @property
    def plan_hash(self) -> str:
        return short_hash(
            {
                "step": self.step,
                "branch_id": self.branch_id,
                "stage": self.stage,
                "sequence_length": self.sequence_length,
                "quotas": dict(sorted(self.quotas.items())),
                "candidate_pools": {
                    lane: list(pool) for lane, pool in sorted(self.candidate_pools.items())
                },
            }
        )

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "branch_id": self.branch_id,
            "stage": self.stage,
            "sequence_length": self.sequence_length,
            "reserved_unlocked": self.reserved_unlocked,
            "quotas": dict(sorted(self.quotas.items())),
            "candidate_pools": {
                lane: list(pool) for lane, pool in sorted(self.candidate_pools.items())
            },
            "plan_hash": self.plan_hash,
        }


@dataclass
class BatchSpec:
    """Step N after selection: what will actually be trained on."""

    step: int
    branch_id: str
    stage: str
    sequence_length: int
    plan_hash: str
    microbatches: List[MicrobatchSpec]
    lane_of_sample: Dict[str, str] = field(default_factory=dict)
    opus_decision_ids: Dict[str, str] = field(default_factory=dict)

    @property
    def sample_ids(self) -> List[str]:
        return [sid for mb in self.microbatches for sid in mb.sample_ids]

    @property
    def batch_id(self) -> str:
        return "b-" + short_hash(
            {
                "step": self.step,
                "branch_id": self.branch_id,
                "plan_hash": self.plan_hash,
                "microbatches": [mb.as_dict() for mb in self.microbatches],
            }
        )[:12]

    def as_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "step": self.step,
            "branch_id": self.branch_id,
            "stage": self.stage,
            "sequence_length": self.sequence_length,
            "plan_hash": self.plan_hash,
            "microbatches": [mb.as_dict() for mb in self.microbatches],
            "sample_ids": self.sample_ids,
            "lane_of_sample": dict(sorted(self.lane_of_sample.items())),
            "opus_decision_ids": dict(sorted(self.opus_decision_ids.items())),
        }


class BatchPlanner:
    """Builds the shuffle index once, then answers plan(step) in O(1)."""

    def __init__(
        self,
        schedule: MixtureSchedule,
        sample_lookup,                     # (seq_len, reserved, lane) -> [PackedSample]
        branch_id: str,
        master_seed: int = None,
        candidate_multiplier: int = None,
    ):
        self.schedule = schedule
        self.branch_id = branch_id
        self.master_seed = CONFIG.master_seed if master_seed is None else master_seed
        self.candidate_multiplier = (
            CONFIG.opus_candidate_multiplier
            if candidate_multiplier is None
            else candidate_multiplier
        )

        # -- shuffle index: one deterministic ordering per (context, lane) ---
        self.streams: Dict[ContextKey, Dict[str, List[str]]] = {}
        contexts = {(q.sequence_length, q.reserved_unlocked) for q in schedule.steps}
        for context in sorted(contexts):
            length, reserved = context
            lanes: Dict[str, List[str]] = {}
            for lane in LANES:
                ids = sorted(s.sample_id for s in sample_lookup(length, reserved, lane))
                rng = random.Random(
                    derive_seed(self.master_seed, self.branch_id, "lane", lane,
                                length, int(reserved))
                )
                rng.shuffle(ids)
                lanes[lane] = ids
            self.streams[context] = lanes

        # -- cursors: prefix sums of pool draws, per context, per lane -------
        # Precomputed so plan(step) never has to walk the history.
        self.cursors: Dict[int, Dict[str, int]] = {}
        running: Dict[ContextKey, Dict[str, int]] = {
            context: {lane: 0 for lane in LANES} for context in contexts
        }
        for quota in schedule.steps:
            context = (quota.sequence_length, quota.reserved_unlocked)
            self.cursors[quota.step] = dict(running[context])
            for lane in LANES:
                running[context][lane] += self._pool_size(quota.counts.get(lane, 0))

    # -- internals --------------------------------------------------------

    def _pool_size(self, quota: int) -> int:
        """How many candidates OPUS is offered for a quota of `quota`."""
        if quota <= 0:
            return 0
        return quota * self.candidate_multiplier

    def _draw(self, context: ContextKey, lane: str, cursor: int, count: int
              ) -> Tuple[List[str], Dict[str, int]]:
        """Take `count` ids from a lane stream, wrapping around.

        Wrapping is how repetition happens, and the pass number is recorded per
        drawn sample so the learning ledger can ask whether a second exposure
        to the same window still reduced loss - the direct measurement of
        whether the repetition budget is exhausted.
        """
        stream = self.streams[context][lane]
        if not stream or count <= 0:
            return [], {}
        ids: List[str] = []
        passes: Dict[str, int] = {}
        for offset in range(count):
            index = cursor + offset
            sample_id = stream[index % len(stream)]
            ids.append(sample_id)
            passes[sample_id] = max(passes.get(sample_id, 0), index // len(stream))
        return ids, passes

    # -- the plan ---------------------------------------------------------

    def plan(self, step: int) -> StepPlan:
        if step < 0 or step >= len(self.schedule.steps):
            raise IndexError(f"step {step} outside the compiled schedule")
        quota = self.schedule.steps[step]
        context = (quota.sequence_length, quota.reserved_unlocked)
        cursor = self.cursors[step]

        pools: Dict[str, List[str]] = {}
        passes: Dict[str, int] = {}
        for lane in LANES:
            count = self._pool_size(quota.counts.get(lane, 0))
            ids, lane_passes = self._draw(context, lane, cursor[lane], count)
            if ids:
                pools[lane] = ids
                passes.update(lane_passes)

        return StepPlan(
            step=step,
            branch_id=self.branch_id,
            stage=quota.stage,
            sequence_length=quota.sequence_length,
            reserved_unlocked=quota.reserved_unlocked,
            quotas={lane: quota.counts.get(lane, 0) for lane in LANES},
            candidate_pools=pools,
            pass_numbers=passes,
        )

    def plan_hash(self, step: int) -> str:
        return self.plan(step).plan_hash

    @property
    def index_hash(self) -> str:
        """Hash of the whole shuffle index.

        Written as an artifact.  If two runs report the same index hash, they
        drew from the same ordering; if they differ, no other comparison
        between those runs is meaningful.
        """
        return hash_obj(
            {
                "branch_id": self.branch_id,
                "master_seed": self.master_seed,
                "candidate_multiplier": self.candidate_multiplier,
                "streams": {
                    f"len{length}{'+reserved' if reserved else ''}": {
                        lane: ids for lane, ids in sorted(lanes.items())
                    }
                    for (length, reserved), lanes in sorted(self.streams.items())
                },
            }
        )

    def as_dict(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "master_seed": self.master_seed,
            "candidate_multiplier": self.candidate_multiplier,
            "index_hash": self.index_hash,
            "schedule_hash": self.schedule.schedule_hash,
            "streams": {
                f"len{length}{'+reserved' if reserved else ''}": {
                    lane: ids for lane, ids in sorted(lanes.items())
                }
                for (length, reserved), lanes in sorted(self.streams.items())
            },
            "per_step_plan_hash": {
                str(quota.step): self.plan_hash(quota.step) for quota in self.schedule.steps
            },
        }


# --------------------------------------------------------------------------
# Laying accepted samples out into ranks and accumulation slots
# --------------------------------------------------------------------------


def assemble_batch(
    step_plan: StepPlan,
    accepted: Dict[str, List[str]],
    opus_decision_ids: Optional[Dict[str, str]] = None,
    world_size: int = None,
    microbatch_size: int = None,
    grad_accum: int = None,
) -> BatchSpec:
    """Distribute the accepted samples across (rank, accumulation step).

    Lanes are interleaved rather than blocked, so a rank does not spend a whole
    microbatch on one lane.  The order is a deterministic function of the lane
    names and the accepted lists, so the layout - and therefore the batch id -
    reproduces exactly on a resume.
    """
    world_size = CONFIG.world_size if world_size is None else world_size
    microbatch_size = CONFIG.microbatch_size if microbatch_size is None else microbatch_size
    grad_accum = CONFIG.grad_accum if grad_accum is None else grad_accum

    ordered: List[Tuple[str, str]] = []       # (sample_id, lane), round robin over lanes
    queues = {lane: list(ids) for lane, ids in sorted(accepted.items()) if ids}
    while queues:
        for lane in sorted(queues):
            ordered.append((queues[lane].pop(0), lane))
            if not queues[lane]:
                del queues[lane]

    capacity = world_size * microbatch_size * grad_accum
    if len(ordered) != capacity:
        raise ValueError(
            f"step {step_plan.step}: assembled {len(ordered)} sequences but the "
            f"batch geometry needs exactly {capacity}"
        )

    microbatches: List[MicrobatchSpec] = []
    cursor = 0
    for accum_index in range(grad_accum):
        for rank in range(world_size):
            slice_ids = [sid for sid, _lane in ordered[cursor:cursor + microbatch_size]]
            cursor += microbatch_size
            microbatches.append(
                MicrobatchSpec(
                    rank=rank,
                    accum_index=accum_index,
                    microbatch_id=(
                        f"s{step_plan.step:04d}-r{rank}-a{accum_index}"
                    ),
                    sample_ids=slice_ids,
                )
            )

    return BatchSpec(
        step=step_plan.step,
        branch_id=step_plan.branch_id,
        stage=step_plan.stage,
        sequence_length=step_plan.sequence_length,
        plan_hash=step_plan.plan_hash,
        microbatches=microbatches,
        lane_of_sample={sid: lane for sid, lane in ordered},
        opus_decision_ids=dict(opus_decision_ids or {}),
    )
