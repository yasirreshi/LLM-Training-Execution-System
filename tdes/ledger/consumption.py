"""The training consumption ledger: what was served, and why.

The field list below is exhaustive, and all of it is recorded.  The point
of the list is that every one of them is needed to answer a question you cannot
answer later any other way:

    global_step + microbatch_id   which optimizer update did this feed
    packed_sample_ids             which windows, exactly
    shard_ids + token_span_ids    which tokens, exactly
    loss_mask_hash                which of them were graded
    mixture_lane + stage          why this batch was chosen
    opus_decision_id              why this batch and not the one beside it
    checkpoint_id + ledger offset where to restart from

Even when the planned order is generated from a seed and an index, the run still
needs an append-only record of the *actual* consumed stream - workers restart,
ranks retry, files go missing, and a plan is not a receipt.

The invariant this module exists to make checkable: exactly one record per
(branch, step, rank, microbatch), with global steps forming a contiguous range.
No skipped batch, no repeated batch.  `integrity_report` proves it from the file.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import ATTENTION_POLICY, CONFIG, LANE_POSITION_POLICY
from .store import LedgerStore

EVENT_CONSUME = "consume_microbatch"
EVENT_STEP = "optimizer_step"
EVENT_CHECKPOINT = "checkpoint_saved"
EVENT_ROLLBACK = "ledger_rollback"
EVENT_RESUME = "run_resumed"
EVENT_CRASH_ARMED = "crash_armed"
EVENT_FORK = "branch_forked"
EVENT_VALIDATION = "validation_evaluated"


class ConsumptionLedger:
    def __init__(
        self,
        store: LedgerStore,
        run_id: str,
        branch_id: str,
        tokenizer_hash: str,
        config_hash: str = "",
    ):
        self.store = store
        self.run_id = run_id
        self.branch_id = branch_id
        self.tokenizer_hash = tokenizer_hash
        self.config_hash = config_hash or CONFIG.config_hash

    # -- writing ----------------------------------------------------------

    def record_microbatch(
        self,
        *,
        global_step: int,
        checkpoint_id: str,
        rank: int,
        microbatch_id: str,
        batch_id: str,
        plan_hash: str,
        stage: str,
        sequence_length: int,
        samples: Sequence,
        lane_of_sample: Dict[str, str],
        opus_decision_ids: Dict[str, str],
        pass_numbers: Dict[str, int],
        rng_fingerprint: str,
    ) -> dict:
        payload = {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "global_step": global_step,
            "checkpoint_id": checkpoint_id,
            "rank": rank,
            "microbatch_id": microbatch_id,
            "batch_id": batch_id,
            "plan_hash": plan_hash,
            "curriculum_stage": stage,
            "sequence_length": sequence_length,
            "packed_sample_ids": [s.sample_id for s in samples],
            "shard_ids": sorted({sid for s in samples for sid in s.shard_ids}),
            "doc_ids": [d for s in samples for d in s.doc_ids],
            "token_span_ids": [span for s in samples for span in s.token_span_ids],
            "tokens_hash": [s.tokens_hash for s in samples],
            "loss_mask_hash": [s.loss_mask_hash for s in samples],
            "sample_content_hash": [s.content_hash() for s in samples],
            "mixture_lane": [lane_of_sample.get(s.sample_id, s.lane) for s in samples],
            "packing_policy": [s.policy for s in samples],
            "attention_policy": ATTENTION_POLICY,
            "position_policy": LANE_POSITION_POLICY.get(samples[0].lane, "segment_relative")
            if samples else "segment_relative",
            "loss_bearing_tokens": sum(s.loss_bearing_count for s in samples),
            "context_only_tokens": sum(s.context_only_count for s in samples),
            "pad_tokens": sum(s.pad_count for s in samples),
            "total_positions": sum(s.sequence_length for s in samples),
            "opus_decision_id": [
                opus_decision_ids.get(s.sample_id, "") for s in samples
            ],
            "repeated_pass_number": [pass_numbers.get(s.sample_id, 0) for s in samples],
            "tokenizer_version": self.tokenizer_hash,
            "dataloader_version": CONFIG.dataloader_version,
            "config_hash": self.config_hash,
            "rng_fingerprint": rng_fingerprint,
        }
        return self.store.append(EVENT_CONSUME, payload)

    def record_step(self, **payload: Any) -> dict:
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("branch_id", self.branch_id)
        return self.store.append(EVENT_STEP, payload)

    def record(self, event_type: str, payload: Dict[str, Any]) -> dict:
        payload = dict(payload)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("branch_id", self.branch_id)
        return self.store.append(event_type, payload)

    # -- reading ----------------------------------------------------------

    def consume_events(self, branch_id: Optional[str] = None) -> List[dict]:
        branch = branch_id or self.branch_id
        return [
            record
            for record in self.store.read_all()
            if record["type"] == EVENT_CONSUME
            and record["payload"].get("branch_id") == branch
        ]

    def events_for_step(self, step: int, branch_id: Optional[str] = None) -> List[dict]:
        return [
            record
            for record in self.consume_events(branch_id)
            if record["payload"]["global_step"] == step
        ]


# --------------------------------------------------------------------------
# Integrity: the "no skipped or repeated batches" proof
# --------------------------------------------------------------------------


def integrity_report(records: Iterable[dict], branch_id: str = "") -> dict:
    """Check the consumption record for gaps and duplicates.

    This is the check the assignment's resume criterion turns on, so it is
    computed from the ledger file rather than from anything the trainer kept in
    memory.
    """
    seen: Dict[tuple, List[int]] = {}
    steps: Dict[int, int] = {}
    lanes: Dict[str, int] = {}
    tokens_by_lane: Dict[str, int] = {}
    stage_lane_tokens: Dict[str, Dict[str, int]] = {}
    loss_tokens = 0
    total_positions = 0
    pad_tokens = 0

    for record in records:
        payload = record["payload"]
        if branch_id and payload.get("branch_id") != branch_id:
            continue
        key = (
            payload["branch_id"],
            payload["global_step"],
            payload["rank"],
            payload["microbatch_id"],
        )
        seen.setdefault(key, []).append(record["seq"])
        steps[payload["global_step"]] = steps.get(payload["global_step"], 0) + 1

        stage = payload.get("curriculum_stage", "")
        per_stage = stage_lane_tokens.setdefault(stage, {})
        for lane, sample_id in zip(
            payload.get("mixture_lane", []), payload.get("packed_sample_ids", [])
        ):
            lanes[lane] = lanes.get(lane, 0) + 1
        # token accounting is per sample, and sequence_length is uniform per step
        seq_len = payload.get("sequence_length", 0)
        for lane in payload.get("mixture_lane", []):
            tokens_by_lane[lane] = tokens_by_lane.get(lane, 0) + seq_len
            per_stage[lane] = per_stage.get(lane, 0) + seq_len

        loss_tokens += payload.get("loss_bearing_tokens", 0)
        pad_tokens += payload.get("pad_tokens", 0)
        total_positions += payload.get("total_positions", 0)

    duplicates = [
        {"key": list(key), "seqs": seqs} for key, seqs in sorted(seen.items()) if len(seqs) > 1
    ]

    observed = sorted(steps)
    gaps: List[int] = []
    if observed:
        expected = set(range(observed[0], observed[-1] + 1))
        gaps = sorted(expected - set(observed))

    per_step_counts = sorted({count for count in steps.values()})
    expected_per_step = CONFIG.world_size * CONFIG.grad_accum

    return {
        "branch_id": branch_id,
        "records": sum(len(v) for v in seen.values()),
        "distinct_microbatches": len(seen),
        "duplicate_microbatches": duplicates,
        "duplicate_count": len(duplicates),
        "steps_observed": observed,
        "step_range": [observed[0], observed[-1]] if observed else [],
        "missing_steps": gaps,
        "microbatches_per_step": per_step_counts,
        "expected_microbatches_per_step": expected_per_step,
        "every_step_complete": per_step_counts in ([], [expected_per_step]),
        "no_duplicates": not duplicates,
        "no_gaps": not gaps,
        "ok": (not duplicates) and (not gaps)
        and per_step_counts in ([], [expected_per_step]),
        "sequences_by_lane": dict(sorted(lanes.items())),
        "tokens_by_lane": dict(sorted(tokens_by_lane.items())),
        "tokens_by_stage_lane": {
            stage: dict(sorted(v.items())) for stage, v in sorted(stage_lane_tokens.items())
        },
        "loss_bearing_tokens": loss_tokens,
        "pad_tokens": pad_tokens,
        "total_positions": total_positions,
        "packing_utilisation": round(
            (total_positions - pad_tokens) / total_positions, 5
        ) if total_positions else 0.0,
        "loss_density": round(loss_tokens / total_positions, 5) if total_positions else 0.0,
    }
