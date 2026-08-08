"""Crash recovery: resume without skipping or repeating a batch.

The wreckage a crash leaves has a specific shape.  The checkpoint on disk is
from step C.  The ledger contains records up to some step K > C, because the
loader records a batch when it serves it, before the optimizer has learned from
it.  And the very last ledger line may be torn, because the process died
mid-write.

Two wrong answers are available and both look reasonable:

*   Resume at K.  The model never sees the data from steps C..K-1 - those
    updates died in memory.  Six steps of data are silently skipped and nothing
    reports it, because the weights and the ledger are each internally
    consistent; they just describe different histories.

*   Resume at C and append.  The model sees C..K-1 correctly, but the ledger now
    contains those steps twice, and any later audit reads it as a repeat.

The right answer is to resume at C *and roll the ledger back to the offset that
checkpoint recorded*.  Those records describe batches served to a model state
that no longer exists; they are not history worth keeping.  This is the reason
a checkpoint stores a byte offset rather than a step number - an offset names
the truncation point, which is the operation actually required.

The rollback discards data, so it does three things rather than one: it logs
the hashes of everything it drops, it re-serves those steps, and it then
*compares* the re-served hashes against the dropped ones.  That turns "we
believe nothing was lost" into a checkable claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..ledger.consumption import EVENT_CONSUME, EVENT_ROLLBACK
from ..ledger.store import LedgerOffset, LedgerStore
from ..training.checkpoint import CheckpointMeta, latest_checkpoint, load_checkpoint


@dataclass
class RecoveryState:
    checkpoint_path: Path
    meta: CheckpointMeta
    torn_tail: Optional[dict]
    discarded: List[dict] = field(default_factory=list)
    resume_step: int = 0

    @property
    def discarded_steps(self) -> List[int]:
        steps = {
            record["payload"]["global_step"]
            for record in self.discarded
            if record["type"] == EVENT_CONSUME
            and "global_step" in record["payload"]
        }
        return sorted(steps)

    def summary(self) -> dict:
        return {
            "checkpoint": self.meta.checkpoint_id,
            "checkpoint_step": self.meta.global_step,
            "resume_step": self.resume_step,
            "ledger_offset": self.meta.ledger_offset,
            "torn_tail_repaired": self.torn_tail is not None,
            "torn_tail": self.torn_tail,
            "discarded_records": len(self.discarded),
            "discarded_steps": self.discarded_steps,
            "discarded_event_hashes": [r["event_hash"] for r in self.discarded],
        }


def _fingerprint(record: dict) -> dict:
    """The identity of one consumption record, for before/after comparison.

    Everything here is derived from integers - sample ids, token spans, token
    and mask hashes.  No float appears, so the comparison is exact.
    """
    payload = record["payload"]
    return {
        "step": payload["global_step"],
        "rank": payload["rank"],
        "microbatch_id": payload["microbatch_id"],
        "batch_id": payload.get("batch_id", ""),
        "plan_hash": payload.get("plan_hash", ""),
        "packed_sample_ids": payload.get("packed_sample_ids", []),
        "token_span_ids": payload.get("token_span_ids", []),
        "tokens_hash": payload.get("tokens_hash", []),
        "loss_mask_hash": payload.get("loss_mask_hash", []),
    }


def prepare_resume(
    branch_id: str,
    consumption_store: LedgerStore,
    learning_store: LedgerStore,
    model=None,
    optimizer=None,
    scheduler=None,
    checkpoint_root: Path = None,
    logger=None,
) -> RecoveryState:
    """Repair, restore and roll back.  Returns everything needed to continue."""
    # 1. A torn final line is the signature of a hard kill mid-write.  Repair
    #    it first, because nothing else can parse the ledger until it is gone.
    torn = consumption_store.repair_torn_tail()
    learning_torn = learning_store.repair_torn_tail()
    if logger is not None:
        if torn:
            logger.event("ledger_torn_tail_repaired", **torn)
        if learning_torn:
            logger.event("ledger_torn_tail_repaired", **learning_torn)

    # 2. The newest *complete* checkpoint.  A directory missing state.pt is
    #    what an interrupted checkpoint looks like and must not be chosen.
    path = latest_checkpoint(branch_id, checkpoint_root)
    if path is None:
        raise RuntimeError(f"no complete checkpoint found for branch {branch_id}")
    meta = load_checkpoint(path, model, optimizer, scheduler)

    # 3. Roll the ledger back to the offset that checkpoint recorded.
    offset = LedgerOffset.from_dict(meta.ledger_offset)
    discarded = consumption_store.rollback_to(offset)
    learning_discarded = learning_store.rollback_to(
        LedgerOffset.from_dict(meta.learning_offset)
    )

    state = RecoveryState(
        checkpoint_path=path,
        meta=meta,
        torn_tail=torn,
        discarded=discarded,
        resume_step=meta.global_step,
    )

    # 4. Record the rollback itself.  Discarding ledger records is the one
    #    non-append operation in the system, so it leaves a trace naming
    #    exactly what it removed.
    consumption_store.append(
        EVENT_ROLLBACK,
        {
            "branch_id": branch_id,
            "reason": "resume_from_checkpoint",
            "checkpoint_id": meta.checkpoint_id,
            "rolled_back_to_offset": offset.as_dict(),
            "discarded_record_count": len(discarded),
            "discarded_learning_records": len(learning_discarded),
            "discarded_steps": state.discarded_steps,
            "discarded_event_hashes": [r["event_hash"] for r in discarded],
            "discarded_fingerprints": [
                _fingerprint(r) for r in discarded if r["type"] == EVENT_CONSUME
            ],
            "torn_tail_repaired": bool(torn),
        },
    )
    return state


def verify_next_batch(
    state: RecoveryState, planner, logger=None
) -> dict:
    """Is the batch we are about to serve the batch step N was supposed to get?

    Compared against `planner.plan()`, which is a pure function of the seed and
    the step, and against the plan hash the checkpoint itself recorded.  Two
    independent sources agreeing is the point; one would only be self-consistent.
    """
    step = state.resume_step
    recomputed = planner.plan_hash(step)
    recorded = state.meta.next_expected_plan_hash
    matched = bool(recorded) and recomputed == recorded

    result = {
        "resume_step": step,
        "checkpoint_id": state.meta.checkpoint_id,
        "expected_plan_hash_from_checkpoint": recorded,
        "recomputed_plan_hash": recomputed,
        "matched": matched,
    }
    if logger is not None:
        logger.check("resume_next_batch_matched", matched, **{
            "step": step,
            "expected": recorded[:16] if recorded else "",
            "recomputed": recomputed[:16],
        })
    return result


def verify_rollback_replay(
    state: RecoveryState,
    consumption_store: LedgerStore,
    branch_id: str,
    logger=None,
) -> dict:
    """Did re-serving the rolled-back steps reproduce them exactly?

    The strong form of "no skipped or repeated batches".  Every record the
    rollback discarded should reappear with an identical batch id, identical
    token spans and identical token and mask hashes.  Anything missing is a
    skip; anything extra or different is a repeat.
    """
    discarded = [
        _fingerprint(r) for r in state.discarded if r["type"] == EVENT_CONSUME
    ]
    if not discarded:
        return {
            "compared": 0,
            "identical": True,
            "note": "nothing was rolled back, so nothing to compare",
        }

    replayed_records = [
        r
        for r in consumption_store.read_all()
        if r["type"] == EVENT_CONSUME
        and r["payload"].get("branch_id") == branch_id
        and r["seq"] > state.meta.ledger_offset["event_seq"]
    ]
    replayed = {
        (f["step"], f["rank"], f["microbatch_id"]): f
        for f in (_fingerprint(r) for r in replayed_records)
    }

    mismatches: List[dict] = []
    missing: List[dict] = []
    for original in discarded:
        key = (original["step"], original["rank"], original["microbatch_id"])
        current = replayed.get(key)
        if current is None:
            missing.append(original)
            continue
        if current != original:
            mismatches.append(
                {
                    "key": list(key),
                    "differing_fields": sorted(
                        field for field in original
                        if original[field] != current.get(field)
                    ),
                    "original": original,
                    "replayed": current,
                }
            )

    identical = not mismatches and not missing
    result = {
        "compared": len(discarded),
        "reserved_records": len(replayed),
        "missing": missing,
        "mismatches": mismatches,
        "identical": identical,
        "discarded_steps": state.discarded_steps,
    }
    if logger is not None:
        logger.check(
            "resume_rollback_replay_identical",
            identical,
            compared=len(discarded),
            steps=state.discarded_steps,
            mismatches=len(mismatches),
            missing=len(missing),
        )
    return result
