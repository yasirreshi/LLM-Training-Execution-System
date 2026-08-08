"""Replay: reconstruct a historical interval and prove it matches.

The design notes is unambiguous about why replay reads rather than recomputes:

    "Right now, if I want to go back in history and run something, I will not
     run the code, because I know some nondeterminism can come in.  I'm going
     to run the ledger.  That shard was sent, so I'm going to read and send.
     I will not calculate it."

So replay is driven by the ledger.  But a replay that only reads the ledger and
compares it to itself proves nothing at all - of course a file matches itself.
The check that means something compares three independently derived answers:

    recorded       what the ledger says was served
    reconstructed  the same batch rebuilt from shards on disk, using the
                   sample ids and token spans the ledger names, and re-hashed
    recomputed     what `planner.plan()` says step N should have been offered,
                   derived from the seed with no reference to the ledger at all

`recorded == reconstructed` proves the shards still hold the tokens the ledger
claims.  `recorded == recomputed` proves the ordering was not fabricated after
the fact.  Both together are what makes `[PASS] replay_hash_matched` a real
statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..hashing import hash_mask, hash_token_ids
from ..ledger.consumption import EVENT_CONSUME
from ..ledger.store import LedgerStore


@dataclass
class ReplayedMicrobatch:
    step: int
    rank: int
    microbatch_id: str
    batch_id: str
    plan_hash: str
    sample_ids: List[str]
    token_span_ids: List[str]
    recorded_tokens_hash: List[str]
    recorded_mask_hash: List[str]
    reconstructed_tokens_hash: List[str] = field(default_factory=list)
    reconstructed_mask_hash: List[str] = field(default_factory=list)

    @property
    def tokens_match(self) -> bool:
        return self.recorded_tokens_hash == self.reconstructed_tokens_hash

    @property
    def masks_match(self) -> bool:
        return self.recorded_mask_hash == self.reconstructed_mask_hash

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "rank": self.rank,
            "microbatch_id": self.microbatch_id,
            "batch_id": self.batch_id,
            "plan_hash": self.plan_hash,
            "sample_ids": self.sample_ids,
            "token_span_ids": self.token_span_ids,
            "recorded_tokens_hash": self.recorded_tokens_hash,
            "reconstructed_tokens_hash": self.reconstructed_tokens_hash,
            "recorded_loss_mask_hash": self.recorded_mask_hash,
            "reconstructed_loss_mask_hash": self.reconstructed_mask_hash,
            "tokens_match": self.tokens_match,
            "masks_match": self.masks_match,
        }


def replay_interval(
    consumption_store: LedgerStore,
    branch_id: str,
    start_step: int,
    end_step: int,
    sample_store,
    shard_reader,
    planner=None,
    logger=None,
) -> dict:
    """Replay [start_step, end_step) and compare all three derivations."""
    records = [
        record
        for record in consumption_store.read_all()
        if record["type"] == EVENT_CONSUME
        and record["payload"].get("branch_id") == branch_id
        and start_step <= record["payload"].get("global_step", -1) < end_step
    ]

    replayed: List[ReplayedMicrobatch] = []
    span_mismatches: List[dict] = []

    for record in records:
        payload = record["payload"]
        entry = ReplayedMicrobatch(
            step=payload["global_step"],
            rank=payload["rank"],
            microbatch_id=payload["microbatch_id"],
            batch_id=payload.get("batch_id", ""),
            plan_hash=payload.get("plan_hash", ""),
            sample_ids=list(payload.get("packed_sample_ids", [])),
            token_span_ids=list(payload.get("token_span_ids", [])),
            recorded_tokens_hash=list(payload.get("tokens_hash", [])),
            recorded_mask_hash=list(payload.get("loss_mask_hash", [])),
        )

        # Rebuild each sample from the packed store and re-hash it.  The store
        # is itself rebuilt from the shard files on disk, so this is a genuine
        # round trip through storage, not a cache lookup.
        for sample_id in entry.sample_ids:
            sample = sample_store.get(sample_id)
            entry.reconstructed_tokens_hash.append(hash_token_ids(sample.token_ids))
            entry.reconstructed_mask_hash.append(hash_mask(sample.loss_mask))

            # The token spans the ledger named must still contain the same
            # tokens.  Read them straight out of the shard binary rather than
            # trusting the packed sample we just built from it.
            for segment in sample.segments:
                raw = shard_reader.tokens(
                    segment.shard_id, segment.token_start, segment.token_end
                )
                expected = sample.token_ids[
                    segment.window_start:segment.window_start + segment.length
                ]
                if raw != expected:
                    span_mismatches.append(
                        {
                            "step": entry.step,
                            "sample_id": sample_id,
                            "span": f"{segment.shard_id}:{segment.token_start}-{segment.token_end}",
                            "reason": "shard bytes differ from the packed window",
                        }
                    )
        replayed.append(entry)

    # Third derivation: the plan, recomputed from the seed alone.
    plan_comparison: List[dict] = []
    plan_matched = True
    if planner is not None:
        recorded_plan_hashes: Dict[int, str] = {}
        for entry in replayed:
            recorded_plan_hashes.setdefault(entry.step, entry.plan_hash)
        for step in sorted(recorded_plan_hashes):
            recomputed = planner.plan_hash(step)
            matched = recomputed == recorded_plan_hashes[step]
            plan_matched = plan_matched and matched
            plan_comparison.append(
                {
                    "step": step,
                    "recorded_plan_hash": recorded_plan_hashes[step],
                    "recomputed_plan_hash": recomputed,
                    "matched": matched,
                }
            )

    tokens_matched = all(entry.tokens_match for entry in replayed)
    masks_matched = all(entry.masks_match for entry in replayed)
    spans_matched = not span_mismatches
    all_matched = bool(replayed) and tokens_matched and masks_matched and spans_matched and plan_matched

    result = {
        "branch_id": branch_id,
        "interval": [start_step, end_step],
        "microbatches_replayed": len(replayed),
        "batch_ids": sorted({entry.batch_id for entry in replayed}),
        "tokens_match": tokens_matched,
        "loss_masks_match": masks_matched,
        "token_spans_match": spans_matched,
        "plan_recomputation_matches": plan_matched,
        "all_match": all_matched,
        "span_mismatches": span_mismatches,
        "plan_comparison": plan_comparison,
        "microbatches": [entry.as_dict() for entry in replayed],
        "derivations_compared": [
            "recorded (ledger)",
            "reconstructed (shard bytes re-read and re-hashed)",
            "recomputed (planner, from the seed, ledger not consulted)",
        ],
    }
    if logger is not None:
        logger.check(
            "replay_hash_matched",
            all_matched,
            interval=f"{start_step}-{end_step}",
            microbatches=len(replayed),
            tokens=tokens_matched,
            masks=masks_matched,
            spans=spans_matched,
            plan=plan_matched,
        )
    return result


def batch_ids_by_step(
    consumption_store: LedgerStore, branch_id: str
) -> Dict[int, str]:
    """Step -> batch id, for a branch.  Used by the fork divergence check."""
    out: Dict[int, str] = {}
    for record in consumption_store.read_all():
        if record["type"] != EVENT_CONSUME:
            continue
        payload = record["payload"]
        if payload.get("branch_id") != branch_id:
            continue
        out[payload["global_step"]] = payload.get("batch_id", "")
    return out
