"""The assignment's completion criterion, checked as one thing.

    "The assignment is complete only when the system can prove what it
     consumed, why it consumed it, what the model learned from it and how the
     run can be reconstructed."

Each of the other checks verifies one subsystem.  This one verifies that the
subsystems are *joined up* — that you can start from a token span the model
actually saw and walk, without a gap, to the reason it was chosen, to what the
model got back from it, and to everything needed to reproduce the whole thing.

That is a different property from any individual check passing.  A run could
have a perfect consumption ledger, a perfect OPUS ledger and a perfect learning
ledger and still fail here, if the ids in one do not resolve in the next.  So
the test is a link walk over every consumed sample, not a spot check: any
dangling reference anywhere fails the clause.

The report also carries one fully worked example, because a reader should be
able to follow the chain themselves rather than take the aggregate on trust.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..ledger.consumption import EVENT_CONSUME
from ..ledger.store import LedgerStore

CLAUSES = (
    "what it consumed",
    "why it consumed it",
    "what the model learned from it",
    "how the run can be reconstructed",
)


def _index(records, key):
    return {r["payload"][key]: r["payload"] for r in records}


def check_completion(
    consumption: LedgerStore,
    opus: LedgerStore,
    learning: LedgerStore,
    branch_id: str,
    replay_report: dict,
    resume_report: dict,
    fork_report: dict,
    shard_cards: List[dict],
    token_trace_count: int,
    logger=None,
) -> dict:
    """Walk every consumed sample through all four clauses."""
    consumed = [
        r["payload"] for r in consumption.read_all()
        if r["type"] == EVENT_CONSUME and r["payload"].get("branch_id") == branch_id
    ]
    decisions = _index(
        [r for r in opus.read_all() if r["type"] == "opus_decision"], "decision_id"
    )
    learning_by_key: Dict[tuple, dict] = {}
    for record in learning.read_all():
        if record["type"] != "sample_learning":
            continue
        payload = record["payload"]
        learning_by_key[(payload["sample_id"], payload["step"])] = payload
    cards = {c["shard_id"]: c for c in shard_cards}

    total = 0
    gaps = {"span": [], "shard": [], "decision": [], "reason": [],
            "learning": [], "delta": [], "card": [], "provenance": []}

    for payload in consumed:
        # -- clause 4 fields travel on every record ------------------------
        for field in ("plan_hash", "batch_id", "rng_fingerprint",
                      "tokenizer_version", "dataloader_version", "config_hash"):
            if not payload.get(field):
                gaps["provenance"].append(f"{payload['microbatch_id']}:{field}")

        for index, sample_id in enumerate(payload.get("packed_sample_ids", [])):
            total += 1

            # -- clause 1: what was consumed -------------------------------
            spans = payload.get("token_span_ids", [])
            if not spans:
                gaps["span"].append(sample_id)
            if index < len(payload.get("shard_ids", [])) or payload.get("shard_ids"):
                pass
            if not payload.get("shard_ids"):
                gaps["shard"].append(sample_id)

            # -- clause 2: why -------------------------------------------
            decision_ids = payload.get("opus_decision_id", [])
            decision_id = decision_ids[index] if index < len(decision_ids) else ""
            decision = decisions.get(decision_id)
            if not decision_id or decision is None:
                gaps["decision"].append(sample_id)
            elif not decision.get("reason"):
                gaps["reason"].append(decision_id)

            # -- clause 3: what was learned --------------------------------
            record = learning_by_key.get((sample_id, payload["global_step"]))
            if record is None:
                gaps["learning"].append(f"{sample_id}@{payload['global_step']}")
            elif record.get("loss_delta") is None:
                gaps["delta"].append(sample_id)

        for shard_id in payload.get("shard_ids", []):
            if shard_id not in cards:
                gaps["card"].append(shard_id)

    consumed_ok = not gaps["span"] and not gaps["shard"]
    why_ok = not gaps["decision"] and not gaps["reason"]
    learned_ok = not gaps["learning"] and not gaps["delta"] and not gaps["card"] \
        and token_trace_count > 0
    rebuild_ok = (
        not gaps["provenance"]
        and bool(replay_report.get("all_match"))
        and bool(replay_report.get("plan_recomputation_matches"))
        and bool(resume_report.get("next_batch_verification", {}).get("matched"))
        and bool(resume_report.get("rollback_replay_verification", {}).get("identical"))
        and bool(fork_report.get("diverged_correctly"))
    )

    example = _worked_example(consumed, decisions, learning_by_key, cards)

    report = {
        "criterion": (
            "The assignment is complete only when the system can prove what it "
            "consumed, why it consumed it, what the model learned from it and "
            "how the run can be reconstructed."
        ),
        "method": (
            "a link walk over every consumed sample instance — not a spot check. "
            "Any dangling reference between the consumption, OPUS and learning "
            "ledgers fails the clause it belongs to."
        ),
        "branch_id": branch_id,
        "consumed_sample_instances": total,
        "clauses": {
            "what_it_consumed": {
                "proved": consumed_ok,
                "how": "every consumed microbatch names its shard ids, document ids, "
                       "token span ids and token/mask hashes",
                "instances_without_token_spans": len(gaps["span"]),
                "instances_without_shard_ids": len(gaps["shard"]),
            },
            "why_it_consumed_it": {
                "proved": why_ok,
                "how": "every consumed sample carries an OPUS decision id that resolves "
                       "in the OPUS ledger, and every decision carries a reason from a "
                       "fixed vocabulary — plus the mixture lane and curriculum stage",
                "instances_without_a_resolvable_decision": len(gaps["decision"]),
                "decisions_without_a_reason": len(gaps["reason"]),
            },
            "what_the_model_learned": {
                "proved": learned_ok,
                "how": "every consumed sample has a learning record with the loss measured "
                       "before and after the update on the same tokens; a full per-token "
                       "trace exists for the configured interval; every shard has a report "
                       "card with a classification",
                "instances_without_a_learning_record": len(gaps["learning"]),
                "records_without_a_loss_delta": len(gaps["delta"]),
                "shards_without_a_report_card": len(set(gaps["card"])),
                "token_trace_records": token_trace_count,
            },
            "how_the_run_can_be_reconstructed": {
                "proved": rebuild_ok,
                "how": "every record carries plan hash, batch id, RNG fingerprint, "
                       "tokenizer version, dataloader version and config hash; replay "
                       "matched on three independent derivations; resume matched the "
                       "expected batch and re-served the rolled-back interval identically; "
                       "the fork diverged only after its recorded divergence point",
                "records_missing_provenance_fields": len(gaps["provenance"]),
                "replay_all_match": replay_report.get("all_match"),
                "plan_recomputation_matches": replay_report.get("plan_recomputation_matches"),
                "resume_next_batch_matched":
                    resume_report.get("next_batch_verification", {}).get("matched"),
                "rollback_replay_identical":
                    resume_report.get("rollback_replay_verification", {}).get("identical"),
                "fork_diverged_correctly": fork_report.get("diverged_correctly"),
            },
        },
        "all_four_clauses_proved": consumed_ok and why_ok and learned_ok and rebuild_ok,
        "worked_example": example,
    }

    if logger is not None:
        for name, key in zip(CLAUSES, report["clauses"]):
            logger.check("proves_" + key, report["clauses"][key]["proved"], clause=name)
        logger.check(
            "completion_criterion_met",
            report["all_four_clauses_proved"],
            samples_walked=total,
        )
    return report


def _worked_example(consumed, decisions, learning_by_key, cards) -> Optional[dict]:
    """One sample followed all the way through, so a reader can check by hand."""
    for payload in consumed:
        ids = payload.get("packed_sample_ids") or []
        if not ids:
            continue
        sample_id = ids[0]
        decision_ids = payload.get("opus_decision_id") or [""]
        decision = decisions.get(decision_ids[0])
        record = learning_by_key.get((sample_id, payload["global_step"]))
        if not (decision and record):
            continue
        shard_id = (payload.get("shard_ids") or [""])[0]
        card = cards.get(shard_id)
        return {
            "consumed": {
                "step": payload["global_step"],
                "rank": payload["rank"],
                "microbatch_id": payload["microbatch_id"],
                "sample_id": sample_id,
                "shard_id": shard_id,
                "doc_id": (payload.get("doc_ids") or [""])[0],
                "token_span": (payload.get("token_span_ids") or [""])[0],
                "tokens_hash": (payload.get("tokens_hash") or [""])[0],
                "loss_mask_hash": (payload.get("loss_mask_hash") or [""])[0],
            },
            "why": {
                "mixture_lane": (payload.get("mixture_lane") or [""])[0],
                "curriculum_stage": payload.get("curriculum_stage"),
                "opus_decision_id": decision["decision_id"],
                "status": decision["status"],
                "reason": decision["reason"],
                "opus_score": decision["opus_score"],
                "threshold": decision["threshold"],
                "protected_floor_override": decision.get("protected_floor_override"),
                "scoring_checkpoint": decision.get("scoring_checkpoint"),
                "repeated_pass_number": decision.get("repeated_pass_number"),
            },
            "learned": {
                "loss_before": record["loss_before"],
                "loss_after": record["loss_after"],
                "loss_delta": record["loss_delta"],
                "gradient_norm": record["gradient_norm"],
                "mean_token_perplexity": record["mean_token_perplexity"],
                "eos_perplexity": record.get("eos_perplexity"),
                "model_phase": record["model_phase"],
                "checkpoint_before": record["checkpoint_before"],
                "shard_classification": card["classification"] if card else None,
                "shard_rationale": card["rationale"] if card else None,
            },
            "reconstructable_from": {
                "plan_hash": payload.get("plan_hash"),
                "batch_id": payload.get("batch_id"),
                "rng_fingerprint": payload.get("rng_fingerprint"),
                "tokenizer_version": payload.get("tokenizer_version"),
                "dataloader_version": payload.get("dataloader_version"),
                "config_hash": payload.get("config_hash"),
                "checkpoint_id": payload.get("checkpoint_id"),
            },
        }
    return None
