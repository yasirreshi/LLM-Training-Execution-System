"""Audit: reconstruct what trained a checkpoint, and what preceded a loss spike.

Two questions are the test of whether the whole apparatus was worth building:

    Which shards influenced the model between 5.4B and 5.6B tokens?
    Which OPUS-selected batches appeared before a loss spike?

Neither is answerable from a loss curve, a folder of shards, or a training log
that prints the mean loss every hundred steps.  Both are answerable here,
because the consumption ledger records the token spans behind every microbatch
and the OPUS ledger records why each of those batches was chosen over the one
beside it.

Everything below reads the ledger files.  Nothing is carried over in memory from
the run, so the audit works equally well on artifacts from a run that finished
months ago in another process.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

from ..ledger.consumption import EVENT_CONSUME, EVENT_STEP
from ..ledger.store import LedgerStore


def _consume_records(store: LedgerStore, branch_id: str) -> List[dict]:
    return [
        r for r in store.read_all()
        if r["type"] == EVENT_CONSUME and r["payload"].get("branch_id") == branch_id
    ]


def _step_records(store: LedgerStore, branch_id: str) -> List[dict]:
    return [
        r for r in store.read_all()
        if r["type"] == EVENT_STEP and r["payload"].get("branch_id") == branch_id
    ]


def shards_between_steps(
    store: LedgerStore, branch_id: str, start_step: int, end_step: int
) -> dict:
    """Which shards, documents and token spans trained the model over an interval."""
    by_shard: Dict[str, dict] = {}
    by_lane: Dict[str, int] = {}
    total_tokens = 0
    total_loss_tokens = 0

    for record in _consume_records(store, branch_id):
        payload = record["payload"]
        step = payload["global_step"]
        if not (start_step <= step < end_step):
            continue
        seq_len = payload.get("sequence_length", 0)
        total_tokens += payload.get("total_positions", 0)
        total_loss_tokens += payload.get("loss_bearing_tokens", 0)

        for lane in payload.get("mixture_lane", []):
            by_lane[lane] = by_lane.get(lane, 0) + seq_len

        for span in payload.get("token_span_ids", []):
            shard_id = span.split(":")[0]
            entry = by_shard.setdefault(
                shard_id,
                {"shard_id": shard_id, "spans": [], "steps": set(), "token_count": 0},
            )
            entry["spans"].append(span)
            entry["steps"].add(step)
            try:
                lo, hi = span.split(":")[1].split("-")
                entry["token_count"] += int(hi) - int(lo)
            except (IndexError, ValueError):
                pass

        for shard_id in payload.get("shard_ids", []):
            by_shard.setdefault(
                shard_id,
                {"shard_id": shard_id, "spans": [], "steps": set(), "token_count": 0},
            )["steps"].add(step)

    shards = [
        {
            "shard_id": entry["shard_id"],
            "distinct_spans": len(set(entry["spans"])),
            "total_span_tokens": entry["token_count"],
            "steps": sorted(entry["steps"]),
            "first_step": min(entry["steps"]) if entry["steps"] else None,
            "last_step": max(entry["steps"]) if entry["steps"] else None,
        }
        for entry in sorted(by_shard.values(), key=lambda e: e["shard_id"])
    ]

    return {
        "question": (
            f"which shards influenced the model between step {start_step} and "
            f"step {end_step}"
        ),
        "branch_id": branch_id,
        "interval_steps": [start_step, end_step],
        "shards_involved": len(shards),
        "total_positions": total_tokens,
        "loss_bearing_tokens": total_loss_tokens,
        "tokens_by_lane": dict(sorted(by_lane.items())),
        "shards": shards,
    }


def shards_between_token_counts(
    store: LedgerStore, branch_id: str, token_start: int, token_end: int
) -> dict:
    """The same question phrased in tokens consumed rather than steps.

    This is the form the question is actually asked - "between 5.4B and 5.6B tokens" - because at
    scale nobody remembers which step that was.
    """
    cumulative = 0
    steps_in_window: List[int] = []
    for record in sorted(
        _consume_records(store, branch_id),
        key=lambda r: (r["payload"]["global_step"], r["seq"]),
    ):
        payload = record["payload"]
        before = cumulative
        cumulative += payload.get("total_positions", 0)
        if before < token_end and cumulative > token_start:
            steps_in_window.append(payload["global_step"])

    if not steps_in_window:
        return {
            "question": f"which shards influenced the model between tokens "
                        f"{token_start} and {token_end}",
            "branch_id": branch_id,
            "token_window": [token_start, token_end],
            "steps_in_window": [],
            "shards_involved": 0,
            "shards": [],
        }

    result = shards_between_steps(
        store, branch_id, min(steps_in_window), max(steps_in_window) + 1
    )
    result["question"] = (
        f"which shards influenced the model between tokens {token_start} and {token_end}"
    )
    result["token_window"] = [token_start, token_end]
    result["steps_in_window"] = sorted(set(steps_in_window))
    return result


def detect_loss_spikes(
    store: LedgerStore, branch_id: str, sigma: float = 1.5
) -> List[dict]:
    """Steps whose loss rose sharply against the local trend.

    A spike is defined against the running mean and standard deviation of the
    step-to-step change, rather than against an absolute threshold, because the
    absolute loss falls throughout the run and any fixed number stops meaning
    the same thing after a few thousand steps.
    """
    records = sorted(_step_records(store, branch_id), key=lambda r: r["payload"]["global_step"])
    losses = [(r["payload"]["global_step"], r["payload"].get("mean_loss", 0.0)) for r in records]
    if len(losses) < 4:
        return []

    deltas = [losses[i][1] - losses[i - 1][1] for i in range(1, len(losses))]
    mean = statistics.fmean(deltas)
    spread = statistics.pstdev(deltas) or 1e-9

    spikes = []
    for index, delta in enumerate(deltas):
        if delta > mean + sigma * spread:
            step = losses[index + 1][0]
            spikes.append(
                {
                    "step": step,
                    "loss": round(losses[index + 1][1], 6),
                    "previous_loss": round(losses[index][1], 6),
                    "delta": round(delta, 6),
                    "z_score": round((delta - mean) / spread, 4),
                    "gradient_norm": records[index + 1]["payload"].get("gradient_norm"),
                }
            )
    return spikes


def batches_before_spike(
    consumption: LedgerStore,
    opus: LedgerStore,
    branch_id: str,
    spike_step: int,
    lookback: int = 2,
) -> dict:
    """What was fed to the model in the steps leading up to a spike."""
    low = max(0, spike_step - lookback)
    window = shards_between_steps(consumption, branch_id, low, spike_step + 1)

    decisions = [
        r["payload"]
        for r in opus.read_all()
        if r["type"] == "opus_decision"
        and r["payload"].get("branch_id") == branch_id
        and low <= r["payload"].get("step", -1) <= spike_step
    ]
    accepted = [d for d in decisions if d["status"] == "accepted"]

    return {
        "question": f"which OPUS-selected batches appeared before the loss spike at step {spike_step}",
        "spike_step": spike_step,
        "lookback_steps": [low, spike_step],
        "shards_in_window": [s["shard_id"] for s in window["shards"]],
        "tokens_by_lane": window["tokens_by_lane"],
        "opus_accepted_in_window": len(accepted),
        "opus_overrides_in_window": sum(
            1 for d in accepted if d.get("protected_floor_override")
        ),
        "accepted_decisions": sorted(
            (
                {
                    "decision_id": d["decision_id"],
                    "step": d["step"],
                    "lane": d["lane"],
                    "candidate_id": d["candidate_id"],
                    "opus_score": d["opus_score"],
                    "gradient_norm": d["gradient_norm"],
                    "reason": d["reason"],
                    "protected_floor_override": d.get("protected_floor_override", False),
                }
                for d in accepted
            ),
            key=lambda d: (d["step"], d["lane"], d["candidate_id"]),
        ),
        "highest_gradient_norm_candidate": max(
            accepted, key=lambda d: d.get("gradient_norm", 0.0), default=None
        ),
    }


def checkpoint_provenance(
    consumption: LedgerStore, branch_id: str, checkpoint_step: int
) -> dict:
    """Everything that trained the model up to a checkpoint."""
    result = shards_between_steps(consumption, branch_id, 0, checkpoint_step)
    result["question"] = (
        f"which data produced checkpoint ckpt_{branch_id}_{checkpoint_step:04d}"
    )
    result["checkpoint_step"] = checkpoint_step
    return result


def run_audit(
    consumption: LedgerStore,
    opus: LedgerStore,
    branch_id: str,
    checkpoint_step: int,
    interval: Tuple[int, int],
    logger=None,
) -> dict:
    """The full audit written to ledgers/audit_report.json."""
    spikes = detect_loss_spikes(consumption, branch_id)
    spike_reports = [
        batches_before_spike(consumption, opus, branch_id, spike["step"])
        for spike in spikes[:3]
    ]

    provenance = checkpoint_provenance(consumption, branch_id, checkpoint_step)
    interval_report = shards_between_steps(consumption, branch_id, *interval)

    # Phrase one query in tokens too, since that is how the question is asked
    # at scale.  Uses the middle third of what this run actually consumed.
    total_positions = provenance["total_positions"]
    token_query = shards_between_token_counts(
        consumption, branch_id, total_positions // 3, 2 * total_positions // 3
    )

    report = {
        "branch_id": branch_id,
        "queries": {
            "checkpoint_provenance": provenance,
            "interval_by_step": interval_report,
            "interval_by_token_count": token_query,
        },
        "loss_spikes_detected": spikes,
        "spike_investigations": spike_reports,
        "answerable": {
            "which_shards_trained_this_checkpoint": provenance["shards_involved"] > 0,
            "which_shards_in_a_token_window": token_query["shards_involved"] > 0,
            "which_batches_preceded_a_spike": bool(spike_reports)
            or not spikes,
        },
    }
    if logger is not None:
        logger.milestone(
            "audit completed",
            shards_traced=provenance["shards_involved"],
            spikes=len(spikes),
            interval=f"{interval[0]}-{interval[1]}",
        )
    return report
