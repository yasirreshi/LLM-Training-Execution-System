#!/usr/bin/env python3
"""Run the complete Training Data Execution System demonstration.

    python run_demo.py

One command, no manual intervention.  It regenerates `submission_artifacts/`
from scratch every time, so the artifacts in the repository are never the
authority - the run is.

Phases, in the order the log records them:

    0  build     corpus -> tokenizer -> shards -> manifests -> admission
    1  firewall  a real eval shard is pushed at the batch gate and blocked
    2  train     subprocess, crashes on purpose partway past a checkpoint
    3  resume    subprocess, repairs, rolls back, finishes the run
    4  replay    an earlier interval, compared three independent ways
    5  fork      subprocess, new branch from an earlier checkpoint
    6  audit     which shards trained what, and what preceded a loss spike
    7  reports   mixture compliance, OPUS, learning, next-corpus feedback, throughput
    8  evidence  generated from the artifacts, then independently re-verified
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Dict

from tdes.audit.auditor import run_audit
from tdes.audit.completion import check_completion
from tdes.audit.evidence import write_evidence
from tdes.config import CONFIG, PATHS, STAGES
from tdes.events import EventLogger
from tdes.firewall.eval_firewall import EvalFirewall
from tdes.fsutil import rmtree, write_json
from tdes.ledger.branch import divergence_report
from tdes.ledger.consumption import EVENT_CONSUME, integrity_report
from tdes.ledger.learning import LearningLedger
from tdes.ledger.store import LedgerStore
from tdes.mixture.compiler import compliance_report
from tdes.opus.selector import OpusDecision, decisions_report
from tdes.packing.masks import check_attention_confinement, validate_masks
from tdes.pipeline import (
    FINGERPRINT_FILE,
    build_data_system,
    finalise_data_system,
    write_manifest_artifacts,
)
from tdes.shards.builder import build_shards
from tdes.shards.registry import FirewallViolation
from tdes.streams.planner import BatchPlanner
from tdes.streams.replay import batch_ids_by_step, replay_interval
from tdes.training.checkpoint import list_checkpoints, prune_checkpoints
from tdes.training.crash import CRASH_EXIT_CODE

PRIMARY_BRANCH = "main"


def run_worker(phase: str, logger: EventLogger) -> int:
    """Launch the trainer as a child process so it can be hard-killed."""
    command = [sys.executable, "-m", "tdes.cli.train_worker", "--phase", phase]
    logger.event("worker_launch", phase=phase, command=" ".join(command[1:]))
    completed = subprocess.run(command, cwd=str(PATHS.root))
    logger.event("worker_exit", phase=phase, exit_code=completed.returncode)
    return completed.returncode


# --------------------------------------------------------------------------


def phase_firewall_demo(system, logger: EventLogger) -> dict:
    """Push real evaluation data at the batch gate and confirm it is refused.

    Asserting that a firewall would block something proves nothing.  This takes
    an actual never-train benchmark shard, packs it into a real training window,
    and hands it to the same `check_batch` the training loop calls before every
    gradient - then requires the exception.
    """
    firewall_store = LedgerStore(PATHS.ledgers / "firewall_events.jsonl", "firewall")
    firewall = EvalFirewall(system.registry, system.fingerprints, firewall_store)

    test_entries = system.registry.test
    if not test_entries:
        logger.check("eval_shard_blocked", False, note="no never-train shard registered")
        return firewall.report()

    # Side 1: the registry refuses it at planning time.
    registry_blocked = not firewall.check_admission(
        test_entries[0].shard_id, context="firewall_demo"
    )

    # Side 2: the batch gate refuses it on the decoded text, even though the
    # sample was constructed and handed over as if it were legitimate.
    injected = system.packer.pack_holdout(
        test_entries[:1], "test", min(s.sequence_length for s in STAGES)
    )
    batch_blocked = False
    block_reason = ""
    canary_seen = False
    if injected:
        sample = injected[0]
        decoded = sample.decoded(system.tokenizer)
        canary_seen = "TDES-CANARY-" in decoded
        try:
            firewall.check_batch(
                batch_id="firewall-demo-batch",
                shard_ids=sample.shard_ids,
                decoded_text=decoded,
                loss_bearing_tokens=sample.loss_bearing_count,
            )
        except FirewallViolation as exc:
            batch_blocked = True
            block_reason = str(exc)

    logger.milestone(
        "evaluation data blocked",
        registry_side=registry_blocked,
        batch_side=batch_blocked,
        canary_present_in_sample=canary_seen,
    )
    logger.check(
        "eval_shard_blocked",
        registry_blocked and batch_blocked,
        registry_side=registry_blocked,
        batch_side=batch_blocked,
        reason=block_reason[:90],
    )
    return firewall.report()


def phase_reproducibility(system, logger: EventLogger) -> dict:
    """Rebuild every shard a second time and compare content hashes.

    "Immutable" has to mean the same code on a different day produces the same
    shard, so this rebuilds them in-process and diffs the hashes rather than
    asserting the property.
    """
    rebuilt = build_shards(
        system.sources, system.tokenizer, system.tokenizer_hash, system.fingerprints
    )
    original = {m.shard_id: m.content_hash for m in system.manifests}
    second = {m.shard_id: m.content_hash for m in rebuilt}

    differing = sorted(
        shard_id for shard_id in original
        if original.get(shard_id) != second.get(shard_id)
    )
    stable = original.keys() == second.keys() and not differing

    logger.check(
        "shards_rebuild_identical",
        stable,
        shards=len(original),
        differing=len(differing),
    )
    return {
        "shards_compared": len(original),
        "content_hashes_stable": stable,
        "shard_ids_stable": sorted(original) == sorted(second),
        "differing_shard_ids": differing,
        "method": "shards rebuilt from the same corpus in-process; hashes diffed",
    }


def phase_mask_validation(system, logger: EventLogger) -> dict:
    """Re-validate every packed sample's masks and attention confinement."""
    pad_id = system.tokenizer.special_id("<pad>")
    mask_problems, attention_problems = [], []

    for sample in system.store.by_id.values():
        result = validate_masks(
            sample.token_ids, sample.loss_mask, sample.segment_ids,
            sample.position_ids, pad_id, sample.graded_flags,
        )
        if not result.ok:
            mask_problems.append({"sample_id": sample.sample_id,
                                  "problems": result.problems[:5]})
        confinement = check_attention_confinement(sample.segment_ids)
        if not confinement.ok:
            attention_problems.append({"sample_id": sample.sample_id,
                                       "problems": confinement.problems[:5]})

    ok = not mask_problems and not attention_problems
    logger.check(
        "masks_and_attention_valid",
        ok,
        samples=len(system.store.by_id),
        mask_problems=len(mask_problems),
        attention_problems=len(attention_problems),
    )
    return {
        "samples_checked": len(system.store.by_id),
        "all_masks_valid": not mask_problems,
        "all_attention_confined": not attention_problems,
        "mask_problems": mask_problems,
        "attention_problems": attention_problems,
        "checks_applied": [
            "no loss on padding",
            "no loss on the first token of a segment",
            "no loss on context-only spans (prompt, tool observation)",
            "position ids segment-relative, contiguous, monotonic",
            "segments contiguous, padding a suffix",
            "attention causal and confined to one packed document",
        ],
    }


def phase_replay(system, logger: EventLogger) -> dict:
    consumption = LedgerStore(
        PATHS.ledgers / f"consumption_{PRIMARY_BRANCH}.jsonl", "consumption"
    )
    planner = BatchPlanner(system.schedule, system.store.lane_samples, PRIMARY_BRANCH)
    start, end = CONFIG.replay_interval
    report = replay_interval(
        consumption_store=consumption,
        branch_id=PRIMARY_BRANCH,
        start_step=start,
        end_step=end,
        sample_store=system.store,
        shard_reader=system.reader,
        planner=planner,
        logger=logger,
    )
    logger.milestone(
        "historical stream replayed",
        interval=f"{start}-{end}",
        microbatches=report["microbatches_replayed"],
        all_match=report["all_match"],
    )
    return report


def phase_fork_divergence(logger: EventLogger) -> dict:
    branches = json.loads((PATHS.ledgers / "branches.json").read_text(encoding="utf-8"))
    fork_ids = [
        b for b, meta in branches["branches"].items() if meta["mode"] == "fork"
    ]
    if not fork_ids:
        logger.check("fork_diverged", False, note="no fork branch recorded")
        return {"diverged_correctly": False}

    fork_id = fork_ids[0]
    parent_store = LedgerStore(
        PATHS.ledgers / f"consumption_{PRIMARY_BRANCH}.jsonl", "consumption"
    )
    fork_store = LedgerStore(
        PATHS.ledgers / f"consumption_{fork_id}.jsonl", "consumption_fork"
    )
    report = divergence_report(
        batch_ids_by_step(parent_store, PRIMARY_BRANCH),
        batch_ids_by_step(fork_store, fork_id),
        CONFIG.fork_from_step,
    )
    report["fork_branch_id"] = fork_id
    logger.check(
        "fork_diverged",
        report["diverged_correctly"],
        branch=fork_id,
        from_step=CONFIG.fork_from_step,
        differing_steps=len(report["differing_after_fork"]),
    )
    return report


def phase_reports(system, logger: EventLogger, firewall_demo: dict) -> dict:
    """Everything derived from the ledgers after training has finished."""
    consumption = LedgerStore(
        PATHS.ledgers / f"consumption_{PRIMARY_BRANCH}.jsonl", "consumption"
    )
    records = [r for r in consumption.read_all() if r["type"] == EVENT_CONSUME]
    integrity = integrity_report(records, PRIMARY_BRANCH)
    write_json(PATHS.ledgers / "consumption_integrity.json", integrity)

    chain_ok, chain_detail = consumption.verify_chain()
    logger.check("ledger_chain_intact", chain_ok, **{
        k: v for k, v in chain_detail.items() if k != "head"
    })
    logger.check(
        "no_skipped_or_repeated_batches",
        integrity["ok"],
        steps=integrity["step_range"],
        duplicates=integrity["duplicate_count"],
        gaps=len(integrity["missing_steps"]),
    )

    # -- mixture compliance: planned versus what the ledger says happened ----
    floors_by_stage = {stage.name: stage.protected_floors for stage in STAGES}
    compliance = compliance_report(
        system.schedule,
        integrity["tokens_by_lane"],
        floors_by_stage=floors_by_stage,
        actual_by_stage=integrity["tokens_by_stage_lane"],
    )
    write_json(PATHS.manifests / "mixture_compliance.json", compliance)
    logger.check(
        "mixture_within_tolerance",
        compliance["all_lanes_within_tolerance"],
        max_delta=compliance["max_abs_delta"],
        tolerance=compliance["tolerance"],
    )
    logger.check(
        "protected_floors_respected",
        compliance["all_floors_respected"] is not False,
        checks=len(compliance["protected_floor_checks"]),
    )

    # -- OPUS ---------------------------------------------------------------
    opus_store = LedgerStore(PATHS.ledgers / "opus_decisions.jsonl", "opus")
    decisions = [
        OpusDecision.from_dict(r["payload"])
        for r in opus_store.read_all()
        if r["type"] == "opus_decision"
    ]
    opus = decisions_report(decisions)
    resume_phase = json.loads(
        (PATHS.ledgers / f"phase_resume_{PRIMARY_BRANCH}.json").read_text(encoding="utf-8")
    )
    opus["proxy_health"] = resume_phase.get("opus_proxy_health", {})
    write_json(PATHS.ledgers / "opus_report.json", opus)
    write_json(PATHS.ledgers / "opus_proxy_health.json", opus["proxy_health"])
    logger.milestone(
        "OPUS decisions recorded",
        candidates=opus["total_candidates_scored"],
        accepted=opus["by_status"].get("accepted", 0),
        rejected=opus["by_status"].get("rejected", 0),
        deferred=opus["by_status"].get("deferred", 0),
        floor_overrides=opus["protected_floor_overrides"],
    )
    logger.check(
        "opus_all_decisions_have_reasons",
        opus["statuses_reconcile"] and opus["total_candidates_scored"] > 0,
        reasons=len(opus["by_reason"]),
    )
    logger.check(
        "opus_protected_floor_override_recorded",
        opus["protected_floor_overrides"] > 0,
        overrides=opus["protected_floor_overrides"],
    )

    # The candidates re-scored after the crash must produce identical scores.
    # Selection is a function of (restored model state, candidate) and nothing
    # else, so this is the sharpest determinism evidence the run produces.
    seen: dict = {}
    rescored, divergent = 0, []
    for decision in decisions:
        previous = seen.get(decision.decision_id)
        if previous is None:
            seen[decision.decision_id] = decision
            continue
        rescored += 1
        if (
            previous.opus_score != decision.opus_score
            or previous.status != decision.status
            or previous.reason != decision.reason
        ):
            divergent.append(decision.decision_id)
    logger.check(
        "opus_rescoring_after_crash_identical",
        rescored > 0 and not divergent,
        rescored_candidates=rescored,
        divergent=len(divergent),
    )

    # -- learning ledger, rebuilt from the durable record --------------------
    learning_store = LedgerStore(
        PATHS.ledgers / f"learning_{PRIMARY_BRANCH}.jsonl", "learning"
    )
    learning = LearningLedger.from_store(learning_store, CONFIG.run_id, PRIMARY_BRANCH)
    aggregates = learning.aggregates()
    recommendations = learning.next_corpus_recommendations()
    write_json(PATHS.ledgers / "learning_aggregates.json", aggregates)
    write_json(PATHS.ledgers / "next_corpus_recommendations.json", recommendations)
    logger.check(
        "learning_trace_linked_to_source",
        aggregates["samples_recorded"] > 0 and aggregates["token_records_written"] > 0,
        samples=aggregates["samples_recorded"],
        tokens_traced=aggregates["token_records_written"],
        shard_cards=len(recommendations["shard_cards"]),
    )
    eos = aggregates.get("eos_perplexity", {})
    logger.check(
        "eos_perplexity_tracked",
        bool(eos.get("samples")),
        samples=eos.get("samples", 0),
        mean=eos.get("mean"),
    )

    # -- firewall: merge the driver demo with what the workers recorded ------
    firewall_ledger = LedgerStore(PATHS.ledgers / "firewall_events.jsonl", "firewall")
    ledger_blocks = [r["payload"] for r in firewall_ledger.read_all()]
    worker_firewall = resume_phase.get("firewall", {})
    combined = {
        "registry_checks": firewall_demo.get("registry_checks", 0)
        + worker_firewall.get("registry_checks", 0),
        "batch_checks": firewall_demo.get("batch_checks", 0)
        + worker_firewall.get("batch_checks", 0),
        "blocks_total": len(ledger_blocks),
        "blocks_by_side": _tally(ledger_blocks, "side"),
        "blocks_by_reason": _tally(ledger_blocks, "reason"),
        "validation_gradient_bearing_tokens": max(
            firewall_demo.get("validation_gradient_bearing_tokens", 0),
            worker_firewall.get("validation_gradient_bearing_tokens", 0),
        ),
        "blocked_shard_ids": sorted({b.get("shard_id", "") for b in ledger_blocks}),
        "fingerprint_registry": firewall_demo.get("fingerprint_registry", {}),
        "blocks": ledger_blocks,
        "note": (
            "blocks are read from ledgers/firewall_events.jsonl, which every "
            "process appends to, so this counts real refusals from the driver "
            "demo, the training worker and the fork worker together"
        ),
    }
    write_json(PATHS.ledgers / "firewall_report.json", combined)
    logger.check(
        "validation_never_gradient_bearing",
        combined["validation_gradient_bearing_tokens"] == 0,
        tokens=combined["validation_gradient_bearing_tokens"],
    )

    return {"integrity": integrity, "compliance": compliance, "opus": opus}


def _tally(records, key):
    out = {}
    for record in records:
        value = record.get(key, "")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def phase_performance(logger: EventLogger, wall_seconds: float) -> dict:
    """Build the performance report from two different sources, on purpose.

    **Efficiency comes from the ledgers.**  Packing utilisation and loss density
    are ratios of token counts, and the consumption ledger holds every one of
    them for every step of every branch - including steps 0-11, whose in-memory
    counters died with the crashed process.  Summing the ledger is both complete
    and independently checkable, which is what the assignment's "must be
    reconstructible" rule asks for.

    **Rates come from the timed phases.**  A tokens-per-second figure needs a
    denominator in seconds, and the only wall-clock measurements that survived
    are the ones the phases that finished wrote down.  Dividing all the tokens
    by only the surviving time would inflate the rate, so the rate is scoped to
    the steps it was actually measured over, and the report says which.
    """
    timings: Dict[str, float] = {}
    phase_counters: Dict[str, int] = {}
    phases: Dict[str, dict] = {}
    for path in sorted(PATHS.ledgers.glob("phase_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        performance = payload.get("performance")
        if not performance:
            continue
        phases[path.stem] = performance
        for key, value in performance.get("timings_seconds", {}).items():
            timings[key] = round(timings.get(key, 0.0) + value, 6)
        for key, value in performance.get("counters", {}).items():
            phase_counters[key] = phase_counters.get(key, 0) + value

    # -- complete token accounting, straight out of every consumption ledger --
    ledger_counters = {
        "raw_positions": 0, "useful_loss_bearing_tokens": 0,
        "pad_tokens": 0, "context_only_tokens": 0, "sequences": 0, "microbatches": 0,
    }
    branches, steps_covered = [], set()
    for path in sorted(PATHS.ledgers.glob("consumption_*.jsonl")):
        store = LedgerStore(path, path.stem)
        for record in store.read_all():
            if record["type"] != EVENT_CONSUME:
                continue
            payload = record["payload"]
            branches.append(payload["branch_id"])
            steps_covered.add((payload["branch_id"], payload["global_step"]))
            ledger_counters["raw_positions"] += payload.get("total_positions", 0)
            ledger_counters["useful_loss_bearing_tokens"] += payload.get(
                "loss_bearing_tokens", 0)
            ledger_counters["pad_tokens"] += payload.get("pad_tokens", 0)
            ledger_counters["context_only_tokens"] += payload.get(
                "context_only_tokens", 0)
            ledger_counters["sequences"] += len(payload.get("packed_sample_ids", []))
            ledger_counters["microbatches"] += 1

    raw = ledger_counters["raw_positions"] or 1
    useful = ledger_counters["useful_loss_bearing_tokens"]
    compute = timings.get("compute", 0.0) or 1e-9
    timed_raw = phase_counters.get("raw_positions", 0)
    timed_useful = phase_counters.get("useful_loss_bearing_tokens", 0)

    report = {
        "counters": ledger_counters,
        "counters_source": (
            "summed from every ledgers/consumption_*.jsonl record; covers all "
            f"{len(steps_covered)} (branch, step) pairs across "
            f"{len(set(branches))} branches, including the pre-crash steps whose "
            "in-memory counters died with the killed process"
        ),
        "timed_phase_counters": phase_counters,
        "timings_seconds": timings,
        "wall_seconds_total": round(wall_seconds, 3),
        "efficiency": {
            "packing_utilisation": round((raw - ledger_counters["pad_tokens"]) / raw, 6),
            "loss_density": round(useful / raw, 6),
            "padding_waste": round(ledger_counters["pad_tokens"] / raw, 6),
            "context_only_share": round(
                ledger_counters["context_only_tokens"] / raw, 6),
            "compute_share_of_wall": round(compute / max(wall_seconds, 1e-9), 6),
        },
        "throughput": {
            "_scope": (
                "rates use the timed phases only (resume + fork), because the "
                "crashed process never reported its wall clock; the token counts "
                "in this block are the matching subset, not the ledger totals"
            ),
            "measured_over_positions": timed_raw,
            "raw_tokens_per_sec_compute": round(timed_raw / compute, 2),
            "useful_tokens_per_sec_compute": round(timed_useful / compute, 2),
            "accepted_tokens_per_sec_compute": round(
                phase_counters.get("accepted_tokens_after_opus", 0) / compute, 2),
            "useful_tokens_per_sec_wall": round(
                timed_useful / max(wall_seconds, 1e-9), 2),
        },
        "opus": _merge_opus_perf(phases),
        "per_phase": phases,
        "how_to_reconstruct": {
            "packing_utilisation": "(raw_positions - pad_tokens) / raw_positions",
            "loss_density": "useful_loss_bearing_tokens / raw_positions",
            "useful_tokens_per_sec_compute": (
                "timed_phase_counters.useful_loss_bearing_tokens / timings.compute"),
            "independent_source": (
                "counters are summed from ledgers/consumption_*.jsonl "
                "(total_positions, pad_tokens, loss_bearing_tokens, "
                "context_only_tokens) - verify_evidence.py re-sums them from the "
                "ledger and compares, rather than trusting this file"
            ),
        },
    }
    write_json(PATHS.performance_json, report)
    logger.milestone(
        "performance measured",
        useful_tokens_per_sec=report["throughput"]["useful_tokens_per_sec_compute"],
        packing_utilisation=report["efficiency"]["packing_utilisation"],
        loss_density=report["efficiency"]["loss_density"],
    )
    return report


def _merge_opus_perf(phases: dict) -> dict:
    scored = rejected = 0
    by_lane = {}
    for performance in phases.values():
        opus = performance.get("opus", {})
        scored += opus.get("candidates_scored", 0)
        rejected += opus.get("candidates_rejected", 0)
        for lane, count in opus.get("rejections_by_lane", {}).items():
            by_lane[lane] = by_lane.get(lane, 0) + count
    return {
        "candidates_scored": scored,
        "candidates_rejected": rejected,
        "rejection_rate": round(rejected / scored, 6) if scored else 0.0,
        "rejections_by_lane": dict(sorted(by_lane.items())),
    }


# --------------------------------------------------------------------------


def main() -> int:
    started = time.perf_counter()

    rmtree(PATHS.submission)
    rmtree(PATHS.work)
    logger = EventLogger.reset_files()

    logger.banner("TDES  Training Data Execution System  ·  full demonstration")
    logger.event(
        "run_configuration",
        run_id=CONFIG.run_id,
        config_hash=CONFIG.short_config_hash,
        seed=CONFIG.master_seed,
        total_steps=CONFIG.total_steps,
        sequences_per_step=CONFIG.sequences_per_step,
        world_size=CONFIG.world_size,
        grad_accum=CONFIG.grad_accum,
    )

    # -- 0. build: corpus -> tokenizer -> shards -> manifests --------------
    logger.banner("PHASE 0  build the data system")
    system = build_data_system(logger=logger)

    reproducibility = phase_reproducibility(system, logger)

    # -- 1. firewall, before anything is scheduled -------------------------
    # The eval firewall runs here, between admission and mixture compilation,
    # because that is where it belongs: never-train data must be refused before
    # the scheduler can plan around it, not after.
    firewall_demo = phase_firewall_demo(system, logger)

    # -- 2. compile the mixture, then materialise the batches ---------------
    finalise_data_system(system, logger=logger)
    write_manifest_artifacts(system)
    write_json(PATHS.manifests / "reproducibility.json", reproducibility)

    masks = phase_mask_validation(system, logger)
    write_json(PATHS.manifests / "mask_validation.json", masks)

    # The reference every worker will rebuild and compare itself against.
    write_json(PATHS.manifests / FINGERPRINT_FILE, system.fingerprint())

    # -- 2. train and crash -------------------------------------------------
    code = run_worker("primary", logger)
    crashed = code == CRASH_EXIT_CODE
    logger.check(
        "crash_simulated_and_process_died",
        crashed,
        exit_code=code,
        expected=CRASH_EXIT_CODE,
    )
    if not crashed:
        logger.warn("primary_phase_did_not_crash", exit_code=code)
        return 1

    # -- 3. resume ----------------------------------------------------------
    if run_worker("resume", logger) != 0:
        logger.warn("resume_phase_failed")
        return 1

    # -- 4. replay ----------------------------------------------------------
    replay = phase_replay(system, logger)
    write_json(PATHS.ledgers / "replay_report.json", replay)

    # -- 5. fork ------------------------------------------------------------
    if run_worker("fork", logger) != 0:
        logger.warn("fork_phase_failed")
        return 1
    divergence = phase_fork_divergence(logger)
    write_json(PATHS.ledgers / "fork_divergence.json", divergence)

    # -- 6. audit -----------------------------------------------------------
    consumption = LedgerStore(
        PATHS.ledgers / f"consumption_{PRIMARY_BRANCH}.jsonl", "consumption"
    )
    opus_store = LedgerStore(PATHS.ledgers / "opus_decisions.jsonl", "opus")
    audit = run_audit(
        consumption, opus_store, PRIMARY_BRANCH,
        checkpoint_step=CONFIG.total_steps,
        interval=CONFIG.replay_interval,
        logger=logger,
    )
    write_json(PATHS.ledgers / "audit_report.json", audit)

    # -- 6b. checkpoint retention -------------------------------------------
    # Keep the weights only where something still depends on them: the fork
    # source and the final state.  Metadata - and therefore the ledger offset
    # behind every checkpoint - is retained for all of them.
    keep = sorted({CONFIG.fork_from_step, CONFIG.total_steps})
    pruned = prune_checkpoints(PRIMARY_BRANCH, keep)
    write_json(
        PATHS.checkpoints / "retention.json",
        {
            "policy": "retain weights for the fork source and the final checkpoint",
            "weights_retained_at_steps": keep,
            "metadata_retained_for_all": True,
            "pruned": pruned,
            "bytes_reclaimed": sum(p["bytes_reclaimed"] for p in pruned),
            "checkpoints": list_checkpoints(PRIMARY_BRANCH),
        },
    )
    logger.event(
        "checkpoints_pruned",
        kept=keep,
        pruned=len(pruned),
        mb_reclaimed=round(sum(p["bytes_reclaimed"] for p in pruned) / 1e6, 1),
    )

    # -- 7. reports ---------------------------------------------------------
    logger.banner("PHASE 7  reports derived from the ledgers")
    phase_reports(system, logger, firewall_demo)
    phase_performance(logger, time.perf_counter() - started)

    # -- 7b. the assignment's completion criterion, as one joined-up check ---
    logger.banner("PHASE 7b  the completion criterion")
    completion = check_completion(
        consumption=LedgerStore(
            PATHS.ledgers / f"consumption_{PRIMARY_BRANCH}.jsonl", "consumption"),
        opus=LedgerStore(PATHS.ledgers / "opus_decisions.jsonl", "opus"),
        learning=LedgerStore(
            PATHS.ledgers / f"learning_{PRIMARY_BRANCH}.jsonl", "learning"),
        branch_id=PRIMARY_BRANCH,
        replay_report=replay,
        resume_report=json.loads(
            (PATHS.ledgers / f"phase_resume_{PRIMARY_BRANCH}.json").read_text(encoding="utf-8")),
        fork_report=divergence,
        shard_cards=json.loads(
            (PATHS.ledgers / "next_corpus_recommendations.json").read_text(encoding="utf-8"))["shard_cards"],
        token_trace_count=json.loads(
            (PATHS.ledgers / "learning_aggregates.json").read_text(encoding="utf-8"))["token_records_written"],
        logger=logger,
    )
    write_json(PATHS.ledgers / "completion_criterion.json", completion)

    # -- 8. evidence --------------------------------------------------------
    logger.banner("PHASE 8  evidence bundle")
    evidence = write_evidence(
        extra={
            "wall_seconds": round(time.perf_counter() - started, 3),
            "python": sys.version.split()[0],
        }
    )
    for requirement in evidence["requirements"]:
        logger.check(
            "evidence_" + requirement["requirement"].lower().replace(" ", "_"),
            requirement["result"] == "PASS",
        )

    verifier = subprocess.run(
        [sys.executable, "-m", "tdes.cli.verify_evidence"], cwd=str(PATHS.root)
    )
    logger.check("evidence_independently_verified", verifier.returncode == 0,
                 exit_code=verifier.returncode)

    elapsed = time.perf_counter() - started
    logger.banner(
        f"COMPLETE  {evidence['requirements_passed']}/{evidence['requirements_total']} "
        f"requirements passed in {elapsed:.1f}s"
    )
    logger.event(
        "summary",
        pass_lines=logger.pass_count,
        fail_lines=logger.fail_count,
        artifacts=str(PATHS.submission.relative_to(PATHS.root)),
    )
    return 0 if evidence["all_passed"] and verifier.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
