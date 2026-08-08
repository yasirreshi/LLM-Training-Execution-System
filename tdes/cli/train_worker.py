"""The training process.

This runs as a *separate OS process* for one reason: it has to be killable.  An
in-process exception unwinds the stack, flushes buffers and runs destructors,
which is precisely the cleanup a real crash does not do.  Running the trainer
here and hard-exiting it with `os._exit(137)` reproduces the state the recovery
path actually has to cope with.

Three phases, selected by `--phase`:

    primary   train from step 0 and die mid-step at the configured crash step
    resume    repair, restore, roll the ledger back, and finish the run
    fork      restore an earlier checkpoint into a new branch and diverge

Each phase rebuilds the data system from the corpus in its own process.  That is
not wasted work - it is the check that shard hashes, packed sample ids and the
shuffle index are functions of the corpus rather than of process state.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from ..config import CONFIG, PATHS
from ..events import EventLogger
from ..fsutil import write_json
from ..ledger.branch import Branch, MODE_PRIMARY
from ..ledger.consumption import ConsumptionLedger
from ..ledger.learning import LearningLedger
from ..ledger.store import LedgerStore
from ..opus.proxy import OpusProxy
from ..opus.selector import OpusSelector
from ..perf.metrics import PerfTracker
from ..pipeline import build_and_finalise
from ..streams.planner import BatchPlanner
from ..streams.resume import prepare_resume, verify_next_batch, verify_rollback_replay
from ..training.checkpoint import checkpoint_dir, load_checkpoint
from ..training.determinism import configure
from ..training.loop import Trainer
from ..training.model import TinyGPT, expected_initial_loss, model_config

PRIMARY_BRANCH = "main"


def ledger_paths(branch_id: str):
    return (
        PATHS.ledgers / f"consumption_{branch_id}.jsonl",
        PATHS.ledgers / f"learning_{branch_id}.jsonl",
    )


def build_optimizer(model):
    from ..training.determinism import torch

    t = torch()
    optimizer = t.optim.AdamW(
        model.parameters(), lr=CONFIG.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )

    def lr_lambda(step: int) -> float:
        # Warm up, then cosine decay.  The scheduler's state is checkpointed so
        # a resume continues the same curve rather than restarting the warmup -
        # a restarted warmup is a silent data-independent change to the
        # experiment, and would confound any comparison across the crash.
        if step < CONFIG.warmup_lr_steps:
            return (step + 1) / CONFIG.warmup_lr_steps
        progress = (step - CONFIG.warmup_lr_steps) / max(
            1, CONFIG.total_steps - CONFIG.warmup_lr_steps
        )
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def make_trainer(system, branch_id: str, logger: EventLogger, seed: int = None,
                 opus_store: LedgerStore = None):
    consumption_path, learning_path = ledger_paths(branch_id)
    consumption_store = LedgerStore(consumption_path, f"consumption_{branch_id}")
    learning_store = LedgerStore(learning_path, f"learning_{branch_id}")
    firewall_store = LedgerStore(PATHS.ledgers / "firewall_events.jsonl", "firewall")
    if opus_store is None:
        opus_store = LedgerStore(PATHS.ledgers / "opus_decisions.jsonl", "opus")

    system.firewall.ledger = firewall_store

    model = TinyGPT(model_config(system.vocab_size, CONFIG))
    optimizer, scheduler = build_optimizer(model)

    planner = BatchPlanner(
        system.schedule, system.store.lane_samples, branch_id,
        master_seed=seed if seed is not None else CONFIG.master_seed,
    )
    proxy = OpusProxy(model, CONFIG.opus_probe_tokens)
    selector = OpusSelector(proxy, system.store)

    consumption = ConsumptionLedger(
        consumption_store, CONFIG.run_id, branch_id, system.tokenizer_hash
    )
    learning = LearningLedger(learning_store, CONFIG.run_id, branch_id)

    trainer = Trainer(
        run_id=CONFIG.run_id,
        branch_id=branch_id,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=system.tokenizer,
        sample_store=system.store,
        planner=planner,
        schedule=system.schedule,
        selector=selector,
        proxy=proxy,
        consumption=consumption,
        learning=learning,
        firewall=system.firewall,
        perf=PerfTracker(),
        logger=logger,
        tokenizer_hash=system.tokenizer_hash,
        opus_store=opus_store,
        probe_samples=system.probe_samples,
        validation_samples=system.validation_samples,
    )
    return trainer, consumption_store, learning_store, opus_store


def write_phase_report(branch_id: str, phase: str, payload: dict) -> Path:
    path = PATHS.ledgers / f"phase_{phase}_{branch_id}.json"
    return write_json(path, payload)


# --------------------------------------------------------------------------


def run_primary(logger: EventLogger) -> int:
    system = build_and_finalise(logger=logger)
    trainer, consumption_store, _learning_store, _opus = make_trainer(
        system, PRIMARY_BRANCH, logger
    )

    logger.event(
        "model_initialised",
        parameters=trainer.model.parameter_count(),
        vocab_size=system.vocab_size,
        expected_initial_loss=round(expected_initial_loss(system.vocab_size), 4),
    )

    # The free sanity check: an untrained model should score about ln(V).
    baseline = trainer.per_sample_loss(
        [system.store.get(s) for s in list(system.store.by_id)[:4]]
    )
    measured = sum(baseline) / len(baseline)
    expected = expected_initial_loss(system.vocab_size)
    logger.check(
        "initial_loss_matches_uniform_prior",
        abs(measured - expected) < 0.6,
        measured=round(measured, 4),
        expected_ln_vocab=round(expected, 4),
        delta=round(measured - expected, 4),
    )

    write_json(
        PATHS.ledgers / "branches_seed.json",
        Branch(
            branch_id=PRIMARY_BRANCH,
            run_id=CONFIG.run_id,
            mode=MODE_PRIMARY,
            seed=CONFIG.master_seed,
            schedule_hash=system.schedule.schedule_hash,
            index_hash=trainer.planner.index_hash,
            config_hash=CONFIG.config_hash,
        ).as_dict(),
    )

    logger.banner(f"PHASE 1  train from step 0, crash injected at step {CONFIG.crash_step}")
    outcome = trainer.run(0, CONFIG.total_steps, crash_at=CONFIG.crash_step)

    # Only reached if the crash did not fire, which is itself a failure of the
    # demonstration and must be visible rather than silently tolerated.
    write_phase_report(PRIMARY_BRANCH, "primary", {
        "completed_steps": len(outcome.steps),
        "crash_fired": False,
        "batch_ids": {str(k): v for k, v in outcome.batch_ids.items()},
    })
    logger.check("crash_injected", False, note="crash step was never reached")
    return 1


def run_resume(logger: EventLogger) -> int:
    system = build_and_finalise(logger=logger)
    trainer, consumption_store, learning_store, _opus = make_trainer(
        system, PRIMARY_BRANCH, logger
    )

    logger.banner("PHASE 2  resume from the last checkpoint")

    state = prepare_resume(
        branch_id=PRIMARY_BRANCH,
        consumption_store=consumption_store,
        learning_store=learning_store,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        logger=logger,
    )
    trainer.current_checkpoint_id = state.meta.checkpoint_id
    trainer.tokens_consumed = state.meta.tokens_consumed
    trainer.loss_bearing_consumed = state.meta.loss_bearing_tokens_consumed

    logger.milestone(
        "run resumed",
        checkpoint=state.meta.checkpoint_id,
        resume_step=state.resume_step,
        discarded_records=len(state.discarded),
        discarded_steps=state.discarded_steps,
        torn_tail_repaired=state.torn_tail is not None,
    )
    logger.check(
        "ledger_torn_tail_repaired",
        state.torn_tail is not None,
        removed_bytes=(state.torn_tail or {}).get("removed_bytes", 0),
        reason=(state.torn_tail or {}).get("reason", "none"),
    )

    next_batch = verify_next_batch(state, trainer.planner, logger)

    outcome = trainer.run(state.resume_step, CONFIG.total_steps)

    rollback = verify_rollback_replay(
        state, consumption_store, PRIMARY_BRANCH, logger
    )

    write_phase_report(PRIMARY_BRANCH, "resume", {
        "recovery": state.summary(),
        "next_batch_verification": next_batch,
        "rollback_replay_verification": rollback,
        "steps_completed": len(outcome.steps),
        "initial_loss": outcome.initial_loss,
        "final_loss": outcome.final_loss,
        "batch_ids": {str(k): v for k, v in outcome.batch_ids.items()},
        "plan_hashes": {str(k): v for k, v in outcome.plan_hashes.items()},
        # The decision records themselves live in ledgers/opus_decisions.jsonl;
        # the driver aggregates them from there rather than from memory, so the
        # report is derived from the durable record.
        "opus_proxy_health": trainer.selector.proxy_health(),
        "firewall": trainer.firewall.report(),
        "performance": trainer.perf.report(),
    })
    return 0


def run_fork(logger: EventLogger) -> int:
    system = build_and_finalise(logger=logger)

    parent_checkpoint = checkpoint_dir(PRIMARY_BRANCH, CONFIG.fork_from_step)
    if not (parent_checkpoint / "meta.json").exists():
        logger.check("branch_forked", False, note=f"no checkpoint at {parent_checkpoint.name}")
        return 1

    parent = Branch(
        branch_id=PRIMARY_BRANCH, run_id=CONFIG.run_id, mode=MODE_PRIMARY,
        seed=CONFIG.master_seed, config_hash=CONFIG.config_hash,
    )
    from ..ledger.branch import BranchRegistry

    registry = BranchRegistry()
    registry.register(parent)
    fork = registry.fork(
        parent,
        from_step=CONFIG.fork_from_step,
        from_checkpoint=f"ckpt_{PRIMARY_BRANCH}_{CONFIG.fork_from_step:04d}",
        note=(
            "deliberate divergence: new branch seed, so the data stream after "
            "the fork point differs and every later difference is attributable"
        ),
        seed=CONFIG.master_seed + 977,
    )

    logger.banner(f"PHASE 4  fork from step {CONFIG.fork_from_step} into {fork.branch_id}")

    trainer, consumption_store, learning_store, _opus = make_trainer(
        system, fork.branch_id, logger, seed=fork.seed
    )
    meta = load_checkpoint(
        parent_checkpoint, trainer.model, trainer.optimizer, trainer.scheduler
    )
    trainer.current_checkpoint_id = meta.checkpoint_id
    trainer.tokens_consumed = meta.tokens_consumed
    trainer.loss_bearing_consumed = meta.loss_bearing_tokens_consumed

    fork.schedule_hash = system.schedule.schedule_hash
    fork.index_hash = trainer.planner.index_hash

    consumption_store.append(
        "branch_forked",
        {
            **fork.as_dict(),
            "parent_checkpoint_state_hash": meta.state_hash,
            "parent_index_hash": meta.index_hash,
        },
    )

    end = min(CONFIG.total_steps, CONFIG.fork_from_step + CONFIG.fork_steps)
    outcome = trainer.run(CONFIG.fork_from_step, end, checkpoint_interval=10 ** 9)
    fork.steps_completed = len(outcome.steps)
    registry.write()

    write_phase_report(fork.branch_id, "fork", {
        "branch": fork.as_dict(),
        "parent_checkpoint": meta.as_dict(),
        "steps": [s.as_dict() for s in outcome.steps],
        "batch_ids": {str(k): v for k, v in outcome.batch_ids.items()},
        "plan_hashes": {str(k): v for k, v in outcome.plan_hashes.items()},
        "firewall": trainer.firewall.report(),
        "performance": trainer.perf.report(),
    })
    logger.milestone(
        "branch forked",
        branch=fork.branch_id,
        parent=PRIMARY_BRANCH,
        from_step=CONFIG.fork_from_step,
        steps=len(outcome.steps),
    )
    return 0


PHASES = {"primary": run_primary, "resume": run_resume, "fork": run_fork}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TDES training worker")
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    args = parser.parse_args(argv)

    configure(seed=CONFIG.master_seed)
    logger = EventLogger()          # append to the log the driver created
    return PHASES[args.phase](logger)


if __name__ == "__main__":
    sys.exit(main())
