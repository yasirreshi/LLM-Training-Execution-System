"""The training loop.

Everything the data system does converges here, so the order of operations in
`run_step` is the important part of this file:

    1. plan the step            pure, recomputable, no model involved
    2. score and select         OPUS, against the current checkpoint's proxy
    3. assemble the batch       accepted samples laid out into ranks/accum slots
    4. measure loss BEFORE      on exactly the tokens about to be trained on
    5. per microbatch:
         firewall check         decoded text, last gate before gradients
         forward + backward     gradients accumulate
         record consumption     ledger written and fsynced per microbatch
    6. optimizer step
    7. measure loss AFTER       same tokens, so the delta is a real measurement
    8. record learning          per sample, plus a full token trace in-interval
    9. checkpoint               model + optimizer + scheduler + ledger offset

Step 5 writing the ledger *before* step 6 is what creates the crash asymmetry
the resume logic exists to handle: the data record is durable before the weight
update is.  That ordering is correct - a batch that was served must be recorded
even if the process dies before learning from it - and it is why resume has to
roll the ledger back rather than simply continue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..config import CONFIG, EOS_TOKEN, PAD_TOKEN
from ..events import EventLogger
from ..ledger.consumption import ConsumptionLedger, EVENT_CHECKPOINT, EVENT_STEP
from ..ledger.learning import LearningLedger, SampleLearning
from ..streams.planner import assemble_batch
from .checkpoint import build_meta, checkpoint_id_for, save_checkpoint
from .crash import die_mid_write
from .determinism import rng_fingerprint, torch


@dataclass
class StepResult:
    step: int
    batch_id: str
    plan_hash: str
    loss: float
    grad_norm: float
    lr: float
    loss_bearing_tokens: int
    checkpoint_id: str = ""

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "batch_id": self.batch_id,
            "plan_hash": self.plan_hash,
            "loss": round(self.loss, 6),
            "grad_norm": round(self.grad_norm, 6),
            "learning_rate": round(self.lr, 8),
            "loss_bearing_tokens": self.loss_bearing_tokens,
            "checkpoint_id": self.checkpoint_id,
        }


@dataclass
class RunOutcome:
    steps: List[StepResult] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    batch_ids: Dict[int, str] = field(default_factory=dict)
    plan_hashes: Dict[int, str] = field(default_factory=dict)


class Trainer:
    def __init__(
        self,
        *,
        run_id: str,
        branch_id: str,
        model,
        optimizer,
        scheduler,
        tokenizer,
        sample_store,
        planner,
        schedule,
        selector,
        proxy,
        consumption: ConsumptionLedger,
        learning: LearningLedger,
        firewall,
        perf,
        logger: EventLogger,
        tokenizer_hash: str,
        opus_store=None,
        probe_samples: Sequence = (),
        validation_samples: Sequence = (),
        token_trace_interval=None,
    ):
        self.run_id = run_id
        self.branch_id = branch_id
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.store = sample_store
        self.planner = planner
        self.schedule = schedule
        self.selector = selector
        self.proxy = proxy
        self.consumption = consumption
        self.learning = learning
        self.firewall = firewall
        self.perf = perf
        self.log = logger
        self.tokenizer_hash = tokenizer_hash
        self.opus_store = opus_store
        self.probe_samples = list(probe_samples)
        self.validation_samples = list(validation_samples)
        self.token_trace_interval = token_trace_interval or CONFIG.token_trace_interval

        self.pad_id = tokenizer.special_id(PAD_TOKEN)
        self.eos_id = tokenizer.special_id(EOS_TOKEN)
        self.current_checkpoint_id = "genesis"
        self.tokens_consumed = 0
        self.loss_bearing_consumed = 0
        self.last_batch_id = ""
        self._opus_milestone_emitted = False

    # -- tensors ----------------------------------------------------------

    def batch_tensors(self, samples: Sequence, truncate: Optional[int] = None) -> dict:
        """Stack packed samples into tensors, optionally using only a prefix.

        `truncate` is how OPUS scores cheaply: the design's construction scores
        the first 512 tokens of a 32k sample, which is a ~1.6% probe.  Here the
        probe is 64 of 256 for the same reason - the score has to be much
        cheaper than the training step or selection stops paying for itself.
        """
        t = torch()
        length = min(truncate, samples[0].sequence_length) if truncate else samples[0].sequence_length
        tokens, masks, segments, positions = [], [], [], []
        for sample in samples:
            tokens.append(sample.token_ids[:length])
            masks.append(sample.loss_mask[:length])
            segments.append(sample.segment_ids[:length])
            positions.append(sample.position_ids[:length])
        return {
            "token_ids": t.tensor(tokens, dtype=t.long),
            "loss_mask": t.tensor(masks, dtype=t.long),
            # Padding keeps its PAD_SEGMENT (-1) marker.  The attention bias
            # only ever compares segment ids for equality, so pad attends to
            # pad and nothing else - and the loss mask discards those rows.
            "segment_ids": t.tensor(segments, dtype=t.long),
            "position_ids": t.tensor(positions, dtype=t.long),
        }

    def per_sample_loss(self, samples: Sequence) -> List[float]:
        """Mean loss per sample, computed in one forward pass, no gradients."""
        t = torch()
        if not samples:
            return []
        batch = self.batch_tensors(samples)
        with t.no_grad():
            per_token, mask, _ = self.model.token_losses(
                batch["token_ids"], batch["position_ids"],
                batch["segment_ids"], batch["loss_mask"],
            )
            totals = (per_token * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return [float(v) for v in (totals / counts)]

    # -- one step ---------------------------------------------------------

    def run_step(self, step: int, crash_at: Optional[int] = None) -> StepResult:
        t = torch()
        cfg = CONFIG

        with self.perf.timer("loader"):
            plan = self.planner.plan(step)

        # A fresh proxy direction at each round boundary.  Rounds are aligned to
        # checkpoints so the direction is always computed from a model state a
        # resume can restore exactly - that is what makes the re-scored
        # decisions after a crash reproduce the originals.
        if step % cfg.opus_round_interval == 0 or self.proxy.direction is None:
            with self.perf.timer("opus"):
                probe = self._probe_batches()
                direction = self.proxy.compute_direction(
                    probe, self.current_checkpoint_id, step
                )
            if not self._opus_milestone_emitted:
                self._opus_milestone_emitted = True
                self.log.milestone(
                    "OPUS decisions recorded",
                    first_round_step=step,
                    proxy_version=direction.version,
                    scoring_checkpoint=self.current_checkpoint_id,
                )
            self.log.event(
                "opus_proxy_direction",
                step=step,
                checkpoint=self.current_checkpoint_id,
                probe_batches=direction.probe_batches,
                dim=int(direction.vector.numel()),
            )

        with self.perf.timer("opus"):
            outcome = self.selector.select_step(
                plan, self.batch_tensors, self.current_checkpoint_id
            )
        for decision in outcome.decisions:
            self.perf.count_candidate(decision.lane)
            if decision.status != "accepted":
                self.perf.count_rejection(decision.lane, decision.effective_tokens)
            self.selector_ledger_append(decision)

        batch = assemble_batch(plan, outcome.accepted, outcome.decision_id_by_sample)
        samples = [self.store.get(sid) for sid in batch.sample_ids]

        loss_before = self.per_sample_loss(samples)

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_targets = 0
        fingerprint = rng_fingerprint(cfg.master_seed, self.branch_id, step)

        for index, microbatch in enumerate(batch.microbatches):
            mb_samples = [self.store.get(sid) for sid in microbatch.sample_ids]

            # --- firewall, batch side: decoded text, last gate before grads ---
            with self.perf.timer("firewall"):
                decoded = "\n".join(s.decoded(self.tokenizer) for s in mb_samples)
                self.firewall.check_batch(
                    batch_id=batch.batch_id,
                    shard_ids=sorted({sid for s in mb_samples for sid in s.shard_ids}),
                    decoded_text=decoded,
                    loss_bearing_tokens=sum(s.loss_bearing_count for s in mb_samples),
                )

            with self.perf.timer("compute"):
                tensors = self.batch_tensors(mb_samples)
                loss, targets = self.model.masked_loss(
                    tensors["token_ids"], tensors["position_ids"],
                    tensors["segment_ids"], tensors["loss_mask"],
                )
                (loss / cfg.grad_accum).backward()

            total_loss += float(loss) * float(targets)
            total_targets += int(targets)
            self.perf.count_microbatch(mb_samples)

            # --- consumption ledger, written and fsynced before the update ---
            self.consumption.record_microbatch(
                global_step=step,
                checkpoint_id=self.current_checkpoint_id,
                rank=microbatch.rank,
                microbatch_id=microbatch.microbatch_id,
                batch_id=batch.batch_id,
                plan_hash=batch.plan_hash,
                stage=batch.stage,
                sequence_length=batch.sequence_length,
                samples=mb_samples,
                lane_of_sample=batch.lane_of_sample,
                opus_decision_ids=batch.opus_decision_ids,
                pass_numbers=plan.pass_numbers,
                rng_fingerprint=fingerprint,
            )
            self.tokens_consumed += sum(s.sequence_length for s in mb_samples)
            self.loss_bearing_consumed += sum(s.loss_bearing_count for s in mb_samples)

            # --- the deliberate crash --------------------------------------
            # Fires mid-step, after some microbatches are durably recorded and
            # before the optimizer step, so the ledger genuinely runs ahead of
            # the last checkpoint.
            if crash_at is not None and step == crash_at and index == cfg.world_size - 1:
                self.log.milestone(
                    "crash simulated",
                    step=step,
                    microbatches_recorded=index + 1,
                    last_checkpoint=self.current_checkpoint_id,
                )
                die_mid_write(
                    self.consumption.store,
                    "consume_microbatch",
                    {
                        "run_id": self.run_id,
                        "branch_id": self.branch_id,
                        "global_step": step,
                        "note": "record interrupted by injected crash",
                    },
                    logger=self.log,
                )

        with self.perf.timer("compute"):
            grad_norm = float(
                t.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        loss_after = self.per_sample_loss(samples)
        mean_loss = total_loss / max(1, total_targets)
        lr = self.optimizer.param_groups[0]["lr"]

        self._record_learning(
            step, plan, batch, samples, loss_before, loss_after, grad_norm, outcome
        )

        self.consumption.record(
            EVENT_STEP,
            {
                "global_step": step,
                "batch_id": batch.batch_id,
                "plan_hash": batch.plan_hash,
                "curriculum_stage": batch.stage,
                "mean_loss": round(mean_loss, 6),
                "perplexity": round(math.exp(min(mean_loss, 20.0)), 6),
                "gradient_norm": round(grad_norm, 6),
                "learning_rate": round(lr, 8),
                "loss_bearing_tokens": total_targets,
                "checkpoint_id": self.current_checkpoint_id,
                "rng_fingerprint": fingerprint,
            },
        )
        self.perf.counters.steps += 1
        self.last_batch_id = batch.batch_id

        return StepResult(
            step=step,
            batch_id=batch.batch_id,
            plan_hash=batch.plan_hash,
            loss=mean_loss,
            grad_norm=grad_norm,
            lr=lr,
            loss_bearing_tokens=total_targets,
        )

    # -- learning ledger --------------------------------------------------

    def _record_learning(
        self, step, plan, batch, samples, loss_before, loss_after, grad_norm, outcome
    ) -> None:
        phase = CONFIG.model_phase(step)
        next_ckpt = checkpoint_id_for(self.branch_id, step + 1)
        scores = {d.candidate_id: d.opus_score for d in outcome.decisions}
        trace_lo, trace_hi = self.token_trace_interval
        in_trace_window = trace_lo <= step < trace_hi

        token_stats = self._token_statistics(samples)

        for index, sample in enumerate(samples):
            stats = token_stats.get(sample.sample_id, {})
            self.learning.record_sample(
                SampleLearning(
                    step=step,
                    branch_id=self.branch_id,
                    sample_id=sample.sample_id,
                    lane=batch.lane_of_sample.get(sample.sample_id, sample.lane),
                    stage=batch.stage,
                    shard_ids=sample.shard_ids,
                    doc_ids=sample.doc_ids,
                    loss_before=loss_before[index],
                    loss_after=loss_after[index],
                    grad_norm=grad_norm,
                    loss_bearing_tokens=sample.loss_bearing_count,
                    model_phase=phase,
                    checkpoint_before=self.current_checkpoint_id,
                    checkpoint_after=next_ckpt,
                    opus_decision_id=batch.opus_decision_ids.get(sample.sample_id, ""),
                    opus_score=scores.get(sample.sample_id),
                    pass_number=plan.pass_numbers.get(sample.sample_id, 0),
                    mean_token_ppl=stats.get("mean_ppl", 0.0),
                    eos_ppl=stats.get("eos_ppl"),
                )
            )

        if in_trace_window:
            for sample in samples:
                records = self._token_trace_records(sample)
                if not records:
                    continue
                self.learning.record_token_trace(
                    step=step,
                    sample_id=sample.sample_id,
                    lane=sample.lane,
                    stage=batch.stage,
                    model_phase=phase,
                    checkpoint_before=self.current_checkpoint_id,
                    opus_decision_id=batch.opus_decision_ids.get(sample.sample_id, ""),
                    pass_number=plan.pass_numbers.get(sample.sample_id, 0),
                    tokens=records,
                )

    def _token_statistics(self, samples: Sequence) -> Dict[str, dict]:
        """Mean per-token perplexity, and perplexity at EOS specifically.

        EOS is tracked separately because it answers a question nothing else
        does: is the model learning that documents end?  A model that never
        gets surprised anywhere except at EOS has not learned boundaries, it has
        learned to keep going.
        """
        t = torch()
        if not samples:
            return {}
        batch = self.batch_tensors(samples)
        with t.no_grad():
            per_token, mask, targets = self.model.token_losses(
                batch["token_ids"], batch["position_ids"],
                batch["segment_ids"], batch["loss_mask"],
            )
        out: Dict[str, dict] = {}
        for index, sample in enumerate(samples):
            row_mask = mask[index]
            row_loss = per_token[index]
            selected = row_loss[row_mask > 0]
            mean_ppl = (
                float(t.exp(selected.clamp(max=20.0)).mean()) if selected.numel() else 0.0
            )
            eos_positions = (targets[index] == self.eos_id) & (row_mask > 0)
            eos_ppl = (
                float(t.exp(row_loss[eos_positions].clamp(max=20.0)).mean())
                if bool(eos_positions.any())
                else None
            )
            out[sample.sample_id] = {"mean_ppl": mean_ppl, "eos_ppl": eos_ppl}
        return out

    def _token_trace_records(self, sample) -> List[dict]:
        """The expensive tier: one record per loss-bearing token."""
        t = torch()
        batch = self.batch_tensors([sample])
        with t.no_grad():
            per_token, mask, targets = self.model.token_losses(
                batch["token_ids"], batch["position_ids"],
                batch["segment_ids"], batch["loss_mask"],
            )
        records: List[dict] = []
        for position in range(per_token.shape[1]):
            if mask[0, position] <= 0:
                continue
            target_index = position + 1        # logits at p predict token p+1
            token_id = int(targets[0, position])
            loss = float(per_token[0, position])
            segment = sample.segment_at(target_index)
            records.append(
                {
                    "token_id": token_id,
                    "preview": self.tokenizer.decode_one(token_id),
                    "position_in_sequence": target_index,
                    "position_in_segment": int(sample.position_ids[target_index]),
                    "doc_id": segment.doc_id if segment else "",
                    "shard_id": segment.shard_id if segment else "",
                    "lang": segment.lang if segment else "",
                    "script": segment.script if segment else "",
                    "lane": sample.lane,
                    "is_special": self.tokenizer.is_special(token_id),
                    "is_eos": token_id == self.eos_id,
                    "loss_mask": 1,
                    "cross_entropy": round(loss, 6),
                    "perplexity": round(math.exp(min(loss, 20.0)), 6),
                }
            )
        return records

    # -- probe / validation ------------------------------------------------

    def _probe_batches(self) -> List[dict]:
        batches = []
        chunk = max(1, CONFIG.microbatch_size)
        selected = self.probe_samples[: CONFIG.opus_probe_batches * chunk]
        for start in range(0, len(selected), chunk):
            group = selected[start:start + chunk]
            if group:
                batches.append(self.batch_tensors(group, truncate=CONFIG.opus_probe_tokens))
        # Reading held-out data for a direction is allowed; producing a
        # gradient update from it is not.  Record the read either way.
        self.firewall.note_validation_read(
            sorted({sid for s in selected for sid in s.shard_ids}), gradient_bearing=False
        )
        return batches

    def evaluate_validation(self, step: int) -> Optional[float]:
        if not self.validation_samples:
            return None
        with self.perf.timer("validation"):
            losses = self.per_sample_loss(self.validation_samples)
        if not losses:
            return None
        mean = sum(losses) / len(losses)
        self.learning.record_validation(
            step=step,
            loss=mean,
            shard_ids={sid for s in self.validation_samples for sid in s.shard_ids},
            tokens=sum(s.loss_bearing_count for s in self.validation_samples),
        )
        return mean

    # -- OPUS decision ledger ---------------------------------------------

    def selector_ledger_append(self, decision) -> None:
        """Every candidate gets a record, accepted or not.

        The rejected and deferred records are the ones with future value: they
        say what the selector considered low value at this model age, which is
        exactly the question the next corpus needs answered.
        """
        if self.opus_store is None:
            return
        self.opus_store.append("opus_decision", decision.as_dict())

    # -- checkpointing -----------------------------------------------------

    def save(self, completed_steps: int, stage: str) -> str:
        next_plan_hash = ""
        if completed_steps < len(self.schedule.steps):
            next_plan_hash = self.planner.plan_hash(completed_steps)

        meta = build_meta(
            run_id=self.run_id,
            branch_id=self.branch_id,
            global_step=completed_steps,
            stage=stage,
            consumption_offset=self.consumption.store.current_offset(),
            learning_offset=self.learning.store.current_offset(),
            tokenizer_hash=self.tokenizer_hash,
            schedule_hash=self.schedule.schedule_hash,
            index_hash=self.planner.index_hash,
            tokens_consumed=self.tokens_consumed,
            loss_bearing_tokens_consumed=self.loss_bearing_consumed,
            last_batch_id=self.last_batch_id,
            next_expected_plan_hash=next_plan_hash,
            parent_checkpoint_id=self.current_checkpoint_id,
        )
        save_checkpoint(
            model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, meta=meta,
        )
        # The checkpoint event is appended *after* the checkpoint is durable, so
        # a crash between the two leaves an unreferenced checkpoint rather than
        # a ledger pointing at one that does not exist.
        self.consumption.record(
            EVENT_CHECKPOINT,
            {
                "checkpoint_id": meta.checkpoint_id,
                "global_step": completed_steps,
                "ledger_offset": meta.ledger_offset,
                "learning_offset": meta.learning_offset,
                "rng_fingerprint": meta.rng_fingerprint,
                "next_expected_plan_hash": next_plan_hash,
                "tokens_consumed": self.tokens_consumed,
                "state_hash": meta.state_hash,
            },
        )
        self.current_checkpoint_id = meta.checkpoint_id
        self.log.milestone(
            "checkpoint saved",
            checkpoint=meta.checkpoint_id,
            step=completed_steps,
            ledger_offset_bytes=meta.ledger_offset["byte_offset"],
        )
        self.log.check(
            "checkpoint_saved",
            True,
            checkpoint=meta.checkpoint_id,
            step=completed_steps,
            ledger_seq=meta.ledger_offset["event_seq"],
            ledger_bytes=meta.ledger_offset["byte_offset"],
        )
        return meta.checkpoint_id

    # -- the run -----------------------------------------------------------

    def run(
        self,
        start_step: int,
        end_step: int,
        crash_at: Optional[int] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> RunOutcome:
        interval = checkpoint_interval or CONFIG.checkpoint_interval
        outcome = RunOutcome()

        for step in range(start_step, end_step):
            result = self.run_step(step, crash_at=crash_at)
            outcome.steps.append(result)
            outcome.batch_ids[step] = result.batch_id
            outcome.plan_hashes[step] = result.plan_hash
            if outcome.initial_loss is None:
                outcome.initial_loss = result.loss
            outcome.final_loss = result.loss

            self.log.event(
                "step",
                n=step,
                stage=self.schedule.steps[step].stage,
                loss=round(result.loss, 4),
                ppl=round(math.exp(min(result.loss, 20.0)), 3),
                grad_norm=round(result.grad_norm, 4),
                batch=result.batch_id,
            )

            completed = step + 1
            if completed % interval == 0:
                validation_loss = self.evaluate_validation(step)
                checkpoint_id = self.save(completed, self.schedule.steps[step].stage)
                outcome.checkpoints.append(checkpoint_id)
                result.checkpoint_id = checkpoint_id
                if validation_loss is not None:
                    self.log.event(
                        "validation",
                        step=step,
                        loss=round(validation_loss, 4),
                        gradient_bearing=False,
                    )

        return outcome
