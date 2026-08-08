"""OPUS selection: accept, reject, defer, and the protected-floor override.

Rejections are the valuable output, not a by-product.  The design is explicit:
a rejected batch tells you what the selector considered low value, what the
protected floors had to rescue, what the model was already comfortable with, and
what may deserve review in a later phase.  So every candidate gets a record,
with a reason from a fixed vocabulary, and rejected data is never discarded -
it goes to the deferred ledger.

Four ledgers, exactly as the design calls for: accepted, rejected,
deferred, protected.

The decision procedure for one lane in one step:

1. Score every candidate against the proxy direction.
2. Drop exact duplicates of a window already accepted this step.
3. Drop candidates whose stage hint does not match the current stage, unless the
   lane is protected - a protected lane cannot afford to be picky.
4. Accept the top `quota` survivors that clear the threshold.
5. If that leaves the lane short:
      protected lane   -> accept anyway, flag `protected_floor_override`
      unprotected lane -> accept anyway, reason `quota_pressure`
   Either way the batch geometry is filled; the difference is *why*, and that
   difference is recorded.
6. Everything scored but not accepted is deferred, not deleted.

The threshold is the median of the round's own scores rather than a constant.
A fixed threshold stops meaning anything as the model moves - early in training
almost everything aligns, late almost nothing does.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..config import PROTECTED_LANES
from ..hashing import short_hash

STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_DEFERRED = "deferred"

REASON_BELOW_THRESHOLD = "below_proxy_threshold"
REASON_QUOTA_PRESSURE = "quota_pressure"
REASON_DUPLICATE = "duplicate"
REASON_STAGE_MISMATCH = "stage_mismatch"
REASON_PROTECTED_LANE_BIAS = "protected_lane_bias"
REASON_ACCEPTED_ON_MERIT = "above_proxy_threshold"

ALL_REASONS = (
    REASON_ACCEPTED_ON_MERIT,
    REASON_BELOW_THRESHOLD,
    REASON_QUOTA_PRESSURE,
    REASON_DUPLICATE,
    REASON_STAGE_MISMATCH,
    REASON_PROTECTED_LANE_BIAS,
)

STAGE_ORDER = {"early": 0, "mid": 1, "anneal": 2}


@dataclass
class OpusDecision:
    decision_id: str
    candidate_id: str          # the packed sample id
    step: int
    branch_id: str
    lane: str
    stage: str
    shard_ids: List[str]
    scoring_checkpoint: str
    proxy_version: str
    opus_score: float
    gradient_norm: float
    candidate_loss: float
    status: str
    reason: str
    protected_floor_override: bool = False
    effective_tokens: int = 0
    pass_number: int = 0
    threshold: float = 0.0

    def as_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "step": self.step,
            "branch_id": self.branch_id,
            "lane": self.lane,
            "curriculum_stage": self.stage,
            "shard_ids": list(self.shard_ids),
            "scoring_checkpoint": self.scoring_checkpoint,
            "proxy_version": self.proxy_version,
            "opus_score": self.opus_score,
            "gradient_norm": self.gradient_norm,
            "candidate_loss": self.candidate_loss,
            "status": self.status,
            "reason": self.reason,
            "protected_floor_override": self.protected_floor_override,
            "effective_token_estimate": self.effective_tokens,
            "repeated_pass_number": self.pass_number,
            "threshold": self.threshold,
        }


    @staticmethod
    def from_dict(d: dict) -> "OpusDecision":
        """Rebuild a decision from its ledger record.

        Used by the demo driver and the evidence verifier, which read the OPUS
        ledger from disk rather than receiving decisions from the process that
        made them.
        """
        return OpusDecision(
            decision_id=d["decision_id"],
            candidate_id=d["candidate_id"],
            step=d["step"],
            branch_id=d["branch_id"],
            lane=d["lane"],
            stage=d.get("curriculum_stage", ""),
            shard_ids=list(d.get("shard_ids", [])),
            scoring_checkpoint=d.get("scoring_checkpoint", ""),
            proxy_version=d.get("proxy_version", ""),
            opus_score=d.get("opus_score", 0.0),
            gradient_norm=d.get("gradient_norm", 0.0),
            candidate_loss=d.get("candidate_loss", 0.0),
            status=d["status"],
            reason=d.get("reason", ""),
            protected_floor_override=bool(d.get("protected_floor_override", False)),
            effective_tokens=d.get("effective_token_estimate", 0),
            pass_number=d.get("repeated_pass_number", 0),
            threshold=d.get("threshold", 0.0),
        )


@dataclass
class RoundOutcome:
    accepted: Dict[str, List[str]] = field(default_factory=dict)
    decisions: List[OpusDecision] = field(default_factory=list)
    decision_id_by_sample: Dict[str, str] = field(default_factory=dict)

    @property
    def counts(self) -> Dict[str, int]:
        out = {STATUS_ACCEPTED: 0, STATUS_REJECTED: 0, STATUS_DEFERRED: 0}
        for decision in self.decisions:
            out[decision.status] = out.get(decision.status, 0) + 1
        return out


class OpusSelector:
    def __init__(self, proxy, sample_store, stage_order=None):
        self.proxy = proxy
        self.store = sample_store
        self.stage_order = stage_order or STAGE_ORDER
        self.history: List[dict] = []

    # -- helpers ----------------------------------------------------------

    def _stage_matches(self, sample_stage_hint: str, current_stage: str) -> bool:
        """A shard hinted for a later phase should not be spent now."""
        hint = self.stage_order.get(sample_stage_hint, 0)
        if "anneal" in current_stage:
            current = 2
        elif "midtrain" in current_stage or "mid" in current_stage:
            current = 1
        else:
            current = 0
        return hint <= current

    # -- the selection ----------------------------------------------------

    def select_step(
        self,
        step_plan,
        batch_provider,
        checkpoint_id: str,
    ) -> RoundOutcome:
        """Run selection for one step across every lane it schedules."""
        outcome = RoundOutcome()
        scored_all: List[dict] = []

        for lane, pool in sorted(step_plan.candidate_pools.items()):
            quota = step_plan.quotas.get(lane, 0)
            if quota <= 0 or not pool:
                continue

            scored: List[dict] = []
            for candidate_id in pool:
                sample = self.store.get(candidate_id)
                tensors = batch_provider([sample], truncate=self.proxy.probe_tokens)
                metrics = self.proxy.score(
                    tensors["token_ids"], tensors["position_ids"],
                    tensors["segment_ids"], tensors["loss_mask"],
                )
                scored.append({"sample": sample, "candidate_id": candidate_id, **metrics})
            scored_all.extend(scored)

            threshold = _median([entry["opus_score"] for entry in scored])

            # Deterministic ordering: score first, then id.  Rounding the score
            # for ranking keeps a knife-edge tie from flipping between the
            # original run and the post-crash re-run.
            ranked = sorted(
                scored,
                key=lambda e: (-round(e["opus_score"], 6), e["candidate_id"]),
            )

            accepted_ids: List[str] = []
            accepted_hashes = set()
            preferred: List[tuple] = []      # stage-appropriate candidates
            demoted: List[tuple] = []        # usable, but only if nothing better

            seen_hashes = set()
            for entry in ranked:
                sample = entry["sample"]
                content = sample.content_hash()

                # An exact duplicate window inside one step teaches nothing the
                # first copy did not, so this is a hard reject.
                if content in seen_hashes:
                    outcome.decisions.append(
                        self._decision(entry, step_plan, lane, checkpoint_id,
                                       STATUS_REJECTED, REASON_DUPLICATE, threshold)
                    )
                    continue
                seen_hashes.add(content)

                if self._stage_matches(sample.stage_hint, step_plan.stage):
                    preferred.append((entry, None))
                elif lane in PROTECTED_LANES:
                    # A protected lane cannot afford to be picky about staging;
                    # record that the preference was overridden rather than
                    # letting the lane fall below its floor.
                    demoted.append((entry, REASON_PROTECTED_LANE_BIAS))
                else:
                    # Demoted, not rejected.  Rejecting outright can empty a
                    # lane whose whole pool is hinted for a later phase, and an
                    # empty lane cannot fill a fixed batch geometry.  These are
                    # used only if the stage-appropriate candidates run out, and
                    # the record says which happened.
                    demoted.append((entry, REASON_STAGE_MISMATCH))

            pending = preferred + demoted

            # `pending` is consumed by index rather than by value: entries hold
            # dicts, and list.remove would match on equality, so two candidates
            # with identical metrics could drop the wrong one.
            protected = lane in PROTECTED_LANES
            consumed = [False] * len(pending)

            # pass 1: accept stage-appropriate candidates that clear the bar
            for index, (entry, note) in enumerate(pending):
                if len(accepted_ids) >= quota:
                    break
                if note is not None or entry["opus_score"] < threshold:
                    continue
                consumed[index] = True
                accepted_ids.append(entry["candidate_id"])
                accepted_hashes.add(entry["sample"].content_hash())
                outcome.decisions.append(
                    self._decision(entry, step_plan, lane, checkpoint_id,
                                   STATUS_ACCEPTED, REASON_ACCEPTED_ON_MERIT, threshold)
                )

            # pass 2: fill the geometry, recording why it had to be filled.
            # A protected lane taking a sub-threshold candidate is a floor
            # override; an unprotected one is ordinary quota pressure; a
            # demoted candidate keeps the reason it was demoted for.
            for index, (entry, note) in enumerate(pending):
                if len(accepted_ids) >= quota:
                    break
                if consumed[index]:
                    continue
                consumed[index] = True
                reason = note or (
                    REASON_BELOW_THRESHOLD if protected else REASON_QUOTA_PRESSURE
                )
                accepted_ids.append(entry["candidate_id"])
                accepted_hashes.add(entry["sample"].content_hash())
                outcome.decisions.append(
                    self._decision(
                        entry, step_plan, lane, checkpoint_id, STATUS_ACCEPTED,
                        reason, threshold,
                        override=protected or note == REASON_PROTECTED_LANE_BIAS,
                    )
                )

            # pass 3: nothing is deleted.  A stage-mismatched candidate that was
            # not needed is a rejection with a reason; anything else is deferred,
            # because a batch that is low value now may be right for a later phase.
            for index, (entry, note) in enumerate(pending):
                if consumed[index]:
                    continue
                if note == REASON_STAGE_MISMATCH:
                    outcome.decisions.append(
                        self._decision(entry, step_plan, lane, checkpoint_id,
                                       STATUS_REJECTED, REASON_STAGE_MISMATCH, threshold)
                    )
                else:
                    outcome.decisions.append(
                        self._decision(entry, step_plan, lane, checkpoint_id,
                                       STATUS_DEFERRED, REASON_BELOW_THRESHOLD, threshold)
                    )

            # The batch geometry is fixed, so a lane that still cannot fill its
            # quota repeats its own accepted candidates in order rather than
            # shrinking the batch.
            if accepted_ids and len(accepted_ids) < quota:
                base = list(accepted_ids)
                cursor = 0
                while len(accepted_ids) < quota:
                    accepted_ids.append(base[cursor % len(base)])
                    cursor += 1

            outcome.accepted[lane] = accepted_ids

        for decision in outcome.decisions:
            if decision.status == STATUS_ACCEPTED:
                outcome.decision_id_by_sample[decision.candidate_id] = decision.decision_id

        self.history.append(
            {
                "step": step_plan.step,
                "scoring_checkpoint": checkpoint_id,
                "candidates": len(scored_all),
                "mean_score": round(_mean([e["opus_score"] for e in scored_all]), 6),
                "mean_gradient_norm": round(
                    _mean([e["gradient_norm"] for e in scored_all]), 6
                ),
                "accepted_mean_gradient_norm": round(
                    _mean(
                        [
                            d.gradient_norm
                            for d in outcome.decisions
                            if d.status == STATUS_ACCEPTED
                        ]
                    ),
                    6,
                ),
                **outcome.counts,
            }
        )
        return outcome

    def _decision(
        self, entry, step_plan, lane, checkpoint_id, status, reason, threshold,
        override: bool = False,
    ) -> OpusDecision:
        sample = entry["sample"]
        decision_id = "opd-" + short_hash(
            {
                "step": step_plan.step,
                "branch": step_plan.branch_id,
                "candidate": entry["candidate_id"],
                "checkpoint": checkpoint_id,
            }
        )[:12]
        return OpusDecision(
            decision_id=decision_id,
            candidate_id=entry["candidate_id"],
            step=step_plan.step,
            branch_id=step_plan.branch_id,
            lane=lane,
            stage=step_plan.stage,
            shard_ids=sample.shard_ids,
            scoring_checkpoint=checkpoint_id,
            proxy_version=self.proxy.direction.version if self.proxy.direction else "none",
            opus_score=entry["opus_score"],
            gradient_norm=entry["gradient_norm"],
            candidate_loss=entry["candidate_loss"],
            status=status,
            reason=reason,
            protected_floor_override=override,
            effective_tokens=sample.loss_bearing_count,
            pass_number=step_plan.pass_numbers.get(entry["candidate_id"], 0),
            threshold=round(threshold, 6),
        )

    # -- proxy health -----------------------------------------------------

    def proxy_health(self) -> dict:
        """Is the selector picking good data, or the best of a bad pool?

        The design's diagnostic: watch the mean gradient norm of the accepted
        set across rounds.  If it collapses while the acceptance rate holds
        steady, OPUS is still filling its quota but there is nothing good left
        to fill it with - which is a statement about the corpus, not about the
        selector, and it is only visible if the numbers were recorded.
        """
        if not self.history:
            return {"rounds": 0, "verdict": "no rounds recorded"}
        norms = [h["accepted_mean_gradient_norm"] for h in self.history]
        first, last = norms[0], norms[-1]
        ratio = (last / first) if first else 0.0
        if ratio < 0.35:
            verdict = (
                "accepted gradient norms collapsed - the pool is likely exhausted, "
                "OPUS is selecting the best of poor candidates"
            )
        elif ratio > 2.5:
            verdict = (
                "accepted gradient norms rose sharply - check for a distribution "
                "shift or an unstable shard entering the stream"
            )
        else:
            verdict = "accepted gradient norms stable across rounds"
        return {
            "rounds": len(self.history),
            "first_round_accepted_grad_norm": first,
            "last_round_accepted_grad_norm": last,
            "ratio_last_over_first": round(ratio, 4),
            "verdict": verdict,
            "per_round": self.history,
        }


def decisions_report(decisions: Sequence[OpusDecision]) -> dict:
    by_status: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    by_lane: Dict[str, Dict[str, int]] = {}
    overrides: List[dict] = []

    for decision in decisions:
        by_status[decision.status] = by_status.get(decision.status, 0) + 1
        by_reason[decision.reason] = by_reason.get(decision.reason, 0) + 1
        lane = by_lane.setdefault(decision.lane, {})
        lane[decision.status] = lane.get(decision.status, 0) + 1
        if decision.protected_floor_override:
            overrides.append(decision.as_dict())

    total = len(decisions)
    accepted = by_status.get(STATUS_ACCEPTED, 0)
    return {
        "total_candidates_scored": total,
        "by_status": dict(sorted(by_status.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "by_lane": {k: dict(sorted(v.items())) for k, v in sorted(by_lane.items())},
        "acceptance_rate": round(accepted / total, 4) if total else 0.0,
        "protected_floor_overrides": len(overrides),
        "protected_floor_override_records": overrides,
        "reason_vocabulary": list(ALL_REASONS),
        "statuses_reconcile": (
            by_status.get(STATUS_ACCEPTED, 0)
            + by_status.get(STATUS_REJECTED, 0)
            + by_status.get(STATUS_DEFERRED, 0)
        ) == total,
    }


def _mean(values: Sequence[float]) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0
