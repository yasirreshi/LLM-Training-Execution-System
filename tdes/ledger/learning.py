"""The learning ledger: what the model got back from the data.

The consumption ledger records what the model saw.  This one attaches the
outcome to it, which is the half that almost never gets written down and the
half matters most, because it cannot be recovered afterwards without re-running the
same model over the same data at the same training state.

Storage is tiered, as section 11 describes, because a full token trace of a real
run is not affordable:

*   **full token trace** for one configured interval of steps - token id,
    decoded preview, position, document, shard, language, lane, special/EOS
    flags, cross-entropy, perplexity, model phase, checkpoint before and after,
    OPUS decision, repeated-pass number;
*   **per-sample records** for every step - mean loss, loss delta across the
    update, gradient norm;
*   **aggregates** for the whole run - by shard, lane, language and phase.

From those, `shard_report_cards` produces the thing the next corpus actually needs: for each
shard, did exposure to it help, and should the next corpus collect more of it,
protect it, repeat it, defer it or drop it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..config import CONFIG
from .store import LedgerStore

EVENT_SAMPLE = "sample_learning"
EVENT_TOKEN_TRACE = "token_trace"
EVENT_SHARD_CARD = "shard_report_card"
EVENT_VALIDATION = "validation_loss"

CLASS_USEFUL = "useful"
CLASS_NEUTRAL = "neutral"
CLASS_HARMFUL = "harmful"
CLASS_EXHAUSTED = "exhausted"


@dataclass
class SampleLearning:
    step: int
    branch_id: str
    sample_id: str
    lane: str
    stage: str
    shard_ids: List[str]
    doc_ids: List[str]
    loss_before: float
    loss_after: float
    grad_norm: float
    loss_bearing_tokens: int
    model_phase: str
    checkpoint_before: str
    checkpoint_after: str
    opus_decision_id: str = ""
    opus_score: Optional[float] = None
    pass_number: int = 0
    mean_token_ppl: float = 0.0
    eos_ppl: Optional[float] = None

    @property
    def loss_delta(self) -> float:
        """Negative means the update reduced loss on these very tokens."""
        return self.loss_after - self.loss_before

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "branch_id": self.branch_id,
            "sample_id": self.sample_id,
            "lane": self.lane,
            "curriculum_stage": self.stage,
            "shard_ids": list(self.shard_ids),
            "doc_ids": list(self.doc_ids),
            "loss_before": round(self.loss_before, 6),
            "loss_after": round(self.loss_after, 6),
            "loss_delta": round(self.loss_delta, 6),
            "gradient_norm": round(self.grad_norm, 6),
            "loss_bearing_tokens": self.loss_bearing_tokens,
            "mean_token_perplexity": round(self.mean_token_ppl, 6),
            "eos_perplexity": round(self.eos_ppl, 6) if self.eos_ppl is not None else None,
            "model_phase": self.model_phase,
            "checkpoint_before": self.checkpoint_before,
            "checkpoint_after": self.checkpoint_after,
            "opus_decision_id": self.opus_decision_id,
            "opus_score": self.opus_score,
            "repeated_pass_number": self.pass_number,
        }


class LearningLedger:
    def __init__(self, store: LedgerStore, run_id: str, branch_id: str):
        self.store = store
        self.run_id = run_id
        self.branch_id = branch_id
        self.samples: List[SampleLearning] = []
        self.token_records_written = 0

    # -- loading ----------------------------------------------------------

    @classmethod
    def from_store(cls, store: LedgerStore, run_id: str, branch_id: str) -> "LearningLedger":
        """Rebuild the in-memory view from the ledger file.

        The demo driver aggregates in a different process from the one that
        trained, so the report cards and next-corpus recommendations are derived from
        the durable record rather than from anything held in memory.
        """
        ledger = cls(store, run_id, branch_id)
        for record in store.read_all():
            if record["type"] == EVENT_SAMPLE:
                payload = record["payload"]
                ledger.samples.append(
                    SampleLearning(
                        step=payload["step"],
                        branch_id=payload["branch_id"],
                        sample_id=payload["sample_id"],
                        lane=payload["lane"],
                        stage=payload.get("curriculum_stage", ""),
                        shard_ids=list(payload.get("shard_ids", [])),
                        doc_ids=list(payload.get("doc_ids", [])),
                        loss_before=payload["loss_before"],
                        loss_after=payload["loss_after"],
                        grad_norm=payload.get("gradient_norm", 0.0),
                        loss_bearing_tokens=payload.get("loss_bearing_tokens", 0),
                        model_phase=payload.get("model_phase", ""),
                        checkpoint_before=payload.get("checkpoint_before", ""),
                        checkpoint_after=payload.get("checkpoint_after", ""),
                        opus_decision_id=payload.get("opus_decision_id", ""),
                        opus_score=payload.get("opus_score"),
                        pass_number=payload.get("repeated_pass_number", 0),
                        mean_token_ppl=payload.get("mean_token_perplexity", 0.0),
                        eos_ppl=payload.get("eos_perplexity"),
                    )
                )
            elif record["type"] == EVENT_TOKEN_TRACE:
                ledger.token_records_written += record["payload"].get("token_count", 0)
        return ledger

    # -- writing ----------------------------------------------------------

    def record_sample(self, record: SampleLearning) -> dict:
        self.samples.append(record)
        return self.store.append(
            EVENT_SAMPLE, {"run_id": self.run_id, **record.as_dict()}
        )

    def record_token_trace(
        self,
        *,
        step: int,
        sample_id: str,
        lane: str,
        stage: str,
        model_phase: str,
        checkpoint_before: str,
        opus_decision_id: str,
        pass_number: int,
        tokens: Sequence[dict],
    ) -> dict:
        """Write the full per-token trace for one packed sample.

        `tokens` carries one entry per loss-bearing position.  This is the
        expensive record, so it is only written for the configured interval.
        """
        self.token_records_written += len(tokens)
        return self.store.append(
            EVENT_TOKEN_TRACE,
            {
                "run_id": self.run_id,
                "branch_id": self.branch_id,
                "step": step,
                "sample_id": sample_id,
                "lane": lane,
                "curriculum_stage": stage,
                "model_phase": model_phase,
                "checkpoint_before": checkpoint_before,
                "opus_decision_id": opus_decision_id,
                "repeated_pass_number": pass_number,
                "token_count": len(tokens),
                "tokens": list(tokens),
            },
        )

    def record_validation(self, step: int, loss: float, shard_ids: Sequence[str],
                          tokens: int) -> dict:
        return self.store.append(
            EVENT_VALIDATION,
            {
                "run_id": self.run_id,
                "branch_id": self.branch_id,
                "step": step,
                "validation_loss": round(loss, 6),
                "validation_perplexity": round(math.exp(min(loss, 20.0)), 6),
                "shard_ids": sorted(shard_ids),
                "tokens_evaluated": tokens,
                "gradient_bearing": False,
            },
        )

    # -- aggregation ------------------------------------------------------

    def shard_report_cards(self) -> List[dict]:
        """Follow each shard across the phases it was seen in."""
        by_shard: Dict[str, List[SampleLearning]] = {}
        for record in self.samples:
            for shard_id in record.shard_ids:
                by_shard.setdefault(shard_id, []).append(record)

        cards: List[dict] = []
        for shard_id in sorted(by_shard):
            records = by_shard[shard_id]
            deltas = [r.loss_delta for r in records]
            ppls = [r.mean_token_ppl for r in records if r.mean_token_ppl > 0]
            norms = [r.grad_norm for r in records]
            scores = [r.opus_score for r in records if r.opus_score is not None]

            by_phase: Dict[str, List[float]] = {}
            for record in records:
                by_phase.setdefault(record.model_phase, []).append(record.loss_delta)

            by_pass: Dict[int, List[float]] = {}
            for record in records:
                by_pass.setdefault(record.pass_number, []).append(record.loss_delta)
            repeat_effect = _repeat_effect(by_pass)

            mean_ppl = _mean(ppls)
            classification, rationale = _classify(
                mean_delta=_mean(deltas),
                mean_ppl=mean_ppl,
                max_grad=max(norms) if norms else 0.0,
                mean_grad=_mean(norms),
                repeat_effect=repeat_effect,
            )

            cards.append(
                {
                    "shard_id": shard_id,
                    "exposures": len(records),
                    "lanes": sorted({r.lane for r in records}),
                    "phases_seen": sorted(by_phase),
                    "mean_loss_delta": round(_mean(deltas), 6),
                    "loss_delta_by_phase": {
                        phase: round(_mean(values), 6)
                        for phase, values in sorted(by_phase.items())
                    },
                    "mean_token_perplexity": round(mean_ppl, 6),
                    "mean_gradient_norm": round(_mean(norms), 6),
                    "max_gradient_norm": round(max(norms) if norms else 0.0, 6),
                    "mean_opus_score": round(_mean(scores), 6) if scores else None,
                    "passes_seen": sorted(by_pass),
                    "repeat_effect": round(repeat_effect, 6),
                    "classification": classification,
                    "rationale": rationale,
                }
            )
        return cards

    def aggregates(self) -> dict:
        """Whole-run statistics, the cheapest storage tier."""
        by_lane: Dict[str, List[SampleLearning]] = {}
        by_phase: Dict[str, List[SampleLearning]] = {}
        for record in self.samples:
            by_lane.setdefault(record.lane, []).append(record)
            by_phase.setdefault(record.model_phase, []).append(record)

        def block(records: Sequence[SampleLearning]) -> dict:
            return {
                "exposures": len(records),
                "mean_loss_before": round(_mean([r.loss_before for r in records]), 6),
                "mean_loss_after": round(_mean([r.loss_after for r in records]), 6),
                "mean_loss_delta": round(_mean([r.loss_delta for r in records]), 6),
                "mean_token_perplexity": round(
                    _mean([r.mean_token_ppl for r in records if r.mean_token_ppl > 0]), 6
                ),
                "mean_gradient_norm": round(_mean([r.grad_norm for r in records]), 6),
                "loss_bearing_tokens": sum(r.loss_bearing_tokens for r in records),
            }

        eos = [r.eos_ppl for r in self.samples if r.eos_ppl is not None]
        return {
            "samples_recorded": len(self.samples),
            "token_records_written": self.token_records_written,
            "by_lane": {lane: block(records) for lane, records in sorted(by_lane.items())},
            "by_phase": {phase: block(records) for phase, records in sorted(by_phase.items())},
            "eos_perplexity": {
                "samples": len(eos),
                "mean": round(_mean(eos), 6),
                "min": round(min(eos), 6) if eos else None,
                "max": round(max(eos), 6) if eos else None,
            },
        }

    # -- feedback for the next corpus -------------------------------------

    def next_corpus_recommendations(self) -> dict:
        """Turn the report cards into an instruction for the next corpus.

        This is the artifact that carries forward: this run telling the next
        what to collect, protect, repeat, defer or reject - and saying, for each
        shard, on what measurement.
        """
        cards = self.shard_report_cards()
        actions: Dict[str, List[dict]] = {
            "collect_more": [], "protect": [], "repeat": [], "defer": [], "reject": [],
        }
        for card in cards:
            classification = card["classification"]
            entry = {
                "shard_id": card["shard_id"],
                "lanes": card["lanes"],
                "mean_loss_delta": card["mean_loss_delta"],
                "mean_token_perplexity": card["mean_token_perplexity"],
                "repeat_effect": card["repeat_effect"],
                "because": card["rationale"],
            }
            if classification == CLASS_USEFUL:
                actions["collect_more"].append(entry)
                if card["mean_token_perplexity"] > 3.0:
                    actions["protect"].append(entry)
            elif classification == CLASS_NEUTRAL:
                actions["defer"].append(entry)
            elif classification == CLASS_EXHAUSTED:
                actions["reject"].append(entry)
            elif classification == CLASS_HARMFUL:
                actions["reject"].append(entry)
            if card["repeat_effect"] > 0 and len(card["passes_seen"]) > 1:
                actions["repeat"].append({**entry, "note": "further passes stopped helping"})

        return {
            "generated_from": "learning ledger sample records",
            "shard_cards": cards,
            "learned_out_ppl_threshold": CONFIG.learned_out_ppl,
            "actions": {k: v for k, v in sorted(actions.items())},
            "summary": {
                classification: sum(1 for c in cards if c["classification"] == classification)
                for classification in (CLASS_USEFUL, CLASS_NEUTRAL, CLASS_HARMFUL, CLASS_EXHAUSTED)
            },
        }


# --------------------------------------------------------------------------


def _classify(mean_delta: float, mean_ppl: float, max_grad: float,
              mean_grad: float, repeat_effect: float):
    """Useful, neutral, harmful or exhausted - with the reason.

    Thresholds come from experience:  a shard whose average perplexity has fallen to around 1.2 while
    the run's average is far higher has nothing left to teach, and training on it
    is wasted computation.  A shard that produces
    gradient spikes needs cleaning, staging or warmup before it is used again.
    """
    if mean_ppl and mean_ppl <= CONFIG.learned_out_ppl:
        return CLASS_EXHAUSTED, (
            f"mean token perplexity {mean_ppl:.3f} is at or below the "
            f"{CONFIG.learned_out_ppl} floor - the model already predicts this "
            f"content, so further exposure buys nothing"
        )
    if mean_grad > 0 and max_grad > 8.0 * mean_grad and max_grad > 1.0:
        return CLASS_HARMFUL, (
            f"gradient spiked to {max_grad:.3f} against a mean of {mean_grad:.3f} - "
            f"needs cleaning, later staging or a warmup before reuse"
        )
    if mean_delta < -1e-4:
        return CLASS_USEFUL, (
            f"exposure reduced loss on its own tokens by {abs(mean_delta):.5f} on average"
        )
    if mean_delta > 1e-4:
        return CLASS_HARMFUL, (
            f"loss on these tokens rose by {mean_delta:.5f} across the update"
        )
    return CLASS_NEUTRAL, (
        f"loss barely moved ({mean_delta:+.6f}); no measurable effect either way"
    )


def _repeat_effect(by_pass: Dict[int, List[float]]) -> float:
    """How much worse the second pass was than the first.

    Positive means repetition stopped paying: the later pass reduced loss less
    than the first one did.  That is the direct read-out of an exhausted
    repetition budget.
    """
    passes = sorted(by_pass)
    if len(passes) < 2:
        return 0.0
    first = _mean(by_pass[passes[0]])
    last = _mean(by_pass[passes[-1]])
    return last - first


def _mean(values: Sequence[float]) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0
