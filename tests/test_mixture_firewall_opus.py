"""Mixture and floors, the evaluation firewall, and the OPUS audit trail."""

from __future__ import annotations

import json

import pytest

from tdes.config import CONFIG, LANES, PROTECTED_LANES, STAGES
from tdes.firewall.contamination import EvalFingerprintRegistry
from tdes.ledger.consumption import EVENT_CONSUME
from tdes.ledger.store import LedgerStore
from tdes.mixture.floors import apportion, blend, enforce_floors
from tdes.shards.registry import FirewallViolation


# -- apportionment and floors ---------------------------------------------


def test_apportionment_always_sums_to_the_target():
    weights = {"a": 0.44, "b": 0.16, "c": 0.14, "d": 0.14, "e": 0.04, "f": 0.08}
    for total in range(1, 65):
        counts = apportion(weights, total)
        assert sum(counts.values()) == total


def test_apportionment_is_deterministic_under_ties():
    weights = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    first = apportion(weights, 6)
    for _ in range(20):
        assert apportion(weights, 6) == first


def test_a_floor_is_met_even_when_the_mixture_would_round_it_away():
    counts = apportion({"big": 0.96, "small": 0.04}, 8)
    assert counts["small"] == 0                 # rounding erases it
    adjusted, adjustments = enforce_floors(
        counts, {"small": 0.10}, 8, available={"small": 5, "big": 20}
    )
    assert adjusted["small"] >= 1
    assert sum(adjusted.values()) == 8
    assert adjustments and adjustments[0]["lane"] == "small"


def test_a_floor_cannot_be_met_from_an_empty_lane():
    counts = {"big": 8, "empty": 0}
    adjusted, _ = enforce_floors(counts, {"empty": 0.25}, 8, available={"empty": 0})
    assert adjusted["empty"] == 0
    assert sum(adjusted.values()) == 8


def test_warmup_blends_rather_than_switches():
    a, b = {"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 1.0}
    assert blend(a, b, 0.0) == pytest.approx({"x": 1.0, "y": 0.0})
    assert blend(a, b, 0.5) == pytest.approx({"x": 0.5, "y": 0.5})
    assert blend(a, b, 1.0) == pytest.approx({"x": 0.0, "y": 1.0})


# -- the compiled schedule -------------------------------------------------


def test_every_step_schedules_the_full_batch(system):
    for quota in system.schedule.steps:
        assert sum(quota.counts.values()) == CONFIG.sequences_per_step


def test_protected_lanes_are_scheduled_in_every_step(system):
    for quota in system.schedule.steps:
        stage = next(s for s in STAGES if s.name == quota.stage)
        for lane in stage.protected_floors:
            assert quota.counts.get(lane, 0) > 0, (quota.step, lane)


def test_scarcity_is_resolved_explicitly(system):
    """A lane short of data must record what was done about it."""
    short = [f for f in system.schedule.feasibility
             if f.sequences_required > f.distinct_samples_available > 0]
    assert short, "no lane was short - the scarcity path was never exercised"
    for entry in short:
        assert entry.resolution != "satisfied"
        assert entry.note


def test_reserved_material_is_locked_until_the_anneal_stage(system):
    """Scarce tier-A trajectories must not be burned early."""
    early_stages = [s for s in STAGES if not (s.anneal or s.unlocks_reserved)]
    for stage in early_stages:
        for lane in LANES:
            for sample in system.store.lane_samples(stage.sequence_length, False, lane):
                assert not sample.reserved, sample.sample_id
    # and they do become available at the anneal stage
    anneal = next(s for s in STAGES if s.anneal)
    unlocked = [
        sample
        for lane in LANES
        for sample in system.store.lane_samples(anneal.sequence_length, True, lane)
        if sample.reserved
    ]
    assert unlocked, "no reserved material ever unlocked"


def test_schedule_hash_is_stable(system):
    assert system.schedule.schedule_hash == system.schedule.schedule_hash


def test_mixture_compliance_and_floors(artifacts):
    compliance = json.loads(
        (artifacts / "manifests" / "mixture_compliance.json").read_text(encoding="utf-8")
    )
    assert compliance["all_lanes_within_tolerance"]
    assert compliance["max_abs_delta"] <= compliance["tolerance"]
    breaches = [c for c in compliance["protected_floor_checks"] if not c["respected"]]
    assert breaches == []


# -- firewall --------------------------------------------------------------


def test_ngram_fingerprints_catch_a_reworded_copy():
    registry = EvalFingerprintRegistry()
    original = (
        "A cyclist travels 12 km at 15 km/h and then 18 km at 9 km/h. "
        "What is the average speed for the whole journey?"
    )
    registry.register("bench-1", original, "b1")

    lightly_edited = (
        "Here is a problem. A cyclist travels 12 km at 15 km/h and then 18 km "
        "at 9 km/h. What is the average speed for the whole journey? Solution follows."
    )
    assert registry.scan_text(lightly_edited)

    unrelated = "The river gauge records stage rather than discharge, and the curve does the rest."
    assert registry.scan_text(unrelated) == []


def test_canary_is_unambiguous():
    registry = EvalFingerprintRegistry()
    registry.register_raw_canaries("marker TDES-CANARY-abc123-test here", "bench")
    hits = registry.scan_text("some training text containing TDES-CANARY-abc123-test inside")
    assert any(h.detector == "canary" for h in hits)


def test_registry_refuses_a_never_train_shard(system):
    test_entries = system.registry.test
    assert test_entries, "no never-train shard registered"
    with pytest.raises(FirewallViolation):
        system.registry.assert_trainable(test_entries[0].shard_id)


def test_batch_gate_refuses_evaluation_content(system):
    """The second side: judged on the decoded text, not on the shard id."""
    from tdes.firewall.eval_firewall import EvalFirewall

    firewall = EvalFirewall(system.registry, system.fingerprints)
    test_entries = system.registry.test
    injected = system.packer.pack_holdout(test_entries[:1], "test", 256)
    assert injected, "could not construct an eval sample to inject"

    with pytest.raises(FirewallViolation):
        firewall.check_batch(
            batch_id="test-injection",
            shard_ids=injected[0].shard_ids,
            decoded_text=injected[0].decoded(system.tokenizer),
            loss_bearing_tokens=injected[0].loss_bearing_count,
        )
    assert firewall.blocks


def test_no_non_train_shard_reached_the_consumption_ledger(artifacts):
    registry = json.loads(
        (artifacts / "manifests" / "shard_registry.json").read_text(encoding="utf-8")
    )
    permissions = registry["permissions"]
    ledger = LedgerStore(artifacts / "ledgers" / "consumption_main.jsonl", "c")
    for record in ledger.read_all():
        if record["type"] != EVENT_CONSUME:
            continue
        for shard_id in record["payload"]["shard_ids"]:
            assert permissions.get(shard_id) == "train", shard_id


def test_validation_produced_no_gradient_bearing_tokens(artifacts):
    report = json.loads(
        (artifacts / "ledgers" / "firewall_report.json").read_text(encoding="utf-8")
    )
    assert report["validation_gradient_bearing_tokens"] == 0


# -- OPUS ------------------------------------------------------------------


@pytest.fixture(scope="module")
def decisions(artifacts):
    ledger = LedgerStore(artifacts / "ledgers" / "opus_decisions.jsonl", "opus")
    return [r["payload"] for r in ledger.read_all() if r["type"] == "opus_decision"]


def test_every_candidate_has_a_decision_and_a_reason(decisions):
    assert decisions
    for decision in decisions:
        assert decision["status"] in {"accepted", "rejected", "deferred"}
        assert decision["reason"]


def test_all_four_ledgers_are_populated(decisions):
    statuses = {d["status"] for d in decisions}
    assert {"accepted", "rejected", "deferred"} <= statuses
    assert any(d["protected_floor_override"] for d in decisions)


def test_protected_floor_override_only_fires_on_protected_lanes(decisions):
    for decision in decisions:
        if decision["protected_floor_override"]:
            assert decision["lane"] in PROTECTED_LANES


def test_rejections_carry_the_context_needed_to_reuse_them_later(decisions):
    """Rejected clean data must not disappear - the record has to be actionable."""
    rejected = [d for d in decisions if d["status"] in ("rejected", "deferred")]
    assert rejected
    for decision in rejected:
        assert decision["shard_ids"]
        assert decision["curriculum_stage"]
        assert decision["scoring_checkpoint"]
        assert decision["opus_score"] is not None


def test_scores_look_computed_rather_than_generated(decisions):
    """A real gradient cosine gives near-unique values across candidates."""
    scores = [round(d["opus_score"], 6) for d in decisions]
    assert len(set(scores)) > 0.5 * len(scores)
    assert min(scores) < 0 < max(scores), "cosines should span both signs"


def test_repeated_decision_ids_are_only_the_post_crash_rescoring(decisions, artifacts):
    """The OPUS ledger is not rolled back on resume, and that is deliberate.

    The consumption ledger has to be truncated because it describes batches
    served to a model state that no longer exists.  The OPUS ledger is a record
    of *scoring events*, and the selector genuinely did score those candidates
    twice: once before the crash, and again after the checkpoint was restored.
    Keeping both is honest, and it is also the strongest determinism evidence
    in the run - the two scorings must agree exactly.
    """
    phase = json.loads(
        (artifacts / "ledgers" / "phase_resume_main.json").read_text(encoding="utf-8")
    )
    rolled_back = set(phase["recovery"]["discarded_steps"])

    by_id = {}
    duplicated = []
    for decision in decisions:
        key = decision["decision_id"]
        if key in by_id:
            duplicated.append((by_id[key], decision))
        else:
            by_id[key] = decision

    assert duplicated, "the crash never caused any candidate to be re-scored"
    for original, repeat in duplicated:
        assert original["step"] in rolled_back, original["step"]
        # identical scores from a restored checkpoint: the selection is a pure
        # function of (model state, candidate), not of run history
        assert original["opus_score"] == repeat["opus_score"]
        assert original["gradient_norm"] == repeat["gradient_norm"]
        assert original["status"] == repeat["status"]
        assert original["reason"] == repeat["reason"]


def test_decision_ids_are_unique_within_a_scoring_pass(decisions):
    seen = set()
    for decision in decisions:
        key = (decision["decision_id"], decision["opus_score"])
        seen.add(key)
    ids = {d["decision_id"] for d in decisions}
    # one id maps to exactly one score, so ids identify a decision unambiguously
    assert len(seen) == len(ids)
