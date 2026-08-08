"""The learning trace, the evidence bundle, and whether the auditor has teeth."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from tdes.audit.evidence import REQUIRED_ROWS, run_checks
from tdes.cli.verify_evidence import verify
from tdes.config import CONFIG
from tdes.ledger.consumption import EVENT_CONSUME, integrity_report
from tdes.ledger.store import LedgerStore
from tdes.training.model import expected_initial_loss

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


# -- learning trace --------------------------------------------------------


@pytest.fixture(scope="module")
def learning(artifacts):
    ledger = LedgerStore(artifacts / "ledgers" / "learning_main.jsonl", "learning")
    return ledger.read_all()


def test_every_loss_record_names_its_source_data(learning):
    samples = [r["payload"] for r in learning if r["type"] == "sample_learning"]
    assert samples
    for sample in samples:
        assert sample["shard_ids"], sample["sample_id"]
        assert sample["doc_ids"], sample["sample_id"]
        assert sample["curriculum_stage"]
        assert sample["model_phase"]


def test_loss_delta_is_measured_across_the_update(learning):
    samples = [r["payload"] for r in learning if r["type"] == "sample_learning"]
    for sample in samples:
        # every operand is stored rounded to 6dp, so allow one unit of that
        assert sample["loss_delta"] == pytest.approx(
            sample["loss_after"] - sample["loss_before"], abs=2e-6
        )
    # and at least some exposures actually reduced loss
    assert any(s["loss_delta"] < 0 for s in samples)


def test_token_trace_carries_the_full_record(learning):
    traces = [r["payload"] for r in learning if r["type"] == "token_trace"]
    assert traces, "no token-level trace was written"
    required = {
        "token_id", "preview", "position_in_sequence", "position_in_segment",
        "doc_id", "shard_id", "lang", "script", "lane", "is_special", "is_eos",
        "loss_mask", "cross_entropy", "perplexity",
    }
    for trace in traces[:5]:
        assert trace["tokens"]
        for token in trace["tokens"][:5]:
            assert required.issubset(token)


def test_perplexity_is_the_exponential_of_the_loss(learning):
    traces = [r["payload"] for r in learning if r["type"] == "token_trace"]
    for trace in traces[:3]:
        for token in trace["tokens"][:20]:
            assert token["perplexity"] == pytest.approx(
                math.exp(min(token["cross_entropy"], 20.0)), rel=1e-4
            )


def test_token_trace_is_confined_to_the_configured_interval(learning):
    traces = [r["payload"] for r in learning if r["type"] == "token_trace"]
    low, high = CONFIG.token_trace_interval
    for trace in traces:
        assert low <= trace["step"] < high


def test_eos_perplexity_is_tracked_separately(artifacts):
    aggregates = json.loads(
        (artifacts / "ledgers" / "learning_aggregates.json").read_text(encoding="utf-8")
    )
    eos = aggregates["eos_perplexity"]
    assert eos["samples"] > 0
    assert eos["mean"] is not None


def test_shard_report_cards_classify_every_shard_seen(artifacts):
    recommendations = json.loads(
        (artifacts / "ledgers" / "next_corpus_recommendations.json").read_text(encoding="utf-8")
    )
    cards = recommendations["shard_cards"]
    assert cards
    for card in cards:
        assert card["classification"] in {"useful", "neutral", "harmful", "exhausted"}
        assert card["rationale"]
        assert card["exposures"] > 0
    assert any(recommendations["actions"].values())


def test_validation_loss_was_recorded_without_gradients(learning):
    validations = [r["payload"] for r in learning if r["type"] == "validation_loss"]
    assert validations
    for record in validations:
        assert record["gradient_bearing"] is False
        assert record["tokens_evaluated"] > 0


def test_initial_loss_started_near_ln_vocab(artifacts):
    """The free sanity check: labels shifted right, mask not inverted."""
    log = (artifacts / "run.log").read_text(encoding="utf-8")
    assert "[PASS] initial_loss_matches_uniform_prior" in log

    tokenizer = json.loads(
        (artifacts / "manifests" / "tokenizer.json").read_text(encoding="utf-8")
    )
    expected = expected_initial_loss(tokenizer["vocab_size"])

    ledger = LedgerStore(artifacts / "ledgers" / "consumption_main.jsonl", "c")
    steps = sorted(
        (r["payload"] for r in ledger.read_all() if r["type"] == "optimizer_step"),
        key=lambda p: p["global_step"],
    )
    assert abs(steps[0]["mean_loss"] - expected) < 0.6


def test_loss_fell_over_the_run(artifacts):
    ledger = LedgerStore(artifacts / "ledgers" / "consumption_main.jsonl", "c")
    steps = sorted(
        (r["payload"] for r in ledger.read_all() if r["type"] == "optimizer_step"),
        key=lambda p: p["global_step"],
    )
    assert len(steps) == CONFIG.total_steps
    assert steps[-1]["mean_loss"] < steps[0]["mean_loss"]


# -- throughput ------------------------------------------------------------


def test_reported_throughput_is_reconstructible(artifacts):
    """The assignment's rule: numbers that cannot be rebuilt earn no credit."""
    performance = json.loads((artifacts / "performance.json").read_text(encoding="utf-8"))
    ledger = LedgerStore(artifacts / "ledgers" / "consumption_main.jsonl", "c")
    records = [r for r in ledger.read_all() if r["type"] == EVENT_CONSUME]
    integrity = integrity_report(records, "main")

    assert performance["efficiency"]["packing_utilisation"] == pytest.approx(
        integrity["packing_utilisation"], abs=0.05
    )
    assert performance["efficiency"]["loss_density"] == pytest.approx(
        integrity["loss_density"], abs=0.05
    )
    assert performance["throughput"]["useful_tokens_per_sec_compute"] > 0
    # loss density can never exceed utilisation: graded tokens are a subset
    assert (
        performance["efficiency"]["loss_density"]
        <= performance["efficiency"]["packing_utilisation"] + 1e-9
    )


# -- evidence bundle -------------------------------------------------------


def test_every_required_row_is_present(evidence):
    names = {r["requirement"] for r in evidence["requirements"]}
    for row in REQUIRED_ROWS:
        assert row in names, row


def test_all_requirements_passed(evidence):
    failing = [r["requirement"] for r in evidence["requirements"] if r["result"] != "PASS"]
    assert failing == []
    assert evidence["all_passed"]


def test_every_row_points_at_a_real_artifact(artifacts, evidence):
    for requirement in evidence["requirements"]:
        assert requirement["evidence"], requirement["requirement"]
        for pointer in requirement["evidence"]:
            if "*" in pointer:
                assert list(artifacts.glob(pointer)), pointer
            else:
                assert (artifacts / pointer).exists(), pointer


def test_markdown_agrees_with_the_json(artifacts, evidence):
    markdown = (artifacts / "evidence.md").read_text(encoding="utf-8")
    for requirement in evidence["requirements"]:
        assert f"| {requirement['requirement']} | {requirement['result']} |" in markdown


def test_checks_recompute_to_the_same_verdicts(artifacts, evidence):
    """Re-running the checks must reproduce the bundle exactly."""
    recomputed = {r.name: ("PASS" if r.passed else "FAIL") for r in run_checks(artifacts)}
    claimed = {r["requirement"]: r["result"] for r in evidence["requirements"]}
    assert recomputed == claimed


def test_the_verifier_passes_on_a_clean_run(artifacts):
    assert verify(artifacts, verbose=False) == 0


def test_verifier_runs_as_a_standalone_program(artifacts):
    result = subprocess.run(
        [sys.executable, "-m", "tdes.cli.verify_evidence", "--quiet"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# -- the mutation test: does the auditor actually bite? --------------------


def test_verifier_detects_a_corrupted_ledger(artifact_copy):
    """An auditor that can never fail is not evidence of anything."""
    path = artifact_copy / "ledgers" / "consumption_main.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    target = next(i for i, line in enumerate(lines) if '"global_step":3' in line)
    lines[target] = lines[target].replace('"global_step":3', '"global_step":999')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert verify(artifact_copy, verbose=False) != 0


def test_verifier_detects_a_doctored_evidence_verdict(artifact_copy):
    evidence_path = artifact_copy / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    # flip a genuine FAIL into a claimed PASS - the classic forgery
    evidence["requirements"][0]["result"] = "FAIL"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    assert verify(artifact_copy, verbose=False) != 0


def test_verifier_detects_a_missing_artifact(artifact_copy):
    (artifact_copy / "performance.json").unlink()
    assert verify(artifact_copy, verbose=False) != 0


# -- the assignment's completion criterion ---------------------------------


@pytest.fixture(scope="module")
def completion(artifacts):
    return json.loads(
        (artifacts / "ledgers" / "completion_criterion.json").read_text(encoding="utf-8")
    )


def test_all_four_clauses_are_proved(completion):
    assert completion["all_four_clauses_proved"]
    for name, clause in completion["clauses"].items():
        assert clause["proved"], (name, clause)


def test_the_walk_covered_every_consumed_sample(artifacts, completion):
    """The criterion is a link walk, so it must have walked everything."""
    from tdes.ledger.consumption import EVENT_CONSUME

    ledger = LedgerStore(artifacts / "ledgers" / "consumption_main.jsonl", "c")
    expected = sum(
        len(r["payload"]["packed_sample_ids"])
        for r in ledger.read_all()
        if r["type"] == EVENT_CONSUME and r["payload"]["branch_id"] == "main"
    )
    assert completion["consumed_sample_instances"] == expected
    assert expected > 0


def test_no_dangling_references_between_ledgers(completion):
    c = completion["clauses"]
    assert c["why_it_consumed_it"]["instances_without_a_resolvable_decision"] == 0
    assert c["why_it_consumed_it"]["decisions_without_a_reason"] == 0
    assert c["what_the_model_learned"]["instances_without_a_learning_record"] == 0
    assert c["what_the_model_learned"]["shards_without_a_report_card"] == 0
    assert c["how_the_run_can_be_reconstructed"]["records_missing_provenance_fields"] == 0


def test_the_worked_example_is_complete_and_consistent(artifacts, completion):
    """A reader should be able to follow the example by hand and have it check out."""
    ex = completion["worked_example"]
    assert ex, "no worked example recorded"

    # the OPUS decision the example names must exist and agree
    opus = LedgerStore(artifacts / "ledgers" / "opus_decisions.jsonl", "o")
    decision = next(
        (r["payload"] for r in opus.read_all()
         if r["payload"]["decision_id"] == ex["why"]["opus_decision_id"]), None)
    assert decision is not None
    assert decision["status"] == ex["why"]["status"]
    assert decision["reason"] == ex["why"]["reason"]

    # the learning record must exist and its delta must be internally consistent
    learn = LedgerStore(artifacts / "ledgers" / "learning_main.jsonl", "l")
    record = next(
        (r["payload"] for r in learn.read_all()
         if r["type"] == "sample_learning"
         and r["payload"]["sample_id"] == ex["consumed"]["sample_id"]
         and r["payload"]["step"] == ex["consumed"]["step"]), None)
    assert record is not None
    assert record["loss_delta"] == pytest.approx(
        ex["learned"]["loss_after"] - ex["learned"]["loss_before"], abs=2e-6)

    # the token span must name a shard that exists and parse as a range
    shard, _, span = ex["consumed"]["token_span"].partition(":")
    lo, _, hi = span.partition("-")
    assert shard == ex["consumed"]["shard_id"]
    assert int(hi) > int(lo)

    # everything needed to reconstruct must be present
    for field in ("plan_hash", "batch_id", "rng_fingerprint", "tokenizer_version",
                  "dataloader_version", "config_hash"):
        assert ex["reconstructable_from"][field], field


def test_completion_criterion_is_an_evidence_row(evidence):
    row = next((r for r in evidence["requirements"]
                if r["requirement"] == "Completion criterion"), None)
    assert row is not None
    assert row["result"] == "PASS"
