"""Ledger integrity, crash recovery, replay and fork.

These are the invariants the assignment's grading turns on:

    if resume repeats or skips a batch, the resume section fails
    if replay produces different batch hashes, the replay section fails
"""

from __future__ import annotations

import json

import pytest

from tdes.config import CONFIG
from tdes.ledger.consumption import EVENT_CONSUME, integrity_report
from tdes.ledger.store import GENESIS_HASH, LedgerOffset, LedgerStore


# -- the store itself ------------------------------------------------------


def test_append_builds_an_unbroken_chain(tmp_path):
    store = LedgerStore(tmp_path / "l.jsonl", "l")
    for i in range(10):
        store.append("event", {"i": i})
    ok, detail = store.verify_chain()
    assert ok and detail["records"] == 10


def test_editing_a_record_breaks_the_chain(tmp_path):
    store = LedgerStore(tmp_path / "l.jsonl", "l")
    for i in range(5):
        store.append("event", {"i": i})
    lines = store.path.read_text(encoding="utf-8").splitlines()
    lines[2] = lines[2].replace('"i":2', '"i":99')
    store.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, detail = LedgerStore(store.path, "l").verify_chain()
    assert not ok
    assert detail["error"] == "event_hash_mismatch"


def test_deleting_a_record_breaks_the_chain(tmp_path):
    store = LedgerStore(tmp_path / "l.jsonl", "l")
    for i in range(5):
        store.append("event", {"i": i})
    lines = store.path.read_text(encoding="utf-8").splitlines()
    del lines[1]
    store.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, detail = LedgerStore(store.path, "l").verify_chain()
    assert not ok
    assert detail["error"] == "seq_gap"


def test_torn_final_line_is_detected_and_repaired(tmp_path):
    store = LedgerStore(tmp_path / "l.jsonl", "l")
    for i in range(5):
        store.append("event", {"i": i})
    before = store.current_offset()

    with open(store.path, "ab") as handle:
        handle.write(b'{"seq":5,"prev_hash":"abc')

    assert len(store.read_all()) == 5           # torn line ignored, not fatal
    repair = store.repair_torn_tail()
    assert repair and repair["reason"] == "torn_final_line"
    assert store.current_offset() == before
    assert store.verify_chain()[0]


def test_rollback_returns_what_it_discarded(tmp_path):
    store = LedgerStore(tmp_path / "l.jsonl", "l")
    for i in range(5):
        store.append("event", {"i": i})
    offset = store.current_offset()
    for i in range(5, 9):
        store.append("event", {"i": i})

    discarded = store.rollback_to(offset)
    assert [r["payload"]["i"] for r in discarded] == [5, 6, 7, 8]
    assert len(store.read_all()) == 5
    assert store.verify_chain()[0]


def test_rollback_to_a_bogus_offset_raises(tmp_path):
    store = LedgerStore(tmp_path / "l.jsonl", "l")
    store.append("event", {"i": 0})
    with pytest.raises(RuntimeError):
        store.rollback_to(LedgerOffset(0, 3, GENESIS_HASH))


# -- the real run ----------------------------------------------------------


@pytest.fixture(scope="module")
def consumption(artifacts):
    return LedgerStore(artifacts / "ledgers" / "consumption_main.jsonl", "consumption")


@pytest.fixture(scope="module")
def integrity(consumption):
    records = [r for r in consumption.read_all() if r["type"] == EVENT_CONSUME]
    return integrity_report(records, "main")


def test_run_ledger_chain_is_intact(consumption):
    ok, detail = consumption.verify_chain()
    assert ok, detail


def test_no_batch_was_repeated(integrity):
    assert integrity["duplicate_microbatches"] == []


def test_no_batch_was_skipped(integrity):
    assert integrity["missing_steps"] == []
    assert integrity["step_range"] == [0, CONFIG.total_steps - 1]


def test_every_step_served_a_complete_batch(integrity):
    expected = CONFIG.world_size * CONFIG.grad_accum
    assert integrity["microbatches_per_step"] == [expected]


def test_resume_served_the_expected_next_batch(artifacts):
    phase = json.loads(
        (artifacts / "ledgers" / "phase_resume_main.json").read_text(encoding="utf-8")
    )
    verification = phase["next_batch_verification"]
    assert verification["matched"]
    assert (
        verification["expected_plan_hash_from_checkpoint"]
        == verification["recomputed_plan_hash"]
    )


def test_rolled_back_batches_were_re_served_identically(artifacts):
    """The strong form of 'no skipped or repeated batches'.

    Every record the rollback discarded must reappear with the same batch id,
    the same token spans and the same token and mask hashes.
    """
    phase = json.loads(
        (artifacts / "ledgers" / "phase_resume_main.json").read_text(encoding="utf-8")
    )
    rollback = phase["rollback_replay_verification"]
    assert rollback["compared"] > 0, "nothing was rolled back - the crash proved nothing"
    assert rollback["missing"] == []
    assert rollback["mismatches"] == []
    assert rollback["identical"]


def test_the_crash_left_the_ledger_ahead_of_the_checkpoint(artifacts):
    """Otherwise the recovery would have been trivial."""
    phase = json.loads(
        (artifacts / "ledgers" / "phase_resume_main.json").read_text(encoding="utf-8")
    )
    recovery = phase["recovery"]
    assert recovery["torn_tail_repaired"] is True
    assert recovery["discarded_records"] > 0
    assert recovery["resume_step"] < CONFIG.crash_step
    assert CONFIG.crash_step in recovery["discarded_steps"]


def test_checkpoints_carry_a_data_position(artifacts):
    directories = sorted((artifacts / "checkpoints").glob("ckpt_*"))
    assert directories
    for directory in directories:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        assert set(meta["ledger_offset"]) == {
            "byte_offset", "event_seq", "last_event_hash"
        }
        assert meta["rng_fingerprint"]
        assert meta["schedule_hash"]


def test_checkpoint_retention_keeps_metadata_for_everything(artifacts):
    """Weights may be pruned; the data position may not.

    The ledger offset is what makes a checkpoint auditable, so it survives even
    when the 16MB of weights beside it does not.
    """
    retention = json.loads(
        (artifacts / "checkpoints" / "retention.json").read_text(encoding="utf-8")
    )
    assert retention["metadata_retained_for_all"]

    # metadata survives for every checkpoint — this is the part that makes a
    # checkpoint auditable, and it is unconditional
    for entry in retention["checkpoints"]:
        assert (artifacts / "checkpoints" / entry["path"] / "meta.json").exists()

    # pruning removed the weights it said it removed
    for pruned in retention["pruned"]:
        assert not (artifacts / "checkpoints" / pruned["checkpoint"] / "state.pt").exists()

    # any weights still present must be at a retained step. `state.pt` is
    # gitignored, so a fresh clone has none until the demo runs — which is why
    # this asserts the *placement* of weights rather than their presence.
    kept = set(retention["weights_retained_at_steps"])
    for directory in sorted((artifacts / "checkpoints").glob("ckpt_main_*")):
        if (directory / "state.pt").exists():
            step = int(directory.name.rsplit("_", 1)[1])
            assert step in kept, f"weights survived at unretained step {step}"


def test_replay_matches_on_all_three_derivations(artifacts):
    report = json.loads(
        (artifacts / "ledgers" / "replay_report.json").read_text(encoding="utf-8")
    )
    assert report["microbatches_replayed"] > 0
    assert report["tokens_match"]
    assert report["loss_masks_match"]
    assert report["token_spans_match"]
    assert report["plan_recomputation_matches"]
    assert report["all_match"]


def test_fork_is_identical_before_and_differs_after_the_divergence(artifacts):
    report = json.loads(
        (artifacts / "ledgers" / "fork_divergence.json").read_text(encoding="utf-8")
    )
    assert report["all_identical_before_fork"]
    assert report["any_divergence_after_fork"]
    assert report["diverged_correctly"]


def test_fork_lineage_is_recorded(artifacts):
    branches = json.loads(
        (artifacts / "ledgers" / "branches.json").read_text(encoding="utf-8")
    )
    forks = [b for b in branches["branches"].values() if b["mode"] == "fork"]
    assert forks
    for fork in forks:
        assert fork["parent_branch_id"] == "main"
        assert fork["forked_from_step"] == CONFIG.fork_from_step
        assert fork["forked_from_checkpoint"]
        assert fork["divergence_note"]


def test_audit_can_answer_the_sessions_questions(artifacts):
    audit = json.loads(
        (artifacts / "ledgers" / "audit_report.json").read_text(encoding="utf-8")
    )
    assert audit["answerable"]["which_shards_trained_this_checkpoint"]
    assert audit["answerable"]["which_shards_in_a_token_window"]
    assert audit["answerable"]["which_batches_preceded_a_spike"]
    provenance = audit["queries"]["checkpoint_provenance"]
    assert provenance["shards_involved"] > 0
    assert provenance["shards"][0]["total_span_tokens"] > 0
