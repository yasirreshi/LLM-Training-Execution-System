"""Packing policies, loss masks, attention confinement and position ids."""

from __future__ import annotations

import pytest

from tdes.config import LANE_LOSS_POLICY, LANE_PACKING_POLICY
from tdes.packing.masks import (
    PAD_SEGMENT,
    build_attention_mask,
    build_position_ids,
    check_attention_confinement,
    validate_masks,
)
from tdes.packing.policies import Item, POLICIES, compare_policies, pack


# -- the policies, on abstract items ---------------------------------------


def _items():
    return [
        Item("a", 300, (100, 200)),
        Item("b", 120),
        Item("c", 90),
        Item("d", 700, (200, 400, 600)),
        Item("e", 40),
    ]


@pytest.mark.parametrize("policy", POLICIES)
def test_no_window_ever_exceeds_capacity(policy):
    result = pack(policy, _items(), 256)
    for window in result.windows:
        assert window.used <= 256


def test_best_fit_beats_greedy_or_ties_on_window_count():
    greedy = pack("greedy", _items(), 256)
    best = pack("best_fit", _items(), 256)
    assert len(best.windows) <= len(greedy.windows)


def test_structure_preserving_never_co_packs():
    result = pack("structure_preserving", _items(), 256)
    for window in result.windows:
        assert len({p.item_id for p in window.placements}) == 1
    assert result.stats()["boundary_crossings"] == 0


def test_long_context_defers_documents_that_need_a_bigger_window():
    items = [Item("needs_512", 200, min_context=512), Item("fits", 200)]
    result = pack("long_context", items, 256)
    assert "needs_512" in result.deferred
    assert "fits" not in result.deferred


def test_long_context_refuses_to_waste_a_window_on_a_short_document():
    result = pack("long_context", [Item("tiny", 10)], 512)
    assert result.deferred == ["tiny"]


def test_pad_only_looks_efficient_but_loses_tokens():
    """Utilisation alone can be gamed; retention is what exposes it."""
    stats = pack("pad_only", _items(), 256).stats()
    assert stats["truncated_tokens"] > 0
    assert stats["retention"] < 1.0


def test_splits_land_on_declared_boundaries():
    item = Item("doc", 700, split_points=(200, 400, 600))
    result = pack("structure_preserving", [item], 256)
    offsets = [p.offset_in_item for w in result.windows for p in w.placements]
    assert offsets[0] == 0
    for offset in offsets[1:]:
        assert offset in item.split_points


def test_every_policy_reports_comparable_stats():
    comparison = compare_policies(_items(), 256)
    assert set(comparison) == set(POLICIES)
    for stats in comparison.values():
        assert {"utilisation", "retention", "document_splits", "windows"} <= set(stats)


# -- masks -----------------------------------------------------------------


def test_position_ids_restart_per_segment():
    segments = [0, 0, 0, 1, 1, PAD_SEGMENT, PAD_SEGMENT]
    assert build_position_ids(segments) == [0, 1, 2, 0, 1, 0, 0]


def test_attention_is_causal_and_segment_confined():
    segments = [0, 0, 1, 1]
    mask = build_attention_mask(segments)
    assert mask[3][2] is True            # same segment, backwards
    assert mask[3][1] is False           # different segment
    assert mask[1][2] is False           # forwards
    assert check_attention_confinement(segments).ok


def test_validator_catches_loss_on_padding():
    result = validate_masks(
        token_ids=[5, 6, 0, 0], loss_mask=[0, 1, 1, 0],
        segment_ids=[0, 0, PAD_SEGMENT, PAD_SEGMENT], position_ids=[0, 1, 0, 0],
        pad_token_id=0,
    )
    assert not result.ok
    assert any("pad" in p for p in result.problems)


def test_validator_catches_loss_on_the_first_token_of_a_segment():
    result = validate_masks(
        token_ids=[5, 6, 7, 8], loss_mask=[0, 1, 1, 1],
        segment_ids=[0, 0, 1, 1], position_ids=[0, 1, 0, 1], pad_token_id=0,
    )
    assert not result.ok
    assert any("first token of a segment" in p for p in result.problems)


def test_validator_catches_loss_on_context_only_tokens():
    result = validate_masks(
        token_ids=[5, 6, 7], loss_mask=[0, 1, 1], segment_ids=[0, 0, 0],
        position_ids=[0, 1, 2], pad_token_id=0,
        graded_flags=[False, False, True],
    )
    assert not result.ok
    assert any("context-only" in p for p in result.problems)


def test_validator_catches_a_real_token_after_padding():
    result = validate_masks(
        token_ids=[5, 0, 7], loss_mask=[0, 0, 0],
        segment_ids=[0, PAD_SEGMENT, 0], position_ids=[0, 0, 1], pad_token_id=0,
    )
    assert not result.ok
    assert any("after padding" in p for p in result.problems)


# -- the real packed samples -----------------------------------------------


def test_every_packed_sample_passes_mask_validation(system):
    pad_id = system.tokenizer.special_id("<pad>")
    for sample in system.store.by_id.values():
        result = validate_masks(
            sample.token_ids, sample.loss_mask, sample.segment_ids,
            sample.position_ids, pad_id, sample.graded_flags,
        )
        assert result.ok, (sample.sample_id, result.problems[:3])


def test_no_packed_sample_leaks_attention_across_documents(system):
    for sample in system.store.by_id.values():
        assert check_attention_confinement(sample.segment_ids).ok, sample.sample_id


def test_each_lane_uses_its_declared_policy(system):
    for (length, reserved), lanes in system.store.by_context.items():
        for lane, samples in lanes.items():
            for sample in samples:
                assert sample.policy == LANE_PACKING_POLICY[lane]


def test_agentic_samples_grade_only_the_model_turns(system):
    """User turns and tool observations are context, not targets."""
    samples = system.store.lane_samples(256, False, "agentic")
    assert samples, "the agentic lane produced no samples"
    assert LANE_LOSS_POLICY["agentic"] == "model_turns"
    for sample in samples:
        # Some positions must be context-only, or the mask is doing nothing.
        assert sample.context_only_count > 0
        assert sample.loss_bearing_count > 0
        assert sample.loss_bearing_count < sample.real_token_count


def test_plain_pretraining_lanes_grade_almost_everything(system):
    samples = system.store.lane_samples(256, False, "general_web")
    assert samples
    for sample in samples:
        # everything except the BOS of each segment
        expected = sample.real_token_count - len(sample.segments)
        assert sample.loss_bearing_count == expected


def test_sample_ids_are_content_addressed(system):
    """Two samples with the same segments must share an id, and vice versa."""
    ids = list(system.store.by_id)
    assert len(ids) == len(set(ids))
    for sample in list(system.store.by_id.values())[:20]:
        assert sample.sample_id.startswith("smp-")
        assert sample.tokens_hash == sample.tokens_hash
