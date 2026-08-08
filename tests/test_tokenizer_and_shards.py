"""Tokenizer contract and shard immutability."""

from __future__ import annotations

import json

import pytest

from tdes.corpus.loader import load_sources, training_text
from tdes.shards.builder import build_shards
from tdes.shards.manifest import (
    REJECT_CLEANING,
    REJECT_EVAL_OVERLAP,
    REJECT_LICENCE,
    REJECT_NEVER_TRAIN,
    admission_decision,
    validate_manifest,
)
from tdes.tokenizer.freeze import (
    TokenizerIntegrityError,
    load_frozen,
    tokenizer_hash,
    train_tokenizer,
    verify_shard_tokenizer,
)
from tdes.tokenizer.normalize import ZWJ, ZWNJ, normalize, preserves_joiners


# -- tokenizer -------------------------------------------------------------


def test_encoding_is_deterministic(system):
    text = "The monsoon reaches the Kerala coast in the first week of June."
    first = system.tokenizer.encode(text)
    second = system.tokenizer.encode(text)
    assert first == second


def test_round_trip_preserves_text(system):
    for text in [
        "Contour lines connect points of equal elevation.",
        "मानसून केरल के तट पर पहुँचता है।",
        "பருவமழை கேரளக் கடற்கரையை அடைகிறது.",
        "def apportion(weights, total):\n    return weights\n",
    ]:
        assert system.tokenizer.decode(system.tokenizer.encode(text)) == text


def test_training_the_tokenizer_twice_gives_the_same_merges():
    sources = load_sources()
    texts = training_text(sources)
    first = train_tokenizer(texts)
    second = train_tokenizer(texts)
    assert first.merges == second.merges
    assert first.vocab_size == second.vocab_size


def test_normalization_preserves_indic_joiners():
    text = f"क{ZWJ}्ष and क{ZWNJ}्ष and क्ष"
    assert preserves_joiners(text)
    assert normalize(text).count(ZWJ) == 1
    assert normalize(text).count(ZWNJ) == 1


def test_normalization_is_idempotent():
    text = "  ragged   text \r\n with\ttabs \n\n\n\n and blanks  "
    once = normalize(text)
    assert normalize(once) == once


def test_nukta_spellings_collide_after_normalization():
    """U+0958 and the decomposed क + nukta must produce one token sequence."""
    precomposed = "क़"
    decomposed = "क़"
    assert normalize(precomposed).strip() == normalize(decomposed).strip()


def test_frozen_tokenizer_hash_reverifies(artifacts):
    tokenizer, recorded = load_frozen(artifacts / "manifests" / "tokenizer.json")
    document = json.loads(
        (artifacts / "manifests" / "tokenizer.json").read_text(encoding="utf-8")
    )
    payload = {k: v for k, v in document.items()
               if k not in ("tokenizer_hash", "frozen_at_unix")}
    assert tokenizer_hash(payload) == recorded


def test_tampered_tokenizer_is_rejected(tmp_path, artifacts):
    document = json.loads(
        (artifacts / "manifests" / "tokenizer.json").read_text(encoding="utf-8")
    )
    document["merges"][0] = [999, 998]          # change one merge
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TokenizerIntegrityError):
        load_frozen(path)


def test_shard_with_foreign_tokenizer_is_refused():
    with pytest.raises(TokenizerIntegrityError):
        verify_shard_tokenizer("aaaa", "bbbb", "sh-test")


def test_indic_fertility_is_worse_than_english(system):
    """A real, reportable measurement, not an assumption.

    The design's argument for an Indic-aware tokenizer rests on this gap: a
    higher fertility means fewer words fit the same context window and the same
    document costs proportionally more to train on.
    """
    english = system.tokenizer.fertility(
        "The river gauge records stage, not discharge, and the rating curve does the rest."
    )
    tamil = system.tokenizer.fertility(
        "பருவமழை கேரளக் கடற்கரையை அடையும்போது விவசாய நாட்காட்டி அதனுடன் இணைகிறது."
    )
    assert english < tamil


# -- shards ----------------------------------------------------------------


def test_rebuilding_shards_gives_identical_content_hashes(system):
    rebuilt = build_shards(
        system.sources, system.tokenizer, system.tokenizer_hash, system.fingerprints
    )
    original = {m.shard_id: m.content_hash for m in system.manifests}
    again = {m.shard_id: m.content_hash for m in rebuilt}
    assert original == again


def test_every_manifest_is_structurally_valid(system):
    problems = [p for m in system.manifests for p in validate_manifest(m)]
    assert problems == []


def test_manifests_carry_the_required_fields(system):
    required = {
        "shard_id", "source_id", "document_ids", "tokenizer_hash", "token_count",
        "languages", "scripts", "lane", "licence", "licence_tier",
        "cleaning_pipeline_hash", "dedup_status", "contamination_status",
        "eval_overlap_status", "content_hash", "parent_shard_ids",
    }
    for manifest in system.manifests:
        assert required.issubset(manifest.as_dict())


def test_shard_files_are_read_only(system):
    from tdes.config import PATHS
    from tdes.fsutil import is_readonly

    for manifest in system.manifests[:5]:
        assert is_readonly(PATHS.shards / f"{manifest.shard_id}.bin")


def test_content_hash_matches_the_bytes_on_disk(system):
    for manifest in system.manifests:
        assert system.reader.verify_content_hash(manifest)


def test_admission_gate_rejects_each_bad_class(system):
    """Every rejection reason must be exercised by a real source, not a stub."""
    reasons = set()
    for manifest in system.manifests:
        reasons.update(manifest.rejection_reasons)
    assert REJECT_LICENCE in reasons
    assert REJECT_CLEANING in reasons
    assert REJECT_EVAL_OVERLAP in reasons
    assert REJECT_NEVER_TRAIN in reasons


def test_admitted_shards_have_no_rejection_reason(system):
    for manifest in system.manifests:
        if manifest.admitted:
            assert manifest.rejection_reasons == []


def test_a_missing_tokenizer_hash_blocks_admission(system):
    import copy

    manifest = copy.deepcopy(system.manifests[0])
    manifest.tokenizer_hash = None
    admitted, reasons = admission_decision(manifest)
    assert not admitted
    assert "tokenizer_hash_missing" in reasons
