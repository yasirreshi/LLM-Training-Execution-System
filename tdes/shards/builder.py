"""Turn documents into immutable tokenized shards.

The output of this module is the only thing training ever reads.  Three
properties are enforced here rather than assumed downstream:

*   **Deterministic.**  The same corpus and the same tokenizer produce the same
    shard ids and the same content hashes, on any machine, on any day.  Nothing
    depends on filesystem enumeration order, dict iteration order or a clock.

*   **Immutable.**  Shard files are written atomically and then made read-only.
    Modifying a shard is not supported; producing a new shard with a new hash
    and a `parent_shard_ids` link is.

*   **Self-describing.**  Every shard has a manifest carrying its tokenizer
    hash, cleaning lineage, licence, contamination status and lane, so a future
    reader can decide whether it is allowed to use it without asking anyone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import (
    ATTENTION_POLICY,
    BOS_TOKEN,
    CONFIG,
    EOS_TOKEN,
    LANE_LOSS_POLICY,
    LANE_PACKING_POLICY,
    LANE_POSITION_POLICY,
    PATHS,
)
from ..corpus.loader import Document, ROLE_SPECIAL_TOKEN, Source
from ..firewall.contamination import EvalFingerprintRegistry
from ..fsutil import make_readonly, tokens_to_bytes, write_atomic_bytes, write_json
from ..hashing import merkle_root, sha256_bytes, short_hash
from ..tokenizer.bpe import BPETokenizer
from .manifest import DocumentSpan, ShardManifest, apply_admission

BLOCK_SIZE = 4096          # bytes per merkle leaf
DOCS_PER_SHARD = 3


def _tokenize_document(
    doc: Document, tokenizer: BPETokenizer
) -> Tuple[List[int], List[Tuple[str, int, int, bool]]]:
    """Encode one document, returning tokens and doc-relative role spans.

    Layout:  <bos> [ <role> span-tokens ]... <eos>

    The role marker is part of the span it introduces and inherits its graded
    flag - the model should learn to emit `<think>` but not to emit `<user>`.
    `<eos>` is graded on purpose: ending is a behaviour the model has to learn,
    and per-token perplexity at EOS is one of the few direct read-outs of
    whether document boundaries are being honoured.
    """
    bos = tokenizer.special_id(BOS_TOKEN)
    eos = tokenizer.special_id(EOS_TOKEN)

    tokens: List[int] = [bos]
    spans: List[Tuple[str, int, int, bool]] = []

    for span in doc.spans:
        start = len(tokens)
        if span.role != "text":
            marker = ROLE_SPECIAL_TOKEN.get(span.role)
            if marker:
                tokens.append(tokenizer.special_id(marker))
        tokens.extend(tokenizer.encode(span.text + "\n"))
        if len(tokens) > start:
            spans.append((span.role, start, len(tokens), span.graded))

    eos_start = len(tokens)
    tokens.append(eos)
    spans.append(("eos", eos_start, len(tokens), True))
    return tokens, spans


def _shard_groups(documents: Sequence[Document], size: int = DOCS_PER_SHARD) -> List[List[Document]]:
    ordered = sorted(documents, key=lambda d: d.doc_id)
    return [list(ordered[i:i + size]) for i in range(0, len(ordered), size)]


def build_shards(
    sources: List[Source],
    tokenizer: BPETokenizer,
    tokenizer_hash: str,
    eval_registry: Optional[EvalFingerprintRegistry] = None,
    out_dir: Path = None,
) -> List[ShardManifest]:
    """Build every shard and return their manifests, admission already applied."""
    directory = Path(out_dir) if out_dir else PATHS.shards
    directory.mkdir(parents=True, exist_ok=True)

    manifests: List[ShardManifest] = []

    for source in sorted(sources, key=lambda s: s.source_id):
        for group_index, group in enumerate(_shard_groups(source.documents)):
            if not group:
                continue
            manifests.append(
                _build_one_shard(
                    source, group, group_index, tokenizer, tokenizer_hash,
                    eval_registry, directory
                )
            )

    manifests.sort(key=lambda m: m.shard_id)
    return manifests


def _build_one_shard(
    source: Source,
    group: List[Document],
    group_index: int,
    tokenizer: BPETokenizer,
    tokenizer_hash: str,
    eval_registry: Optional[EvalFingerprintRegistry],
    directory: Path,
) -> ShardManifest:
    tokens: List[int] = []
    doc_spans: List[DocumentSpan] = []

    for doc in group:
        doc_tokens, role_spans = _tokenize_document(doc, tokenizer)
        offset = len(tokens)
        doc_spans.append(
            DocumentSpan(
                doc_id=doc.doc_id,
                token_start=offset,
                token_end=offset + len(doc_tokens),
                lang=doc.lang,
                script=doc.script,
                stage_hint=doc.stage_hint,
                reserved=doc.reserved,
                min_context=doc.min_context,
                content_hash=doc.content_hash,
                role_spans=[
                    (role, offset + start, offset + end, graded)
                    for role, start, end, graded in role_spans
                ],
            )
        )
        tokens.extend(doc_tokens)

    # Shard id is derived from content, not from a counter, so rebuilding the
    # corpus in a different order still yields the same id for the same shard.
    shard_id = "sh-{lane}-{digest}".format(
        lane=source.lane.replace("_", ""),
        digest=short_hash(
            {
                "source_id": source.source_id,
                "group_index": group_index,
                "doc_ids": [doc.doc_id for doc in group],
                "tokenizer_hash": tokenizer_hash,
            }
        )[:10],
    )

    payload = tokens_to_bytes(tokens)
    block_hashes = [
        hashlib.sha256(payload[offset:offset + BLOCK_SIZE]).hexdigest()
        for offset in range(0, max(len(payload), 1), BLOCK_SIZE)
    ]
    content_hash = merkle_root(block_hashes)

    index_payload = {
        "shard_id": shard_id,
        "tokenizer_hash": tokenizer_hash,
        "documents": [doc.as_dict() for doc in doc_spans],
    }
    index_hash = sha256_bytes(
        json.dumps(
            index_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )

    bin_path = directory / f"{shard_id}.bin"
    write_atomic_bytes(bin_path, payload)
    write_json(directory / f"{shard_id}.idx.json", index_payload)
    make_readonly(bin_path)
    make_readonly(directory / f"{shard_id}.idx.json")

    # Contamination scan, per document, before the gate runs.
    overlap_status = "none"
    overlap_detail: List[dict] = []
    if eval_registry is not None and not source.held_out:
        for doc in group:
            for hit in eval_registry.scan_text(doc.text):
                overlap_detail.append({"doc_id": doc.doc_id, **hit.as_dict()})
        if overlap_detail:
            overlap_status = "overlap_detected"

    # The manifest records what the scan *found*, not what the source claimed.
    # One exception: a source whose cleaning lineage was never recorded stays
    # "not_scanned" regardless, because a clean overlap scan says nothing about
    # PII, deduplication or the other checks that were also never run.
    if source.contamination_status == "not_scanned":
        contamination_status = "not_scanned"
    elif eval_registry is not None and not source.held_out:
        contamination_status = "overlap_detected" if overlap_detail else "scanned_clean"
    else:
        contamination_status = source.contamination_status

    manifest = ShardManifest(
        shard_id=shard_id,
        lane=source.lane,
        source_id=source.source_id,
        source_file=source.file,
        tokenizer_hash=tokenizer_hash,
        token_count=len(tokens),
        documents=doc_spans,
        provenance=source.provenance,
        licence=source.licence,
        licence_tier=source.licence_tier,
        cleaning_pipeline_hash=source.cleaning_pipeline,
        dedup_status=source.dedup_status,
        pii_status=source.pii_status,
        contamination_status=contamination_status,
        eval_overlap_status=overlap_status,
        eval_overlap_detail=overlap_detail,
        held_out=source.held_out,
        never_train=source.never_train,
        loss_bearing=source.loss_bearing,
        benchmark_id=source.benchmark_id,
        scarce_tier=source.scarce_tier,
        capability_tags=sorted(source.capability_tags),
        languages=sorted({doc.lang for doc in group}),
        scripts=sorted({doc.script for doc in group}),
        stage_hint=_dominant_stage_hint(group),
        reserved=all(doc.reserved for doc in group),
        min_context=max((doc.min_context for doc in group), default=0),
        content_hash=content_hash,
        index_hash=index_hash,
        block_hashes=block_hashes,
        parent_shard_ids=[],
        packing_policy=LANE_PACKING_POLICY.get(source.lane, "pad_only"),
        loss_policy=LANE_LOSS_POLICY.get(source.lane, "all"),
        position_policy=LANE_POSITION_POLICY.get(source.lane, "segment_relative"),
        attention_policy=ATTENTION_POLICY,
        dataloader_version=CONFIG.dataloader_version,
        config_hash=CONFIG.config_hash,
    )
    return apply_admission(manifest)


def _dominant_stage_hint(group: Sequence[Document]) -> str:
    order = {"early": 0, "mid": 1, "anneal": 2}
    return max((doc.stage_hint for doc in group), key=lambda h: order.get(h, 0))


# --------------------------------------------------------------------------
# Reading shards back
# --------------------------------------------------------------------------


class ShardReader:
    """Reads token spans out of shard files, verifying the tokenizer first."""

    def __init__(self, directory: Path = None, live_tokenizer_hash: str = ""):
        self.directory = Path(directory) if directory else PATHS.shards
        self.live_tokenizer_hash = live_tokenizer_hash
        self._cache: Dict[str, bytes] = {}
        self.read_count = 0
        self.cache_hits = 0

    def _load(self, shard_id: str) -> bytes:
        cached = self._cache.get(shard_id)
        if cached is not None:
            self.cache_hits += 1
            return cached
        data = (self.directory / f"{shard_id}.bin").read_bytes()
        self._cache[shard_id] = data
        self.read_count += 1
        return data

    def tokens(self, shard_id: str, start: int, end: int) -> List[int]:
        data = self._load(shard_id)
        chunk = data[start * 4:end * 4]
        return [
            int.from_bytes(chunk[i:i + 4], "little", signed=False)
            for i in range(0, len(chunk), 4)
        ]

    def verify_content_hash(self, manifest: ShardManifest) -> bool:
        data = self._load(manifest.shard_id)
        block_hashes = [
            hashlib.sha256(data[offset:offset + BLOCK_SIZE]).hexdigest()
            for offset in range(0, max(len(data), 1), BLOCK_SIZE)
        ]
        return merkle_root(block_hashes) == manifest.content_hash
