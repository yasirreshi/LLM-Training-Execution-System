"""Shard manifests and the admission gate.

A manifest is what makes a pile of token ids into a training object you can
reason about later.  The field list below is the full one, and every field is present.  Each one answers a question that becomes
unanswerable once the shard is a year old and the person who built it has left:

    content_hash            which shard is this, exactly
    tokenizer_hash          what do these integers mean
    cleaning_pipeline_hash  how did raw text become admitted data
    contamination_status    is it allowed in at all
    capability_lane         where may the scheduler use it
    parent_shard_ids        what was it derived from

The admission gate is deliberately strict and deliberately noisy: a rejection
writes a record naming the rule that fired, so "why is this shard not in the
run" always has an answer on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

MANIFEST_FORMAT_VERSION = "tdes-manifest-1"

TRAINABLE_LICENCE_TIERS = frozenset({"A", "B", "C"})

# Rejection reasons.  Fixed vocabulary so the admission report is aggregatable.
REJECT_LICENCE = "licence_tier_not_trainable"
REJECT_CLEANING = "cleaning_lineage_unknown"
REJECT_TOKENIZER = "tokenizer_hash_missing"
REJECT_EVAL_OVERLAP = "eval_overlap_detected"
REJECT_NEVER_TRAIN = "never_train_flag_set"
REJECT_NOT_SCANNED = "contamination_not_scanned"
REJECT_EMPTY = "shard_has_no_tokens"
REJECT_UNKNOWN_LANE = "capability_lane_unknown"

ALL_REJECT_REASONS = (
    REJECT_LICENCE,
    REJECT_CLEANING,
    REJECT_TOKENIZER,
    REJECT_EVAL_OVERLAP,
    REJECT_NEVER_TRAIN,
    REJECT_NOT_SCANNED,
    REJECT_EMPTY,
    REJECT_UNKNOWN_LANE,
)


@dataclass
class DocumentSpan:
    """Where one document sits inside a shard, and which parts carry loss."""

    doc_id: str
    token_start: int
    token_end: int
    lang: str
    script: str
    stage_hint: str
    reserved: bool
    min_context: int
    content_hash: str
    # [(role, start, end, graded)] offsets are shard-relative
    role_spans: List[Tuple[str, int, int, bool]] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "lang": self.lang,
            "script": self.script,
            "stage_hint": self.stage_hint,
            "reserved": self.reserved,
            "min_context": self.min_context,
            "content_hash": self.content_hash,
            "role_spans": [list(span) for span in self.role_spans],
        }

    @staticmethod
    def from_dict(d: dict) -> "DocumentSpan":
        return DocumentSpan(
            doc_id=d["doc_id"],
            token_start=d["token_start"],
            token_end=d["token_end"],
            lang=d["lang"],
            script=d["script"],
            stage_hint=d["stage_hint"],
            reserved=d["reserved"],
            min_context=d["min_context"],
            content_hash=d["content_hash"],
            role_spans=[tuple(span) for span in d.get("role_spans", [])],
        )


@dataclass
class ShardManifest:
    shard_id: str
    lane: str
    source_id: str
    source_file: str
    tokenizer_hash: Optional[str]
    token_count: int
    documents: List[DocumentSpan]

    provenance: str = ""
    licence: str = ""
    licence_tier: str = ""
    cleaning_pipeline_hash: Optional[str] = None
    dedup_status: str = "unknown"
    pii_status: str = "unknown"
    contamination_status: str = "not_scanned"
    eval_overlap_status: str = "none"
    eval_overlap_detail: List[dict] = field(default_factory=list)
    held_out: bool = False
    never_train: bool = False
    loss_bearing: bool = True
    benchmark_id: Optional[str] = None
    scarce_tier: Optional[str] = None
    capability_tags: List[str] = field(default_factory=list)

    languages: List[str] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)
    stage_hint: str = "early"
    reserved: bool = False
    min_context: int = 0

    content_hash: str = ""
    index_hash: str = ""
    block_hashes: List[str] = field(default_factory=list)
    parent_shard_ids: List[str] = field(default_factory=list)

    packing_policy: str = ""
    loss_policy: str = ""
    position_policy: str = ""
    attention_policy: str = ""

    dataloader_version: str = ""
    config_hash: str = ""
    format_version: str = MANIFEST_FORMAT_VERSION

    # filled in by the gate
    admitted: Optional[bool] = None
    rejection_reasons: List[str] = field(default_factory=list)

    @property
    def document_ids(self) -> List[str]:
        return [doc.doc_id for doc in self.documents]

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["documents"] = [doc.as_dict() for doc in self.documents]
        payload["document_ids"] = self.document_ids
        payload["doc_count"] = len(self.documents)
        return payload

    @staticmethod
    def from_dict(d: dict) -> "ShardManifest":
        data = dict(d)
        data.pop("document_ids", None)
        data.pop("doc_count", None)
        data["documents"] = [DocumentSpan.from_dict(x) for x in d["documents"]]
        return ShardManifest(**data)


# --------------------------------------------------------------------------
# Validation and admission
# --------------------------------------------------------------------------


def validate_manifest(manifest: ShardManifest) -> List[str]:
    """Structural validation - is the manifest itself well formed.

    Separate from admission, which is a *policy* question.  A manifest can be
    perfectly valid and still be refused entry.
    """
    problems: List[str] = []
    if not manifest.shard_id:
        problems.append("missing shard_id")
    if not manifest.content_hash:
        problems.append("missing content_hash")
    if not manifest.documents:
        problems.append("no documents")
    if manifest.token_count <= 0:
        problems.append("token_count must be positive")

    covered = 0
    cursor = 0
    for doc in manifest.documents:
        if doc.token_start != cursor:
            problems.append(
                f"document {doc.doc_id} starts at {doc.token_start}, expected {cursor}"
            )
        if doc.token_end <= doc.token_start:
            problems.append(f"document {doc.doc_id} has a non-positive span")
        for role, start, end, _graded in doc.role_spans:
            if start < doc.token_start or end > doc.token_end:
                problems.append(
                    f"document {doc.doc_id} role span {role} escapes the document"
                )
        cursor = doc.token_end
        covered += doc.token_count
    if covered != manifest.token_count:
        problems.append(
            f"documents cover {covered} tokens but token_count says {manifest.token_count}"
        )
    return problems


def admission_decision(manifest: ShardManifest) -> Tuple[bool, List[str]]:
    """Apply the admission contract.  Returns (admitted, reasons)."""
    from ..config import LANES

    reasons: List[str] = []

    if manifest.token_count <= 0 or not manifest.documents:
        reasons.append(REJECT_EMPTY)
    if not manifest.tokenizer_hash:
        reasons.append(REJECT_TOKENIZER)
    if manifest.licence_tier not in TRAINABLE_LICENCE_TIERS:
        reasons.append(REJECT_LICENCE)
    if not manifest.cleaning_pipeline_hash:
        reasons.append(REJECT_CLEANING)
    if manifest.never_train:
        reasons.append(REJECT_NEVER_TRAIN)
    if manifest.eval_overlap_status not in ("none", ""):
        reasons.append(REJECT_EVAL_OVERLAP)
    if manifest.contamination_status not in ("scanned_clean",):
        reasons.append(REJECT_NOT_SCANNED)
    if manifest.lane not in LANES and not manifest.held_out:
        reasons.append(REJECT_UNKNOWN_LANE)

    return (not reasons), sorted(set(reasons))


def apply_admission(manifest: ShardManifest) -> ShardManifest:
    admitted, reasons = admission_decision(manifest)
    manifest.admitted = admitted
    manifest.rejection_reasons = reasons
    return manifest


def admission_report(manifests: List[ShardManifest]) -> Dict[str, Any]:
    by_reason: Dict[str, List[str]] = {reason: [] for reason in ALL_REJECT_REASONS}
    admitted, rejected = [], []
    for manifest in manifests:
        if manifest.admitted:
            admitted.append(manifest.shard_id)
        else:
            rejected.append(
                {
                    "shard_id": manifest.shard_id,
                    "lane": manifest.lane,
                    "source_id": manifest.source_id,
                    "reasons": manifest.rejection_reasons,
                    "eval_overlap_detail": manifest.eval_overlap_detail,
                }
            )
            for reason in manifest.rejection_reasons:
                by_reason.setdefault(reason, []).append(manifest.shard_id)
    return {
        "total_shards": len(manifests),
        "admitted_count": len(admitted),
        "rejected_count": len(rejected),
        "admitted": sorted(admitted),
        "rejected": sorted(rejected, key=lambda r: r["shard_id"]),
        "rejected_by_reason": {k: sorted(v) for k, v in sorted(by_reason.items()) if v},
        "reason_vocabulary": list(ALL_REJECT_REASONS),
    }
