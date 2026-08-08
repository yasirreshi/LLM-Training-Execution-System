"""Build packed training samples from admitted shards.

This is where a shard stops being a token array and becomes a training example:
fixed length, with a loss mask that says what is graded, segment ids that stop
documents leaking into each other through attention, and position ids that
restart per document.

Every sample is content-addressed.  `sample_id` is a hash of the lane, the
policy, the window length and the exact (shard, offset, length) segments it
contains - so the same corpus always produces the same sample ids, and a sample
id in a ledger written six months ago still names one specific window of tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import (
    ATTENTION_POLICY,
    LANE_LOSS_POLICY,
    LANE_PACKING_POLICY,
    LANE_POSITION_POLICY,
    PAD_TOKEN,
)
from ..firewall.eval_firewall import EvalFirewall
from ..hashing import hash_mask, hash_token_ids, short_hash
from ..shards.builder import ShardReader
from ..shards.manifest import DocumentSpan, ShardManifest
from ..shards.registry import ShardRegistry
from ..tokenizer.bpe import BPETokenizer
from .masks import PAD_SEGMENT, build_position_ids, validate_masks
from .policies import Item, compare_policies, pack


@dataclass
class SegmentRef:
    """One contiguous stretch of one document inside a packed window."""

    shard_id: str
    doc_id: str
    token_start: int          # shard-relative
    token_end: int            # shard-relative
    window_start: int         # index inside the packed window
    lane: str
    lang: str
    script: str

    @property
    def length(self) -> int:
        return self.token_end - self.token_start

    def as_dict(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "doc_id": self.doc_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "window_start": self.window_start,
            "lane": self.lane,
            "lang": self.lang,
            "script": self.script,
        }


@dataclass
class PackedSample:
    sample_id: str
    lane: str
    policy: str
    sequence_length: int
    token_ids: List[int]
    loss_mask: List[int]
    segment_ids: List[int]
    position_ids: List[int]
    graded_flags: List[bool]
    segments: List[SegmentRef]
    reserved: bool = False
    min_context: int = 0
    stage_hint: str = "early"
    _decoded: Optional[str] = field(default=None, repr=False)

    # -- derived counts ---------------------------------------------------

    @property
    def pad_count(self) -> int:
        return sum(1 for s in self.segment_ids if s == PAD_SEGMENT)

    @property
    def real_token_count(self) -> int:
        return self.sequence_length - self.pad_count

    @property
    def loss_bearing_count(self) -> int:
        return sum(self.loss_mask)

    @property
    def context_only_count(self) -> int:
        return self.real_token_count - self.loss_bearing_count

    @property
    def utilisation(self) -> float:
        return self.real_token_count / self.sequence_length

    @property
    def loss_density(self) -> float:
        """Fraction of the window that actually teaches the model anything.

        The number that matters more than raw utilisation: a window can be 100%
        full of tokens and still be mostly context the model is not graded on.
        """
        return self.loss_bearing_count / self.sequence_length

    # -- identity ---------------------------------------------------------

    @property
    def tokens_hash(self) -> str:
        return hash_token_ids(self.token_ids)

    @property
    def loss_mask_hash(self) -> str:
        return hash_mask(self.loss_mask)

    @property
    def shard_ids(self) -> List[str]:
        return sorted({s.shard_id for s in self.segments})

    @property
    def doc_ids(self) -> List[str]:
        return [s.doc_id for s in self.segments]

    @property
    def token_span_ids(self) -> List[str]:
        """Stable names for the token ranges this sample consumed."""
        return [f"{s.shard_id}:{s.token_start}-{s.token_end}" for s in self.segments]

    def content_hash(self) -> str:
        return short_hash(
            {
                "tokens": self.tokens_hash,
                "loss_mask": self.loss_mask_hash,
                "segments": [s.as_dict() for s in self.segments],
            }
        )

    def segment_at(self, index: int) -> Optional[SegmentRef]:
        """Which packed document owns window position `index`.

        Needed by the per-token trace: a token's loss is only interpretable
        once you know which document and shard it came from.
        """
        segment = self.segment_ids[index]
        if segment == PAD_SEGMENT or segment >= len(self.segments):
            return None
        return self.segments[segment]

    def decoded(self, tokenizer: BPETokenizer) -> str:
        if self._decoded is None:
            self._decoded = tokenizer.decode(
                [t for t, s in zip(self.token_ids, self.segment_ids) if s != PAD_SEGMENT]
            )
        return self._decoded

    def as_index_entry(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "lane": self.lane,
            "policy": self.policy,
            "sequence_length": self.sequence_length,
            "segments": [s.as_dict() for s in self.segments],
            "token_span_ids": self.token_span_ids,
            "shard_ids": self.shard_ids,
            "doc_ids": self.doc_ids,
            "pad_count": self.pad_count,
            "loss_bearing_tokens": self.loss_bearing_count,
            "context_only_tokens": self.context_only_count,
            "utilisation": round(self.utilisation, 4),
            "loss_density": round(self.loss_density, 4),
            "tokens_hash": self.tokens_hash,
            "loss_mask_hash": self.loss_mask_hash,
            "content_hash": self.content_hash(),
            "reserved": self.reserved,
            "stage_hint": self.stage_hint,
            "attention_policy": ATTENTION_POLICY,
            "position_policy": LANE_POSITION_POLICY.get(self.lane, "segment_relative"),
            "loss_policy": LANE_LOSS_POLICY.get(self.lane, "all"),
        }


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def _graded_array(doc: DocumentSpan) -> List[bool]:
    """Per-token graded flags for one document, doc-relative."""
    flags = [False] * doc.token_count
    for _role, start, end, graded in doc.role_spans:
        if not graded:
            continue
        for i in range(start - doc.token_start, end - doc.token_start):
            if 0 <= i < len(flags):
                flags[i] = True
    return flags


def _split_points(doc: DocumentSpan) -> Tuple[int, ...]:
    """Turn boundaries, doc-relative - the only places a split may land."""
    return tuple(
        sorted({span[1] - doc.token_start for span in doc.role_spans if span[1] > doc.token_start})
    )


class Packer:
    def __init__(
        self,
        registry: ShardRegistry,
        reader: ShardReader,
        tokenizer: BPETokenizer,
        firewall: Optional[EvalFirewall] = None,
    ):
        self.registry = registry
        self.reader = reader
        self.tokenizer = tokenizer
        self.firewall = firewall
        self.pad_id = tokenizer.special_id(PAD_TOKEN)
        self.policy_comparison: Dict[str, Dict[str, dict]] = {}
        self.pack_stats: Dict[str, dict] = {}

    # -- item construction ------------------------------------------------

    def _lane_items(
        self, lane: str, allow_reserved: bool
    ) -> Tuple[List[Item], Dict[str, Tuple[str, DocumentSpan]]]:
        items: List[Item] = []
        lookup: Dict[str, Tuple[str, DocumentSpan]] = {}

        for entry in sorted(self.registry.trainable_by_lane(lane), key=lambda e: e.shard_id):
            manifest: ShardManifest = entry.manifest
            # Registry side of the firewall, at planning time.
            if self.firewall is not None and not self.firewall.check_admission(
                manifest.shard_id, context=f"packing:{lane}"
            ):
                continue
            for doc in manifest.documents:
                if doc.reserved and not allow_reserved:
                    continue
                item_id = f"{manifest.shard_id}#{doc.doc_id}"
                items.append(
                    Item(
                        item_id=item_id,
                        size=doc.token_count,
                        split_points=_split_points(doc),
                        min_context=doc.min_context,
                    )
                )
                lookup[item_id] = (manifest.shard_id, doc)

        items.sort(key=lambda i: i.item_id)
        return items, lookup

    # -- materialisation --------------------------------------------------

    def _materialise(
        self,
        lane: str,
        policy: str,
        sequence_length: int,
        window,
        lookup: Dict[str, Tuple[str, DocumentSpan]],
    ) -> PackedSample:
        token_ids: List[int] = []
        loss_mask: List[int] = []
        segment_ids: List[int] = []
        graded_flags: List[bool] = []
        segments: List[SegmentRef] = []

        for segment_index, placement in enumerate(window.placements):
            shard_id, doc = lookup[placement.item_id]
            start = doc.token_start + placement.offset_in_item
            end = start + placement.length

            piece = self.reader.tokens(shard_id, start, end)
            doc_graded = _graded_array(doc)
            piece_graded = doc_graded[
                placement.offset_in_item:placement.offset_in_item + placement.length
            ]

            window_start = len(token_ids)
            token_ids.extend(piece)
            graded_flags.extend(piece_graded)
            segment_ids.extend([segment_index] * len(piece))
            # The first token of a segment has nothing before it to condition
            # on, so it can never be a target however it was tagged.
            loss_mask.extend(
                [0] + [1 if g else 0 for g in piece_graded[1:]]
            )

            segments.append(
                SegmentRef(
                    shard_id=shard_id,
                    doc_id=doc.doc_id,
                    token_start=start,
                    token_end=end,
                    window_start=window_start,
                    lane=lane,
                    lang=doc.lang,
                    script=doc.script,
                )
            )

        pad_needed = sequence_length - len(token_ids)
        if pad_needed < 0:
            raise ValueError(f"packed window overflows {sequence_length} by {-pad_needed}")
        token_ids.extend([self.pad_id] * pad_needed)
        loss_mask.extend([0] * pad_needed)
        segment_ids.extend([PAD_SEGMENT] * pad_needed)
        graded_flags.extend([False] * pad_needed)

        position_ids = build_position_ids(segment_ids)

        sample_id = "smp-" + short_hash(
            {
                "lane": lane,
                "policy": policy,
                "sequence_length": sequence_length,
                "segments": [s.as_dict() for s in segments],
            }
        )[:12]

        sample = PackedSample(
            sample_id=sample_id,
            lane=lane,
            policy=policy,
            sequence_length=sequence_length,
            token_ids=token_ids,
            loss_mask=loss_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            graded_flags=graded_flags,
            segments=segments,
            reserved=any(lookup[p.item_id][1].reserved for p in window.placements),
            min_context=max(
                (lookup[p.item_id][1].min_context for p in window.placements), default=0
            ),
            stage_hint=max(
                (lookup[p.item_id][1].stage_hint for p in window.placements),
                key=lambda h: {"early": 0, "mid": 1, "anneal": 2}.get(h, 0),
                default="early",
            ),
        )

        validation = validate_masks(
            sample.token_ids, sample.loss_mask, sample.segment_ids,
            sample.position_ids, self.pad_id, sample.graded_flags,
        )
        if not validation.ok:
            raise ValueError(
                f"packed sample {sample_id} failed mask validation: {validation.problems[:5]}"
            )
        return sample

    # -- public -----------------------------------------------------------

    def count_windows(
        self, lane: str, sequence_length: int, allow_reserved: bool
    ) -> Tuple[int, int]:
        """How many windows a lane would yield, without materialising tokens.

        The packing policies work on abstract items carrying only a size and
        their split points, so the window count is known before a single token
        is read from a shard.  That is what lets the mixture be compiled - and
        checked for feasibility - *before* the samples are built, rather than
        after, which is the order the schedule actually needs.

        Returns (window_count, packable_tokens).
        """
        items, _lookup = self._lane_items(lane, allow_reserved)
        if not items:
            return 0, 0
        result = pack(LANE_PACKING_POLICY[lane], items, sequence_length)
        windows = [w for w in result.windows if w.used > 1]
        return len(windows), sum(w.used for w in windows)

    def pack_lane(
        self, lane: str, sequence_length: int, allow_reserved: bool
    ) -> List[PackedSample]:
        policy = LANE_PACKING_POLICY[lane]
        items, lookup = self._lane_items(lane, allow_reserved)
        if not items:
            return []

        key = f"{lane}@{sequence_length}{'+reserved' if allow_reserved else ''}"
        self.policy_comparison[key] = compare_policies(items, sequence_length)

        result = pack(policy, items, sequence_length)
        stats = result.stats()
        stats["policy"] = policy
        stats["lane"] = lane
        self.pack_stats[key] = stats

        return [
            self._materialise(lane, policy, sequence_length, window, lookup)
            for window in result.windows
            if window.used > 1          # a one-token window has no target
        ]


    def pack_holdout(
        self, entries, label: str, sequence_length: int, policy: str = "greedy"
    ) -> List[PackedSample]:
        """Pack validation or probe shards.

        Deliberately does not run the registry firewall check, because these
        shards are *supposed* to fail it - they have `validation` permission,
        not `train`.  They are packed so their loss can be measured; the
        firewall's batch-side gate still refuses them if anything ever tries to
        route them into a gradient path, and `EvalFirewall.note_validation_read`
        records the read.
        """
        items: List[Item] = []
        lookup: Dict[str, Tuple[str, DocumentSpan]] = {}
        for entry in sorted(entries, key=lambda e: e.manifest.shard_id):
            for doc in entry.manifest.documents:
                item_id = f"{entry.manifest.shard_id}#{doc.doc_id}"
                items.append(
                    Item(item_id=item_id, size=doc.token_count,
                         split_points=_split_points(doc), min_context=doc.min_context)
                )
                lookup[item_id] = (entry.manifest.shard_id, doc)
        if not items:
            return []
        items.sort(key=lambda i: i.item_id)
        result = pack(policy, items, sequence_length)
        self.pack_stats[f"{label}@{sequence_length}"] = {
            "policy": policy, "lane": label, **result.stats()
        }
        return [
            self._materialise(label, policy, sequence_length, window, lookup)
            for window in result.windows
            if window.used > 1
        ]


class PackedSampleStore:
    """Every packed sample, indexed by (sequence_length, reserved_unlocked)."""

    def __init__(self):
        self.by_context: Dict[Tuple[int, bool], Dict[str, List[PackedSample]]] = {}
        self.by_id: Dict[str, PackedSample] = {}

    def add(
        self, sequence_length: int, reserved_unlocked: bool, lane: str,
        samples: Sequence[PackedSample],
    ) -> None:
        context = self.by_context.setdefault((sequence_length, reserved_unlocked), {})
        context.setdefault(lane, []).extend(samples)
        for sample in samples:
            self.by_id[sample.sample_id] = sample

    def lane_samples(
        self, sequence_length: int, reserved_unlocked: bool, lane: str
    ) -> List[PackedSample]:
        return self.by_context.get((sequence_length, reserved_unlocked), {}).get(lane, [])

    def get(self, sample_id: str) -> PackedSample:
        return self.by_id[sample_id]

    def __len__(self) -> int:
        return len(self.by_id)

    def index(self) -> dict:
        return {
            "sample_count": len(self.by_id),
            "contexts": {
                f"len{length}{'+reserved' if reserved else ''}": {
                    lane: [s.sample_id for s in samples]
                    for lane, samples in sorted(lanes.items())
                }
                for (length, reserved), lanes in sorted(self.by_context.items())
            },
            "samples": [
                self.by_id[sid].as_index_entry() for sid in sorted(self.by_id)
            ],
        }


def build_all_samples(
    packer: Packer, contexts: Iterable[Tuple[int, bool]], lanes: Sequence[str]
) -> PackedSampleStore:
    store = PackedSampleStore()
    for sequence_length, reserved_unlocked in contexts:
        for lane in lanes:
            store.add(
                sequence_length, reserved_unlocked, lane,
                packer.pack_lane(lane, sequence_length, reserved_unlocked),
            )
    return store
