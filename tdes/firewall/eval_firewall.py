"""The evaluation and validation firewall, enforced from both sides.

One side is not enough. Block the data from the registry side, *and* have the
training code ask before it consumes anything - because a copy mistake can
still happen.

So there are two independent gates:

**Registry side** - `check_admission` refuses a shard whose permission is not
`train`.  This runs when the mixture is compiled and again when a packed sample
is built, so never-train data never reaches the planner.

**Batch side** - `check_batch` runs immediately before the loss-bearing forward
pass, on the *decoded text* of what is actually about to be trained on.  It does
not trust the shard id it was handed; it looks at the tokens.  A canary string
in that text is unambiguous evidence the first gate failed.

Both gates write a `firewall_block` record when they fire, so a block is
evidence rather than a silent skip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..ledger.store import LedgerStore
from ..shards.registry import FirewallViolation, ShardRegistry
from .contamination import ContaminationHit, EvalFingerprintRegistry

SIDE_REGISTRY = "registry"
SIDE_BATCH = "batch"


@dataclass
class BlockRecord:
    side: str
    shard_id: str
    reason: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"side": self.side, "shard_id": self.shard_id,
                "reason": self.reason, "detail": self.detail}


class EvalFirewall:
    def __init__(
        self,
        registry: ShardRegistry,
        fingerprints: EvalFingerprintRegistry,
        ledger: Optional[LedgerStore] = None,
    ):
        self.registry = registry
        self.fingerprints = fingerprints
        self.ledger = ledger
        self.blocks: List[BlockRecord] = []
        self.checks_run = 0
        self.batch_checks_run = 0
        self.validation_gradient_tokens = 0     # must stay at exactly zero

    # -- recording --------------------------------------------------------

    def _record(self, record: BlockRecord) -> None:
        self.blocks.append(record)
        if self.ledger is not None:
            self.ledger.append("firewall_block", record.as_dict())

    # -- side 1: registry -------------------------------------------------

    def check_admission(self, shard_id: str, context: str = "") -> bool:
        """True when the shard may enter a loss-bearing batch.

        Returns rather than raises, because the caller is usually filtering a
        candidate list and a block here is expected behaviour, not an error.
        """
        self.checks_run += 1
        entry = self.registry.get(shard_id)
        if entry is None:
            self._record(BlockRecord(SIDE_REGISTRY, shard_id, "shard_not_registered",
                                     {"context": context}))
            return False
        if entry.permission != "train":
            self._record(
                BlockRecord(
                    SIDE_REGISTRY, shard_id,
                    f"permission_{entry.permission}",
                    {
                        "context": context,
                        "lane": entry.lane,
                        "never_train": entry.manifest.never_train,
                        "benchmark_id": entry.manifest.benchmark_id,
                        "admitted": entry.manifest.admitted,
                        "rejection_reasons": entry.manifest.rejection_reasons,
                    },
                )
            )
            return False
        return True

    # -- side 2: batch ----------------------------------------------------

    def check_batch(
        self,
        batch_id: str,
        shard_ids: Sequence[str],
        decoded_text: str,
        loss_bearing_tokens: int,
    ) -> None:
        """Last gate before gradients.  Raises FirewallViolation on any hit.

        Deliberately re-derives everything from what is in the batch: the shard
        permissions are looked up again, and the text is scanned again. If the
        registry gate had been bypassed, this is where it shows up.
        """
        self.batch_checks_run += 1

        for shard_id in shard_ids:
            entry = self.registry.get(shard_id)
            if entry is None or entry.permission != "train":
                permission = entry.permission if entry else "unregistered"
                self._record(
                    BlockRecord(SIDE_BATCH, shard_id, f"permission_{permission}",
                                {"batch_id": batch_id})
                )
                if entry is not None and entry.permission == "validation":
                    self.validation_gradient_tokens += loss_bearing_tokens
                raise FirewallViolation(
                    f"batch {batch_id} sources shard {shard_id} with permission "
                    f"'{permission}' - refusing to compute gradients"
                )

        hits: List[ContaminationHit] = self.fingerprints.scan_token_text(decoded_text)
        if hits:
            worst = max(hits, key=lambda h: h.overlap_ratio)
            self._record(
                BlockRecord(
                    SIDE_BATCH, ",".join(sorted(set(shard_ids))),
                    f"contamination_{worst.detector}",
                    {"batch_id": batch_id, "hits": [h.as_dict() for h in hits]},
                )
            )
            raise FirewallViolation(
                f"batch {batch_id} contains benchmark content "
                f"({worst.detector}, overlap {worst.overlap_ratio:.2f}, "
                f"item {worst.benchmark_doc_id}) - refusing to compute gradients"
            )

    # -- validation accounting -------------------------------------------

    def note_validation_read(self, shard_ids: Sequence[str], gradient_bearing: bool) -> None:
        """Record that validation data was read for evaluation.

        Reading it is allowed.  Letting it produce a gradient is not, and the
        counter this maintains is asserted to be zero in the evidence bundle.
        """
        if not gradient_bearing:
            return
        for shard_id in shard_ids:
            entry = self.registry.get(shard_id)
            if entry is not None and entry.permission == "validation":
                self.validation_gradient_tokens += 1
                self._record(
                    BlockRecord(SIDE_BATCH, shard_id, "validation_gradient_leak", {})
                )

    # -- reporting --------------------------------------------------------

    def report(self) -> dict:
        by_side: Dict[str, int] = {}
        by_reason: Dict[str, int] = {}
        for block in self.blocks:
            by_side[block.side] = by_side.get(block.side, 0) + 1
            by_reason[block.reason] = by_reason.get(block.reason, 0) + 1
        return {
            "registry_checks": self.checks_run,
            "batch_checks": self.batch_checks_run,
            "blocks_total": len(self.blocks),
            "blocks_by_side": dict(sorted(by_side.items())),
            "blocks_by_reason": dict(sorted(by_reason.items())),
            "validation_gradient_bearing_tokens": self.validation_gradient_tokens,
            "blocked_shard_ids": sorted({b.shard_id for b in self.blocks}),
            "fingerprint_registry": self.fingerprints.as_dict(),
            "blocks": [b.as_dict() for b in self.blocks],
        }
