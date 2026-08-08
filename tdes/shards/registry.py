"""The shard registry: one place that knows every shard and its permission.

Train shards are admitted into the data stream, test shards are registered
*precisely so they can be blocked*, and validation shards sit in between -
readable for evaluation, never gradient-bearing.

Registering test data rather than excluding it is the important idea.  The
system has to know the evaluation set exists in order to prevent it entering a
batch, and in order to answer "was this benchmark jump real" afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..config import PATHS
from ..fsutil import write_json
from .manifest import ShardManifest

PERMISSION_TRAIN = "train"                  # admitted, gradient-bearing
PERMISSION_VALIDATION = "validation"        # readable, never gradient-bearing
PERMISSION_TEST = "test"                    # never read by training at all
PERMISSION_BLOCKED = "blocked"              # failed admission


@dataclass
class RegistryEntry:
    manifest: ShardManifest
    permission: str

    @property
    def shard_id(self) -> str:
        return self.manifest.shard_id

    @property
    def lane(self) -> str:
        return self.manifest.lane


class ShardRegistry:
    def __init__(self, manifests: Iterable[ShardManifest]):
        self.entries: Dict[str, RegistryEntry] = {}
        for manifest in sorted(manifests, key=lambda m: m.shard_id):
            self.entries[manifest.shard_id] = RegistryEntry(
                manifest, _permission_for(manifest)
            )

    # -- lookup -----------------------------------------------------------

    def __contains__(self, shard_id: str) -> bool:
        return shard_id in self.entries

    def get(self, shard_id: str) -> Optional[RegistryEntry]:
        return self.entries.get(shard_id)

    def manifest(self, shard_id: str) -> ShardManifest:
        return self.entries[shard_id].manifest

    def by_permission(self, permission: str) -> List[RegistryEntry]:
        return [e for e in self.entries.values() if e.permission == permission]

    @property
    def trainable(self) -> List[RegistryEntry]:
        return self.by_permission(PERMISSION_TRAIN)

    @property
    def validation(self) -> List[RegistryEntry]:
        return self.by_permission(PERMISSION_VALIDATION)

    @property
    def test(self) -> List[RegistryEntry]:
        return self.by_permission(PERMISSION_TEST)

    @property
    def blocked(self) -> List[RegistryEntry]:
        return self.by_permission(PERMISSION_BLOCKED)

    def trainable_by_lane(self, lane: str) -> List[RegistryEntry]:
        return [e for e in self.trainable if e.lane == lane]

    # -- the registry side of the firewall --------------------------------

    def assert_trainable(self, shard_id: str) -> None:
        """Refuse a shard that is not allowed to produce gradients.

        Called by the packer before a shard's tokens are placed into a
        loss-bearing sample.  This is the first of the two firewall sides; the
        second runs at batch build time on the decoded text, so a shard that
        somehow reached this point with the wrong permission is still caught.
        """
        entry = self.entries.get(shard_id)
        if entry is None:
            raise FirewallViolation(f"shard {shard_id} is not in the registry")
        if entry.permission != PERMISSION_TRAIN:
            raise FirewallViolation(
                f"shard {shard_id} has permission '{entry.permission}' and may not "
                f"enter a loss-bearing batch"
            )

    def is_trainable(self, shard_id: str) -> bool:
        entry = self.entries.get(shard_id)
        return entry is not None and entry.permission == PERMISSION_TRAIN

    # -- reporting --------------------------------------------------------

    def summary(self) -> dict:
        by_lane: Dict[str, Dict[str, int]] = {}
        for entry in self.entries.values():
            lane = by_lane.setdefault(entry.lane, {})
            lane[entry.permission] = lane.get(entry.permission, 0) + 1
            lane["tokens"] = lane.get("tokens", 0) + entry.manifest.token_count
        return {
            "total_shards": len(self.entries),
            "train": len(self.trainable),
            "validation": len(self.validation),
            "test": len(self.test),
            "blocked": len(self.blocked),
            "trainable_tokens": sum(e.manifest.token_count for e in self.trainable),
            "by_lane": {k: by_lane[k] for k in sorted(by_lane)},
            "permissions": {
                shard_id: entry.permission
                for shard_id, entry in sorted(self.entries.items())
            },
        }

    def write(self, path: Path = None) -> Path:
        target = Path(path) if path else PATHS.manifests / "shard_registry.json"
        return write_json(target, self.summary())


class FirewallViolation(RuntimeError):
    """Raised when data without training permission reaches a gradient path."""


def _permission_for(manifest: ShardManifest) -> str:
    if manifest.never_train:
        return PERMISSION_TEST
    if manifest.held_out or not manifest.loss_bearing:
        return PERMISSION_VALIDATION
    if not manifest.admitted:
        return PERMISSION_BLOCKED
    return PERMISSION_TRAIN
