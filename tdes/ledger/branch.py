"""Branch lineage: making "which data stream" part of the experiment definition.

The design's rule for a trustworthy comparison:

    experiment = model checkpoint + optimizer state + data stream + code/config

Restoring an old checkpoint and letting the loader produce whatever the current
seed and shard set happen to generate satisfies the first two and quietly breaks
the third.  The model starts from known weights, the data silently changed, and
if the new run looks better nobody can say whether the change or the data did it.

So a run is always bound to a branch, and there are only two legitimate ways to
continue from an old checkpoint:

*   **replay** - feed the same historical stream, recorded in the ledger;
*   **fork** - start a new branch, with a new id, and record the exact point of
    divergence so every later difference is attributable.

This module keeps the lineage so both remain answerable later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..config import PATHS
from ..fsutil import write_json
from ..hashing import short_hash

MODE_PRIMARY = "primary"
MODE_RESUME = "resume"
MODE_REPLAY = "replay"
MODE_FORK = "fork"


@dataclass
class Branch:
    branch_id: str
    run_id: str
    mode: str
    parent_branch_id: Optional[str] = None
    forked_from_step: Optional[int] = None
    forked_from_checkpoint: Optional[str] = None
    divergence_note: str = ""
    seed: Optional[int] = None
    schedule_hash: str = ""
    index_hash: str = ""
    config_hash: str = ""
    steps_completed: int = 0

    def as_dict(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "run_id": self.run_id,
            "mode": self.mode,
            "parent_branch_id": self.parent_branch_id,
            "forked_from_step": self.forked_from_step,
            "forked_from_checkpoint": self.forked_from_checkpoint,
            "divergence_note": self.divergence_note,
            "seed": self.seed,
            "schedule_hash": self.schedule_hash,
            "index_hash": self.index_hash,
            "config_hash": self.config_hash,
            "steps_completed": self.steps_completed,
        }


class BranchRegistry:
    def __init__(self, path: Path = None):
        self.path = Path(path) if path else PATHS.ledgers / "branches.json"
        self.branches: Dict[str, Branch] = {}

    def register(self, branch: Branch) -> Branch:
        self.branches[branch.branch_id] = branch
        return branch

    def get(self, branch_id: str) -> Optional[Branch]:
        return self.branches.get(branch_id)

    def fork(
        self,
        parent: Branch,
        from_step: int,
        from_checkpoint: str,
        note: str,
        seed: Optional[int] = None,
    ) -> Branch:
        branch_id = "br-" + short_hash(
            {
                "parent": parent.branch_id,
                "step": from_step,
                "checkpoint": from_checkpoint,
                "note": note,
            }
        )[:10]
        return self.register(
            Branch(
                branch_id=branch_id,
                run_id=parent.run_id,
                mode=MODE_FORK,
                parent_branch_id=parent.branch_id,
                forked_from_step=from_step,
                forked_from_checkpoint=from_checkpoint,
                divergence_note=note,
                seed=seed if seed is not None else parent.seed,
                config_hash=parent.config_hash,
            )
        )

    def lineage(self, branch_id: str) -> List[str]:
        chain, cursor = [], branch_id
        while cursor:
            chain.append(cursor)
            branch = self.branches.get(cursor)
            cursor = branch.parent_branch_id if branch else None
        return list(reversed(chain))

    def as_dict(self) -> dict:
        return {
            "branches": {
                branch_id: branch.as_dict()
                for branch_id, branch in sorted(self.branches.items())
            },
            "lineages": {
                branch_id: self.lineage(branch_id) for branch_id in sorted(self.branches)
            },
        }

    def write(self, path: Path = None) -> Path:
        return write_json(path or self.path, self.as_dict())


def divergence_report(
    parent_batches: Dict[int, str], fork_batches: Dict[int, str], fork_point: int
) -> dict:
    """Prove a fork actually diverged, and only after the divergence point.

    Both halves matter.  Identical batch ids after the fork point would mean the
    fork changed nothing, so the comparison would be measuring noise.  Differing
    batch ids *before* it would mean the fork did not really start from the
    parent's state, so the comparison would be measuring two unrelated runs.
    """
    shared = sorted(set(parent_batches) & set(fork_batches))
    before = [s for s in shared if s < fork_point]
    after = [s for s in shared if s >= fork_point]

    identical_before = [s for s in before if parent_batches[s] == fork_batches[s]]
    differing_after = [s for s in after if parent_batches[s] != fork_batches[s]]

    return {
        "fork_point_step": fork_point,
        "shared_steps": shared,
        "steps_before_fork": before,
        "steps_after_fork": after,
        "identical_before_fork": identical_before,
        "differing_after_fork": differing_after,
        "all_identical_before_fork": len(identical_before) == len(before),
        "any_divergence_after_fork": bool(differing_after),
        "diverged_correctly": (len(identical_before) == len(before)) and bool(differing_after),
        "parent_batch_ids": {str(k): v for k, v in sorted(parent_batches.items())},
        "fork_batch_ids": {str(k): v for k, v in sorted(fork_batches.items())},
    }
