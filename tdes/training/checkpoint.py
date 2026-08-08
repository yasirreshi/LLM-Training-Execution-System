"""Checkpoints that carry a data position.

"A checkpoint without a data position is incomplete."  Everything in this module
follows from that one line of the design.

A checkpoint therefore stores five things, not one:

    model weights          where the model is
    optimizer state        how the weights were moving
    scheduler state        where the learning rate was going
    RNG position           which is derived, not stored raw - see determinism.py
    ledger offset          where the data stream was

The ledger offset is the part that makes crash recovery correct rather than
approximately correct.  A step number tells you where to resume counting.  An
offset tells you where to *truncate*, and truncation is the operation actually
required, because a crash leaves the ledger describing batches that were served
to a model state which no longer exists.

Writes are atomic: temp file, fsync, rename.  The demo hard-kills the trainer
mid-step on purpose, so a half-written checkpoint has to be impossible rather
than unlikely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from ..config import CONFIG, PATHS
from ..fsutil import write_atomic_bytes, write_json
from ..hashing import hash_obj
from ..ledger.store import LedgerOffset
from .determinism import rng_fingerprint, torch

CHECKPOINT_FORMAT = "tdes-checkpoint-1"


@dataclass
class CheckpointMeta:
    checkpoint_id: str
    run_id: str
    branch_id: str
    global_step: int              # steps completed; the next step to serve
    stage: str
    ledger_offset: dict
    learning_offset: dict
    rng_fingerprint: str
    tokenizer_hash: str
    schedule_hash: str
    index_hash: str
    config_hash: str
    parent_checkpoint_id: Optional[str] = None
    tokens_consumed: int = 0
    loss_bearing_tokens_consumed: int = 0
    last_batch_id: str = ""
    next_expected_plan_hash: str = ""
    format_version: str = CHECKPOINT_FORMAT

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def state_hash(self) -> str:
        return hash_obj(self.as_dict())


def checkpoint_dir(branch_id: str, step: int, root: Path = None) -> Path:
    base = Path(root) if root else PATHS.checkpoints
    return base / f"ckpt_{branch_id}_{step:04d}"


def checkpoint_id_for(branch_id: str, step: int) -> str:
    return f"ckpt_{branch_id}_{step:04d}"


def save_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    meta: CheckpointMeta,
    root: Path = None,
) -> Path:
    """Write a checkpoint atomically.  Returns the directory."""
    t = torch()
    directory = checkpoint_dir(meta.branch_id, meta.global_step, root)
    directory.mkdir(parents=True, exist_ok=True)

    import io

    buffer = io.BytesIO()
    t.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        buffer,
    )
    write_atomic_bytes(directory / "state.pt", buffer.getvalue())
    write_json(directory / "meta.json", meta.as_dict())
    return directory


def load_checkpoint(
    directory: Path, model=None, optimizer=None, scheduler=None
) -> CheckpointMeta:
    t = torch()
    directory = Path(directory)
    meta = CheckpointMeta(**json.loads((directory / "meta.json").read_text(encoding="utf-8")))

    state = t.load(directory / "state.pt", map_location="cpu", weights_only=False)
    if model is not None:
        model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return meta


def latest_checkpoint(branch_id: str, root: Path = None) -> Optional[Path]:
    """The highest-numbered complete checkpoint for a branch.

    A directory missing `meta.json` or `state.pt` is skipped rather than
    trusted: that is what a checkpoint interrupted by the crash would look
    like, and picking it would defeat the recovery.
    """
    base = Path(root) if root else PATHS.checkpoints
    if not base.exists():
        return None
    candidates = []
    for directory in base.glob(f"ckpt_{branch_id}_*"):
        if (directory / "meta.json").exists() and (directory / "state.pt").exists():
            try:
                step = int(directory.name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.append((step, directory))
    if not candidates:
        return None
    return max(candidates)[1]


def list_checkpoints(branch_id: str = "", root: Path = None) -> List[dict]:
    base = Path(root) if root else PATHS.checkpoints
    out: List[dict] = []
    if not base.exists():
        return out
    pattern = f"ckpt_{branch_id}_*" if branch_id else "ckpt_*"
    for directory in sorted(base.glob(pattern)):
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["path"] = directory.name
        meta["complete"] = (directory / "state.pt").exists()
        out.append(meta)
    return sorted(out, key=lambda m: (m.get("branch_id", ""), m.get("global_step", 0)))


def prune_checkpoints(
    branch_id: str, keep_steps: List[int], root: Path = None
) -> List[dict]:
    """Drop the weights of superseded checkpoints, keeping their metadata.

    Real runs do this because storage is finite - the classic failure is a 200GB checkpoint that will not save because nobody
    deleted the old ones.

    What is kept and what goes is chosen carefully:

    *   `meta.json` is **never** removed.  It is small, and it carries the
        ledger offset, the RNG fingerprint and the schedule hash - which is the
        part an auditor needs to answer "where was the data stream when this
        checkpoint was taken".  Pruning that would break the audit trail.
    *   `state.pt` is removed for checkpoints nothing still depends on.  It can
        always be regenerated by re-running, and it is 99% of the bytes.

    Returns a record of what was pruned, for the ledger.
    """
    base = Path(root) if root else PATHS.checkpoints
    keep = set(keep_steps)
    pruned: List[dict] = []

    for directory in sorted(base.glob(f"ckpt_{branch_id}_*")):
        try:
            step = int(directory.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if step in keep:
            continue
        state = directory / "state.pt"
        if not state.exists():
            continue
        size = state.stat().st_size
        state.unlink()
        pruned.append(
            {"checkpoint": directory.name, "global_step": step,
             "bytes_reclaimed": size, "metadata_retained": True}
        )
    return pruned


def build_meta(
    *,
    run_id: str,
    branch_id: str,
    global_step: int,
    stage: str,
    consumption_offset: LedgerOffset,
    learning_offset: LedgerOffset,
    tokenizer_hash: str,
    schedule_hash: str,
    index_hash: str,
    tokens_consumed: int,
    loss_bearing_tokens_consumed: int,
    last_batch_id: str,
    next_expected_plan_hash: str,
    parent_checkpoint_id: Optional[str] = None,
) -> CheckpointMeta:
    return CheckpointMeta(
        checkpoint_id=checkpoint_id_for(branch_id, global_step),
        run_id=run_id,
        branch_id=branch_id,
        global_step=global_step,
        stage=stage,
        ledger_offset=consumption_offset.as_dict(),
        learning_offset=learning_offset.as_dict(),
        rng_fingerprint=rng_fingerprint(CONFIG.master_seed, branch_id, global_step),
        tokenizer_hash=tokenizer_hash,
        schedule_hash=schedule_hash,
        index_hash=index_hash,
        config_hash=CONFIG.config_hash,
        parent_checkpoint_id=parent_checkpoint_id,
        tokens_consumed=tokens_consumed,
        loss_bearing_tokens_consumed=loss_bearing_tokens_consumed,
        last_batch_id=last_batch_id,
        next_expected_plan_hash=next_expected_plan_hash,
    )
