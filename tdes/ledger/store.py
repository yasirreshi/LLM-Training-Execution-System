"""Append-only, hash-chained ledger.

This is the run's memory.  Three properties matter and each is enforced here
rather than assumed:

1. **Append-only.**  Records are only ever added at the end.  The one exception
   is `rollback_to`, which truncates back to a checkpoint's recorded offset -
   and that operation is itself logged, with the hashes of everything it
   discarded, so it is auditable.

2. **Hash chained.**  Each record carries `prev_hash` and `event_hash`.  A
   modified or removed record breaks the chain, and `verify_chain` finds it.
   This is what lets the evidence verifier claim the ledger was not doctored.

3. **Crash safe.**  Every append is flushed and fsynced, so a complete line is
   durable.  A hard kill mid-write leaves a *torn final line*; `repair_torn_tail`
   detects it, truncates it, and reports it.  The corruption is expected - the
   demo deliberately causes it - and its clean repair is part of the evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..hashing import canonical_json, sha256_text

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class LedgerOffset:
    """A position in a ledger, durable enough to resume from.

    Stored inside every checkpoint.  A checkpoint without a data position is
    incomplete (the execution layer, section 14).
    """

    byte_offset: int
    event_seq: int          # seq of the last record at/ before this offset; -1 if empty
    last_event_hash: str

    def as_dict(self) -> dict:
        return {
            "byte_offset": self.byte_offset,
            "event_seq": self.event_seq,
            "last_event_hash": self.last_event_hash,
        }

    @staticmethod
    def from_dict(d: dict) -> "LedgerOffset":
        return LedgerOffset(
            byte_offset=int(d["byte_offset"]),
            event_seq=int(d["event_seq"]),
            last_event_hash=str(d["last_event_hash"]),
        )

    @staticmethod
    def empty() -> "LedgerOffset":
        return LedgerOffset(0, -1, GENESIS_HASH)


def compute_event_hash(seq: int, prev_hash: str, event_type: str, payload: Any) -> str:
    return sha256_text(
        canonical_json(
            {"seq": seq, "prev_hash": prev_hash, "type": event_type, "payload": payload}
        )
    )


class LedgerStore:
    """One append-only JSONL ledger file."""

    def __init__(self, path: Path, name: str = ""):
        self.path = Path(path)
        self.name = name or self.path.stem
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_bytes(b"")
        self._tail_seq, self._tail_hash = self._scan_tail()

    # -- internals --------------------------------------------------------

    def _scan_tail(self) -> Tuple[int, str]:
        seq, last_hash = -1, GENESIS_HASH
        for rec in self.read_all():
            seq = rec["seq"]
            last_hash = rec["event_hash"]
        return seq, last_hash

    # -- writing ----------------------------------------------------------

    def append(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        seq = self._tail_seq + 1
        prev_hash = self._tail_hash
        event_hash = compute_event_hash(seq, prev_hash, event_type, payload)
        record = {
            "seq": seq,
            "prev_hash": prev_hash,
            "event_hash": event_hash,
            "type": event_type,
            "payload": payload,
        }
        line = canonical_json(record) + "\n"
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._tail_seq, self._tail_hash = seq, event_hash
        return record

    def append_many(self, event_type: str, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.append(event_type, p) for p in payloads]

    # -- reading ----------------------------------------------------------

    def read_all(self, strict: bool = False) -> List[Dict[str, Any]]:
        """Parse every complete record.

        An incomplete trailing line (hard kill mid-write) is ignored here rather
        than raised, so recovery code can open a torn ledger.  A *complete* but
        unparsable line is skipped too - which leaves a gap in `seq` that
        `verify_chain` will report, so corruption is never silently absorbed.
        """
        out: List[Dict[str, Any]] = []
        if not self.path.exists():
            return out
        raw = self.path.read_bytes()
        if not raw:
            return out
        lines = raw.split(b"\n")
        # split always yields a final element: b"" when the file ends with a
        # newline (every line complete), otherwise the torn fragment.  Either
        # way the complete records are everything before it.
        torn = lines[-1] if lines[-1] != b"" else None
        for chunk in lines[:-1]:
            if not chunk.strip():
                continue
            try:
                out.append(json.loads(chunk.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if strict:
                    raise
                continue
        if torn is not None and strict:
            raise ValueError(f"{self.path.name}: torn final line of {len(torn)} bytes")
        return out

    def read_range(self, start_seq: int, end_seq: int) -> List[Dict[str, Any]]:
        return [r for r in self.read_all() if start_seq <= r["seq"] < end_seq]

    def __len__(self) -> int:
        return self._tail_seq + 1

    # -- offsets ----------------------------------------------------------

    def current_offset(self) -> LedgerOffset:
        size = self.path.stat().st_size if self.path.exists() else 0
        return LedgerOffset(size, self._tail_seq, self._tail_hash)

    # -- integrity --------------------------------------------------------

    def repair_torn_tail(self) -> Optional[Dict[str, Any]]:
        """Truncate an incomplete final line left by a hard kill.

        Returns a description of what was removed, or None if the file was
        already clean.
        """
        if not self.path.exists():
            return None
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            # Even a newline-terminated file can hold a garbled last record.
            if raw:
                last = raw.rstrip(b"\n").split(b"\n")[-1]
                try:
                    json.loads(last.decode("utf-8"))
                except Exception:
                    keep = raw.rstrip(b"\n")[: -len(last)]
                    self._truncate(len(keep))
                    return {"ledger": self.name, "removed_bytes": len(raw) - len(keep),
                            "reason": "unparsable_final_record"}
            return None
        torn = raw.split(b"\n")[-1]
        keep_len = len(raw) - len(torn)
        self._truncate(keep_len)
        return {"ledger": self.name, "removed_bytes": len(torn),
                "reason": "torn_final_line",
                "torn_fragment_preview": torn[:120].decode("utf-8", "replace")}

    def _truncate(self, size: int) -> None:
        with open(self.path, "r+b") as fh:
            fh.truncate(size)
            fh.flush()
            os.fsync(fh.fileno())
        self._tail_seq, self._tail_hash = self._scan_tail()

    def rollback_to(self, offset: LedgerOffset) -> List[Dict[str, Any]]:
        """Truncate back to a checkpoint's offset, returning discarded records.

        Needed because a crash leaves the ledger *ahead* of the surviving model
        state: those events describe batches consumed by a model that no longer
        exists.  Keeping them would make resume look like it skipped work.  The
        discarded hashes are returned so the caller can log them and later prove
        the re-consumed batches are identical.
        """
        discarded = [r for r in self.read_all() if r["seq"] > offset.event_seq]
        self._truncate(offset.byte_offset)
        if self._tail_seq != offset.event_seq or self._tail_hash != offset.last_event_hash:
            raise RuntimeError(
                f"{self.name}: rollback landed at seq={self._tail_seq} "
                f"hash={self._tail_hash[:12]} but checkpoint recorded "
                f"seq={offset.event_seq} hash={offset.last_event_hash[:12]}"
            )
        return discarded

    def verify_chain(self) -> Tuple[bool, Dict[str, Any]]:
        """Recompute the whole hash chain.  Any edit anywhere shows up here."""
        prev = GENESIS_HASH
        expected_seq = 0
        for rec in self.read_all():
            if rec["seq"] != expected_seq:
                return False, {"ledger": self.name, "error": "seq_gap",
                               "at": rec["seq"], "expected": expected_seq}
            if rec["prev_hash"] != prev:
                return False, {"ledger": self.name, "error": "prev_hash_mismatch",
                               "at": rec["seq"]}
            recomputed = compute_event_hash(
                rec["seq"], rec["prev_hash"], rec["type"], rec["payload"]
            )
            if recomputed != rec["event_hash"]:
                return False, {"ledger": self.name, "error": "event_hash_mismatch",
                               "at": rec["seq"], "recomputed": recomputed[:16],
                               "recorded": str(rec["event_hash"])[:16]}
            prev = rec["event_hash"]
            expected_seq += 1
        return True, {"ledger": self.name, "records": expected_seq, "head": prev}
