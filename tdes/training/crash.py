"""Deliberate crash injection.

The assignment requires the final run to crash on purpose and then prove the
resume was exact.  A crash that is merely an exception proves very little - the
process unwinds, buffers flush, and destructors run.  What has to be reproduced
is the state a `SIGKILL` leaves behind, which has two distinct symptoms:

1.  **The ledger runs ahead of the model.**  Batches were served and recorded,
    then the process died before the optimizer step and before the next
    checkpoint.  The durable model state is older than the durable data record.
    This is the symptom that makes resume non-trivial: continuing from the
    ledger would skip data, and continuing from the checkpoint without
    truncating would look like a repeat.

2.  **The last ledger line is torn.**  The process died partway through a
    write, leaving bytes that are not a complete record.

`die_mid_write` reproduces both: it emits the leading fragment of a real record,
flushes it without a terminating newline, and then calls `os._exit`, which skips
cleanup handlers, buffered flushes and atexit hooks - the closest a process can
get to being killed from outside while remaining self-inflicted and therefore
deterministic enough to demonstrate on demand.

This module is only reachable when crash injection is explicitly requested by
the demo driver, and it says so in the log before it fires.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..hashing import canonical_json
from ..ledger.store import LedgerStore, compute_event_hash

CRASH_EXIT_CODE = 137          # 128 + SIGKILL, the code a killed process reports


def die_mid_write(
    ledger: LedgerStore,
    event_type: str,
    payload: Dict[str, Any],
    fraction: float = 0.55,
    logger=None,
) -> None:
    """Write a partial ledger record, then terminate immediately.

    Never returns.
    """
    seq = ledger._tail_seq + 1
    prev_hash = ledger._tail_hash
    record = {
        "seq": seq,
        "prev_hash": prev_hash,
        "event_hash": compute_event_hash(seq, prev_hash, event_type, payload),
        "type": event_type,
        "payload": payload,
    }
    line = canonical_json(record)
    cut = max(16, int(len(line) * fraction))
    fragment = line[:cut]

    if logger is not None:
        logger.event(
            "crash_injected",
            ledger=ledger.name,
            partial_bytes=len(fragment),
            full_record_bytes=len(line) + 1,
            exit_code=CRASH_EXIT_CODE,
        )

    with open(ledger.path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(fragment)
        handle.flush()
        os.fsync(handle.fileno())

    # os._exit skips atexit handlers, buffered stdout flushes and destructors.
    os._exit(CRASH_EXIT_CODE)


def arm_marker(path: Path, payload: dict) -> None:
    """Record that a crash is about to be injected, before injecting it.

    Written durably so the driver can tell a deliberate crash apart from a real
    failure when it inspects the wreckage.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(canonical_json(payload))
        handle.flush()
        os.fsync(handle.fileno())


def read_marker(path: Path) -> Optional[dict]:
    import json

    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
