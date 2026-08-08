"""Durable file operations.

Two rules the rest of the system relies on:

*   Nothing is ever written in place.  Writes go to a temporary file in the same
    directory, are fsynced, then renamed over the target.  A reader - or a
    crash - therefore sees the complete old file or the complete new one, never
    a half-written mixture.  This is what lets a checkpoint survive the hard
    kill in the middle of the demo.

*   A shard, once written, is made read-only.  It is an immutable training
    object; if it needs to change it becomes a new shard with a new hash and a
    new lineage.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Union

from .hashing import canonical_json


REPLACE_ATTEMPTS = 6


def write_atomic_bytes(path: Union[str, Path], data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _replace_with_retry(tmp: str, path: Path) -> None:
    """os.replace, hardened for Windows.

    Two things can make the rename fail on Windows even though the data is
    already durable in the temp file:

    *   the destination is marked read-only, which shards deliberately are;
    *   another process holds a transient handle on it - a virus scanner
        opening the freshly created file is the usual culprit, and it clears
        within milliseconds.

    Clearing the read-only bit handles the first.  A short backoff handles the
    second.  If the rename still will not go through, fall back to removing the
    destination first; that briefly leaves no file at the target path, which is
    weaker than a true atomic swap, so it is only used as a last resort.
    """
    last: OSError = None
    for attempt in range(REPLACE_ATTEMPTS):
        _unlock(path)
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last = exc
            time.sleep(0.02 * (attempt + 1))
    try:
        if path.exists():
            _unlock(path)
            os.unlink(path)
        os.rename(tmp, path)
        return
    except OSError:
        pass
    raise last


def write_atomic_text(path: Union[str, Path], text: str) -> Path:
    return write_atomic_bytes(path, text.encode("utf-8"))


def write_json(path: Union[str, Path], obj: Any, indent: int = 2) -> Path:
    """Human-readable JSON artifact (manifests, reports)."""
    return write_atomic_text(
        path, json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"
    )


def write_canonical_json(path: Union[str, Path], obj: Any) -> Path:
    """Canonical JSON - byte-stable, for anything that gets hashed or diffed."""
    return write_atomic_text(path, canonical_json(obj) + "\n")


def read_json(path: Union[str, Path]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_readonly(path: Union[str, Path]) -> None:
    """Drop write permission.  Works on POSIX and Windows."""
    path = Path(path)
    mode = path.stat().st_mode
    path.chmod(mode & ~stat.S_IWRITE & ~stat.S_IWGRP & ~stat.S_IWOTH)


def is_readonly(path: Union[str, Path]) -> bool:
    return not (Path(path).stat().st_mode & stat.S_IWRITE)


def _unlock(path: Path) -> None:
    """Restore write permission so a read-only file can be replaced."""
    if path.exists():
        try:
            path.chmod(path.stat().st_mode | stat.S_IWRITE)
        except OSError:
            pass


def rmtree(path: Union[str, Path]) -> None:
    """Remove a tree that may contain read-only shard files."""
    import shutil

    path = Path(path)
    if not path.exists():
        return

    def _on_error(func, name, _exc):
        try:
            os.chmod(name, stat.S_IWRITE)
            func(name)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_on_error)


def tokens_to_bytes(token_ids) -> bytes:
    """uint32 little-endian image of a token id array."""
    buf = bytearray()
    for token in token_ids:
        buf += int(token).to_bytes(4, "little", signed=False)
    return bytes(buf)


def bytes_to_tokens(data: bytes) -> list:
    return [
        int.from_bytes(data[i:i + 4], "little", signed=False)
        for i in range(0, len(data), 4)
    ]
