"""Canonical hashing primitives.

Every identity in this system is a hash of *integers and strings* - token ids,
spans, masks, manifest fields - never of a floating point number.  That is a
deliberate design choice: CPU float reductions are allowed to drift between
runs, so any hash that included a loss value would make replay verification
flaky.  Losses are compared with a tolerance instead (see `tdes.audit`).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

HASH_LEN = 16  # short hex length used for ids; full digests kept for chains


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 safe.

    Two structurally equal objects always produce the same string on any
    platform and any Python build, which is what makes the hashes comparable
    across the original run, the resume and the replay.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_obj(obj: Any) -> str:
    """Full sha256 hex digest of a canonically serialised object."""
    return sha256_text(canonical_json(obj))


def short_hash(obj: Any) -> str:
    """Short id form used for shard ids, batch ids, sample ids."""
    return hash_obj(obj)[:HASH_LEN]


def hash_token_ids(token_ids: Sequence[int]) -> str:
    """Hash a token id sequence via its little-endian uint32 byte image.

    Using the byte image rather than the JSON list makes this cheap for long
    sequences and independent of list formatting.
    """
    buf = bytearray()
    for t in token_ids:
        buf += int(t).to_bytes(4, "little", signed=False)
    return sha256_bytes(bytes(buf))


def hash_mask(mask: Sequence[int]) -> str:
    """Hash a 0/1 mask as a packed bit image."""
    out = bytearray()
    acc = 0
    nbits = 0
    for bit in mask:
        acc = (acc << 1) | (1 if bit else 0)
        nbits += 1
        if nbits == 8:
            out.append(acc)
            acc = 0
            nbits = 0
    if nbits:
        out.append(acc << (8 - nbits))
    return sha256_bytes(bytes(out))


def merkle_root(leaves: Iterable[str]) -> str:
    """Binary merkle root over hex-digest leaves.

    Used for shard content hashes so a shard's identity is verifiable block by
    block rather than only as one opaque digest.
    """
    level = [bytes.fromhex(h) for h in leaves]
    if not level:
        return sha256_bytes(b"")
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(hashlib.sha256(left + right).digest())
        level = nxt
    return level[0].hex()


def derive_seed(*parts: Any) -> int:
    """Derive a 63-bit seed from arbitrary parts.

    RNG state in this system is a *function of position* (run, branch, step)
    rather than a mutating global generator.  That is what makes resuming at
    step N produce exactly the stream step N would have produced originally.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
