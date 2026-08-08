"""Train the tokenizer once, freeze it, hash it, and verify it on every load.

"Frozen" has an operational meaning here, not a documentary one.  The merge
table is serialised to `tokenizer.json`, that file is hashed, and the hash is
written into every shard manifest.  Any component that opens a shard checks the
manifest's tokenizer hash against the tokenizer it actually holds, and refuses
the shard on a mismatch.  A shard is only meaningful under the exact tokenizer
that created it; without that check, a vocabulary change would silently
reinterpret every token id in the archive.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

from ..config import CONFIG, PATHS, SPECIAL_TOKENS
from ..hashing import sha256_text, canonical_json
from .bpe import BPETokenizer, BPETrainer

TOKENIZER_FORMAT_VERSION = "tdes-bpe-1"


class TokenizerIntegrityError(RuntimeError):
    """Raised when a shard's tokenizer hash does not match the live tokenizer."""


def train_tokenizer(texts: List[str], num_merges: int = None) -> BPETokenizer:
    merges = BPETrainer(num_merges or CONFIG.bpe_merges).train(texts)
    return BPETokenizer(merges, SPECIAL_TOKENS)


def serialise(tokenizer: BPETokenizer, corpus_hash: str) -> dict:
    return {
        "format": TOKENIZER_FORMAT_VERSION,
        "special_tokens": list(tokenizer.special_tokens),
        "num_merges": len(tokenizer.merges),
        "vocab_size": tokenizer.vocab_size,
        "merges": [list(pair) for pair in tokenizer.merges],
        "trained_on_corpus_hash": corpus_hash,
    }


def tokenizer_hash(payload: dict) -> str:
    """Hash of the tokenizer definition itself.

    Deliberately excludes anything about *when* it was trained, so retraining
    from the same corpus yields the same hash.
    """
    return sha256_text(canonical_json(payload))


def freeze(tokenizer: BPETokenizer, corpus_hash: str, out_dir: Path = None) -> Tuple[Path, str]:
    """Write tokenizer.json and return (path, hash)."""
    directory = Path(out_dir) if out_dir else PATHS.manifests
    directory.mkdir(parents=True, exist_ok=True)
    payload = serialise(tokenizer, corpus_hash)
    digest = tokenizer_hash(payload)

    document = dict(payload)
    document["tokenizer_hash"] = digest
    document["frozen_at_unix"] = int(time.time())

    path = directory / "tokenizer.json"
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (directory / "tokenizer_hash.txt").write_text(digest + "\n", encoding="utf-8")
    return path, digest


def load_frozen(path: Path = None) -> Tuple[BPETokenizer, str]:
    """Load tokenizer.json and re-verify its hash before returning it.

    The recomputation is the point.  Reading a hash out of a file and trusting
    it proves nothing; recomputing it from the merge table proves the file was
    not edited after it was frozen.
    """
    location = Path(path) if path else PATHS.manifests / "tokenizer.json"
    document = json.loads(location.read_text(encoding="utf-8"))
    recorded = document.pop("tokenizer_hash")
    document.pop("frozen_at_unix", None)

    recomputed = tokenizer_hash(document)
    if recomputed != recorded:
        raise TokenizerIntegrityError(
            f"{location.name}: recorded hash {recorded[:16]} but the merge table "
            f"hashes to {recomputed[:16]} - the frozen tokenizer was modified"
        )
    tokenizer = BPETokenizer(
        [tuple(pair) for pair in document["merges"]],
        document["special_tokens"],
    )
    return tokenizer, recorded


def verify_shard_tokenizer(manifest_hash: str, live_hash: str, shard_id: str = "") -> None:
    """Refuse a shard whose token ids were produced by a different vocabulary."""
    if manifest_hash != live_hash:
        raise TokenizerIntegrityError(
            f"shard {shard_id or '<unknown>'} was tokenized with "
            f"{str(manifest_hash)[:16]} but the live tokenizer is {live_hash[:16]}"
        )
