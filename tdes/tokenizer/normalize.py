"""Text normalization for the frozen tokenizer contract.

The tokenizer contract: the same raw text must always produce the same token ids.
That requires pinning the normalization, not only the merge table, because a
normalization change silently repartitions the vocabulary.

The Indic-specific care here is not decoration.  Two things go wrong routinely:

*   **Nukta spellings.**  क़ can be written as U+0958, or as क (U+0915) followed
    by the nukta sign (U+093C).  They render identically.  NFC maps both to the
    decomposed pair, because U+0958..U+095F are in Unicode's composition
    exclusion table, so applying NFC is what makes the two spellings collide
    into one token sequence instead of two.

*   **Joiners.**  ZWJ (U+200D) and ZWNJ (U+200C) select between conjunct and
    half-form rendering.  क्‍ष and क्ष are different code point sequences and
    can be different words.  A normalizer that strips joiners "to clean things
    up" corrupts the text.  This one keeps them, and a test asserts it.

What is removed: the BOM, and the invisible formatting characters that carry no
linguistic content (soft hyphen, word joiner, directional marks).  Those do
merge tokens spuriously and nothing is lost by dropping them.
"""

from __future__ import annotations

import re
import unicodedata

ZWJ = "‍"
ZWNJ = "‌"

# Invisible characters that carry no linguistic content in this corpus.
# Deliberately does NOT include ZWJ/ZWNJ.
_STRIP_CHARS = (
    "﻿"  # byte order mark
    "­"  # soft hyphen
    "⁠"  # word joiner
    "​"  # zero width space
    "‎"  # left-to-right mark
    "‏"  # right-to-left mark
)
_STRIP_RE = re.compile("[" + re.escape(_STRIP_CHARS) + "]")

_TRAILING_WS_RE = re.compile(r"[ \t]+(?=\n)")
_MANY_BLANKS_RE = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Canonicalise text.  Idempotent: normalize(normalize(x)) == normalize(x)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _STRIP_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _MANY_BLANKS_RE.sub("\n\n", text)
    return text.strip() + "\n"


def preserves_joiners(text: str) -> bool:
    """True when normalization kept every ZWJ/ZWNJ the input had."""
    normalized = normalize(text)
    return (
        text.count(ZWJ) == normalized.count(ZWJ)
        and text.count(ZWNJ) == normalized.count(ZWNJ)
    )


def script_of(text: str) -> str:
    """Best-effort dominant script tag, used for manifest metadata.

    Counting by Unicode block rather than by language guess, because the block
    is a property of the bytes and a language is not.
    """
    counts: dict = {}
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            key = "Deva"
        elif 0x0980 <= code <= 0x09FF:
            key = "Beng"
        elif 0x0B80 <= code <= 0x0BFF:
            key = "Taml"
        elif 0x0C00 <= code <= 0x0C7F:
            key = "Telu"
        elif 0x0A80 <= code <= 0x0AFF:
            key = "Gujr"
        elif ch.isalpha() and code < 0x0250:
            key = "Latn"
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "Zyyy"
    return max(sorted(counts), key=lambda k: counts[k])
