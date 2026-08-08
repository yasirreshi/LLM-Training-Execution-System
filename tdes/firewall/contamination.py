"""Contamination fingerprints for the evaluation firewall.

Three detectors, because each one misses what the others catch:

*   **Content hash.**  Exact duplicate of a benchmark item.  Free, and useless
    the moment anyone reformats a line.
*   **N-gram fingerprints.**  Overlapping word n-grams from every benchmark
    item.  Catches lightly edited copies, which is the realistic case.  Needs a
    ratio threshold, because a low overlap is just ordinary shared phrasing.
*   **Canary strings.**  Unique markers planted in the test set.  A canary in a
    loss-bearing batch is unambiguous - there is no innocent explanation.

The ratio matters.  A 0.9 overlap over 180 tokens is a copy.  A 0.35 overlap
over 30 tokens is two documents both containing the phrase "in one hour the
pipe fills".  Reporting both as "contamination" trains people to ignore the
alarm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set

from ..hashing import sha256_text

NGRAM_SIZE = 8
DEFAULT_OVERLAP_THRESHOLD = 0.55

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_CANARY_RE = re.compile(r"TDES-CANARY-[0-9a-zA-Z\-]+")


def words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def ngram_fingerprints(text: str, n: int = NGRAM_SIZE) -> Set[str]:
    tokens = words(text)
    if len(tokens) < n:
        return {sha256_text(" ".join(tokens))} if tokens else set()
    return {
        sha256_text(" ".join(tokens[i:i + n]))[:16]
        for i in range(len(tokens) - n + 1)
    }


def extract_canaries(text: str) -> Set[str]:
    return set(_CANARY_RE.findall(text))


@dataclass
class ContaminationHit:
    benchmark_doc_id: str
    benchmark_id: str
    detector: str            # content_hash | ngram | canary
    overlap_ratio: float
    matched_ngrams: int = 0
    canary: str = ""

    def as_dict(self) -> dict:
        return {
            "benchmark_doc_id": self.benchmark_doc_id,
            "benchmark_id": self.benchmark_id,
            "detector": self.detector,
            "overlap_ratio": round(self.overlap_ratio, 4),
            "matched_ngrams": self.matched_ngrams,
            "canary": self.canary,
        }


@dataclass
class EvalFingerprintRegistry:
    """Fingerprints of everything that must never be trained on."""

    content_hashes: Dict[str, str] = field(default_factory=dict)      # hash -> doc_id
    ngrams: Dict[str, Set[str]] = field(default_factory=dict)         # doc_id -> fingerprints
    canaries: Dict[str, str] = field(default_factory=dict)            # canary -> doc_id
    benchmark_of: Dict[str, str] = field(default_factory=dict)        # doc_id -> benchmark id
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD

    # -- construction -----------------------------------------------------

    def register(self, doc_id: str, text: str, benchmark_id: str = "") -> None:
        self.content_hashes[sha256_text(text)] = doc_id
        self.ngrams[doc_id] = ngram_fingerprints(text)
        self.benchmark_of[doc_id] = benchmark_id
        for canary in extract_canaries(text):
            self.canaries[canary] = doc_id

    def register_raw_canaries(self, text: str, doc_id: str = "registry-file") -> None:
        """Pick up canaries declared in a file header rather than a document."""
        for canary in extract_canaries(text):
            self.canaries[canary] = doc_id

    # -- scanning ---------------------------------------------------------

    def scan_text(self, text: str) -> List[ContaminationHit]:
        hits: List[ContaminationHit] = []

        digest = sha256_text(text)
        if digest in self.content_hashes:
            doc_id = self.content_hashes[digest]
            hits.append(ContaminationHit(doc_id, self.benchmark_of.get(doc_id, ""),
                                         "content_hash", 1.0))

        for canary, doc_id in self.canaries.items():
            if canary in text:
                hits.append(ContaminationHit(doc_id, self.benchmark_of.get(doc_id, ""),
                                             "canary", 1.0, canary=canary))

        candidate = ngram_fingerprints(text)
        if candidate:
            for doc_id, reference in self.ngrams.items():
                if not reference:
                    continue
                shared = len(reference & candidate)
                if shared == 0:
                    continue
                # Ratio against the *benchmark item*, not the candidate: a long
                # document that quotes one short test item in full is fully
                # contaminated with respect to that item, and dividing by the
                # long document's length would hide it.
                ratio = shared / len(reference)
                if ratio >= self.overlap_threshold:
                    hits.append(
                        ContaminationHit(doc_id, self.benchmark_of.get(doc_id, ""),
                                         "ngram", ratio, matched_ngrams=shared)
                    )
        return hits

    def scan_token_text(self, text: str) -> List[ContaminationHit]:
        """Batch-time scan.  Same detectors, kept as a named entry point so the
        two firewall sides are distinguishable in the code and in the log."""
        return self.scan_text(text)

    def as_dict(self) -> dict:
        return {
            "benchmark_items": sorted(self.ngrams),
            "content_hash_count": len(self.content_hashes),
            "ngram_size": NGRAM_SIZE,
            "fingerprints_per_item": {k: len(v) for k, v in sorted(self.ngrams.items())},
            "canaries": sorted(self.canaries),
            "overlap_threshold": self.overlap_threshold,
        }


def build_registry(test_documents: Iterable, raw_files: Sequence[str] = ()) -> EvalFingerprintRegistry:
    """Build the registry from the held-out test documents."""
    registry = EvalFingerprintRegistry()
    for doc in test_documents:
        registry.register(doc.doc_id, doc.text, getattr(doc, "benchmark_id", "") or "")
    for raw in raw_files:
        registry.register_raw_canaries(raw)
    return registry
