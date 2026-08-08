"""Read the bundled corpus and attach the source contract provenance to every document.

Nothing is downloaded.  The whole corpus is in `corpus/`, so the run is offline
and the documents are byte-identical on every machine, which is a precondition
for shards whose content hashes reproduce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config import PATHS
from ..hashing import sha256_text
from ..tokenizer.normalize import normalize, script_of

DOC_SEPARATOR = "===DOC==="
HEADER_SEPARATOR = "---"

# Role markers for the agentic and reasoning lanes.  The role decides the loss
# mask: context roles are conditioned on, graded roles are learned.
ROLE_MARKERS = {
    "@user:": "user",
    "@think:": "think",
    "@tool_call:": "tool_call",
    "@tool_result:": "tool_result",
    "@answer:": "answer",
}

CONTEXT_ROLES = frozenset({"user", "tool_result"})
GRADED_ROLES = frozenset({"think", "tool_call", "answer"})

# Which special token announces each role in the token stream.
ROLE_SPECIAL_TOKEN = {
    "user": "<user>",
    "think": "<think>",
    "tool_call": "<tool_call>",
    "tool_result": "<tool_result>",
    "answer": "<answer>",
    "assistant": "<assistant>",
}


@dataclass
class Span:
    role: str            # "text" for plain prose, otherwise a marker role
    text: str

    @property
    def graded(self) -> bool:
        return self.role == "text" or self.role in GRADED_ROLES


@dataclass
class Document:
    doc_id: str
    source_id: str
    lane: str
    lang: str
    script: str
    title: str
    spans: List[Span]
    stage_hint: str = "early"
    reserved: bool = False
    min_context: int = 0
    file: str = ""

    @property
    def text(self) -> str:
        return "\n".join(span.text for span in self.spans)

    @property
    def content_hash(self) -> str:
        return sha256_text(self.text)

    @property
    def structured(self) -> bool:
        return any(span.role != "text" for span in self.spans)


@dataclass
class Source:
    source_id: str
    file: str
    lane: str
    provenance: str
    licence: str
    licence_tier: str
    cleaning_pipeline: Optional[str]
    dedup_status: str
    pii_status: str
    contamination_status: str
    held_out: bool
    loss_bearing: bool
    capability_tags: List[str] = field(default_factory=list)
    never_train: bool = False
    benchmark_id: Optional[str] = None
    scarce_tier: Optional[str] = None
    documents: List[Document] = field(default_factory=list)

    def as_manifest_fields(self) -> dict:
        return {
            "source_id": self.source_id,
            "provenance": self.provenance,
            "licence": self.licence,
            "licence_tier": self.licence_tier,
            "cleaning_pipeline_hash": self.cleaning_pipeline,
            "dedup_status": self.dedup_status,
            "pii_status": self.pii_status,
            "contamination_status": self.contamination_status,
            "held_out": self.held_out,
            "loss_bearing": self.loss_bearing,
            "never_train": self.never_train,
            "benchmark_id": self.benchmark_id,
            "scarce_tier": self.scarce_tier,
            "capability_tags": sorted(self.capability_tags),
        }


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _split_roles(body: str) -> List[Span]:
    """Split a body into role spans, or return one plain span if unmarked."""
    lines = body.split("\n")
    has_marker = any(
        line.startswith(marker) for line in lines for marker in ROLE_MARKERS
    )
    if not has_marker:
        return [Span("text", body.strip())]

    spans: List[Span] = []
    current_role: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        if current_role is None:
            return
        text = "\n".join(buffer).strip()
        if text:
            spans.append(Span(current_role, text))

    for line in lines:
        matched = None
        for marker, role in ROLE_MARKERS.items():
            if line.startswith(marker):
                matched = (marker, role)
                break
        if matched:
            flush()
            marker, current_role = matched
            buffer = [line[len(marker):].strip()]
        else:
            buffer.append(line)
    flush()
    return spans


def parse_corpus_file(path: Path, source: Source) -> List[Document]:
    raw = normalize(path.read_text(encoding="utf-8"))
    chunks = raw.split(DOC_SEPARATOR)
    documents: List[Document] = []

    for chunk in chunks[1:]:            # chunks[0] is the file header comment
        chunk = chunk.strip("\n")
        if not chunk.strip():
            continue
        if f"\n{HEADER_SEPARATOR}\n" not in chunk:
            raise ValueError(f"{path.name}: document without a '---' header separator")
        header_text, body = chunk.split(f"\n{HEADER_SEPARATOR}\n", 1)

        meta: Dict[str, str] = {}
        for line in header_text.strip().split("\n"):
            if not line.strip() or ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()

        for required in ("doc_id", "lang", "script"):
            if required not in meta:
                raise ValueError(f"{path.name}: document missing '{required}'")

        spans = _split_roles(body)
        documents.append(
            Document(
                doc_id=meta["doc_id"],
                source_id=source.source_id,
                lane=source.lane,
                lang=meta["lang"],
                script=meta.get("script") or script_of(body),
                title=meta.get("title", ""),
                spans=spans,
                stage_hint=meta.get("stage_hint", "early"),
                reserved=_parse_bool(meta.get("reserved", "false")),
                min_context=int(meta.get("min_context", "0")),
                file=source.file,
            )
        )
    return documents


def load_sources(corpus_root: Path = None) -> List[Source]:
    """Load sources.json and every document it points at."""
    root = Path(corpus_root) if corpus_root else PATHS.corpus
    spec = json.loads((root / "sources.json").read_text(encoding="utf-8"))

    sources: List[Source] = []
    for entry in spec["sources"]:
        source = Source(
            source_id=entry["source_id"],
            file=entry["file"],
            lane=entry["lane"],
            provenance=entry["provenance"],
            licence=entry["licence"],
            licence_tier=entry["licence_tier"],
            cleaning_pipeline=entry.get("cleaning_pipeline"),
            dedup_status=entry.get("dedup_status", "unknown"),
            pii_status=entry.get("pii_status", "unknown"),
            contamination_status=entry.get("contamination_status", "not_scanned"),
            held_out=bool(entry.get("held_out", False)),
            loss_bearing=bool(entry.get("loss_bearing", True)),
            capability_tags=list(entry.get("capability_tags", [])),
            never_train=bool(entry.get("never_train", False)),
            benchmark_id=entry.get("benchmark_id"),
            scarce_tier=entry.get("scarce_tier"),
        )
        source.documents = parse_corpus_file(root / source.file, source)
        sources.append(source)

    # Deterministic order so shard ids do not depend on filesystem enumeration.
    sources.sort(key=lambda s: s.source_id)
    return sources


def all_documents(sources: List[Source]) -> List[Document]:
    docs: List[Document] = []
    for source in sources:
        docs.extend(source.documents)
    docs.sort(key=lambda d: d.doc_id)
    return docs


def training_text(sources: List[Source]) -> List[str]:
    """Text the tokenizer is trained on.

    Held-out sources are excluded.  Fitting a vocabulary to the benchmark is a
    subtle contamination channel: it does not leak answers, but it does let the
    model spend fewer, better-shaped tokens on exactly the text it will be
    evaluated on.
    """
    return [
        doc.text
        for source in sources
        if not source.held_out
        for doc in source.documents
    ]
