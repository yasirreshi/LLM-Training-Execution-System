"""Build the whole data system from the corpus, deterministically.

Both the demo driver and the training subprocess call `build_data_system`, and
both get byte-identical state: same tokenizer hash, same shard content hashes,
same packed sample ids, same schedule hash, same shuffle index.

That the *training worker rebuilds all of it in its own process* rather than
receiving it over a pipe is deliberate.  It is the strongest available check
that the construction is genuinely a function of the corpus and the config: if
anything depended on process state, a clock, dict ordering or a filesystem
listing, the resumed worker would derive a different index hash and every later
comparison would fail loudly instead of quietly drifting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import LANES, PATHS, SEQUENCE_LENGTHS, STAGES
from .corpus.loader import Source, all_documents, load_sources, training_text
from .firewall.contamination import EvalFingerprintRegistry, build_registry
from .firewall.eval_firewall import EvalFirewall
from .fsutil import write_json
from .hashing import sha256_text
from .mixture.compiler import MixtureSchedule, compile_schedule
from .packing.packer import Packer, PackedSampleStore, build_all_samples
from .shards.builder import ShardReader, build_shards
from .shards.manifest import ShardManifest, admission_report, validate_manifest
from .shards.registry import ShardRegistry
from .tokenizer.freeze import freeze, load_frozen, train_tokenizer


@dataclass
class DataSystem:
    sources: List[Source]
    tokenizer: object
    tokenizer_hash: str
    corpus_hash: str
    manifests: List[ShardManifest]
    registry: ShardRegistry
    reader: ShardReader
    fingerprints: EvalFingerprintRegistry
    firewall: EvalFirewall
    packer: Packer
    store: PackedSampleStore
    schedule: Optional[MixtureSchedule]
    admission: dict
    validation_samples: List = field(default_factory=list)
    probe_samples: List = field(default_factory=list)
    manifest_problems: List[str] = field(default_factory=list)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def fingerprint(self) -> dict:
        """A short summary of everything the build produced.

        The driver writes this; every worker recomputes it in its own process
        and compares.  Any dependence on process state, dict ordering, a clock
        or filesystem enumeration would show up here as a mismatch instead of
        drifting silently into the ledger.
        """
        from .hashing import merkle_root, short_hash

        return {
            "corpus_hash": self.corpus_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "vocab_size": self.vocab_size,
            "shard_count": len(self.manifests),
            "shard_content_root": merkle_root(
                m.content_hash for m in sorted(self.manifests, key=lambda x: x.shard_id)
            ),
            "admitted_shards": self.admission.get("admitted_count"),
            "schedule_hash": self.schedule.schedule_hash if self.schedule else None,
            "packed_sample_count": len(self.store),
            "packed_sample_root": short_hash(sorted(self.store.by_id)),
        }


def _corpus_hash(sources: List[Source]) -> str:
    """One hash over every document in the corpus, in a stable order."""
    return sha256_text(
        "\n".join(
            f"{doc.doc_id}:{doc.content_hash}"
            for doc in sorted(all_documents(sources), key=lambda d: d.doc_id)
        )
    )


def build_data_system(
    logger=None, ledger_for_firewall=None, corpus_root: Path = None,
    milestones: bool = True,
) -> DataSystem:
    sources = load_sources(corpus_root)
    corpus_hash = _corpus_hash(sources)

    # -- tokenizer: train, freeze, then load it back and re-verify the hash --
    tokenizer = train_tokenizer(training_text(sources))
    _, tokenizer_hash = freeze(tokenizer, corpus_hash)
    tokenizer, verified_hash = load_frozen()
    if logger is not None:
        logger.check(
            "tokenizer_hash_verified",
            verified_hash == tokenizer_hash,
            tokenizer_hash=tokenizer_hash[:16],
            vocab_size=tokenizer.vocab_size,
            merges=len(tokenizer.merges),
        )

    # -- evaluation fingerprints, built before any shard is admitted --------
    test_docs = []
    raw_test_files: List[str] = []
    for source in sources:
        if source.never_train or source.lane == "test":
            for doc in source.documents:
                doc.benchmark_id = source.benchmark_id or ""
                test_docs.append(doc)
            raw_test_files.append(
                (PATHS.corpus / source.file).read_text(encoding="utf-8")
            )
    fingerprints = build_registry(test_docs, raw_test_files)

    # -- shards -------------------------------------------------------------
    manifests = build_shards(sources, tokenizer, tokenizer_hash, fingerprints)
    problems = [
        f"{manifest.shard_id}: {problem}"
        for manifest in manifests
        for problem in validate_manifest(manifest)
    ]
    admission = admission_report(manifests)
    registry = ShardRegistry(manifests)
    reader = ShardReader(live_tokenizer_hash=tokenizer_hash)

    announce = _announcer(logger, milestones)
    if logger is not None:
        announce(
            "shards created",
            shards=len(manifests),
            tokens=sum(m.token_count for m in manifests),
            documents=sum(len(m.documents) for m in manifests),
        )
        announce(
            "manifests validated",
            structural_problems=len(problems),
            admitted=admission["admitted_count"],
            rejected=admission["rejected_count"],
        )
        for rejected in admission["rejected"]:
            logger.event(
                "shard_rejected",
                shard=rejected["shard_id"],
                lane=rejected["lane"],
                reasons=rejected["reasons"],
            )

    firewall = EvalFirewall(registry, fingerprints, ledger_for_firewall)
    packer = Packer(registry, reader, tokenizer, firewall)

    return DataSystem(
        sources=sources,
        tokenizer=tokenizer,
        tokenizer_hash=tokenizer_hash,
        corpus_hash=corpus_hash,
        manifests=manifests,
        registry=registry,
        reader=reader,
        fingerprints=fingerprints,
        firewall=firewall,
        packer=packer,
        store=PackedSampleStore(),
        schedule=None,
        admission=admission,
        manifest_problems=problems,
    )


def finalise_data_system(
    system: DataSystem, logger=None, milestones: bool = True
) -> DataSystem:
    """Compile the mixture, then materialise the packed samples.

    That order is the real one, not a presentational choice.  The packing
    policies operate on item sizes, so how many windows a lane will yield is
    known before a single token is read - which means the schedule can be
    compiled, checked for feasibility and have its scarcity resolved *first*,
    and the samples are then built to serve a plan that already exists.
    Building first and planning around whatever came out would be the tail
    wagging the dog, and it is how a run ends up with a mixture nobody chose.
    """
    contexts = sorted(
        {(stage.sequence_length, bool(stage.unlocks_reserved) or stage.anneal)
         for stage in STAGES}
    )

    def availability(sequence_length: int, reserved_unlocked: bool, lane: str):
        return system.packer.count_windows(lane, sequence_length, reserved_unlocked)

    schedule = compile_schedule(availability)

    if logger is not None:
        unsatisfied = [
            f.as_dict() for f in schedule.feasibility if f.resolution != "satisfied"
        ]
        _announcer(logger, milestones)(
            "mixture compiled",
            steps=schedule.total_steps,
            schedule_hash=schedule.schedule_hash[:16],
            floor_adjustments=len(schedule.floor_adjustments),
            scarcity_resolutions=len(unsatisfied),
        )
        for entry in unsatisfied:
            logger.event(
                "mixture_scarcity",
                stage=entry["stage"],
                lane=entry["lane"],
                resolution=entry["resolution"],
                repeat_factor=entry["repeat_factor"],
            )

    store = build_all_samples(system.packer, contexts, LANES)
    validation_samples = system.packer.pack_holdout(
        system.registry.validation, "validation", min(SEQUENCE_LENGTHS)
    )
    for sample in validation_samples:
        store.by_id[sample.sample_id] = sample

    # The plan assumed a window count; confirm the materialised store agrees.
    # A divergence here would mean the schedule was compiled against a supply
    # that does not exist, which is exactly the failure the compiler is for.
    mismatches = []
    for length, unlocked in contexts:
        for lane in LANES:
            planned, _tokens = availability(length, unlocked, lane)
            actual = len(store.lane_samples(length, unlocked, lane))
            if planned != actual:
                mismatches.append(
                    {"lane": lane, "sequence_length": length,
                     "planned_windows": planned, "materialised": actual}
                )

    schedule.anneal_reserve = {
        lane: ids
        for lane, ids in (
            (
                lane,
                sorted(
                    sample.sample_id
                    for (_length, unlocked), lanes in store.by_context.items()
                    if unlocked
                    for sample in lanes.get(lane, [])
                    if sample.reserved
                ),
            )
            for lane in LANES
        )
        if ids
    }

    system.store = store
    system.schedule = schedule
    system.validation_samples = validation_samples
    system.probe_samples = list(validation_samples)

    if logger is not None:
        _announcer(logger, milestones)(
            "batches packed",
            packed_samples=len(store),
            contexts=len(contexts),
            validation_samples=len(validation_samples),
        )
        logger.check(
            "packed_supply_matches_compiled_plan",
            not mismatches,
            contexts=len(contexts),
            mismatches=len(mismatches),
        )
    return system


def _announcer(logger, milestones: bool):
    """Milestone lines in the driver; plain events in a worker.

    The workers genuinely rebuild everything, but repeating the milestone lines
    once per process would make run.log look as though the pipeline ran four
    times.  The rebuild is reported instead as a fingerprint comparison, which
    is the part that carries information.
    """
    if logger is None:
        return lambda *a, **k: None
    if milestones:
        return logger.milestone
    return lambda text, **payload: logger.event(
        "rebuild:" + text.replace(" ", "_"), **payload
    )


FINGERPRINT_FILE = "build_fingerprint.json"


def build_and_finalise(
    logger=None, ledger_for_firewall=None, milestones: bool = False
) -> DataSystem:
    """Rebuild the data system in a worker process and prove it matches.

    The driver writes its fingerprint after the initial build.  Every worker
    recomputes the same fingerprint here and compares, so a determinism defect
    surfaces as a failed check at the start of the phase rather than as a
    mysterious hash mismatch during replay.
    """
    system = build_data_system(
        logger=logger, ledger_for_firewall=ledger_for_firewall, milestones=milestones
    )
    finalise_data_system(system, logger=logger, milestones=milestones)

    reference_path = PATHS.manifests / FINGERPRINT_FILE
    fingerprint = system.fingerprint()
    if logger is not None and reference_path.exists():
        import json as _json

        reference = _json.loads(reference_path.read_text(encoding="utf-8"))
        differing = sorted(
            key for key in fingerprint
            if reference.get(key) != fingerprint[key]
        )
        logger.check(
            "data_system_deterministic_across_processes",
            not differing,
            differing_fields=differing,
            tokenizer_hash=fingerprint["tokenizer_hash"][:16],
            schedule_hash=(fingerprint["schedule_hash"] or "")[:16],
            shard_root=fingerprint["shard_content_root"][:16],
        )
    return system


def write_manifest_artifacts(system: DataSystem) -> None:
    """Emit everything under submission_artifacts/manifests/."""
    directory = PATHS.manifests
    directory.mkdir(parents=True, exist_ok=True)

    shard_dir = directory / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for manifest in system.manifests:
        write_json(shard_dir / f"{manifest.shard_id}.manifest.json", manifest.as_dict())

    write_json(directory / "admission_report.json", system.admission)
    write_json(
        directory / "manifest_validation.json",
        {
            "shards_checked": len(system.manifests),
            "structural_problems": system.manifest_problems,
            "all_valid": not system.manifest_problems,
        },
    )
    system.registry.write(directory / "shard_registry.json")
    write_json(directory / "eval_registry.json", system.fingerprints.as_dict())
    write_json(directory / "mixture_schedule.json", system.schedule.as_dict())
    write_json(
        directory / "packing_report.json",
        {
            "per_lane": system.packer.pack_stats,
            "policy_comparison": system.packer.policy_comparison,
            "note": (
                "policy_comparison runs every policy over the same items so the "
                "chosen policy's advantage is measured rather than asserted. "
                "Read utilisation together with retention: pad_only reaches high "
                "utilisation by truncating whatever does not fit."
            ),
        },
    )
    write_json(directory / "packed_samples_index.json", system.store.index())
    write_json(
        directory / "corpus_report.json",
        {
            "corpus_hash": system.corpus_hash,
            "sources": len(system.sources),
            "documents": sum(len(s.documents) for s in system.sources),
            "tokenizer_hash": system.tokenizer_hash,
            "vocab_size": system.tokenizer.vocab_size,
            "fertility_by_language": _fertility(system),
        },
    )


def _fertility(system: DataSystem) -> Dict[str, float]:
    """Tokens per word by language - the Indic cost the design cares about."""
    totals: Dict[str, List[int]] = {}
    for source in system.sources:
        for doc in source.documents:
            words = len(doc.text.split())
            if not words:
                continue
            tokens = len(system.tokenizer.encode(doc.text))
            bucket = totals.setdefault(doc.lang, [0, 0])
            bucket[0] += tokens
            bucket[1] += words
    return {
        lang: round(tokens / words, 4)
        for lang, (tokens, words) in sorted(totals.items())
        if words
    }
