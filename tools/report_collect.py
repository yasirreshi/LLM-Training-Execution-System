"""Concrete run material for the report: real source, real records, real files.

Everything here is pulled out of the working tree or out of
`submission_artifacts/` at build time. Nothing is transcribed by hand, so the
code the report displays is the code that ran, and the records it shows are
records the run actually wrote.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional

from tools.source_notes import PICKS

# stage id -> substrings that identify that stage's lines in run.log
LOG_PICKS = {
    "documents":   ["shards created"],
    "tokenizer":   ["tokenizer_hash_verified"],
    "shards":      ["shards created", "shards_rebuild_identical"],
    "manifests":   ["manifests validated", "shard_rejected"],
    "firewall":    ["evaluation data blocked", "eval_shard_blocked",
                    "validation_never_gradient_bearing"],
    "mixture":     ["mixture compiled", "mixture_scarcity", "mixture_within_tolerance",
                    "protected_floors_respected"],
    "packing":     ["batches packed", "packed_supply_matches_compiled_plan"],
    "masks":       ["masks_and_attention_valid"],
    "planner":     ["data_system_deterministic_across_processes"],
    "opus":        ["OPUS decisions recorded", "opus_proxy_direction",
                    "opus_all_decisions_have_reasons", "opus_protected_floor_override_recorded",
                    "opus_rescoring_after_crash_identical"],
    "training":    ["model_initialised", "initial_loss_matches_uniform_prior"],
    "consumption": ["ledger_chain_intact", "no_skipped_or_repeated_batches"],
    "learning":    ["learning_trace_linked_to_source", "eos_perplexity_tracked"],
    "checkpoint":  ["checkpoint saved", "checkpoint_saved", "checkpoints_pruned"],
    "crash":       ["crash simulated", "crash_injected", "crash_simulated_and_process_died",
                    "worker_exit"],
    "resume":      ["run resumed", "ledger_torn_tail_repaired", "resume_next_batch_matched",
                    "resume_rollback_replay_identical"],
    "replay":      ["historical stream replayed", "replay_hash_matched", "branch forked",
                    "fork_diverged"],
    "audit":       ["audit completed"],
    "throughput":  ["performance measured"],
}


def extract_source(root: Path, rel: str, symbol: str, notes=(), summary: str = "",
                   max_lines: int = 70) -> dict:
    """Pull one function verbatim, and anchor its commentary to real line numbers.

    Notes anchor by a substring rather than a line number so they survive the
    code being edited.  An anchor that no longer matches is reported rather
    than dropped, so stale commentary is caught at build time.
    """
    path = root / rel
    if not path.exists():
        return {"file": rel, "symbol": symbol, "code": "# file not found", "lines": 0,
                "summary": summary, "notes": [], "unmatched": [n[0] for n in notes]}
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    owner, _, name = symbol.rpartition(".")

    def find(nodes):
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                return n
        return None

    node = None
    if owner:
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and n.name == owner:
                node = find(n.body)
                break
    else:
        node = find(tree.body)
    if node is None:
        return {"file": rel, "symbol": symbol, "code": "# symbol not found", "lines": 0,
                "summary": summary, "notes": [], "unmatched": [n[0] for n in notes]}

    lines = text.splitlines()
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    end = node.end_lineno
    body = lines[start:end]
    total = len(body)
    truncated = total > max_lines
    if truncated:
        body = body[:max_lines] + ["    # ... truncated; see the file"]
    pad = min((len(l) - len(l.lstrip()) for l in body if l.strip()), default=0)
    body = [l[pad:] if len(l) > pad else l for l in body]

    # anchor each note to the first line containing its substring
    anchored, unmatched, used = [], [], set()
    for anchor, commentary in notes:
        probe = " ".join(anchor.split())
        hit = None
        for i, line in enumerate(body):
            if i in used:
                continue
            if probe in " ".join(line.split()):
                hit = i
                break
        if hit is None:
            unmatched.append(anchor)
            continue
        used.add(hit)
        anchored.append({"line": hit + 1, "text": commentary})
    anchored.sort(key=lambda n: n["line"])
    for index, note in enumerate(anchored, start=1):
        note["n"] = index

    return {"file": rel, "symbol": symbol, "code": chr(10).join(body), "lines": total,
            "truncated": truncated, "first_line": start + 1,
            "summary": summary, "notes": anchored, "unmatched": unmatched}


def collect_source(root: Path) -> dict:
    out, stale = {}, []
    for stage, picks in PICKS.items():
        rendered = []
        for pick in picks:
            src = extract_source(root, pick["file"], pick["symbol"],
                                 pick.get("notes", ()), pick.get("summary", ""),
                                 pick.get("max_lines", 70))
            if src["unmatched"]:
                stale.append((pick["file"], pick["symbol"], src["unmatched"]))
            rendered.append(src)
        out[stage] = rendered
    if stale:
        print("  WARNING: source annotations that no longer anchor:")
        for f, sym, anchors in stale:
            for a in anchors:
                print(f"    {f}::{sym}  ->  {a[:70]}")
    return out


def collect_files(artifacts: Path) -> List[dict]:
    out = []
    for path in sorted(artifacts.rglob("*")):
        if path.is_dir():
            continue
        out.append({"path": path.relative_to(artifacts).as_posix(),
                    "bytes": path.stat().st_size})
    return out


def _first(path: Path, predicate=None) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if predicate is None or predicate(rec):
                return rec
    return None


def collect_records(artifacts: Path) -> dict:
    """One verbatim example of each record type the run wrote."""
    cons = artifacts / "ledgers" / "consumption_main.jsonl"
    out: Dict[str, object] = {
        "consume": _first(cons, lambda r: r["type"] == "consume_microbatch"),
        "step": _first(cons, lambda r: r["type"] == "optimizer_step"),
        "checkpoint_event": _first(cons, lambda r: r["type"] == "checkpoint_saved"),
        "rollback": _first(cons, lambda r: r["type"] == "ledger_rollback"),
        "opus": _first(artifacts / "ledgers" / "opus_decisions.jsonl",
                       lambda r: r["payload"].get("protected_floor_override")),
        "learning": _first(artifacts / "ledgers" / "learning_main.jsonl",
                           lambda r: r["type"] == "sample_learning"),
        "validation": _first(artifacts / "ledgers" / "learning_main.jsonl",
                             lambda r: r["type"] == "validation_loss"),
        "firewall": _first(artifacts / "ledgers" / "firewall_events.jsonl"),
    }
    for path in sorted(artifacts.glob("manifests/shards/*.manifest.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if m["admitted"] and "manifest_ok" not in out:
            out["manifest_ok"] = m
        if not m["admitted"] and m["rejection_reasons"] and "manifest_bad" not in out:
            out["manifest_bad"] = m
    metas = sorted((artifacts / "checkpoints").glob("ckpt_*/meta.json"))
    if metas:
        out["checkpoint_meta"] = json.loads(metas[0].read_text(encoding="utf-8"))
    trace = _first(artifacts / "ledgers" / "learning_main.jsonl",
                   lambda r: r["type"] == "token_trace" and r["payload"]["tokens"])
    if trace:
        out["token"] = trace["payload"]["tokens"][0]
    return out


def collect_config(config_module) -> dict:
    d = config_module.CONFIG.as_dict()
    d.pop("stages", None)
    return d


def collect_log(artifacts: Path) -> dict:
    lines = (artifacts / "run.log").read_text(encoding="utf-8").splitlines()
    by_stage = {}
    for stage, needles in LOG_PICKS.items():
        picked, seen = [], set()
        for line in lines:
            if any(n in line for n in needles):
                key = line.strip()
                if key in seen:
                    continue
                seen.add(key)
                picked.append(line.rstrip())
            if len(picked) >= 8:
                break
        by_stage[stage] = picked
    return {"total_lines": len(lines), "by_stage": by_stage}
