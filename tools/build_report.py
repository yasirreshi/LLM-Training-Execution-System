#!/usr/bin/env python3
"""Render docs/report.html from the generated artifacts.

    python tools/build_report.py

The interactive report is held to the same rule as the evidence bundle: every
number in it is read out of `submission_artifacts/`, never typed in.  The
template carries the prose, the layout and the interactions; this script
supplies the data and nothing else.  Change the run and the report changes with
it - or fails loudly because an artifact it expected is not there.

Two figures need the packed samples themselves rather than a report file (the
token inspector and the per-token perplexity strip), so the pipeline is rebuilt
in-process.  That is the same deterministic rebuild the training workers do, so
the sample ids in the report match the ones in the ledgers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tdes.config import PATHS                                    # noqa: E402
from tdes.ledger.store import LedgerStore                        # noqa: E402
from tdes.pipeline import build_data_system, finalise_data_system  # noqa: E402
from tools import report_collect as RC  # noqa: E402

A = PATHS.submission
TEMPLATE = ROOT / "tools" / "report_template.html"
OUTPUT = ROOT / "docs" / "index.html"   # docs/index.html is the GitHub Pages root

LANES = ["general_web", "code", "math_science", "indic", "agentic", "reasoning"]


def read(rel: str):
    path = A / rel
    if not path.exists():
        raise SystemExit(
            f"missing artifact: {rel}\nRun `python run_demo.py` before building the report."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def collect() -> dict:
    out: dict = {}

    # -- corpus, sources, shards, admission --------------------------------
    sources = json.loads((PATHS.corpus / "sources.json").read_text(encoding="utf-8"))["sources"]
    out["sources"] = [
        {"id": s["source_id"], "file": s["file"], "lane": s["lane"], "licence": s["licence"],
         "tier": s["licence_tier"], "clean": s.get("cleaning_pipeline"),
         "contam": s.get("contamination_status"), "heldout": s.get("held_out", False),
         "never": s.get("never_train", False), "scarce": s.get("scarce_tier"),
         "expected": s.get("_expected", "")}
        for s in sources
    ]
    mans = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(A.glob("manifests/shards/*.manifest.json"))]
    out["shards"] = [
        {"id": m["shard_id"], "lane": m["lane"], "src": m["source_id"], "tokens": m["token_count"],
         "docs": m["doc_count"], "admitted": m["admitted"], "reasons": m["rejection_reasons"],
         "tier": m["licence_tier"], "overlap": m["eval_overlap_status"],
         "hash": m["content_hash"][:12], "langs": m["languages"],
         "policy": m["packing_policy"], "loss": m["loss_policy"], "reserved": m["reserved"]}
        for m in mans
    ]
    adm = read("manifests/admission_report.json")
    out["admission"] = {"total": adm["total_shards"], "admitted": adm["admitted_count"],
                        "rejected": adm["rejected_count"],
                        "by_reason": {k: len(v) for k, v in adm["rejected_by_reason"].items()}}
    reg = read("manifests/shard_registry.json")
    out["registry"] = {k: reg[k] for k in
                       ["train", "validation", "test", "blocked", "trainable_tokens", "total_shards"]}
    out["corpus"] = read("manifests/corpus_report.json")

    # -- packing ------------------------------------------------------------
    pr = read("manifests/packing_report.json")
    out["policy_comparison"] = pr["policy_comparison"]
    out["pack_stats"] = pr["per_lane"]
    out["mask_validation"] = read("manifests/mask_validation.json")
    idx = read("manifests/packed_samples_index.json")
    summary: dict = {}
    for s in idx["samples"]:
        key = f'{s["lane"]}@{s["sequence_length"]}'
        b = summary.setdefault(key, {"n": 0, "util": 0.0, "dens": 0.0, "policy": s["policy"]})
        b["n"] += 1
        b["util"] += s["utilisation"]
        b["dens"] += s["loss_density"]
    for b in summary.values():
        b["util"] = round(b["util"] / b["n"], 4)
        b["dens"] = round(b["dens"] / b["n"], 4)
    out["samples_summary"] = summary

    # -- two real packed samples, for the token inspector -------------------
    system = finalise_data_system(build_data_system())

    def strip(sample, limit=140):
        return {
            "id": sample.sample_id, "lane": sample.lane, "policy": sample.policy,
            "seq": sample.sequence_length, "pad": sample.pad_count,
            "loss_tokens": sample.loss_bearing_count, "ctx": sample.context_only_count,
            "segments": [{"doc": s.doc_id, "shard": s.shard_id, "start": s.window_start,
                          "len": s.length, "lang": s.lang} for s in sample.segments],
            "tokens": [{"t": system.tokenizer.decode_one(t), "m": m, "s": sg, "p": p}
                       for t, m, sg, p in zip(sample.token_ids[:limit], sample.loss_mask[:limit],
                                              sample.segment_ids[:limit],
                                              sample.position_ids[:limit])],
        }

    out["sample_agentic"] = strip(system.store.lane_samples(256, False, "agentic")[0])
    out["sample_web"] = strip(system.store.lane_samples(256, False, "general_web")[0])

    # -- mixture -------------------------------------------------------------
    sch = read("manifests/mixture_schedule.json")
    out["stages"] = sch["stages"]
    out["per_step"] = [{"step": q["step"], "stage": q["stage"], "seq": q["sequence_length"],
                        "counts": q["counts"], "warmup": q["warmup_t"]} for q in sch["per_step"]]
    out["feasibility"] = sch["feasibility"]
    out["floor_adjustments"] = sch["floor_adjustments"]
    out["intent_vs_compiled"] = sch["stage_intent_vs_compiled"]
    out["compliance"] = read("manifests/mixture_compliance.json")

    # -- training curve and per-step consumption ----------------------------
    cons = LedgerStore(A / "ledgers" / "consumption_main.jsonl", "c")
    records = cons.read_all()
    steps = sorted((r["payload"] for r in records if r["type"] == "optimizer_step"),
                   key=lambda p: p["global_step"])
    out["curve"] = [{"s": p["global_step"], "loss": round(p["mean_loss"], 4),
                     "ppl": round(p["perplexity"], 1), "gn": round(p["gradient_norm"], 4),
                     "lr": p["learning_rate"], "stage": p["curriculum_stage"],
                     "ckpt": p["checkpoint_id"]} for p in steps]
    learn = LedgerStore(A / "ledgers" / "learning_main.jsonl", "l")
    learn_records = learn.read_all()
    out["validation"] = [{"s": r["payload"]["step"], "loss": round(r["payload"]["validation_loss"], 4)}
                         for r in learn_records if r["type"] == "validation_loss"]

    per_step_lane: dict = {}
    for r in records:
        if r["type"] != "consume_microbatch":
            continue
        d = per_step_lane.setdefault(r["payload"]["global_step"], {})
        for lane in r["payload"]["mixture_lane"]:
            d[lane] = d.get(lane, 0) + 1
    out["actual_lane_counts"] = {str(k): v for k, v in sorted(per_step_lane.items())}
    out["integrity"] = read("ledgers/consumption_integrity.json")

    # -- OPUS ----------------------------------------------------------------
    opus = LedgerStore(A / "ledgers" / "opus_decisions.jsonl", "o")
    out["opus_fields"] = ["step", "lane", "status", "reason", "score", "grad_norm", "loss",
                          "override", "pass", "candidate", "threshold"]
    out["opus"] = [
        [d["step"], d["lane"], d["status"], d["reason"], round(d["opus_score"], 4),
         round(d["gradient_norm"], 4), round(d["candidate_loss"], 3),
         1 if d["protected_floor_override"] else 0, d["repeated_pass_number"],
         d["candidate_id"], round(d["threshold"], 4)]
        for d in (r["payload"] for r in opus.read_all())
    ]
    rep = read("ledgers/opus_report.json")
    out["opus_report"] = {k: rep[k] for k in
                          ["total_candidates_scored", "by_status", "by_reason", "by_lane",
                           "acceptance_rate", "protected_floor_overrides"]}
    out["proxy_health"] = read("ledgers/opus_proxy_health.json")

    # -- crash, resume, replay, fork ----------------------------------------
    phase = read("ledgers/phase_resume_main.json")
    out["recovery"] = phase["recovery"]
    out["next_batch"] = phase["next_batch_verification"]
    out["rollback"] = {k: phase["rollback_replay_verification"][k]
                       for k in ["compared", "reserved_records", "identical", "discarded_steps"]}
    replay = read("ledgers/replay_report.json")
    out["replay"] = {k: replay[k] for k in
                     ["interval", "microbatches_replayed", "tokens_match", "loss_masks_match",
                      "token_spans_match", "plan_recomputation_matches", "all_match",
                      "batch_ids", "derivations_compared"]}
    out["replay_rows"] = [
        {"step": m["step"], "rank": m["rank"], "mb": m["microbatch_id"], "batch": m["batch_id"],
         "rec": m["recorded_tokens_hash"][0][:16], "rebuilt": m["reconstructed_tokens_hash"][0][:16],
         "ok": m["tokens_match"], "spans": m["token_span_ids"][:2]}
        for m in replay["microbatches"][:12]
    ]
    out["plan_comparison"] = replay["plan_comparison"]
    out["fork"] = read("ledgers/fork_divergence.json")
    out["branches"] = read("ledgers/branches.json")
    out["checkpoints"] = [
        {"id": m["checkpoint_id"], "step": m["global_step"], "stage": m["stage"],
         "offset": m["ledger_offset"], "rng": m["rng_fingerprint"], "tokens": m["tokens_consumed"],
         "next_plan": m["next_expected_plan_hash"],
         "weights": (A / "checkpoints" / m["checkpoint_id"] / "state.pt").exists()}
        for m in (json.loads((d / "meta.json").read_text(encoding="utf-8"))
                  for d in sorted((A / "checkpoints").glob("ckpt_*")))
    ]
    out["retention"] = read("checkpoints/retention.json")["pruned"]

    # -- learning ledger and next-corpus feedback -------------------------------------
    out["learning_agg"] = read("ledgers/learning_aggregates.json")
    rec = read("ledgers/next_corpus_recommendations.json")
    out["shard_cards"] = rec["shard_cards"]
    out["corpus_verdicts"] = rec["summary"]
    out["corpus_actions"] = {k: len(v) for k, v in rec["actions"].items()}

    traces = [r["payload"] for r in learn_records if r["type"] == "token_trace"]
    if traces:
        biggest = max(traces, key=lambda t: t["token_count"])
        out["token_trace"] = {
            "step": biggest["step"], "lane": biggest["lane"], "sample": biggest["sample_id"],
            "phase": biggest["model_phase"], "n": biggest["token_count"],
            "tokens": [{"t": x["preview"], "ppl": round(x["perplexity"], 2),
                        "ce": round(x["cross_entropy"], 3), "doc": x["doc_id"],
                        "eos": x["is_eos"], "sp": x["is_special"],
                        "pos": x["position_in_sequence"]} for x in biggest["tokens"][:220]],
        }
    out["trace_lanes"] = sorted({t["lane"] for t in traces})

    # -- audit, firewall, performance, evidence ------------------------------
    audit = read("ledgers/audit_report.json")
    provenance = audit["queries"]["checkpoint_provenance"]
    out["audit"] = {
        "spikes": audit["loss_spikes_detected"],
        "provenance": {k: provenance[k] for k in
                       ["shards_involved", "total_positions", "loss_bearing_tokens", "tokens_by_lane"]},
        "shards": provenance["shards"],
        "spike_reports": [
            {"step": s["spike_step"], "lookback": s["lookback_steps"],
             "shards": s["shards_in_window"], "accepted": s["opus_accepted_in_window"],
             "overrides": s["opus_overrides_in_window"],
             "top": s["highest_gradient_norm_candidate"]}
            for s in audit["spike_investigations"]
        ],
        "token_window": audit["queries"]["interval_by_token_count"],
    }
    fw = read("ledgers/firewall_report.json")
    out["firewall"] = {k: fw[k] for k in
                       ["registry_checks", "batch_checks", "blocks_total", "blocks_by_side",
                        "blocks_by_reason", "validation_gradient_bearing_tokens", "blocked_shard_ids"]}
    out["firewall_blocks"] = fw["blocks"]
    out["fingerprints"] = fw["fingerprint_registry"]
    out["perf"] = read("performance.json")

    evidence = read("evidence.json")
    out["evidence"] = [{"req": r["requirement"], "res": r["result"], "note": r["note"],
                        "files": r["evidence"], "detail": r["detail"]}
                       for r in evidence["requirements"]]
    out["evidence_meta"] = {k: evidence[k] for k in
                            ["run_id", "config_hash", "requirements_passed", "requirements_total",
                             "all_passed", "wall_seconds", "generation_method"]}
    out["completion"] = read("ledgers/completion_criterion.json")
    out["run_log_tail"] = [line for line in (A / "run.log").read_text(encoding="utf-8").splitlines()
                           if line.startswith("[PASS]") or line.startswith("[FAIL]")]

    # -- concrete run material: what we built, and what it actually wrote -----
    import tdes.config as _cfg
    out["source"] = RC.collect_source(ROOT)
    out["files"] = RC.collect_files(A)
    out["records"] = RC.collect_records(A)
    out["config"] = RC.collect_config(_cfg)
    out["log"] = RC.collect_log(A)
    return out


# The template is a body fragment.  Opened directly in a browser it would run
# in quirks mode with no declared encoding, so the Devanagari and Tamil text
# would mis-decode.  Standalone output therefore gets a real document shell.
DOCUMENT_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="description" content="Stage-by-stage record of one training-data pipeline run.">
"""
DOCUMENT_MID = """</head>
<body>
"""
DOCUMENT_TAIL = """</body>
</html>
"""


def wrap_document(fragment: str) -> str:
    """Turn the body fragment into a standalone, valid HTML document.

    The <title> and <style> at the top of the fragment belong in <head>; every-
    thing after the style block is body content.  Splitting there leaves the
    fragment itself untouched, so the same template serves both outputs.
    """
    marker = "</style>"
    index = fragment.find(marker)
    if index < 0:
        return DOCUMENT_HEAD + DOCUMENT_MID + fragment + DOCUMENT_TAIL
    head = fragment[: index + len(marker)]
    body = fragment[index + len(marker):]
    return DOCUMENT_HEAD + head + DOCUMENT_MID + body + DOCUMENT_TAIL


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Render the run report")
    parser.add_argument("--fragment", action="store_true",
                        help="emit a body fragment instead of a standalone document")
    parser.add_argument("--out", default=None, help="output path")
    args = parser.parse_args(argv)

    if not TEMPLATE.exists():
        raise SystemExit(f"missing template: {TEMPLATE}")
    data = collect()
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    # a literal closing script tag inside the JSON block would end it early
    payload = payload.replace("</script>", "<" + chr(92) + "/script>")

    html = TEMPLATE.read_text(encoding="utf-8")
    if "__DATA__" not in html:
        raise SystemExit("template has no __DATA__ placeholder")
    rendered = html.replace("__DATA__", payload)
    if not args.fragment:
        rendered = wrap_document(rendered)

    out = Path(args.out) if args.out else OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")

    kind = "fragment" if args.fragment else "standalone document"
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(data)} data sections, {kind})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
