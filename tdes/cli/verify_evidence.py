"""Independently re-derive every claim in evidence.json.

    python -m tdes.cli.verify_evidence

This is a separate program from the one that wrote the bundle, and it trusts
nothing the run said.  It re-runs every check against `submission_artifacts/`,
compares its own conclusions with what `evidence.json` claims, and additionally
re-derives the headline numbers a second way - by summing the consumption ledger
directly rather than reading any report.

Exits non-zero on any disagreement.  That is the point: an auditor that can
never fail is not evidence of anything, which is why `tests/test_evidence.py`
corrupts an artifact and requires this program to notice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from ..audit.evidence import Artifacts, run_checks
from ..config import PATHS
from ..ledger.consumption import EVENT_CONSUME, integrity_report
from ..ledger.store import LedgerStore

TOLERANCE = 1e-6


def verify(root: Path = None, verbose: bool = True) -> int:
    root = Path(root) if root else PATHS.submission
    artifacts = Artifacts(root)
    problems: List[str] = []

    claimed = artifacts.json("evidence.json")
    if claimed is None:
        print("FAIL  evidence.json is missing", file=sys.stderr)
        return 2

    # -- 1. re-run every check ---------------------------------------------
    recomputed = {r.name: r for r in run_checks(root)}
    claimed_by_name = {
        r["requirement"]: r for r in claimed.get("requirements", [])
    }

    for name, result in sorted(recomputed.items()):
        stated = claimed_by_name.get(name)
        if stated is None:
            problems.append(f"{name}: present in re-run but absent from evidence.json")
            continue
        expected = "PASS" if result.passed else "FAIL"
        if stated["result"] != expected:
            problems.append(
                f"{name}: evidence.json says {stated['result']} but re-running the "
                f"check on the artifacts gives {expected}"
            )
        if verbose:
            mark = "ok  " if stated["result"] == expected else "MISMATCH"
            print(f"  [{mark}] {name:28s} {stated['result']}")

    for name in claimed_by_name:
        if name not in recomputed:
            problems.append(f"{name}: claimed in evidence.json but no check produces it")

    # -- 2. every ledger's hash chain must still verify ---------------------
    for path in sorted((root / "ledgers").glob("*.jsonl")):
        store = LedgerStore(path, path.stem)
        ok, detail = store.verify_chain()
        if not ok:
            problems.append(f"{path.name}: hash chain broken - {detail}")
        elif verbose:
            print(f"  [ok  ] chain {path.name:34s} {detail['records']} records")

    # -- 3. headline numbers, re-derived straight from the ledger -----------
    consumption = root / "ledgers" / "consumption_main.jsonl"
    if consumption.exists():
        store = LedgerStore(consumption, "consumption")
        records = [r for r in store.read_all() if r["type"] == EVENT_CONSUME]
        integrity = integrity_report(records, "main")

        if not integrity["no_duplicates"]:
            problems.append(
                f"consumption ledger contains {integrity['duplicate_count']} "
                f"duplicate microbatches"
            )
        if not integrity["no_gaps"]:
            problems.append(
                f"consumption ledger is missing steps {integrity['missing_steps']}"
            )
        if not integrity["every_step_complete"]:
            problems.append(
                f"steps have {integrity['microbatches_per_step']} microbatches, "
                f"expected {integrity['expected_microbatches_per_step']}"
            )

        performance = artifacts.json("performance.json") or {}
        reported = performance.get("efficiency", {}).get("packing_utilisation")
        if reported is not None:
            drift = abs(reported - integrity["packing_utilisation"])
            if drift > 0.05:
                problems.append(
                    f"performance.json reports packing utilisation {reported} but "
                    f"summing the ledger gives {integrity['packing_utilisation']:.6f}"
                )
            elif verbose:
                print(
                    f"  [ok  ] utilisation reported {reported} vs ledger "
                    f"{integrity['packing_utilisation']:.6f}"
                )

    # -- 4. the markdown table must agree with the JSON ---------------------
    markdown = artifacts.text("evidence.md")
    for name, stated in sorted(claimed_by_name.items()):
        row = f"| {name} | {stated['result']} |"
        if row not in markdown:
            problems.append(f"evidence.md is missing or disagrees with the row for {name}")

    # -- 5. the required submission structure -------------------------------
    for required in ("run.log", "evidence.json", "evidence.md", "performance.json",
                     "manifests", "ledgers", "checkpoints"):
        if not (root / required).exists():
            problems.append(f"submission_artifacts/{required} is missing")

    # -- verdict ------------------------------------------------------------
    if problems:
        print("\nEVIDENCE VERIFICATION FAILED", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    total = len(recomputed)
    passed = sum(1 for r in recomputed.values() if r.passed)
    print(
        f"\nevidence verified independently: {passed}/{total} requirements pass, "
        f"all hash chains intact, reported figures match the ledger"
    )
    return 0 if passed == total else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Independently verify evidence.json")
    parser.add_argument("--root", default=None, help="submission_artifacts directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    return verify(Path(args.root) if args.root else None, verbose=not args.quiet)


if __name__ == "__main__":
    sys.exit(main())
