"""Shared fixtures.

Two kinds of test live here.

*   **Unit tests** exercise the pure pieces - hashing, the ledger, packing
    policies, mask validators, apportionment - and need nothing on disk.

*   **Artifact tests** assert properties of a real run.  They read
    `submission_artifacts/`, and if it is not there the `artifacts` fixture runs
    `python run_demo.py` once for the whole session.  That makes `pytest -q`
    self-contained from a clean clone at the cost of about a minute on the first
    invocation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tdes.config import PATHS  # noqa: E402
from tdes.pipeline import build_data_system, finalise_data_system  # noqa: E402


@pytest.fixture(scope="session")
def artifacts() -> Path:
    """submission_artifacts/, running the demo once if it is absent."""
    marker = PATHS.submission / "evidence.json"
    if not marker.exists():
        result = subprocess.run(
            [sys.executable, "run_demo.py"], cwd=str(REPO_ROOT),
            capture_output=True, text=True,
        )
        if not marker.exists():
            pytest.fail(
                "run_demo.py did not produce evidence.json\n"
                f"exit={result.returncode}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
            )
    return PATHS.submission


@pytest.fixture(scope="session")
def evidence(artifacts: Path) -> dict:
    return json.loads((artifacts / "evidence.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def system():
    """A freshly built data system, in-process.

    Deliberately independent of whether the demo has run: these tests check the
    construction itself, not the artifacts a run left behind.
    """
    built = build_data_system()
    return finalise_data_system(built)


@pytest.fixture(scope="session")
def artifact_copy(artifacts: Path, tmp_path_factory) -> Path:
    """A throwaway copy, for tests that need to corrupt something."""
    destination = tmp_path_factory.mktemp("artifacts_copy") / "submission_artifacts"
    shutil.copytree(artifacts, destination)
    return destination


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
