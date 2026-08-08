"""Determinism controls.

Reproducibility here is engineered in two layers, and it is worth being precise
about which one carries the guarantee.

**Integer determinism - exact, and what the evidence rests on.**  Batch ids,
token spans, loss masks and content hashes are all functions of integers.  They
reproduce bit for bit on any machine, regardless of thread count, BLAS build or
CPU.  Every `[PASS]` claim about resume and replay is an integer comparison.

**Float determinism - best effort.**  Losses and gradients are floating point,
and floating point addition is not associative, so a different reduction order
gives a slightly different number.  The settings below (single thread, fixed
seeds, deterministic kernels) make the run reproduce on the same machine and
build, but the system never *depends* on that: losses are compared with a
tolerance and are never hashed.

RNG is handled by derivation rather than by carrying state.  Instead of a global
generator whose position must be checkpointed and restored exactly, each step
derives its seed from (master_seed, branch_id, step).  Resuming at step N
therefore reconstructs step N's randomness by definition, with nothing to get
wrong.
"""

from __future__ import annotations

import os
import random

from ..hashing import derive_seed

_TORCH = None


def torch():
    """Import torch lazily, with a clear message if it is missing."""
    global _TORCH
    if _TORCH is None:
        try:
            import torch as _t
        except ImportError as exc:                                  # pragma: no cover
            raise SystemExit(
                "PyTorch is required to run the training loop.\n"
                "Install the CPU build with:  pip install -r requirements.txt"
            ) from exc
        _TORCH = _t
    return _TORCH


def configure(seed: int = 0, single_thread: bool = True) -> None:
    """Pin everything that can be pinned, before any tensor is created."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    t = torch()
    if single_thread:
        # Reduction order in a threaded BLAS depends on how work was split,
        # which makes the same sum differ run to run.  One thread removes that
        # source of drift; the model is far too small for it to cost anything.
        t.set_num_threads(1)
        t.set_num_interop_threads(1)

    random.seed(seed)
    t.manual_seed(seed)
    try:
        t.use_deterministic_algorithms(True, warn_only=True)
    except Exception:                                               # pragma: no cover
        pass


def step_generator(master_seed: int, branch_id: str, step: int):
    """A torch Generator whose state is a pure function of the position."""
    t = torch()
    generator = t.Generator()
    generator.manual_seed(derive_seed(master_seed, branch_id, "step", step))
    return generator


def step_rng(master_seed: int, branch_id: str, step: int, purpose: str = "") -> random.Random:
    return random.Random(derive_seed(master_seed, branch_id, "step", step, purpose))


def rng_fingerprint(master_seed: int, branch_id: str, step: int) -> str:
    """A short, checkable name for a step's randomness.

    Stored in the checkpoint and in the ledger so a resume can prove it
    reconstructed the same RNG position rather than merely asserting it.
    """
    return format(derive_seed(master_seed, branch_id, "step", step), "016x")
