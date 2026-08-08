"""Frozen run configuration.

Everything that influences the data stream lives here and is hashed into
`RunConfig.config_hash`.  That hash goes into every checkpoint and every ledger
event, so a run can never be silently compared against a run that used
different settings.

The rule for a trustworthy comparison:  experiment = model checkpoint + optimizer state + data stream
+ code/config.  If any one changes silently, the comparison is worthless.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

from .hashing import hash_obj, short_hash

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    corpus: Path = REPO_ROOT / "corpus"
    work: Path = REPO_ROOT / "artifacts_work"
    shards: Path = REPO_ROOT / "artifacts_work" / "shards"
    packed: Path = REPO_ROOT / "artifacts_work" / "packed"
    submission: Path = REPO_ROOT / "submission_artifacts"

    @property
    def manifests(self) -> Path:
        return self.submission / "manifests"

    @property
    def ledgers(self) -> Path:
        return self.submission / "ledgers"

    @property
    def checkpoints(self) -> Path:
        return self.submission / "checkpoints"

    @property
    def run_log(self) -> Path:
        return self.submission / "run.log"

    @property
    def events_jsonl(self) -> Path:
        return self.submission / "events.jsonl"

    @property
    def evidence_json(self) -> Path:
        return self.submission / "evidence.json"

    @property
    def evidence_md(self) -> Path:
        return self.submission / "evidence.md"

    @property
    def performance_json(self) -> Path:
        return self.submission / "performance.json"


PATHS = Paths()


# --------------------------------------------------------------------------
# Capability lanes  (capability lanes)
# --------------------------------------------------------------------------

LANES: Tuple[str, ...] = (
    "general_web",
    "code",
    "math_science",
    "indic",
    "agentic",
    "reasoning",
)

# Packing policy per lane.  The policy is a *training decision*, not a storage
# detail: it decides whether unrelated documents may share an attention window.
LANE_PACKING_POLICY: Dict[str, str] = {
    "general_web": "concat_chop",
    "code": "best_fit",
    "math_science": "greedy",
    "indic": "best_fit",
    "agentic": "structure_preserving",
    "reasoning": "long_context",
}

# Whether documents from different sources may be co-packed into one window.
LANE_ALLOWS_COPACKING: Dict[str, bool] = {
    "general_web": True,
    "code": True,
    "math_science": True,
    "indic": True,
    "agentic": False,   # tool traces must not leak into each other
    "reasoning": False,  # a reasoning trace needs the whole window to finish
}

# Which spans carry loss.  "all" = plain next-token pretraining.
# "response_only" = prompt is context, the answer is graded (SFT contract).
# "model_turns" = user turns and tool observations are context; the model's
#                 planning, tool calls and final answer are graded.
LANE_LOSS_POLICY: Dict[str, str] = {
    "general_web": "all",
    "code": "all",
    "math_science": "all",
    "indic": "all",
    "agentic": "model_turns",
    "reasoning": "response_only",
}

# Position id policy.  Segment-relative positions restart at 0 for each packed
# document, so a document does not inherit its neighbour's offsets and learn
# that it "began" halfway through a window.
LANE_POSITION_POLICY: Dict[str, str] = {lane: "segment_relative" for lane in LANES}

ATTENTION_POLICY = "causal_block_diagonal_per_segment"

PROTECTED_LANES: Tuple[str, ...] = ("indic", "agentic", "reasoning")


# --------------------------------------------------------------------------
# Curriculum stages
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    sequence_length: int
    mixture: Dict[str, float]
    protected_floors: Dict[str, float]
    warmup_steps: int
    anneal: bool = False
    # Lanes whose shards are reserved and must not be consumed before this
    # stage.  Enforced by the mixture compiler, asserted by tests.
    unlocks_reserved: Tuple[str, ...] = ()


STAGES: Tuple[Stage, ...] = (
    Stage(
        name="foundation-en",
        steps=10,
        sequence_length=256,
        mixture={
            "general_web": 0.44,
            "code": 0.16,
            "math_science": 0.14,
            "indic": 0.14,
            "agentic": 0.04,
            "reasoning": 0.08,
        },
        protected_floors={"indic": 0.10, "agentic": 0.03, "reasoning": 0.05},
        warmup_steps=0,
    ),
    Stage(
        name="reasoning-heavy-midtrain",
        steps=8,
        sequence_length=256,
        mixture={
            "general_web": 0.28,
            "code": 0.22,
            "math_science": 0.18,
            "indic": 0.12,
            "agentic": 0.08,
            "reasoning": 0.12,
        },
        protected_floors={"indic": 0.10, "agentic": 0.06, "reasoning": 0.10},
        warmup_steps=4,
    ),
    Stage(
        name="long-context-anneal",
        steps=6,
        sequence_length=512,          # long context is its own regime
        mixture={
            "general_web": 0.18,
            "code": 0.20,
            "math_science": 0.16,
            "indic": 0.14,
            "agentic": 0.14,
            "reasoning": 0.18,
        },
        protected_floors={"indic": 0.12, "agentic": 0.10, "reasoning": 0.16},
        warmup_steps=3,
        anneal=True,
        unlocks_reserved=("agentic", "reasoning"),
    ),
)


# --------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    run_id: str = "v5-s6-demo"
    master_seed: int = 20260807
    dataloader_version: str = "tdes-loader-1.0.0"

    # Batch geometry.  global batch = ranks * microbatch * grad_accum sequences.
    world_size: int = 2          # ranks, simulated sequentially on CPU
    microbatch_size: int = 2     # sequences per rank per accumulation step
    grad_accum: int = 2

    # Model (tiny on purpose - the model is a prop, the data system is the work)
    vocab_target: int = 4096
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.0
    max_position: int = 512
    learning_rate: float = 3e-3
    warmup_lr_steps: int = 6
    grad_clip: float = 1.0

    # Checkpoint / crash / fork geometry.
    # The crash lands 4 steps *past* the step-12 checkpoint on purpose: the
    # ledger then genuinely runs ahead of the surviving model state, so resume
    # has to roll back rather than merely continue.  A crash that happened to
    # coincide with a checkpoint would prove nothing.
    checkpoint_interval: int = 6      # checkpoints at steps 6, 12, 18, 24
    crash_step: int = 16              # dies mid-step 16; last checkpoint is 12
    fork_from_step: int = 12
    fork_steps: int = 4
    replay_interval: Tuple[int, int] = (6, 12)
    token_trace_interval: Tuple[int, int] = (6, 9)

    # OPUS.  Rounds are aligned to the checkpoint interval so a round is always
    # scored against a model state that a resume can restore exactly - that is
    # what makes the re-scored decisions after a crash reproduce the originals.
    opus_round_interval: int = 6      # a selection round every N steps
    opus_candidate_multiplier: int = 2
    opus_probe_tokens: int = 64       # prefix scored per candidate
    opus_probe_batches: int = 4       # golden probe batches for proxy direction

    # Tokenizer
    bpe_merges: int = 3800

    # Analysis thresholds (from a shard at ~1.2 perplexity is exhausted)
    learned_out_ppl: float = 1.2
    mixture_tolerance: float = 0.06   # absolute share tolerance, planned vs actual

    @property
    def sequences_per_step(self) -> int:
        return self.world_size * self.microbatch_size * self.grad_accum

    @property
    def total_steps(self) -> int:
        return sum(s.steps for s in STAGES)

    def stage_for_step(self, step: int) -> Stage:
        acc = 0
        for stage in STAGES:
            if step < acc + stage.steps:
                return stage
            acc += stage.steps
        return STAGES[-1]

    def stage_local_step(self, step: int) -> int:
        acc = 0
        for stage in STAGES:
            if step < acc + stage.steps:
                return step - acc
            acc += stage.steps
        return step - acc + STAGES[-1].steps

    def sequence_length_for_step(self, step: int) -> int:
        return self.stage_for_step(step).sequence_length

    def model_phase(self, step: int) -> str:
        stage = self.stage_for_step(step)
        if stage.anneal:
            return "anneal"
        frac = step / max(1, self.total_steps)
        if frac < 0.34:
            return "early"
        if frac < 0.67:
            return "mid"
        return "late"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["stages"] = [asdict(s) for s in STAGES]
        d["lanes"] = list(LANES)
        d["lane_packing_policy"] = dict(LANE_PACKING_POLICY)
        d["lane_loss_policy"] = dict(LANE_LOSS_POLICY)
        d["lane_position_policy"] = dict(LANE_POSITION_POLICY)
        d["attention_policy"] = ATTENTION_POLICY
        d["protected_lanes"] = list(PROTECTED_LANES)
        return d

    @property
    def config_hash(self) -> str:
        return hash_obj(self.as_dict())

    @property
    def short_config_hash(self) -> str:
        return short_hash(self.as_dict())


CONFIG = RunConfig()

# Sequence lengths actually used anywhere in the run - packed sample stores are
# built once per distinct length.
SEQUENCE_LENGTHS: Tuple[int, ...] = tuple(sorted({s.sequence_length for s in STAGES}))


# --------------------------------------------------------------------------
# Special tokens (frozen tokenizer contract, the tokenizer contract)
# --------------------------------------------------------------------------

SPECIAL_TOKENS: Tuple[str, ...] = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<user>",
    "<assistant>",
    "<tool_call>",
    "<tool_result>",
    "<think>",
    "<answer>",
)

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

CLEANING_PIPELINE_VERSION = "clean-1.0.0"


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
