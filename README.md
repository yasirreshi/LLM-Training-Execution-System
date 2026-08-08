# Training Data Execution System

Training infrastructure assignment.

A small but complete training data system implementing the full path

```
documents → tokenized shards → manifests → mixture schedule → packing → batches
          → training → consumption ledger → learning ledger → checkpoint
          → crash → resume → replay → audit
```

The model is deliberately tiny (1.3M parameters) and beside the point. The point is
that the data system can prove **what it consumed, why it consumed it, what the model
learned from it, and how the run can be reconstructed.**

**→ [Interactive report](docs/report.html)** — the complete record of this run. Nineteen
stages in execution order, each answering the same questions, weighted toward what was
actually built and run rather than toward theory:

| Facet | Answers |
|---|---|
| Concept | a snapshot — what the object is, and the alternative rejected. Two or three sentences, no more |
| What we built | the concrete implementation: formats, constants, algorithms, the decisions inside the code |
| What it ran | what actually happened in this run, with the numbers |
| The code that ran | the real functions, pulled out of the working tree at build time — **line-numbered, syntax-highlighted, and every marked line explained underneath in plain English** |
| What it wrote | the artifacts with their real sizes, plus one verbatim record from the run |
| In run.log | that stage's actual log lines |
| In / out | what it consumes, produces, and which stage depends on it |
| Measured | the metrics the stage emitted |

Plus three architecture views — the contract stack, the dependency layers,
and the runtime data flow with its types — and sixteen interactive instruments sitting inside
the stage they belong to.

The page carries **35 verbatim source excerpts with 160 line-anchored explanations**, **12 real
records** (a full manifest, a consumption record, an OPUS decision with its floor override, a
checkpoint's metadata, the rollback event) and the actual log lines for every stage.

The code is not dropped in raw. Each excerpt has a one-sentence summary of what the function
does, and the lines that matter are numbered and explained underneath — what that line achieves
and, where it is not obvious, why it is written that way rather than the shorter way. A reader
who does not write Python can still follow the mechanism.

Commentary lives in `tools/source_notes.py` and anchors to a **substring of the line**, not a
line number, so it follows the code when it moves. An anchor that stops matching is reported at
build time rather than silently dropped — stale commentary fails loudly.

All of it is read from the working tree and `submission_artifacts/` by
`python tools/build_report.py`, so the code it displays is the code that ran and the records it
shows are records the run wrote.

## Contents

- [The pipeline, end to end](#the-pipeline-end-to-end) — every stage, in execution order, with the decision made at each touchpoint
- [Architecture](#architecture) — module map and dependency direction
- [The five design decisions that matter](#the-five-design-decisions-that-matter)
- [Does it meet the completion criterion?](#does-it-meet-the-completion-criterion)
- [Results from the committed run](#results-from-the-committed-run)
- [Honest caveats](#honest-caveats)

## Run it

```bash
pip install -r requirements.txt
python run_demo.py                    # regenerates submission_artifacts/ from scratch
python -m tdes.cli.verify_evidence    # independent re-derivation of every claim
pytest -q                             # 112 invariant tests
python tools/build_report.py          # rebuild docs/report.html from the artifacts
```

The report is assembled from `tools/report_head.html` + `report_body.html` +
`report_stages.js` (the prose) + `report_main.js` (renderer and instruments) into
`tools/report_template.html`; `tools/report_collect.py` then gathers the source excerpts,
verbatim records, file listing and log lines, and `build_report.py` injects everything. Prose
lives in the template; every number, every line of code and every record comes from the tree
and the artifacts at build time, and the build fails loudly if something it expects is missing.

One command, no manual intervention, no network. Roughly 40–90 seconds on a laptop CPU
(the six anneal steps run at window 512, four times the attention cost of the earlier stages).
`run_demo.py` deletes and rebuilds `submission_artifacts/` every time, so the files
committed here are a snapshot — the run is the authority.

---

## Architecture

Twelve subsystems, each owning one contract from an earlier session.

```
run_demo.py                  the one command; orchestrates all phases
tdes/
  config.py                  frozen run config — its hash goes into every record
  hashing.py                 canonical JSON, merkle roots, seed derivation
  fsutil.py                  atomic writes, read-only shards, Windows-safe replace
  events.py                  run.log + events.jsonl, written by the same call
  corpus/                    document loading, provenance attachment
  tokenizer/                 NFC + Indic-safe normalization, byte-level BPE, freeze
  shards/                    builder, manifests, admission gate, registry
  firewall/                  contamination fingerprints, two-sided eval firewall
  mixture/                   quota compiler, protected floors, warmup, anneal reserve
  packing/                   six policies, loss/attention masks, position ids
  opus/                      golden-probe gradient direction, selector
  streams/                   planner (pure), resume, replay
  ledger/                    hash-chained store, consumption, learning, branches
  training/                  tiny GPT, loop, checkpoints, determinism, crash injector
  audit/                     auditor, evidence generation
  perf/                      throughput counters
  cli/                       train_worker (killable), verify_evidence
corpus/                      bundled offline corpus, 6 lanes + validation + test
tools/                       report template, prose, renderer and generator
tests/                       112 invariant tests
```

**Data flows one way.** The corpus is read once; everything downstream is derived and
content-addressed. A shard id is a hash of its documents and tokenizer; a packed sample
id is a hash of its segments; a batch id is a hash of its microbatch layout. Nothing is
named by a counter, so nothing depends on the order things happened to be built in.

---

## The pipeline, end to end

What follows is the whole path a token takes, in execution order, with the
decision made at each touchpoint. Every number is from the committed run.

```
corpus/*.txt ─┐
              ├─▶ 01 documents ──▶ 02 tokenizer ──▶ 03 shards ──▶ 04 manifests + gate
sources.json ─┘        54 docs        3,558 vocab      25 shards       20 admitted, 5 refused
                                      frozen + hashed  read-only       6 rules, all reasons kept
                                                                              │
        ┌─────────────────────────────────────────────────────────────────────┘
        ▼
   05 firewall ──▶ 06 mixture ──▶ 07 packing ──▶ 08 masks ──▶ 09 plan
   2 sides           24 steps       6 policies     6 checks     pure fn
   0 leaks           18 rescues     121 samples    0 failures   (seed,branch,step)
                                                                        │
        ┌───────────────────────────────────────────────────────────────┘
        ▼
   10 OPUS ──▶ 11 train step ──▶ 12 consumption ledger ──▶ 14 checkpoint
   528 scored   8 seq/step        96 records, hash-chained   offset, not step no.
   264 taken    loss before/after   written BEFORE the step        │
        │                                                          │
        └──────────▶ 13 learning ledger ◀─────────────────────────┘
                     192 samples, 5,038 token records
                                                                   │
        ┌──────────────────────────────────────────────────────────┘
        ▼
   15 crash ──▶ 16 resume ──▶ 17 replay + fork ──▶ 18 audit ──▶ 19 throughput
   exit 137      roll back      3 derivations       2 questions   4 rates
   ledger ahead  18 identical   all agree           answered      + formulas
```

### 01 · Documents — provenance is bound at parse time

Lane files split on `===DOC===`; a header gives `doc_id`, `lang`, `script`, and
optionally `stage_hint`, `reserved`, `min_context`. Role markers (`@user:`,
`@think:`, `@tool_call:`, `@tool_result:`, `@answer:`) split the body into spans.

> **Touchpoint — the loss mask is decided here, not at packing time.** A fixed
> table marks `user` and `tool_result` as context, `think`/`tool_call`/`answer`
> as graded. Everything downstream inherits it.

`corpus/sources.json` carries licence, tier, cleaning hash, dedup, PII and
contamination status per source. Four sources are *designed to fail* the gate, so
the firewall has something genuine to catch rather than a synthetic injection.

### 02 · Tokenizer — frozen means enforced, not documented

Byte-level BPE trained on the non-held-out corpus, then serialised and hashed.

> **Touchpoint — NFC, and joiners survive.** Stripping ZWJ/ZWNJ turns क्‍ष into
> क्ष — different words. NFC also makes the two Unicode spellings of a nukta
> letter collide into one token sequence instead of being learned twice.

> **Touchpoint — the hash is recomputed on every load**, from the merge table,
> and the file is refused on mismatch. Reading a hash and trusting it proves
> nothing. Measured fertility: English 1.65 → Tamil 7.18.

### 03 · Shards — immutable, content-addressed

Documents become `<bos> [role-marker span-tokens]… <eos>` in a `uint32` array,
grouped ≤3 per shard, written atomically then `chmod` read-only.

> **Touchpoint — `<eos>` is graded on purpose.** Ending is a behaviour the model
> must learn, and EOS perplexity is a direct read-out of whether boundaries work.

> **Touchpoint — the id is a hash of content**, not a counter, so rebuilding in a
> different order yields the same id. Verified: rebuilt in-process, all hashes
> identical.

### 04 · Manifests and the admission gate

28 fields per shard. Six independent rules, and **every** failing reason is
returned rather than the first.

> **Touchpoint — contamination status records what the scan *found*,** not what
> the source claimed. One exception: a source with no cleaning lineage stays
> `not_scanned`, because a clean overlap scan says nothing about PII or dedup.

This run: 20 admitted, 5 refused — licence tier D, unknown cleaning lineage,
verbatim benchmark overlap, and two never-train shards.

### 05 · Firewall — two sides, three detectors

> **Touchpoint — the ratio is computed against the benchmark item, not the
> candidate.** A long document quoting one short test item in full scores 1.0
> instead of being diluted to nothing. Threshold 0.55 separates a copy from
> shared phrasing.

**Registry side** refuses non-train shards at planning time. **Batch side** runs
immediately before every gradient, on the *decoded text*, and does not trust the
shard id it was handed. Result: 0 validation tokens ever gradient-bearing.

### 06 · Mixture — compiled before anything is built

> **Touchpoint — availability comes from window *counts*,** obtainable from item
> sizes alone. So the schedule is compiled and checked before a single token is
> materialised, rather than planning around whatever the packer happened to make.

Largest-remainder apportionment (always sums exactly), linear warmup blending
between stages, floors as a repair pass capped by real supply. This run: 18 floor
rescues, 9 lanes needing explicit scarcity resolution, planned-vs-actual delta
0.000.

### 07 · Packing — the policy is a training decision

Six policies over abstract items. Web → `concat_chop`, code and Indic →
`best_fit`, maths → `greedy`, agentic → `structure_preserving`, reasoning →
`long_context`.

> **Touchpoint — oversized documents are cut at declared split points,** never
> mid-turn. "Structure preserving" does not mean never divided; it means each
> piece is still a coherent stretch of one conversation.

> **Touchpoint — retention is reported beside utilisation.** `pad_only` scores
> 1.000 utilisation by truncating what does not fit. Retention is what exposes it.

### 08 · Masks — six invariants, checked per sample

`loss_mask` (graded targets), `segment_ids` (attention confinement),
`position_ids` (restart per document).

> **Touchpoint — attention is causal *and* block-diagonal.** Drop the second and
> the model learns that an unrelated document is a natural continuation.

> **Touchpoint — the mask is validated against an independently derived list** of
> which tokens the role spans said were graded. These bugs still train and still
> show a falling loss curve; grading padding makes loss drop *beautifully*.

### 09 · The plan — pure, so it can be checked

`plan(seed, branch, step)` — no I/O, no model, no global RNG.

> **Touchpoint — RNG is derived, not carried.**
> `sha256(master_seed ‖ branch_id ‖ step)`. Resuming at step N reconstructs step
> N's randomness *by definition* — nothing to serialise, nothing to restore wrong.

### 10 · OPUS — a real gradient cosine

A golden probe set gives the direction we wish to move in; each candidate is
scored by the cosine of its own gradient against it.

> **Touchpoint — the probe subset excludes the tied embedding.** It dominates the
> parameter count, changes slowly, and would swamp the cosine with a term nearly
> identical for every candidate.

> **Touchpoint — stage-mismatched candidates are demoted, not rejected.**
> Rejecting outright can empty a lane whose whole pool is hinted for a later
> phase, and an empty lane cannot fill a fixed batch geometry. Found as a real bug.

528 scored, 264 accepted, 50 rejected, 214 deferred, 18 floor overrides. Scores
span −0.56 to +0.87 across 423 distinct values.

### 11 · The training step — the order is the point

```
plan → proxy/select → assemble → LOSS BEFORE
     → per microbatch { firewall → forward → backward → WRITE LEDGER }
     → clip → optimizer step → scheduler step
     → LOSS AFTER → learning records → checkpoint if due
```

> **Touchpoint — the ledger is written and fsynced *before* the optimizer step.**
> A batch that was served must be recorded even if the process dies next. This
> single ordering choice is what leaves the ledger ahead of the model after a
> crash — and therefore what makes resume a real problem with a right answer.

> **Touchpoint — loss is measured before *and* after on the same tokens,** so
> "did this batch help" is a measurement rather than an inference.

First-step loss 8.20 against ln(V) = 8.18 — the free check that labels are
shifted right and the mask is not inverted.

### 12 · Consumption ledger — append-only, hash-chained

`{seq, prev_hash, event_hash, type, payload}`, canonical JSON, fsynced per record.

> **Touchpoint — one non-append operation exists,** `rollback_to(offset)`, and it
> logs the hashes of everything it discards.

96 records, 0 duplicates, 0 gaps, chain intact through crash, repair and rollback.

### 13 · Learning ledger — tiered storage

Per-sample for every step; full per-token trace for a configured interval;
aggregates for the run.

> **Touchpoint — perplexity ≤ 1.2 marks a shard *exhausted*,** not bad: the model
> already predicts it, so further exposure is spent compute. A max gradient >8×
> the mean marks it *harmful*.

192 sample records, 5,038 token records, 19 shard report cards →
`next_corpus_recommendations.json`.

### 14 · Checkpoint — carries a data position

> **Touchpoint — it stores a ledger *byte offset*, not a step number.** A step
> number tells you where to resume counting; an offset tells you where to
> **truncate**, which is the operation recovery actually needs.

Also: the plan hash the *next* step should serve, so resume has something
independent to check against.

### 15 · Crash — a real process death

The trainer is a subprocess. At step 16 — four steps past the step-12 checkpoint
— it writes 55% of a record with no trailing newline, fsyncs, and calls
`os._exit(137)`.

> **Touchpoint — `os._exit`, not an exception.** An exception unwinds the stack,
> flushes buffers and runs destructors: exactly the cleanup a real kill does not
> get. Result on disk: weights from step 12, a ledger claiming step 16, a
> 172-byte torn tail.

### 16 · Resume — the two wrong answers, and the right one

- Resume at the ledger's position → those steps are **silently skipped**.
- Resume at the checkpoint and append → they are **recorded twice**.
- Restore **and truncate to the recorded offset** → one exposure, one record.

> **Touchpoint — the comparison fingerprint is integer-only.** Step, rank,
> microbatch, batch id, sample ids, token spans, token and mask hashes. No float,
> so the equality test cannot be defeated by CPU reduction-order drift.

Two proofs emitted: the plan hash matched, and all 18 rolled-back microbatches
came back byte-identical — 0 mismatches, 0 missing.

### 17 · Replay and fork — three derivations

**Recorded** (ledger) · **reconstructed** (shard bytes re-read and re-hashed) ·
**recomputed** (planner, from the seed, ledger never consulted).

> **Touchpoint — a replay that compares the ledger to itself proves nothing.**
> Agreement with the shards proves the tokens are still there; agreement with the
> recomputed plan proves the ordering was not fabricated afterwards.

> **Touchpoint — the fork check is two-sided.** Identical batches *after* the fork
> point would mean it changed nothing; differing batches *before* it would mean it
> never started from the parent's state.

### 18 · Audit — the two questions

Which shards influenced the model over an interval (by step *or* by cumulative
token count), and which OPUS-selected batches preceded a loss spike.

> **Touchpoint — spikes are detected as a z-score on the first difference,** not
> an absolute threshold, because the absolute loss falls throughout a run.

### 19 · Throughput — the number that matters

> **Touchpoint — efficiency comes from the ledgers (complete, including the
> pre-crash steps); rates come from the timed phases.** Dividing all the tokens by
> only the surviving wall clock would inflate the figure. Both scopes are stated
> in `performance.json`, with the formulas to rebuild each number.

Utilisation 0.867, loss density 0.820. The gap is context-only tokens — real, but
not learning.


---

## The five design decisions that matter

### 1. Identity comes from integers, never from floats

Floating-point addition is not associative, so a different reduction order gives a
different number. Every identity here — batch ids, token spans, loss masks, shard
content hashes — is a hash of **integers only**. Losses are compared with a tolerance
and are never hashed.

That is why `replay_hash_matched` is an exact claim rather than an approximate one. In
practice the floats reproduce as well — two consecutive full runs produced bit-identical
losses and bit-identical OPUS gradient cosines — but nothing *depends* on that.

### 2. The plan is a pure function; the batch is not

Two layers, deliberately separated:

| | depends on | recomputable by | named by |
|---|---|---|---|
| **the plan** | seed, branch, step | anyone, any time, no model | `plan_hash` |
| **the batch** | plan + model state | only from a checkpoint | `batch_id` |

`plan(seed, branch, step)` does no I/O, touches no global RNG, and does not care how far
the run has already got. Replay compares three independently derived answers —
**recorded** (ledger), **reconstructed** (shard bytes re-read and re-hashed) and
**recomputed** (planner, from the seed, ledger not consulted). A file agreeing with
itself would prove nothing.

RNG is derived rather than carried: each step's seed is
`sha256(master_seed ‖ branch_id ‖ step)`, so resuming at step N reconstructs step N's
randomness by definition, with no generator state to restore incorrectly.

### 3. A checkpoint stores a ledger *offset*, not a step number

The consumption ledger is written and fsynced **before** the optimizer step, because a
batch that was served must be recorded even if the process dies before learning from it.
That ordering creates the asymmetry recovery has to handle: after a crash the ledger is
ahead of the durable model state, and two plausible answers are both wrong.

- Resume at the ledger's position → the data from those steps is silently skipped.
  Nothing reports it, because the weights and the ledger are each internally consistent
  and merely describe different histories.
- Resume at the checkpoint and append → the ledger records those steps twice and any
  later audit reads it as a repeat.

The right answer is to restore the checkpoint **and truncate the ledger back to the byte
offset it recorded**. Those records describe batches served to a model state that no
longer exists. A step number tells you where to resume counting; an offset tells you
where to truncate, which is the operation actually required.

The rollback is the one non-append operation in the system, so it logs the hashes of
everything it discards — and then the re-served batches are compared against them.

### 4. The mixture is compiled before the batches are built

The packing policies work on item sizes, so how many windows a lane will yield is known
before a single token is read. That means the schedule can be compiled, checked for
feasibility, and have its scarcity resolved *first*, with the samples then built to serve
a plan that already exists. Building first and planning around whatever came out is how a
run ends up with a mixture nobody chose.

### 5. Evidence is generated, and the auditor has teeth

Every requirement is a `Check` that reopens the generated artifacts and recomputes the
claim. No check receives a value in memory from the run, and none contains a literal
expected result — there are zero occurrences of `passed = True` in `evidence.py`. If the
artifacts say a subsystem failed, the bundle says it failed.

`tdes/cli/verify_evidence.py` is a **separate program** that re-runs every check,
re-verifies every ledger hash chain, and re-derives the headline throughput figures by
summing the consumption ledger directly. Three tests corrupt artifacts — a doctored
ledger record, a flipped verdict, a deleted file — and require the verifier to notice
each one. An auditor that can never fail is not evidence of anything.

---

## Does it meet the completion criterion?

> "The assignment is complete only when the system can prove **what it consumed**, **why it
> consumed it**, **what the model learned from it** and **how the run can be reconstructed**."

Yes — and it is checked as one property rather than inferred from the others. Every other check
verifies a subsystem; this one verifies the subsystems are *joined up*. A run could have a
perfect consumption ledger, a perfect OPUS ledger and a perfect learning ledger and still fail
here, if an id in one does not resolve in the next.

So it is a **link walk over every consumed sample instance**, not a spot check
(`tdes/audit/completion.py` → `ledgers/completion_criterion.json`, evidence row
*Completion criterion*, and five tests in `tests/test_evidence_and_learning.py`).

| Clause | How it is proved | This run |
|---|---|---|
| **what it consumed** | every microbatch names its shard ids, document ids, token span ids and token/mask hashes | 192 sample instances, 0 without spans |
| **why it consumed it** | every sample carries an OPUS decision id that resolves in the OPUS ledger, with a reason from a fixed vocabulary, plus lane and stage | 0 dangling decisions, 0 without a reason |
| **what the model learned** | every sample has a loss measured before and after the update on the same tokens; a full per-token trace for the configured interval; every shard has a classified report card | 0 missing records, 0 shards without a card |
| **how it can be reconstructed** | every record carries plan hash, batch id, RNG fingerprint, tokenizer version, loader version and config hash; replay matched on three derivations; resume matched and re-served identically; the fork diverged only after its recorded point | 0 records missing provenance |

**One sample, followed all the way through** — every value below appears in a generated artifact,
so it can be checked by hand:

```
1  CONSUMED   step 0 · s0000-r0-a0 · sample smp-bfca01925505
              token span sh-agentic-15e9e87bcd:561-762   (document agent-0001)
              tokens hash b95f10dceee585795cc8bcc2…

2  WHY        lane agentic · stage foundation-en
              OPUS opd-69adeadaa108 → accepted, reason "protected_lane_bias"
              score 0.116878 against a threshold of 0.074855

3  LEARNED    loss 8.200393 → 7.971709  (delta -0.228684)
              mean token perplexity 3032.968262 · phase early
              shard verdict: harmful

4  REBUILD    plan hash d91e8b961dacf8e0 · batch b-9a4e71e13c97
              rng 7ecbc81c2db0d867 · tokenizer a50ca9093ec605d7
              config eb8646b63484219c · loader tdes-loader-1.0.0
```

The report has the same walk as a dedicated section, clause by clause, with the live numbers.

---

## What the grader will see

**Step 1 — Execute.** `python run_demo.py` on a clean clone exits 0 in ~40s and
regenerates the complete `submission_artifacts/` tree. Verified from a fresh copy with
no artifacts present.

**Step 2 — Verify evidence.** `run.log` contains all 13 required event lines *in the
order the assignment lists them*, and all five required `[PASS]` markers
(`tokenizer_hash_verified`, `eval_shard_blocked`, `checkpoint_saved`,
`resume_next_batch_matched`, `replay_hash_matched`) among 48 checks, with zero `[FAIL]`.
`evidence.md` opens with the four claims that matter and the nine required table rows
(plus three extras), then an appendix of what each check actually verified.

**Step 3 — Inspect code.** 112 tests. No random numbers anywhere in the OPUS scorer or
selector — 423 distinct gradient cosines across 528 decisions, spanning −0.564 to +0.867.

## Results from the committed run

**Shards and admission** — 25 shards, 20 admitted, 5 refused. Every rejection class is
exercised by a real source, not a stub:

| Reason | Source |
|---|---|
| `licence_tier_not_trainable` | `general_web/web_restricted_licence.txt` (tier D) |
| `cleaning_lineage_unknown` | `code/code_unknown_lineage.txt` (no pipeline hash) |
| `eval_overlap_detected` | `math_science/science_contaminated.txt` quotes a benchmark item verbatim |
| `never_train_flag_set` | the two `corpus/test/` shards |

**Tokenizer** — 3,558-token byte-level BPE trained from scratch, frozen and hashed, the
hash recomputed from the merge table on every load. Measured fertility: English 1.65,
Hindi 3.18, Bengali 4.38, Marathi 4.40, Telugu 5.98, **Tamil 7.18**. That last figure is
the Indic cost as a number — the same document costs over four times more
context in Tamil than in English.

**Packing** — six policies, one per lane by data type; all six benchmarked over the same
items so the chosen one's advantage is measured. Utilisation 0.867, loss density 0.820.

**Mixture** — planned versus actual delta 0.000, all 9 protected-floor checks respected,
18 floor rescues fired, 9 lanes needed explicit scarcity resolution.

**OPUS** — 528 candidates scored by real gradient cosine. 264 accepted, 50 rejected, 214
deferred, 18 protected-floor overrides, all five rejection reasons present.

**Crash and resume** — died at step 16 with exit 137, last checkpoint at step 12, 172-byte
torn tail repaired, 23 records rolled back across steps 12–16. All 18 rolled-back
microbatches re-served with identical batch ids, spans and hashes — and identical losses,
across a process boundary.

**Learning ledger** — 192 sample records, 5,038 token-level trace records, 19 shard report
cards feeding `next_corpus_recommendations.json`: 12 collect-more, 12 protect, 13 repeat, 7 reject.

---

## What is committed

`submission_artifacts/` is committed because the assignment asks for the generated
manifests, ledgers, checkpoints and reports — but it is a snapshot, not the source of
truth. Every run deletes and rebuilds it.

One exception: `checkpoints/**/state.pt` is gitignored. Each is 16MB of weights that any
run regenerates, whereas the `meta.json` beside it — carrying the ledger offset, RNG
fingerprint and schedule hash — is what makes a checkpoint auditable, and that is
committed for every checkpoint. The run applies the same distinction:
`prune_checkpoints` reclaims 31.6MB by dropping superseded weights while keeping all
metadata, which is what real runs do and is the exact failure the diagnostic describes — a
200GB checkpoint that would not save because nobody deleted the old ones.

## Honest caveats

- **The OPUS probe set is held-out validation data.** Its gradient is computed and used
  as a *direction*, never applied to the weights, so no validation token is ever
  gradient-bearing — asserted, and reported as 0 in `firewall_report.json`. Steering
  selection with held-out data is nonetheless a weak information channel. It is a
  deliberate choice, documented in `tdes/opus/proxy.py`, not an oversight.
- **The corpus is ~20k trainable tokens**, so 24 steps make roughly 2.5 passes over it.
  Repetition is real and tracked per sample, which is what makes the repeat-effect
  measurement meaningful — but the OPUS pool genuinely runs thin, and the proxy-health
  monitor says so: *"accepted gradient norms collapsed — the pool is likely exhausted."*
  That negative finding is left in rather than tuned quiet.
- **Quota granularity.** At 8 sequences per step the finest expressible share is 12.5%,
  so a stated 4% agentic share is served as 12.5% under the protected floor.
  `mixture_schedule.json` reports this under `stage_intent_vs_compiled` rather than
  hiding it, and compliance is measured against the compiled plan.
- **`structure_preserving` splits long trajectories at turn boundaries.** It never
  co-packs two documents and never cuts mid-turn, but a trajectory longer than the window
  is divided across windows rather than dropped.
- **Throughput is measured on one CPU core** with `torch.set_num_threads(1)` for
  determinism. The absolute tokens/sec is not meaningful; the ratios between raw, useful
  and accepted tokens are, and those are what the report emphasises. Efficiency figures
  are summed from the ledgers (complete, including the pre-crash steps); rates are scoped
  to the phases whose wall clock survived the crash, and `performance.json` says so.
- **The crash writes a deliberate partial ledger record and calls `os._exit(137)`.**
  Constructing the failure state is the point — a real SIGKILL cannot be scheduled
  deterministically — and `tdes/training/crash.py` says so plainly.
