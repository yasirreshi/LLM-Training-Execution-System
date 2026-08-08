# Evidence bundle

Run `v5-s6-demo`  ·  config hash `eb8646b63484219c`

**13 of 13 requirements passed.**

Every figure below was produced by reopening the generated artifacts and
recomputing the claim. Nothing here is hardcoded, and nothing was carried
over in memory from the run that produced the artifacts.

## The four claims that matter

1. **No batch was skipped or repeated.** Steps [0, 23] recorded across 96 microbatches; duplicates none, gaps none. After the crash, 18 rolled-back microbatches were re-served and compared: identical = True.
2. **The resumed batch was the expected batch.** The checkpoint recorded `55b37c62e5244119`; the planner, recomputed from the seed alone, produced `55b37c62e5244119`.
3. **Replay reproduced the original stream.** Interval [6, 12], 24 microbatches. Token hashes True, loss masks True, token spans True, independent plan recomputation True.
4. **No evaluation data reached a gradient.** 2 never-train and 1 contaminated shards blocked at admission; 0 non-train shards found in the consumption ledger; validation gradient-bearing tokens = 0.

Packing utilisation 0.85568, loss density 0.8021, across 121 packed samples using 5 policies.

## Requirements

| REQUIREMENT | RESULT | EVIDENCE |
|---|---|---|
| End-to-end execution | PASS | `run.log` |
| Shards and manifests | PASS | `manifests/shards/*.manifest.json`, `manifests/manifest_validation.json`, `manifests/reproducibility.json` |
| Tokenizer integrity | PASS | `manifests/tokenizer.json`, `manifests/shards/*.manifest.json` |
| Packing correctness | PASS | `manifests/packed_samples_index.json`, `manifests/mask_validation.json`, `manifests/packing_report.json` |
| Mixture compliance | PASS | `manifests/mixture_compliance.json`, `manifests/mixture_schedule.json` |
| OPUS audit trail | PASS | `ledgers/opus_decisions.jsonl`, `ledgers/opus_report.json` |
| Evaluation firewall | PASS | `ledgers/firewall_report.json`, `manifests/admission_report.json`, `ledgers/consumption_main.jsonl` |
| Crash recovery | PASS | `ledgers/phase_resume_main.json`, `ledgers/consumption_main.jsonl`, `checkpoints/` |
| Replay | PASS | `ledgers/replay_report.json`, `ledgers/consumption_main.jsonl` |
| Fork | PASS | `ledgers/branches.json`, `ledgers/fork_divergence.json` |
| Learning trace | PASS | `ledgers/learning_main.jsonl`, `ledgers/learning_aggregates.json`, `ledgers/next_corpus_recommendations.json` |
| Throughput | PASS | `performance.json`, `ledgers/consumption_main.jsonl` |
| Completion criterion | PASS | `ledgers/completion_criterion.json` |

## Appendix: what each check actually verified

### End-to-end execution — PASS
_log scanned for the required event sequence and for any [FAIL] line_

- **fail_lines_in_log**: none
- **milestones_found**: 13
- **milestones_in_order**: yes
- **milestones_required**: 13
- **missing_milestones**: none
- **pass_lines_in_log**: 40
- **submission_structure_complete**: yes

### Shards and manifests — PASS
_shards rebuilt in-process and content hashes compared_

- **manifests_missing_content_hash**: none
- **manifests_missing_lineage_field**: none
- **rebuild_reproducibility**: content_hashes_stable=True, differing_shard_ids=[], method=shards rebuilt from the same corpus in-process; hashes diffed, shard_ids_stable=True, shards_compared=25
- **shard_manifests**: 25
- **structural_problems**: none
- **structurally_valid**: yes

### Tokenizer integrity — PASS
_hash recomputed from the merge table, not read back from the file_

- **all_shards_share_one_tokenizer**: yes
- **distinct_tokenizer_hashes_in_manifests**: `a50ca9093ec605d7690619bb07fe5d14f6a367065be1ad62ca1d523bce69610d`
- **hash_matches**: yes
- **merges**: 3293
- **recomputed_from_merge_table**: `a50ca9093ec605d7690619bb07fe5d14f6a367065be1ad62ca1d523bce69610d`
- **recorded_hash**: `a50ca9093ec605d7690619bb07fe5d14f6a367065be1ad62ca1d523bce69610d`
- **shard_manifests_checked**: 25
- **vocab_size**: 3558

### Packing correctness — PASS
_utilisation recomputed per sample from pad counts_

- **aggregate_loss_density**: 0.8021
- **aggregate_packing_utilisation**: 0.85568
- **mask_validation**: all_attention_confined=True, all_masks_valid=True, attention_problems=[], checks_applied=['no loss on padding', 'no loss on the first token of a segment', 'no loss on context-only spans (prompt, tool observation)', 'position ids segment-relative, contiguous, monotonic', 'segments contiguous, padding a suffix', 'attention causal and confined to one packed document'], mask_problems=[], samples_checked=121
- **packed_samples**: 121
- **policies_exercised**: `best_fit`, `concat_chop`, `greedy`, `long_context`, `structure_preserving`
- **policy_comparison_present**: yes
- **samples_with_inconsistent_utilisation**: none

### Mixture compliance — PASS
_actual shares summed from the consumption ledger, not from the planner_

- **all_lanes_within_tolerance**: yes
- **all_protected_floors_respected**: yes
- **floor_breaches**: none
- **lanes**: {'actual_share': 0.29583, 'actual_tokens': 18176, 'delta': 0.0, 'lane': 'general_web', 'planned_share': 0.29583, 'planned_tokens': 18176, 'within_tolerance': True}, {'actual_share': 0.15417, 'actual_tokens': 9472, 'delta': 0.0, 'lane': 'code', 'planned_share': 0.15417, 'planned_tokens': 9472, 'within_tolerance': True}, {'actual_share': 0.125, 'actual_tokens': 7680, 'delta': 0.0, 'lane': 'math_science', 'planned_share': 0.125, 'planned_tokens': 7680, 'within_tolerance': True}, {'actual_share': 0.125, 'actual_tokens': 7680, 'delta': 0.0, 'lane': 'indic', 'planned_share': 0.125, 'planned_tokens': 7680, 'within_tolerance': True}, {'actual_share': 0.125, 'actual_tokens': 7680, 'delta': 0.0, 'lane': 'agentic', 'planned_share': 0.125, 'planned_tokens': 7680, 'within_tolerance': True}, {'actual_share': 0.175, 'actual_tokens': 10752, 'delta': 0.0, 'lane': 'reasoning', 'planned_share': 0.175, 'planned_tokens': 10752, 'within_tolerance': True}
- **max_abs_delta**: 0
- **tolerance**: 0.06

### OPUS audit trail — PASS
_scores are gradient cosines, so near-unique values are expected_

- **by_reason**: above_proxy_threshold=221, below_proxy_threshold=214, protected_lane_bias=18, quota_pressure=13, stage_mismatch=62
- **by_status**: accepted=264, deferred=214, rejected=50
- **decisions_missing_a_reason**: none
- **decisions_recorded**: 528
- **distinct_scores**: 423
- **ledger_chain_detail**: head=f65d9a058553349d4a2773c7c778de271f0e325e19e4ce5e1f077c0f1d86d6ab, ledger=opus_decisions, records=528
- **ledger_chain_intact**: yes
- **protected_floor_overrides**: 18
- **score_range**: -0.563792, 0.86712
- **statuses_present**: `accepted`, `deferred`, `rejected`

### Evaluation firewall — PASS
_consumption ledger re-scanned against registry permissions independently_

- **batch_side_checks_run**: 49
- **canaries_registered**: 1
- **eval_overlap_shards_blocked**: `sh-mathscience-d3e3ce9616`
- **never_train_shards_blocked**: `sh-test-24dae7d4c5`, `sh-test-65f6387680`
- **non_train_shards_found_in_consumption_ledger**: none
- **registry_side_blocks**: 1
- **validation_gradient_bearing_tokens**: 0

### Crash recovery — PASS
_gap and duplicate check recomputed from the ledger file_

- **crash_step**: 16
- **expected_plan_hash**: `55b37c62e5244119`
- **ledger_chain_intact**: yes
- **ledger_integrity**: distinct_microbatches=96, every_step_complete=True, no_duplicates=True, no_gaps=True, step_range=[0, 23]
- **microbatches_compared**: 18
- **next_batch_matched**: yes
- **recomputed_plan_hash**: `55b37c62e5244119`
- **records_rolled_back**: 23
- **resume_step**: 12
- **resumed_from_checkpoint**: `ckpt_main_0012`
- **rollback_replay_identical**: yes
- **steps_rolled_back**: 12, 13, 14, 15, 16
- **torn_tail_repaired**: yes

### Replay — PASS
_three independent derivations compared, not a file against itself_

- **batch_ids**: `b-435e5f934156`, `b-68f7d9bc465f`, `b-abb8e3c69282`, `b-af43476e7607`, `b-e8f94e6f8187`, `b-febc9f3de71c`
- **derivations_compared**: `recorded (ledger)`, `reconstructed (shard bytes re-read and re-hashed)`, `recomputed (planner, from the seed, ledger not consulted)`
- **interval**: 6, 12
- **loss_masks_match**: yes
- **microbatches_replayed**: 24
- **plan_recomputation_matches**: yes
- **token_spans_match**: yes
- **tokens_match**: yes

### Fork — PASS
_parent must be identical before the fork point and differ after it_

- **branches**: `br-dce229fce9`, `main`
- **diverged_after_fork**: yes
- **fork_point_step**: 12
- **identical_before_fork**: yes
- **steps_after_fork_compared**: 12, 13, 14, 15

### Learning trace — PASS
_every loss record carries the shard and document that produced it_

- **classification_summary**: exhausted=0, harmful=7, neutral=0, useful=12
- **eos_perplexity**: max=13331.81543, mean=992.307705, min=151.86879, samples=99
- **missing_token_fields**: none
- **sample_records**: 192
- **samples_without_source_link**: none
- **shard_report_cards**: 19
- **token_fields_present**: 14 entries
- **token_trace_records**: 23
- **tokens_traced**: 5038

### Throughput — PASS
_utilisation and loss density recomputed by summing ledger counters_

- **opus_rejection_rate**: 0.5
- **padding_waste**: 0.133416
- **raw_tokens_per_sec_compute**: 3601.2
- **recomputed_from_ledger**: 0.86273
- **recomputed_loss_density**: 0.81455
- **reconstruction_formulas**: independent_source=counters are summed from ledgers/consumption_*.jsonl (total_positions, pad_tokens, loss_bearing_tokens, context_only_tokens) - verify_evidence.py re-sums them from the ledger and compares, rather than trusting this file, loss_density=useful_loss_bearing_tokens / raw_positions, packing_utilisation=(raw_positions - pad_tokens) / raw_positions, useful_tokens_per_sec_compute=timed_phase_counters.useful_loss_bearing_tokens / timings.compute
- **reported_loss_density**: 0.820241
- **reported_packing_utilisation**: 0.866584
- **useful_tokens_per_sec_compute**: 2933.41

### Completion criterion — PASS
_a link walk over every consumed sample, not a spot check_

- **consumed_sample_instances**: 192
- **criterion**: `The assignment is complete only when the system can prove what it consumed, why it consumed it, what the model learned from it and how the run can be reconstructed.`
- **dangling_opus_decisions**: 0
- **proves_how_it_can_be_reconstructed**: yes
- **proves_what_it_consumed**: yes
- **proves_what_the_model_learned**: yes
- **proves_why_it_consumed_it**: yes
- **records_missing_provenance**: 0
- **samples_without_a_learning_record**: 0

