"""Which code the report shows, and the commentary that makes it readable.

The report's audience is not assumed to read Python. So every excerpt carries a
one-sentence summary of what the function does, and numbered notes anchored to
specific lines explaining what that line achieves and — where it is not
obvious — why it is written that way rather than the shorter way.

Notes anchor by a substring of the line rather than by line number, so they
follow the code when it moves.  A note whose anchor no longer matches is
reported at build time rather than silently dropped.
"""

from __future__ import annotations

# stage -> [ {file, symbol, summary, notes:[(anchor, commentary)]} ]
PICKS = {

"documents": [
 {"file":"tdes/corpus/loader.py", "symbol":"_split_roles",
  "summary":"Splits a document body into role-tagged spans — which is where the loss mask is really decided.",
  "notes":[
   ("has_marker = any(",
    "First decide whether this document is structured at all. Plain prose has no role markers, so it "
    "becomes one span tagged <code>text</code> and everything in it is graded."),
   ("for marker, role in ROLE_MARKERS.items()",
    "Walk the lines looking for <code>@user:</code>, <code>@think:</code>, <code>@tool_call:</code> and "
    "friends. The marker table is a module constant, so the vocabulary of roles is fixed rather than "
    "invented per file."),
   ("flush()",
    "When a new marker appears, close off the span that was being accumulated. This is what makes a "
    "span run from one marker to the next rather than stopping at a line break."),
   ("buffer = [line[len(marker):].strip()]",
    "The text on the marker line itself belongs to the span. Dropping it would silently lose the first "
    "line of every turn."),
  ]},
 {"file":"tdes/corpus/loader.py", "symbol":"parse_corpus_file",
  "summary":"Reads one corpus file into Document objects, attaching the source's provenance to each.",
  "notes":[
   ("raw = normalize(",
    "Normalization happens on the way in, once. Every later hash is taken over normalized text, so two "
    "files differing only in line endings produce the same shard."),
   ("chunks[1:]",
    "Skip the first chunk — it is the file header comment above the first <code>===DOC===</code> "
    "separator, not a document."),
   ('raise ValueError(f"{path.name}: document without',
    "Fail loudly on a malformed document rather than silently skipping it. A quietly dropped document "
    "would change the corpus hash with no explanation."),
   ("for required in (",
    "Three fields are mandatory. A document that cannot say what language it is in cannot be tagged, "
    "and an untagged document cannot be scheduled into a lane."),
   ("source_id=source.source_id",
    "Every document inherits its source's identity here — which is how licence and cleaning lineage "
    "reach the manifest later without being re-derived."),
  ]},
],

"tokenizer": [
 {"file":"tdes/tokenizer/normalize.py", "symbol":"normalize",
  "summary":"Canonicalises text so the same input always tokenizes identically — while leaving Indic joiners intact.",
  "notes":[
   ('text.replace("\\r\\n", "\\n")',
    "Fold Windows line endings first. Otherwise the same document checked out on two machines produces "
    "two different hashes."),
   ("_STRIP_RE.sub(",
    "Remove invisible formatting characters that carry no meaning — BOM, soft hyphen, word joiner, "
    "directional marks. Note what is <b>not</b> in that set: ZWJ and ZWNJ, which do carry meaning in "
    "Indic scripts."),
   ('unicodedata.normalize("NFC", text)',
    "The important line. NFC is what makes the two Unicode spellings of a nukta letter — क़ as one code "
    "point, or क plus a combining mark — collide into one token sequence instead of being learned twice."),
   ("_MANY_BLANKS_RE.sub(",
    "Collapse runs of blank lines so formatting noise does not consume tokens."),
  ]},
 {"file":"tdes/tokenizer/freeze.py", "symbol":"load_frozen",
  "summary":"Loads the frozen tokenizer and re-derives its hash before trusting it.",
  "notes":[
   ('recorded = document.pop("tokenizer_hash")',
    "Take the claimed hash out of the document, because the hash must be computed over everything "
    "<i>except</i> itself."),
   ('document.pop("frozen_at_unix", None)',
    "Drop the timestamp too. Retraining from the same corpus should produce the same hash, and a clock "
    "reading would make that impossible."),
   ("recomputed = tokenizer_hash(document)",
    "Recompute from the merge table. This is the whole point: reading a hash out of a file and trusting "
    "it proves nothing about whether the file was edited afterwards."),
   ("raise TokenizerIntegrityError(",
    "Refuse rather than warn. A tokenizer that has drifted reinterprets every token id in the archive, "
    "so continuing would corrupt everything downstream silently."),
  ]},
],

"shards": [
 {"file":"tdes/shards/builder.py", "symbol":"_tokenize_document",
  "summary":"Turns one document into tokens, recording which ranges are graded and which are context.",
  "notes":[
   ("tokens: List[int] = [bos]",
    "Every document starts with <code>&lt;bos&gt;</code>. That token is never graded — nothing precedes "
    "it to condition on — and it is what makes the first-position rule work when documents are packed "
    "together later."),
   ("marker = ROLE_SPECIAL_TOKEN.get(span.role)",
    "A role becomes a real special token in the stream, not a comment. The model sees "
    "<code>&lt;think&gt;</code> as a token it can learn to emit."),
   ("spans.append((span.role, start, len(tokens), span.graded))",
    "Record the range and whether it is graded. These offsets travel into the manifest and are what the "
    "packer reads to build the loss mask."),
   ('spans.append(("eos", eos_start, len(tokens), True))',
    "<code>&lt;eos&gt;</code> is graded on purpose. Ending is a behaviour the model has to learn, and "
    "perplexity at EOS is one of the few direct read-outs of whether document boundaries are working."),
  ]},
 {"file":"tdes/fsutil.py", "symbol":"_replace_with_retry",
  "summary":"Renames a temp file over the target atomically, working around two real Windows failures.",
  "notes":[
   ("_unlock(path)",
    "Clear the read-only bit before each attempt. Shards are deliberately read-only, so the rename would "
    "otherwise fail on every rebuild."),
   ("os.replace(tmp, path)",
    "<code>os.replace</code> is atomic on both POSIX and Windows — a reader sees the complete old file or "
    "the complete new one, never a half-written mixture."),
   ("time.sleep(0.02 * (attempt + 1))",
    "Back off and retry. A virus scanner opening the freshly created file holds a handle for a few "
    "milliseconds and makes the rename fail; this was hit during development, not imagined."),
   ("os.unlink(path)",
    "Last resort only. Removing the destination first briefly leaves no file at that path, which is "
    "weaker than an atomic swap — so it runs only after the retries are exhausted."),
  ]},
],

"manifests": [
 {"file":"tdes/shards/manifest.py", "symbol":"admission_decision",
  "summary":"The gate. Six independent rules; returns every reason the shard failed, not just the first.",
  "notes":[
   ("reasons: List[str] = []",
    "Collect reasons instead of returning early. One shard can fail for two unrelated causes, and an "
    "operator needs to see both — fixing one and re-running only to fail again wastes a cycle."),
   ("if not manifest.tokenizer_hash:",
    "No tokenizer hash means the token ids cannot be interpreted. That is disqualifying regardless of how "
    "good the text is."),
   ("if manifest.licence_tier not in TRAINABLE_LICENCE_TIERS:",
    "Licence is checked as data, from the source contract — not inferred from a filename or a folder."),
   ("if not manifest.cleaning_pipeline_hash:",
    "The admission contract: unknown cleaning lineage is a refusal. Nobody can vouch for data whose "
    "processing history is missing."),
   ('if manifest.contamination_status not in ("scanned_clean",):',
    "Note the polarity — the shard must have been <i>scanned and found clean</i>. Absence of evidence is "
    "not evidence of cleanliness, so <code>not_scanned</code> fails."),
   ("return (not reasons), sorted(set(reasons))",
    "Sorted and deduplicated, so the report is stable across runs and countable across shards."),
  ]},
 {"file":"tdes/shards/registry.py", "symbol":"_permission_for",
  "summary":"Maps a manifest to one of four permissions. Order matters — the first match wins.",
  "notes":[
   ("if manifest.never_train:",
    "Checked first. Test data is refused before anything else is considered, so no later condition can "
    "accidentally let it through."),
   ("return PERMISSION_VALIDATION",
    "Validation sits between train and test: readable, so loss can be measured, but never gradient-bearing."),
   ("if not manifest.admitted:",
    "A shard that failed the gate becomes <code>blocked</code> rather than disappearing — it stays in the "
    "registry so the audit can still see it exists and why it was refused."),
  ]},
],

"firewall": [
 {"file":"tdes/firewall/eval_firewall.py", "symbol":"EvalFirewall.check_batch",
  "summary":"The last gate before gradients. Re-derives everything from the batch rather than trusting its labels.",
  "notes":[
   ("entry = self.registry.get(shard_id)",
    "Look the permission up again, even though the packer already checked. This gate exists precisely "
    "because the earlier one might have been bypassed."),
   ('raise FirewallViolation(',
    "Raise, do not skip. A skipped batch would leave a hole in the schedule; an exception stops the run "
    "and makes the failure impossible to miss."),
   ("hits: List[ContaminationHit] = self.fingerprints.scan_token_text(decoded_text)",
    "Scan the <b>decoded text of what is about to be trained on</b>. Not the shard id, not the manifest — "
    "the actual tokens, decoded back to characters."),
   ("worst = max(hits, key=lambda h: h.overlap_ratio)",
    "Report the strongest hit in the error, so the message names a specific benchmark item rather than "
    "saying contamination happened somewhere."),
  ]},
 {"file":"tdes/firewall/contamination.py", "symbol":"EvalFingerprintRegistry.scan_text",
  "summary":"Runs three detectors with different blind spots and returns every hit.",
  "notes":[
   ("if digest in self.content_hashes:",
    "Exact match. Free to compute, and defeated by a single reformatted whitespace character — which is "
    "why it is not the only detector."),
   ("if canary in text:",
    "A canary is a unique string planted in the benchmark. Finding one in training text has no innocent "
    "explanation, which makes it the least ambiguous signal available."),
   ("shared = len(reference & candidate)",
    "Set intersection of 8-word shingles. This is what catches a copy that has been lightly edited, which "
    "is the realistic contamination case."),
   ("ratio = shared / len(reference)",
    "The critical detail: divide by the <b>benchmark item's</b> length, not the candidate's. A long "
    "document quoting one short test item in full scores 1.0 instead of being diluted to near zero."),
   ("if ratio >= self.overlap_threshold:",
    "0.55 separates a copy from ordinary shared phrasing. Below it, two documents merely both contain "
    "the phrase “in one hour the pipe fills”."),
  ]},
],

"mixture": [
 {"file":"tdes/mixture/floors.py", "symbol":"apportion",
  "summary":"Splits a whole number of sequences across lane weights so the parts always sum to the target.",
  "notes":[
   ("exact = {lane: total * max(0.0, w) / mass",
    "The ideal, fractional allocation — what each lane would get if sequences could be split."),
   ("floors = {lane: int(value) for lane, value in exact.items()}",
    "Take the whole part of each. This always under-allocates, which is the point: the shortfall is then "
    "distributed deliberately rather than by rounding accidents."),
   ("leftover = total - sum(floors.values())",
    "Exactly how many sequences are still unassigned."),
   ("order = sorted(exact, key=lambda lane: (-(exact[lane] - floors[lane]), lane))",
    "Rank by the size of the fraction that was discarded — largest remainder first. Ties break on the "
    "lane name, which is what makes the split reproducible on a replay rather than dependent on dict order."),
   ("floors[lane] += 1",
    "Hand out the leftovers. The result now sums to exactly <code>total</code>, every step, with no drift "
    "accumulating across the run."),
  ]},
 {"file":"tdes/mixture/floors.py", "symbol":"enforce_floors",
  "summary":"Raises any lane below its protected minimum, paying for it from the largest lane that can afford it.",
  "notes":[
   ("required = math.ceil(floor_fraction * total - 1e-9)",
    "Round up — a floor is a minimum, so half a sequence has to become a whole one. The epsilon avoids a "
    "floating-point value like 0.9999999 rounding up to two."),
   ("cap = available.get(lane, required)",
    "A floor cannot be met from an empty lane. Capping at what actually exists is what stops the schedule "
    "promising data that was never built."),
   ("donors = sorted(",
    "Take from the largest <i>unprotected</i> lane first, so satisfying one floor never breaks another."),
   ("adjustments.append(",
    "Record every intervention. A floor that quietly rewrote the mixture would make the planned shares a "
    "fiction; this is what makes the 18 rescues in this run visible."),
  ]},
],

"packing": [
 {"file":"tdes/packing/policies.py", "symbol":"pack_best_fit",
  "summary":"Longest piece first, into the tightest window that still fits.",
  "notes":[
   ("cuts = _split_at_boundaries(item, capacity)",
    "Oversized documents are cut first, at legal boundaries, so the packer only ever handles pieces that "
    "can fit."),
   ("pieces.sort(key=lambda p: (-p[2], p[0], p[1]))",
    "Descending by size — the decreasing in best-fit-decreasing. Large pieces placed first leave gaps that "
    "small pieces can fill; the reverse order strands the large ones. The two extra sort keys make the "
    "order deterministic when sizes tie."),
   ("slack = capacity - window.used - length",
    "How much space would be left over. Best fit means minimising this, not finding the first window that "
    "happens to work."),
   ("if slack == 0:",
    "A perfect fit cannot be improved on, so stop searching immediately."),
   ("result.windows.append(window)",
    "No existing window had room, so open a new one."),
  ]},
 {"file":"tdes/packing/policies.py", "symbol":"_split_at_boundaries",
  "summary":"Cuts an oversized document into window-sized pieces without cutting through a turn.",
  "notes":[
   ("boundaries = sorted(set(item.split_points))",
    "The legal cut points — for a conversation, the turn boundaries. A plain prose document has none, so "
    "it is cut anywhere."),
   ("usable = [b for b in boundaries if offset < b <= limit]",
    "Which boundaries fall inside the piece being cut."),
   ("cut = usable[-1] if usable else limit",
    "Take the <b>last</b> usable boundary, so the piece is as full as possible while still ending cleanly. "
    "If no boundary fits, fall back to cutting at the capacity — a document with one enormous turn cannot "
    "be helped."),
  ]},
],

"masks": [
 {"file":"tdes/packing/masks.py", "symbol":"validate_masks",
  "summary":"Six checks over one packed sample. These catch the bugs that still train and never throw.",
  "notes":[
   ("problems.append(f\"position {i}: loss on a pad token\")",
    "Check 1. Grading padding is the classic silent disaster: the loss drops beautifully because "
    "predicting <code>&lt;pad&gt;</code> four thousand times is effortless."),
   ("for i in first_positions(segment_ids):",
    "Check 2. The first token of each packed document has nothing before it to condition on, so it can "
    "never be a target however it was tagged."),
   ("if position_ids[i] != expected:",
    "Check 3. Positions must restart at 0 per document and increment by one — otherwise a document packed "
    "second inherits its neighbour's offsets."),
   ("problems.append(\"segments are interleaved rather than contiguous\")",
    "Check 4. Block-diagonal attention assumes each document occupies one contiguous run. Interleaved "
    "segments would make the mask wrong in a way nothing else notices."),
   ("problems.append(f\"position {i}: real token after padding began\")",
    "Check 5. Padding must be a suffix. A hole in the middle breaks the assumption that pad positions can "
    "be ignored wholesale."),
   ("if loss_mask[i] and not graded_flags[i]:",
    "Check 6, and the strongest one: compare the mask against an independently derived list of which "
    "tokens the role spans said were graded. This is what catches loss leaking onto a prompt."),
  ]},
 {"file":"tdes/training/model.py", "symbol":"build_attention_bias",
  "summary":"Builds the attention mask that is both causal and confined to one packed document.",
  "notes":[
   ("causal = t.tril(",
    "Lower triangle — no position may attend to a later one. Standard for a decoder."),
   ("same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)",
    "Broadcast comparison producing an L×L boolean: true where query and key belong to the same packed "
    "document. This one line is the whole cross-document defence."),
   ("allowed = causal & same_segment",
    "Both constraints at once. Without the second, packing would teach the model that an unrelated "
    "document is a natural continuation."),
   ("allowed = allowed | eye",
    "Keep the diagonal alive so no row is entirely masked. A fully masked row makes softmax produce NaN; "
    "padding rows attend only to themselves and are discarded by the loss mask anyway."),
   ('bias.masked_fill_(~allowed.unsqueeze(1), float("-inf"))',
    "Turn the boolean into an additive bias — 0 where allowed, −∞ where not — which softmax turns into "
    "exactly zero attention weight."),
  ]},
],

"planner": [
 {"file":"tdes/streams/planner.py", "symbol":"BatchPlanner.plan",
  "summary":"Returns what step N may consume. Pure: same inputs, same answer, forever.",
  "notes":[
   ("quota = self.schedule.steps[step]",
    "The per-lane counts, compiled before the run started. The planner does not decide the mixture, it "
    "serves it."),
   ("cursor = self.cursors[step]",
    "Precomputed prefix sums. This is why the function is O(1) — it never walks the history to work out "
    "where it is."),
   ("ids, lane_passes = self._draw(context, lane, cursor[lane], count)",
    "Draw this lane's candidates from its shuffled stream, wrapping if the stream is shorter than the run "
    "needs."),
   ("return StepPlan(",
    "Nothing has touched a model, a file or a random generator. That is what makes the plan recomputable "
    "months later by anyone holding the seed."),
  ]},
 {"file":"tdes/streams/planner.py", "symbol":"assemble_batch",
  "summary":"Lays accepted samples across ranks and accumulation slots, then names the batch by its contents.",
  "notes":[
   ("while queues:",
    "Round-robin across lanes rather than taking one lane at a time, so no rank spends a whole microbatch "
    "on a single lane."),
   ("if len(ordered) != capacity:",
    "The batch geometry is fixed. Refuse loudly rather than train a short batch, which would quietly "
    "change the effective batch size for that step."),
   ("for accum_index in range(grad_accum):",
    "Fill accumulation slot by accumulation slot, rank within slot — the same order the trainer will "
    "consume them in."),
   ("microbatch_id=(",
    "A readable, deterministic id: step, rank, accumulation index. This is the key the resume comparison "
    "matches on."),
  ]},
],

"opus": [
 {"file":"tdes/opus/proxy.py", "symbol":"OpusProxy.score",
  "summary":"Scores one candidate by how closely its gradient points the way the golden set wants to go.",
  "notes":[
   ("self.model.zero_grad(set_to_none=True)",
    "Clear gradients first, so the score measures this candidate alone and nothing accumulated earlier."),
   ("loss.backward()",
    "A real backward pass. This is what makes the score a measurement rather than a heuristic — and it is "
    "why scoring uses only a 64-token prefix."),
   ("flat = _flat_grad(self.params)",
    "Flatten the gradient over the probe parameters — the last block plus the final norm — into one "
    "vector, so it can be compared with the direction by a dot product."),
   ("self.model.zero_grad(set_to_none=True)",
    "Clear again immediately. Scoring must leave no trace on the training step that follows."),
   ("alignment = float(t.dot(flat, self.direction.vector) / grad_norm)",
    "Cosine similarity: the direction vector is already unit length, so dividing by this gradient's norm "
    "completes it. Positive means this batch pushes the weights the way the golden set would."),
  ]},
 {"file":"tdes/opus/selector.py", "symbol":"OpusSelector.select_step", "max_lines":130,
  "summary":"Scores a lane's candidates, accepts the quota, and records a reason for every one — accepted or not.",
  "notes":[
   ("threshold = _median(",
    "The bar is the round's own median, not a constant. A fixed threshold stops meaning anything as the "
    "model moves and scores drift."),
   ("key=lambda e: (-round(e[\"opus_score\"], 6), e[\"candidate_id\"])",
    "Rank by score, ties broken on id. Rounding to 6dp first means a knife-edge tie cannot flip between "
    "the original run and the re-run after the crash."),
   ("if content in seen_hashes:",
    "Two identical windows in one step teach nothing the first did not, so the duplicate is a hard "
    "rejection."),
   ("demoted.append((entry, REASON_STAGE_MISMATCH))",
    "Demoted, not rejected. Rejecting outright can empty a lane whose whole pool is hinted for a later "
    "phase — and an empty lane cannot fill a fixed batch geometry. This was a real bug, found and fixed."),
   ("reason = note or (",
    "When the quota still is not full, the reason records <i>why</i> a weak candidate was taken: a floor "
    "override for a protected lane, ordinary quota pressure otherwise."),
   ("# pass 3: nothing is deleted",
    "Everything unused is deferred, never deleted. A batch that is low value now may be exactly right "
    "during annealing."),
  ]},
],

"training": [
 {"file":"tdes/training/loop.py", "symbol":"Trainer.run_step", "max_lines":150,
  "summary":"One optimizer update. The order of these operations is the load-bearing part.",
  "notes":[
   ("plan = self.planner.plan(step)",
    "Step 1: what may be consumed. Pure, no model involved."),
   ("if step % cfg.opus_round_interval == 0",
    "Recompute the proxy direction at round boundaries only, and align those to checkpoints so a resume "
    "can restore the exact state the direction came from."),
   ("loss_before = self.per_sample_loss(samples)",
    "Measure before the update, on exactly the tokens about to be trained on. Without this, “did this "
    "batch help” is guesswork."),
   ("self.firewall.check_batch(",
    "The last gate. Runs on decoded text, immediately before any gradient exists."),
   ("(loss / cfg.grad_accum).backward()",
    "Divide by the accumulation count so the accumulated gradient equals the mean over the whole global "
    "batch rather than the sum."),
   ("self.consumption.record_microbatch(",
    "<b>The critical ordering.</b> The record is written and fsynced here — before the optimizer step. A "
    "batch that was served must be recorded even if the process dies next, and that is exactly what "
    "leaves the ledger ahead of the model after the crash."),
   ("if crash_at is not None and step == crash_at",
    "The injected crash fires mid-step, after some microbatches are durable and before the update."),
   ("self.optimizer.step()",
    "Only now do the weights change."),
   ("loss_after = self.per_sample_loss(samples)",
    "Measure again on the same tokens. The difference is a measurement, not an inference."),
  ]},
],

"consumption": [
 {"file":"tdes/ledger/store.py", "symbol":"LedgerStore.append",
  "summary":"Appends one record, chained to the previous by hash and flushed to disk before returning.",
  "notes":[
   ("prev_hash = self._tail_hash",
    "Each record embeds the previous record's hash. That is the chain: remove or edit anything and every "
    "hash after it stops matching."),
   ("event_hash = compute_event_hash(seq, prev_hash, event_type, payload)",
    "The record's own identity, computed over its sequence number, its predecessor, its type and its "
    "payload together."),
   ("line = canonical_json(record) ",
    "Canonical JSON — sorted keys, no incidental whitespace — so byte offsets are stable and two equal "
    "payloads always produce identical bytes."),
   ("os.fsync(fh.fileno())",
    "Force it to the platter before returning. Without this the crash would lose records that were "
    "genuinely served, which is the failure the whole ordering was designed to avoid."),
  ]},
 {"file":"tdes/ledger/store.py", "symbol":"LedgerStore.verify_chain",
  "summary":"Walks the whole ledger recomputing every hash. Any edit, deletion or reordering shows up here.",
  "notes":[
   ('return False, {"ledger": self.name, "error": "seq_gap"',
    "Sequence numbers must be dense. A missing record leaves a gap — which is also how a corrupted line "
    "that was skipped during reading gets caught."),
   ('if rec["prev_hash"] != prev:',
    "Each record must point at the one actually before it, so records cannot be reordered."),
   ("recomputed = compute_event_hash(",
    "Recompute rather than compare stored values. A tampered payload produces a different hash even "
    "though the file still parses."),
   ("prev = rec[\"event_hash\"]",
    "Advance the chain and continue. One forward pass verifies the entire history."),
  ]},
],

"learning": [
 {"file":"tdes/ledger/learning.py", "symbol":"_classify",
  "summary":"Turns a shard's measurements into a verdict the next corpus can act on.",
  "notes":[
   ("if mean_ppl and mean_ppl <= CONFIG.learned_out_ppl:",
    "The design's threshold. A shard whose average perplexity has fallen to ~1.2 is already predicted by "
    "the model — training on it further is spent compute, so it is <b>exhausted</b> rather than bad."),
   ("if mean_grad > 0 and max_grad > 8.0 * mean_grad",
    "A gradient spiking to eight times the mean marks a shard as <b>harmful</b>: it destabilises the "
    "update and needs cleaning, later staging, or a warmup before reuse."),
   ("if mean_delta < -1e-4:",
    "Loss on the shard's own tokens went down across the update — the direct evidence that exposure "
    "helped. <b>Useful</b>."),
   ("return CLASS_NEUTRAL,",
    "Everything else moved the loss by less than noise. Recorded as neutral rather than guessed at."),
  ]},
 {"file":"tdes/ledger/learning.py", "symbol":"LearningLedger.next_corpus_recommendations",
  "summary":"Converts the report cards into an instruction for the next corpus: collect, protect, repeat, defer, reject.",
  "notes":[
   ("cards = self.shard_report_cards()",
    "Built from the per-sample records, so every recommendation traces back to measurements rather than "
    "to an opinion."),
   ('actions["collect_more"].append(entry)',
    "A shard that reduced loss is worth more of. This is the signal the next corpus needs."),
   ('if card["mean_token_perplexity"] > 3.0:',
    "Useful <i>and</i> still surprising means the model has not exhausted it — worth protecting with a "
    "floor rather than merely collecting more of."),
   ('if card["repeat_effect"] > 0 and len(card["passes_seen"]) > 1:',
    "The later pass helped less than the first, so the repetition budget for that shard is spent."),
  ]},
],

"checkpoint": [
 {"file":"tdes/training/checkpoint.py", "symbol":"build_meta",
  "summary":"Assembles the metadata that makes a checkpoint resumable and auditable.",
  "notes":[
   ("next_expected_plan_hash=next_expected_plan_hash",
    "The checkpoint records what the <i>next</i> step should serve. On resume this is compared against a "
    "fresh recomputation from the seed — two independent sources agreeing."),
   ("ledger_offset=consumption_offset.as_dict()",
    "The data position: byte offset, event sequence and the hash at that point. This is the field that "
    "makes recovery a truncation rather than a guess."),
   ("rng_fingerprint=rng_fingerprint(CONFIG.master_seed, branch_id, global_step)",
    "RNG is derived from position, so this is a checkable fingerprint rather than a blob of generator "
    "state that has to be restored perfectly."),
   ("config_hash=CONFIG.config_hash",
    "Every setting that influences the data stream, hashed. A checkpoint can never be silently compared "
    "against a run that used different settings."),
  ]},
 {"file":"tdes/training/checkpoint.py", "symbol":"latest_checkpoint",
  "summary":"Finds the newest checkpoint that is actually complete.",
  "notes":[
   ('if (directory / "meta.json").exists() and (directory / "state.pt").exists():',
    "Both files must be present. A directory with only one of them is what an <i>interrupted</i> "
    "checkpoint looks like — choosing it would defeat the recovery it is meant to enable."),
   ("candidates.append((step, directory))",
    "Only complete checkpoints become candidates."),
   ("return max(candidates)[1]",
    "Highest step number wins."),
  ]},
],

"crash": [
 {"file":"tdes/training/crash.py", "symbol":"die_mid_write",
  "summary":"Writes part of a real ledger record and terminates the process without cleanup.",
  "notes":[
   ("record = {",
    "Build a genuine record first. The fragment left on disk has to look like a real record that was "
    "interrupted, not like arbitrary garbage."),
   ("cut = max(16, int(len(line) * fraction))",
    "Cut at 55% of the line. Enough to be unmistakably a partial record, not so little that it could be "
    "confused with an empty line."),
   ("handle.write(fragment)",
    "Write the fragment with <b>no terminating newline</b> — that missing newline is precisely what makes "
    "the tail detectable as torn."),
   ("os.fsync(handle.fileno())",
    "Force the partial bytes to disk, so the wreckage survives the process dying."),
   ("os._exit(CRASH_EXIT_CODE)",
    "<code>os._exit</code>, not <code>sys.exit</code>. It skips atexit handlers, buffered flushes and "
    "destructors — the cleanup a real kill does not get to do. Code 137 is 128 + SIGKILL."),
  ]},
],

"resume": [
 {"file":"tdes/streams/resume.py", "symbol":"prepare_resume", "max_lines":95,
  "summary":"Repairs, restores and rolls back — in that order, because each step depends on the last.",
  "notes":[
   ("torn = consumption_store.repair_torn_tail()",
    "First. Nothing can parse the ledger while the last line is a fragment, so the repair has to precede "
    "everything else."),
   ("path = latest_checkpoint(branch_id, checkpoint_root)",
    "The newest <i>complete</i> checkpoint — incomplete ones are skipped inside this call."),
   ("offset = LedgerOffset.from_dict(meta.ledger_offset)",
    "Read the data position the checkpoint recorded. This is the whole reason the offset is stored."),
   ("discarded = consumption_store.rollback_to(offset)",
    "Truncate. The discarded records describe batches served to a model state that no longer exists — "
    "keeping them would make the resume look like a repeat."),
   ("EVENT_ROLLBACK,",
    "Log the rollback itself, with the hashes of everything it removed. This is the one non-append "
    "operation in the system, so it leaves a trace rather than silently shrinking the file."),
  ]},
 {"file":"tdes/streams/resume.py", "symbol":"verify_rollback_replay",
  "summary":"Proves the re-served batches are byte-identical to the ones the rollback threw away.",
  "notes":[
   ("discarded = [",
    "The fingerprints captured before truncation — step, rank, microbatch, batch id, sample ids, token "
    "spans, token and mask hashes. All integers, so the comparison is exact."),
   ('and r["seq"] > state.meta.ledger_offset["event_seq"]',
    "Only look at records written <i>after</i> the rollback point — those are the re-served ones."),
   ("missing.append(original)",
    "A discarded record that never came back is a <b>skip</b>."),
   ("if current != original:",
    "A record that came back different is a <b>repeat</b> of something else. Either way the resume claim "
    "fails."),
   ("identical = not mismatches and not missing",
    "Both conditions must hold. This is the strong form of “no skipped or repeated batches” — the "
    "assignment's exact criterion, checked rather than asserted."),
  ]},
],

"replay": [
 {"file":"tdes/streams/replay.py", "symbol":"replay_interval", "max_lines":110,
  "summary":"Rebuilds a past interval three independent ways and requires all three to agree.",
  "notes":[
   ("records = [",
    "Derivation 1: what the ledger says was served. Read, not recomputed — the design notes is explicit that "
    "re-running risks nondeterminism."),
   ("entry.reconstructed_tokens_hash.append(hash_token_ids(sample.token_ids))",
    "Derivation 2: rebuild the sample and re-hash it from scratch."),
   ("raw = shard_reader.tokens(",
    "And go further — re-read the named byte range straight out of the shard file. This is a genuine round "
    "trip through storage, not a cache lookup."),
   ("recomputed = planner.plan_hash(step)",
    "Derivation 3: what the planner says step N should have been offered, derived from the seed with the "
    "ledger never consulted."),
   ("all_matched = bool(replayed) and tokens_matched",
    "All three must agree. Agreement between the ledger and the shards proves the tokens are still there; "
    "agreement with the recomputed plan proves the ordering was not fabricated afterwards."),
  ]},
 {"file":"tdes/ledger/branch.py", "symbol":"divergence_report",
  "summary":"Checks a fork diverged — and only after the point it was supposed to.",
  "notes":[
   ("before = [s for s in shared if s < fork_point]",
    "Steps the two branches share before the divergence point."),
   ("identical_before = [s for s in before if parent_batches[s] == fork_batches[s]]",
    "These must match. If they do not, the fork did not really start from the parent's state and any "
    "comparison between the two is meaningless."),
   ("differing_after = [s for s in after if parent_batches[s] != fork_batches[s]]",
    "These must differ. If they do not, the fork changed nothing and the experiment is measuring noise."),
   ('"diverged_correctly": (len(identical_before) == len(before)) and bool(differing_after)',
    "Both halves together. Either alone would pass while the experiment was still confounded."),
  ]},
],

"audit": [
 {"file":"tdes/audit/auditor.py", "symbol":"detect_loss_spikes",
  "summary":"Finds steps where the loss jumped relative to how it normally moves.",
  "notes":[
   ("deltas = [losses[i][1] - losses[i - 1][1]",
    "Work on the step-to-step <i>change</i>, not the absolute loss. The absolute value falls throughout a "
    "run, so any fixed threshold stops meaning the same thing after a while."),
   ("spread = statistics.pstdev(deltas) or 1e-9",
    "How much the loss normally moves between steps. The fallback avoids dividing by zero on a perfectly "
    "flat series."),
   ("if delta > mean + sigma * spread:",
    "A spike is a change that is unusual <i>for this run</i> — 1.5 standard deviations above the mean "
    "change."),
   ('"z_score": round((delta - mean) / spread, 4)',
    "Report how unusual, not just that it happened, so a reader can judge severity."),
  ]},
 {"file":"tdes/audit/auditor.py", "symbol":"batches_before_spike",
  "summary":"Answers the second question: what was fed to the model just before a spike.",
  "notes":[
   ("window = shards_between_steps(consumption, branch_id, low, spike_step + 1)",
    "Which shards and token spans were consumed in the steps leading up to the spike."),
   ('if r["type"] == "opus_decision"',
    "Pull the selection decisions from the same window — this is only possible because OPUS recorded them "
    "at the time."),
   ('accepted = [d for d in decisions if d["status"] == "accepted"]',
    "Narrow to what actually entered the batch."),
   ('"opus_overrides_in_window": sum(',
    "Count the protected-floor overrides specifically. A spike preceded by several forced acceptances is a "
    "different story from one preceded by high-scoring data."),
   ('"highest_gradient_norm_candidate": max(',
    "Name the single most likely culprit, so the answer points at a shard rather than at a step range."),
  ]},
],

"throughput": [
 {"file":"tdes/perf/metrics.py", "symbol":"PerfTracker.report",
  "summary":"Turns raw counters into the four throughput figures, and ships the formulas to rebuild them.",
  "notes":[
   ("compute = self.timers.get(\"compute\", 0.0) or 1e-9",
    "Compute time only — loader and scoring time are tracked separately so they cannot inflate the rate."),
   ('"useful_tokens_per_sec_compute": round(c.useful_tokens / compute, 2)',
    "The figure that matters: tokens the model was actually graded on, per second. Raw throughput counts "
    "padding and context, which teach nothing."),
   ('"packing_utilisation": _ratio(',
    "How much of each window held a real token."),
   ('"loss_density": _ratio(c.useful_tokens, c.raw_positions)',
    "How much of each window was graded. Always the smaller number — the gap is context-only tokens."),
   ('"how_to_reconstruct"',
    "Ship the arithmetic alongside the numbers. The assignment is explicit that figures which cannot be "
    "rebuilt earn no credit, so the report says exactly how to rebuild each one."),
  ]},
],
}
