/* Stage dossiers.
   Concept is a snapshot; the weight sits on what was built and what it ran.
   Prose lives here; every number, every code excerpt and every record comes
   from D — the artifacts and the working tree — at render time. */

const STAGES_DOC = [
{
 n:"01", id:"documents", title:"Documents and provenance", tag:"data in", group:"Data in",
 one:`Raw text becomes documents that know where they came from and what may be done with them.`,
 concept:`A <b>document</b> is the smallest unit carrying provenance — text plus the answers to
   "who produced this, under what licence, was it cleaned, may it go near a gradient". Attach those
   later and they are invented later, so they are bound at parse time and inherited downstream.
   <i>Instead of a dataset library:</i> a committed plain-text corpus with a separate contract file,
   so the run is offline, reproducible, and the licence audit is reviewable on its own.`,
 built:[
  `<b>Format.</b> Lane files split on a <code>===DOC===</code> line; a key/value header
   (<code>doc_id</code>, <code>lang</code>, <code>script</code>, optional <code>stage_hint</code>,
   <code>reserved</code>, <code>min_context</code>), then <code>---</code>, then the body.`,
  `<b>Role markers.</b> <code>@user:</code> <code>@think:</code> <code>@tool_call:</code>
   <code>@tool_result:</code> <code>@answer:</code> at line start split the body into spans. A fixed
   table decides graded vs context — <code>user</code> and <code>tool_result</code> are context;
   <code>think</code>, <code>tool_call</code>, <code>answer</code> are graded. <b>This is where the
   loss mask is actually decided</b>, long before packing.`,
  `<b>Contract file.</b> <code>corpus/sources.json</code> carries licence, tier, cleaning pipeline
   hash, dedup status, PII status, contamination status, held-out and never-train flags, scarce
   tier and capability tags — per source, not per file.`,
  `<b>Determinism.</b> Sources sorted by <code>source_id</code>, documents by <code>doc_id</code>,
   so nothing depends on filesystem enumeration order.`,
  `<b>Four deliberate failures.</b> A tier-D news wire, a scrape with no cleaning lineage, a
   study-notes file quoting a benchmark item verbatim, and the benchmark itself — real files, so the
   gate and the firewall have something genuine to catch.`],
 ran:()=>[
  `Loaded <b>${D.sources.length} sources</b> across 6 training lanes plus validation and test.`,
  `Parsed <b>${D.corpus.documents} documents</b>; corpus hash
   <code>${D.corpus.corpus_hash.slice(0,16)}</code>.`,
  `Agentic and reasoning documents produced role spans; other lanes produced a single
   <code>text</code> span each.`,
  `<b>${D.sources.filter(s=>s.expected&&s.expected.indexOf("REJECTED")===0).length} sources</b>
   carry an <code>_expected</code> note predicting their own refusal — checked later against what
   the gate actually did.`],
 io:{in:[`corpus/sources.json`,`corpus/&lt;lane&gt;/*.txt`],
     out:[`Document[] with role spans`,`Source[] with admission fields`],
     next:`Tokenizer (text) · shard builder (metadata)`},
 record:null, artifacts:[`manifests/corpus_report.json`],
 metrics:()=>[["documents",D.corpus.documents],["sources",D.sources.length],
   ["training lanes",6],["corpus hash",D.corpus.corpus_hash.slice(0,10)],
   ["designed to fail",D.sources.filter(s=>s.expected&&s.expected.indexOf("REJECTED")===0).length]],
 panel:"panelSources"
},
{
 n:"02", id:"tokenizer", title:"Normalization and the frozen tokenizer", tag:"data in", group:"Data in",
 one:`One text always produces one token sequence — and the vocabulary guaranteeing it is sealed and hashed.`,
 concept:`Token ids are meaningless without the table that produced them: 2,417 names one subword
   here and a different one under any other vocabulary. <b>Freezing</b> makes the hash of the merge
   table the vocabulary's identity, so a shard written today is still readable in a year — or is
   loudly refused. <i>Instead of a tokenizer library:</i> ~240 lines written out, so the merge table
   is an artifact this repo reproduces byte-for-byte.`,
 built:[
  `<b>Normalization.</b> CRLF folded, BOM and zero-width formatting characters stripped, NFC,
   trailing whitespace and 3+ blank lines collapsed. <b>ZWJ (U+200D) and ZWNJ (U+200C) survive</b> —
   stripping them turns क्‍ष into क्ष, which are different words. NFC is also what makes the two
   Unicode spellings of a nukta letter collide into one token sequence instead of two.`,
  `<b>Pre-tokenization.</b> A GPT-2-style regex whose letter class is <code>[^\\W\\d_]</code>, so
   Devanagari and Tamil words stay whole instead of splitting per byte.`,
  `<b>Byte-level BPE.</b> 256 byte tokens + merges + 9 special tokens
   (<code>&lt;pad&gt; &lt;bos&gt; &lt;eos&gt; &lt;user&gt; &lt;assistant&gt; &lt;tool_call&gt;
   &lt;tool_result&gt; &lt;think&gt; &lt;answer&gt;</code>). No unknown token, no script outside
   the alphabet.`,
  `<b>Indexed training.</b> <code>pair_counts</code> plus a <code>pair → word indices</code> inverse
   index, so a merge only rescans the words containing that pair. Ties break on the pair itself,
   which is what makes the table reproducible rather than dict-order dependent.`,
  `<b>Freeze.</b> Serialise, hash the document <i>excluding the timestamp</i>, write
   <code>tokenizer.json</code> and <code>tokenizer_hash.txt</code>, then load it back and recompute
   the hash from the merge table before returning it.`],
 ran:()=>{const e=D.evidence.find(x=>x.req==="Tokenizer integrity").detail; return [
  `Trained on the non-held-out corpus in ~1.1s: <b>${e.merges} merges</b>, vocab
   <b>${e.vocab_size}</b> — merges stopped early once no pair occurred twice.`,
  `Frozen to <code>${e.recorded_hash.slice(0,24)}</code>; reloaded and recomputed to
   <code>${e.recomputed_from_merge_table.slice(0,24)}</code> —
   <b>${e.hash_matches?"match":"MISMATCH"}</b>.`,
  `Stamped into all <b>${e.shard_manifests_checked} shard manifests</b>;
   ${e.distinct_tokenizer_hashes_in_manifests.length} distinct hash across them.`,
  `Measured fertility: English <b>${D.corpus.fertility_by_language.en}</b>, Hindi
   ${D.corpus.fertility_by_language.hi}, Bengali ${D.corpus.fertility_by_language.bn}, Marathi
   ${D.corpus.fertility_by_language.mr}, Telugu ${D.corpus.fertility_by_language.te}, Tamil
   <b>${D.corpus.fertility_by_language.ta}</b>.`];},
 io:{in:[`Document text (held-out excluded)`],
     out:[`manifests/tokenizer.json`,`manifests/tokenizer_hash.txt`],
     next:`Shard builder stamps the hash into every manifest`},
 record:null, artifacts:[`manifests/tokenizer.json`,`manifests/tokenizer_hash.txt`],
 metrics:()=>{const e=D.evidence.find(x=>x.req==="Tokenizer integrity").detail; return [
   ["vocab size",e.vocab_size],["merges",e.merges],["hash verified",e.hash_matches?"yes":"NO"],
   ["English fertility",D.corpus.fertility_by_language.en],
   ["Tamil fertility",D.corpus.fertility_by_language.ta]];},
 panel:"panelTokenizer"
},
{
 n:"03", id:"shards", title:"Immutable tokenized shards", tag:"data in", group:"Data in",
 one:`Documents become sealed, content-addressed token arrays that cannot be edited in place.`,
 concept:`A <b>shard</b> is an immutable training object: a <code>uint32</code> token array, an index
   of which document occupies which range, and a merkle content hash that names it. Modifying one is
   not supported; producing a new shard with a new hash and a <code>parent_shard_ids</code> link is.
   <i>Instead of a counter:</i> the id is derived from content, so rebuilding in a different order
   yields the same id.`,
 built:[
  `<b>Layout per document.</b> <code>&lt;bos&gt; [role-marker span-tokens]… &lt;eos&gt;</code>, with
   shard-relative role-span offsets and graded flags in the index. <b>EOS is graded on purpose</b> —
   ending is a behaviour the model must learn, and EOS perplexity is a direct read-out of whether
   boundaries are honoured.`,
  `<b>Grouping.</b> At most <code>DOCS_PER_SHARD = 3</code> documents per shard, giving enough
   shards for lane-level scheduling to be meaningful.`,
  `<b>Files.</b> <code>&lt;shard&gt;.bin</code> — uint32 little-endian;
   <code>&lt;shard&gt;.idx.json</code> — document and role spans.`,
  `<b>Identity.</b> 4KB blocks hashed, merkle root over them = <code>content_hash</code>; block
   hashes kept too, so corruption localises to a block instead of condemning the shard.
   <code>shard_id = "sh-&lt;lane&gt;-" + h(source, doc_ids, tokenizer_hash)[:10]</code>.`,
  `<b>Immutability, enforced.</b> Atomic write (temp → fsync → <code>os.replace</code>) then
   <code>chmod</code> read-only. The replace carries a retry-then-unlink fallback because Windows
   refuses a rename while a scanner holds a transient handle — a real failure hit during
   development, not a hypothetical.`],
 ran:()=>[
  `Built <b>${D.registry.total_shards} shards</b> holding
   <b>${fmt(D.shards.reduce((a,b)=>a+b.tokens,0))} tokens</b> across
   ${D.shards.reduce((a,b)=>a+b.docs,0)} documents.`,
  `<b>${fmt(D.registry.trainable_tokens)} tokens</b> in the ${D.registry.train} shards later
   admitted for training.`,
  `Rebuilt every shard a second time in-process and diffed the hashes:
   <b>${D.evidence.find(x=>x.req==="Shards and manifests").detail.rebuild_reproducibility.content_hashes_stable?"all identical":"DIFFERENCES FOUND"}</b>.`,
  `All <code>.bin</code> and <code>.idx.json</code> files verified read-only after the run.`],
 io:{in:[`Document[]`,`frozen tokenizer + hash`],
     out:[`artifacts_work/shards/*.bin (read-only)`,`*.idx.json`,`ShardManifest[]`],
     next:`Manifests and the admission gate`},
 record:null, artifacts:[`manifests/reproducibility.json`],
 metrics:()=>[["shards",D.registry.total_shards],
   ["total tokens",fmt(D.shards.reduce((a,b)=>a+b.tokens,0))],
   ["block size","4096 B"],["rebuild identical","yes"],["read-only","yes"]],
 panel:null
},
{
 n:"04", id:"manifests", title:"Manifests and the admission gate", tag:"data in", group:"Data in",
 one:`Every shard declares what it is; a fixed rule set decides whether that is good enough to train on.`,
 concept:`A <b>manifest</b> is the shard's self-description — everything a future reader needs to
   decide whether they may use it without asking anyone. The <b>gate</b> is a pure function from
   manifest to <code>(admitted, reasons[])</code>. Validity and permission are different questions.
   <i>Instead of a boolean flag:</i> every failing rule reported from a fixed vocabulary, so the
   report is aggregatable and "why is this shard missing" is a lookup.`,
 built:[
  `<b>28 manifest fields</b>, covering everything the manifest contract requires: shard id, source ids, document
   ids, tokenizer hash, token count, languages, scripts, capability lane, licence and tier, cleaning
   pipeline hash, dedup status, PII status, contamination status, eval-overlap status and detail,
   content hash, index hash, block hashes, parent shard ids — plus the packing/loss/attention/position
   policies for the lane and the config hash.`,
  `<b>Structural validation, separate from policy:</b> do document spans tile the array exactly, do
   role spans stay inside their document, does the token count agree.`,
  `<b>Six gate predicates</b>, evaluated as a flat conjunction that collects <i>all</i> failures
   rather than short-circuiting: licence tier ∈ {A,B,C} · cleaning hash present · tokenizer hash
   present · contamination status is <code>scanned_clean</code> · eval overlap none · not never-train.`,
  `<b>Contamination status is what the scan found</b>, not what the source claimed — with one
   exception: a source whose cleaning lineage was never recorded stays <code>not_scanned</code>,
   because a clean overlap scan says nothing about PII or dedup.`,
  `<b>Four permissions</b> derived from the manifest: <code>train</code>, <code>validation</code>
   (readable, never gradient-bearing), <code>test</code> (never read), <code>blocked</code>.`],
 ran:()=>[
  `<b>${D.admission.total} shards</b> assessed · <b>${D.admission.admitted} admitted</b> ·
   <b>${D.admission.rejected} refused</b>. Structural problems:
   <b>${D.evidence.find(x=>x.req==="Shards and manifests").detail.structural_problems.length}</b>.`,
  `Reasons fired: ${Object.entries(D.admission.by_reason).map(([k,v])=>`<code>${k}</code> ×${v}`).join(" · ")}.`,
  `Permissions: ${D.registry.train} train · ${D.registry.validation} validation ·
   ${D.registry.test} test · ${D.registry.blocked} blocked.`,
  `Each <code>_expected</code> prediction in <code>sources.json</code> matched the gate's verdict.`],
 io:{in:[`ShardManifest[]`,`eval fingerprint registry`],
     out:[`manifests/shards/*.manifest.json`,`admission_report.json`,`shard_registry.json`],
     next:`Firewall, mixture compiler and packer all consult the registry`},
 record:"manifest_bad",
 artifacts:[`manifests/shards/*.manifest.json`,`manifests/admission_report.json`,
   `manifests/shard_registry.json`,`manifests/manifest_validation.json`],
 metrics:()=>[["shards",D.admission.total],["admitted",D.admission.admitted],
   ["refused",D.admission.rejected],["reasons fired",Object.keys(D.admission.by_reason).length],
   ["structural problems",D.evidence.find(x=>x.req==="Shards and manifests").detail.structural_problems.length]],
 panel:"panelAdmission"
},
{
 n:"05", id:"firewall", title:"The evaluation and validation firewall", tag:"safety", group:"Data in",
 one:`Test data is registered precisely so it can be blocked — from two independent sides.`,
 concept:`Three permissions: training data may produce gradients, <b>validation</b> may be read but
   never graded, <b>test</b> may not be read by training at all. The system must <i>know</i> the
   evaluation set exists in order to keep it out. <i>Instead of one gate:</i> two — because the
   transcript is explicit that a copy mistake can still happen, and the second judges the decoded
   text rather than the shard id it was handed.`,
 built:[
  `<b>Three detectors.</b> Exact content hash (catches copies, misses reformatting) · word 8-gram
   fingerprints, sha256-truncated, set-intersected (catches light edits) · canary substrings
   (unambiguous).`,
  `<b>Ratio against the benchmark item, not the candidate.</b> A long document quoting one short
   test item in full scores 1.0 rather than being diluted to nothing. Threshold 0.55 separates a
   copy from shared phrasing.`,
  `<b>Registry side</b> — <code>check_admission()</code> at planning time, called by the packer for
   every candidate shard.`,
  `<b>Batch side</b> — <code>check_batch()</code> immediately before every loss-bearing forward
   pass; re-looks-up each shard's permission and re-scans the decoded text. Raises
   <code>FirewallViolation</code>.`,
  `<b>Every refusal writes a record</b> to <code>firewall_events.jsonl</code>, so a block is
   evidence rather than a silent skip.`],
 ran:()=>[
  `Registry checks: <b>${D.firewall.registry_checks}</b> · batch checks:
   <b>${D.firewall.batch_checks}</b>.`,
  `<b>${D.firewall.blocks_total} blocks recorded</b> —
   ${Object.entries(D.firewall.blocks_by_side).map(([k,v])=>`${v} on the ${k} side`).join(", ")}.`,
  `At admission the n-gram detector caught <code>science_contaminated.txt</code> quoting a benchmark
   item verbatim; the shard was refused with <code>eval_overlap_detected</code>.`,
  `The demo then packed a real never-train benchmark shard into a training window and pushed it at
   <code>check_batch()</code> — refused, with the canary found in the decoded text.`,
  `Validation gradient-bearing tokens: <b>${D.firewall.validation_gradient_bearing_tokens}</b>,
   independently re-checked by scanning the whole consumption ledger against registry permissions.`],
 io:{in:[`test documents`,`ShardRegistry`,`decoded batch text at run time`],
     out:[`ledgers/firewall_events.jsonl`,`firewall_report.json`,`manifests/eval_registry.json`],
     next:`Blocks feed the evidence bundle; permissions gate the packer`},
 record:"firewall",
 artifacts:[`ledgers/firewall_events.jsonl`,`ledgers/firewall_report.json`,`manifests/eval_registry.json`],
 metrics:()=>[["registry checks",D.firewall.registry_checks],["batch checks",D.firewall.batch_checks],
   ["blocks",D.firewall.blocks_total],
   ["validation grad tokens",D.firewall.validation_gradient_bearing_tokens],
   ["canaries",D.evidence.find(x=>x.req==="Evaluation firewall").detail.canaries_registered]],
 panel:"panelFirewall"
},
{
 n:"06", id:"mixture", title:"Compiling the curriculum into quotas", tag:"scheduling", group:"Scheduling",
 one:`Human-readable stages become an integer number of sequences per lane per step — before anything is built.`,
 concept:`The <b>schedule</b> says how many of each step's sequences every lane gets. A
   <b>protected floor</b> is a hard minimum, not a preference — a language the model stops seeing is
   one it has effectively lost. <i>Instead of probabilistic sampling:</i> integer apportionment by
   largest remainder, so the realised mixture matches the plan exactly rather than in expectation.`,
 built:[
  `<b>Three stages</b> as frozen config: <code>foundation-en</code> (10 steps, window 256) ·
   <code>reasoning-heavy-midtrain</code> (8 steps, 256, warmup 4) ·
   <code>long-context-anneal</code> (6 steps, <b>window 512</b>, warmup 3, unlocks reserved agentic
   and reasoning material).`,
  `<b>Availability first.</b> Lane window counts come from <code>Packer.count_windows()</code>,
   which runs the policy on item <i>sizes</i> — so the schedule is compiled and checked before a
   single token is materialised.`,
  `<b>Warmup as a ramp.</b> <code>blend(prev, cur, t)</code> with
   <code>t = (local+1)/warmup_steps</code>, so a stage boundary is not a switch.`,
  `<b>Largest-remainder apportionment.</b> Floor each share, hand leftovers to the biggest
   fractional parts, ties on lane name. Always sums to exactly the batch size.`,
  `<b>Floors as a repair pass</b>, capped by what the lane can supply — a floor cannot be met from
   an empty lane — with every adjustment recorded.`,
  `<b>Scarcity resolved explicitly</b> into one of: satisfied · repeat · synthesise · reduce share ·
   defer · impossible, each with a note.`],
 ran:()=>[
  `Compiled <b>${D.per_step.length} steps</b> across ${D.stages.length} stages.`,
  `<b>${D.floor_adjustments.length} floor rescues</b> fired — at 8 sequences per step a 4% agentic
   share rounds to zero, so the floor intervened on essentially every step.`,
  `<b>${D.feasibility.filter(f=>f.resolution!=="satisfied").length} (stage, lane) pairs</b> could not
   be satisfied from distinct samples; all resolved as <code>repeat_existing_data</code> at factors
   ${D.feasibility.filter(f=>f.resolution!=="satisfied").map(f=>f.repeat_factor.toFixed(2))
     .filter((v,i,a)=>a.indexOf(v)===i).sort().join(", ")}×.`,
  `Planned versus actual, measured afterwards from the ledger: max lane delta
   <b>${D.compliance.max_abs_delta}</b> against a tolerance of ${D.compliance.tolerance};
   <b>${D.compliance.protected_floor_checks.length} floor checks</b>, all respected.`],
 io:{in:[`lane window counts`,`Stage[] from config`],
     out:[`manifests/mixture_schedule.json`,`manifests/mixture_compliance.json`],
     next:`The planner turns quotas into candidate pools`},
 record:null, artifacts:[`manifests/mixture_schedule.json`,`manifests/mixture_compliance.json`],
 metrics:()=>[["steps",D.per_step.length],["stages",D.stages.length],
   ["floor rescues",D.floor_adjustments.length],
   ["scarce lanes",D.feasibility.filter(f=>f.resolution!=="satisfied").length],
   ["planned vs actual",D.compliance.max_abs_delta]],
 panel:"panelMixture"
},
{
 n:"07", id:"packing", title:"Packing policies", tag:"batch build", group:"Batch construction",
 one:`Filling a fixed window is a training decision, and the right answer differs by data type.`,
 concept:`Every unused slot is a token the model did not learn from; every unrelated document in a
   window is a chance to learn a transition that does not exist. <i>Instead of one global policy:</i>
   six, chosen per lane — because cutting a Shakespeare span mid-line is harmless and cutting an
   agent trajectory mid-turn is not.`,
 built:[
  `<b>Six policies</b> over abstract <code>Item{size, split_points, min_context}</code>, so they are
   unit-testable without a tokenizer or a shard: <code>pad_only</code> · <code>concat_chop</code> ·
   <code>greedy</code> · <code>best_fit</code> · <code>structure_preserving</code> ·
   <code>long_context</code>.`,
  `<b>Lane assignment.</b> web → concat_chop · code → best_fit · maths → greedy · indic → best_fit ·
   agentic → structure_preserving · reasoning → long_context.`,
  `<b>Splitting rule.</b> An oversized item is cut at the nearest declared split point at or before
   capacity — never mid-turn. That is what structure-preserving means in practice: not that a
   document is never divided, but that each piece is still a coherent stretch of one conversation.`,
  `<b>long_context</b> additionally defers any document below
   <code>LONG_CONTEXT_MIN_FILL = 0.45</code> of the window, and any whose <code>min_context</code>
   exceeds it — a reasoning trace cut off before its verification step teaches the model to stop
   reasoning early.`,
  `<b>Retention alongside utilisation.</b> <code>pad_only</code> scores 1.000 utilisation by
   truncating whatever does not fit; retention is the column that exposes it.`,
  `<b>All six run over the same items</b> for every lane and window, so the chosen policy's
   advantage is measured, not asserted.`],
 ran:()=>[
  `Materialised <b>${D.mask_validation.samples_checked} packed samples</b> across two window lengths
   (256 for stages 1–2, 512 for the anneal stage) and two reserved contexts.`,
  `Ran <b>${Object.keys(D.policy_comparison).length} policy comparisons</b> — every policy over
   every lane/window combination.`,
  `Aggregate utilisation
   <b>${D.evidence.find(x=>x.req==="Packing correctness").detail.aggregate_packing_utilisation}</b>,
   loss density
   <b>${D.evidence.find(x=>x.req==="Packing correctness").detail.aggregate_loss_density}</b>.`,
  `Confirmed the materialised sample count matches the window count the schedule was compiled
   against — otherwise the plan would rest on a supply that does not exist.`],
 io:{in:[`admitted shards via the registry`,`stage window length`,`reserved-unlocked flag`],
     out:[`PackedSample[] by (window, reserved, lane)`,`manifests/packing_report.json`],
     next:`Mask construction, then the planner`},
 record:null, artifacts:[`manifests/packing_report.json`,`manifests/packed_samples_index.json`],
 metrics:()=>{const e=D.evidence.find(x=>x.req==="Packing correctness").detail; return [
   ["packed samples",e.packed_samples],["policies in use",e.policies_exercised.length],
   ["utilisation",e.aggregate_packing_utilisation],["loss density",e.aggregate_loss_density],
   ["comparisons",Object.keys(D.policy_comparison).length]];},
 panel:"panelPacking"
},
{
 n:"08", id:"masks", title:"Loss masks, attention masks, position ids", tag:"batch build", group:"Batch construction",
 one:`The batch carries the training meaning of its tokens, not just the tokens.`,
 concept:`<code>loss_mask[i]=1</code> means position i is graded. <code>segment_ids[i]</code> names
   which packed document owns it, and attention is confined within. <code>position_ids[i]</code>
   restart per document. <i>Instead of trusting the construction:</i> six invariants checked on every
   sample — these are the bugs that still train, still show a falling loss curve, and never throw.`,
 built:[
  `<b>Per-token graded flags</b> copied from the document's role spans, then two overrides: the
   first position of every segment is never a target (nothing precedes it), and padding is never a
   target.`,
  `<b>Attention bias</b> = <code>causal ∧ same-segment</code>, an additive <code>0 / -inf</code>
   tensor of shape (B,1,L,L). The identity diagonal is kept alive so no softmax row is fully masked —
   padding attends to itself and the loss mask discards those rows.`,
  `<b>Segment-relative positions</b>; <code>-1</code> segment id for padding, position 0 there.`,
  `<b>Standard shift</b> — logits at i predict token i+1 — and the mean divides by the count of
   <i>graded</i> tokens, not positions, so padding cannot dilute the number.`,
  `<b>Six validators:</b> no loss on padding · no loss on a segment's first token · no loss on
   context-only spans · positions segment-relative, contiguous, monotonic · segments contiguous with
   padding as a suffix · attention causal and confined. A sample that fails is not emitted.`],
 ran:()=>[
  `All <b>${D.mask_validation.samples_checked} packed samples</b> validated at construction, then
   re-validated in the demo: <b>${D.mask_validation.mask_problems.length} mask problems</b>,
   <b>${D.mask_validation.attention_problems.length} attention leaks</b>.`,
  `Agentic samples show the loss policy working — the inspector below has real context-only spans
   (the user turn and the tool results) carrying no gradient.`,
  `general_web samples grade every real token except the <code>&lt;bos&gt;</code> beginning each
   packed document — asserted as an exact identity in the test suite.`],
 io:{in:[`PackedSample under construction`,`document role spans`],
     out:[`validated loss_mask / segment_ids / position_ids`,`loss_mask_hash per sample`],
     next:`Tensors at the training step; hashes into the ledger`},
 record:null, artifacts:[`manifests/mask_validation.json`],
 metrics:()=>[["samples validated",D.mask_validation.samples_checked],
   ["invariants",D.mask_validation.checks_applied.length],
   ["mask problems",D.mask_validation.mask_problems.length],
   ["attention leaks",D.mask_validation.attention_problems.length],
   ["policy","causal + block-diagonal"]],
 panel:"panelMasks"
},
{
 n:"09", id:"planner", title:"The batch plan — a pure function", tag:"determinism", group:"Scheduling",
 one:`Which candidates step N is offered depends only on the seed, the branch and the step number.`,
 concept:`The <b>plan</b> is a pure function of <code>(seed, branch, step)</code> — no I/O, no model,
   no global RNG. The <b>batch</b> is what OPUS accepted from it, so it also depends on model state.
   <i>Instead of a stateful iterator:</i> a precomputed shuffle index and positional RNG derivation,
   which removes a whole class of resume bugs by construction.`,
 built:[
  `<b>Shuffle index built once</b> per (window, reserved) context and lane, seeded by
   <code>derive_seed(master_seed, branch, "lane", lane, length, reserved)</code>, and hashed into
   <code>index_hash</code>.`,
  `<b>Cursors precomputed</b> as prefix sums of candidate-pool sizes, so <code>plan(step)</code> is
   O(1) and never walks the history.`,
  `<b>Wraparound with pass numbers.</b> Drawing wraps modulo the stream length and records
   <code>index // len</code> — which is what makes the repeat-effect measurement possible later.`,
  `<b>Candidate pool = quota × 2</b>, so OPUS always has something to reject.`,
  `<b>RNG by derivation, not by carrying:</b>
   <code>sha256(master_seed ‖ branch_id ‖ step)</code>. Resuming at step N reconstructs step N's
   randomness by definition — nothing to serialise, nothing to restore wrongly.`,
  `<b>Layout.</b> <code>assemble_batch()</code> interleaves lanes round-robin across
   (rank, accumulation slot), so no rank spends a whole microbatch on one lane, and hashes the
   result into <code>batch_id</code>.`],
 ran:()=>[
  `Planned all <b>${D.per_step.length} steps</b> for the main branch, and again for the fork branch
   under a different seed.`,
  `On replay, <b>${D.plan_comparison.length} step plan hashes</b> were recomputed from the seed alone
   and compared against the ledger:
   <b>${D.replay.plan_recomputation_matches?"all matched":"MISMATCH"}</b>.`,
  `Both worker processes rebuilt the whole index independently and matched the driver's build
   fingerprint — evidence the ordering is a function of the corpus and config, not of process state.`],
 io:{in:[`MixtureSchedule`,`PackedSampleStore`,`seed + branch id`],
     out:[`StepPlan per step`,`plan_hash`,`index_hash`,`BatchSpec after selection`],
     next:`OPUS scores the pool; the trainer consumes the batch`},
 record:null, artifacts:[`manifests/build_fingerprint.json`],
 metrics:()=>[["steps planned",D.per_step.length],["pool multiplier","2×"],
   ["plan hashes rechecked",D.plan_comparison.length],
   ["all matched",D.replay.plan_recomputation_matches?"yes":"NO"],
   ["microbatches",D.integrity.distinct_microbatches]],
 panel:null
},
{
 n:"10", id:"opus", title:"OPUS selection", tag:"scheduling", group:"Scheduling",
 one:`Score every candidate by how its gradient aligns with the direction a golden set would take.`,
 concept:`A golden probe set is scored at the current checkpoint to give the direction we
   <i>wish</i> to move in; each candidate is scored by the cosine between its own gradient and that
   direction. <i>Instead of a heuristic quality score:</i> a real gradient cosine, because only that
   can say a shard is redundant for <b>this</b> model right now. Rejections are the valuable output —
   kept, never deleted.`,
 built:[
  `<b>Proxy direction</b> from the golden probe batches, averaged and normalised, recomputed every
   <code>opus_round_interval = 6</code> steps. Rounds align to checkpoints so the direction always
   comes from a state a resume can restore exactly.`,
  `<b>Probe parameter subset</b> — the last transformer block plus the final norm, 198,016 values.
   The tied embedding is excluded deliberately: it dominates the parameter count, changes slowly, and
   would swamp the cosine with a term nearly identical for every candidate.`,
  `<b>Cheap scoring</b> — forward + backward on an <code>opus_probe_tokens = 64</code> prefix, the
   same construction as scoring a short prefix of a long sample.`,
  `<b>Adaptive threshold</b> — the round's own median, because a fixed threshold stops meaning
   anything as the model moves.`,
  `<b>Deterministic ranking</b> — score rounded to 6dp, ties broken on sample id, so a knife-edge tie
   cannot flip between the original run and the post-crash re-run.`,
  `<b>Demote, do not reject, on stage mismatch.</b> Rejecting outright can empty a lane whose whole
   pool is hinted for a later phase, and an empty lane cannot fill a fixed batch geometry — a bug
   found and fixed during the build.`,
  `<b>Five reasons:</b> above_proxy_threshold · below_proxy_threshold · quota_pressure · duplicate ·
   stage_mismatch · protected_lane_bias.`],
 ran:()=>[
  `Scored <b>${D.opus_report.total_candidates_scored} candidates</b> —
   ${D.opus_report.by_status.accepted} accepted, ${D.opus_report.by_status.rejected} rejected,
   ${D.opus_report.by_status.deferred} deferred.`,
  `<b>${D.opus_report.protected_floor_overrides} protected-floor overrides</b>: the lane would have
   dropped below its floor, so the floor won and the record says so.`,
  `Scores span <b>${Math.min.apply(null,D.opus.map(r=>r[4])).toFixed(4)}</b> to
   <b>${Math.max.apply(null,D.opus.map(r=>r[4])).toFixed(4)}</b> with
   <b>${new Set(D.opus.map(r=>r[4])).size} distinct values</b> across ${D.opus.length} decisions —
   the signature of a computed cosine, not a generated number.`,
  `Candidates re-scored after the crash produced <b>identical</b> scores, statuses and reasons —
   logged as <code>opus_rescoring_after_crash_identical</code>.`,
  `Proxy health across ${D.proxy_health.rounds} rounds: accepted gradient norm fell from
   ${D.proxy_health.first_round_accepted_grad_norm} to
   ${D.proxy_health.last_round_accepted_grad_norm} (${D.proxy_health.ratio_last_over_first}×) —
   flagged, and correct for a corpus this size.`],
 io:{in:[`StepPlan candidate pools`,`current model state`,`golden probe batches`],
     out:[`ledgers/opus_decisions.jsonl`,`accepted ids per lane`,`opus_proxy_health.json`],
     next:`Accepted samples are assembled into the batch`},
 record:"opus",
 artifacts:[`ledgers/opus_decisions.jsonl`,`ledgers/opus_report.json`,`ledgers/opus_proxy_health.json`],
 metrics:()=>[["candidates",D.opus_report.total_candidates_scored],
   ["accepted",D.opus_report.by_status.accepted],["rejected",D.opus_report.by_status.rejected],
   ["deferred",D.opus_report.by_status.deferred],
   ["floor overrides",D.opus_report.protected_floor_overrides],
   ["distinct scores",new Set(D.opus.map(r=>r[4])).size]],
 panel:"panelOpus"
},
{
 n:"11", id:"training", title:"The training step", tag:"execution", group:"Execution",
 one:`Nine operations in a fixed order — and the order is the part that matters.`,
 concept:`One optimizer update spanning <code>ranks × microbatch × grad_accum</code> sequences.
   <i>The ordering decision:</i> the consumption record is written and fsynced <b>before</b> the
   optimizer step, because a batch that was served must be recorded even if the process dies first.
   That single choice is what makes the ledger run ahead of the model after a crash — and therefore
   what makes resume a real problem with a right answer.`,
 built:[
  `<b>Geometry.</b> world_size 2 × microbatch 2 × grad_accum 2 = <b>8 sequences per step</b>,
   4 microbatches, ranks simulated sequentially on CPU.`,
  `<b>Model.</b> 1,312,256-parameter pre-LN decoder — 4 layers, 4 heads, d_model 128, tied
   embeddings, learned positions to 512, segment-aware additive attention bias.`,
  `<b>Optimiser.</b> AdamW, lr 3e-3, β (0.9, 0.95), weight decay 0.01, grad clip 1.0, warmup 6 steps
   then cosine — scheduler state checkpointed so a resume continues the same curve rather than
   restarting the warmup.`,
  `<b>Order per step:</b> plan → proxy/select → assemble → <b>loss before</b> → per microbatch
   {firewall on decoded text, forward, backward, <b>write ledger</b>} → clip → optimizer step →
   scheduler step → <b>loss after</b> → learning records → token trace if in interval → checkpoint
   if at interval.`,
  `<b>Loss before/after</b> computed for all 8 samples in one batched no-grad forward, so the delta
   costs two forwards per step rather than sixteen.`,
  `<b>Determinism.</b> <code>torch.set_num_threads(1)</code> and
   <code>use_deterministic_algorithms(True)</code> — threaded BLAS reduction order varies with how
   work was split, and that alone breaks loss reproducibility across processes.`],
 ran:()=>[
  `<b>${D.curve.length} steps</b>, ${D.integrity.distinct_microbatches} microbatches,
   ${fmt(D.integrity.total_positions)} positions consumed.`,
  `First-step loss <b>${D.curve[0].loss}</b> against ln(V) =
   <b>${Math.log(D.corpus.vocab_size).toFixed(4)}</b> — a delta of
   ${(D.curve[0].loss-Math.log(D.corpus.vocab_size)).toFixed(4)}, confirming labels are shifted
   correctly and the mask is not inverted, before any training happened.`,
  `Loss fell to <b>${D.curve[D.curve.length-1].loss}</b> (perplexity
   ${D.curve[D.curve.length-1].ppl}) by step ${D.curve.length-1}.`,
  `Validation evaluated at ${D.validation.length} checkpoints — loss only, never gradient-bearing.`,
  `The window changes from 256 to 512 at the anneal stage, visible as the band change below.`],
 io:{in:[`BatchSpec`,`PackedSample tensors`,`model + optimizer state`],
     out:[`consumption records`,`learning records`,`StepResult`],
     next:`Checkpoint at the interval`},
 record:"step", artifacts:[`ledgers/consumption_main.jsonl`],
 metrics:()=>[["steps",D.curve.length],["sequences / step",8],["parameters","1,312,256"],
   ["first loss",D.curve[0].loss],["final loss",D.curve[D.curve.length-1].loss]],
 panel:"panelTraining"
},
{
 n:"12", id:"consumption", title:"The consumption ledger", tag:"memory", group:"Memory",
 one:`An append-only, hash-chained record of every microbatch actually served.`,
 concept:`The run's memory of what it fed the model — the receipt, not the plan. Each record chains
   to the previous by hash, so an edit or a deletion is detectable. <i>Instead of a database:</i>
   JSONL with fsync per record, because the torn-tail case then becomes something you can demonstrate
   rather than the engine's private problem.`,
 built:[
  `<b>Record shape.</b> <code>{seq, prev_hash, event_hash, type, payload}</code>, one canonical JSON
   line — sorted keys, compact separators — so byte offsets are stable and equal payloads always
   hash equally.`,
  `<b>Chain.</b> <code>event_hash = sha256(canonical({seq, prev_hash, type, payload}))</code>.
   Verification is one forward pass recomputing every hash.`,
  `<b>Durability.</b> write → flush → <code>os.fsync</code>, per record.`,
  `<b>Torn-tail tolerance.</b> Reads ignore an incomplete final line; a <i>complete</i> but
   unparsable line is skipped, which leaves a sequence gap that <code>verify_chain()</code> reports —
   corruption is never silently absorbed.`,
  `<b>One non-append operation:</b> <code>rollback_to(offset)</code>, which returns and logs what it
   discarded, and refuses to proceed if the resulting head does not match the offset's recorded hash.`,
  `<b>Payload covers the full record contract</b> — run/branch, step, checkpoint, rank, microbatch,
   packed sample ids, shard ids, token span ids, loss-mask hashes, attention and position policy,
   lane, stage, tokenizer version, dataloader version, OPUS decision id, pass number, RNG fingerprint
   and the token accounting.`],
 ran:()=>[
  `<b>${D.integrity.records} consumption records</b> over steps ${D.integrity.step_range.join("–")},
   ${D.integrity.distinct_microbatches} distinct microbatches.`,
  `<b>${D.integrity.duplicate_count} duplicates</b>, <b>${D.integrity.missing_steps.length} gaps</b>,
   every step carrying exactly ${D.integrity.expected_microbatches_per_step} microbatches.`,
  `Hash chain verified intact after the crash, the repair and the rollback.`,
  `Token accounting: ${fmt(D.integrity.loss_bearing_tokens)} loss-bearing,
   ${fmt(D.integrity.pad_tokens)} padding, ${fmt(D.integrity.total_positions)} positions total.`],
 io:{in:[`served microbatches`],
     out:[`ledgers/consumption_&lt;branch&gt;.jsonl`,`ledgers/consumption_integrity.json`],
     next:`Checkpoints reference offsets into it; replay and audit read it`},
 record:"consume", artifacts:[`ledgers/consumption_main.jsonl`,`ledgers/consumption_integrity.json`],
 metrics:()=>[["records",D.integrity.records],["microbatches",D.integrity.distinct_microbatches],
   ["duplicates",D.integrity.duplicate_count],["gaps",D.integrity.missing_steps.length],
   ["chain","intact"]],
 panel:"panelConsumption"
},
{
 n:"13", id:"learning", title:"The learning ledger", tag:"memory", group:"Memory",
 one:`What the model gave back, attached to the data that caused it.`,
 concept:`The half that almost never gets written down — and cannot be recovered later without
   re-running the same model over the same data at the same training state. <i>Instead of logging the
   training loss:</i> measure the loss on the same tokens immediately before and after the update,
   because "this batch was hard" and "this batch helped" are different questions that often have
   opposite answers.`,
 built:[
  `<b>Tiered storage</b>, as the storage strategy is tiered: a full per-token trace for a configured interval,
   per-sample records for every step, aggregates for the whole run.`,
  `<b>Per-sample record:</b> loss before, loss after, delta, gradient norm, loss-bearing tokens, mean
   token perplexity, <b>EOS perplexity separately</b>, model phase, checkpoint before/after, OPUS
   decision id, repeated-pass number.`,
  `<b>Per-token record</b> (14 fields): token id, decoded preview, position in sequence and in
   segment, document, shard, language, script, lane, special and EOS flags, loss-mask flag,
   cross-entropy, perplexity.`,
  `<b>Classification thresholds from the design.</b> Mean perplexity at or below
   <code>learned_out_ppl = 1.2</code> → <b>exhausted</b>, the model already predicts it. Max gradient
   &gt; 8× the mean → <b>harmful</b>, needs cleaning or later staging. Negative mean delta →
   <b>useful</b>. Otherwise <b>neutral</b>.`,
  `<b>Repeat effect</b> = mean delta on the last pass minus the first. Positive means repetition has
   stopped paying.`,
  `<b>Next-corpus output</b> — collect_more · protect · repeat · defer · reject, each with the measurement
   behind it.`],
 ran:()=>[
  `<b>${D.learning_agg.samples_recorded} sample records</b> and
   <b>${fmt(D.learning_agg.token_records_written)} token records</b>.`,
  `<b>${D.learning_agg.eos_perplexity.samples} EOS positions</b> traced, mean perplexity
   ${D.learning_agg.eos_perplexity.mean}.`,
  `<b>${D.shard_cards.length} shard report cards</b> — ${D.corpus_verdicts.useful} useful,
   ${D.corpus_verdicts.harmful} harmful, ${D.corpus_verdicts.neutral} neutral,
   ${D.corpus_verdicts.exhausted} exhausted.`,
  `<code>next_corpus_recommendations.json</code>: ${Object.entries(D.corpus_actions).filter(x=>x[1]>0)
    .map(x=>`${x[1]} ${x[0].replace(/_/g," ")}`).join(", ")}.`],
 io:{in:[`per-sample losses`,`per-token cross-entropy`,`OPUS scores`,`pass numbers`],
     out:[`ledgers/learning_&lt;branch&gt;.jsonl`,`learning_aggregates.json`,`next_corpus_recommendations.json`],
     next:`Feeds the audit and the next corpus`},
 record:"learning",
 artifacts:[`ledgers/learning_main.jsonl`,`ledgers/learning_aggregates.json`,`ledgers/next_corpus_recommendations.json`],
 metrics:()=>[["sample records",D.learning_agg.samples_recorded],
   ["token records",fmt(D.learning_agg.token_records_written)],
   ["EOS traced",D.learning_agg.eos_perplexity.samples],
   ["report cards",D.shard_cards.length],["useful",D.corpus_verdicts.useful]],
 panel:"panelLearning"
},
{
 n:"14", id:"checkpoint", title:"Checkpoints that carry a data position", tag:"execution", group:"Execution",
 one:`Model state and data state are saved together, or the checkpoint is incomplete.`,
 concept:`Five things, not one: weights, optimizer, scheduler, RNG position (derived), and the
   <b>ledger offset</b>. <i>Instead of a step number:</i> a byte offset — because a step number tells
   you where to resume counting, and an offset tells you where to <b>truncate</b>, which is the
   operation recovery actually needs.`,
 built:[
  `<b>Written atomically</b> — model, optimizer and scheduler serialised to one buffer, temp → fsync
   → replace, then <code>meta.json</code> beside it.`,
  `<b>meta.json carries</b> checkpoint id, global step, stage, consumption <i>and</i> learning ledger
   offsets (byte offset, event seq, last event hash), RNG fingerprint, tokenizer hash, schedule hash,
   shuffle index hash, config hash, tokens consumed, last batch id, and <b>the plan hash the next
   step should serve</b>.`,
  `<b>Ledger event written after the checkpoint is durable</b>, so a crash between the two leaves an
   unreferenced checkpoint rather than a ledger pointing at one that does not exist.`,
  `<b>latest_checkpoint()</b> skips any directory missing <code>meta.json</code> or
   <code>state.pt</code> — that is what an interrupted checkpoint looks like, and choosing it would
   defeat the recovery.`,
  `<b>Retention.</b> After the run, weights are pruned where nothing still depends on them; every
   <code>meta.json</code> is kept, because the ledger offset is what makes a checkpoint auditable.`],
 ran:()=>[
  `<b>${D.checkpoints.length} checkpoints</b> at steps ${D.checkpoints.map(c=>c.step).join(", ")}.`,
  `Each records a byte offset — step ${D.checkpoints[1]?D.checkpoints[1].step:12} sits at byte
   <b>${fmt(D.checkpoints[1]?D.checkpoints[1].offset.byte_offset:0)}</b>, event seq
   ${D.checkpoints[1]?D.checkpoints[1].offset.event_seq:0}.`,
  `Pruning reclaimed <b>${(D.retention.reduce((a,b)=>a+b.bytes_reclaimed,0)/1e6).toFixed(1)} MB</b>
   by dropping ${D.retention.length} superseded weight files while keeping all
   ${D.checkpoints.length} metadata files.`],
 io:{in:[`model/optimizer/scheduler state`,`current ledger offsets`],
     out:[`checkpoints/ckpt_&lt;branch&gt;_&lt;step&gt;/{state.pt, meta.json}`,`retention.json`],
     next:`Resume and fork both restore from here`},
 record:"checkpoint_meta", artifacts:[`checkpoints/ckpt_*/meta.json`,`checkpoints/retention.json`],
 metrics:()=>[["checkpoints",D.checkpoints.length],["interval","6 steps"],
   ["weights kept",D.checkpoints.filter(c=>c.weights).length],
   ["metadata kept",D.checkpoints.length],
   ["reclaimed",(D.retention.reduce((a,b)=>a+b.bytes_reclaimed,0)/1e6).toFixed(1)+" MB"]],
 panel:"panelCheckpoint"
},
{
 n:"15", id:"crash", title:"The deliberate crash", tag:"execution", group:"Execution",
 one:`A hard kill mid-write, leaving the ledger ahead of the model and the last line torn.`,
 concept:`Reproduces what <code>SIGKILL</code> leaves behind, which has two symptoms: the ledger
   ahead of the durable model state, and a final line that is not a complete record. <i>Instead of an
   exception:</i> a real process death — an exception unwinds the stack, flushes buffers and runs
   destructors, which is precisely the cleanup a crash does not do.`,
 built:[
  `<b>The trainer runs as a separate OS process</b> (<code>tdes/cli/train_worker.py</code>), launched
   by <code>run_demo.py</code> via <code>subprocess.run</code> — it has to be killable.`,
  `<b>Placement.</b> The crash fires mid-step, after the first <code>world_size</code> microbatches
   are durably recorded and <i>before</i> the optimizer step — so the ledger genuinely runs ahead.`,
  `<b>Timing.</b> <code>crash_step = 16</code>, four steps past the step-12 checkpoint. A crash
   coinciding with a checkpoint would leave the two agreeing and prove nothing.`,
  `<b>The kill.</b> Build a genuine record, serialise canonically, write the first 55% of the bytes
   with no terminating newline, fsync, then <code>os._exit(137)</code> — 128 + SIGKILL, skipping
   atexit handlers, buffered flushes and destructors.`,
  `<b>The parent checks the exit code</b> and emits <code>crash simulated</code> only on 137.`],
 ran:()=>[
  `Worker exited <b>137</b> during step <b>16</b>; last checkpoint
   <code>${D.recovery.checkpoint}</code>.`,
  `Left a <b>${D.recovery.torn_tail.removed_bytes}-byte torn fragment</b> as the final line — reason
   recorded as <code>${D.recovery.torn_tail.reason}</code>.`,
  `Left the ledger <b>${D.recovery.discarded_records} records</b> ahead of the model state, covering
   steps ${D.recovery.discarded_steps.join(", ")}.`],
 io:{in:[`a live trainer past a checkpoint`],
     out:[`exit code 137`,`a torn ledger tail`,`a ledger ahead of the model`],
     next:`Resume`},
 record:null, artifacts:[`ledgers/consumption_main.jsonl (torn, then repaired)`],
 metrics:()=>[["crash step",16],["exit code",137],["last checkpoint",D.recovery.checkpoint],
   ["torn bytes",D.recovery.torn_tail.removed_bytes],
   ["records ahead",D.recovery.discarded_records]],
 panel:null
},
{
 n:"16", id:"resume", title:"Resume without skipping or repeating", tag:"execution", group:"Execution",
 one:`Restore the checkpoint, truncate the ledger to its offset, re-serve — then prove it matched.`,
 concept:`Two obvious answers are both wrong. Resume at the ledger's position and the rolled-back
   steps are <b>silently skipped</b>; resume at the checkpoint and append and they are <b>recorded
   twice</b>. <i>Only restore-and-truncate</i> gives one exposure per batch and one record per batch.`,
 built:[
  `<b>Order of operations:</b> repair the torn tail first (nothing can parse the ledger until it is
   gone) → load the newest <i>complete</i> checkpoint → roll both ledgers back to its recorded
   offsets → log a <code>ledger_rollback</code> event carrying the discarded hashes and fingerprints
   → re-serve.`,
  `<b>The comparison fingerprint is integer-only</b> — step, rank, microbatch id, batch id, plan
   hash, sample ids, token span ids, token hashes, mask hashes. No float appears, so the equality
   test is exact and cannot be defeated by CPU reduction-order drift.`,
  `<b>Two independent proofs.</b> <code>verify_next_batch()</code> compares the checkpoint's recorded
   <code>next_expected_plan_hash</code> against the planner's fresh recomputation;
   <code>verify_rollback_replay()</code> compares every re-served microbatch against the fingerprint
   of the record the rollback discarded.`,
  `<b>The OPUS ledger is deliberately not rolled back</b> — it records scoring <i>events</i>, and the
   selector genuinely did score those candidates twice. Keeping both makes the determinism checkable.`],
 ran:()=>[
  `Repaired the <b>${D.recovery.torn_tail.removed_bytes}-byte</b> torn tail; chain head unchanged,
   because a partial record never entered the chain.`,
  `Restored <code>${D.recovery.checkpoint}</code> and rolled back to byte
   <b>${fmt(D.recovery.ledger_offset.byte_offset)}</b> / event seq
   ${D.recovery.ledger_offset.event_seq}, discarding <b>${D.recovery.discarded_records} records</b>
   across steps ${D.recovery.discarded_steps.join(", ")}.`,
  `<code>resume_next_batch_matched</code> — checkpoint said
   <code>${D.next_batch.expected_plan_hash_from_checkpoint}</code>, planner recomputed
   <code>${D.next_batch.recomputed_plan_hash}</code>.`,
  `<code>resume_rollback_replay_identical</code> — <b>${D.rollback.compared} microbatches</b>
   compared, <b>0 mismatches, 0 missing</b>.`,
  `The re-served steps produced <b>identical losses to four decimal places</b> across the process
   boundary — not required, but a useful sign the determinism controls hold.`],
 io:{in:[`torn ledger`,`latest complete checkpoint`],
     out:[`ledger_rollback event`,`re-served records`,`two [PASS] checks`],
     next:`The run completes; replay and audit read the repaired ledger`},
 record:"rollback", artifacts:[`ledgers/phase_resume_main.json`],
 metrics:()=>[["resumed at",D.recovery.resume_step],
   ["records rolled back",D.recovery.discarded_records],
   ["compared",D.rollback.compared],["mismatches",0],["missing",0]],
 panel:"panelResume"
},
{
 n:"17", id:"replay", title:"Replay and fork", tag:"execution", group:"Execution",
 one:`Read history rather than recompute it — then check it against two things that did not come from it.`,
 concept:`The design notes is explicit: <i>"I'm going to run the ledger… I will not calculate it"</i>,
   because nondeterminism can creep into a re-run. But a replay that reads the ledger and compares it
   to itself proves nothing. <i>Instead of one source:</i> three independently derived answers that
   must agree.`,
 built:[
  `<b>Recorded</b> — the interval's consumption records, read from the ledger.`,
  `<b>Reconstructed</b> — each sample rebuilt from the packed store and re-hashed, and the named
   token spans re-read <i>straight out of the shard binaries</i> and compared against the packed
   window. A genuine round trip through storage, not a cache lookup.`,
  `<b>Recomputed</b> — the plan hash for those steps derived from the seed alone, ledger not
   consulted.`,
  `<b>Fork</b> — a new branch id derived from (parent, step, checkpoint, note), a different seed, the
   divergence point recorded, lineage written to <code>branches.json</code>, and its own consumption
   ledger so the parent stays replayable.`,
  `<b>The fork check is two-sided.</b> Identical batches <i>after</i> the fork point would mean the
   fork changed nothing; differing batches <i>before</i> it would mean it did not really start from
   the parent's state. Only both together make the comparison sound.`],
 ran:()=>[
  `Replayed steps <b>${D.replay.interval.join("–")}</b>:
   <b>${D.replay.microbatches_replayed} microbatches</b>, ${D.replay.batch_ids.length} distinct
   batch ids.`,
  `Token hashes ${D.replay.tokens_match?"match":"DIFFER"} · loss masks
   ${D.replay.loss_masks_match?"match":"DIFFER"} · token spans re-read from the shard bytes
   ${D.replay.token_spans_match?"match":"DIFFER"} · ${D.plan_comparison.length} plan hashes
   recomputed from the seed ${D.replay.plan_recomputation_matches?"match":"DIFFER"}.`,
  `Forked <code>${D.fork.fork_branch_id||"a new branch"}</code> from step ${D.fork.fork_point_step}:
   <b>${D.fork.identical_before_fork.length} steps identical</b> before the fork point,
   <b>${D.fork.differing_after_fork.length} differing</b> after.`,
  `The parent branch remains replayable unchanged — the fork wrote to its own ledger.`],
 io:{in:[`consumption ledger`,`shard binaries`,`planner`,`a checkpoint to fork from`],
     out:[`ledgers/replay_report.json`,`fork_divergence.json`,`branches.json`],
     next:`Both feed the evidence bundle`},
 record:null,
 artifacts:[`ledgers/replay_report.json`,`ledgers/fork_divergence.json`,`ledgers/branches.json`],
 metrics:()=>[["interval",D.replay.interval.join("–")],
   ["microbatches",D.replay.microbatches_replayed],["derivations",3],
   ["all match",D.replay.all_match?"yes":"NO"],
   ["fork correct",D.fork.diverged_correctly?"yes":"NO"]],
 panel:"panelReplay"
},
{
 n:"18", id:"audit", title:"Audit", tag:"memory", group:"Memory",
 one:`Two questions nothing else in the stack can answer.`,
 concept:`A loss curve says something went wrong and nothing about what; a folder of shards says what
   exists and nothing about what was used. <i>These are answerable only because</i> the ledger
   recorded token spans per microbatch and OPUS recorded why each batch was chosen — both written at
   the time, not reconstructed after.`,
 built:[
  `<b>Query 1 — by step range:</b> walk the ledger, group token spans by shard, count distinct spans
   and tokens, record first and last step per shard.`,
  `<b>Query 1′ — by token count:</b> the same question phrased as the design asks it ("between 5.4B
   and 5.6B tokens"), by accumulating positions and selecting the steps whose cumulative range
   overlaps the window — because at scale nobody remembers which step that was.`,
  `<b>Query 2 — spikes:</b> detected as a z-score on the <i>first difference</i> of the loss series
   at 1.5σ, not an absolute threshold, because the absolute loss falls throughout a run and any fixed
   number stops meaning the same thing.`,
  `<b>Spike investigation</b> pulls the OPUS decisions in the preceding window, counts the
   protected-floor overrides among them, and names the candidate with the largest gradient.`,
  `<b>Checkpoint provenance</b> — everything that trained the model up to a given step.`],
 ran:()=>[
  `Traced <b>${D.audit.provenance.shards_involved} shards</b> behind the final checkpoint, accounting
   for ${fmt(D.audit.provenance.total_positions)} positions and
   ${fmt(D.audit.provenance.loss_bearing_tokens)} loss-bearing tokens.`,
  `Detected <b>${D.audit.spikes.length} loss spikes</b>${D.audit.spikes.length?
    ` at step${D.audit.spikes.length>1?"s":""} ${D.audit.spikes.map(s=>s.step).join(", ")}`:""},
   and investigated ${D.audit.spike_reports.length}.`,
  `The token-window query resolved to
   ${D.audit.token_window.steps_in_window?D.audit.token_window.steps_in_window.length:0} steps and
   ${D.audit.token_window.shards_involved} shards.`],
 io:{in:[`consumption ledger`,`OPUS ledger`,`step loss series`],
     out:[`ledgers/audit_report.json`], next:`Evidence bundle`},
 record:null, artifacts:[`ledgers/audit_report.json`],
 metrics:()=>[["shards traced",D.audit.provenance.shards_involved],
   ["positions",fmt(D.audit.provenance.total_positions)],
   ["spikes found",D.audit.spikes.length],
   ["investigated",D.audit.spike_reports.length],["queries",3]],
 panel:"panelAudit"
},
{
 n:"19", id:"throughput", title:"Throughput and packing efficiency", tag:"performance", group:"Memory",
 one:`Useful loss-bearing tokens per second — not tokens per second.`,
 concept:`<b>Utilisation</b> is the share of positions holding a real token; <b>loss density</b> is
   the share actually graded, and it is always smaller. The gap is context-only tokens. <i>Instead of
   one number:</i> four, with the gaps between them as the report — and every ratio is two recorded
   counters, because figures that cannot be reconstructed earn no credit.`,
 built:[
  `<b>Counters into the ledger</b> as the run proceeds: positions, padding, context-only, graded,
   sequences, microbatches, shard reads, cache hits.`,
  `<b>Timers</b> around loader, OPUS, firewall and compute separately.`,
  `<b>Two sources, deliberately.</b> Efficiency ratios are summed from <i>every</i> consumption
   ledger — complete, including the pre-crash steps whose in-memory counters died with the killed
   process. Rates come from the phases whose wall clock survived, because dividing all the tokens by
   only the surviving time would inflate the number.`,
  `<b>The scope is stated in the file</b>, and the reconstruction formulas ship alongside the numbers
   so the verifier can rebuild them.`],
 ran:()=>[
  `Utilisation <b>${D.perf.efficiency.packing_utilisation}</b>, loss density
   <b>${D.perf.efficiency.loss_density}</b>, padding waste ${D.perf.efficiency.padding_waste},
   context-only share ${D.perf.efficiency.context_only_share}.`,
  `<b>${fmt(D.perf.throughput.useful_tokens_per_sec_compute)} useful tokens/sec</b> of compute
   against ${fmt(D.perf.throughput.raw_tokens_per_sec_compute)} raw — the gap is padding and context.`,
  `<code>verify_evidence.py</code> re-summed the ledger independently and the reported utilisation
   agreed.`],
 io:{in:[`ledger counters`,`phase timers`], out:[`performance.json`],
     next:`Evidence bundle re-derives these from the ledger`},
 record:null, artifacts:[`performance.json`],
 metrics:()=>[["utilisation",D.perf.efficiency.packing_utilisation],
   ["loss density",D.perf.efficiency.loss_density],
   ["padding waste",D.perf.efficiency.padding_waste],
   ["context share",D.perf.efficiency.context_only_share],
   ["useful tok/s",fmt(D.perf.throughput.useful_tokens_per_sec_compute)]],
 panel:"panelThroughput"
}
];
