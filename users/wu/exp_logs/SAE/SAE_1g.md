# SAE_1g — Evidence for a simple weak SAE initialization

## State
<!-- Overwritten in place, never appended; deleted at phase close. In-flight runs (job dir + the
question each answers), blockers, next action, proposals for the planner. -->

State as of 2026-08-23 -- 1g.9 is CLOSED by its own off-ramp; the USER resolved the direction fork
toward the decode route and 1g.10 (full-model LM-aware descriptive decode) is COMPLETE with its
table BLOCKED by its own pre-registered explanation duty; 1g.2 is READ and CLOSED on its gate;
1g.2a (H4-LM) is open with items 1-4 complete.

**1g.10a IS COMPLETE: VERDICT DISCHARGED, both tests passing**
(`H4CrossBeamDefectJob.2pV5rHuWJW3d`, 2 h 28 m, peak RSS 782 MB against an 8 GiB request).
TEST D re-decoded 81 utterances (three disclosed cells, twice each) with ZERO violations at
1e-12 nats. TEST U checked all 1,944 banked winners against their exact unpruned totals with
ZERO violations at 1e-6 nats. Exact-currency context: of 384 differing-winner cases,
exact(w512) beats exact(w256) in 352 and LOSES in 32, median gain +0.0109 nats per retained
unit -- the wider beam usually but not always finds the better-scoring sequence, which is the
search-error signature and not a scorer defect.

**1g.10b IS BUILT, TESTED AND LAUNCHED under the planner's option (b) ruling**
(`sae/h4_beam1024_probe.py`, speech-llm `8e2c841`; 36 probe cells plus one parity cell;
read `H4Beam1024ReadJob.tKbQ0MHLdX03`; manager `sae_1g_h4_full_model_decode` pid 450055). The
planner verified my blocker in the code and rejected option (a); the registration line "the
existing chunk class at the same contract" is amended by replacement.

- WHY A DEDICATED CLASS, in one line: extending `DECODER_BEAMS` would have orphaned the banked
  global-beam table (its `cells` is a hashed argument) and, through it, the discharged 1g.10
  read. VERIFIED AFTER THE CHANGE rather than assumed -- both
  `H4FullModelDecodeReadJob.MXhi20TtG1I0` and `H4CrossBeamDefectJob.2pV5rHuWJW3d` are
  hash-unchanged in the rebuilt graph.
- GUARD (1), NO COPIED CODE: `prefix_beam_decode`, `kenlm_phone_callbacks`, `decoder_record`,
  `_load_h1_units`, `_retained_runs`, `_load_channel` and `_validate_contract` are all imported
  from the modules that own them; the probe class contributes sharding arithmetic and
  bookkeeping only. A test greps the module source to enforce both halves -- that each name is
  imported and that none is redefined -- so the guard cannot rot silently.
- GUARD (2), THE BEAM IS HARD-PINNED: only 1024 and the 512 parity mode construct; 64, 256 and
  2048 are refused, so the probe cannot become a general beam knob.
- GUARD (3), THE PARITY CELL GATES EVERYTHING: the probe class runs at beam 512 on the contract
  shard for the median observed-agreement cell and must reproduce the banked production chunk's
  one-best sequences AND scores exactly; the reader emits no beam-1024 column otherwise. The
  cell is READ from the 1g.10a artifact's own disclosed selection
  (`controlled/reference|lambda=0.5|beta=-1`), never typed, and the config refuses to build
  unless that artifact's verdict is DISCHARGED.
- THE PRE-REGISTERED QUOTING BAR is in the module docstring verbatim: cross-channel comparisons
  only from cells whose 512-vs-1024 agreement is at least 26 of 27, every quote naming its
  27-utterance support. Every column carries `read_on` with that support so no quote can pass as
  an 890-utterance number.
- COST: 37 decode jobs on the contract shard (27 utterances each), 8 h and 4 GiB requested
  against the contract's 2 h / 2 GiB floor, because beam 1024 roughly doubles the beam-512 work.
- TESTED BEFORE LAUNCH: `scripts/h4_beam1024_probe_test.py` 8/8, the load-bearing one being that
  a failing parity cell suppresses every beam-1024 column -- without it, "this class decodes like
  production" would be an assertion rather than a measurement, which is the property that
  justified the dedicated class over extending the beam tuple.

**1g.10 IS COMPLETE, AND ITS TABLE IS BLOCKED BY ITS OWN PRE-REGISTERED EXPLANATION DUTY**
(verdicts 30-31; `H4FullModelDecodeReadJob.MXhi20TtG1I0`; all 1,332 chunks and 36 merges finished
with ZERO error markers). This is a blocking result for the planner, not a fallback decision.

- THE DUTY FIRED ON THE DECODER-DEFECT BRANCH. It was written into the producing module before any
  statistic existed: tiny margins where instability is measured confirm verdict 28's flat-score
  mechanism; wide margins with persisting instability indicate a decoder defect and BLOCK every
  cell. Zero of 36 cells reaches the registered 0.999 adjacent-beam agreement (min 0.2222, median
  0.6111, max 0.8889) and zero of 36 has a median score margin at or below the registered 1e-3
  nats per retained unit (min 1.210e-03, median 4.345e-03, max 1.540e-02). The report prints
  "DECODER DEFECT SUSPECTED -- no cell of this table may be read until that is explained".
- I AM NOT READING THE CELLS, and no row is compared against another. The per-cell correct-phone,
  total-variation and babble-null columns are banked in approach 16 as the record of what the run
  produced, under the duty's block.
- THE POSITIVE CONTROL IS HEALTHY, which is what makes the finding pointed rather than trivial:
  `controlled/reference` gives 7 of 12 readable cells and a best correct-phone fraction of 0.6010
  against the LM-blind local decoder's 0.5832 on the same channel. The decoder produces sensible
  content on a channel known to carry content while its adjacent beams still disagree on roughly
  one utterance in three.
- TWO OBSERVATIONS FOR THE PLANNER, offered as facts and explicitly NOT as rescues. The margin
  population is a mixture rather than uniformly wide: between 6.1 and 46.6 percent of utterances
  WITHIN a cell do sit at or below the flat threshold, so the cell medians summarize two
  populations. And the agreement column is read on the probe's 27 utterances, where one
  disagreeing utterance is 3.7 points and the 0.999 test cannot be met by anything short of
  perfect agreement. Neither softens a measured agreement of 0.2222. The threshold and the duty
  are pre-registered and I have not adjusted either; whether the suspected defect is investigated,
  and how, is the planner's call.
- CORRECTION TO MY EARLIER STATE ENTRY: I wrote that the agreement and drift columns would be read
  on 28 utterances, from the contract's 2,466 retained units. The artifact says 27, and 27 is the
  number the reader carries in its payload, its report header and its `beam_probe_note`. Every
  quote of those two columns is on 27 utterances.
- THE BEAM-256 CUT AS EXECUTED (planner ruling 2026-08-23): beam 512 ran the full registered scope
  and every decoded surface, margin, babble null and PER column reads from it; beam 256 ran ONE
  shard per cell, the shard READ from the measured contract's own `shard` block (index 28, 2,466
  retained units, the canonical heaviest selection-role shard `heaviest_shard` picked when the
  contract was measured) rather than a number I chose. 1,116 of 2,304 chunks dropped (31 of 32
  shards on each of 36 beam-256 cells; the ruling's "1,152" counted the retained 36 as well),
  leaving 1,152 beam-512 chunks + 36 beam-256 probes + 36 merges = 1,332, which is the on-disk
  census.
- SHAPE, exactly as experiment (1) registers it after the same-day amendment: the three audited
  count-4 channels (`controlled/reference` as positive control, `real/pseudo_pair_seed0` as the
  collapsed row, and `real/espum_seed0_update30000`, which the user promoted in as the old
  approach's projection into this route), all 12 registered grid points, on the 890 selection-role
  utterances. Channels are bound BY NAME through the existing count adapters, never by a hash I
  typed. The registered decoder is used unchanged -- no new modelling code, no new job class:
  existing `H4SequenceDecodeChunkJob`/`H4SequenceDecodeMergeJob` at the passing measured SELECTION
  resource contract, its fixed 32-way sharding, the deleted-silence boundary policy, the frozen
  duration law, the banked KenLM 4-gram replacing the fitting bigram. The global-beam eligibility
  flag is read for provenance and deliberately NOT applied, per "beam is not an eligibility bar
  here".
- SHARDING is not mine to choose: `H4_NUM_SHARDS = 32` is validated inside the job class and the
  merge refuses any other count. After the beam-256 cut the graph is 1,152 beam-512 chunks (3 x 12
  x 32) plus 36 merges, plus 36 single-shard beam-256 probes with no merge -- 1,188 chunk jobs in
  all, each shard about 28 of the 890 utterances.
- RESOURCES per chunk are the contract's own 1.5x rounding, which the job class enforces as a
  floor: 1 cpu, 2 GiB, 2 h. The measured selection-role maximum is 3,069 s and 0.91 GiB at beam
  512, so 2 h has ~2.4x headroom and the timeout doubling never needs to fire.
- BUDGET, and the one thing the planner should look at: prior chunk jobs each landed on their own
  booster node (checked across 20 finished update-role chunks -- 20 distinct hosts), so 1,536
  single-cpu jobs was on the order of 900-2,100 GH200 node-hours for a CPU-only descriptive read,
  which is what prompted the cut above. After it the graph is 1,188 chunk jobs, roughly half that.
  I flagged the budget rather than trimming it on my own, because scaling the registered scope
  down is the planner's call, and the planner took the cut.
- CONVENTIONS I had to pin, all in the producing module's docstring before any result exists.
  (i) `p_text` and `r_target` are 1g.9's pins IMPORTED from that module, not restated, so every
  cell is read against the same channel-independent target. (ii) One alignment convention for the
  whole job -- the plain unit-cost Levenshtein of `h4_validation_jobs.edit_distance`, which is the
  measure the funded 1g.2 descriptive read uses; pooled PER is total edits over total reference
  phones, and correct-phone fraction is `1 - pooled PER`, which may go negative when insertions
  outnumber the reference and is meant to. (iii) The babble null is unigram-MATCHED per cell: each
  draw keeps every utterance's decoded length and the cell's own decoded phone histogram and
  replaces only the ordering and identity. That is the direct answer to verdict 29 -- a cell
  cannot clear its own null by matching a histogram, which is how a content-free control passed
  1g.9's clause 1. 1,000 draws, seed 42, keyed per cell by a digest of the cell name; the bar is
  the empirical 99th percentile, reported with the null mean, standard deviation and maximum.
  (iv) Beam agreement and score drift are descriptive columns and never an eligibility bar, and
  the reader carries the registered explanation duty: it states FLAT SCORES when the measured
  instability sits on near-tied margins (verdict 28's mechanism) and DECODER DEFECT SUSPECTED when
  margins are wide while beams disagree, in which case no cell may be read.
- TESTED before launch, no artifact and no decoder: `scripts/h4_full_model_decode_test.py` 8/8.
  The load-bearing check is that the vectorised batched Levenshtein used for the null reproduces
  the scalar 1g.2 `edit_distance` exactly on random inputs -- otherwise the observed number and
  its null would be two different measures. Also checked: a content-free decode does not clear its
  own matched null, the null is still beatable so the bar is not vacuous, and the reader refuses a
  grid with any cell dropped.
- NOT DONE and not authorized: experiment (2)'s extension to fingerprint and random-map, which the
  registration puts behind the planner's read of (1), and the count-0 B-table cell, which the
  registration leaves to the planner and which is in any case a different object from the banked
  count-0 direct-Q read. Nothing here opens a selection surface.

**1g.9 EXPERIMENT 1 IS COMPLETE AND ITS RESULT IS A BLOCKING ONE FOR THE PLANNER**
(`H4CollapseLocateJob.gZ9d6e3E7ZGu`, 31 minutes; approach 15, verdicts 26-29). The registered
clause-0 off-ramp condition IS MET: at count 4 -- the planner's pre-stated decision read -- all five
starts already satisfy both proposed targets in the posterior (total variation 0.0108-0.0736 against
0.15; rate residual -5.5 to 0.0 percent against 20 percent), while the decode collapses on
`real/pseudo_pair_seed0` alone. `lambda_equal` is 8.1e+05 to 1.5e+08, so neither constraint can be
felt at any weight anyone would set. Per the gate, "the constrained-training arm does not run as
specced ... and the finding returns to the planner with the diagnostic as the deliverable" --
experiment 2's graph does not exist and I am not building it. RULED 2026-08-22 (PLAN_1G.md 1g.9
Status): clause 0 FIRED, the subphase CLOSES as the registered off-ramp outcome, experiments 2 and 3
do not run. The direction fork exceeded 1g.9's scope and went to the USER, who RESOLVED it
2026-08-23 toward the decode route -- the language model was never in the production decode and is
to be used -- so 1g.10 is registered and the "no further 1g work" hold is lifted for it alone.

Two readings the planner should carry into that ruling, both banked: a near-zero posterior total
variation is satisfied most easily by the LEAST informative channel (verdict 28 -- the collapsed
start has the lowest divergence and the worst likelihood, barely moving from count 0 while two other
starts move 1.3-1.6 nats per unit), and the registered `random_map` control has the SMALLEST decoded
unigram distance of all five starts, so the 1g.9 gate's own clause-1 readability criterion is passed
by a content-free null (verdict 29). Reported, not acted on: clause 1 is pre-registered.

Experiment 1 ran FIRST AND ALONE as the subphase requires; every input was already frozen, and the
job added only forward-backward posteriors and gradients, reading no gold.

- What it banks, for the five 1g.2a starts at repair counts 0 and 4: the posterior expected
  symbol-ENTRY distribution `q_bar` and the posterior expected rate `N/T` at the frozen H1
  duration, the same two statistics recomputed from the banked decoded one-bests (no new decode),
  and each proposed constraint term's gradient. Under the accepted two-state topology a symbol's
  first-position state is reachable only by entering that symbol, so its summed posterior occupancy
  IS the entry count and no transition posterior is needed.
- DECISIONS I made and pinned in the producing module's docstring, because the subphase left them
  open. (i) `r_target` is DERIVED, not chosen: the frozen H1 length-law fit maximized the
  geometric-duration marginal over the 6,414 update utterances, and `r_target` is that same law's
  posterior `E[N|T]` per retained unit at the accepted `p`; the memoryless reading `1-p` = 0.7644
  is reported beside it, and so is the count of forced symbol boundaries at deleted-silence gaps,
  which neither reading models and which is the one term by which a healthy posterior may
  legitimately exceed the target. (ii) A gradient norm is meaningless without a parameterization
  and an absolute norm compares nothing, so gradients are taken in `B = softmax(theta)` and
  reported as `lambda_equal = ||grad L_HMM|| / ||grad L_term||` -- the weight at which each
  constraint first pushes as hard as the likelihood. That also gives experiment 2's "one lambda
  magnitude" a traceable origin instead of an invented one. (iii) The constraints act on the update
  role but the banked decodes exist only on the 890 selection utterances, so a posterior-versus-
  decode read drawn across those folds would confound stage with fold: the posterior is computed on
  BOTH, and the clause-0 read is taken on the matched 890 with the update-role figure beside it.
- ASYMMETRY the reader must carry, stated in the artifact: at count 0 the banked decode reads the
  start's direct `Q` while the posterior reads the start's `B`; they are not two views of one
  table. At count 4 both read the same repaired `B`.
- A prior observation that sharpens what clause 0 is testing: the frozen local decoder is a
  per-unit argmax over `Q * prior` followed by run collapse (`channel_h.frozen_local_decode`). It
  consults neither the language model nor the duration law, so a collapse that lives in the decoder
  rather than in the objective is a live possibility the diagnostic can actually separate.
- TESTED before launch, no artifact and no scorer: `scripts/h4_collapse_locate_test.py` 6/6 -- the
  torch forward-backward transcription reproduces `channel_h.marginal_forward_backward` to 2.2e-16,
  all three gradients match central finite differences (worst 6.6e-09), and `E[N|T]` matches a
  direct sum and pins to the atom of a degenerate length prior. The job re-checks the two
  forward-backward implementations against each other at run time and refuses to report a gradient
  if they disagree.


**1g.2a, the user-mandated matched trigram/4-gram arm** (approach 14). Funded scope is Experiments
items 1-4; the F arm and every selector-shaped consequence stay closed under the 1g.2 verdict.

- Item 2, the exact context-state repair engine, is COMPLETE: `h4_context_engine.py`, 24/24
  synthetic checks. Orders 2/3/4 reproduce exhaustive path enumeration in likelihood, posteriors
  and counts with and without the deleted-silence boundary rule; instantiated with `legacy-2g` it
  reproduces the accepted dense engine; reachable histories come out at 1+39+39^2+39^3 = 60,880.
- Item 1, the matched LM artifacts, is BUILT AND RUN. All four automata exist: `legacy-2g`
  (`H4LegacyLmJob.lZI6TrYdVpev`) and matched 2/3/4 (`H4MatchedLmJob.T8ImJUXHaB0l` /
  `.Jb2m4aM2fUTy` / `.VpVkGMMy7xKW`) from `KenLMplzJob.ef5FXMvv8af5` / `.tis71OtNidgL` /
  `.bg0iYRzBQynx`. The legacy rebuild's phone-sequence hash reproduced the accepted H1's recorded
  hash EXACTLY over 39,630,169 phone lines, which is the binding that says this bigram is the one
  the accepted surface was fitted on. Both reruns and the reasoning for each are recorded in
  approach 14 and nowhere else.
- Item 3, the measured resource gate, is COMPLETE and PASSES (verdict 22):
  `config/sae_1g_h4_context_resource.py`, `H4ContextResourceGateJob.HA1vzRL7MEAz`. Exact order 4
  on the heaviest update chunk costs about 50 s and 0.67 GiB, so the request is 1 h per shard and
  5 h for the whole 32-shard fold in one process, at 2 GiB either way, against
  11.5 h and 256 GiB -- the same 1.5x multiplier and limits the accepted decoder resource contract
  used, so the two are comparable. It ran no M-step and applied no pruning. Independent
  cross-check: its lowest-entropy probe table is the same arm, count and array hash the accepted
  decoder contract selected months earlier, with the entropy agreeing to all 16 digits, so the two
  implementations of the probe rule agree.
- This config SUBSUMES `sae_1g_h4_matched_lm`, whose graph it contains; that config is now BLOCKED
  in `sis_managers.sh` so a second manager cannot double-submit the shared LM jobs.
- The EM driver itself (`h4_context_em.py`) is BUILT and verified, including exact sub-batching:
  the engine's alpha is `(batch, time, 2, 40^(order-1))`, so at order 4 a whole 200-utterance
  shard packed at once would need on the order of 150 GB, and the shard E-step now takes a
  `max_batch` that changes memory and nothing else.
- Item 4, the fixed-duration diagnostic on the reference plus the four accepted H3 starts, is
  BUILT: `config/sae_1g_h4_context_diagnostic.py`, 20 jobs
  (`H4ContextRepairJob`, five starts x four fitting LMs), each requesting 5 h and 2 GiB read from
  the gate artifact at graph-build time. It refuses to build unless the gate's `whole_fold_verdict`
  is PASS. LAUNCHED and COMPLETE 2026-08-22: all 20 cells finished, no errors; the count-4 table
  is in approach 14 and the smoothing-bridge reading is verdict 23. This config now subsumes
  `sae_1g_h4_context_resource`, which is BLOCKED in `sis_managers.sh` for the same
  double-submission reason `sae_1g_h4_matched_lm` is.

  The DECODE and descriptive phone-error half of item 4 is BUILT, RUN and COMPLETE 2026-08-22 at
  the user's request (verdict 24, table in approach 14): `config/sae_1g_h4_context_diagnostic_per.py`, module `h4_context_decode.py`,
  121 new jobs (60 channel adapters, 60 local decodes, one error read), all finished with no
  errors. The frozen decoder for all
  five of these starts is the LOCAL decoder -- `H4ProvisionalMaximaJob.ejmy4sdTOcS3` records
  `decoder.kind == "local"` with no lambda, no insertion penalty and no beam for every baseline row
  -- so no sequence decode, no beam and no `G_dec` binary enters this half at all. Counts 1/2/4 are
  decoded from the repaired tables `H4ContextRepairJob` already banked; count 0 is READ from the
  frozen 1g.2 direct-Q decode, because at count 0 the accepted method decodes the start's direct
  `Q`, that column cannot depend on the fitting LM, and the read job re-hashes all four columns and
  refuses the grid if they ever disagree. The bed is the same 890 selection-role utterances and the
  same gold the 1g.2 descriptive read used, so the count-0 and `legacy-2g` columns are directly
  comparable with verdict 21's numbers.

  One design point I had to settle, recorded here because it is the only place the two halves could
  have diverged: H4-LM-D freezes the decoder and changes only `G_fit` DURING REPAIR, so the local
  decoder's phone prior stays the accepted `phone_lm.npz` in every one of the twenty columns. That
  is why these cells cannot reuse `H4LocalDecodeJob`: that job pins its prior FILE to the channel's
  fitting LM, which is the right binding in 1g.2 where the two are one object and the wrong one
  here. The binding is re-aimed, not dropped -- the hypotheses still carry the channel's
  `fitting_lm_sha256`, so the frozen scorer still refuses a channel and a decode from different
  fitting LMs.

  STILL MISSING, so item 4 is NOT finished and no order may be chosen from what is banked: the
  own-minus-donor half. It needs the same 60 channel adapters (already built and hash-stable, so
  wiring it later costs no rework) fanned into `H4FixedTextScoreJob` at the ten frozen donor
  assignments -- 600 further jobs -- plus an aggregator that reproduces the 1g.2 surface's
  normalization exactly. That aggregator is the piece that needs writing.
- Nothing in 1g.2a reranks a maximum or reads a label. No 5-gram is funded. (Item 4 DOES repair
  utterances -- that is what the diagnostic is -- but only on the five funded starts, at the frozen
  `p` and topology, and it refits nothing.)

**RESOLVED 2026-08-22, both halves of a double outage.** The login node's `/tmp` filled (1.4 TB,
essentially all `/tmp/mmfs` GPFS traces -- NOT this user's files, whose whole Claude tree was 739 MB)
and every harness Bash call died with ENOSPC; separately the project fileset hit its quota and every
write failed with `EDQUOT`/`Errno 122`. The Bash half is fixed by pointing `CLAUDE_CODE_TMPDIR` at
`/e/scratch/spell/wu24/claude-tmp` (the project fileset is the wrong target -- it was the other
casualty). The quota half eased on its own; `jutil` reads the project fileset at 46.8/53.7 TB and
3.58M/4.0M inodes but its `last-updated` was three days stale, so it never showed the violation.
Cost: two D8.1a jobs killed mid-write (see SAE_3E1.md), no 1g.2a artifact affected.

**1g.2 is closed and unchanged.**

- The controlled validation read is COMPLETE (`config/sae_1g_h4_controlled_validation.py`;
  approach 12; verdicts 18-20): `H4ProvisionalWinnerAuditJob.kBCapQOpk1Hj` and
  `H4ControlledValidationJob.Otv6GBVY8ZUj`. **The selector verdict is NEGATIVE** -- `Sel` is
  inverted, not merely uninformative -- while the count method-level safety read PASSES with all
  three nonzero counts safe, and the sequence family is UNRESOLVED because no eligible sequence
  tuple exists. `h4_lm_trigger` is False, which is why 1g.2a is out-of-trigger user-funded work
  rather than a triggered follow-up.
- A STALLED watcher verdict on a finished 1g config is the known artifact: the manager exits on
  sisyphus's interactive "All calculations are done ... (v/u/c)?" prompt, which under `nohup` raises
  `EOFError` and looks like a crash, and the one-shot console then calls finished consumer jobs
  `waiting`. The on-disk `finished` marker and "Job finished successfully" settle it.
- H4 pre-label selection surfaces remain COMPLETE and unchanged: surface
  `work/speech_llm/sae/h4_selector_jobs/H4SelectionSurfaceJob.MKHfnUO9XwkU`, maxima
  `.../H4ProvisionalMaximaJob.ejmy4sdTOcS3`. The 821-job H4 prerequisite graph, the beam table and
  all 85 provisional maxima are preserved.
- The user-funded descriptive real-seed PER read is COMPLETE (`config/sae_1g_h4_real_seed_per.py`,
  `H4RealSeedPerJob.vu6Dp6HkJ2pH`; approach 13, verdict 21). It is a measurement over the closed
  gate and selects nothing; the 1,112-ID evaluation stays sealed. It was rerun once on 2026-08-22
  after the verifier's hand-back, to bank split-resolved PER in the job's own artifact instead of
  citing numbers from an unregistered console command; under the fixed seed every previously logged
  interval, point estimate and verdict reproduced identically.
- No `H4SelectorFreezeJob` was built, the final refits (7,304 construction IDs and 4,455 dev IDs)
  and the 1,112-ID evaluation stay closed, and the four real H3 rows stay sealed at every boundary
  the controlled read touches.

Blockers: none. Nothing is waiting on the user, the cluster or the verifier.

Proposal for the planner, refining the 2026-08-22 storage-placement policy in `PLAN.md`: that
paragraph classifies the H4/H4-LM family as not high-inode because it "writes 2-3 files per cell",
which is true of OUTPUTS and misses where this family actually spends inodes. A job dir also holds
one symlink in `input/` per upstream JOB, so `H4ContextResourceGateJob`, whose registered probe
population is the 340 corrected start/count tables, carries 342 input symlinks -- more than a
hundred times its own output count. Sisyphus keeps a cleared job dir forever, so every `-co` rerun
duplicates that whole tree: my three gate reruns left 1,026 symlink inodes of debris behind
(`H4ContextResourceGateJob.HA1vzRL7MEAz.cleared.0001..0003`), against 342 for the live artifact.
Small in absolute terms on a 4.0M limit, but it is the wrong shape, it is self-inflicted, and the
policy as written would not flag it. Suggested amendment: the inode test for a job design is
outputs per cell PLUS upstream fan-in, times the number of reruns a job is likely to need. The
cleared dirs hold no result and are safe to remove, but deletion is the user's call, not mine.

Proposal for the planner (shared-tree item, NOT mine to resolve): `config_sae_1g_v1.py` carries an
UNCOMMITTED working-tree edit adding a `corrective_h1()` builder that registers a second
`Phase1gH1Job` with `gold_json` inside a Phase-1g config. I did not write it and have neither
committed nor removed it -- committing another session's uncommitted work, or deleting it, are both
wrong from here. It does not enter the 1g.2 or 1g.2a graphs: neither
`config_sae_1g_h4_controlled_validation_v1` nor `config_sae_1g_h4_matched_lm_v1` imports it, and
the accepted H1 stays pinned at `Phase1gH1Job.HbxKiuBTJ8aN`. Its label-boundary status needs
whoever wrote it to explain it.

Proposals for the planner:

1. RATIFIED 2026-08-22 by the planner as exactly the registered local-winner exemption, and now
   asserted in the graph by `H4ProvisionalWinnerAuditJob` with empty audit mappings (approach 12):
   the frozen-versus-next-beam winner audit is satisfied without running it because all 85
   provisional winners are `decoder.kind = "local"`. The at-most-320 budgeted shard cells were not
   spent.
2. The label-free half of the baseline pre-evaluation-ready condition already reads positive -- the
   selector assigns a nonzero repair count to two of the four real starts
   (`espum_seed0_update30000` and `pseudo_pair_seed0`, both count 4) -- so the only outstanding
   input to that condition is the controlled method-level safety read, which is a label read.

## Approach

This log contains experimental evidence only; the current method, gates, and future work are defined
in `PLAN_1G.md`. `T_phi` below means the unpaired text converted to 39 stress-free ARPAbet phones. A
“rung” is one fixed audio-unit stream: adjacent-deduplicated raw codes or one of the pooled streams
`seg16`, `seg12.5`, and `seg9`.

1. **Channel-shape screen (1g.0).** On 2,703 dev-clean and 2,864 dev-other utterances, measure how
   much adjacent audio units depend on one another and compare it with three increasingly flexible
   channel shapes:

   - one audio segment per text symbol;
   - variable duration with conditionally independent emissions;
   - two ordered emission states per text symbol.

   The decision statistic is measured audio-pair mutual information divided by the maximum allowed
   by the tested shape. A ratio at or below 2 is admissible. The plug-in and Miller–Madow-corrected
   estimates must agree or the cell is indeterminate. The one-segment column needs no gold duration.
   The independent-duration and two-state columns were read at within-symbol rates measured from
   gold boundaries; they are historical diagnostics and do not set a prospective candidate's
   duration.

   Dev-other inputs to the decision:

   | audio stream | lag-1 mutual information | gold within-symbol pair rate | cross-utterance floor | gold cross-boundary share |
   |---|---:|---:|---:|---:|
   | adjacent-deduplicated raw codes | 3.2964 | 0.7093 | 0.2423 | 0.291 |
   | `seg16` | 2.6171 | 0.4226 | 0.4005 | 0.577 |
   | `seg12.5` | 2.3730 | 0.3164 | 0.4995 | 0.684 |
   | `seg9` | 1.9596 | 0.2037 | 0.6444 | 0.796 |

   Resulting ratios on dev-other, with the duration-bearing columns at that gold-derived operating
   point; parenthetical values are the only materially different dev-clean reads:

   | stream / text symbols | one segment | independent duration | two ordered states |
   |---|---:|---:|---:|
   | raw / phones | 5.76, rejected | 1.69, admissible | 1.30, admissible |
   | raw / characters | 5.62, rejected | 1.96, admissible (**2.14, rejected**) | 1.47, admissible |
   | `seg16` / phones | 4.58, rejected | 2.42, rejected | 1.50, admissible |
   | `seg16` / characters | 4.47, rejected | 2.73, rejected | 1.66, admissible |
   | `seg12.5` / phones | 4.15, rejected | 2.81, rejected | 1.64, admissible |
   | `seg12.5` / characters | 4.05, rejected | 3.08, rejected | 1.79, admissible (**1.99, indeterminate**) |
   | `seg9` / phones | 3.43, rejected | 3.01, rejected | 1.72, admissible (**1.91, indeterminate**) |
   | `seg9` / characters | 3.34, rejected | 3.14, rejected | 1.84, admissible (**2.03, rejected**) |

   Subtracting the complete cross-utterance floor still leaves every pooled independent-duration
   cell above 2 (2.02–2.79). The raw-character independent-duration result is split-dependent, not a
   clean pass.

2. **Spectral two-class anchor (1g.4).** Split text symbols and audio units into a
   syllabic/non-syllabic pair using the largest-eigenvalue eigenvector of the symmetric normalized
   Laplacian. The text side is a positive control. “Mass accuracy” weights each unit type by its
   number of evaluated occurrences; the majority is the score from always choosing the dominant
   class. “Containment” checks whether the proposed unit–phone mask retains the class of each unit's
   majority phone. The registered hard gate requires mass accuracy at least 0.85, at least 0.20 above
   the measured majority, and containment at least 0.85.

   The initial audio read used segment duration to orient the two classes because the descriptor dump
   did not yet exist. The registered energy/periodicity orientation was subsequently run; it flips
   `seg16` and `seg12.5` and leaves the verdict unchanged. The canonical dev-other read is the fixed
   572-utterance evaluation fifth; the 540-utterance dev-clean fifth gives the same all-fail verdict.

   | side / stream | top-eigengap check | canonical mass accuracy | measured majority | verdict |
   |---|---|---:|---:|---|
   | text phones (`T_phi`) | pass | 1.0000 | 0.6095 | positive control passes |
   | text characters | pass | 0.9764 | 0.6130 | positive control passes |
   | audio raw | fail | 0.5488 | 0.5711 | fail |
   | audio `seg16` | fail | 0.4452 | 0.5449 | fail |
   | audio `seg12.5` | pass | 0.4968 | 0.5312 | fail |
   | audio `seg9` | unstable | 0.7867 | 0.5154 | fail |

   The later permutation and bootstrap uncertainty reads were not persisted in a catalogued
   artifact, so this log does not treat their reported numbers as evidence. The saved point-estimate
   jobs do preserve every accuracy above, and every stream fails the 0.85 accuracy bar independently
   of either pre-check.

3. **Deterministic hard two-class descriptor screen (1g.4).** Compute seven waveform descriptors on
   the frozen wav2vec2 50 Hz grid, average each descriptor per unit over the 8,416-utterance seed bed
   (3,685,941 frames), and set its binary mass cut from the syllabic proportion of unpaired text. No
   label enters the descriptor dump or cut. The canonical reads use the same fixed 540/572 evaluation
   fifths and silence-unit convention as Approach 2.

   | stream | dev-clean best accuracy | dev-other best accuracy | dev-other majority | dev-other margin | gate |
   |---|---:|---:|---:|---:|---|
   | raw | 0.7894 | 0.7929 | 0.5711 | +0.2218 | fail |
   | `seg16` | 0.7849 | **0.8130** | 0.5449 | +0.2681 | fail |
   | `seg12.5` | 0.7629 | 0.7859 | 0.5312 | +0.2548 | fail |
   | `seg9` | 0.7588 | 0.7824 | 0.5154 | +0.2671 | fail |

   Energy is the best descriptor on every stream. These seven descriptors are seven alternative
   estimators of the **same** syllabic/non-syllabic target. They are not the six independent
   articulatory memberships required by the registered soft-product specification, so this screen
   does not answer that soft-product prerequisite.

4. **Exploratory phone repair rehearsal (E5).** The completed job exercises the soft
   two-sub-state Baum–Welch implementation on `seg12.5`, but its configuration does not implement the
   corrected experiment in `PLAN_1G.md`:

   | item | completed exploratory implementation |
   |---|---|
   | population | all 2,864 dev-other utterances used both to fit the oracle map and to score it |
   | preprocessing | differs from the frozen 1f fixture |
   | x-axis | code field `fraction_correct` is misnamed: it is the probability of retaining the reference label; `1` retains all and `0` redraws all |
   | initial channel | hard unit-to-phone map with 0.9 mass on assigned units and a uniform remainder |
   | state split | independently randomized ±10% perturbations |
   | duration | fixed mean 1.463, derived from the old gold-boundary operating point |
   | text sample | every 80th line, capped at 300,000 lines |
   | update | 30 soft emission-only Baum–Welch steps under a pinned phone bigram |
   | decoder / stopping | posterior argmax; stopping iteration selected by weighted phone-LM perplexity |
   | real candidates | neither ESPUM nor fingerprint nor the two treated controls is run |

   The old prose called retention 0 the reference endpoint and retention 1 content-free; the code
   does the reverse. The endpoint results were:

   | configured retention | realized units matching reference | start PER | LM-selected PER | step-30 PER |
   |---:|---:|---:|---:|---:|
   | 0 | 0.016 | 1.0109 | 0.8409 | 0.8409 |
   | 1 | 1.000 | 0.4865 | 0.4589 | 0.6699 |

   These show that the implementation runs and that the fixed 30 repair steps can drift even from
   its own fitted reference. Because the same utterances built and scored that reference, and because
   no real seed or treated control was run, the numbers are engineering evidence only and cannot
   fire a gate.

5. **Banked phone-seed artifact audit.** Neither 1f seed persisted a per-unit map, and both original
   seeds were fitted on all 8,416 utterances, including the fixed evaluation audio. The deterministic
   fingerprint map can be recomputed with its original arguments to verify its complete recorded
   error decomposition, but that full-bed reconstruction is provenance only. ESPUM did persist the
   selected neural checkpoint, a context-dependent convolution with `conv.weight` shape
   `(39, 500, 4)` rather than a per-unit table; that checkpoint is likewise transductive provenance,
   not a held-out input. A decisive held-out row requires construction-only fingerprint recomputation
   or ESPUM retraining and a newly measured operating point.

   The frozen encoder normalization, PCA, and K-means were fitted only on the 2,849 dedicated train
   utterances, and segment pooling is per utterance; those common transforms did not fit evaluation
   audio. The historical `UnitWordStreamJob.eIxgmMh99RSE` did learn its proxy-silence mask from all
   8,416 utterances, however. That stream is valid for fixture reproduction but not as the unchanged
   prospective held-out stream.

6. **Construction-only topology read (H1).** On the frozen 6,414-utterance update partition from the
   8,416-utterance seed bed, fit each route's duration from unpaired complete text and update audio,
   then read both channel shapes on those same masked update sequences. The execution snapshot archived
   the four imported source modules before computation; its SHA-256 is
   `b939c19d669b1b5c585915cb7a634196d31b64f38113698cabab35a1503832d9`. Both plug-in and
   Miller--Madow ratios must be at most 2 for a shape to be admissible.

   | route | retained units | fitted mean duration | one-state ratios (plug-in, MM) | two-state ratios (plug-in, MM) |
   |---|---:|---:|---:|---:|
   | `seg12.5` / phones | 397 | 1.308221 | 3.2424, 3.1845 | 1.8525, 1.8194 |
   | raw / characters | 395 | 2.601966 | 2.6394, 2.6322 | 1.8086, 1.8037 |

7. **Corrected H2/H3 phone calibration path.** H2 keeps a zero-probability duration self-loop
   impossible and treats the normalized, once-floored emission table `B(unit | phone)` as its
   canonical scoring, decoding, perturbation, and repair input. Its validator requires finite,
   positive rows whose sums agree with one at zero relative tolerance. The production decoder now
   rejects coerced grid/shard/count values and non-finite scalar evidence and proves deterministic
   shard coverage. The actual wired start has the intended 39-phone by 500-unit inventory and exact
   H1/H3 provenance. It retains eight alternatives as an output-only cap; one-best and confidence use
   the complete surviving beam. Deleted silence is one shared duration boundary for fixed scoring,
   decoding, and repair forward--backward, while phone-LM history continues across the gap. H3
   reconstructs `seg12.5` tokens as maximal runs on the
   original frame raster, removes frozen-mask silence runs as chunk boundaries, and pools ESPUM logits
   over each run's complete frame span. On the real 8,416-utterance seed bed this construction contains
   exactly 715,099 retained runs in 72,842 chunks, matching the verifier's reference counts. The
   calibration graph fits fingerprint, random-map, pseudo-pair, and ESPUM rows on the H1 update role;
   ESPUM reads the disjoint selection role label-free. The strict calibration projection averages the
   frozen selected checkpoint posterior on exactly the update role to persist `Q(phone | unit)` and
   `B(unit | phone)`. A separate graph wires blind construction-role fingerprint, random-map,
   pseudo-pair, and ESPUM refits plus the ESPUM projection. Resume state includes NumPy, CPU/CUDA
   Torch, exact CUDA device count and one nonempty RNG tensor per device, permutation, and batch-offset
   state; a serialized GH200 interrupted-versus-uninterrupted trajectory check exercises it. Projection
   manifests hash all eight modules imported by the runtime path. Its final graph contains all four
   initializer families; the trainer verifies runtime source hashes before work begins.

8. **H4 controlled calibration and repair production stage.** The first production H4 graph freezes
   the calibration inputs before any selection or evaluation labels are read. Its preparation job
   uses the accepted H1 roles and silence mask, the exact `seg12.5` frame raster, every line of
   `T_phi`, and pinned dev-clean/dev-other MFA parquet snapshots. It fits the positive-reference
   channel only on the 3,565 labelled dev utterances inside the 6,414-utterance update role and
   persists content hashes for those external inputs and every local runtime source. The fixed
   library contains the reference, 50 retain/redraw maps (ten `q` levels by five initial draws), ten
   soft-damage rows, and 20 marginal-random maps. The same update-only two-state repair is then wired
   at counts 0, 1, 2, and 4 for all 81 controlled starts and the four accepted H3 calibration starts.
   This stage does not yet decode, score the 890-utterance selection role, freeze the selector, or
   construct the final 7,304-utterance refits; those remain downstream H4 work.

   The graph completed on 2026-08-20. `H4CalibrationPreparationJob.DPv4aIqwPEzM` produced the
   start bundle, phone LM, and one donor table covering all 8,416 construction/evaluation-bed
   utterances. All 85 repair trajectories now contain finite normalized two-state tables at exactly
   counts 0, 1, 2, and 4: 81 controlled trajectories plus fingerprint, random-map seed 1000,
   pseudo-pair seed 0, and ESPUM seed 0/update 30,000. This is completed calibration infrastructure,
   not an H4 gate result.

9. **Corrected H4 recovery and decoder-resource preflight.** A role firewall first materializes only
   the 3,565 labelled dev utterances inside the 6,414-utterance update role. From those sealed inputs,
   the recovery stage regenerates `Q(phone | unit)` for the 71 non-soft controlled starts, imports the
   four persisted H3 `Q`/`B` pairs, and authorizes reuse only when canonical `Q`-to-`B` conversion and
   the old count-0 repair state match exactly. It replaces exactly the ten old B-space soft starts by
   the registered Q-space mixtures and reruns only those trajectories. Every one of the 85 direct-Q
   adapters and 340 lossless count adapters depends on the recovery authorization.

   Before selector freeze, the donor graph contains only the exact 890-utterance selection role. Its
   same-speaker support law requires donor retained length at least the source length and donor chunk
   count at most the source count before duration/rate ranking; sources without compatible donors are
   retained as `no_swap`. The resource graph then measures the exact 32-way update and selection
   workloads over three hash-deduplicated representative channel tables and all 48 decoder settings,
   followed by the globally slowest/highest-RSS setting on the heaviest real shard. It contains no
   construction/evaluation donor, decode, scoring, selector, or final-refit job.

10. **H4 label-free global-beam boundary.** The baseline `legacy-2g` continuation reuses the frozen
    three-table update inventory and passing 10-hour/2-GiB update contract. On its canonical heaviest
    shard it runs exactly 144 cells: three representatives by the fixed 12
    `(language-model scale, insertion penalty)` settings by beams 64/128/256/512. For each setting,
    the first adjacent beam pair passing at least 99.9% exact one-best agreement and strictly less
    than `1e-4` absolute decoder-score change per retained unit on every representative freezes the
    smaller beam; a setting with no passing pair is ineligible. This stage reads update audio only and
    stops before selection-surface decoding, controlled labels, construction refit, or evaluation.

    The boundary completed on 2026-08-21, and none of the 12 sequence-decoder settings has a stable
    beam under the registered criterion. Even the adjacent pair with the best worst-representative
    one-best agreement reaches only 0.7313 (required 0.999), while the pair with the smallest
    worst-representative score change reaches 0.005448 nats per retained unit (required strictly
    below 0.0001). Thus the baseline H4 surface retains the local decoder only; no sequence setting
    may enter selection, and the graph stopped without opening selection labels or evaluation.

11. **H4 pre-label selection surfaces (1g.2).** With the sequence family ruled out by approach 10,
    the baseline surface is local-only: 340 local decodes (85 starts by repair counts 0/1/2/4),
    3,400 fixed-text donor scores (10 frozen assignments per tuple), one selection surface and one
    provisional-maxima read, over the 890 selection utterances and reading no label. The statistic
    itself contributes from the 513 donor-eligible sources of those 890 -- 235 dev-clean and 278
    dev-other, the same set in every tuple and every assignment -- because the 377 `no_swap` sources
    have no eligible donor and are absent by construction. Each tuple's statistic is the
    own-minus-donor fixed-text channel rate
    (`own_logp/own_retained_units - donor_logp/donor_retained_units`), summed with `math.fsum` in
    sorted-id order over the eligible rows of each split, then weighted by the registered fixed
    shares 432/890 and 458/890 -- the no-renormalization rule, so the weights are the split sizes of
    the full 890 and not of the 513 that contribute. The maximum over the
    10 frozen assignments is that tuple's provisional maximum; ties break on the registered order
    `legacy-2g;repair_count;local;lambda_outer_beta_inner;initializer;seed;update`. Both artifacts
    carry `contains_labels: false` and `frozen_pre_label: true`, and their recorded `code_identity`
    sha256 of `h4_selector_jobs.py` (`517401b9...`) matches the committed file at `84808a8` byte for
    byte.

    The graph completed 2026-08-21. All 85 starts (81 controlled, 4 real) produced a finite
    provisional maximum and every one of the 85 winners is `decoder.kind = "local"` and
    `eligible = true`. The five registered cross-start rows:

    | cross-start row | provisional maximum | winning repair count | winning assignment |
    |---|---:|---:|---:|
    | `real/random_map_seed1000` | 10.7753 | 0 | 0 |
    | `real/fingerprint` | 10.1520 | 0 | 0 |
    | `controlled/reference` | 5.8265 | 4 | 6 |
    | `real/espum_seed0_update30000` | 4.2613 | 4 | 0 |
    | `real/pseudo_pair_seed0` | 0.1437 | 4 | 5 |

    Two internal consistency reads, neither a gate: `controlled/random_map_seed1000` returns
    10.7753, identical to its `real/` twin at every printed digit, and all 324 controlled
    within-sequence choices (81 rows by 4 counts) plus all 4,080 `global_beam_ineligible` entries
    (340 tuples by 12 settings) are ineligible, so the surface carries no sequence-decoder score
    anywhere -- which is what approach 10 requires.

    The two channel degeneracies among the 81 controlled starts are CONSTRUCTION, not a copy error,
    and both are the same fact: `Q_LEVELS` ends at 1.0 (`h4_jobs.py:34`), so level index 09 is the
    undamaged endpoint of each damage ladder. In the map ladder `keep_count` is then the whole live
    set, so `assignment[keep] = reference_map[keep]` overwrites every drawn entry and the draw seed
    cannot survive -- `controlled/map_q09_draw00..04` are necessarily one channel, and they return
    10.3872214431 at all five draws. In the soft ladder `canonical_soft_q` early-returns
    `reference.copy()` at `q_level == 1.0` (`h4_production.py:223-224`), so `controlled/soft_q09` IS
    the reference channel and returns the reference's 5.8264784397 exactly. The two q09 endpoints do
    NOT coincide with each other: the map ladder builds a hard one-hot `q` from the reference map
    while the soft ladder keeps the reference's own soft rows, which is why one reads 10.3872 and
    the other 5.8265. Effective independent controls are therefore 76 of 81, a property of the
    registered ladder design that any clustered interval or null spread over the controls must
    respect.

12. **H4 controlled validation read (1g.2 label boundary).** The planner opened the controlled
    reference labels on 2026-08-22 (`PLAN_1G.md` Status; `PLAN.md` queue item 1, user priority).
    Two jobs sit on the frozen graph and nothing else: `H4ProvisionalWinnerAuditJob.kBCapQOpk1Hj`
    emits the audited maxima -- with EMPTY audit mappings, which is how the local-winner exemption
    is ASSERTED rather than assumed, because that job errors on a sequence winner lacking an audit
    -- and `H4ControlledValidationJob.Otv6GBVY8ZUj` is the only label reader in Phase 1g
    (`speech_llm/sae/h4_validation_jobs.py`, config `sae_1g_h4_controlled_validation`). No decode,
    score, refit or maximum is recomputed: every input was already on disk, so the read is one CPU
    job over frozen artifacts plus one gold file.

    The four real H3 rows are not passed to the reader at all; it refuses any non-`controlled/`
    key at construction, so their own errors cannot be read even by mistake. Registered statistics
    (transcribed into the module docstring, so the reporting rule lives with the code that
    produces the numbers): reference vs the strongest null under a simultaneous 95 % interval,
    formed by bootstrapping `Sel(reference) - max_over_controls Sel` so bounding the maximum
    bounds every control at once; Spearman(`Sel`, -error) globally and inside the predeclared
    starting-PER band 0.80-0.93; the reference-start paired count safety read PER(r)-PER(0) for
    r in 1/2/4 against the 0.05 margin; and the within-trajectory rank/regret/count-0 bounds.
    Bounds are the ONE-SIDED 5th/95th percentiles with the two-sided pair reported beside them,
    10,000 resamples at seed 20260822, resampled channel cluster then donor assignment then
    utterance within split.

    Duplicate channels are collapsed by the artifacts' own `channel_array` sha256 rather than by
    parsing arm names; that rule independently reproduces approach 11's degeneracy finding from
    the surface alone, which is why it is the rule rather than a hand-written exception list.
    85 starts carry 79 distinct channels through THREE duplicate groups: the five `map_q09` draws
    (one channel), `soft_q09` with `controlled/reference`, and -- cross-namespace, so the reader's
    controlled-only view never sees it -- `controlled/random_map_seed1000` with
    `real/random_map_seed1000`. Effective independent CONTROLS are 76 of 81, unaffected by the
    third group since it pairs a controlled arm with a real one. (Corrected 2026-08-22: the first
    version named two groups and so could not reach 79 from 85; the verifier found the third in
    the audited maxima.) Before any interval is taken the reader rebuilds every controlled
    tuple's frozen `Sel` from that tuple's own stored per-utterance deltas and refuses to continue
    unless it matches to 1e-9 -- otherwise the bootstrap would be resampling a different statistic
    than the one the surface froze, which nothing downstream would catch.

    `scripts/h4_validation_test.py` carries 53 synthetic-only checks: the interval readings in all
    three directions, a planted quality-tracking selector that must PASS and its inversion that
    must read NEGATIVE, a harmless and a damaging repair count, the verdict table, and the label
    firewall. It builds every fixture in memory and reads no real artifact, so it runs before the
    boundary opens and cannot launder a real number into a passing check.

    `H4SelectorFreezeJob` is deliberately NOT in this graph. It raises unless the selector verdict
    is PASS -- correct, since a failed selector must not freeze a maximum -- so adding it now
    would place a job in the graph whose outcome is not yet known. It is built in a follow-up
    config only if the validation artifact reads PASS.

13. **User-funded descriptive PER read over the four real H3 seeds (1g.2, gate already closed).**
    Registered by the planner on 2026-08-22 from the user's request to compute PER on real
    dev-other data, as a measurement over the closed gate rather than a revision of it
    (`PLAN_1G.md` 1g.2 Status). One CPU job, `H4RealSeedPerJob.vu6Dp6HkJ2pH`, in its own module
    and its own config: plain per-split PER on the 890 selection-role utterances (432 dev-clean,
    458 dev-other) for the four real rows at counts 0/1/2/4, from the frozen surface's EXISTING
    decode artifacts against the same `GoldPhonesJob.ZGSp0hxyd2YP` gold. No decode was run, no
    score recomputed, no maximum reranked.

    `H4ControlledValidationJob` and its label firewall were not touched, as the registration
    requires. The two jobs are mirror images and refuse each other's inputs -- the validation
    reader rejects any non-`controlled/` key, this one requires exactly the four registered real
    rows at exactly the four registered counts -- so neither can quietly become the other. The
    reporting rule lives in the job's docstring AND in its output payload: descriptive and
    evaluation-only, selects nothing, funds nothing, and any later decision that uses these
    numbers to pick a seed or count must be re-registered with the label circularity disclosed as
    a supervision cost. The 1,112-ID held-out evaluation stays sealed.

14. **1g.2a: matched higher-order fitting LMs and the exact context-state repair engine.**
    User-mandated trigram/4-gram work, registered by the planner on 2026-08-22 as out-of-trigger
    (`h4_lm_trigger` is False from verdict 18). Funded scope is Experiments items 1-4; the F arm
    and every selector-shaped consequence stay closed under the 1g.2 verdict.

    The question item 1 answers is separation, not size. The accepted baseline `legacy-2g` is an
    add-one bigram, so raising the order alone would move SMOOTHING and ORDER together. The
    matched family is unpruned modified Kneser-Ney at orders 2/3/4 built from the SAME pinned
    complete `T_phi`, the same 39-phone inventory and the same BOS/EOS convention, with the
    canonical `phoneme_ngram_lm` settings (`interpolate_unigrams=True`, `pruning=None`, discount
    fallback 0.5/1.0/1.5 -- Kneser-Ney needs the fallback because a ~40-symbol vocabulary has no
    singleton unigrams). Matched 2 vs 3 vs 4 is then the order contrast; `legacy-2g` vs matched 2
    is the separate smoothing bridge.

    An ARPA is a backoff SCORING function, not a generator: `<unk>` carries mass, `<s>` is a
    token, and a history's continuations need not sum to one over the symbols the repair law
    permits. The compiler therefore evaluates the full backoff recursion at every history, drops
    `<unk>`/`<s>` as successors, explicitly renormalizes over the 39 phones plus EOS, and RECORDS
    the removed mass per history instead of absorbing it -- that recorded number is the audit
    trail for a leaking ARPA. The engine's BOS-padded history collapses to a single `<s>` before
    the lookup, since repeating it would query n-grams the training text never contains.

    `H4LegacyLmJob` exists because the baseline was never an artifact: `Phase1gH1Job` builds the
    add-one bigram inside its own run and keeps only what it derived. The job rebuilds it through
    the same code path and refuses to continue unless the rebuilt phone-sequence hash and line
    count match what the accepted H1 recorded. That check, not a file hash, is what binds the
    bigram to the `T_phi` the accepted surface was fitted on.

    Item 2 is the engine those automata feed. The accepted engine is bigram-specific and its dense
    transition matrix at order 4 would be about 118 GB, so it cannot represent the arm at all. The
    ruled form keeps the state as (duration sub-state, BOS-padded history), makes duration moves
    elementwise-diagonal, and leaves exactly one contraction over phone-exit arcs against the
    normalized per-history table; emissions stay tied by (phone, sub-state) and broadcast rather
    than materializing per history, and the backward pass is shared across sub-states, which the
    topology proves. Reachable histories come out at exactly 1+39+39^2+39^3 = 60,880 and an
    order-4 E-step over the update fold costs about 5.6e12 operations, both derived here and both
    matching the plan's independent projection.

    | artifact | deviating parameters | reachable histories / arcs; max renormalized mass | job hash |
    | --- | --- | --- | --- |
    | `legacy-2g` | add-one, order 2 (the accepted baseline) | 40 / 1,600; 0 (add-one needs none) | `H4LegacyLmJob.lZI6TrYdVpev` |
    | `matched-2g` | MKN, order 2 | 40 / 1,600; 7.05e-08 | `H4MatchedLmJob.T8ImJUXHaB0l` (`KenLMplzJob.ef5FXMvv8af5`) |
    | `matched-3g` | MKN, order 3 | 1,561 / 62,440; 1.274e-04 | `H4MatchedLmJob.Jb2m4aM2fUTy` (`KenLMplzJob.tis71OtNidgL`) |
    | `matched-4g` | MKN, order 4 | 60,880 / 2,435,200; 1.274e-04 | `H4MatchedLmJob.VpVkGMMy7xKW` (`KenLMplzJob.bg0iYRzBQynx`) |

    All four normalize to machine precision (max absolute error 5.6e-16). The renormalized mass is
    what each ARPA was spending on symbols this automaton cannot emit, and it is small: the
    matched family loses at most 1.3e-04 at any history, so the order contrast is not an artifact
    of throwing mass away. The legacy rebuild reproduced the accepted H1's recorded phone-sequence
    hash exactly over 39,630,169 phone lines, which is the binding that says this bigram is the
    one the accepted surface was fitted on -- a file hash would not have shown that.

    Two reruns happened in this phase and they were decided differently, which is worth stating in
    one place because the record otherwise reads as contradictory. The three MATCHED compiles were
    rerun, because their first outputs were provably wrong: one had read a superseded ARPA. The
    LEGACY artifact was rerun for a cosmetic reason -- to write the matched family's manifest field
    names so the four automata could be compared without translating between two vocabularies. That
    second rerun was a MISTAKE OF PROCESS, and the verifier was right to hand it back
    (2026-08-22): unlike the 1g.2 validation rerun, this artifact had consumers, and rewriting a
    manifest that finished cells had already bound could have stranded them. It happens to be safe
    -- the rerun completed before any diagnostic cell started, and every legacy cell's
    `input_content_sha256.automaton` equals the manifest's current `automaton_sha256` (`0aa488aa`),
    checked directly -- but that is timing, not design. The rule stands as the 1g.2 rerun set it: a
    finished artifact may be rerun only when nothing has consumed it, and a cosmetic gain never
    clears that bar.

    A residue of that decision remains and is left alone deliberately: `code_identity` hashes the
    whole module, and `h4_lm_artifacts.py` holds both job classes, so the legacy artifact now
    records a different module hash from the three matched ones. The difference is the legacy job's
    own manifest fields and memory request, neither of which the matched compile path executes.
    Re-running three correct artifacts to erase a cosmetic hash difference would repeat the mistake
    above, so the coarseness is recorded here instead.

    Orders are read from each job's own `identity`/`order` fields in the loaded graph, not from a
    hand-written hash-to-order list. Time and memory for the KenLM jobs trace to the order-4 run
    of the same job class over this same corpus (`KenLMplzJob.0aJeN88X6EdW`: 0.05 h elapsed,
    3.3 GiB max RSS at `mem=16`, `time=2`).

    Verification is synthetic-only and reads no real artifact, so it runs before anything is
    measured and cannot launder a real number into a passing check: `scripts/h4_context_engine_test.py`
    24/24 -- orders 2/3/4 reproduce exhaustive path enumeration in likelihood, posteriors AND
    counts, with and without the deleted-silence boundary rule, and the `legacy-2g` instantiation
    reproduces the accepted dense engine's likelihood, posteriors and M-step; `scripts/h4_lm_artifacts_test.py`
    24/24 -- backoff evaluation, BOS/EOS handling, renormalization audit, and the plan's separate
    requirement that the matched order-2 automaton reproduce exhaustive enumeration.

    The engine is the E-step only, so the trainer itself is a third piece:
    `h4_context_em.py` is the counterpart of the accepted `soft_reestimation_curve`, preserving
    everything that routine owns -- the two-state symmetry break applied once immediately before
    repair step 1, the unperturbed count-0 snapshot, the single common floor, the pinned fitting LM
    the M-step never touches, and the meaning of a snapshot at count `n` (the table after `n`
    M-steps with the likelihood evaluated at that table). It is written in shard form even for one
    shard, because the plan requires shards to aggregate likelihood and expected counts before ONE
    common M-step: a per-shard M-step is a different estimator, not a parallel implementation of
    the same one, and making the single-process path the one-shard case is what keeps that true
    rather than merely intended. Shards return UNNORMALIZED counts; only the driver floors and
    normalizes. `scripts/h4_context_em_test.py` 36/36, the decisive check being that the order-2
    instantiation reproduces the accepted trainer snapshot for snapshot across counts 0/1/2/4 --
    without that, a higher-order likelihood would not be comparable to the banked baseline at all
    -- plus 2/3/5-way sharding changing no number and the boundary contract refusing every way of
    getting it wrong.

    A real corruption was caught while item 1 first ran, and it is the reason `parse_arpa` now
    checks the ARPA's own declared per-order counts against what it read. The login-node
    LocalEngine ran each `KenLMplzJob` more than once (order 2 twice, order 3 three times), and a
    compile read `lm.gz` while lmplz was rewriting it. The order-2 compile failed loudly. The
    order-3 compile did NOT: it finished, its `ngram_counts` match the final file exactly, and
    only its banked `arpa_sha256` gives it away -- no file on disk carries that hash any more.
    Structurally-plausible truncation is invisible to every other check, so all matched compiles
    are rerun under the hardened reader and the family shares one compiler identity.

    Item 3, the measured resource gate, is COMPLETE and PASSES (verdict 22 and its correction):
    exact order 4 on the heaviest of the 32 update chunks costs about 50 s and 0.67 GiB, giving a
    1 h per-shard and 5 h whole-fold request at 2 GiB against 11.5 h and 256 GiB. Nothing higher
    than the baseline bigram was requested before that measurement existed.

    Item 4, the fixed-duration diagnostic, is COMPLETE in its label-free likelihood half: all 20
    cells (five starts x four fitting LMs) finished, each running the accepted 0/1/2/4 repair
    trajectory over the whole 6,414-utterance update fold (584,424 retained audio units) with only
    the fitting LM changed. Per-audio-unit log likelihood at repair count 4, read from each cell's
    own `row_name` and `fitting_lm.identity` fields:

    | start | `legacy-2g` | `matched-2g` | `matched-3g` | `matched-4g` |
    | --- | --- | --- | --- | --- |
    | `controlled/reference` | -5.2736 | -5.2736 | -5.2418 | -5.2201 |
    | `real/espum_seed0_update30000` | -5.3568 | -5.3568 | -5.3653 | -5.3821 |
    | `real/fingerprint` | -5.6262 | -5.6262 | -5.6082 | -5.6079 |
    | `real/pseudo_pair_seed0` | -5.8930 | -5.8930 | -5.8839 | -5.8822 |
    | `real/random_map_seed1000` | -5.6547 | -5.6547 | -5.6255 | -5.6410 |

    These numbers are NOT an order choice and cannot become one. Each column is the likelihood of
    the same audio under a DIFFERENT fitting LM, so the columns are not readings of one model on a
    common scale; a higher-order LM is a larger model, and the plan's own reporting rule -- carried
    in every cell's payload -- says D selects nothing, chooses no order, funds no final refit and
    cannot close the unrun coherent matched-4 arm. The own-minus-donor and descriptive-error halves
    of item 4 are not yet built, and the ordering question is not answerable from this table alone.

    The DECODE and descriptive-error half of item 4 is COMPLETE 2026-08-22
    (`config/sae_1g_h4_context_diagnostic_per.py`, `h4_context_decode.py`): 60 channel adapters,
    60 local decodes and one error read, all finished with no errors. The frozen decoder for all
    five of these starts is the LOCAL decoder -- `H4ProvisionalMaximaJob.ejmy4sdTOcS3` records
    `decoder.kind == "local"` with no lambda, no insertion penalty and no beam on every baseline row
    -- so no beam search, no sequence decode and no decoding 4-gram enter this half. Counts 1/2/4
    decode the repaired tables the 20 repair cells already banked; count 0 is read from the frozen
    1g.2 direct-Q decode, and the read job re-hashes that column in all four fitting-LM positions
    and refuses the grid if they ever disagree. Bed and gold are the 890 selection-role utterances
    of the 1g.2 descriptive read, so the count-0 and `legacy-2g` columns are comparable with
    verdict 21.

    Pooled corpus phone error rate (total edits over total reference phones), 890 selection
    utterances:

    | start | count | `legacy-2g` | `matched-2g` | `matched-3g` | `matched-4g` |
    | --- | --- | --- | --- | --- | --- |
    | `controlled/reference` | 0 | 0.3934 | 0.3934 | 0.3934 | 0.3934 |
    | `controlled/reference` | 1 | 0.3913 | 0.3913 | 0.3886 | 0.3882 |
    | `controlled/reference` | 2 | 0.4042 | 0.4042 | 0.3941 | 0.3930 |
    | `controlled/reference` | 4 | 0.4168 | 0.4168 | 0.4089 | 0.3985 |
    | `real/espum_seed0_update30000` | 0 | 0.8573 | 0.8573 | 0.8573 | 0.8573 |
    | `real/espum_seed0_update30000` | 1 | 0.8579 | 0.8579 | 0.8603 | 0.8600 |
    | `real/espum_seed0_update30000` | 2 | 0.8603 | 0.8603 | 0.8576 | 0.8555 |
    | `real/espum_seed0_update30000` | 4 | 0.8528 | 0.8528 | 0.8466 | 0.8492 |
    | `real/fingerprint` | 0 | 0.8673 | 0.8673 | 0.8673 | 0.8673 |
    | `real/fingerprint` | 1 | 0.8673 | 0.8673 | 0.8673 | 0.8673 |
    | `real/fingerprint` | 2 | 0.8656 | 0.8656 | 0.8649 | 0.8651 |
    | `real/fingerprint` | 4 | 0.8586 | 0.8586 | 0.8557 | 0.8564 |
    | `real/pseudo_pair_seed0` | 0 | 0.9136 | 0.9136 | 0.9136 | 0.9136 |
    | `real/pseudo_pair_seed0` | 1 | 0.8757 | 0.8757 | 0.8774 | 0.8774 |
    | `real/pseudo_pair_seed0` | 2 | 0.8563 | 0.8563 | 0.8585 | 0.8562 |
    | `real/pseudo_pair_seed0` | 4 | 0.8096 | 0.8096 | 0.8114 | 0.8103 |
    | `real/random_map_seed1000` | 0 | 0.9015 | 0.9015 | 0.9015 | 0.9015 |
    | `real/random_map_seed1000` | 1 | 0.9015 | 0.9015 | 0.9015 | 0.9015 |
    | `real/random_map_seed1000` | 2 | 0.9001 | 0.9001 | 0.8985 | 0.8990 |
    | `real/random_map_seed1000` | 4 | 0.8921 | 0.8921 | 0.8868 | 0.8874 |

    The `legacy-2g` and `matched-2g` columns are not a copying error and were checked as such: the
    two decodes are separate artifacts with different content hashes, produced from repaired tables
    with different array hashes, and their decoded phone sequences are nevertheless BYTE-IDENTICAL
    on all 890 utterances in all 15 repaired cells. For contrast, `matched-4g` differs from
    `legacy-2g` on 664, 796 and 849 of 890 utterances at counts 1, 2 and 4 on the reference.

    One design point, recorded because it is the only place the two halves of item 4 could have
    diverged: H4-LM-D freezes the decoder and changes only the fitting LM DURING REPAIR, so the
    local decoder's phone prior stays the accepted `phone_lm.npz` in all twenty columns. That is
    why these cells cannot reuse `H4LocalDecodeJob`, which pins its prior FILE to the channel's
    fitting LM -- the right binding in 1g.2 where the two are one object, the wrong one here. The
    binding is re-aimed, not dropped: the hypotheses still carry the channel's `fitting_lm_sha256`,
    so the frozen scorer still refuses a channel and a decode from different fitting LMs.

    The OWN-MINUS-DONOR half is COMPLETE 2026-08-22 as well, so item 4 is now finished on both
    trajectories the plan asks for. It reuses the same 60 channel adapters plus the 60 decodes above
    fanned into `H4FixedTextScoreJob` at the ten frozen donor assignments (600 jobs), and the
    count-0 column reuses the frozen 1g.2 adapter and its ten already-finished score jobs. The
    statistic is NOT re-derived: `H4ContextOwnMinusDonorJob` calls `compute_selection_aggregate`
    from the 1g.2 selector module, the same function the frozen selection surface calls, so a cell
    here and a cell there are the same statistic by construction. All 80 cells came back eligible;
    no cell was dropped. The reader lives in its own module `h4_context_scores.py` rather than in
    `h4_context_decode.py`, because both modules hash their own source into every job identity and
    adding a reader to the decode module would have moved the hash of all 121 finished decode cells,
    re-run them and orphaned the artifacts this Catalog cites.

    Own-minus-donor (own log probability per own audio unit minus donor log probability per donor
    audio unit, averaged over eligible non-`no_swap` utterances within each split, weighted 432/890
    and 458/890, then averaged over the ten frozen donor assignments):

    | start | count | `legacy-2g` | `matched-2g` | `matched-3g` | `matched-4g` |
    | --- | --- | --- | --- | --- | --- |
    | `controlled/reference` | 0 | 4.2212 | 4.2212 | 4.2212 | 4.2212 |
    | `controlled/reference` | 1 | 4.5828 | 4.5828 | 4.6133 | 4.5886 |
    | `controlled/reference` | 2 | 5.0198 | 5.0198 | 5.0152 | 4.9173 |
    | `controlled/reference` | 4 | 5.8265 | 5.8265 | 5.6251 | 5.3723 |
    | `real/espum_seed0_update30000` | 0 | 3.2048 | 3.2048 | 3.2048 | 3.2048 |
    | `real/espum_seed0_update30000` | 1 | 3.3912 | 3.3912 | 3.3297 | 3.2423 |
    | `real/espum_seed0_update30000` | 2 | 3.6519 | 3.6519 | 3.5188 | 3.3815 |
    | `real/espum_seed0_update30000` | 4 | 4.2613 | 4.2613 | 3.9725 | 3.7627 |
    | `real/fingerprint` | 0 | 10.1520 | 10.1520 | 10.1520 | 10.1520 |
    | `real/fingerprint` | 1 | 6.0444 | 6.0441 | 4.6789 | 4.5984 |
    | `real/fingerprint` | 2 | 5.1106 | 5.1104 | 3.9494 | 3.7163 |
    | `real/fingerprint` | 4 | 4.5317 | 4.5316 | 3.5672 | 3.2584 |
    | `real/pseudo_pair_seed0` | 0 | -0.0272 | -0.0272 | -0.0272 | -0.0272 |
    | `real/pseudo_pair_seed0` | 1 | 0.0266 | 0.0266 | 0.0245 | 0.0244 |
    | `real/pseudo_pair_seed0` | 2 | 0.0506 | 0.0506 | 0.0480 | 0.0510 |
    | `real/pseudo_pair_seed0` | 4 | 0.1437 | 0.1437 | 0.1390 | 0.1404 |
    | `real/random_map_seed1000` | 0 | 10.7753 | 10.7753 | 10.7753 | 10.7753 |
    | `real/random_map_seed1000` | 1 | 7.1353 | 7.1351 | 6.3593 | 6.7122 |
    | `real/random_map_seed1000` | 2 | 5.6890 | 5.6888 | 4.5740 | 4.3967 |
    | `real/random_map_seed1000` | 4 | 4.8425 | 4.8424 | 3.7748 | 3.4494 |

    Not funded and not run: the 81-row controlled library, any selector refit, any order choice, any
    final refit.

15. **1g.9 experiment 1: locate the phone-repair collapse.** For the five 1g.2a starts at repair
    counts 0 and 4, under the accepted two-state topology at the frozen H1 duration p=0.235603 and
    the `legacy-2g` fitting LM, compute the posterior expected symbol-ENTRY distribution `q_bar`
    and the posterior expected rate by forward-backward, the same two statistics from the banked
    frozen local one-bests (no new decode), and each proposed constraint term's gradient. `p_text`
    is the accepted calibration `phone_lm`'s `phone_prior` over the complete 39,630,169-line
    unpaired phone corpus; `r_target` = 0.7644 symbols per retained unit, derived from the frozen
    H1 length-law fit (the memoryless reading `1-p` agrees to four decimals, and 53,498 update /
    5,110 selection forced boundaries at deleted-silence gaps are the one term a healthy posterior
    may legitimately exceed it by). Gradients are reported as `lambda_equal = ||grad L_HMM|| /
    ||grad L_term||` in the `B = softmax(theta)` parameterization. Label-free; selects nothing.
    One job, `H4CollapseLocateJob.gZ9d6e3E7ZGu`, 31 minutes.

    Posterior on the 890 selection utterances, matched to the decode. Residuals are relative to
    `r_target`; `cl0` is the registered clause-0 pattern (posterior within total variation 0.15 and
    rate within 20 percent, while the decode meets fewer).

    | start | count | post TV | post rate res | dec TV | dec rate res | distinct | cl0 |
    |---|---|---|---|---|---|---|---|
    | `controlled/reference` | 0 | 0.0498 | -0.014 | 0.0854 | +0.148 | 37 | no |
    | `controlled/reference` | 4 | 0.0436 | -0.054 | 0.0658 | +0.121 | 37 | no |
    | `real/espum_seed0_update30000` | 0 | 0.0334 | +0.153 | 0.1058 | +0.266 | 35 | yes |
    | `real/espum_seed0_update30000` | 4 | 0.0690 | +0.000 | 0.0847 | +0.222 | 36 | yes |
    | `real/fingerprint` | 0 | 0.1165 | +0.208 | 0.1545 | +0.233 | 39 | no |
    | `real/fingerprint` | 4 | 0.0317 | -0.042 | 0.1357 | +0.209 | 38 | yes |
    | `real/pseudo_pair_seed0` | 0 | 0.0023 | +0.027 | 0.8345 | -0.856 | 3 | yes |
    | `real/pseudo_pair_seed0` | 4 | 0.0120 | -0.001 | 0.6871 | -0.506 | 9 | yes |
    | `real/random_map_seed1000` | 0 | 0.0851 | +0.118 | 0.0291 | +0.263 | 37 | yes |
    | `real/random_map_seed1000` | 4 | 0.0664 | -0.055 | 0.0402 | +0.240 | 37 | yes |

    Update-role posterior (the fold the constrained objective would run on), the two constraint
    terms' `lambda_equal`, and the per-retained-unit log likelihood on each fold.

    | start | count | TV | KL coverage | rate | lam coverage | lam rate | sel LL/unit | upd LL/unit |
    |---|---|---|---|---|---|---|---|---|
    | `controlled/reference` | 0 | 0.0473 | 0.0068 | 0.7563 | 6.068e+06 | 9.440e+07 | -5.6506 | -5.6452 |
    | `controlled/reference` | 4 | 0.0439 | 0.0064 | 0.7208 | 1.223e+06 | 5.283e+06 | -5.2971 | -5.2736 |
    | `real/espum_seed0_update30000` | 0 | 0.0365 | 0.0039 | 0.8825 | 3.296e+06 | 5.820e+06 | -5.7715 | -5.7663 |
    | `real/espum_seed0_update30000` | 4 | 0.0736 | 0.0160 | 0.7624 | 1.034e+06 | 4.040e+07 | -5.3792 | -5.3568 |
    | `real/fingerprint` | 0 | 0.1205 | 0.0449 | 0.9242 | 1.422e+06 | 1.927e+07 | -7.2593 | -7.2930 |
    | `real/fingerprint` | 4 | 0.0352 | 0.0052 | 0.7259 | 1.987e+06 | 1.077e+07 | -5.6502 | -5.6262 |
    | `real/pseudo_pair_seed0` | 0 | 0.0006 | 0.0000 | 0.7861 | 1.454e+08 | 1.532e+07 | -5.9457 | -5.9438 |
    | `real/pseudo_pair_seed0` | 4 | 0.0108 | 0.0008 | 0.7620 | 3.509e+06 | 2.812e+07 | -5.9037 | -5.8930 |
    | `real/random_map_seed1000` | 0 | 0.0809 | 0.0712 | 0.8519 | 2.166e+06 | 7.495e+06 | -6.9734 | -6.9771 |
    | `real/random_map_seed1000` | 4 | 0.0630 | 0.0175 | 0.7178 | 8.085e+05 | 8.302e+06 | -5.6775 | -5.6547 |

    Not funded and not run: 1g.9 experiment 2's constrained refits, experiment 3's unigram-matched
    babble null, and every constrained-arm lambda. Their graph does not exist.

16. **1g.10: the full-model (LM-aware) sequence decode of the audited count-4 channels.** The
    registered prefix-mass beam decoder under the frozen H1 two-state law with the banked KenLM
    phone 4-gram REPLACING the fitting bigram, run on the three audited count-4 channels over the
    12-point grid (lm_scale in {0.5, 1, 2, 4} x insertion penalty in {-2, -1, 0}). Beam 512 is the
    decision beam and runs the full 890-utterance selection role over 32 shards per cell (1,152
    chunk jobs plus 36 merges); beam 256 exists only to supply the adjacent-pair agreement and
    drift columns and runs ONE shard per cell on the planner's budget ruling -- the heaviest
    selection-role shard taken from the measured resource contract's own `shard` block, 27
    utterances, the same index for every cell. The global-beam eligibility flag is read for
    provenance and deliberately NOT applied: measuring the surface that bar was hiding is the
    point. Nothing is refit and nothing is selected; the held-out evaluation stays sealed.

    | row | readable cells of 12 | cells clearing the babble null | best correct-phone fraction (cell) |
    |---|---|---|---|
    | `controlled/reference` (positive control) | 7 | 12 | 0.6010 (lam 0.5, beta 0) |
    | `real/espum_seed0_update30000` | 7 | 12 | 0.1907 (lam 2, beta 0) |
    | `real/pseudo_pair_seed0` (collapsed) | 1 | 12 | 0.1756 (lam 1, beta 0) |

    The LM-blind local decoder on the same channels, for reference (this is what 1g.9 measured):
    `controlled/reference` correct-phone 0.5832 at TV 0.0658 and length ratio 1.1205;
    `real/espum_seed0_update30000` 0.1472 at TV 0.0847 / ratio 1.2224; `real/pseudo_pair_seed0`
    0.1904 at TV 0.6871 / ratio 0.4941 on only 9 distinct phones.

    Beam-stability explanation duty, over all 36 cells (agreement and drift on the probe's 27
    utterances; margins on all 890):

    | quantity | min | median | max |
    |---|---|---|---|
    | one-best agreement, beam 256 against beam 512 | 0.2222 | 0.6111 | 0.8889 |
    | median score margin, nats per retained unit | 1.210e-03 | 4.345e-03 | 1.540e-02 |
    | fraction of utterances in a cell at or below the flat threshold | 0.061 | -- | 0.466 |

    ZERO of 36 cells reaches the registered 0.999 agreement level and ZERO of 36 has a median
    margin at or below the registered flat threshold of 1e-3 nats per retained unit.


## Verdicts

1. **Approach 1: one segment per text symbol is rejected.** It exceeds the registered ratio on all
   eight dev-other cells and on both estimators. This is a result about that channel shape, not about
   every Phase-1 initializer.

2. **WRONG AS AN UNSCOPED CLAIM (old Approach-1 conclusion): independent duration is rejected.** At
   the historical gold-derived duration point it is rejected on every pooled stream, passes for raw
   phones, and is split-dependent for raw characters. Subtracting the measured cross-utterance floor
   does not rescue a pooled cell. This does not fix or reject a duration fitted prospectively without
   labels.

3. **WRONG AS STATED (old Approach-1 two-state conclusion): seven dev-clean cells pass and one is
   indeterminate.** The exception rows were attached incorrectly. All eight dev-other cells pass. On
   dev-clean, five pass, `seg12.5`/characters and `seg9`/phones are indeterminate because the two
   estimators straddle the threshold, and `seg9`/characters is rejected. These are gold-duration
   diagnostics; a prospective duration must be fitted and checked without labels.

4. **Approach 2: the spectral anchor fails its registered gate.** The phone and character text
   controls pass near ceiling, while every audio stream misses the 0.85 accuracy requirement. The
   failure remains valid after the polarity correction.

5. **WRONG / UNVERIFIED (old Conclusions 2, 23, and 24): the silence pre-check demonstrated that the
   eigenvector tracked silence, and later uncertainty tests repaired the pre-check.** No catalogued
   artifact preserves the later permutation or bootstrap outputs, so neither claim can support a
   verdict. This is not load-bearing: every audio stream independently fails the saved accuracy gate.

6. **WRONG AS STATED (old Conclusion 4): the audio partition carries no information about the text
   partition.** The experiment supports only the narrower claim that this fixed binary partition and
   its registered containment/accuracy metrics did not recover a useful correspondence.

7. **Approach 3: the deterministic hard descriptor route also fails.** Its best held-out result is
   energy on `seg16`, 0.8130 mass accuracy versus the required 0.85. The +0.20-over-majority half
   passes on dev-other, but the gate is a conjunction. Both evaluated splits fail.

8. **WRONG AND SUPERSEDED (old descriptor population read):** the first report used all 2,703/2,864
   labelled utterances. That operating point is not comparable with the registered fixed fifth and
   must not support a verdict. The corrected 540/572-utterance reports give the 0.7588–0.8130 range
   in Approach 3. The all-utterance artifacts remain provenance only.

9. **WRONG (old Conclusion 5): the descriptor route replaces the failed spectral route.** That was a
   temporary next-step statement. The hard descriptor experiment subsequently ran; both exercised
   1g.4 routes failed their registered gates. Current funding status belongs in `PLAN_1G.md`.

10. **WRONG / NOT ANSWERABLE (old Conclusion 19): the six-factor soft product failed its registered
    prerequisite.** The implementation counted seven alternative descriptors for one binary target,
    rather than testing six independent memberships. No six-factor channel or prerequisite screen was
    run, so no experimental verdict exists; current funding status belongs in `PLAN_1G.md`.

11. **WRONG AND SUPERSEDED (old E5 endpoint and hard-stop interpretation).** The completed code uses
    retention 1 for the reference and 0 for a random redraw, fits and evaluates on the same
    utterances, and does not run the actual seeds or controls. Its reference endpoint moved from
    0.4865 PER to an LM-selected 0.4589 and then drifted to 0.6699 at step 30; its random endpoint
    moved from 1.0109 to 0.8409. These observations are non-decisive and fire no gate.

12. **Approach 5 establishes the seed-provenance constraint.** The original fingerprint and ESPUM
    artifacts saw the evaluation audio and are transductive rows only. Neither qualifies for the
    held-out gate or can silently inherit its original headline; `PLAN_1G.md` specifies the required
    construction-only operating point.

13. **Approach 5 localizes the preprocessing correction.** The frozen encoder, PCA/K-means, and
    per-utterance pooling need no refit. The proxy-silence mask does: its historical construction saw
    evaluation audio, so the prospective route must learn that mask on update audio and freeze it.

14. **Approach 6 selects the two-state phone channel.** On the construction-only `seg12.5`/phone
    read, the one-state channel is decisively rejected while the two-state channel is admissible under
    both estimators. Freeze `p=0.23560298` (mean duration 1.308221) and the two-state topology for the
    H3/H4 phone path. The raw-character row makes the same topology choice, but does not unblock H6's
    separately gated handoff.

15. **Approach 7 freezes ESPUM seed 0 at update 30,000.** All three full-loss ESPUM generators were
    fitted on the exact 6,414-utterance H1 update population and selected without labels on the
    disjoint 890-utterance H1 selection population. Weighted phone-language-model perplexity (lower
    is better; ordinary perplexity divided by squared emitted-inventory coverage) was 32.5352 for
    seed 0 at update 30,000, 32.5912 for seed 1 at update 38,000, and 33.1554 for seed 2 at update
    34,000. This freezes the seed/update for projection and the later 7,304-utterance construction
    refit; it is not an evaluation phone-error-rate result.

16. **Approach 10 rejects every baseline sequence-decoder setting at the label-free beam boundary.**
    Across all 12 language-model-scale/insertion-penalty settings, no adjacent beam pair from
    64/128/256/512 passes both stability clauses on all three frozen update representatives. H4 is
    not failed by this result: its mechanically admissible baseline surface is reduced to the local
    decoder, which is the only decoder family that may proceed to the selection stage.

17. **Approach 11 persists the five pre-label provisional maxima, and every one of them is a
    local winner.** All 85 starts carry a finite own-minus-donor maximum computed with no label
    read (`contains_labels: false`, `frozen_pre_label: true`), and all 85 winning tuples are
    `decoder.kind = "local"`. `PLAN_1G.md` requires a frozen-versus-next-beam winner audit only for
    a sequence winner and states that a local winner needs none, so the audit precondition standing
    in front of the controlled labels is discharged by construction rather than by running the
    audit. The label-free half of the baseline pre-evaluation-ready condition also reads positive:
    of the four real starts the selector assigns a nonzero repair count to two
    (`espum_seed0_update30000` and `pseudo_pair_seed0`, both count 4). The controlled method-level
    safety read is a label read and has not run, so pre-evaluation readiness itself is not decided
    here.

18. **The frozen own-minus-donor selector FAILS 1g.2: it is not uninformative but systematically
    INVERTED, so H4 is unresolved and no baseline maximum may freeze.** The controlled reference
    labels opened once (`H4ControlledValidationJob.Otv6GBVY8ZUj`, approach 12; 81 controlled arms,
    76 effective channels, 10,000 resamples at seed 20260822). Under `Sel` the reference channel
    scores 5.826478 while the strongest null `controlled/random_map_seed1007` scores 10.807694 and
    the strongest control of either damage family `controlled/map_q05_draw00` scores 10.848025:
    reference minus strongest control is -5.0215 with a one-sided 95 % interval of
    [-5.071922, -4.978244], so the reference loses to a content-free random map by about five
    units and the interval is nowhere near zero. Rank agreement runs the wrong way at every scale
    -- global Spearman(`Sel`, -error) -0.7493 [-0.826461, -0.622575], inside the predeclared
    starting-PER band 0.80-0.93 (48 channels) -0.5125 [-0.702978, -0.252155], and within
    trajectory [-0.928046, -0.789989]. All five NEGATIVE clauses -- the three correlation clauses
    just listed plus the two reference-versus-control comparison clauses -- take the registered
    reading for an upper bound at or below zero; none is merely unresolved. (Corrections
    2026-08-22, verifier hand-back: the point estimate reads -5.0215 not -5.03, and the earlier
    "all four correlation clauses" miscounted -- the artifact carries three correlation clauses
    and two comparison clauses. No interval, verdict or consequence moves.)

    This is a property of the frozen score, not of the reader. It reproduces the pre-label
    cross-start ordering banked before any label existed (the random-map null above the reference).
    The error instrument reads plausibly against the independently banked anchors, with the
    comparison stated at its own operating point: on the 458 dev-other utterances OF THE SELECTION
    ROLE the reference channel's count-0 PER is 0.4149 and `random_map_seed1007` reads 0.9094
    (`controlled_evidence.json`, `per_arm[*].per_by_count_and_split`), against SAE_1f's
    memoryless-oracle-map 0.4148 and random-map-control 0.8946 computed on the DISJOINT
    572-utterance dev-other evaluation fifth. Different utterance sets and, for the null, a
    different draw, so this is corroboration that the instrument measures ordinary phone error in
    the expected regime -- it is NOT an identity check, and the closeness of 0.4149 to 0.4148
    carries no more weight than that. (Correction 2026-08-22, verifier hand-back: the first
    version of this paragraph cited both numbers as an "essentially exact" match without naming
    either utterance set, and they existed in no artifact field because I had computed them in an
    unregistered console command. The split-resolved PER is now produced by the job itself and the
    read was rerun to bank it; every other number in this verdict was reproduced identically under
    the fixed seed.) So the selector is measuring something real and ranking it upside down.

    The two margin clauses pass and do not rescue it: mean selection regret is
    [0.015849, 0.020977] and selected-minus-count-0 is [-0.006181, 0.004722], both inside the 0.05
    margin. A small regret is what an inverted score looks like when the counts within one channel
    differ little -- it constrains the damage per pick, not the validity of the ranking.

    Consequence, taken from the pre-registered gate and not softened after the fact: `Sel` has
    failed, so H4 has no selector, likelihood cannot rescue it, and no contrastive update may be
    invented after labels are read. The 85 provisional maxima stay frozen and unreranked, no
    `H4SelectorFreezeJob` is built, and the 7,304-ID and 4,455-ID final refits and the 1,112-ID
    evaluation stay closed. The failure closes the tested score/channel-shape/decoder/
    representation combination only -- it does not close all Phase-1 initializers, and it is a
    decision not to fund this selector rather than a finding that repair cannot work.

19. **The count method-level safety read PASSES: every nonzero repair count is safe at the
    controlled operating point.** On the reference start's local decoder the paired differences are
    PER(1)-PER(0) [-0.003457, -0.000787], PER(2)-PER(0) [0.008906, 0.012588] and PER(4)-PER(0)
    [0.021156, 0.025474], all with an upper bound below the registered 0.05 margin, so all three
    are SAFE and the read is PASS (reference PER by count 0.3934 / 0.3913 / 0.4042 / 0.4168; count
    1 is a slight improvement). The H4-LM trigger therefore does NOT fire: it is fired by a count
    read that finds no safe count, and this one found three. Read with verdict 18 this is the
    informative split -- repair itself is not what failed at this operating point; the score that
    was supposed to choose among repairs is. Baseline H4 is still not pre-evaluation-ready,
    because that condition needs the safe nonzero count AND a valid frozen selector, and the
    selector is negative.

20. **The sequence family is UNRESOLVED, and untested rather than failed.** No mechanically
    eligible sequence tuple exists to test, because the baseline global beam table ruled all 12
    grid points ineligible (verdict 16). This blocks nothing at this boundary: every provisional
    maximum is local, so the registered local-winner exemption applies, and the sequence verdict
    would only have bound a sequence winner.

21. **DESCRIPTIVE (selects nothing): on plain PER over the selection role, no real seed beats a
    content-free control, and the best number in the whole table belongs to a control.** Read
    under approach 13's registered rule -- these values may not pick a seed, a count or a setting,
    and they do not touch the closed 1g.2 gate. Pooled PER over the 890 selection utterances, best
    count per row, with dev-other beside it (458 utterances of the selection role): ESPUM
    seed-0/update-30,000 0.8528 pooled / 0.8624 dev-other at count 4; fingerprint 0.8586 / 0.8691
    at count 4; pseudo-pair 0.8096 / 0.8105 at count 4; random-map 0.8921 / 0.9022 at count 4. At
    count 0 the same rows read 0.8573 / 0.8673 / 0.9136 / 0.9015 pooled.

    Two of these four rows are content-free CONTROLS, not candidate seeds -- pseudo-pair and
    random-map -- so the comparison that matters is candidates against controls, and it does not
    favour the candidates. The pseudo-pair control moves furthest under repair (0.9136 to 0.8096
    pooled, -0.104 from count 0, the largest movement of any row) and ends BELOW both candidate
    seeds; ESPUM and fingerprint move by -0.0045 and -0.0087. So at this operating point a
    content-free control's repaired output has lower phone error than either real seed's best,
    which is the opposite of what a content-bearing seed should show. The candidate-versus-
    random-map margin at best count is 0.0393 for ESPUM and 0.0335 for fingerprint, in the
    neighbourhood of the historical 0.0365 recorded for selected ESPUM over the stronger control.

    Provenance and limits, so this is not over-read: these are selection-role numbers on the full
    890-utterance selection bed (432 dev-clean + 458 dev-other; the artifact's own
    `selection_ids_count`), while the banked SAE_1f anchors (ESPUM 0.8580,
    fingerprint 0.8809, pseudo-pair 0.9239, random-map 0.8946) were computed on the DISJOINT
    572-utterance dev-other evaluation fifth at their own historical transductive operating
    points; the rows are in the same regime but are not the same measurement, and no held-out
    number exists here because the 1,112-ID evaluation stays sealed. Nothing in this verdict
    changes verdict 18: the gate closed on the selector, and this read neither rescues nor
    re-argues it.



22. **A14: exact order-4 repair CLEARS the measured resource gate, with a wide margin.** On the
    heaviest of the 32 update chunks (chunk 2, 19,515 retained units) one exact order-4 E-step
    takes 49.12 s and peaks at 0.67 GiB, so a full repair curve out to count 4 is requested at
    1 h and 2 GiB against limits of 11.5 h and 256 GiB. Order 3 costs 0.05 s and 0.17 GiB per
    probe utterance against order 4's 0.90 s and 0.67 GiB. Measured reachability is 60,879
    histories and 2,435,160 arcs at order 4 and 1,560 histories and 62,400 arcs at order 3 --
    one below the `1+39+39^2+39^3` and `1+39+39^2` bounds in each case, the all-BOS history
    being unreachable once the first phone is emitted. This licenses item 4 to run each
    (start, fitting LM) curve in ONE process over the whole update fold rather than as 32
    sharded jobs: 32 chunks x 49 s x 6 E-steps is about 2.6 h, inside the 11.5 h clamp.
    Artifact `H4ContextResourceGateJob.HA1vzRL7MEAz`. This is a statement about affordability on
    this machine, not about repair.

    CORRECTION 2026-08-22, the request clause only: the "1 h" first reported was the PER-SHARD
    request, and item 4 runs the whole fold in one process, so it would have asked for a
    thirty-second of what it needs and died at the wall with nothing saved -- these jobs do not
    resume. The gate now emits both figures and a separate `whole_fold_verdict`; on re-measurement
    the heaviest chunk read 50.82 s, giving 1 h per shard and 5 h whole-fold at 2 GiB, both PASS,
    with 6.5 h of headroom. Memory does not scale with the fold, because shards are processed one
    sub-batch at a time. The PASS verdict and every other number above stand.

    Second correction, same date, the probe timing only: the order-4 probe cells read 0.9366-0.9499 s
    in the artifact as it now stands, not the 0.90 s above (verifier). A few percent of node-to-node
    variation on a sub-second cell changes nothing, because the request is sized from the chunk
    rerun and not from the probe.

23. **A14: the smoothing bridge is empirically NULL, so an order-3 or order-4 difference is
    attributable to order rather than to smoothing.** On the label-free update-fold likelihood,
    add-one `legacy-2g` and matched modified-Kneser-Ney `matched-2g` agree to within 4e-5 per audio
    unit on all five D starts at all four repair counts, so rounded displays can still differ in the
    fourth decimal; the largest disagreement anywhere is `real/fingerprint` count 0, -7.293028
    against -7.293065, a gap of 3.7e-5. (Corrected 2026-08-22: the first wording said "agree to four
    decimals", which its own example contradicts.) This is the expected result rather
    than a surprise: smoothing only matters where counts are sparse, and a 39-phone inventory over
    39,630,169 lines of text leaves no sparse bigram, so both estimators sit on essentially the
    maximum-likelihood bigram. It matters anyway, because the plan's whole reason for building a
    matched family was that raising the order alone would confound order with smoothing -- and the
    confound turns out to be negligible at this corpus size. SCOPE: this is the label-free
    likelihood half of item 4 only, on the update role. It says nothing about which order decodes
    better, and the own-minus-donor and descriptive-error halves are not yet run. Artifacts are the
    20 `H4ContextRepairJob` cells under `work/speech_llm/sae/h4_context_diagnostic/`.

24. **A14: on descriptive phone error rate the matched order-3 and order-4 fitting models are not
    a usable gain over the baseline bigram at this operating point, and the smoothing bridge is
    null here too.** Bed: the 890 selection-role utterances (432 dev-clean, 458 dev-other), plain
    pooled corpus PER against the same gold as verdict 21, at the frozen local decoder with the
    accepted phone prior held fixed across every column. (a) `legacy-2g` and `matched-2g` decode to
    byte-identical phone sequences in all 15 repaired cells, which is the decode-side counterpart
    of verdict 23. (b) On the prospective reference, higher fitting order reduces the DAMAGE repair
    does rather than producing a gain: every fitting LM is worse at counts 2 and 4 than the
    unrepaired 0.3934, and order 4 shrinks that loss from +0.0234 to +0.0051 at count 4. (c) On the
    four real H3 starts, count-4 repair helps under every fitting LM (largest,
    `real/pseudo_pair_seed0`, 0.9136 to 0.8096), but changing order on top of that moves PER by at
    most 0.0062 in either direction and does not move it consistently: at count 4, order 4 beats
    the baseline bigram on espum (-0.0037), fingerprint (-0.0022) and random_map (-0.0047) and
    loses on pseudo_pair (+0.0006), while order 3 beats order 4 on three of the four. Every real
    start stays in the 0.81-0.91 PER band under every fitting LM, so the start dominates the
    fitting order by an order of magnitude. SCOPE AND STATUS: this is descriptive and reads labels.
    PLAN_1G 1g.2a Gate says in as many words that perplexity and PER cannot select order, so this
    verdict does not choose a fitting LM, does not authorize the coherent matched-4 arm (item 5) and
    does not close it; the own-minus-donor half of item 4 is still unrun. Artifacts: the 60
    `H4ContextLocalDecodeJob` cells and `H4ContextDiagnosticPerJob.IYHS4cX3j3XV` under
    `work/speech_llm/sae/h4_context_decode/`.

25. **A14: item 4 is COMPLETE on both trajectories, and its two halves DISAGREE about fitting
    order, so the arm answers no order question.** Same bed and operating point as verdict 24, plus
    the label-free own-minus-donor statistic on all 80 cells, every one eligible. (a) The smoothing
    bridge is null on this statistic too: add-one and matched Kneser-Ney at order 2 differ by at
    most 2.7e-4 anywhere (`real/fingerprint` count 1), against cell-to-cell spreads of whole units.
    It is small-but-nonzero here where the decoded sequences were byte-identical, which is the
    expected asymmetry -- the score reads the channel, the decode reads only its argmax. (b) Raising
    the fitting order LOWERS own-minus-donor at count 4 at every one of the five starts, by -0.0033
    (`real/pseudo_pair_seed0`), -0.4542 (reference), -0.4986 (espum), -1.2733 (fingerprint) and
    -1.3931 (random map). Higher is what the frozen selector maximizes, so on the label-free
    statistic the higher-order fitting model looks WORSE, and by a margin far larger than the
    smoothing bridge. (c) That is the opposite direction from verdict 24, where order 3 or 4 gave a
    small phone-error improvement on three of the four real starts. The two halves are not merely
    differently sized, they point opposite ways: within a start, the rank correlation between the
    two over the twelve repaired cells is +0.917, -0.629, +0.822, -0.993 and +0.907, and +0.112
    pooled over all 60 -- no consistent sign. (Corrected 2026-08-22 after a verifier hand-back: the
    first printed values +0.902, -0.636, +0.734, -0.979, +0.853 and +0.110 rested on no artifact and
    used positional instead of average ranks for ties. The values above are Spearman with
    average-rank ties on the unrounded banked numbers, computed and banked by
    `H4ContextAgreementJob.zd6RBdYcvzti`, which carries the convention in its own output; the
    qualitative picture and this verdict's conclusion are unchanged.) This reproduces the 1g.2 finding that the
    own-minus-donor selector is inverted rather than uninformative, now with fitting order as the
    moving coordinate instead of the arm, and it is why the label-free half cannot be read as an
    order preference in either direction. CONSEQUENCE AND SCOPE: item 4 is complete and its answer
    is that a fixed-duration diagnostic at this operating point does not identify a better fitting
    order -- the label-free statistic and the error rate disagree, and PLAN_1G 1g.2a already forbids
    the error rate from selecting order. This closes no method: per the gate, "a negative
    fixed-duration result cannot close the coherent higher-order method", so the unrun coherent
    matched-4 arm (item 5) is untouched, neither authorized nor refused, and remains the planner's
    call. Artifacts: `H4ContextOwnMinusDonorJob.SygqXhY8F2Qt` under `work/speech_llm/sae/h4_context_scores/` and
    the 600 `H4FixedTextScoreJob` cells it reads.

26. **A15: the phone-repair collapse is NOT in the posterior the 1g.9 constraints would act on --
    at count 4 every one of the five starts already satisfies both proposed targets.** On the 890
    selection utterances at repair count 4 (the planner's pre-stated clause-0 decision read),
    posterior total variation to `p_text` runs 0.0108 to 0.0736 against the 0.15 criterion and the
    posterior rate residual runs -5.5 to 0.0 percent against the 20 percent band -- all five starts
    pass both, including the one whose decode collapses. The constraint gradients say the same in
    their own currency: `lambda_equal` is 8.1e+05 to 1.5e+08, so either term would need a weight
    six to eight orders of magnitude above the likelihood's scale before the optimizer could feel
    it, which is what a penalty on an already-satisfied quantity looks like. This is the registered
    clause-0 off-ramp condition, and the ruling on it is the planner's.

27. **A15: the collapse is DECODE-RESIDENT and specific to ONE start.** At count 4, decoded total
    variation to `p_text` is 0.0402 to 0.1357 for four of the five starts, which emit 36 to 38 of
    the 39 phones; `real/pseudo_pair_seed0` emits 9 and sits at 0.6871, with a decoded rate 50.6
    percent BELOW `r_target` while the other four run 12 to 24 percent above it. At count 0 it
    emits 3 phones at total variation 0.8345 and 85.6 percent below target. Its AH overproduction
    against `p_text` is +0.835 at count 0 and +0.415 at count 4, beside the 1g.2 audit's
    +0.417 for the same cell.
    CORRECTION 2026-08-22: "independently reproducing the audit's figure" overstated it. The +0.415
    here is an excess over `p_text`, the audit's +0.417 an excess over the gold 890 unigram -- two
    different references agreeing to 0.002, which is corroboration, not identity. Every other
    number in the verdict stands. So the "babble" the 1g.2
    audit characterized is a property of one start under the frozen local decoder -- a per-unit
    argmax over `Q * prior` with run collapse, which consults neither the fitting language model
    nor the duration law -- and not a property the repair objective produces: the very same
    emission table yields a healthy posterior when the language model and duration law are summed
    over.

28. **A15: a near-zero posterior total variation is NECESSARY but NOT SUFFICIENT for a healthy
    posterior, and this table contains the counterexample.** `real/pseudo_pair_seed0` has the
    LOWEST posterior total variation of all ten cells (0.0006 at count 0, 0.0108 at count 4) and
    the worst decode. Its per-retained-unit log likelihood is also the worst of the count-4 column
    (-5.8930 update, against -5.2736 to -5.6547 for the other four) and it barely moves from count
    0 (-5.9438), while `real/fingerprint` moves -7.2930 to -5.6262 and `real/random_map_seed1000`
    -6.9771 to -5.6547. An emission table that carries little audio information leaves the
    posterior to fall back on the fitting language model's own marginal, which IS approximately
    `p_text` -- so the clause-0 statistic is satisfied most easily by the least informative
    channel. Any reading of clause 0 must carry the likelihood column beside the divergence.

29. **A15: decoded unigram distance to `p_text` does not discriminate, because the registered
    random-map control passes it best.** `real/random_map_seed1000` has the SMALLEST decoded total
    variation of all five starts at both counts (0.0291 and 0.0402), better than the reference
    (0.0854, 0.0658). Bearing on the pre-registered 1g.9 gate: clause 1's readability criterion
    (decoded total variation <= 0.30) is met by a control that carries no phonetic content, so
    clause 1 admits cells rather than evidencing them, and clause 2's comparison against the babble
    null is doing all the discriminating work. Reported for the planner; clause 1 is
    pre-registered and I am not proposing to change it.

30. **A16: 1g.10's TABLE IS BLOCKED BY ITS OWN PRE-REGISTERED EXPLANATION DUTY -- the beams
    disagree while the score margins are wide, which is the decoder-defect branch, not verdict
    28's flat-score branch.** The duty was written into the producing module before any statistic
    of this job existed: tiny margins where instability is measured confirm verdict 28's mechanism
    (an uninformative channel rides the language-model marginal, so hypotheses tie), whereas wide
    margins with persisting instability indicate a decoder defect and BLOCK any reading of the
    cells. The numbers land squarely on the second branch. Zero of 36 cells reaches the registered
    0.999 adjacent-beam agreement level (min 0.2222, median 0.6111, max 0.8889), and zero of 36
    has a median score margin at or below the registered flat threshold of 1e-3 nats per retained
    unit (min 1.210e-03, median 4.345e-03, max 1.540e-02). The job therefore prints "DECODER
    DEFECT SUSPECTED -- no cell of this table may be read until that is explained", and the
    per-cell columns below it are NOT read here. WHAT THIS LICENSES AND WHAT IT DOES NOT: it
    licenses "the registered sequence decoder does not currently produce a readable surface on
    these channels", and it does NOT license any comparison among the three rows, any statement
    about the phone-repair route's viability, or any reading of the correct-phone, total-variation
    or babble-null columns -- which is exactly what the duty exists to prevent. Two facts the
    planner will want beside this, offered as observations and not as rescues: the population is
    heterogeneous rather than uniformly wide (between 6.1 and 46.6 percent of utterances WITHIN a
    cell do sit at or below the flat threshold, so the cell medians are summarizing a mixture),
    and the agreement column is read on the probe's 27 utterances, where a single disagreeing
    utterance is 3.7 percentage points and the 0.999 test cannot be passed by anything short of
    perfect agreement -- though neither observation softens a measured agreement of 0.2222. The
    threshold and the duty are pre-registered and are not adjusted here; whether the suspected
    defect is investigated, and how, is the planner's call.

31. **A16: the positive control's own reading is internally consistent with the blocked verdict,
    and is reported only as a decoder-health observation.** `controlled/reference` produces 7 of 12
    readable cells and its best cell reaches a correct-phone fraction of 0.6010 (lam 0.5, beta 0),
    against 0.5832 for the LM-blind local decoder on the same channel -- so the full model is not
    grossly broken on a channel known to carry content. That is what makes the instability finding
    pointed rather than trivial: the decoder produces sensible content on the control while its
    adjacent beams still disagree on 1 utterance in 3. This entry records the observation; under
    verdict 30 it is NOT a licence to read the control's cells as a result, and no row is compared
    against another.

32. **A16: 1g.10a returns DISCHARGED -- the adjacent-beam instability is pruning reshuffle in a
    correct scorer, not a decoder defect, so the duty's block is satisfied.**
    `H4CrossBeamDefectJob.2pV5rHuWJW3d`. TEST D (determinism): 81 utterances across three
    disclosed cells, decoded twice each through the same entry point, ZERO violations at 1e-12
    nats. TEST U (upper bound): all 1,944 banked winners -- both beams, every probe
    utterance-cell -- rescored against their exact unpruned all-alignments totals, ZERO
    violations at 1e-6 nats, so no pruned sum exceeds the total it is a subset of. The exact
    score is an identity rather than a second implementation: its channel part is the
    `marginal_path_log_score` the chunk jobs already bank, and the language-model and insertion
    terms are alignment-independent and factor out of the path sum. WHAT THIS LICENSES: that the
    beam-512 table may be read AS DESCRIPTIVE with each quoted cell carrying its own 256-vs-512
    agreement beside it. It does NOT license cross-channel comparison, which the registration
    holds pending 1g.10b, and it does not revisit verdict 30, which stands as written for its
    date -- the block was correct until it was explained.

33. **A16: beam 512 is NOT converged, and the exact-currency column says so in the only currency
    where the question is well posed.** Rescoring both beams' winners unpruned and differencing:
    of 384 differing-winner cases exact(w512) beats exact(w256) in 352 and LOSES in 32 (8.3
    percent), median gain +0.0109 nats per retained unit, range -0.0567 to +0.1657. A wider beam
    that were merely searching a superset could never lose, so this is the non-nested kept-set
    effect the scouting round saw, now measured in a beam-independent currency: `prune` ranks
    whole prefixes by their surviving mass, and because those masses grow with the beam the
    ranking reshuffles, so a prefix kept at 256 can fall outside 512's top set. This is the
    quantitative reason 1g.10b's beam-1024 probe exists, and it is context rather than a gate.

## Catalog

1g.10a cross-beam defect diagnostic, verdict DISCHARGED (verdicts 32-33): `work/speech_llm/sae/h4_cross_beam_defect/H4CrossBeamDefectJob.2pV5rHuWJW3d` (`cross_beam_defect.json`, `cross_beam_defect.txt`); code `sae/h4_cross_beam_defect.py`, `scripts/h4_cross_beam_defect_test.py` (9/9) at speech-llm `294c8fc`.

1g.10 full-model decode read, BLOCKED by its explanation duty (verdicts 30-31): `work/speech_llm/sae/h4_full_model_decode/H4FullModelDecodeReadJob.MXhi20TtG1I0` (`full_model_decode.json`, `full_model_decode.txt`); 1,152 beam-512 chunks + 36 merges + 36 single-shard beam-256 probes under `work/speech_llm/sae/h4_decode_jobs/`; code `sae/h4_full_model_decode.py`, `configs/config_sae_1g_h4_full_model_decode_v1.py`, `config/sae_1g_h4_full_model_decode.py`, `scripts/h4_full_model_decode_test.py` (9/9) at speech-llm `359dbeb`.

| evidence | concrete artifact or source |
|---|---|
| 1g.9 experiment 1, locate the collapse (approach 15, verdicts 26-29) | `work/speech_llm/sae/h4_collapse_locate/H4CollapseLocateJob.gZ9d6e3E7ZGu`; code `sae/h4_collapse_locate.py`, `configs/config_sae_1g_h4_collapse_locate_v1.py`, `config/sae_1g_h4_collapse_locate.py`, `scripts/h4_collapse_locate_test.py` (6/6) at speech-llm `d08cd88` |
| 1g.2a item 4 rank agreement between the two halves | `work/speech_llm/sae/h4_context_agreement/H4ContextAgreementJob.zd6RBdYcvzti` |
| 1g.2a item 4 label-free own-minus-donor, five starts x four fitting LMs | `work/speech_llm/sae/h4_context_scores/H4ContextOwnMinusDonorJob.SygqXhY8F2Qt` |
| 1g.2a item 4 descriptive PER, five starts x four fitting LMs | `work/speech_llm/sae/h4_context_decode/H4ContextDiagnosticPerJob.IYHS4cX3j3XV` |
| 1g.2a item 4 repaired-table decodes, 60 cells at counts 1/2/4 | `work/speech_llm/sae/h4_context_decode/H4ContextLocalDecodeJob.*` (60 dirs) |
| 1g.2a fitting LM `legacy-2g` (add-one bigram) | `work/speech_llm/sae/h4_lm_artifacts/H4LegacyLmJob.lZI6TrYdVpev` |
| 1g.2a fitting LM matched-2g / 3g / 4g (unpruned MKN) | `work/speech_llm/sae/h4_lm_artifacts/H4MatchedLmJob.T8ImJUXHaB0l`; `.Jb2m4aM2fUTy`; `.VpVkGMMy7xKW` |
| 1g.2a matched MKN ARPAs, orders 2/3/4 | `work/i6_core/lm/kenlm/KenLMplzJob.ef5FXMvv8af5`; `.tis71OtNidgL`; `.bg0iYRzBQynx` |
| 1g.2a fixed-duration diagnostic, 5 starts x 4 fitting LMs (item 4) | `work/speech_llm/sae/h4_context_diagnostic/H4ContextRepairJob.*` (20 cells) |
| 1g.2a measured resource gate (item 3) | `work/speech_llm/sae/h4_context_resource/H4ContextResourceGateJob.HA1vzRL7MEAz` |
| 1g.0 structure screen, dev-clean | `work/speech_llm/sae/structure_screen/StructureScreenJob.Xyy7r1zTK9hU` |
| 1g.0 structure screen, dev-other | `work/speech_llm/sae/structure_screen/StructureScreenJob.U3QYclOJHgq2` |
| spectral duration-polarity reads, clean/other | `work/speech_llm/sae/spectral_split/SpectralVCJob.AK0OUD2QcPXz`; `work/speech_llm/sae/spectral_split/SpectralVCJob.ZA7uvQ2s7Zta` |
| spectral registered-polarity reads, clean/other | `work/speech_llm/sae/spectral_split/SpectralVCJob.dP9A1geKgd45`; `work/speech_llm/sae/spectral_split/SpectralVCJob.koxlC99UA0t6` |
| raw and pooled unit streams | `work/speech_llm/sae/quantize_states/MergeUnitsPklJob.ncxcd3vouD5E`; `work/speech_llm/sae/repr_pool/SegmentPoolUnitsJob.IHRNqQfnxrQ3` |
| frozen encoder states and train-fit quantizer | `work/speech_llm/sae/av_states/AvStatesJob.c4Ak1rACchRC`; `work/speech_llm/sae/quantize_states/QuantizeStatesJob.FWpGhC941JMi` |
| historical full-bed silence-delimited stream | `work/speech_llm/sae/lexfree_match/UnitWordStreamJob.eIxgmMh99RSE` |
| phone text (`T_phi`) | `work/i6_experiments/users/wu/experiments/posterior_hmm/data/phon_lm/TextToPhonemeJob.THKMON3k9LJQ` |
| normalized character text | `work/i6_core/tools/download/DownloadJob.g4jClO48cAvP` |
| descriptor dump | `work/speech_llm/sae/descriptors/UnitDescriptorsJob.cSmt6LY5WVOu` |
| canonical descriptor read, dev-clean 540 | `work/speech_llm/sae/descriptors/UnitClassReadJob.m0usvL4Oxlv2` |
| canonical descriptor read, dev-other 572 | `work/speech_llm/sae/descriptors/UnitClassReadJob.kvZb0zdRznOY` |
| superseded all-utterance descriptor reads | `work/speech_llm/sae/descriptors/UnitClassReadJob.yeB6P7J3rdwz`; `work/speech_llm/sae/descriptors/UnitClassReadJob.FIwiUeQ5bgGv` |
| descriptor audio manifest | `work/speech_llm/sae/gua_jobs/GuaAudioManifestJob.rdVx8r37h78h` |
| completed exploratory E5 rehearsal | `work/speech_llm/sae/seed_basin/SeedBasinJob.Zm3EuTveSGBL` |
| original fingerprint reads, clean/other | `work/speech_llm/sae/fingerprint_match/FingerprintMatchJob.O4dpJTesB66u`; `work/speech_llm/sae/fingerprint_match/FingerprintMatchJob.MHmUIV85g8Ry` |
| selected ESPUM checkpoint | `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.lALR9ldNG8f1` |
| completed but invalid H1 read | `work/speech_llm/sae/channel_h/Phase1gH1Job.Bz5bcz5grt8B` (runtime source was not frozen; see verifier feedback) |
| accepted construction-only H1 read | `work/speech_llm/sae/channel_h/Phase1gH1Job.HbxKiuBTJ8aN` (`phase1g_h1.json`, source snapshot, and progress log) |
| H2 common channel and bounded raw-character fitting | commits `0556513`, `80408cf`; `src/speech_llm/sae/channel_h.py` |
| H3 role manifest and valid frozen fingerprint fixture | `work/speech_llm/sae/h3_jobs/H3RoleManifestJob.3hl4qCJKKlUN`; `work/speech_llm/sae/lexfree_match/LexFreeMatchJob.lQqgQGjbVe6A`; `work/speech_llm/sae/h3_jobs/H3FrozenFingerprintFixtureAssertJob.Vv2w4KN5173K` |
| invalid H3 prospective stream (quarantined) | `work/speech_llm/sae/h3_jobs/H3MaskedEspumStreamJob.423jpYfDsDkM` |
| invalid H3 prospective initializer batch (quarantined) | `work/speech_llm/sae/h3_jobs/H3InitializerJob.mV1ulU6v75Zr`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.wuJrQHnNPq9m`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.lQQcUgcUry7z` |
| invalid H3 prospective ESPUM batch (quarantined) | `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.nJBsngMMc59s`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.CCohNXNZgPsX`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.jU2ODb1ahIlK`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.F0AcQFAYmv0E` |
| corrected H2/H3 core implementation | commits `8c8eec5`, `5be4263`, `3ba4917`, `04a0b3a`, `925c0be`, `4ba0f32`, `88762f2`, `bda896a`; `src/speech_llm/sae/channel_h.py`; `src/speech_llm/sae/channel_decode_jobs.py`; `src/speech_llm/sae/h3_projection.py`; `src/speech_llm/sae/h3_resume_equivalence.py`; `src/speech_llm/sae/espum_jobs.py` |
| corrected H3 pooled-run stream and calibration starts | `work/speech_llm/sae/h3_jobs/H3MaskedEspumStreamJob.GqAphDUVZJ7f`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.6ifXwi6C9o4b`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.wP5OnAoxzDow`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.gNAARAXeogOt` |
| corrected H3 ESPUM calibration fan-out and frozen pick | `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.97FwGhhItdpO`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.eQyuM6m4rPX2`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.lk3V9mM67j0m`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.h4LngSZ4YvKL`; `work/speech_llm/sae/h3_jobs/H3EspumPickJob.ezmw64E1JwzI` |
| selected calibration projection and GPU resume-equivalence evidence | `work/speech_llm/sae/h3_projection/H3CalibrationEspumProjectionJob.s4GWy36bdWxZ`; `work/speech_llm/sae/h3_resume_equivalence/H3EspumResumeEquivalenceJob.yL2E4UjTDxQ6` |
| live H2 count-0 channel snapshot | `work/speech_llm/sae/channel_decode_jobs/Phase1gChannelSnapshotJob.TJWNeqBXGjfy` |
| completed H2 decoder timing grid | `work/speech_llm/sae/channel_decode_jobs/Phase1gDecoderPreflightJob.egplrTqzH7Ys` (fastest cell); `work/speech_llm/sae/channel_decode_jobs/Phase1gDecoderPreflightJob.5xQSQqaShtXI` (largest elapsed-time cell) |
| H3 construction-population final initializers | `work/speech_llm/sae/h3_jobs/H3InitializerJob.uKw59MBJC4Hj`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.ABTGA9vIwwI8`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.BS1nPUwf1fel` |
| H3 selected construction-population ESPUM refit and strict projection | `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.t1l7N4lQ9dtY`; `work/speech_llm/sae/h3_projection/H3FinalEspumProjectionJob.PJMwUGUXUb7s` |
| H4 artifact, donor, bootstrap, and gate harnesses | commits `93f6261`, `4e67695`; `src/speech_llm/sae/h4_harness.py` |
| H4 calibration preparation and update-only repair graph | commit `c2e930b`; `work/speech_llm/sae/h4_jobs/H4CalibrationPreparationJob.DPv4aIqwPEzM`; reference `H4RepairJob.x1TyHJMfEVpb`; fingerprint `.iUFh7IwniCMl`; random-map seed 1000 `.Ds0zM1NTY2C1`; pseudo-pair seed 0 `.aeetC3NfgPxB`; ESPUM seed 0/update 30,000 `.ViPSmq4Am8vX` |
| H4 corrected recovery and decoder-resource preflight | commit `436ea50`; update-only reference `work/speech_llm/sae/h4_production_jobs/H4UpdateReferenceArtifactJob.DZa7gIj8rZNj`; Q recovery `work/speech_llm/sae/h4_production_jobs/H4QRecoveryJob.ar34r8ltGTGW`; selection donor `work/speech_llm/sae/h4_production_jobs/H4RoleDonorTableJob.w2RMXcCJyGoy`; update contract `work/speech_llm/sae/h4_decode_jobs/H4ResourceContractJob.kFA99bygctlt`; selection contract `work/speech_llm/sae/h4_decode_jobs/H4ResourceContractJob.kyMk7fwm027C` |
| H4 baseline global-beam boundary | code commit `3de988a`; reducer `work/speech_llm/sae/h4_beam_jobs/H4GlobalBeamTableJob.ro6L8QCnqYpx` |
| H4 pre-label selection surface and provisional maxima | commit `84808a8`; `src/speech_llm/sae/h4_selector_jobs.py`; surface `work/speech_llm/sae/h4_selector_jobs/H4SelectionSurfaceJob.MKHfnUO9XwkU`; maxima `work/speech_llm/sae/h4_selector_jobs/H4ProvisionalMaximaJob.ejmy4sdTOcS3` |
| H5 handoff and H6 character-route interfaces | commit `ce265ce`; `src/speech_llm/sae/handoff.py`; `src/speech_llm/sae/character_route.py` |

| 1g.2 audited provisional maxima (local-winner exemption asserted) | `work/speech_llm/sae/h4_selector_jobs/H4ProvisionalWinnerAuditJob.kBCapQOpk1Hj` |
| 1g.2 controlled validation read (verdicts 18-20; the only label reader) | `work/speech_llm/sae/h4_validation_jobs/H4ControlledValidationJob.Otv6GBVY8ZUj` |
| controlled reference gold phones | `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/eval/GoldPhonesJob.ZGSp0hxyd2YP` |
| 1g.2 descriptive real-seed PER (verdict 21; selects nothing) | `work/speech_llm/sae/h4_real_seed_per/H4RealSeedPerJob.vu6Dp6HkJ2pH` |

## Verifier feedback

- 2026-08-20 — The pre-rewrite theory battery has been audited by provenance and relevance. The
  completed, decision-bearing evidence is now summarized in `PLAN_1G.md`: the raw-versus-pooled
  rate/identity tradeoff, the need for two-state within-symbol structure, the weakness of aggregate
  matching as content evidence, the positional design's rank limitation, the damage from hard unit
  coarsening, and the scoped likelihood/error warning. The generic finite-HMM identifiability theorem
  is literature context only and does not establish identifiability of the live tied duration model.
  The archived corpus-size, pair/triple nullity, moment-sample, repair-basin, decoder-gain, and anchor
  predictions are uncommitted synthetic motivation, not results or gate evidence.

- 2026-08-20 — H1 remains accepted at the exact 6,414/890/7,304/1,112 partition and two-state phone
  choice `p=0.23560298`; do not rerun it. H2 is now closed at the common-engine level: commits
  `88762f2` and `bda896a` require and propagate the same explicit deleted-silence boundary vector
  through repair, scoring, and decoding. The current channel suite passes 23/23, including exact
  boundary-aware repair enumeration. This resolves the previously material mismatch at 53,498 gaps
  affecting 97.71% of the update utterances. The accepted 39-phone by 500-unit snapshot, strict
  evidence/merge checks, 48 timing cells, and eight-alternative output-only cap remain unchanged.

  H3 calibration remains valid: the exact 6,414/890 roles selected full-loss ESPUM seed 0/update
  30,000 at weighted phone-LM perplexity 32.5352; strict projection and GH200 resume equivalence pass.
  Final-refit fingerprint `H3InitializerJob.uKw59MBJC4Hj`, random map `.ABTGA9vIwwI8`, and
  pseudo-pair `.BS1nPUwf1fel` finished on all 7,304 construction IDs. The selected ESPUM construction
  refit `EspumMatchTrainJob.t1l7N4lQ9dtY` and strict final projection
  `H3FinalEspumProjectionJob.PJMwUGUXUb7s` also finished. The projected 500-by-39
  `Q(phone | unit)` and 39-by-500 `B(unit | phone)` are finite and normalized; their manifest binds
  all 7,304 construction IDs, the frozen seed-0/update-30,000 choice, every input, and all eight
  runtime sources. The refit does not hash selection IDs. Never relaunch calibration or this final
  graph.

  H4's first calibration graph exposed two material interface defects before any selection read: its
  donor law did not guarantee the exact support condition `C_d <= N <= T_d`, and its controlled bundle
  omitted the original count-0 `Q(phone | unit)`. Exact `-inf` donor scores must not be clipped, and
  `Q` cannot be recovered provenance-safely from normalized `B`. The corrective directives from this
  read are superseded by the verified 2026-08-21 recovery entry below; no selector, final refit, or
  evaluation result existed to reinterpret.

- 2026-08-21 — The corrected H4 prerequisite graph is independently verified and complete. All 71
  regenerated non-soft controls reproduce their retained count-0 `B` exactly, and all four imported
  H3 pairs reproduce canonical `B` from persisted `Q`; therefore exactly 75 trajectories are reused
  and the ten soft Q-space trajectories are rerun. The graph exposes 85 direct-Q starts and the full
  85-by-4 lossless channel inventory. The production selection donor law yields 513/890 eligible
  sources (235/432 clean and 278/458 other), with 377 explicit `no_swap`; these are construction-time
  facts about the frozen table, not a content result. Resource preflight is exactly 288 probes: three
  representatives by 48 cells on each of update and selection, plus one global worst-cell shard rerun
  and contract per role. Both contracts pass the registered 1.5-times resource rule. The update role
  measured 23,768.19 seconds and 1.043 GiB on its heaviest 19,515-unit shard, yielding a 10-hour,
  2-GiB production request; selection measured 3,069.04 seconds and 0.908 GiB on its heaviest
  2,466-unit shard, yielding 2 hours and 2 GiB. The graph has 821/821 jobs finished and no scheduler
  or problem state. Evaluation audio, labels, donors, decodes, and scoring remain absent.

  The reported selector blocker is **VERIFIED and resolved prospectively**. Independent graph
  reconstruction finds no full-role local or sequence decode, fixed-text score, selector, final refit,
  or evaluation job among the 821 completed prerequisites. The raw scorer interface deliberately
  retains `ell_own`, `ell_donor`, `T_source`, `T_donor`, and phone length without choosing a
  normalization. Unequal lengths are material in the frozen donor table: among 5,130 assignment rows,
  4,803 (93.6%) have a longer donor; the median `T_donor/T_source` is 1.159 and the maximum is 5.25.
  The stored update likelihood has no decoder-family, `lambda`, `beta`, or beam coordinate and rises
  at every retained 0/1/2/4 step in all 85 trajectories; using it would choose count 4 everywhere
  while leaving the rest of the deployment tuple undefined.

  `PLAN_1G.md` now freezes own-minus-donor as the **sole deployment selector**. For tuple `c`, fixed
  source decode `z_ic`, and donor assignment `s`, it is

      Delta_ics = log P_Bc(U_i | z_ic)/T_i
                  - log P_Bc(U_d(i,s) | z_ic)/T_d(i,s).

  Each denominator is that input's own positive retained-unit count after the frozen silence mask.
  `P_Bc` marginalizes the duration/state paths, including their exits, and excludes `G_fit`, `G_dec`,
  insertion penalty, beam score, and posterior terms. A common source denominator or phone-length
  denominator is forbidden: for a content-independent per-unit rate and `T_d > T_i`, either creates a
  positive own-audio advantage from donor length alone, whereas separate rate normalization gives
  zero.

  For each of ten frozen assignments, take equal-utterance clean and other means over the common
  eligible rows, combine them with fixed weights 432/890 and 458/890, then average the ten assignment
  values equally; higher wins. Do not renormalize after `no_swap`, clip non-finite values, or drop
  tuple-specific rows. Beam is not a score-selected coordinate: one global beam per
  `(G_fit,lambda,beta)` is derived on the three frozen representative tables and heaviest update
  shard, adding at most 144 stability cells **per fitting-LM identity**. The audit targets are exactly
  the prospective label-fitted reference plus the ESPUM, fingerprint, random-map seed-1000, and
  pseudo-pair H3 rows: any provisional sequence winner must pass a two-beam audit on all 6,414 update
  IDs. Each changed H3 final-refit table must pass again on its 7,304 construction IDs, and the
  reference on its distinct 4,455 dev construction IDs, before evaluation. Each five-row audit adds at
  most 320 shard cells. A triggered coherent matched-4 arm gets its own representatives/global-beam
  table; baseline evidence transfers only by exact tuple and hash identity. The controlled PER reads issue method-
  level repair and sequence-family verdicts; they never prune or rank an individual real-start count
  or grid point.
  H4 maximizes this point statistic over admissible settings within each fixed H3 start; deterministic
  ties prefer `legacy-2g` when applicable, lower repair count, local decoding, the registered
  `(lambda,beta)` enumeration, and canonical start/seed order. Construction likelihood remains the EM
  objective and a finite-value/manifest health diagnostic only; it is never a selector, fallback, or
  tiebreaker. A failed selector or selected-winner beam audit leaves H4 unresolved without reranking.

  Preserve all 821 completed jobs. Add the bounded representative/global-beam extension and new
  deterministic consumer boundary. Before any controlled label read, generate every controlled score
  plus the reference/four-H3 selection surfaces, persist their provisional maxima, and finish the
  update-role winner audits. Only then open controlled labels to validate the fixed selector and
  method families; the four H3 error rows remain sealed, and a pass freezes the unchanged maxima. A
  triggered H4-LM repeats this label-reader-free boundary before its new controlled verdict. Final-
  refit the four H3 rows on 7,304 construction IDs and the reference on 4,455 dev IDs, build the
  evaluation-release artifact after all required final-table audits, and only then open the 1,112-ID
  evaluation once. Do not edit the source-hash-bound `h4_production.py` or `h4_decode_jobs.py`; add new
  consumer jobs/module/config around their artifacts. No prerequisite repair or resource-preflight
  rerun is required.

- 2026-08-20 — Higher-order `G_fit` is registered as conditional H4-LM work, not H6; H6 remains the
  character route after a valid H5 phone handoff. First complete the corrected baseline bigram H4
  assay prerequisites: mechanics, positive controls, donor-score calibration/correlation, and selector
  validity. The method-specific nonzero-count/update-health outcome is not a prerequisite. A failed
  prerequisite is fixed first and does not trigger an LM arm. Trigger H4-LM before evaluation only if
  those prerequisites pass and either the controlled method-level read finds no safe nonzero count or
  the label-free selector assigns count 0 to both real phone seeds. Controlled labels never attach
  safety to or prune a selected real count. Passing both independent conditions is merely pre-
  evaluation-ready, not a held-out content result. The motivating Ney
  order sweep changes recognition
  order for its learned- and correct-channel rows together, so it does not isolate fitting order or
  predict the project effect.

  H4-LM separates the legacy add-one bigram from a matched unpruned modified-Kneser--Ney 2/3/4
  fitting family while keeping the banked 4-gram decoder fixed. H2-LM must implement exact context-
  state repair, reproduce dense `legacy-2g` and exhaustive tiny matched order-2/3/4 reads, and obtain
  measured trigram/4-gram timing and RSS; the 48 H2 beam-decoder cells cannot size repair. The fixed-
  `p` diagnostic is bounded to the reference and four accepted H3 starts: matched order 2 reads the
  smoothing bridge, order 3 is directional, and order 4 is the proposed context probe. It cannot feed
  H5. Only matched order 4 receives a full controlled-library arm, with a separate H1-LM EOS-
  conditioned duration/topology read on the 6,414 update IDs and its own representative/global-beam
  contract. Label-reader-free jobs freeze the combined provisional maxima before matched-4 controlled
  errors are read; F then repeats the expanded selector and sequence-family gates rather than
  inheriting the latter from baseline. It advances only if matched-4 has a safe nonzero controlled
  point and, independently, ESPUM or fingerprint has an unchanged nonzero combined maximum; otherwise
  evaluation remains closed without label-based count substitution. Only unchanged passing maxima may
  refit/release. Held-out LM perplexity is descriptive and never selects order. This is planned work,
  not a result.

- 2026-08-21 — H4 global-beam boundary VERIFIED; approach 10 and conclusion 16 CONFIRMED. Full
  independent recomputation from the 144 raw `H4SequenceDecodeChunkJob` cells reproduces every
  claim: no `(lambda,beta)` setting has an adjacent beam pair passing both clauses on all three
  representatives; the best worst-representative one-best agreement is 0.7313432835820896
  (lambda=2, beta=-2, pair 256->512) and the smallest worst-representative score change is
  0.005448072638504445 nats per retained unit (lambda=0.5, beta=-2, 256->512), matching the
  logged 0.7313/0.005448; the reducer's own per-setting verdicts agree cell-for-cell with the
  recomputation, no anomalies. Frame verified at source and artifact: `beam_pair_statistics`
  implements exact one-best symbol-sequence equality and the fsum-over-retained-units
  denominator with inclusive 0.999 and strict 1e-4 bounds, and `derive_global_beams` walks
  ascending adjacent pairs requiring EVERY representative, freezing the smaller beam of the
  first passing pair; the three representatives are the frozen resource-contract triple (340
  tables deduplicated to 316 hashes, entropy-sorted, indices 0/157/315:
  `controlled/map_q04_draw03` r0, `controlled/map_q06_draw01` r2, `controlled/soft_q00` r0);
  the shard is the canonical heaviest update chunk `update[2::32]` (201 IDs, 19,515 retained
  units, hash-bound to contract `H4ResourceContractJob.kFA99bygctlt`); zero selection or
  evaluation IDs appear in any cell; every `code_identity` sha256 matches the current checkout,
  and commits `3de988a`/`84808a8` touched neither `h4_production.py` nor `h4_decode_jobs.py`
  (last touched by `436ea50`). Consequence ratified as plan-consistent: all 12 sequence
  settings are label-free ineligible, the baseline surface retains the LOCAL decoder only, H4
  is NOT failed, no provisional sequence winner can exist so the sequence-family gate cannot
  bind at this boundary, and local winners need no beam audit. Reading note: with 201 shard
  utterances the 99.9% clause is effectively 201/201 unchanged one-bests — a property of the
  frozen shard size, not an implementation choice. The `sae_1g_h4_prelabel_surfaces` launch
  condition ("12 grid verdicts verified") is MET.

- 2026-08-22 — Pre-label selection-surfaces launch VERIFIED (speech-llm `84808a8`; manager pid
  3796121 with watcher attached). The graph's new work is exactly 340 `H4LocalDecodeJob` (85
  starts x counts 0/1/2/4), 3,400 `H4FixedTextScoreJob` (ten frozen assignments per tuple), one
  `H4SelectionSurfaceJob` and one `H4ProvisionalMaximaJob`; no sequence decode or merge exists,
  and the config enforces local-only by construction (`_load_completed_global_table` raises
  unless the beam table classifies all 12 points, and an ineligible row contributes no tuple).
  All 966 pre-existing dirs (821 prerequisites + 144 beam cells + the reducer) show zero files
  modified after launch except two startup auto-cleanup re-tars with outputs untouched; nothing
  rerun, no error markers. Label firewall confirmed at source: `h4_selector_jobs.py` contains no
  transcript, edit-count, or evaluation reader; the selector implements Section 4's `Sel`
  exactly (own-minus-donor per-unit contrast, 432/890 and 458/890 weights, ten assignments
  averaged with `math.fsum` in sorted order; beam and likelihood excluded from the tie order);
  the later-boundary freeze/audit classes exist in the module but are not instantiated, and both
  launched selector jobs re-assert their own source identity at run time. `h4_production.py` and
  `h4_decode_jobs.py` remain untouched since `436ea50`. Both log migrations verified: SAE_1g
  Conclusion -> Verdicts byte-identical (16 items, all 8 WRONG markers verbatim), SAE_3E1
  append-only (59-61 added after untouched 1-58) — no plan reference can dangle. Progress at
  ~22:20: all 340 local decodes finished, 1,420+/3,400 scores done, zero errors.
  State corrections handed to the implementer (State is implementer-owned): (i) "the best
  adjacent-beam pair keeps only 62%" is unsupported by the artifact — the binding best
  worst-representative agreement is 0.7313 (best mean/pooled 0.7678); 0.6202 is the best pair of
  the WORST setting (lm_scale 0.5, beta 0), the likely misread; (ii) the launch split "635
  finished / 343 runnable / 3,730 waiting" is the known one-shot console status misreport —
  ground truth 966 finished / 3,742 unfinished (the 4,708 total and "nothing in a problem
  state" are correct); (iii) the cleanup note's "input/ goes" is wrong — `JOB_CLEANUP_KEEP_INPUT
  = True` is effective and cleaned dirs on disk retain `input/`; cleanup keeps `output/`,
  `info`, and `input/`, removes internal `work/`, and tars logs; (iv) trivial: manager start
  21:48 not 21:47, and "two orders above the 1e-4 bound" overstates (smallest
  worst-representative change is 54x). The JOB_AUTO_CLEANUP console fact itself is verified
  correct (`sisyphus/__main__.py:217-218` forces False for every non-manager subcommand;
  `settings.py:238` True; no effective override). One harmless exception: startup cleanup
  failed on `GoldPhonesJob.ZGSp0hxyd2YP` (its `work/` never existed), which keeps a plain
  `finished` marker.

- 2026-08-22 (H4 pre-label surfaces VERIFIED; controlled labels ruled open; two observations
  banked for the validation read). Approach 11 and verdict 17 confirmed against the raw
  artifacts: 85 starts (81 controlled + 4 real, counted from the maxima JSON), all finite, every
  winner `decoder.kind="local"` and eligible, and every winner the true argmax over its four
  tuples (0/85 violations); the five registered cross-start rows match at all printed digits and
  the controlled/real random-map twins are bit-identical at full float64 (shared channel content
  hash, only name-bearing hashes differ). Surface counts confirmed by construction and by input
  symlinks: 340 tuples x 10 assignments = 3,400 `H4FixedTextScoreJob` inputs, 4,080
  global_beam_ineligible entries, 324 controlled within-sequence choices all ineligible — no
  sequence-decoder score exists anywhere in either artifact. `code_identity` sha256 equals the
  committed `h4_selector_jobs.py` at `84808a8` exactly; the maxima file's recorded
  `selection_surface_sha256` equals the surface file's actual digest; `choices_sha256` and
  `expected_tuple_keys_sha256` recompute exactly. Label firewall re-confirmed from the job
  `info` inputs (no transcript/WER/corpus reader among all 4,167 surface inputs) and both
  artifacts carry `contains_labels: false` / `frozen_pre_label: true`. The Section-4 aggregation
  is implemented as registered: fixed 432/890, 458/890 weights over per-split eligible-row
  means, NO renormalization after `no_swap` removal — all 3,400 weighted means and all 340 `Sel`
  values reproduce exactly. Verdict 17's local-winner reading is frame-correct against
  `PLAN_1G.md` ("A local winner needs no beam audit"); the ruling opening the controlled
  reference labels is in `PLAN_1G.md` Status. Both hand-backs CLOSED same day (commit
  `1c0ab7a88`, verifier-checked): the approach-11 coverage wording now states the 513
  donor-eligible contributing sources and the no-renormalization weights correctly, and the q09
  degeneracies are confirmed as construction at source — `Q_LEVELS` ends at 1.0
  (`h4_jobs.py:34`) so level 09 is each ladder's undamaged endpoint; at q=1.0 the map ladder's
  `keep_count` is the whole live set and `assignment[keep] = reference_map[keep]` discards the
  draw (`h4_jobs.py:307-316`), and `canonical_soft_q` early-returns `reference.copy()`
  (`h4_production.py:223-224`) — all three citations verified against the committed source.
  Two observations for the validation stage, no action now: (i) only 79 distinct channel
  content hashes among the 85 starts — `controlled/map_q09_draw00..04` are all one channel and
  `controlled/soft_q09` is bit-identical to `controlled/reference`, both by the ladder
  construction above, and (correction 2026-08-22, third group found in the audited maxima)
  `controlled/random_map_seed1000` is bit-identical to `real/random_map_seed1000`, which is
  what makes 85 go to 79 rather than 80 — so effective independent controls are 76 of 81
  (unaffected by the third, cross-namespace group), which any clustered CI
  or null spread over the controls must
  respect; (ii) the cross-start surface ranks the random-map null 9th of 85 (10.7753)
  and the reference 73rd (5.8265), with the registered five rows at ranks 9/72/73/76/84 — the
  controlled label validation must be read knowing the pre-label `Sel` ordering places nulls
  above the reference, and winning repair counts across all 85 starts are {0: 72, 4: 13} (counts
  1 and 2 never win).

- 2026-08-22 (1g.2 controlled validation read VERIFIED; verdicts 18-20 ACCEPTED on the gate;
  one material hand-back). Code audit (`h4_validation_jobs.py` at `31ea348`, tree matches the
  commit): the label firewall is CONSTRUCTION-time (`h4_validation_jobs.py:691-698`, non-
  `controlled/` keys raise before outputs exist) with a runtime backstop cross-checking each
  artifact's own `arm_name`/`repair_count`/`role` fields; the job recomputes no score and
  reranks no maximum (only writes are its own three outputs); the `Sel`-rebuild guard sits
  before the RNG and hard-fails at 1e-9 (`:456-465`); duplicate collapse is by the artifacts'
  own `channel_array` sha256 (`:329`, provenance to `h4_production_jobs.py:884`); verdict logic
  implements the registered readings exactly (correlation NEGATIVE iff upper95 <= 0, count SAFE
  iff upper95 <= 0.05 matching the gate's "no greater than"; `h4_lm_trigger` fires iff no safe
  count). Test suite re-run by the verifier: 50/50, synthetic-only, no real artifact read.
  Artifact cross-check (`H4ControlledValidationJob.Otv6GBVY8ZUj`, finished 5m34s; audit job
  `kBCapQOpk1Hj`): every clause interval, point estimate, and verdict word in verdicts 18-20
  traces to `controlled_evidence.json` fields and brackets its point estimate; bootstrap
  metadata as registered (10,000 resamples, seed 20260822, three levels); `n_finite` = 10,000
  on EVERY interval block (discharging the audit's no-minimum-finite-gate concern for this
  run); no NaN anywhere; the audited maxima are value-identical to the provisional maxima
  except the schema string plus two provenance keys, zero remappings, all 85 winners
  `decoder.kind = "local"` counted in the artifact itself, and the validation artifact records
  the audited file's sha256. Instrument validity is artifact-traceable WITHOUT the anchor
  sentence: the reference channel reads 0.3934-0.4168 PER by count under the error measure
  while the 48 in-band controls were constructed into the 0.80-0.93 starting-PER band.
  HAND-BACK CLOSED (a501d2fd0 + speech-llm 7146bc6, rerun verified 2026-08-22): verdict 18's
  instrument numbers (count-0 dev-other PER 0.4149; random-map 0.9094) had come from an
  unregistered console command and existed in no artifact field. The job now emits
  `per_by_count_and_split` and `split_sizes` and the read was RERUN at the same hash. My direct
  read of the rerun artifact: ALL previously banked values reproduce identically (17
  interval/point endpoints, the three count intervals, per_by_count, all verdicts, trigger
  False, n_finite 10,000 everywhere), and the anchors are now artifact-traceable — reference
  count-0 dev-other 0.4149 and `random_map_seed1007` 0.9094 on the 458 selection-role dev-other
  utterances (`split_sizes` {432, 458} matching the registered 432/890, 458/890 weights) —
  with the verdict text correctly restating the SAE_1f comparison as corroboration across
  DISJOINT utterance sets, not an identity check. The rerun is accepted ONLY on this pairing:
  nothing had consumed the old artifact (no freeze job exists; everything downstream is closed
  by the gate) and reproduction is bit-identical under the fixed seed — an artifact with any
  consumer would have required a new-hash job instead. Suite re-run by the verifier: 53/53;
  dead code (shadowed `spearman`, `resample_sel`, `has_sequence`) removed in the same commit;
  working tree matches 7146bc6. Wording corrections all absorbed (five NEGATIVE clauses =
  three correlation + two comparison; -5.0215; the third cross-namespace duplicate group).
  Shared-tree item stands, correctly untouched by the implementer (neither of us wrote it): the
  UNCOMMITTED `config_sae_1g_v1.py` edit wiring a second `Phase1gH1Job` with `gold_json` does
  not enter the 1g.2 graph and the accepted H1 stays pinned, but its label-boundary status
  needs its author to explain it — raised to the USER (plus untracked
  `config_sae_3e1_d6_swap_cont_v1.py`).

- 2026-08-22 (user-funded descriptive PER read VERIFIED; verdict 21 accepted; one wording
  hand-back). `H4RealSeedPerJob.vu6Dp6HkJ2pH` finished; the code (`h4_real_seed_per.py` at
  speech-llm `97662ff`, tree matches) implements the registration exactly: construction-time
  exact-key firewall (the four `real/` rows at counts 0/1/2/4, nothing else accepted), runtime
  cross-check of each decode artifact's own `arm_name`/`repair_count`/`role` fields, the
  890-utterance selection-role assertion, the shared verified `pooled_per`/`utterance_errors`
  measure, the reporting rule in docstring AND payload, provenance sha256s, and
  `held_out_evaluation: SEALED` in the artifact; `H4ControlledValidationJob` untouched. All
  sixteen pooled and all quoted dev-other values in verdict 21, every best count (4 for all four
  rows), and both candidate-versus-random-map margins (0.0393/0.0335) reproduce from the
  artifact's own fields exactly; split sizes {432, 458}. Anchor attributions verified in
  SAE_1f.md: n2 is the pseudo-pair null in map form (:91, :459) so 0.9239 is correctly
  attributed; the 0.8809 row is the unary fingerprint candidate (:84, :222, :407); the
  historical "+0.0365" (:218, :397) reproduces as ESPUM's distance to the stronger null n1
  (0.8946 - 0.8580), the same construction as today's margins — SAE_1f's own M1-bar phrasing
  differs, so the margins are read as null-distances, which is what verdict 21 does. Suite
  re-run by the verifier: 58/58. WORDING HAND-BACK (one number): verdict 21's
  "892-utterance-bed subset" — the bed is 890 (432 + 458; artifact `selection_ids_count` 890).
  Verdict 21's framing is correct where it matters: descriptive, selects nothing, the gate
  untouched, and the headline honestly stated — no real seed beats a content-free control on
  plain PER at this operating point, and the table's best number belongs to the pseudo-pair
  control.
  Notes, no action: the bootstrap draws its three levels independently per iteration and shares
  them across arms rather than literally nesting (defensible — utterances are shared across
  channels, and every observed interval is far from its threshold); the winner-audit
  local-exemption loop asserts the five baseline rows, not all 85 maxima (immaterial here — the
  artifact itself counts 85 local winners); dead code in the label-reading module (`spearman`
  defined twice, unused `resample_sel`, dead `has_sequence` parameter) was a review hazard —
  removed hash-neutrally in speech-llm `7146bc6` (this note absorbed 2026-08-22); the suite never asserts an end-to-end overall PASS (moot for this
  NEGATIVE read). Gate consequences are ruled in `PLAN_1G.md` 1g.2 Status (2026-08-22): H4
  unresolved, selector closed on this combination, maxima frozen, refits and the 1,112-ID
  evaluation stay closed, H4-LM not triggered; the direction fork goes to the user.

- 2026-08-22 (1g.2a items 1, 3, and item 4's likelihood half VERIFIED; one process hand-back,
  three wording nits; no banked number moves). Verified READ-ONLY under the harness temp-full
  blocker (no Bash, so no test re-execution; see deferred items). Engine
  (`h4_context_engine.py`): line-level review against the accepted dense law — BOS start into
  sub-state 0, duration moves p into sub-state 1 from BOTH sub-states, exits q*P(j|h) into
  sub-state 0 of the shifted history from both, boundary rule kills duration moves and keeps
  history through the exit contraction, terminal q*P(EOS|h) after the last real frame, per-frame
  scaling — every arc reproduces `repair_hmm`/`marginal_forward_backward`, and the shared
  backward variable is proven by the topology (both sub-states have identical outgoing arcs);
  the base-40 padded history axis with the reachability mask is exact by construction. Cells:
  eight of 20 payloads covering ALL five rows and ALL four fitting LMs reproduce the approach-14
  count-4 table digit for digit (also fingerprint count-0 -7.293028 / -7.293065 behind verdict
  23); every cell banks the same accepted H1 (`5a3ef9de`), unit stream (`59192a3d`), and an
  `input_content_sha256.automaton` equal to the CURRENT manifest's `automaton_sha256` (legacy
  `0aa488aa`, matched-3g `bc176309`, matched-4g `f38eedfc`), so the cells provably consumed the
  post-hardening automata and the caught mid-rewrite ARPA corruption reached no banked number.
  Gate (`resource_gate.json`): verdict 22 and its correction reproduce exactly — 340 candidate
  tables deduplicated to 316, selected indices 0/157/315, probe utterance 2902-9006-0015 (353
  units), heaviest chunk 2 (19,515 units, 201 utterances) re-measured 50.82 s / 0.6734 GiB,
  multiplier 1.5 giving 1 h per-shard and 5 h whole-fold at 2 GiB, `m_steps_run` 0, pruning
  none, `whole_fold_verdict` PASS, reached 60,879/2,435,160 (order 4) and 1,560/62,400 (order
  3), one below the bounds as read. Matched manifests reproduce the approach-14 automaton table
  (histories/arcs/renormalized-mass/normalization error) and the diagnostic config enforces the
  whole-fold gate, exact scope (5x4), alias plus registered output, and gate-read (never
  hand-copied) resource requests.
  HAND-BACK (process, the one material item): the in-flight `legacy-2g` cosmetic rerun
  contradicts approach 14's own recorded principle ("the artifacts were NOT rerun ... to chase a
  cosmetic hash") and, unlike the 1g.2 validation rerun, has CONSUMERS — the five finished
  legacy cells bank `input_content_sha256.automaton_manifest = a01ee7ce...` and `.automaton =
  0aa488aa...`, so a rewrite that adds manifest fields changes the manifest digest (and the
  automaton digest too unless `np.savez_compressed` is byte-stable here), stranding finished
  cells citing content no longer on disk. Park the rerun if it has not finished; if it has,
  verify both output files byte-identical to the banked digests, and on ANY difference restore
  the banked content and move field-name alignment into a new job for future consumers only.
  Either way, reconcile the State bullet ("is rerunning to write the same manifest field names")
  with the approach-14 paragraph and record the final decision in exactly one place.
  Wording nits: (i) verdict 23 "agree to four decimals" is contradicted by its own example
  (-7.2930 vs -7.2931 differ in the fourth displayed decimal; the actual gap is 3.7e-5 — say
  "within 4e-5 per audio unit, so rounded displays can differ in the fourth decimal"); (ii)
  verdict 22's "order 4's 0.90 s" per probe utterance is measured 0.9366-0.9499 s (say 0.95 s);
  (iii) State's item-3 bullet still carries the pre-correction single "1 h and 2 GiB" figure —
  align with the corrected per-shard/whole-fold pair.
  Deferred items DISCHARGED 2026-08-22 evening, shell restored (replaces the deferred list, same
  date, because all four items ran): (a) three suites re-executed green — engine 36/36, LM
  artifacts 36/36, EM driver 45/45; the counts exceed the banked 24/24/24/24/36/36 because the
  scripts grew during the hardening round (mtimes 11:12-11:49, before any diagnostic cell ran at
  12:29+), so the banked counts describe an earlier revision and every current check passes.
  (b) Entropy cross-check confirmed and stronger than State's claim: the accepted decoder
  contract `H4ResourceContractJob.kFA99bygctlt` and the gate agree to all 16 digits on ALL THREE
  probe indices (1.9224904277792318 / 3.06327078252676 / 5.860070935311313), same array sha
  `c6d85886` for the lowest. (c) Legacy-2g rerun timeline verified on disk: rerun finished
  12:25:39, earliest diagnostic cell started 12:29:31 — the "completed before any cell started"
  narrative holds, with under four minutes of margin. (d) Feedback commits pushed. The State
  fan-in inode proposal is ACCEPTED and the PLAN.md storage-placement paragraph amended
  (planner re-measured: 342 input symlinks in the live gate dir and in each of the three cleared
  copies, 1,026 debris); cleared-dir deletion stays the user's call.
- 2026-08-22 (1g.2a item 4 descriptive PER half VERIFIED; verdict 24 ACCEPTED, no hand-backs).
  Everything checked reproduces from the artifacts, not the log: all 80 table cells match
  `H4ContextDiagnosticPerJob.IYHS4cX3j3XV` `per_pooled` at four decimals; the bed is the
  registered one by content hash (`gold_sha256` and `accepted_h1_sha256` both equal the
  verdict-21 artifact `H4RealSeedPerJob.vu6Dp6HkJ2pH`; 890 = 432+458); the count-0 column
  equals the frozen 1g.2 direct-Q PERs for all four real starts and is hash-identical across
  the four fitting-LM positions in the payload. The frozen-decoder claim was verified at its
  source: all 425 decoder blocks in `H4ProvisionalMaximaJob.ejmy4sdTOcS3` are kind local with
  null lm_scale/insertion/beam, and all 60 decode cells carry ONE decoder block and ONE
  `phone_prior_sha256` (`9b4a00f4`), so the prior is fixed across every column as the design
  point requires. Verdict 24(a) was reproduced from the raw hypotheses, not the file hashes:
  legacy-2g and matched-2g decoded sequences are byte-identical in 15/15 repaired cells (their
  differing `hypotheses_sha256` is the re-aimed `fitting_lm_sha256` binding, present as
  described), and the reference-row divergence counts against matched-4g are exactly
  664/796/849 of 890 at counts 1/2/4. Verdict arithmetic checked unrounded: count-4
  order-4-minus-legacy deltas are -0.00365 (espum; the verdict's -0.0037 is the correct round,
  the displayed-table -0.0036 is not), -0.00221, -0.00469, +0.00064, and the largest order
  effect is espum matched-3g -0.00624. Scope discipline is right: the no-selection reporting
  rule is pre-registered inside the artifact with the label-circularity disclosure, held-out is
  SEALED, and the verdict claims no order choice with the own-minus-donor half still open.
  Observation, no action needed: `h4_context_decode.py` has no synthetic suite of its own, but
  it reuses the suite-covered decode kernel (`repaired_b_local_decode`) and PER arithmetic
  (`pooled_per`/`utterance_errors`), and the count-0 byte-equality to the frozen 1g.2 decode is
  a stronger end-to-end anchor for the wiring than a synthetic suite would be.
- 2026-08-22 (1g.2a item 4 own-minus-donor half: verdicts 25(a)/(b) and the frame VERIFIED;
  ONE HAND-BACK on the 25(c) correlation numbers, conclusion unaffected). Verified from the
  artifacts: all 80 table cells reproduce from `H4ContextOwnMinusDonorJob.SygqXhY8F2Qt` as the
  mean over ten `weighted_mean` entries at four decimals; every cell carries exactly ten
  assignments with constant eligible counts 235/278 and `ineligible_cells` is empty; the
  smoothing-bridge maximum is exactly 2.702e-4 at `real/fingerprint` count 1 as claimed; the
  five count-4 order-4-minus-legacy deltas reproduce to the digit; `donor_table_sha256` equals
  the frozen 1g.2 donor table (present in the maxima, winner-audit and selection-surface
  artifacts); `accepted_h1_sha256` equals the PER half's; the module imports
  `compute_selection_aggregate` from the 1g.2 selector module as stated; and the decisive
  count-0 reuse claim is proven at bit level — all five count-0 aggregates occur bit-equal
  inside `H4SelectionSurfaceJob.MKHfnUO9XwkU`'s frozen surface. The one hand-back of this
  round — verdict 25(c)'s six correlation numbers rested on no artifact and reproduced under
  none of nine conventions tried — is ABSORBED and CLOSED same day: the corrected verdict now
  prints +0.917, -0.629, +0.822, -0.993, +0.907 and +0.112 pooled, banked by
  `H4ContextAgreementJob.zd6RBdYcvzti` with the convention stated in its own output
  (Spearman, average-rank ties, unrounded pooled PER against the ten-assignment mean, twelve
  repaired cells per start, count 0 excluded as one artifact in four columns). Verifier closure
  checks, all passing: the banked values equal the verifier's independent computation to full
  float precision; the artifact's `per_sha256` and `scores_sha256` are exactly the sha256 of
  the two half payloads, so the agreement read is content-bound to what it correlates; and the
  correction's stated mechanism is proven rather than asserted — positional (first-occurrence)
  tie-breaking reproduces the six original numbers EXACTLY (+0.902, -0.636, +0.734, -0.979,
  +0.853, +0.110), so the discrepancy is fully explained and both number sets are understood.
  The verdict's conclusion was never in question; verdict 25 is now ACCEPTED in full and item 4
  is closed on both trajectories.
- 2026-08-22 (1g.9 experiment-1 launch VERIFIED; three implementer pins RATIFIED; one clause-0
  reading pre-stated before results). The launch is real: `H4CollapseLocateJob.gZ9d6e3E7ZGu` on
  disk and running under a live engine at check, the experiment-2 constrained arms correctly
  unbuilt until the planner rules on this output. Suite re-run by the verifier: 6/6 (torch
  forward-backward against `channel_h.marginal_forward_backward`, gradients against central
  finite differences, `E[N|T]` against a direct sum). Docstring review (speech-llm `d08cd88`) —
  every pinned constant traces: `p_text` is the `phone_prior` array of the accepted calibration
  over the complete 39,630,169-line unpaired phone corpus, the exact corpus and preprocessing the
  fitting LM is estimated from, unsmoothed; `r_target` is DERIVED from the frozen H1 length-law
  fit exactly as the subphase registers, with the memoryless reading `1-p` = 0.7644 beside it and
  the forced deleted-silence-boundary count reported as the one term a healthy posterior may
  legitimately exceed the target by — RATIFIED, the caveat is real and correctly separated;
  `lambda_equal = ||grad L_HMM|| / ||grad L_term||` in the `B = softmax(theta)` parameterization
  — RATIFIED, it answers the dead-band check and gives experiment 2's lambda magnitude a
  traceable origin; the fold pin (gradients on the 6,414 update role, clause-0 read on the
  matched 890 with the update-role posterior beside) — RATIFIED, it removes a stage/fold
  confound the registered clause 0 had left open. Pre-stated by the planner NOW, before any
  artifact exists (also in `PLAN_1G.md` 1g.9 Status): the clause-0 decision read is the COUNT-4
  row, where posterior and decode read the same repaired emission table; count 0 is context
  only, because there the decode reads the start's direct `Q` while the posterior reads its `B`
  — the asymmetry State itself discloses. Also noted for the clause-0 reading: the frozen local
  decoder is a per-unit argmax with run collapse and consults neither the language model nor the
  duration law, so a decoder-resident collapse is a live mechanism this diagnostic can genuinely
  separate, not a straw man.
- 2026-08-22 (1g.9 experiment-1 result VERIFIED in full; verdicts 26-29 ACCEPTED; clause-0
  ruling issued in `PLAN_1G.md` 1g.9 Status). Every number in both approach-15 tables
  reproduces from `H4CollapseLocateJob.gZ9d6e3E7ZGu/output/collapse_locate.json` under the
  verifier's independent recomputation: all ten (posterior TV, posterior rate residual, decoded
  TV, decoded rate residual, distinct symbols, clause-0 flag) tuples to the printed precision,
  the `lambda_equal` range 8.085e+05 to 1.454e+08, the pseudo-pair decoded AH excesses +0.8345
  and +0.4151, and the per-retained-unit log likelihoods to all four decimals including the
  divisors (60,604 selection / 584,424 update retained units). The pinned constants are stated
  inside the artifact as required: `r_target` 0.7643970 with its derivation text, `p` 0.23560298,
  the memoryless reading, the forced deleted-silence boundaries (53,498 update / 5,110
  selection), and `p_text`'s source with the 39,630,169-line count. The clause-0
  operationalization (posterior passes both targets while the decode meets fewer) is a faithful
  reading of the registered pattern, and the decision does not rest on it: the all-five-pass
  fact plus `lambda_equal` decides. One precision note, non-blocking: verdict 27's "+0.415
  reproduces the 1g.2 audit's +0.417" compares an excess over `p_text` with the audit's excess
  over the gold 890 unigram — two different references agreeing to 0.002, which is
  corroboration, not identity. Verdict 28's counterexample is verified and is the round's
  deepest finding: the lowest posterior total variation of all ten cells belongs to the start
  with the worst likelihood and the worst decode, so the proposed constraint statistic is
  satisfied most easily by the channel carrying the least audio information. Clause 0 FIRED;
  experiments 2-3 stay unbuilt (verified: no constrained-arm graph exists); the direction fork
  is with the user.
- 2026-08-23 (1g.10 launch round; VERIFIED AND ACCEPTED; beam-256 cut ruled in `PLAN_1G.md`
  1g.10 Status). A sampled chunk (`H4SequenceDecodeChunkJob.S25eY8DyW2cx`: pseudo-pair, count
  4, beam 256, lm_scale 1.0, insertion penalty -1, chunk 26) traces every constant to the
  registration: KenLM `CreateBinaryLMJob.hvZoC014xnIe`, H1 `Phase1gH1Job.HbxKiuBTJ8aN`, count
  adapter, contract `H4ResourceContractJob.kyMk7fwm027C`, 2 GiB / 2 h; 1,773 of the 2,304 new
  chunk dirs existed on disk mid-submission at verification time. The docstring pins are
  ratified as registered (imported 1g.9 targets; 1g.2 `edit_distance`; per-cell matched babble
  null, 1,000 draws, seed 42; the two-way explanation duty). The implementer's budget flag was
  correct and material: the planner takes the offered cut -- beam 256 restricted to one fixed
  shard per cell (canonical heaviest selection-role shard, same index everywhere), dropping
  1,152 chunks; surplus beam-256 jobs are deleted per the standing delete rule; beam 512 is
  uncut. The reader's key set re-registration is with the implementer.
- 2026-08-23 later (beam-256 cut VERIFIED EXECUTED; hand-back ABSORBED). On-disk chunk count
  is exactly the predicted post-cut census (1,332 = 144 historical beam-table cells + 1,152
  beam-512 + 36 single-shard beam-256 probes); the probe shard traces to the measured
  contract's own shard block (`H4ResourceContractJob.kyMk7fwm027C`: chunk_index 28, 2,466
  retained units), not a typed number. The implementer's arithmetic correction is accepted:
  the surplus is 1,116 (31 of 32 shards on 36 cells), not the ruling's 1,152, which would
  have deleted the kept probes. Agreement/drift columns carry their 28-utterance support in
  payload, header and `beam_probe_note` as ruled.
- 2026-08-23 (1g.10 result round VERIFIED; the duty's block is endorsed; 1g.10a registered in
  `PLAN_1G.md` 1g.10 Status 2026-08-23 result). Verified on disk: 1,332 chunk dirs with ZERO
  `error.run*` markers (completion proven by the reader having consumed every merge;
  per-chunk `finished` markers live inside `finished.tar.gz` after auto-cleanup);
  `H4FullModelDecodeReadJob.MXhi20TtG1I0`'s payload carries the duty verdict verbatim
  ("DECODER DEFECT SUSPECTED -- ... no cell of this table may be read until that is
  explained"), `beam_probe_utterances` = 27, and the printed grid matches approach 16's table
  (agreement min/median/max 0.2222/0.6111/0.8889; margins 1.210e-03/4.345e-03/1.540e-02;
  zero cells passing either branch bar). The implementer's refusal to read any cell, the
  fenced decoder-health observation (verdict 31), and the facts-not-rescues framing of the
  mixture and 27-utterance observations are all exactly right. The 27-vs-28 correction is
  accepted from the artifact -- the prior round's "28-utterance support" line above is
  OBJECTIVELY WRONG on that count (the ruling's shard index 28 was conflated with the
  utterance count; the probe shard holds 27 utterances). NEXT: the 1g.10a cross-beam defect
  diagnostic (banked data only; step-zero selection-rule question; scoring-determinism and
  pruning-monotonicity tests at 1e-9 nats per retained unit; pre-registered consequences
  including the conditional beam-1024 probe) is registered and ready for the implementer.
- 2026-08-23 (step-zero STOP VERIFIED AND ENDORSED; tests re-registered). The stop was exactly
  the registered behavior. Planner verified both code claims in `channel_h.py`: selection is
  the registered argmax form, but `prune` ranks prefixes by the logsumexp of their surviving
  states and discards whole prefixes -- the banked score is a pruned path-sum and is
  legitimately beam-dependent, so the original TEST A would convict a decoder behaving as
  designed (117 of 588 same-winner scores at 1e-9 is description, not a defect count). The
  scouting anomalies (13 same-sequence scores lower at beam 512; 8 lower total retained
  masses) are consistent with non-nested kept sets under this pruning rule and carry no
  verdict. REPLACEMENT (in `PLAN_1G.md` 1g.10 Status 2026-08-23 re-rule): TEST D
  (bit-determinism, double-decode of three disclosed cells at beam 256, 1e-12 nats) and
  TEST U (banked pruned score <= exact all-alignments forced score of the same sequence
  + 1e-6 nats, for all 972 utterance-cells and both beams' winners -- the impossibility
  bound that also polices the scouting anomalies), plus the exact-currency winner-gap
  distribution as context. Consequences unchanged: any violation blocks; both passing
  discharges the suspicion as a designed-in approximation and unlocks branches (ii)/(iii)
  as registered.
- 2026-08-23 (1g.10a launch AND result round VERIFIED; DISCHARGED; consequences applied in
  `PLAN_1G.md` 1g.10 Status 2026-08-23 discharge). Launch verified: the module docstring
  carries the re-ruled rule verbatim; the exact score is the pre-existing
  `marginal_path_log_score` (the chunk jobs already bank it as `channel_log_probability`),
  so the identity test is not circular -- and the test suite's own load-bearing check (real
  decoder at a prune-nothing beam must reproduce the exact score to 1e-9, all alternatives,
  silence policy on, plus a pruning-bites check against vacuity) is exactly the right
  validation; tests re-run by the verifier, 9/9; inputs trace to the registered KenLM
  `CreateBinaryLMJob.hvZoC014xnIe`, frozen H1 `Phase1gH1Job.HbxKiuBTJ8aN`, and the banked
  chunk artifacts. Result verified from the artifact: TEST D 81 utterances re-decoded, 0
  violations at 1e-12; TEST U 1,944 checks, 0 violations at 1e-6; exact-currency 352
  positive / 32 negative of 384, median +0.0109 nats per retained unit. The report's fences
  (reads no cell, quotes no comparison, authorizes nothing itself) are exactly right. The
  planner applies the pre-registered consequences: beam-512 table readable as descriptive
  with per-cell agreement disclosed; 1g.10b (beam-1024 probe, 36 cells x contract shard 28,
  reader extension, 26-of-27 cross-channel bar) is registered and ready to build.
- 2026-08-23 (1g.10b blocker VERIFIED; ruled OPTION (b) in `PLAN_1G.md` 1g.10 Status
  2026-08-23 ruling). The stop was right and all three load-bearing claims verify in the
  code: `DECODER_BEAMS = (64, 128, 256, 512)` (`channel_h.py`), `cells` is a hashed
  constructor argument of `H4GlobalBeamTableJob` (`h4_beam_jobs.py:399`), and the 1g.10 read
  consumes `out_global_beams` (`config_sae_1g_h4_full_model_decode_v1.py:222`) -- option (a)
  would orphan the banked global-beam table and the discharged read exactly as reported. The
  self-correction on the source-identity guard (runtime drift guard, not a hash input) is
  the right kind of checking. Ruled: option (b) with identity guards -- imports not copies,
  beam hard-pinned to 1024 plus a 512 parity mode, and a mandatory parity cell (probe class
  at beam 512 on the median observed-agreement cell must byte-reproduce the banked chunk's
  one-bests and scores; the 1g.10b reader refuses beam-1024 columns without it). Option (c)
  stays rejected as analyzed.
