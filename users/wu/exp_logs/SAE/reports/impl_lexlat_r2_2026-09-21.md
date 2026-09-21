# SAE 4A lexlat -- round 2 implementation (2026-09-21)

Status: **DONE_WITH_CONCERNS**. Two commits on `haotian_modality_matching_jupiter` in
`recipe/2025-10-speech-llm`: **a05ab16** (the E0 OOM priority insert) and **f0ab314** (round 2).
Nothing launched; no project document edited.

## 1. Priority insert: the E0 CUDA OOM (commit a05ab16)

`lexlat.py`: the candidate expansion is chunked over the DESTINATION axis in blocks of a module
constant `DEST_BLOCK = 1024`, at both copies of the pattern (`_step`, `_step_max`). `m_max` stays
GLOBAL inside a block, which is what makes the change bit-identical: the `logsumexp` / `max` over
the arc axis then covers exactly the same set, in the same order, with the same internal max-shift.
(A per-block `m_b` would NOT be bit-identical: an all-`NEG_INF` row's `logsumexp` is
`NEG_INF + log(width)` and the width would change.) The guard now also checks the quantity that is
actually allocated, `b * n_blk * m_max` per block, against `max_candidates`; at the ceiling cell
(C = 16384, m_max = 6014) `n_blk = 1024` makes the limit ~16x larger than the block, which clears
by about 4x, so the ceiling cell does not raise. The existing arc-count assert is KEPT beside it as
a cheap sanity bound (the debug report recommends it) -- a deviation from a strict reading of
"rather than max_candidates", flagged here.

No `Job.__init__` kwarg and no `__sis_version__` moved: `LexlatCensusJob.RDMTvt4ngAPB` reruns at
its hash (confirmed by the census in section 4).

Tests added to `test_lexlat.py`:
* `test_destination_blocking_is_bit_identical` -- 6 parametrisations (block 1 / 2 / 3, escape on and
  off): `torch.equal` on 16 `LexlatOutput` fields, on `max_multiplicity`, and on
  `lexlat_viterbi`'s `phones` / `words` / `word_ids` / `score` / `ok`. All pass.
* `test_manual_backward_matches_autograd_on_a_PRUNED_lattice` -- the check deferred from round 1
  (review finding C4): 6 parametrisations that really truncate (`pruned_max > 0` asserted), at
  budgets (C, C_esc) in {(3,0), (4,1), (4,2)} x two lengths / temperatures / checkpoint strides.

One cell was dropped from that parametrisation and why: at `C = 2, C_esc = 1` only ONE free
non-escape slot remains, pruning removes every complete path, log Z collapses to the `NEG_INF`
sentinel (-1e30) and the manual backward correctly returns all-zero posteriors while autograd
returns an arbitrary sub-gradient of the sentinel. That is not a backward bug -- forward and
backward agree -- but the posterior identity is undefined there, so the test now ASSERTS a finite
log Z (an invalid premise fails loudly) and runs `(4,2)`, which prunes 0.445 of the mass and passes
at 8.9e-16. Measured for the record at s_len 9: (2,1) log Z = -1e30; (3,0) -4.966; (4,1) -4.761;
(4,2) -4.966; (8,1) -4.336 -- i.e. only `C - C_esc = 1` degenerates.

## 2. Training integration (Design 1-4)

New module `sae/emc/lexlat_train.py` (no banked blank-free module gains a class):

* `lexlat_lambda` / `lambda_schedule` / `assert_ramp` -- the curriculum of Design 3. At on-set 8,
  ramp 3: `[0]*7 + [1/3, 2/3, 1] + [1]*10`; at on-set 5 the ramp is 5, 6, 7.
* `LexlatRuntime` -- a PLAIN object (nothing in `state_dict`) holding the spec, the cached
  resources per (device, dtype), `step_params`, `lattice_loss`, `fd_passes`, `monitors`.
  `step_params` returns `(None, None)` while `lam_lex == 0`, which is the switch the train step
  branches on: before the on-set the arm runs the BANKED `lattice.py` call, not the augmented DP at
  `lam_lex = 0` (amendment 2). `lattice_loss` reproduces `lattice.lattice_loss`' contract and its
  arc-posterior surrogate exactly, with the module's own manual backward (never autograd through
  the lattice).
* `derive_max_candidates` (the census formula with the batch made explicit) and
  `truncate_to_bigram` (the declared E1 fallback, `word_lm_order = 2`).

`prefix_lm/model/definitions/sae_blankfree.py` and `.../train_steps/sae_blankfree.py` gain a
DEFAULT-OFF `lexlat_*` keyword block and a `getattr`-guarded branch. Both are hashed by
`code_object_path` + `hashed_arguments`, not by file content, so the edit is hash-neutral (proved in
section 4). With no lexicon stated -- every banked arm, including one re-importing this tree on a
resubmit -- `lex_params` stays `None`, nothing is imported and every line is the one it was.

**The rate term goes through the same DP after the on-set.** `rate_term._fd_passes` hard-calls
`lattice_forward_backward`; tilting the banked lattice after the on-set would differentiate a
different objective, since `E[N]` is then the lexicon-shaped expectation. The rate TERM is
unchanged (same `lam_rate`, `rho`, denominator, `eps`, central difference), which is how Design 4(a)
is read. Stated as an assumption.

**Log keys** (`ctx.mark_as_loss(..., as_error=True)`, so they appear per sub-epoch in the RETURNN
log and in `learning_rates`): `lexlat_lam`, `lexlat_term_mean`, `lexlat_pruned_mass`,
`lexlat_pruned_p95`, `lexlat_pruned_max`, `lexlat_neg_inf`, `lexlat_escape_phone_frac`,
`lexlat_expected_phones_per_word`, `lexlat_expected_words`, `lexlat_expected_escape_words`,
`lexlat_contexts_per_frame`. The Gate's four engagement clauses read the first, second, third and
seventh / eighth of these.

Two wording points, resolved and named:
* The dispatch says "phones per word of the max-plus path"; Design 4 and the Gate define
  `lexlat_expected_phones_per_word` as the POSTERIOR-expected quantity, which `LexlatOutput` returns
  free and which the [0.7, 1.4] x mean-pronunciation-length band is written against. The
  posterior-expected quantity is what is logged.
* `lexlat_neg_inf` is LOGGED, not raised. Amendment 3 says one `NEG_INF` utterance aborts the arm;
  raising inside the step would kill the other three arms' allocation, so the abort is a reading
  rule on the log, not an exception.

## 3. Deliverables 2 and 3

`sae/emc/lexlat_train_jobs.py`:
* `LexlatEfficiencyProbeJob` (E1). One GH200, `rqmt` cpu 16 / mem 64 / gpu_mem 96. Runs the ARM'S
  OWN training config, so the run shape (88,000 padded frames, max_seqs 128), the laplace ordering
  and the lexicon arguments are the pack's; parameters pinned to the same banked `ctrl_50` ep10
  checkpoint round 1's probes pin. It walks the sub-epoch TWICE (walk 1: shape census, batch count,
  largest padded T; walk 2: the timed steps), restores a deep copy of the cold model and optimizer
  state before every timed step OUTSIDE the timing, and times h2d / forward / backward /
  optimizer step separately. Sampling plan: four windows of `ceil(100/4) = 25` at evenly spaced
  points PLUS the largest-T batch -- never the first 100 steps. Outputs median and mean seconds per
  step, the extrapolated sub-epoch seconds against 601 s, peak GiB, and PASS/FAIL against the bar
  (<= 1202 s, <= 80 GiB), which is written in the class docstring and in class constants.
  Reading convention, stated because the Design does not name the statistic: the gate is read on
  `torch.cuda.max_memory_RESERVED` (the stricter, OOM-relevant number) and
  `max_memory_allocated` is reported beside it.
  Reality check the job reports honestly: a sub-epoch of this bed is ~52-56 batches, far fewer than
  100 steps, so the four windows overlap and `n_measured` < `n_steps`; both are in the json.
* `LexlatWordCountsJob` (word unigram counts and equal-token-mass deciles off the replayed unbiased
  window; a job of its own, because an extra output on the FINISHED `LexiconTrieBuildJob` would move
  its hash) and `LexlatFreqStratPerJob` (per-decile PER; each gold phone inherits the MFA word
  containing its midpoint, the reconstructed gold string is asserted against the banked `gold.json`).

`configs/config_sae_4a_lexlat_pack_v1.py` + shims `config/sae_4a_lexlat_pack.py` (`py`) and
`config/sae_4a_lexlat_e1.py` (`py_e1`). TWO entry points on purpose: E1 gates the pack, so it must
be launchable without the four training arms in the graph.
* Four arms, verified from the resolved configs: `lexlat_20` (on-set 8), `lexshuf_20` (on-set 8,
  `lexlat_shuffled = True`), `lexlat_20_e5` (on-set 5), `lexlat_20_s1` (on-set 8, `random_seed` 1,
  `random_seed_offset` 1000, its own `FlatRecognizerInitJob(seed=1)` -- the seed set read from
  `config_sae_4a_prepro_pack_v1.CTRL_S1_SEED`, never re-typed). All at C = 1024, C_esc = 64,
  lam_lex = 1, ramp 3, word-LM order 3, batch 88,000 / max_seqs 128, schedules
  tau[:4] = [8.0, 5.04, 3.17, 2.0] and lr[:3] = [1e-5, 1e-4, 1e-4] -- the prepro pack's own.
* One constants block (`LAM_LEX`, `MAX_CONTEXTS`, `ESCAPE_BUDGET`, `WORD_LM_ORDER`, `RAMP`), so
  E0's re-declaration of C and E1's two declared fallbacks are a one-line change.
* `ctrl_20` / `ctrl_20_s1` are consumed as FROZEN PATHS resolved to the producing
  `BlankfreeGreedyPerJob` work dirs and pinned by a key carrying a sha of that real path, so a
  repointed prepro output moves this read's hash instead of silently changing its input. All 16
  (2 arms x 4 epochs x 2 splits) exist on disk.
* Registered: 32 greedy PER reads (PER with S/D/I, rate, decode stats) and 32 derangement gaps
  (4 arms x 4 kept epochs x 2 splits), 48 `PairedPerDeltaJob` (6 pairs x 4 epochs x 2 splits) and
  2 `LexlatFreqStratPerJob` (dev-other, ep10 and ep20). Pairs: `lexlat_20` vs `ctrl_20` (primary
  and, at ep4, the cross-pack floor F), `lexlat_20` vs `lexshuf_20` (the null clause and Design 5's
  ep4 identity check), `lexshuf_20` vs `ctrl_20`, `lexlat_20_e5` vs `ctrl_20`, `lexlat_20_s1` vs
  `ctrl_20_s1` (ruling 4's ep4 check), `lexlat_20_s1` vs `lexlat_20`.

## 4. Hash census (the requirement the round turns on)

Job ids of each config's whole graph, computed in a fresh process with the two banked-model files
at HEAD content ("before") and with round 2 applied ("after"):

| config | before | after | verdict |
|---|---|---|---|
| `config_sae_4a_prepro_pack_v1` | 133 jobs | 133 jobs | every id IDENTICAL |
| `config_sae_4a_budget_pack_v1` | 775 jobs | 775 jobs | every id IDENTICAL |
| `config_sae_4a_lexlat_probes_v1` | 3 jobs | 3 jobs | every id IDENTICAL |

The three round-1 probe jobs keep their hashes: `LexiconTrieBuildJob.rlMsnTBSZXsB`,
`LexlatEquivalenceProbeJob.lq61PSAg1DcC`, `LexlatCensusJob.RDMTvt4ngAPB`.

Containment, the other half of the requirement: the new E1 graph is 10 jobs and the new pack graph
189, and NEITHER contains any `lexlat_jobs` job (the trie is a frozen path, the two probes are not
reconstructed) nor the prepro pack's `PackedBlankfreeTrainJob.5EIGJJ1MkcO9`; the 9 jobs shared with
the prepro graph are the bed's own upstream (features, splits, prior, flat init). New job ids:
`PackedBlankfreeTrainJob.XaDOVLQZkzgh`, `LexlatEfficiencyProbeJob.r4Iaa72mU27T`,
`LexlatWordCountsJob.X2YnVYfqN7aV`, `LexlatFreqStratPerJob.u0b3NEWqD9DQ` / `.vVCLDycreVYe`.

## 5. Tests

`test_lexlat.py` 43 passed / 1 skipped (was 31/1; +12 from the two new parametrised tests).
New `test_lexlat_train.py`, 12 tests:
* the pre-on-set bit-identity, at BOTH on-sets and at every sub-epoch before them: the train step's
  own branch versus the banked `lattice.lattice_loss`, `torch.equal` on the loss AND on the
  gradients w.r.t. `log_q` and the segment table;
* the on-set really switches (the branch flips, `lam_lex` is the ramp value, the loss moves), so
  the identity test is not passing because the lexicon is dead;
* the ramp schedule at both on-sets;
* E1's sampling plan (four 25-wide windows at 0/100/200/300 plus the longest index; a 55-batch
  sub-epoch samples every batch exactly once);
* a CPU smoke of the step body E1 times (augmented loss, autograd-checked finite gradients through
  the surrogate, the tilted rate passes, all 11 monitor keys finite);
* the frequency-stratified aligner's S/D/I equals the banked `eval_jobs.edit_counts` on 200 random
  pairs.
Whole suite together with the blank-free modules: **61 passed, 1 skipped**
(`test_lexlat.py`, `test_lexlat_train.py`, `test_blankfree_lattice.py`, `test_blankfree_pack.py`).
`test_blankfree_pack.py` needs `/e/project1/spell/wu24/env/sis_env/bin` on PATH (`black`); without
it it fails in `i6_core` before reaching any lexlat code.

What the tests do NOT cover, stated so the checks are not over-read: no GPU was run, so the
E1 job's RETURNN engine path, the h2d copy, the optimizer step, CUDA peak memory and the real run
shape are unexercised; and loading a config proves the graph builds, never that the training branch
takes effect in a real run.

## 6. Not done / left for the orchestrator

1. The Gate names two further disclosed reads that the dispatch's deliverable list does not:
   the Step 0 prior-gap rerun (`PriorGapAnalysisJob` per arm decode) and the wav2vec-U 2.0 selection
   statistic per kept epoch. Neither is registered in the pack config. Both are cheap CPU reads and
   can be added without moving any arm's hash; proposed for a follow-up round.
2. Design 5's pre-launch check (a) -- "the step-1 loss of `lexlat_20` matches `ctrl_20`'s logged
   step-1 loss to 1e-4" -- is a read off two training logs after the pack starts, not a job; it is
   not automated here.
3. The `_plan` sampling is honest about a sub-epoch shorter than 100 steps but the Design's "100
   steps" is then unreachable; E1 reports `n_total` and `n_measured` rather than choosing a
   substitute.
4. The workspace shims are untracked files in the setup dir (not a git repo), so they are not in
   either commit.

Full paths: `/e/project1/spell/wu24/2026-07-13_unsupervised/recipe/i6_experiments/users/wu/exp_logs/SAE/reports/impl_lexlat_r2_2026-09-21.md`

---

## 7. Round 2b (same day, after the E0 rerun died on the new guard)

**299922f -- the destination block is sized to the frame.** The rerun (Slurm 1922930) failed at
`lexlat.py` `_dest_block`, frame 6 of the ceiling cell: `m_max` = 25,511 arcs on the single busiest
destination, so a fixed 1024-wide block asks for 1 x 1024 x 25,511 = 26,123,264 padded slots against
`max_candidates` = 24,883,200. The same run measured `m_max` between 376 and 25,511, so no fixed
width is right at both ends. The width is now
`max(1, min(DEST_BLOCK, n_new, max_candidates // (B * m_max)))`, computed per frame, with
`DEST_BLOCK = 1024` kept as the CAP; the assert can fire only when a SINGLE destination is over
budget, which no block width could fix, and it then raises with B, `m_max` and the limit. Values do
not move with the width by the same argument as before (every destination keeps its full arc list at
the unchanged global `m_max`). Hash-neutral: no `Job` kwarg, no `__sis_version__`, so
`LexlatCensusJob.RDMTvt4ngAPB` reruns at its hash. Two new tests: the width arithmetic at the census
numbers (975 at `m_max` 25,511; the cap at 6,014 and at 376; `n_new` when smaller; the
single-destination raise) and a spy-checked width-1 run bit-identical to the un-blocked one.

**6cbb3f9 -- a NEG_INF utterance now ABORTS the arm, in the step.** The coordinator is right and my
round-2 reading was wrong. Design 2 (c) / amendment 3 states it without qualification: "the census
and the training log report ... the count of utterances with log Z = NEG_INF (must be 0 in every
sub-epoch; one such utterance is an abort of that arm)". No line of the phase file says otherwise;
the Gate's own abort rule lists NaN, |surrogate| > 100 and the rate clause and does not mention
NEG_INF, but it does not license tolerating it either. My stated reason for logging only -- "raising
would take the other three arms down" -- is FALSE: `pack_jobs` runs each arm as its own `rnn.py`
subprocess on its own GPU, polls them independently and "waits for all of them, and fails NAMING the
arm(s) that failed", so a raise ends exactly one arm. `LexlatRuntime.assert_no_neg_inf` therefore
raises, and only from the on-set on (before it the arm is the banked path and the bed's tolerant
`z_zero` behaviour stands). Baseline measured before arming it: the prepro `ctrl_20` arm logs
`train_loss_blankfree_z_zero_frac` = 0.0 and `dev_..._z_zero_frac` = 0.0 in all 20 sub-epochs
(`PackedBlankfreeTrainJob.5EIGJJ1MkcO9/output/ctrl_20/learning_rates`), so the clause cannot fire
from the bed. The counter stays in the log for the pre-on-set sub-epochs. Tested end to end on
C = 2 / C_esc = 1, a budget that really leaves no complete path.

**Phones per word, confirmed against the phase file's own words.** Design 4 (c): "New monitor
`lexlat_expected_phones_per_word` (posterior-expected phones between word closes) must stay in
[0.7, 1.4] x the lexicon's token-weighted mean pronunciation length". The Gate repeats it:
"`lexlat_expected_phones_per_word` outside [0.7, 1.4] x the build job's token-weighted mean
pronunciation length for 3 consecutive sub-epochs at full ramp -> the anti-deletion guard has fired;
UNINFORMATIVE." What is implemented is that posterior-expected quantity
(`expected_word_phones / expected_words`, per utterance, averaged over the kept utterances of the
batch). The band is [2.505, 5.010] from the Results table's token-weighted mean of 3.579 phones. The
dispatch's "max-plus path" wording is the one that differs from the phase file; the phase file is
what the monitor follows.

Tests after 2b: **65 passed, 1 skipped**. Census re-run after the train-step edit: prepro_pack 133,
budget_pack 775, lexlat_probes 3 -- every job id still identical.
