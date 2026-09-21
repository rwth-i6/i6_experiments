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

---

## 8. Round-2 review fixes (speech-llm `b39ce3c`, 2026-09-21)

The four findings of `review_lexlat_r2_2026-09-21.md`, in one commit (explicit paths, not pushed).

### Fix 1 (BLOCKING) -- the E1 shim exported no `py`

`config/sae_4a_lexlat_e1.py` defined only `py_e1` and `run`; `tools/sisyphus/sisyphus/loader.py`
looks up the attribute literally named `py` and only WARNS when it is missing, so `sis m
config/sae_4a_lexlat_e1.py` would have loaded an empty graph and reported nothing to run. The shim
now defines `def py(): return py_e1()` (and keeps `run = py`). The workspace `config/` tree is not
a git repository, so this file is not in the commit.

VERIFIED by replaying the loader's own lookup (`scratchpad/shimcheck.py`: chdir to the setup,
`importlib.import_module("config.sae_4a_lexlat_e1")`, `getattr(mod, "py")()`, list
`tk.sis_graph.jobs()`): **10 jobs, including `LexlatEfficiencyProbeJob.r4Iaa72mU27T`** -- byte-equal
to the job-id list `py_e1()` produced before the fixes.

### Fix 2 (BLOCKING) -- the destination block is chosen from a BYTE budget

`lexlat._dest_block` took `max_candidates // (B * m_max)`, and `max_candidates` itself scales with
B (`lexlat_train.derive_max_candidates` = arc bound x batch), so the width was B-independent and
the `DEST_BLOCK = 1024` cap bound at the training shape: one block is
`3 x [128, 51, 1024 x 376]` float64, about 60 GiB before the slot columns.

The width now comes from a byte budget: `n_blk = max(1, min(DEST_BLOCK, n_new, DEST_BLOCK_BYTES //
(_EXPANSION_COPIES * B * o_n * m_max * itemsize)))`, with `DEST_BLOCK_BYTES = 8 << 30` and
`_EXPANSION_COPIES = 3` as module constants. The single-destination assert stays and now prints B,
o_n, m_max, the itemsize and the resulting GiB. No kwarg, no `__sis_version__`: values stay
bit-identical (`test_adaptive_destination_block_is_bit_identical_at_width_one` forces width 1 with a
spy on `_dest_block` and compares the full DP output).

`test_adaptive_destination_block_width` now asserts the budget at the TRAINING shape (B = 128,
o_n = 51, m_max = 376: the chosen width keeps the block under 8 GiB and one more destination would
exceed it), at the E0 census frames (m_max 25,511 / 6,014 / 376 at B = 1), that a small `n_new` is
never padded up, that a huge `n_new` is capped at `DEST_BLOCK`, and that a single destination over
budget raises.

### Fix 3 -- E1's PASS statistic and the banked-path ratio

`LexlatEfficiencyProbeJob.run` now measures each planned batch TWICE from the same cold parameters:
once with the lexicon and once with `model.lexlat = None`, which is exactly the banked `lattice.py`
path the train step falls back to. Both numbers are the COMPLETE update (h2d, forward, loss and
manual backward, optimizer step); the loader wait is outside the timed region on both sides and is
reported separately as excluded.

The extrapolation is no longer `median x n_total`: when the four windows cover the whole sub-epoch
(`len(measured) == n_total`, the bed's ~55 batches -- the actual case) the number IS the SUM of the
measured full-step times over every batch; otherwise it is the MEAN x the batch count. The basis is
printed with the number (`seconds_per_subepoch_basis`, `windows_cover_the_sub_epoch`). The json and
`summary.txt` add `seconds_per_subepoch_banked` and `ratio_over_banked`. The class docstring states
that **the bar is read on the ABSOLUTE seconds per sub-epoch (<= 1202 s) and the ratio is disclosed,
never gating**, and that loader wait is excluded on both sides.

### Fix 4 -- the two pre-registered Gate reads are registered

**(a) Step 0 prior-gap rerun.** `_register_prior_gap` builds one `prior_gap.PriorGapAnalysisJob` per
arm and epoch with the arm's own `greedy_raw.json` / `greedy_phones.json` as the private-code input
and EVERY other argument read from `config_sae_4a_prior_gap_v1` (the same `_pin` objects for the
window, word corpus and two lexicons, the same `kenlm_binaries()`, `KENLM_ORDERS = (4, 6)`,
`WORD_LM_ORDER = 3`, `VERSION = 2`) and `config_sae_4a_private_code_v1.prior_npz()` (the frozen
`PhoneNgramPriorJob.RtzbESkOedsT` file -- the same file this pack's own prior job produces). No
Step 0 job is reconstructed: `PriorGapAnalysisJob.2RkbKYl0v1XK` is NOT in the graph.

SCOPE, and a deviation to rule on: the review says "at each kept epoch"; the phase file heads BOTH
disclosed label-using reads with "Disclosed label-using reads at kept epochs 10 and 20", so
`PRIOR_GAP_EPOCHS = FREQ_STRAT_EPOCHS` = (10, 20) and `= KEEP_EPOCHS` is the one-line widening. Each
instance is a 4 CPU / 16 GB / 4 h job that replays the window texts (about 325 MB) and refits its
own KenLM models (about 490 MB) in its work directory, so all four kept epochs would be 24 such
jobs instead of 12. `PRIOR_GAP_ARMS` is the four lexicalised arms AND `ctrl_20` / `ctrl_20_s1`
(frozen decodes), so the Gate's "like-for-like pairing" has its baseline under the same read at the
same N -- the banked 2.32 / 0.237 are `ctrl_50` ep10 numbers.

**(b) The wav2vec-U 2.0 selection statistic.** New `LexlatSelectionStatJob` in `lexlat_train_jobs`
(hand-bumped `__sis_version__ = 1`, so no banked job in that file re-hashes). ONE instance holds
every arm at every kept epoch on `dev-other`, so its ranking is over arms AND checkpoints. The
convention is in the docstring and printed into both outputs: corpus perplexity
`exp(-sum_u log p(u) / sum_u len(u))` of the SIL-stripped decode under the banked 4-gram phone
KenLM, `<s>`-conditioned per-token `BaseScore` with no `</s>` in nats (`prior_gap.kenlm_utt_log_prob`,
the campaign's convention), divided by the SQUARE of `|phone types seen| / 39` over
`prior.ARPABET_39`; lower is better; LABEL-FREE and reference only. A SIL token in the string and an
LM of the wrong order both raise.

**CONCERN, needs a ruling -- the banked 4-gram LM is not a durable frozen path.**
`PriorGapAnalysisJob` builds its KenLM models in its WORK directory and registers none of them as an
output, and the Step 0 v2 instance's work directory has already been auto-cleaned
(`PriorGapAnalysisJob.2RkbKYl0v1XK` keeps a 9.8 kB `finished.tar.gz` and nothing else). The file is
therefore taken from a later Step 0b instance that still holds it; the three surviving copies
(`pg14aEYJyiva`, `OO0iAEVLKgOO`, `m6lUxAO65f6A`) are byte-identical, md5
`61eae5635495b66cec12ef638bafa6d4`, so the fit is deterministic given the pinned window.
`_step0_phone_lm_4gram` pins the first surviving copy by a sha of the FILE'S CONTENT, not of its
path, so repointing it at another copy does not move the read's hash. Those work directories are
cleanup-eligible: if all three go, the config RAISES at graph-build time (it never refits an LM to
stay loadable). A durable fix is outside this dispatch -- either copy the file somewhere stable, or
re-register Step 0's LMs as outputs.

### Checks

* `test_lexlat.py` + `test_lexlat_train.py`: **60 passed, 1 skipped** (the count differs from round
  2's 65 because the pruned-autograd and block tests were re-parametrised in fixes 2 and 3; one new
  test, `test_selection_statistic_is_perplexity_over_the_squared_vocabulary_fraction`, stubs KenLM
  and checks the arithmetic, the empty-utterance rule, the ranking order and both guards).
* `test_blankfree_pack.py`, `test_blankfree_lattice.py`, `test_blankfree_attrib.py`: 11 passed.
* Census, one process per config: `config_sae_4a_prepro_pack_v1` **133**,
  `config_sae_4a_budget_pack_v1` **775**, `config_sae_4a_lexlat_probes_v1` **3** -- every job id
  byte-identical to the round-2 census. `py_e1` still 10 jobs, identical list, E1 still
  `LexlatEfficiencyProbeJob.r4Iaa72mU27T`.
* The pack graph goes 189 -> **202 jobs**: exactly 13 additions and no id moved, so
  `PackedBlankfreeTrainJob.XaDOVLQZkzgh` is unchanged. The new ids are
  `LexlatSelectionStatJob.3HXluQmd65EE` and twelve `PriorGapAnalysisJob`
  (`64uK3sqCde8h`, `BM7uzYpbAse4`, `Bnkr92723Noo`, `DvaNhMgGvweN`, `LaDsQFRa7KDK`, `U7Q2OEFBJR3j`,
  `aSYfGWGD7rJO`, `bTTQ9WGBOWqz`, `d8amOzlVfwGR`, `gQzZOfc7PHCp`, `vTOEg1CwN2Pb`, `x51tSxaPXuoR`);
  none of them has a job directory on disk, so nothing banked is re-funded.
* Noted, not changed: `lexlat_neg_inf` is a normalised batch count in the sub-epoch column; since
  `6cbb3f9` any NEG_INF utterance after the on-set raises, so the column is read as "nonzero".

---

## 9. C re-declared at 4096 after E0 (speech-llm `373eb78`, 2026-09-21)

E0's curve (`output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat/e0_census/summary.txt`) puts
C = 1024 at a log Z gap of **0.0512** nats per retained frame against C = 4096, over the funding
rule's 0.05 (C = 256: 0.0913). Design 6: "If C = 1024 misses the rule, C is re-declared at the
smallest measured value meeting it, the change recorded before launch, and E1 measured at that C."
C = 4096 is the only measured value meeting the rule (pruned median 0.0000 at every lam_lex, zero
NEG_INF, about 4,032 live contexts per frame against 1,009 at C = 1024).

### What changed

* `MAX_CONTEXTS = 4096` in `config_sae_4a_lexlat_pack_v1`, with E0's numbers in the constant's
  comment. All four pack arms carry it (verified by resolving each arm's `get_model`
  `hashed_arguments`: C 4096, C_esc 64, lam_lex 1.0, order 3, on-sets 8 / 8 / 5 / 8, `lexlat_20_s1`
  still `random_seed` 1 with offset 1000, batch `{'features': 88000}` / `max_seqs` 128).
* **C_esc stays 64.** Design 2 writes "an ESCAPE sub-budget of C_esc = 64 of the C slots is
  reserved" -- an absolute slot count, not a fraction of C -- so nothing in Design 2 scales it with
  C and it is left where it is.
* **`max_candidates` needs no edit**: `lexlat_max_candidates` is never written by the config, so the
  model derives it from C and the batch (`lexlat_train.derive_max_candidates` =
  `1.25 * C * (2K + 1) * (2 + wmax) * B`), i.e. the arc guard scales linearly with the new C by
  itself.
* C is threaded through `_model_args_delta` / `_arm_train_config` / `efficiency_probe`, so the two
  E1 registrations differ **in C alone**: same arm, same pinned checkpoint (`epoch.010.pt`), same
  sub-epoch 10, same recorded seed and seed offset, same batch plan, same job `name`. `py_e1` is
  frozen at `E1_CONTEXTS_LOWER = 1024` and still rebuilds the RUNNING
  `LexlatEfficiencyProbeJob.r4Iaa72mU27T` at its own hash; the new `py_e1_c4096` registers the
  4096 probe under its own alias and its own `e1_efficiency_c4096` output prefix, so the two jobs
  can never repoint each other's output symlink.
* New workspace shim **`/e/project1/spell/wu24/2026-07-13_unsupervised/config/sae_4a_lexlat_e1_c4096.py`**,
  defining `py` (the workspace `config/` tree is not a git repository, so it is not in the commit).
  `config/sae_4a_lexlat_e1.py` and the running job are untouched.

### The destination block in the E1 output (review item 3 of this round)

The width of the expansion's destination block is a MEASURED quantity at a given C and B, not
something the config states, so the probe now measures it: it wraps `lexlat._dest_block` for the
timed walk (the banked path never calls it, so every record belongs to a lexicalised step) and
reports, per step in `measurements[i]["dest_block"]` and over the walk in `dest_block`: the number
of frames, the width range, the width histogram, the busiest frame's `m_max` with its `n_new`, B,
O, the chosen width and the GiB that block actually took, and the widest block of the walk. The
`dest_block_convention` field and the class docstring carry the formula and the arithmetic at the
pack's shape: with `DEST_BLOCK_BYTES = 8 GiB`, three copies, B = 128, O = 51 and float64 one
destination costs `3 x 128 x 51 x m_max x 8 = 156,672 x m_max` bytes, so the width is about
`54,827 // m_max`, capped at `DEST_BLOCK = 1024` -- e.g. 145 at m_max 376, 9 at 6,014, 2 at 25,511,
and the block stays at or under 8 GiB at every C. The spy costs one python call per frame inside
the timed region and that is disclosed in the report; the DP's values do not depend on the width
(bit-identical blocking, `test_lexlat`), so this is a memory reading and never a result.

### Checks

* Tests: **72 passed, 1 skipped** (`test_lexlat.py`, `test_lexlat_train.py`,
  `test_blankfree_pack.py`, `test_blankfree_lattice.py`, `test_blankfree_attrib.py`), including the
  new `test_destination_block_stats_report_the_widest_block_of_the_walk`.
* Census, one process per config: `config_sae_4a_prepro_pack_v1` **133**,
  `config_sae_4a_budget_pack_v1` **775**, `config_sae_4a_lexlat_probes_v1` **3** and `py_e1`'s
  **10** job ids -- all byte-identical to the previous census, so the running C = 1024 probe keeps
  `LexlatEfficiencyProbeJob.r4Iaa72mU27T`.
* New hashes: **`LexlatEfficiencyProbeJob.eHrOFiwUqKAf`** (`py_e1_c4096`, a 9-job graph, confirmed
  through the loader's own lookup on the new shim) and **`PackedBlankfreeTrainJob.DPiivOfTWAdM`**
  (the pack, still 202 jobs: the training job and everything below it re-hash with C, the 14 shared
  upstream jobs do not).
* Note for the plan, not acted on: Design 7's first declared E1 fallback is C = 512, which E0's own
  funding rule now excludes (its gap would exceed 1024's 0.0512). If E1 fails at C = 4096 the
  remaining declared fallbacks are the word bigram CSR and stop.

## 10. `LexlatWordCountsJob` paired the wrong window sample (fix, commit `154fc5d`)

`LexlatWordCountsJob.X2YnVYfqN7aV` died at `lexlat_train_jobs.py:570` ->
`prior_gap.replay_window_texts` (`prior_gap.py:919`) on that function's own consistency check, at
window line 1: the phone line `['<SIL>','AH','<SIL>','AH','F','R','EH','N','D','AH','K','AY']`
against the word line `['AH','AH','HH','AO','N','T','AH','D','HH','AW','S']`.

**Which two inputs the job pairs.** The frozen files were already right: the job takes
`probes.WORD_CORPUS` (the word corpus), `probes.WINDOW_PHN` (the phone window), `BLISS_LEXICON` and
`G2P_LEXICON` -- the same four `tk.Path`s the FINISHED `LexiconTrieBuildJob.rlMsnTBSZXsB` takes
(`config_sae_4a_lexlat_probes_v1.py:108-115`). What was wrong were the **replay constants**.
`replay_window_texts` does not read a stored index; it REPLAYS the sampling job that produced the
window, `text_sample.sample_line_indices(n_kept, n_window_lines, sample_seed)`, and walks the word
corpus's kept lines in step with the window file. My defaults were `n_window_lines = 500_000` and
`held_stride = 50`, invented at the call site. The banked window is `SampleLinesJob.orN768ARKwlt`
with `n_out = DEFAULT_COUNT_LINES + DEFAULT_HELD_LINES = 1,010,000`, and a uniform sample of
500,000 lines is not the head of a sample of 1,010,000, so window line *j* was paired with a
different corpus line -- the mismatch above. `held_stride` was likewise wrong (50 against
`prior.HELD_STRIDE = 101`), which would have counted a subset that is not the trigram's.

**The banked jobs that pair them correctly** are `lexlat_jobs.LexiconTrieBuildJob.__init__`
(lines 123-143) and `prior_gap.PriorGapAnalysisJob.__init__` (lines 1409-1436): both take
`Optional[int] = None` and resolve from `prior`'s constants. The fix makes this job resolve them the
same way (and says so in the docstring); the config keeps omitting them, so no call site invents a
value. This is the standing "n-grams always from the unbiased window" rule applied to a count read.

**Test** (`test_lexlat_train.py::test_word_count_job_replays_the_window_with_the_banked_sampling_constants`):
(a) the job's resolved `n_window_lines` / `held_stride` / `sample_seed` equal
`LexiconTrieBuildJob`'s and equal 1,010,000 / 101 / 0; (b) a synthetic corpus plus the window built
from it through the **real** `sample_line_indices` replays through the **real**
`replay_window_texts`, and the first three written lines are the sampled corpus lines and phonemize
consistently; (c) a replay with a deliberately wrong `n_window_lines` (the smallest whose sample
starts at a different corpus line, found by scanning, so the mispairing is on line 0) raises the
same "does not phonemize to its word line" assertion.

**Checks.** 62 passed + 1 skipped in `test_lexlat_train.py` + `test_lexlat.py`; 27 passed in
`test_prior_gap.py` + `test_text_sample.py`. Census (one process per entry point, diffed against the
section 9 census): prepro 133, budget 775, probes 3, `py_e1_c4096` 9 job ids **byte-identical**;
`py_e1` still 10 jobs and the pack still 202, with exactly three ids moved in the pack graph and one
in `py_e1` -- `LexlatWordCountsJob.X2YnVYfqN7aV` -> **`1ZJy5dFbOAHD`** and the two
`LexlatFreqStratPerJob`s that consume its `out_deciles`
(`DCkFkyFxf5jX` -> `96Jl9MdOfCft`, `Rqn0ReKyNnfL` -> `wkGNg7CuPdPU`). `LexlatEfficiencyProbeJob`
`r4Iaa72mU27T` / `eHrOFiwUqKAf`, `PackedBlankfreeTrainJob.DPiivOfTWAdM` and
`LexlatSelectionStatJob.DLhM5v5w9wbi` are unchanged. Per the dispatch, neither the E1 job nor
`lexlat.py` was touched in this commit.

**For the executor, not acted on here:** the failed `X2YnVYfqN7aV` job dir is now orphaned (the read
reruns at `1ZJy5dFbOAHD`); its error marker is a leftover, not a live failure.

## 11. The E1 OOM remedy: the DP blocked over the BATCH axis (commit `d2f6a5e`)

Answering the debugger's reading (`reports/debug_lexlat_e1_oom_2026-09-21.md`): the OOM is in the
MANUAL BACKWARD of `lexlat_forward_backward`, whose per-frame arc section holds five
`[B, K, m_max, O]` float64 tensors at once -- 64 GiB at B = 114, m_max = 7416 -- after the forward,
which `_dest_block` already bounds, had succeeded.

**What was changed (`lexlat.py`, one place per leg).** The backward's arc section and the forward's
band reduction (`_band_reduce`, the forward leg's twin allocation, `[B * K, m_max, O]`) are cut into
blocks of batch ROWS:

```
b_c = max(1, min(BATCH_BLOCK, B, BATCH_BLOCK_BYTES // (copies * K * m_max * O * itemsize)))
```

with `BATCH_BLOCK_BYTES = 8 GiB` (`DEST_BLOCK_BYTES`' twin), `copies = 5` in the backward and 3 in
the band reduction. At the measured shape that is **14 rows**, i.e. 5 x 14 x 39 x 7416 x 51 x 8 B =
**7.9 GiB in place of 64 GiB**; the rest of the step (the checkpoint store, the replay cache, the
slot columns, the model and the optimizer) is untouched, so this is a reduction of the term the
diagnosis names and not a reading of the new peak.

**Why it is bit-identical rather than equal to a tolerance.** Every width the blocks pad to -- the
live-arc count `m_n`, the band group's `m_max`, the context counts -- is computed on the WHOLE batch
BEFORE the loop and handed to every block, so a row's padded rectangle, its reduction set, its order
and `_logmm`'s per-row max-shift along the contracted axis are exactly the un-blocked ones; only the
rows are processed in groups. (Recomputing the widths per block would NOT be: a group of rows has a
smaller `m_max`, and a destination whose arcs are all NEG_INF reduces to `NEG_INF + log(width)`.)

**Placement, and the one deviation from the dispatch's wording.** The dispatch's own sizing formula
contains `m_max`, which is a PER-FRAME runtime quantity: it does not exist before the frame runs, so
the chunking is applied inside the frame (where the coordinator's formula is evaluable with the
actual `m_max`) rather than by slicing the whole call into sub-batches and concatenating. Slicing
the call would additionally have changed every padded width per sub-batch, which is the one thing
that can move a value. The per-row results are written back into the full-batch outputs, which is
the "concatenation" the dispatch asks for.

**Tests** (`test_lexlat.py`): `test_batch_blocking_is_bit_identical` builds a padded batch of six
toy utterances and asserts, for forced blocks of 1 / 2 / 4 rows, with and without the escape, that
log Z, `post_q`, `seg_post`, every `expected_*` readout, the pruning monitors and the autograd
gradient of `lexlat_log_z` (both `d log Z / d log_q` and `d log Z / d seg`) are `torch.equal` to the
un-blocked run; `test_batch_blocking_leaves_the_manual_backward_equal_to_autograd` re-checks
`post_q = tau * d log Z / d log_q` on the blocked path (the B = 1 gradient tests never enter the
loop); `test_batch_block_width` pins the byte rule at the OOM's own shape.

**E1 (`LexlatEfficiencyProbeJob`), the three instrumentation items.** (a) One line per batch PER LEG,
printed and flushed the moment the leg finishes: index, padded T, B, the `m_max` seen, seconds,
`max_memory_allocated` and `max_memory_reserved`; plus `partial.json`, rewritten after every timed
batch, so a kill leaves every completed measurement on disk (`efficiency.json` is written only by a
completed run). (b) A new HASHED kwarg `max_timed_batches`: the timed set becomes the largest-T
batch plus the first `k = (max_timed_batches - 1) // n_points` batches of each window, never larger
than the cap; the set is then a sample, so the sub-epoch number is the MEAN full-step time times
`n_total`, the rule already written for a sampled set. (c) `STEP_ABORT_SEC = 1200.0` (the
coordinator's value): a lexicalised step above it ends the lexicalised leg, the banked leg still
runs on the planned batches, and the job writes `verdict = FAIL` with the seconds marked a LOWER
BOUND and exits 0.

**Registrations and hashes.** Both probes carry `E1_MAX_TIMED_BATCHES = 9` (k = 2) and
`E1_TIME_RQMT_HOURS = 4.0`, and nothing else moved (same checkpoint `epoch.010.pt`, epoch 10, name
`lexlat_20/ctrl_50_ep10`, C_esc 64; C alone differs). Both therefore re-hash:

| entry point | C | before | after |
| --- | --- | --- | --- |
| `py_e1` | 1024 | `r4Iaa72mU27T` | **`x3LaY6KlB6I4`** |
| `py_e1_c4096` | 4096 | `eHrOFiwUqKAf` | **`fY18YN7LVdAe`** |

Census (one process per entry point, diffed against section 9/10's): prepro **133**, budget **775**,
probes **3** and the pack graph (**202** jobs, `PackedBlankfreeTrainJob.DPiivOfTWAdM`,
`LexlatWordCountsJob.1ZJy5dFbOAHD`) byte-identical; in `py_e1` / `py_e1_c4096` the only changed id is
the probe's own. `lexlat.py` is imported at run time and is not hashed, so the blocking itself moves
nothing.

**Checks.** 85 passed + 1 skipped over `test_lexlat`, `test_lexlat_train`, `test_blankfree_lattice`,
`test_blankfree_attrib`, `test_blankfree_pack`. All of it is CPU: bit-identity is asserted at the
toy shape in float64 on CPU, and the memory claim above is arithmetic on the diagnosis's measured
`m_max`, not a GPU reading. Whether the blocked backward actually fits 80 GiB at the run shape is
E1's own measurement.
