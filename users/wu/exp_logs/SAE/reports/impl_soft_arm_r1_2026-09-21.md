# Implementer: the soft (straight-through) scorer arm, round 1 (2026-09-21)

Built to `SAE_4A_prior.md` "Training arm" (A1-A5) and the binding "Training arm, reopened by user
ruling (2026-09-21)". Pattern: the lexlat round-2 build. Commit **f99f9f6** on
`haotian_modality_matching_jupiter` of `recipe/2025-10-speech-llm` (explicit paths, NOT pushed).
Nothing was launched.

## 1. Files

| file | what |
|---|---|
| `src/speech_llm/sae/emc/soft_scorer.py` (new, 873 l) | the whole term |
| `src/speech_llm/sae/emc/soft_scorer_jobs.py` (new, 488 l) | `SoftLamProbeJob` |
| `src/speech_llm/sae/emc/test_soft_scorer.py` (new, 442 l) | 22 tests |
| `src/speech_llm/prefix_lm/model/definitions/sae_blankfree.py` (+37/-1) | default-off `soft_*` block |
| `src/speech_llm/prefix_lm/model/train_steps/sae_blankfree.py` (+26) | the guarded branch + monitors |
| `.../librispeech/configs/config_sae_4a_soft_pack_v1.py` (new, 552 l) | probe graph + pack graph |
| `config/sae_4a_soft_probe.py`, `config/sae_4a_soft_pack.py` (setup dir, untracked) | shims defining `py` |

## 2. `soft_scorer.py`

* `viterbi_blankfree` -- a max-plus (Viterbi) pass of the blank-free lattice. None existed: the
  banked DP is log-sum-exp and the sampler is FFBS. It reuses `lattice._forward_ctx`,
  `_band_matrix`, `_prior_term` and `blankfree_sampler._collapse`, replaces `logsumexp -> amax` and
  `logaddexp -> maximum`, and -- since there is no max-plus GEMM -- chunks the history reduction
  over the symbol axis (`SYMBOL_BLOCK`) and the band reduction over the successor-group axis
  (`GROUP_BLOCK`). Checkpointed forward + chunked replay backtrace with `argmax`, under `no_grad`.
  The walk asserts it lands on `(s=0, h=start, f=0)`.
* `segment_conditionals` -- the A2 conditional in CLOSED FORM, no second DP: under a substitution of
  one token's symbol exactly the token's recognizer frames, its segment score, three prior terms and
  three legality masks move. Validated value-by-value against brute-force path enumeration (trigram
  and bigram history, tau in {0.5,1,2,8}, beta in {0,1}).
* `straight_through`, `derangement` / `assert_derangement`, `SoftScorer` (frozen `model.pt`, `eval`,
  `requires_grad_(False)`, `config["phones"] == PHONES` asserted; it does NOT touch global cuDNN or
  TF32 flags), `soft_log_prob` (BOS context, teacher forcing, no EOS, reduction at the soft vector's
  own dtype), `unigram_log_prob`, `SoftRuntime`.
* `SoftRuntime.step` -> `(term, monitors)`: Viterbi -> conditionals -> two checks -> SIL dropped ->
  `>512` masked and counted -> compaction -> optional derangement -> `strong - uni` ->
  `term = -sum(r/retained)/n_scored` (A5). Monitors: `soft_lam`, `soft_masked_long`,
  `soft_argmax_mismatch`, `soft_scored`, `soft_term_mean`, `soft_reward_mean` (per token, hard
  string), `soft_artefact_gap` (scorer(soft) - scorer(hard) per token), `soft_tokens`.
  Two hard failures (`FloatingPointError`): a path that does not re-add to the forward's maximum,
  and a conditional whose argmax is not the Viterbi symbol by more than `ARGMAX_TOL`.
* The frozen scorer is held in a per-device cache, never a submodule: `soft` appears in NO
  `state_dict` key (checked on the real written config, section 5), so a banked checkpoint still
  loads strictly and the scorer can never enter the optimizer.

## 3. `soft_scorer_jobs.SoftLamProbeJob`

GPU (cpu 16 / mem 64 / gpu_mem 96 / time 1 h), `__sis_version__ = 1` hand-bumped, one output pair
(`summary.txt`, `probe.json`). It rebuilds the model from the FROZEN written `returnn.config` of the
finished prepro pack's `ctrl_20`, loads the ep4 checkpoint, attaches a `SoftRuntime` and runs the
ARM'S OWN `train_step` on the arm's own train loader at the run shape (`batch_size` and `max_seqs`
asserted against the config). Both terms come from ONE forward: `l_tau` and `soft` are read off
`ctx.losses` AS MARKED (unscaled) and each is differentiated with `torch.autograd.grad` over theta +
phi; the ratio is `|grad soft| / |grad l_tau|`, the reported number is the median over the batches,
`lam_soft = target / ratio` for 0.1 and 0.3. `lam_tau` is ASSERTED to be 1.0 (it is, for ctrl_20),
which is what makes "the l_tau gradient norm" unambiguous; at any other value the job refuses rather
than picking a convention. A5 gives ONE convention here (both terms are already per retained frame),
unlike the score-function probe which had to report two. The step's own monitors -- every `as_error`
loss, `soft_*` included -- are recorded per batch and rendered.

## 4. `config_sae_4a_soft_pack_v1.py`

* `py_probe()` -> ONE job, `SoftLamProbeJob.wAJQ26T7iZzX`; every input is a pinned frozen path
  (`probes._pin`: `hash_overwrite` carries a sha of `os.path.realpath`, so repointing an input moves
  the hash), so the probe graph contains no other job and rebuilds nothing.
* `py()` -> the pack: one `PackedBlankfreeTrainJob` with `soft_20`, `soft_20_s1`, `soft_20_r03`,
  `softshuf_20`; N = 20, kept epochs 1/4/10/20, schedules, streams, priorshuf prior, flat init and
  seed set all taken from `config_sae_4a_prepro_pack_v1` and never re-typed. Per-epoch greedy PER +
  decode stats + derangement gap; `PairedPerDeltaJob` on the spec's five pairings at all four kept
  epochs and both dev splits (40 jobs); `PriorGapAnalysisJob` on each arm's decode at ep10 and ep20
  (8 jobs), every other argument read from `config_sae_4a_prior_gap_v1` /
  `config_sae_4a_private_code_v1`.
* `LAM_01 = LAM_03 = None`. `build()` calls `_lam(arm)` for every arm BEFORE constructing anything,
  so the pack cannot be built before the probe is read. What sis says today:
  `AssertionError: soft_20 takes lam_soft = LAM_01, which is still None: run the lam_soft probe
  (config/sae_4a_soft_probe.py), read its summary.txt and fill LAM_01 in ...`.
  The pack's job hash CANNOT be reported yet: `lam_soft` is in the model args, hence in the arm's
  config, hence in the pack hash. With PLACEHOLDER weights (0.125 / 0.375) the graph builds to 186
  jobs and `PackedBlankfreeTrainJob.ngpp1GNOMmy5` -- a placeholder id, not the arm's.

## 5. Checks run

* `pytest speech_llm/sae/emc/test_soft_scorer.py -q` -> **22 passed** (torch 2.7.1, pytest 9.1.1):
  Viterbi = the enumeration's arg maximum (5 cells) and checkpoint-stride invariance; the segment
  conditional = the substitution posterior value-by-value (7 cells, trigram and bigram); a long
  silence segment admits no other symbol; the reward = `prior_gap`'s neural minus
  `witten_bell_utt_log_prob(order=1)` on the same strings, by CALLING those primitives with real
  objects, agreement < 1e-5 at production dtype; the unigram term is linear in the soft vector; the
  straight-through gradient EQUALS the soft-vector gradient at the hard string (`rtol=0, atol=0`);
  the derangement is fixed-point free and fixed by its seed; with the block off the loss and BOTH
  gradients are `torch.equal` to the banked path (plus a non-vacuity check); a string over 512
  positions is masked out and counted; the null scores the deranged string; the probe's summary
  renders every field it banks.
* Neighbouring suites: `test_lexlat_train.py`, `test_blankfree_permute.py`, `test_infomax.py` ->
  35 passed. One failure, `test_the_arm_configs_differ_from_ctrl_only_in_the_infomax_arguments`,
  is the known missing-`black`-on-PATH trap and NOT this change: with
  `PATH=/e/project1/spell/wu24/env/sis_env/bin:$PATH` all 9 infomax tests pass.
* The call into the shared primitive, end to end: `soft_20`'s and `softshuf_20`'s ReturnnConfigs
  were WRITTEN with `ReturnnConfig.write` and re-loaded with RETURNN's own `Config.load_file`, then
  `get_model(epoch=1, step=0)` was called. `model.soft` is a `SoftRuntime` with the right spec, the
  null's recorded permutation is fixed-point free with SIL (39) fixed, and
  `[k for k in model.state_dict() if "soft" in k] == []`.
* Census (fresh process per graph, `tk.sis_graph.jobs()` ids sorted):
  `config_sae_4a_prepro_pack_v1` **133**, `config_sae_4a_budget_pack_v1` **775**,
  `config_sae_4a_lexlat_probes_v1` **3** -- the banked counts of
  `reports/impl_lexlat_r2_2026-09-21.md` section 4, with `LexiconTrieBuildJob.rlMsnTBSZXsB`,
  `LexlatEquivalenceProbeJob.lq61PSAg1DcC`, `LexlatCensusJob.RDMTvt4ngAPB`,
  `PackedBlankfreeTrainJob.5EIGJJ1MkcO9` and `PackedBlankfreeTrainJob.ks7CbtlvpcIL` all present and
  unmoved. Containment: the new pack graph holds exactly ONE `PackedBlankfreeTrainJob` (its own),
  shares 9 upstream jobs with the prepro graph and 8 with the budget graph (same ids, nothing
  re-funded).
* NOT run, and it cannot be: `SoftLamProbeJob.run()` itself. It needs a GPU node and this round
  launches nothing, so the job is code-reviewed, not measured. Its RETURNN calls are copied from
  `LexlatEfficiencyProbeJob` (which has run) and `SampledRewardProbeJob`.

## 6. Resolved inputs

* **Scorer**: `work/speech_llm/sae/emc/neural_phone_lm/NeuralPhoneLmTrainJob.xObXEwRpvmzd/output/model.pt`
  -- the file `PriorGapAnalysisJob.5wNIQs2lpC5P` consumed (its `input/` symlink). Its `train_log.json`
  gives `selected_epoch 10`, `heldout_perplexity 5.0906`, `parameters 3312170`, and the checkpoint's
  own config is `4 layers / 256 wide / 512 positions / 42 vocab` -- the spec's instance (a).
  **Discrepancy**: the spec (line 568) and the dispatch name the class `NeuralPhoneLmTrainJobV2`.
  The only `NeuralPhoneLmTrainJobV2` on disk (`vkNGAOeLgNsy`) is a 25,525,290-parameter model at
  perplexity 3.96 -- a different instance. The path the scored row came from wins; the class name
  in the spec looks wrong.
* **Unigram (A1 baseline)**: `PhoneNgramPriorJob.RtzbESkOedsT/output/prior.npz`, i.e.
  `private_code.prior_npz()` -- the arm's own `prior_npz_path` in `ctrl_20`'s written config, the
  unbiased 1,010,000-line window (1,000,000 counted) the scorer was fitted on.
* **Probe checkpoint / config**: `PackedBlankfreeTrainJob.5EIGJJ1MkcO9/output/ctrl_20/`
  (`models/epoch.004.pt`, `returnn.config`; `batch_size {"features": 88000}`, `max_seqs 128`,
  `lam_tau` unset = 1.0), consumed as frozen paths.
* **Controls**: the prepro pack's `ctrl_20` / `ctrl_20_s1` greedy decodes, resolved through the
  registered output names to their `BlankfreeGreedyPerJob` work directories.

## 7. Constants I chose (the spec does not fix them)

| constant | value | why |
|---|---|---|
| `DERANGEMENT_SEED` | 0 | the bed's null convention: lexlat's shuffled-pronunciation null is the seed-0 derangement, `PriorGapAnalysisJob.null_seed` is 0. The permutation itself is printed at model construction and banked in `describe()`. |
| `ARGMAX_TOL` | 1e-6 | the numerical tolerance of the argmax-equals-Viterbi assert; a tie must not raise. |
| `SYMBOL_BLOCK` / `GROUP_BLOCK` | 4 / 4 | memory chunking of the max-plus reductions; they change no value (the checkpoint-stride test and the enumeration test cover it). |
| `soft_checkpoint` | `None` (= the DP's own `lattice_checkpoint`) | the max-plus pass runs at the stride the arm's lattice already runs at. |
| `PROBE_LAM` | 1.0 | a placeholder; `SoftRuntime` refuses 0 and the probe reads only unscaled gradients. |
| probe batches | the FIRST 3 of the arm's own sub-epoch-4 ordering | the spec says "3 batches" and no more. Each batch's B / T / padded frames / retained frames and its own ratio are reported, so the spread a median of three hides is visible. |
| `time_rqmt` | 1.0 h | an allocation, not a measurement ("minutes" per the spec). |
| the derangement is applied to the string for BOTH halves of the reward | -- | the null's reward is the same function composed with a bijection; applying it to `p_strong` alone would also change the baseline's support. |
| `soft_artefact_gap` | the monitor's name | the spec names the quantity, not a key. |

## 8. Points I could not implement as written

* The PACK's job hash cannot exist before the probe is read (section 4). This is the spec's own
  ordering, not a defect.
* The control rows of the prior-gap read (`ctrl_20` / `ctrl_20_s1` at ep10 / ep20) are NOT
  registered here: `PriorGapAnalysisJob`'s `name` is hashed, so a copy under this phase's name
  would be four more 4-hour CPU jobs computing what `config_sae_4a_lexlat_pack_v1` already
  registers on the same frozen control decodes. Stated in the module docstring.
* `SoftLamProbeJob.run()` is unexecuted (section 5).
