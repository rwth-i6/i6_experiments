# Survey: what a sampled score-function prior term needs in the blank-free step — 2026-09-20

Read-only code survey (Explore agent) of `recipe/2025-10-speech-llm/src/speech_llm/`, facts only,
for the `SAE_4A_prior.md` training arm (sampled strings from the lattice posterior, reward
log p_strong(y) − log p_3(y), per-utterance baseline). Paths relative to `src/speech_llm/`.

## 1. Manual forward-backward and existing samplers
- Forward table `table [b, n_store, o_n, h_n, 2]`, n_store = ceil(T/ck), filled under no_grad at
  t % ck == 0 (lattice.py:1099-1100, 1007-1008). Blank-free: ck = 32 (blankfree_train_jobs.py:176),
  W = 25 so o_n = 51, trigram h_n = 1681, float64 (blankfree_train_jobs.py:177).
- Backward recomputes each 32-frame chunk from its checkpoint with the SAME `_forward_step` and
  `cx`, bit-identical (lattice.py:1107, 1154-1161, 1060-1062, 1144-1147). Per-frame transients are
  not stored; `seg_pad` (lattice.py:903-927) and `cx` (`_forward_ctx`, lattice.py:794-858) are
  resident. A backward-sampling pass needs exactly what the backward has: checkpointed table,
  `cx`, `seg_pad`, recomputed per-frame `fwd`.
- Existing FFBS: `candidates.sample_joint_paths` (:31) / `sample_joint_paths_batch` (:119) draw
  complete JOINT paths (frame symbols, durations, states, collapsed `phones`), return `log_joint`
  (with prior x prior_weight and 1/tau) and `log_z` (`JointPathSamples`, :16-23); float64,
  no_grad, keep the FULL forward table `(batch, t_max, n_offsets, n_hist, 2)` without
  checkpointing (:132). FLAG: CTC-hard-coded (blank at q[t,0], phone at q[t,phone+1], :78-80,
  :104-105, :178-180, :205-206) and stride 1 (`o_keep = s - t + band`, :75, :176). The blank-free
  arm has 40 symbols, no blank, stride 3 (lattice.py:243, sae_blankfree.py:262);
  `_forward_step` / `_arc_weights` are topology- and stride-aware (lattice.py:511-518, 870, 893).
- Memory notes: `path_training._sampling_bytes` (:36-47), `candidate_training._conditional_bytes`
  (:68-79), `DP_STORE_BYTES = 48 GiB` (candidate_training.py:12); callers shrink the group and
  raise "single-utterance complete-path FFBS exceeds the approved memory estimate"
  (path_training.py:70, candidate_training.py:107). Diagnostics: path_ess, unique_phone_strings,
  phone_ess (path_training.py:111-115).
- `reverse.sample()` (:368) draws z ~ p_phi(z | y, eta), not lattice paths (BT); bt_aux /
  bt_blankfree draw sentences from text, not from q_theta.

## 2. Scoring a fixed string under the recognizer
- `candidates.fixed_phone_log_marginals` (candidates.py:282-295): log A_tau(y) without the text
  prior, marginal over alignments and segmentations including G and 1/tau; no_grad, float64,
  state `[N, U+1, n_offsets, 2]` (:249); same CTC / stride-1 caveat.
- `candidate_lattice_losses` (candidates.py:319-419): differentiable finite-Y version, surrogate
  `-((post_q*log_q).sum + (seg_post*seg_table).sum)/tau` (:417-419), per-string `log_a`.
- Blank-free recognizer-only: `blankfree_seed.transcript_logprob` (blankfree_seed.py:10-27),
  differentiable, `log_q [B,T,K]`, `targets [B,U]`, O(T) loop over `[B,U]`, rejects
  adjacent-identical targets; used by bt_blankfree (:82) and the seed step (:30-46). The
  derangement-gap job scores phi only (blankfree_eval_jobs.py:187-195).

## 3. Live trigram outside the lattice
- `prior.PhoneNgramPrior.per_token_log_probs` / `.log_prob(seq, order)` (prior.py:313-330):
  numpy, BOS-padded (`BOS_ID = 40`), no end-of-sequence term (:267-273).
- Torch batched: `candidate_training.candidate_sequence_scores(lm, sequences, order, prior)`
  (candidate_training.py:29-59), order 3 walks `prior_log_bi [N_CTX^2, K]`, history
  `k*(k+1)+k` as BOS-BOS.

## 4. Train step structure (train_steps/sae_blankfree.py)
- Terms marked individually via `ctx.mark_as_loss(value, name, scale=model.lam_X)`: l_tau (:89),
  agg (:90), rate (:110); no single summed surrogate.
- Monitors: `values` dict -> `mark_as_loss(torch.tensor([float(v)]), f"blankfree_{k}",
  as_error=True)` (:136-153). Cleaner template: `infomax_terms` returns
  `{name: InfomaxTerm(value, scale, as_error)}` (:122-130; infomax.py:103-117, 264-294), guarded by
  `getattr(model, ...)`.
- Weights/schedules: `model_args` in `blankfree_train_jobs.build_blankfree_train_config`
  (:161-178) -> `PartialImport(hashed_arguments=model_args)` (:221-224), every key hash-relevant.
  `learning_rates` per sub-epoch list (:256), length = num_subepochs (:150-154); schedules from
  `blankfree_budget_jobs.budget_learning_rates` / `budget_temperature_schedule` (:77-156);
  `sae_emc.schedule_value` (1-based, last held).
- lam_bt template: `SaeEmcModelV1(lam_bt, bt_ramp_epochs, bt_depth, bt_text_path)`
  (definitions/sae_emc.py:216-221, 448-487); config keys written only when lam_bt != 0
  (blankfree_train_jobs.py:179-216); step dispatches on a marker tag, imports inside the guard
  (train_steps/sae_blankfree.py:29-33, 155-165); ramp `lam_bt_effective` (bt_aux.py:168-177); BT
  DOUBLES the step count per sub-epoch (bt_aux.py:28-32); ramped weight rides the loss value not
  `scale` (bt_aux.py:35-43); `BT_MONITOR_KEYS` (bt_blankfree.py:112-119).

## 5. GRPO / reward code still in tree
- sae/grpo/grpo.py:11-41 `group_advantages`: A = (r − mean_G)/(std_G + eps), population std;
  `normalize_std=False` = centre-only (recorded pitfall: std-division equalises easy and
  degenerate groups). `grpo_policy_loss` (:75-129) PPO-clipped, agg seq_mean / token_mean /
  seq_sum ("removes the length bias").
- sae/grpo/reward.py:115-146 records the LENGTH EXPLOIT (per-token normalisation makes the mean
  rise with length; corr +0.22/+0.30; per-unit is the sign guarantee); `length_hinge` (:86-91);
  DEAD BAND (trainer.py:284-287; test_trainer.py:222); per-term within-group std is the
  calibration column (trainer.py:255-261); a -inf in a group destroys mean/std (psi_scorer.py:14).

## 6. Loading a frozen external torch LM
- sha256-pinned precedent: `candidate_counts_path` + `candidate_source_sha256` ->
  `candidate_training.load_candidate_lm` (:15-26); wired definitions/sae_emc.py:246-248, 527-535;
  configs/config_sae_4a_path512_train_v1.py:20-43.
- Checkpoint-path precedent: `recognizer_checkpoint_path`, `reverse_checkpoint_path`,
  `prior_npz_path`, `eta_table_path` (definitions/sae_emc.py:222-225), `_load_state`
  (:102-127), `_load_prior` (:563-571); all hashed model_args.
- No torch transformer phone LM exists in the tree (KenLM 4/6 in prior_gap.py; SparsePhoneLM
  orders 1-6).

## 7. Memory / step-rate facts
- Batch budget estimator `estimate_peak_gib = 6.8e-5 * B * T_max`, `check_batch_budget`,
  MAX_UTTS_PER_BATCH 128, MAX_BATCH_FRAMES 128_000 (lattice.py:140-188), fitted on a CTC / bigram
  / fp32 / stride-1 GH200 bench (lattice.py:146-151). FLAG: no measured number for the blank-free
  trigram float64 step.
- Operating point: 88_000 frames / max_seqs 128 (emc_train_jobs.py:263, 270), accum 1; rqmt cpu
  16 / mem 64 / gpu_mem 96 (blankfree_train_jobs.py:60-63). About 601 s per sub-epoch, ~56 EMC
  steps per sub-epoch (blankfree_budget_jobs.py:57-58; bt_aux.py:133-134). Monitors
  `blankfree_frames_per_sec`, `blankfree_rate_dp_calls` (train_steps/sae_blankfree.py:147-149).
- DP calls per step today: 1 forward+backward (`lattice_loss`) + the rate finite-difference
  tilts (2 stacked into one call when `fd_batch_fits`, else 2 sequential; rate_term.py:538-639),
  so the step already pays about 2 DP passes.

Undetermined: no per-step GiB / s measurement for the blank-free trigram float64 configuration;
no blank-free-topology sampler and no blank-free fixed-string A_tau(y) exist.
