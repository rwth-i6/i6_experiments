# Code review: L2-1 phi-first probe launch (A3 / A8 / A9), 2026-09-23

Verdict: APPROVE_WITH_NOTES. I found nothing that blocks the launch. No line produces a wrong
number, a wasted run or an unreproducible result. The notes below are non-blocking and are ordered
by relevance.

Reviewed:
- Branch haotian_modality_matching_jupiter in recipe/2025-10-speech-llm, commits 35c1eb26,
  d0048d7f, afa6b3dd, 6e917b83, e1cd038c and 3dc01561 (HEAD 3dc01561). The working tree changes
  only files unrelated to this launch (config_sae_1g_v1.py, and the untracked
  config_sae_3e1_d6_swap_cont_v1.py).
- The shim config/sae_4a_lexlat_v2_em_probe.py.
- The implementer reports impl_l21_phifirst_2026-09-23.md and impl_durprior_2026-09-23.md.
- The spec SAE_4A_lexlat_v2.md: L2-1, A1-A3, A7, A8 and A9.

Abbreviations: C = .../librispeech/configs/config_sae_4a_lexlat_v2_em_v1.py,
M = prefix_lm/model/definitions/sae_blankfree.py, T = prefix_lm/model/train_steps/sae_blankfree.py,
R = sae/emc/blankfree_phifirst_probe_read.py, J = sae/emc/blankfree_phifirst_jobs.py,
G = sae/emc/blankfree_genmarg_jobs.py.

## 1. The null recognizer is in effect: PASS
- M:642 refuses null_recognizer unless freeze_recognizer is set. It also refuses
  zero_reverse_emission, ent, cons, bt, selfdistill, a nonzero anchor, content, soft and sf.
- M:721 null_log_q returns a constant, -log 40.
- T:63-64 swaps in the null log_q. The recognizer forward never runs; T uses
  recognizer.output_lengths, which is a length function only.
- T:125 and T:134 mark agg and the rate term as_error: they are logged but carry no gradient.
- T:149 skips the tilted FD passes. T:268 is a second guard.
- l_tau enters at scale lam_tau = 1.0, so phi's gradient comes from the lattice term alone.
- emc_param_groups skips requires_grad=False parameters and asserts that nothing is trainable
  outside the recognizer and reverse. Under freeze, theta has no parameter group.

## 2. A3 implemented: PASS
All of the following were checked on the written config (cfg of TVCw6EcU5ahd):
- learning_rates [1e-4]*4 x phi multiplier 30 gives a constant 3e-3. No LR control, warmup or
  dynamic LR keys are present.
- gradient_clip_global_norm 5.0; Adam weight_decay 0.0.
- betas [0.9, 0.999] and eps 1e-8. These are the torch.optim.Adam defaults of the only phi-alone
  precedent, the supervised reverse fit (reverse.py:635, blankfree_reverse_init_jobs.py:198, FitConfig
  reverse.py:97-100).
- temperature_schedule [4, 1, 1, 1].
- num_epochs 4 with partition_epoch 4 and laplace:.1000 over train.segments (PfpCPQRCfIAk), so 4
  sub-epochs are one pass.
- cleanup_old_models keep [1, 2, 3, 4], keep_best_n 0.
- stage1_config (C:278) asserts all of the above on every written config.

Scoring:
- Each checkpoint gets a genmarg marginal plus a gendecode on GenMargSampleJob over the CV holdout
  (the same cv.segments as the training dev set, 285 utterances, disjoint from train.segments).
- The marginal is taken at tau = 1 with the log_q constant removed.
- The emitted rate is pooled as 50 x non-SIL tokens / original frames (A8). The expected rate is
  report only.
- The lattice arguments equal the probe's: band 25, prior RtzbESkOedsT at weight 1.0, eta
  U4etvcSpsQi4, matmul reduction, float64.

## 3. The six arms differ only as intended: PASS
Probe versus bed: I compared the probe's written config key by key against the bed reference
WEmvqF2yBFG8/supphi_plain/returnn.config. The complete list of differences:
- keep [1,2,4,8] becomes [1,2,3,4].
- null_recognizer and freeze_recognizer are added.
- recognizer_checkpoint_path points to the flat init. The recognizer is loaded but never run under
  the null.
- reverse_checkpoint_path is removed, so phi starts cold.
- tau [2]*8 becomes [4,1,1,1].
- learning_rates [1e-5, 1e-4 x7] becomes [1e-4]*4.
- num_epochs 8 becomes 4.
- betas/eps (0.5, 0.98)/1e-6 become (0.9, 0.999)/1e-8.
- random_seed and random_seed_offset are added.
- torch_log_memory_usage is added.
Every item is part of the declared stage-1 delta or is a monitoring switch.

Across the six arms:
- s1 and s2 differ only in random_seed (the engine's model-init seed) and the feats
  random_seed_offset. With partition 4, the order seed is 1 + s, constant within the pass
  (C:317-323).
- durinit and durfrz each equal their uniform counterpart plus reverse_duration_prior and its mode
  (C:294).
- The prior path is the job output of BlankfreeDurationPriorMeanJob.ReQtJKYpZgsN (C:181, C:243).
- Under freeze, the M:703-708 hook zeroes the phone-row gradients only. With wd 0 the phone rows move
  exactly 0, while the SIL row moves (engine test: 26.0 to 25.9457).

## 4. The reader's rule: PASS
- R:214-215: gain = S(ep3) - S(ep4), paired over utterances possible in both, compared with
  < 0.01.
- R:216-220: rate = the ep4 gendecode emitted pooled_hz in [5.80, 14.49].
- Any None among the clauses gives CANNOT_TELL.
- The wave setting is the passing candidate with the lowest mean S at ep4, paired over every passing
  json (R:258-263); this is A9's "higher mean held-out tau = 1 marginal". No pass gives NO WAVE
  SETTING; a tie gives TIE. Uniform is report only.
- The projection is startup + 4 x (train + dev) of sub-epoch 1 (J:348-351). The watcher regexes
  match RETURNN's log formats.

## 5. Resources: PASS
- Training asks for 1.65 h (1.5 x 1.1), 1 GPU, 16 CPUs, 64 GB and 96 GB of GPU memory. The expected
  wall is about 0.4-0.75 h for 4 sub-epochs plus startup, with theta not running.
- Memory: 15.8 GiB was measured at batch 88000 / 128 seqs. The bed, which also runs the recognizer,
  is an upper bound.

## 6. The shared prior job: resolved by events
- The D17 manager (pid 2658727, config/sae_4a_lexlat_durprior.py) started at 16:49:03.
- ReQtJKYpZgsN finished at 16:50:06 with output/prior.json mean_frames 4.413787447505655.
- A finished job shared between managers is safe, so the probe manager may start now.
- At the time of review, no probe manager was running and no PhiFirstProbeTrainingJob dir existed.

## 7. Dry-load, census and overlap: PASS
- The shim loads in sis console without the engine.
- The census is 57 jobs: 48 ReturnnForwardJobV2, 6 PhiFirstProbeTrainingJob, the prior job,
  GenMargSampleJob and PhiFirstProbeReadJob.RuFm51PHSz4q. All were new at 16:47, except the prior
  job, which is now finished.
- None of the 57 is in the graphs of managers 1770648, 2228146, 3640835, 3600681 or 3927592.
- The D17 graph (70 jobs) overlaps only on ReQtJKYpZgsN.

## Notes (non-blocking)
1. G GenMargSampleJob.b7aFZd9Tse5X is also in the graph of the unlaunched full ladder shim
   config/sae_4a_lexlat_v2_ladder.py (220 jobs). Whichever of the two managers starts second should
   start only after this job has finished.
2. G:1456 builds 48 GPU forward jobs at 2 h each. The genmarg forward has not yet run as a real
   sisyphus job, so the first real-job failure, if any, would show at these reads rather than in
   training.
3. R:218 requires the projection clause for BOTH seeds, where A3 says "a restart projects to
   <= 1.5 h". This is stricter than the spec. The outcome only changes if one seed projects above
   1.5 h, which is unlikely at the expected 0.4-0.75 h, and a projection FAIL only re-derives the
   budget.
4. J:351 leaves out the gaps between sub-epochs (checkpoint save, dataset re-init). measured.hours
   is reported beside the projection.
5. The prior mean m uses the lengths of all 28539 train-shard utterances, including the 285 CV
   holdout utterances. These are label-free lengths and the effect on m is negligible.
