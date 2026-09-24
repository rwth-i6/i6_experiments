# Tests closing review findings 1-3 (rate clock, agg weights, train step -> k2 plumbing), 2026-09-24

Status: DONE. No package code changed; no FINDING (in all three places the code matches the method).

## What was added
Appended section "T2.5" to `PKG/tests/test_model_blankfree.py` (+~200 lines; no other file edited).
Every pinned constant is a literal from the method, not read from the code's config:

- `test_t2_5_ctrl_20_rate_clock_is_50_hz`: the model built from ctrl_20's own `get_model` args has
  `lattice_cfg.frame_rate_hz == 50.0` (the value `train_step.py:110` divides rho by) and rho 9.6619373279.
- `test_t2_5_rate_term_follows_original_length_at_50_hz`: same ctrl_20 model, same batch (S 6/5/1),
  original lengths [11,5,4] then [23,9,4]. E[N] comes from the enumeration oracle and is identical in
  both runs. The step's rate term equals mean_kept(((E[N]/(orig/50) - rho)/rho)^2) to 1e-10 in each
  run and in the difference (0.57137 against 0.19274).
- `test_t2_5_agg_weights_are_the_bed_s`: the ctrl_20 model's agg weights are (1.0, 1.0). The step's
  agg equals 1*KL_uni + 1*KL_bi, computed from path-enumerated run counts: 23.10601 against the
  oracle's 23.10601, within 1e-5 (float32). KL_uni is 6.76 and KL_bi 16.34, so a weight change is
  far outside the tolerance.
- `test_t2_5_train_step_k2_plumbing[7,8,9,10,20]`: the model is built from the k2lat_20_ma3000 arm's
  args, with a recording fake in place of `LexlatK2Runtime.step`. Asserted: `retained == S` ([6,5,1],
  not original [11,5,4], not T); `feat_lens == ceil(S/3)` ([2,2,1]); `keep == [1,1,0]`, which equals
  ~z_zero; epoch and tau(e). The marked loss equals B4's mean_kept[(-L_lex)/S], and its scale equals
  the literal lam_lex (0, 1/3, 2/3, 1, 1). total - (l_tau + 3 rate + 0.1 agg) equals lam x term to 1e-6.

## Reference agg weights and their source
The reference is 1 and 1: SAE_4A.md:126-127, the JUPITER pre-registration of 2026-09-15, reads
"lambda_agg = 0.1 on KL(c_text || c_hat) summed over unigram and bigram terms". SAE_i6_ref_objective
section 4.3 and the SAE_i6_ref_blankfree section 1 L_agg row give the same unweighted form.
SAE_4A_blankfree.md "Registered first model" and SAE_4A_attrib.md state no per-order weight.

## Mutation checks (scratchpad plugin `-p mutplug`, applied in-process; no package file edited)
- M12, frame_rate 50 -> 50/3 (a `LatticeConfig.__post_init__` wrap): both rate tests fail. The
  clock assert shows 16.67 against 50; the end-to-end check shows step 0.312 against oracle 0.571.
- M13, agg_bigram_weight default 1.0 -> 0.5 (`__kwdefaults__`): the agg test fails with (1.0, 0.5)
  against (1.0, 1.0).
- M14, retained=original (an exec'd mutated copy of train_step.py): all 4 active k2 cases fail with
  retained [11,5,4] against [6,5,1]. Epoch 7 passes, as it should, because the step is not called.
Logs: session scratchpad mut_M1{2,3,4}.log.

## Checks
- New tests: 8 passed (7.8 s).
- Full suite, run once: 503 passed, 14 skipped, 12 xfailed, 0 failed (66 s). This is 495 plus the 8 new tests.

## Undetermined / notes
- lam_lex at sub-epoch 10: B4 says the sources disagree. The D15 log (what ran) gives 1.0, while the
  DP design text puts full weight from 11. The test pins the D15 log value, which the code also gives.
- M13 fails at the config assert first. The end-to-end agg comparison would also fail under it
  (8.17 nats off against a 2.3e-4 tolerance), but that failure was not run separately.
- The fake stands in for the runtime's own reduction and its empty-lattice handling (numerator only).
  Those are T1.20's (`test_model_lexlat_k2.py`, k2 fixture). The real runtime cannot sit under the
  train step on the T1.19 fixture, because the fixture has 4 phones and the bed has 40.
