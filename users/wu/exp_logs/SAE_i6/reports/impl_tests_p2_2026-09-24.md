# Priority-2 tests (T2.1-T2.13): implementation report, 2026-09-24

Status: DONE_WITH_CONCERNS. All 13 plan items are written and green, apart from one strict xfail that
pins a small float32 defect (Finding 1). No package file was changed, and no test file outside my
assignment was touched.

PKG = recipe/i6_experiments/users/wu/experiments/unsupervised_asr. Spec: reports/test_plan_2026-09-24.md
sections 0, 2 and 4. Oracles: tests/lattice_oracle.py, reused unchanged.

## Files (all under PKG/tests/)

| file | change | result |
|---|---|---|
| test_model_blankfree.py | appended T2.1-T2.4 | 13 passed, 1 xfailed |
| test_training_checkpoints.py | new, T2.5 | 6 passed |
| test_analysis_per.py | appended T2.7 and T2.6 | 2 passed |
| test_analysis_paired.py | appended T2.8 | 3 passed |
| test_reverse_genmarg_decode.py | new, T2.9 | 36 passed |
| test_reverse_genmarg.py | appended T2.10 | 1 passed |
| test_analysis_jsd.py | appended T2.11 | 1 passed |
| test_training_init.py | new, T2.12 | 2 passed |
| test_reverse_duration_prior.py | appended T2.13 | 4 passed |

Total: 68 passed and 1 xfailed, in about 17 s on CPU. None of these tests is marked slow. T2.6 ran
(it was not skipped): the RETURNN CPU forward takes about 7 s.

## Finding (pinned by a strict xfail)

1. **The null recognizer's constant is float32.**
   - Test: `test_model_blankfree.py::test_t2_4_null_offset_is_exact_log_40`.
   - Cause: `model/blankfree_model.py:617` builds `null_log_q` as float32 −log 40, which is
     −3.68887949 instead of −3.6888794541.
   - Effect: the null step's l_tau_b·S_b equals (T_rec/τ)·3.68887949 − log Z(log_q ≡ 0). It does not
     equal the plan's (T_rec/τ)·log 40 − log Z(0), nor genmarg's `l_tau2 + uniform_q_offset_tau2`,
     which uses float64 log 40 (`reverse_model/genmarg_steps.py:287`).
   - Size: T_rec·3.6e-8/τ nats per utterance. Measured: 3.58e-8 at S = 6, T_rec = 2, τ = 2.
   - The passing test `test_t2_4_null_recognizer` asserts the same identity with the float32 constant
     to 1e-10 (the difference measured was 0).
   - The effect is negligible for any reported number. It matters only for an identity check at 1e-10.

## T2.1 measurements

Setup: the T1.6 model (band 25, D 25/50, trigram, float64 matmul, checkpoint 2), batch S [6, 5, 1],
original [11, 5, 4], epoch 2, τ = 2.

- **Kept rows.** The oracle finds S = 1 infeasible, z_zero = [F, F, T], and n_keep = 2.
- **log Z and E[N].** Both match the oracle to 1e-10.
- **Loss terms** (step value against oracle value):

  | term | step | oracle |
  |---|---|---|
  | l_tau | 4.099256573809582 | 4.099256573809582 |
  | rate | 0.588916778023622 | 0.5889167780236216 |
  | agg | 23.10601425 (float32) | 23.10601413 |
  | total | 8.17660838 | 8.17660832 |

  - The total equals 1·l_tau + 3·rate + 0.1·agg of the marked values. The residual of 6e-8 is the
    float32 rounding of agg.
  - The scales are (1, 3, 0.1) and none of the three terms is marked as_error.
  - blankfree_expected_phone_rate_hz equals 50·mean_kept(E[N]/orig).
- **Gradient on θ at ε = 1e-4.** The logit gradient differs from the exact oracle by 4.5e-9
  (5.5e-8 relative). The θ-parameter gradients match the oracle gradient pulled back through the
  recognizer to within 4.5e-8 (scaled).
- **Gradient on θ at the bed's ε = 0.25:**
  - The step equals the oracle's own finite-difference surrogate to 5.2e-9, so the assembly is exact.
  - The step differs from the exact gradient by **1.03e-5** absolute. That is **1.27e-4** of the
    largest exact gradient and **4.2e-4** of the rate part's largest gradient. This is recorded and
    not asserted.
  - For comparison, T1.15's g-bias was 1.9e-3.
- **Gradient on φ.** It is bitwise equal to that of a run with lam_rate = lam_agg = 0. It also equals
  autograd of the kept-row mean of l_tau alone, with deviation 0.0. So φ receives gradient from l_tau
  only.

## Other recorded values

- T2.6: the dumped rows equal ConvRecognizer.eval() to 2.4e-7, relative to each frame's max |log q|
  (which reaches 131.6). The argmax is identical on every frame. The PER job's value equals the direct
  PER exactly (8811/6666).
- T2.9: log_w equals the oracle's maximum term to 1e-10 in all 36 cases. The returned latent is
  admissible and scores that maximum. segment_conditionals re-adds the path weight to 1e-10, with
  mismatch 0. This holds for both history builds: the blank-free one, and the ctc-built production
  one, checked against the SIL-split oracle.
- T2.11: max |js − jensenshannon²| is 3.3e-16 over 205 pairs.
- T2.13: the float64 max-entropy table matches the brentq oracle to 1.7e-16. The model's float32 law
  matches to 1.6e-8.

## Choices and gaps in the spec

- **T2.1 oracle semantics.** The oracle enumerates the step's actual latent set, which is the SIL-split
  set. The prior is the float32-rounded log_tri walked over tokens. per_token_log_probs is never used.
  The only Z = 0 row is S = 1, which matmul detects, so nothing relies on the `_logmm` floor.
- **T2.1 gradient comparison.** θ's gradient is compared on the logits (g − q·Σg) and on the
  parameters, not on the raw log_q.
  - Reason: agg's closed-form run counts assume Σq = 1 but the enumeration does not, so their raw
    log_q gradients differ by λ_t·q_t.
  - Measured before projection: 0.046. After projection: 4.5e-9.
  - This is not a defect.
- **T2.1 agg rows.** The agg term counts every row, including the z_zero row. It is not a kept-row
  mean, and the oracle follows the code here.
- **T2.2.**
  - The prior and eta paths are replaced by fixtures, and recognizer_checkpoint_path is dropped.
  - The real default φ dimensions are used.
  - The reverse multiplier is exactly 30.0.
- **T2.3.** The k2 runtime is built from nonexistent graph paths. Its `step` is replaced by a recorder,
  so the real k2 step is not exercised here (T1.20 covers it).
- **T2.6.**
  - The net uses in_dim 16, not the 1024 of NET_ARGS.
  - The data are 2703 random tiny utterances, because the PER job asserts 2703 for dev-clean.
  - The comparison tolerance is 1e-6 relative to the frame's max, because the dump is a batched
    float32 forward.
- **T2.8.** delta_per is compared to 1e-15, because the job computes it as per_b − per_a. Both CIs are
  compared bitwise.
- **T2.12.** Besides in_dim 16, the NET_ARGS checkpoint is also loaded through the model's strict
  recognizer_checkpoint_path.
- **T2.13.**
  - The optimizer step uses torch's Adam with the bed's betas, eps and zero weight decay, not RETURNN's
    updater.
  - The 1e-9 tolerance is applied to the float64 table. The float32 model law is checked at 1e-6.

## Checks

- Full suite, run once with the setup PYTHONPATH:
  - Result: 495 passed, 14 skipped, 12 xfailed, 0 failed, in 64 s.
  - Skipped: 7 artefact, 4 ffmpeg and 3 gpu tests.
  - xfailed: my 1, plus 11 existing ones.
  - Log: /work/asr4/hwu/tmp_probe/tests/full_suite_p2.log.
- A check that passes shows only what it exercised. None of these tests reads a banked artefact or
  runs on a GPU.
