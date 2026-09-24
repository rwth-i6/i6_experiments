# Oracle tests of the lattice and rate terms (T1.1-T1.8, T1.14, T1.15): implementation report, 2026-09-24

Status: DONE_WITH_CONCERNS. The tests are written and green, apart from 4 strict xfails. Those pin two
defects in the package code. No package file was changed.

PKG = recipe/i6_experiments/users/wu/experiments/unsupervised_asr. Spec: reports/test_plan_2026-09-24.md,
sections 0, 2 and 3.

## Files (all new, all under PKG/tests/)

- `conftest.py`: registers the `slow`, `gpu`, `k2` and `artefact` markers.
  - `gpu` is skipped without CUDA.
  - `k2` is skipped unless `import k2` succeeds.
  - `artefact` is skipped unless SAE_ARTEFACT_DIR is set.
  - `slow` is a marker only, so it is deselected with `-m "not slow"`.
  - It does not touch sys.path. As before, the suite needs
    `PYTHONPATH=<setup>/recipe:<setup>/recipe/returnn:<setup>/sisyphus`; without it, every existing test
    fails at collection with "No module named i6_experiments".
- `lattice_oracle.py`: the brute-force enumerator of T1.1.
  - It imports nothing from the package.
  - It lists every (frame path, token reading, segmentation) latent and scores each one as a
    differentiable float64 torch vector.
  - Band readings: "code" (the default), "literal" (the NOTE reading) and "next_frame".
  - `sil_split`: off by default, which is the run-collapse definition. When on, a SIL run may split into
    several SIL tokens; this exists only for the T1.6 diagnosis.
  - Hand-checked on the T1.3 instances: counts 3/0/3 and 2/0/3 match the manual enumeration.
- `test_model_lattice.py`: 81 passed, 4 xfailed, 1 skipped (T1.8 gpu). This includes the slow test T1.7.
- `test_model_rate_term.py`: 23 passed.

## Checks run

- Non-slow set of my two files: 103 passed, 1 skipped, 4 xfailed, 1 deselected, in 3.6 s.
- T1.7 alone: 23 s on the 14-thread CPU.
- Full suite, `pytest -q tests/` with the PYTHONPATH above, run once at the end: 328 passed, 5 skipped
  (4 ffmpeg-pin tests and T1.8), 6 xfailed (4 mine, 2 in the other implementer's
  test_lm_phone_prior.py), 0 failed.
- Logs are in /work/asr4/hwu/tmp_probe/tests/ (full_suite2.log, lattice3.log, rate1.log, t17.log).

## Findings

The code is not modified. Each finding below is pinned by a strict xfail.

1. The production lattice lets a SIL run split into several SIL tokens.
   - Tests: `test_t1_6_log_z_vs_run_collapse_oracle` and `test_t1_6_model_history_is_the_blankfree_one`.
   - Cause: `model/emc_model.py:367` builds `prior_history` while `lattice_cfg.topology` is still "ctc".
     That table exempts SIL from the repeat diagonal (`lattice.py:355-357`).
     `model/blankfree_model.py:481` then switches the topology to "blankfree" but never rebuilds the
     history, and `train_step.py` passes `history=model.prior_history`.
   - Effect: on a SIL run of frames, each frame after the first may open a new SIL token. Each such token
     carries its own prior term and its own segment. These latents are outside the run-collapse
     definition of NOTE §4.1 and of plan T1.1/T1.4(c).
   - Evidence, all at T1.6:
     - The captured log Z of the step equals the SIL-split enumeration with difference 0.
     - The same inputs rerun through the DP with the history rebuilt from the blank-free config equal the
       run-collapse oracle, also with difference 0.
     - Against the run-collapse oracle, the step is off by 3.9e-8 (utterance 0). The gap is small here
       only because this corpus prior gives P(SIL | BOS, SIL) = e^-8.6.
   - Not measured: the effect at the bed. It grows where SIL→SIL has more prior mass, and for SIL runs
     longer than D_sil = 50 unit frames, which can then be covered by several SIL tokens.
   - Owner decision needed: is the SIL split intended? It matches the stale docstring (plan item S10).
2. `_logmm` turns an impossible product into finite mass.
   - Tests: `test_t1_4c_sil_sil_is_not_a_latent[matmul]` and `test_t1_5_z_zero_row[matmul]`.
   - Cause: `model/lattice.py:673-674`. When both operands have finite maxima but no aligned finite pair,
     the product is floored to a_max + b_max − 708 rather than NEG_INF.
   - Effect on the matmul path: a Z = 0 utterance comes out with a finite log Z, `z_zero` False, nonzero
     posteriors, and a finite loss. Measured: log Z = −708.4 in T1.4(c) and −711.7 in T1.5, where the loss
     is 711.7. The elementwise path returns z_zero correctly on the same inputs.
   - Scope: the bed's only reachable Z = 0 case, S = 1, is still detected on both paths
     (`test_t1_5_production_z_zero_s1`). On feasible utterances the spurious mass is about e^-708
     relative, so it has no effect there.
3. Also hit, and already pinned by the other implementer: `model/prior.py:246`.
   `per_token_log_probs(order=3)` double-counts a one-token string. At the orchestrator's instruction,
   the T1.6 oracle walks the float32-rounded trigram table itself instead.

## Recorded measurements

- T1.3, band convention (S2):
  - Instance 1: the code gives exp(log Z) = 3 (3.0000000000000004), which is the code-band count of 3.
    The NOTE's literal reading, |s_end − 3 t_emit| ≤ W, gives 0.
  - Instance 2 (a = A A B, S 9, T 3, W 2, d_max 6): the code gives 2. The reading that checks only the
    emission frame (without the repeat frames) gives 3. The literal reading gives 0.
- T1.15, rate-term finite difference at the bed's eps = 0.25. "g bias" is the maximum of
  |g − exact| / max|exact|, taken per utterance and then maximised over utterances.

  | case | g bias | fd_expected bias against E[N] |
  |---|---|---|
  | W 2, trigram, tau 2 | 1.88e-3 | 3.1e-4 |
  | W 2, trigram, tau 8 | 5.0e-5 | 2.2e-5 |
  | W 40, bigram, tau 2 | 1.60e-3 | 3.4e-4 |

  The plan's provisional bound is 5e-2; 2e-3 would be a safe tightening. At eps = 1e-4 the g bias is at
  most 3.0e-10 and the fd_expected error at most 5.3e-11. The stacked and sequential passes agree to
  1e-12.
- T1.7, fp32 against fp64 log Z, as relative differences: 1.0e-7, 1.1e-7 and 1.4e-6. The fp64 values
  are 3.66, 0.259 and −0.213, so the absolute differences are 3.7e-7, 2.9e-8 and 2.9e-7.
- T1.7, other checks:
  - Elementwise against matmul: at most 8.9e-16 on log Z and 7e-14 on expected_prior.
  - Checkpoints 0, 7 and 32 are bit-identical.
  - Directional derivatives: at most 8.7e-11.
  - Tilt identity: at most 1.5e-11.
- T1.2(e): the maximum relative FD deviation over the 20 coordinates is 7.6e-9.

## Choices and gaps in the spec

- T1.5: the plan's feasible row "S = 10" cannot be feasible at d_max = 2 when T = ceil(S/3), because
  2T = 8. I used S = 4, T = 2 instead, and W = 6.
  - The DP also raises a shape error if T_max exceeds what the seg table's s axis covers, i.e. when
    3(T−1) + 2W + 1 > S_max + 1 + 2W. The train step never produces such an input.
- T1.2(e): FD coordinates are drawn only where the derivative is at least 0.02. At smaller derivatives,
  float64 round-off at h = 1e-6 is too large for a 1e-6 relative check.
- T1.15 "relative" is read as norm-relative per utterance.
- T1.7 and T1.8 use T1.1's random generators (seed 17). The plan says only "random tables".
- T1.8 has not been run, because this host has no CUDA.
- The T1.6 corpus is a copy of test_lm_phone_prior.CORPUS, because that file was being edited
  concurrently.
- T1.3 asserts exp(log Z) = 3 within 1e-12 rather than bit-exact, because the float value is
  3.0000000000000004.
