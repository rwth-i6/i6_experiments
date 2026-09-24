# S3b-R rate term -- pre-compute code review (2026-09-15)

Verdict: **PASS_WITH_CONCERNS**. The term is real, wired, label-free, hash-neutral, and the three
arms differ from S3 in exactly the four rate keys. Nothing here blocks the GPU spend. Three
concerns below; F1 affects how the gate number is read, F2 is a wasted change, F3 is a units
discrepancy between the dispatch text and the implemented (phase-file) form.

Reviewed: commits `dddbb56`, `629d5ae` on `haotian_modality_matching_jupiter` in
`recipe/2025-10-speech-llm` (working tree is at `d69f938`; the census below isolates the two
rate-term commits by restoring only their six files). No file was edited by this review.

## Findings

**F1 (read defect).** `src/speech_llm/sae/emc/emc_train_jobs.py:201`
`TEXT_PHONE_RATE_HZ = 9.8  # dev text phone rate (SAE_1f.md:533-536)` -- the GOLD dev rate the spec
declares reporting-only. `DecodeStatsJob` (`emc_train_jobs.py:867-868`, constructed with defaults at
`emc_train_jobs.py:1362`) writes `rate_band: [5.88, 14.70]` and the boolean `rate_in_band` into
`decode_stats.json`, which `config_sae_4a_s3b_rate_v1.py:207-209` registers per sub-epoch per split.
The pre-registered G4a.3b-R band is `[0.6, 1.5] x rho = [5.797, 14.493]` (config docstring, line 41).
Failure path: an arm decoding at 5.80-5.87 /s (or 14.50-14.70 /s) reads IN BAND off the registered
artifact and OUT of band under the gate as written. No training or selection defect -- the gold
number enters no loss and no selection -- but the gate must be read from the separately registered
`{base}/{split}/phone_rate` output var (the raw non-SIL phones / audio-second, `emc_train_jobs.py:953`)
compared by hand against `[5.797, 14.493]`, NEVER from `rate_in_band`. Pre-existing job, unchanged by
this commit; it becomes a hazard only because G4a.3b-R re-bases the band on rho. Fixing it in the job
would move S3's hashes, so the fix is a reading rule, not code.

**F2 (a change that buys nothing at this operating point).** `src/speech_llm/sae/emc/rate_term.py:530`
(`_stacked_fd_call`) / `:514` (`fd_batch_fits`). Commit 629d5ae exists to make the two tilts one DP
call. Confirmed independently that it never fires here: the resolved config of arm `DF6blPpto23t`
carries `max_seqs = 128` (line 69) and `batch_size = 88000` (line 39), and
`fd_batch_fits(128, 500, 2)` returns
`False, '2 x B = 256 > MAX_UTTS_PER_BATCH = 128 ... ~8.70 GiB (one pass: ~4.35 GiB)'`.
So every step of all three arms takes the sequential fallback: 3 DP calls, `emc/rate_dp_calls` = 2.
The fallback is correct (see V5) and disclosed in the config COST block and in the implementer's
report; the point is only that the second commit's stated benefit is zero unless `lattice.py`'s caps
are raised (off limits) or `max_seqs` is halved (which would change the realized batch the S3 arms
are compared at, i.e. a second delta -- do not do it). Cost consequence: the ~45 min/arm projection
in the config is unmeasured; read `emc/sec_per_step` and `emc/rate_dp_calls` off the first 100 steps
before trusting `time_rqmt = 6.0`.

**F3 (units, dispatch vs implementation).** `rate_term.py:640` implements the RELATIVE form
`((rate - rho)/rho)^2`, which is what `SAE_4A.md` S3b-R states. The review dispatch stated the
ABSOLUTE form `(rate - rho)^2`. They differ by `1/rho^2 = 26.8`, so the arm grid lam_rate 1/3/10 in
the code equals 0.037/0.112/0.37 in the absolute parameterization. The code follows the phase file
(the spec of record); flagged only so the lam values are not compared against an absolute-form
intuition when the gate is read.

**F4 (operational, low).** `config/sae_4a_s3b_rate.py` contains the S0b graph and shares
`UnitsHdfJob` with S1b/S2/S3; `config/sae_4a_phase.py` (1021 jobs, currently managed) contains S3 and
S0b. Launching the rate wrapper as its own manager while the phase manager runs puts two managers on
the shared init/units jobs. The wrapper's docstring says so; either add the rate config to the phase
manager or stop the phase manager first.

## Checks completed (evidence)

V1. **Hash census, replicated independently** (not taken from the report). Shadow tree = current
`src` with `definitions/sae_emc.py`, `train_steps/sae_emc.py`, `emc_train_jobs.py` restored from
`dddbb56~1` and `rate_term.py`, `test_rate_term.py`, `config_sae_4a_s3b_rate_v1.py` deleted; both
graphs enumerated under the sisyphus venv from the setup dir.
`config_sae_4a_s3_v1` (+S0b): **98 jobs before, 98 after, sorted id lists diff to nothing.**
`config/sae_4a_phase.py` (S2b+S2c+S3+decode_temp): **1021 before, 1021 after, identical.**
Scripts: scratchpad `census.py`, outputs `s3.before/.after`, `ph.before/.after`.

V2. **Arm ids** rebuilt from the config: `WCh1fMyr88yu` (lam1), `DF6blPpto23t` (lam3),
`axf11lrsap5i` (lam10) -- match the reviewed values; no job directory exists on disk for any of them,
so nothing is orphaned or re-funded.

V3. **One delta only, at the resolved config.** Serialized the RETURNN config of the S3 job
(`sBlPYBA1YcIQ`) and of `DF6blPpto23t` and diffed them. The entire difference is:
`'lam_rate': 3.0, 'rate_rho_hz': 9.6619373279, 'rate_fd_eps': 0.25, 'rate_fd_mode': 'central'`
in `model_args`, plus the `model = .../DF6blPpto23t/output/models/epoch` path. Identical
`temperature_schedule [8.0, 5.04, 3.17, 2.0, 2.0, 2.0, 2.0, 2.0]`, `anchor_weight_schedule 0.0`,
`lam_agg 0.1`, `count_ema_decay 0.99`, `band 25`, `prior_weight 1.0`, `prior_order 2`, prior npz,
eta npz, recognizer_kwargs, `freeze_recognizer False`, flat init checkpoint, `batch_size 88000`,
`max_seqs 128`, optimizer, seed, data and segment lists. `rate_fd_batched` is correctly absent
(default True), which is why the arm hashes are the reviewed ones. Read differences vs S3 are a
strict subset (no word-decode chain, no `SelectPosteriorsJob`), stated in the config docstring.

V4. **Not a silent no-op; the flag reaches the model and the model reaches the step.**
`emc_train_jobs.py:452-460`: `PartialImport(SaeEmcModelV1, hashed_arguments=model_args)` ->
`get_model`, and `Import(speech_llm.sae.emc.emc_train_jobs.train_step)` -> that wrapper calls
`prefix_lm.model.train_steps.sae_emc.train_step` (`emc_train_jobs.py:314,325`).
`definitions/sae_emc.py:274-291` assigns `self.lam_rate` and refuses `lam_rate != 0` without a
positive `rate_rho_hz`/valid mode/positive eps at construction. `train_steps/sae_emc.py:212-227`
reads `model.lam_rate`, converts `rho_hz / cfg.frame_rate_hz` to tokens per frame and calls
`rate_loss`; `mark_as_loss(rate_term, "rate", scale=lam_rate)`.

V5. **Gradient reaches theta; stacked == sequential; phi untouched.** Ran my own check (not the
implementer's suite): B=4, T=40, default `LatticeConfig`, bigram history, tau=2, rho=0.1932387,
eps=0.25, central.
fp64: loss 3.18540725, `|grad log_q|_1 = 1.3828e+00`, `seg.grad is None`, `fd_check 8.57e-05`.
fp32: loss 3.04205227, `|grad log_q|_1 = 1.6329e+00`, `seg.grad is None`, `fd_check 1.04e-04`.
batched=True vs batched=False: **max|dloss| = 0.000e+00, max|dgrad| = 0.000e+00** at both dtypes,
`n_dp_calls` 1 vs 2. So the stacked path is bit-identical and the fallback is silently correct.
The small `fd_check` confirms the Gibbs identity `tau dlogZ/db = E[N_nonSIL]` the surrogate rests on.
Sign: `coef = 2(rate-rho)/(rho^2 T)` (`rate_term.py:637`) is negative under a collapse, so descent
moves `log_q` along `+g` and raises the rate -- the intended direction.
Chain of detachment is right: `coef`/`g`/`value`/`keep` all come from a detached DP output or a
`no_grad` block, `log_q` is the only grad-carrying factor, and
`loss = surrogate + (value - surrogate).detach()` puts the exact value forward.

V6. **Non-SIL only.** `rate_term.py:334` `nonsil_type_mask` zeroes `cfg.sil_id` (39); `:341`
`tilt_segment_table` broadcasts `b * mask` over (d, s) so SIL and blank entries are bit-identical;
`:357` `expected_nonsil_tokens` = `seg_post.sum(2,3).sum(1) - mass[:, sil_id]`, explicitly NOT
`out.expected_tokens` (which includes SIL). Illegal entries are the finite -1e30 floor and are
unchanged by a +/-0.25 shift.

V7. **rho is label-free and arithmetically right.** `analysis/out/rho.rate_term.txt`:
2,784,159,269 phones / 778,025,128 words = 3.5784953066 phones/word; x 2.7 words/s = 9.6619373279 /s;
/50 Hz = 0.193238746559 tokens/frame. Recomputed: 3.57849530664, 9.661937327942, 0.193238746559 --
matches to every digit written, and `[0.6, 1.5] x rho = [5.797162, 14.492906]` matches the config's
`[5.80, 14.49]`. `verified_phone_count == phones`, i.e. the lexicon+G2P replay reproduces T_phi's own
token count exactly, which also establishes that T_phi carries no SIL token, so rho and E[N_nonSIL]
are the same currency. `compute_rho` (`rate_term.py:190`) has no gold/transcript/alignment parameter
and opens only T_phi, its source text and the two lexicons. `RATE_RHO_HZ = 9.6619373279` is the
literal in `config_sae_4a_s3b_rate_v1.py:100`. The gold 9.8/9.4 appears nowhere in `rate_term.py`,
nowhere in the new config, and nowhere in the loss -- except via F1's `DecodeStatsJob` band.

V8. **No gold in selection; no train/eval contact.** `UnsupervisedCheckpointSelectionJob` is
unchanged and reads `cv_loss_key = emc_train_jobs.CV_LOSS_KEY = "dev_loss_l_tau"`
(`emc_train_jobs.py:198`), i.e. the L_tau column only -- adding a `rate` loss column does not move
the selection criterion, and the selection is otherwise the KenLM score on posteriors. The CV set is
the seed-0 1 % holdout of train-clean-100 (`pl_split.out_cv_segments`), label-free; dev-clean/dev-other
are read-only. `S3DerangementGapJob` takes `gold_json` as the eligible tag list only, as in S3.

V9. **Monitors gated on lam_rate.** `train_steps/sae_emc.py:84-93` defines `RATE_MONITOR_KEYS`
(`emc/expected_phone_rate_hz`, `emc/rate_fd_check`, `emc/rate_dp_calls`) and `:271-273` appends them
only when `rate_values is not None`, i.e. only when `lam_rate != 0`. So no other arm's
`learning_rates` gains a column -- consistent with V1's identical censuses.

V10. **c5 exercises the shared rate_term code.** `analysis/emc_cold_fixed_point.py:625` imports
`expected_nonsil_tokens`, `tilt_segment_table`; `:740` imports `solve_rate_tilt`; `:650` imports
`read_rho_cache`, `rho_report_lines`. So rho, the non-SIL tilt and E[N_nonSIL] are literally the same
code as the train step. It does NOT exercise `rate_loss`/`_fd_passes` -- correct, since c5 is a
network-free fixed point with no gradient; the FD surrogate has no counterpart there. `--rate-tilt`
refuses the banked frozen-phi comparison, and at `b = 0` `tilt_segment_table` returns the caller's own
object, so c1-c4 stay numerically untouched.

## Checks NOT reached

- Did not run the implementer's `test_rate_term.py` / `test_sae_emc.py` / `--self-test` suites; V5 is
  my own independent check at fp64 and fp32, and it covers the stacked-vs-sequential and
  gradient-reaches-theta claims but not the Z=0, forward-mode, container-refusal or rho-signature tests.
- No GPU timing. The "3x step, ~45 min/arm, `time_rqmt = 6.0` is enough" claim is a projection from
  S3's 110 s/sub-epoch, not a measurement (F2).
- Did not open `analysis/out/emc_cold_fixed_point.cmd` to check the c5a/c5b command lines verbatim
  against the sbatch wrapper in the implementer's report; only the script's `--rate-tilt` code path
  was verified.
- Did not verify `analysis/out/rho.rate_term.txt` by re-running `compute_rho` over the 40 M-line
  source text (CPU-hours); the internal consistency check (replayed phone count == T_phi's own token
  count, recorded in the cache) was accepted instead.
- The four setup-dir files (`analysis/emc_cold_fixed_point.py`, `.cmd`, `rho.rate_term.txt`,
  `config/sae_4a_s3b_rate.py`) are untracked anywhere; unchanged from the implementer's section 8 and
  left as an orchestrator decision.

## Recommendation

Fund the three arms as they stand. Carry two reading rules into the phase file before the gate is
read: (a) G4a.3b-R's rate clause is read off `{base}/{split}/phone_rate` against [5.797, 14.493],
never off `rate_in_band` in `decode_stats.json` (F1); (b) `emc/rate_dp_calls` is expected to read 2
and `emc/sec_per_step` should be checked on the first 100 steps against S3's 110 s/sub-epoch (F2).
