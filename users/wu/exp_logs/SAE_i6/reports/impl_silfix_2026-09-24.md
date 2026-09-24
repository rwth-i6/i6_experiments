# SIL-run split fix: `sil_run_collapse` option and arm `ctrl_20_rc` (2026-09-24)

Status: DONE_WITH_CONCERNS. The option is in place behind a hashed model argument, and the default
path is bit-identical: every existing job id is unchanged, and the default train step matches HEAD
bitwise. With the option on, T1.6 matches the run-collapse oracle to 1e-10 through the train step's
own arguments. The concerns are listed at the end: the D_sil analysis shows that run collapse forces
a non-SIL token into every silence longer than about 1 s, the rate of such silences on retained
frames has not been measured, and nothing has run on a GPU.

The live `recipe/` tree was not touched, nothing was committed, and no branch was created. The work
was done in the detached worktree `/work/asr4/hwu/tmp_dev/silfix` at f35f5f1db. The live HEAD is
dc52c8783, which differs from f35f5f1db only in `SAE_i6_P0.md`, and `git apply --check` of the patch
on the live tree passes. Only this report was written into the live tree (`reports/`, untracked).

## Deliverables

- Package patch `/work/asr4/hwu/tmp_dev/silfix.patch` (`git diff HEAD`; 6 files, +390/-13):
  - `model/blankfree_model.py` (+27/-1):
    - new keyword `sil_run_collapse: bool = False`;
    - after `lattice_cfg` is switched to "blankfree", when True, `self.prior_history =
      build_prior_history(self.lattice_cfg, self.prior_history_name)`, re-validated with `_use_matmul`;
    - `self.sil_run_collapse` is always set;
    - a class comment.
    - False leaves the history object the base class built under "ctc" untouched.
  - `training/config.py` (+7/-1): `build_train_config(sil_run_collapse=False)`, which writes
    `model_args["sil_run_collapse"] = True` only when True (after `prior_weight_schedule`), plus
    docstrings.
  - `training/arms.py` (+12): `ctrl_20_rc` is `ctrl_20` plus `sil_run_collapse=True`, and nothing
    else. It is added to `__all__` and `ARM_PRESETS`; at 60 sub-epochs it is named `ctrl_20_rc_x60`.
  - Tests: see Checks.
- Setup config diff `/work/asr4/hwu/tmp_dev/silfix_config.diff` (against the live
  `config/sae_i6_p0.py`; `patch -p1 --dry-run` from the setup dir passes):
  - `train_and_read(arms.ctrl_20_rc)`;
  - `paired_delta(ctrl_rc, ctrl, epoch=e)` for e in 1/4/10/20 (tag `ctrl_20_rc_vs_ctrl_20`, delta =
    rc - ctrl, the G0.RC sign);
  - `js_rows("p0_rc", {ctrl_20, ctrl_20_rc}, rows=[ctrl_20_rc_vs_ctrl_20])`;
  - the returned dict gains `ctrl_20_rc`, `rc_delta` and `js_rows_rc`.
  - `config/sae_i6_p0_screen.py` is unchanged.

## Consumer inventory (file:line at f35f5f1db, package `unsupervised_asr/`)

Built or changed by the option:

- `model/emc_model.py:367`: builds `prior_history` under "ctc". It is unchanged; the rebuild happens
  after `blankfree_model.py:481` (the topology switch).

These read `model.prior_history` or `model.lattice_cfg`, so they follow the option by construction:

- `model/train_step.py:86-97`: the dp dict (`history=model.prior_history`) and `lattice_loss`, giving
  l_tau. The same step serves RETURNN's dev evaluation. Affected, and tested (T1.6 rc, T2.1 rc).
- `model/train_step.py:111` `expected_nonsil_tokens(out)`, and `out.expected_tokens`, the prior and
  reverse monitors: all read the same `out`. Affected, and tested (T2.1 rc `e_n`).
- `model/train_step.py:124`, `rate_term.py:256,298`: `_fd_passes` copies dp, including the history,
  for the tilted passes. Affected, and tested (T2.1 rc: theta's gradient vs the FD surrogate).
- phi's training signal: the l_tau DP's `seg_post` through `build_segment_table`. Affected, and tested
  (T2.1 rc reverse gradient).
- `reverse_model/genmarg_steps.py:204-246,375-392` and `genmarg_decode.py` (via `dp["history"]`) would
  follow. However, `reverse_model/genmarg.py:474-478` (`bed_from_train_config`) refuses any model key
  outside `BED_MODEL_ARG_KEYS`, so a genmarg read on an rc bed fails loudly at graph time. No genmarg
  read is registered for ctrl_20_rc.

Not affected (no lattice, or already run-collapse):

- `model/blankfree.py` `expected_run_counts` (agg): already run-collapse, SIL included, from log_q
  alone.
- `model/lexlat_k2.py` `h_topology`, `lexlat_k2_train.py:363,459`: already strict run collapse; their
  docstring (`lexlat_k2.py:38-45`) states SIL->SIL is forbidden. The fix aligns the train lattice with
  them.
- `analysis/posterior_steps.py` and `analysis/per.py`: recognizer only, argmax then run collapse.
- `analysis/gaps.py`: reverse-only model (`reverse_kwargs`) on decoded strings.
- `analysis/jsd.py`: decoded strings.
- `reverse_model/p0_steps.py`: its own `transcript_logprob`.
- The p0 and gold-phi trainings do not use this class's lattice.

## Job ids

- The shadow setup `/work/asr4/hwu/tmp_dev/silfix_setup` has `recipe/i6_experiments` pointing to the
  worktree, plus the live i6_core, returnn and i6_models, and an empty `work/`. It reproduced the live
  graph byte for byte: `live_*.tsv` equals `head_*.tsv` for both configs.
- With the worktree:
  - the unmodified `sae_i6_p0.py` gives the same 139 ids as HEAD;
  - `sae_i6_p0_screen.py` gives the same 75 ids as HEAD;
  - the modified `sae_i6_p0.py` gives 164 jobs, and all 139 old ids are present.
- Unchanged ids include ctrl_20 `ReturnnTrainingJob.GiT88bxzoZbZ`, ctrl_20_s1 `DvVfxf1LrCBi`,
  k2lat_20_ma3000 `jcKXbLMDk4hl`, gold phi `Ac2eioZbRX7d` and p0 `Y1vbqR6KeJSx`.
- 25 jobs are new, all ctrl_20_rc's:
  - training `ReturnnTrainingJob.llSFybyKXkbL`: the same rqmt as ctrl_20 (16 cpu, 64 GB, gpu_mem 96,
    11.5 h), so it goes to gpu_48gb. Its model args are ctrl_20's 16 plus `sil_run_collapse=True`.
  - 5 checkpoint extracts;
  - 4 posterior dumps (gpu_mem 24, so gpu_24gb);
  - 4 PER jobs;
  - decode and derangement gaps, with 2 gap-item jobs and 2 CPU reverse-score forwards;
  - 4 `PairedPerDeltaJob`s (`ctrl_20_rc_vs_ctrl_20/ep{1,4,10,20}`);
  - `JsRowsReadJob.8fdSriRNB5z0` (`js_rows/p0_rc`).
- Id lists are in `/work/asr4/hwu/tmp_dev/silfix_setup/probe/`: `head_*.tsv`, `live_*.tsv`, `wt_*.tsv`
  and `new_jobs.tsv`.

## Checks

- Full CPU suite, worktree first on PYTHONPATH: 526 passed, 14 skipped, 12 xfailed in 105 s
  (`/work/asr4/hwu/tmp_dev/logs/suite_after.log`). The baseline was 503/14/12, so there are 23 new
  tests and the same 12 xfails.
  - The two T1.6 strict xfails (`test_t1_6_log_z_vs_run_collapse_oracle`,
    `test_t1_6_model_history_is_the_blankfree_one`) still xfail on the default model. Their reason
    text now names the option.
- Default path, bitwise: T2.1's train step (eps 1e-4 and 0.25) was run with HEAD's package and with
  the worktree. All 72 tensors (log Z, the total, every logged loss, every parameter gradient) are
  `torch.equal` (`probe/bitid.py`, `head.pt`, `wt.pt`).
- T1.6 rc (`test_model_lattice.py`, 3 tests):
  - The captured log Z equals the run-collapse oracle with diff 0.0 on both utterances (tolerance
    1e-10).
  - Every l_tau argument except the history, and the step's log_q and seg, equal the default's
    bitwise. The history is the model's own, rebuilt from the blank-free cfg, and it differs from the
    default only in SIL's diagonal.
  - The SIL-split oracle no longer matches on utterance 0: the diff is 3.9e-8, 18 times the
    tolerance. On utterance 1 the diff is 1.6e-11, inside the tolerance, because at this tiny shape the
    split latents are 3 of 4723 and 2 of 3162. The test therefore requires the mismatch on at least
    one utterance, and a strict decrease of log Z on every one (G0.RC: rc log Z <= ctrl_20's).
- T2.1 rc (`test_model_blankfree.py`):
  - The assembly at eps 1e-4 and 0.25 against the run-collapse oracle passes: log Z and E[N] to 1e-10,
    l_tau and rate to 1e-10, theta's logit and parameter gradients to 1e-6.
  - The reverse gradient equals the l_tau-only run bitwise, and matches the oracle (deviation 0.0).
  - The rc l_tau (4.099256576283) differs from the default's (4.099256573810) by 6e-10 relative, so
    the 1e-10 check separates the two latent sets.
  - Plus `test_sil_run_collapse_moves_only_the_history`: the default history equals the ctc-built one
    table for table; the rc history equals the blankfree-built one; the state_dicts are identical.
- `tests/test_model_sil_run_collapse.py` (new, 16 tests):
  - The config key is present only when True.
  - Removing it restores the base hash.
  - `ctrl_20_rc` (20 and 60 sub-epochs) is ctrl_20 plus that one key: undoing it gives ctrl_20's
    hash, and the job is distinct with an equal rqmt.
  - The D_sil cases below.
- NOT run: nothing on a GPU (this desktop has none), and no training. G0.RC(A) "step 1 of ctrl_20_rc
  has the same batch and inputs as ctrl_20, log Z <= ctrl_20's" is shown only at test shapes.

## D_sil = 50: what enforcing run collapse on SIL does to long silences

These cases are built on the DP at the bed's W 25, d_min 2, D 25, D_sil 50 and stride 3. They are
cross-checked by an independent feasibility recursion over the oracle's band (`lattice_oracle._in_band`).

- **The binding cap is the band, not D_sil.** On every frame of a run, the token's end s must stay
  within W of 3t. So one token spans at most floor(2W/3)+1 = 17 recognizer frames (about 1.02 s)
  inside an utterance, whatever its type. The DP and the recursion agree on this for R = 10..23.
  - At the start, s_0 = 0, so the first token's unit span is at most W+3 = 28 units (0.56 s).
  - At the end, a trailing run is at most 10 recognizer frames (0.6 s).
  - D_sil = 50 then caps the SIL token's unit span. It binds only after the band.
  - Phones were already under these caps before the fix. The SIL split was the only way the default
    covered a longer silence.
- **Such paths are infeasible, not forced into z_zero.** Under the option, a recognizer path with a
  SIL run over the cap has no reading and gets weight 0. Its mass moves to paths that put a non-SIL
  frame and token inside the silence (SIL X SIL). Under the default the same path is read as several
  SIL tokens.
- Test case, an all-silence utterance (SIL 0, other symbols -30 nats):
  - Expected non-SIL tokens under rc are 0 up to S = 28, 1 for S = 29..75, 2 at 100, 3 at 150.
  - The default inserts none, but reads the silence as about S/4.5 SIL tokens (66 at S = 300).
  - At -10 nats the rc inserts about 1 non-SIL token per 30 units.
- **z_zero and masking are unchanged.** Z stays > 0, because a non-SIL alternative always exists. On
  random inputs for S = 1..60, the z_zero rows are identical with and without the option (S = 1 only),
  and rc log Z <= default log Z throughout.
  - The step masks z_zero rows out of l_tau and rate via `keep`. It does not mask them out of agg.
  - A long-silence utterance is therefore NOT masked. It stays in the l_tau and rate means with a
    larger l_tau.
  - Theta's gradient pushes a non-SIL class up at one frame inside the silence, and the rate term
    counts the forced token.
- **How often.** Internal inter-phone gaps longer than 50 units (1.0 s) in the MFA alignments
  (`get_mfa_alignments`) on ORIGINAL audio, before rVAD:
  - dev-other: 76 of 6677 gaps, in 62 of 2864 utterances (2.2%); median gap 9.5 units; maximum 105.5.
  - dev-clean: 39 of 6015 gaps, in 35 of 2703 utterances (1.3%); maximum 79.5.
  - rVAD removes about 60% of gold SIL frames, so these are upper bounds.
  - Retained-frame figures, edge silences and the train-clean-100 rates are NOT measured: the VAD job
    had not finished, and the train MFA data is not downloaded.
- Watch in the G0.RC reads: PER insertions inside long pauses, and the emitted rate.

## Assumptions and open points

- The JS row is a separate `JsRowsReadJob` (`p0_rc`), so the three-arm `p0` JS job keeps its id. The
  brief did not say "same job".
- The option does not extend genmarg: `BED_MODEL_ARG_KEYS` still refuses the key. Extending it would
  be a separate change if an rc genmarg read is wanted.
- The worktree's index has an intent-to-add entry for the new test file (needed for the patch). No
  commit was made.
- Scripts: `/work/asr4/hwu/tmp_dev/silfix_setup/probe/` (`bitid.py`, `explore.py`, `emb.py`, `rqmt.py`)
  and `/work/asr4/hwu/tmp_dev/run_pytest.sh`.
