# LexlatStepProfileJob: the identity check killed a measured step (fix 1)

DONE. Hash did NOT move: `speech_llm/sae/emc/lexlat_profile_jobs/LexlatStepProfileJob.ItFHebFGK6nb`
(same id before and after the edit, read by loading `config/sae_4a_lexlat_profile.py` with
`sis_env`). Commit `10ccb37` on `haotian_modality_matching_jupiter` in `recipe/2025-10-speech-llm`
(not pushed).

## The defect

`LexlatStepProfileJob.ItFHebFGK6nb` ran 1 h 14 min, measured the reference, timed, banked and
profiled steps of batch 0, exported the 574 MB trace, and then died in `_loss_equality`:
`AssertionError: the timed step's loss 'blankfree_frames_per_sec' moved by 9.488e+01 relative`
(tolerance `LOSS_REL_TOL = 1e-6`). Because the per-batch `_write(partial=True)` came AFTER that
assertion, `summary.txt` held only the header: no phase table, no per-frame rows, no kernel table.

`blankfree_frames_per_sec` is `retained.sum() / elapsed` with `elapsed = now -
model._last_step_time` (`prefix_lm/model/train_steps/sae_blankfree.py:36,217`) -- a function of the
wall clock. The phase timers add a `cuda.synchronize` per wrapped call, so this monitor MUST move;
asserting on it asserts that the instrumentation has no cost, which is the opposite of what the job
measures (the job reports that cost as the timed/reference ratio).

## The change (2 files, both in `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/`)

`lexlat_profile_jobs.py` (+103 lines, no signature change, `__sis_version__` untouched at 1):

* new class constant `LOSS_TIMING_MARKERS = ("per_sec", "sec_per", "time", "elapsed")` with the
  reasoning in its doc comment, and module-level `is_timing_monitor(name, markers)` (case-folded
  substring test; module level so it is callable on the plain namespaces the tests use).
* `_loss_equality` now only BUILDS the table. Every key carries `timing_monitor` / `checked`, and
  the table carries `timing_markers` and `excluded_keys`. A timing monitor keeps its reported
  `value`/`abs_delta`/`rel_delta`; it is just not asserted on.
* new `_assert_loss_equality(eq)` (staticmethod) raises: `total_loss` per step, then every key with
  `checked` true, against `rel_tol`.
* `run()` order is now: build the table -> `batches.append(entry)` -> `_write(partial=True)` (the
  phase table, `per_frame.json`, `profile.json` with `partial=True`, the kernel table) -> assert.
  On `AssertionError` the entry gets `loss_equality["identity_failed"] = str(exc)`, the report is
  rewritten, and the error is re-raised.
* `_summary` renders two new lines: the excluded monitors ("wall-clock monitors reported but NOT
  checked: ...") and, when set, `IDENTITY CHECK FAILED: <msg>` with a one-line reading rule.

### Keys excluded on this arm, read off the train step

The arm's step is `train_steps.sae_blankfree.train_step` (wired by
`blankfree_train_jobs.py:225`). Exactly ONE of its keys matches the markers:

* `blankfree_frames_per_sec` -- EXCLUDED (wall clock).

Everything else stays under `LOSS_REL_TOL`: the loss terms `l_tau`, `agg`, `rate`, `soft`, `sf`;
the monitors `blankfree_{l_tau_per_frame, agg_kl_unigram, agg_kl_bigram, reverse_per_frame,
prior_per_token, phone_rate_original_hz, phone_rate_retained_hz, expected_phone_rate_hz,
expected_tokens, z_zero_frac, rate_fd_check, rate_dp_calls, temperature}`; the `lexlat_*` monitors
of `lexlat_train.LexlatRuntime.monitors` (all deterministic functions of the batch); and the
`soft_*` / `sf_*` / infomax keys, none of which contains any marker substring (checked by grep).
`total_loss` is unaffected by the monitors (they are `as_error=True`).

`test_lexlat_profile.py` (+63 lines): `test_a_throughput_monitor_does_not_fail_the_identity_check`
(only `blankfree_frames_per_sec` differs, by 94.877 -> `_assert_loss_equality` passes, the key is in
`excluded_keys`, `checked is False`, its delta is still reported) and
`test_a_real_loss_term_moving_still_fails_the_identity_check` (`rate` differs by 1e-3 ->
`pytest.raises(AssertionError, match="'rate' moved")`). The existing summary test gained two
asserts so the new rendered lines cannot be computed-but-never-printed.

## Checks run

* `pytest speech_llm/sae/emc/test_lexlat_profile.py -q` under the conda env
  (`/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python`, PYTHONPATH per the toolchain
  note): **8 passed in 19.4 s** (6 pre-existing + 2 new).
* Hash read before AND after the edit with
  `python tools/sisyphus/sis console --script -c "..." config/sae_4a_lexlat_profile.py` under
  `sis_env`: both print `LexlatStepProfileJob.ItFHebFGK6nb`. The job dir was not touched.
* No added line exceeds 101 columns (the file's existing width).

These are unit checks of the identity table and the render. They do not prove the on-GPU ordering
change (report-before-assert) takes effect; only a rerun does.

## Left undetermined / not done

* The finiteness assertion (`total_loss_finite`, just above the append) still runs BEFORE the
  write, so a non-finite loss would still fail with no phase table. The dispatch named only the
  identity assertion; moving it too is a one-line change, not made.
* `__sis_version__` was NOT bumped, so a rerun resumes into the existing failed dir
  `work/.../LexlatStepProfileJob.ItFHebFGK6nb`. Clearing it (error marker AND `submit_log`) is the
  executor's call; nothing was deleted or launched here.
* Whether the timed step's values actually reproduce the reference's on GPU is unknown: the run
  never reached that verdict for any non-timing key, since `blankfree_frames_per_sec` was compared
  first in dict order. The rerun now answers it.
