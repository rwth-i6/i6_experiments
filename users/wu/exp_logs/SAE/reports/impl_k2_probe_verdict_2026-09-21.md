# Implementer: the k2 probe's summary.txt gets a VERDICT line (2026-09-21)

Status: DONE.  Committed as `0c812b1` on `haotian_modality_matching_jupiter` in
`recipe/2025-10-speech-llm` (two explicit paths, not pushed).  Nothing was launched, no config
was touched, no `Job.__init__` signature and no hashed attribute moved.

Follow-up to `3454206`, answering concerns 1, 3 and 4 of
`reports/review_k2_probe_chunk_2026-09-21.md`.  Concern 2 (the pre-flight fan-out guard's
unreachable threshold) was NOT in the dispatch and was not touched.

## 1. `summary.txt`'s last line is a VERDICT (`lexlat_k2_jobs.py`)

The reads and the whole text of `summary.txt` moved out of `LexlatK2ProbeJob.run()` into a new
module-level writer, `probe_reads_and_summary(probe, ...) -> (per_max_active, verdict, text)`,
so the rule is testable off a record on disk and lives with the code that writes it.  `run()`
calls it once and writes `text`; `probe.json` carries the same word as `"verdict"` (and the new
`"child_failure_reason"`).  The rule, pre-registered in that function's docstring:

* `VERDICT: PASS` -- every PLANNED cell was measured AND every rung's read passes the bar.  A
  planned cell is one of `len(plan) x len(max_active)`; `plan` is the job's own sampled batch
  set (`max_timed_batches`), so a PASS is still the protocol's sampled extrapolation over the
  sub-epoch, which the per-rung `seconds_per_subepoch_basis` line names as before.
* `VERDICT: INCOMPLETE (<measured>/<planned> cells; <reason>)` -- the child timed out (`timeout`),
  died of a signal (`abort`, which is what `K2_ABORT=1` produces on a k2 fatal) or exited non-zero
  (`error`); or cells are missing with no named cause (`error`).  On such a run EVERY per-rung
  line is written `PARTIAL READ: ...` instead of `READ: ...`, so no `READ: PASS` can be grepped
  out of the file whichever way the cells that did run fell, and a paragraph above the verdict
  says nothing in the file may be transcribed into `PROBE_SEC_PER_SUBEPOCH`.
* `VERDICT: FAIL` -- every planned cell is accounted for and some rung's read fails.  "Accounted
  for" is measured OR skipped because THAT RUNG hit `STEP_ABORT_SEC` on a measured cell: the
  child's only early exit is the rung abort (`lexlat_k2.run_probe`), and such a run is the
  deliverable measured FAIL, not a gap.  JUDGMENT CALL, stated because the dispatch's three
  reason words all name a child death: a rung-aborted run reads FAIL, not INCOMPLETE.

`_write_failure` (child died before any cell) ends its file with
`VERDICT: INCOMPLETE (0/<planned> cells; <reason>)`.  `lexlat_k2.py` writes NO per-rung line of
its own -- the child only writes `probe.json` / `partial.json` -- so nothing there needed the
same rule; verified by grep for `READ`/`PASS`/`VERDICT` in that file.

## 2. The GPU-check child has a clock (`lexlat_k2_jobs.py:511` as reviewed)

`GPU_CHECK_TIMEOUT_SEC = 600.0`; the check now runs through `_run_with_timeout` (the same
`start_new_session` + process-group `SIGTERM`/`SIGKILL` as the main child) instead of a bare
`sp.run`.  `_run_with_timeout` gained `capture=False`; with it set it returns the child's stdout
and stderr, which `gpu_check.txt` still records verbatim, and it drains them after the kill.  Its
return is now a 4-tuple `(rc, note, stdout, stderr)`; the one other call site was updated.  On
expiry the job writes `summary.txt` with `VERDICT: INCOMPLETE (0/<planned> cells;
gpu_check_timeout)` and raises.  ASSUMPTION, stated: at that point no batch has been planned (the
plan needs the loader's walk), so `<planned>` is the job's own cap, `max_timed_batches x
len(max_active)` (27 for the live probes), and `?` when no cap is set.

## 3. The child budget cannot exceed the real allocation

`budget_hours = min(float(self.rqmt["time"]), SLURM_TIME_CLAMP_HOURS)` with
`SLURM_TIME_CLAMP_HOURS = 11.5`, the clamp `settings.py`'s `check_engine_limits` applies at
submit time (`current_rqmt["time"] = min(11.5, ...)`).  `settings.py` was NOT touched; the
constant is read here with that reference in its comment.  At the probes' `PROBE_TIME_HOURS = 2.0`
the two agree, so no current run changes: 2.0 h -> 6480 s of child budget, as before.  The build
job's sibling line (`:247`, at 0.95) was outside the dispatch and was left alone.

## Checks run

* `test_lexlat_k2.py` + `test_lexlat_k2_train.py` + `test_lexlat_k2_official.py` +
  `test_lexlat_k2_overcount_exact.py` in `/e/scratch/spell/wu24/envs/sae_k2/bin/python`:
  **92 passed, 1 skipped** (baseline 88 passed, 1 skipped; the skip is the pre-existing `kaldilm`
  one).  The four new tests also pass in the shared training env, since they need no k2.
* NEW in `test_lexlat_k2.py`, all calling the REAL writer on a synthetic record written to and
  read back from a `partial.json` / `probe.json` in `tmp_path`:
  * `test_summary_verdict_is_pass_only_on_a_complete_run` -- 6 of 6 cells, inside the bar:
    `VERDICT: PASS` is the last line, both rungs read `READ: PASS`, no `PARTIAL` anywhere;
  * `test_summary_verdict_is_incomplete_when_the_child_died` -- ONE measured cell of six, and it
    would pass the bar (`per_ma["1000"]["read"] == "PASS"`): last line
    `VERDICT: INCOMPLETE (1/6 cells; timeout)`, the rung lines are `PARTIAL READ: PASS` and
    `PARTIAL READ: FAIL`, and `re.search(r"(?m)^\s*READ:", text) is None`.  `abort` and `error`
    are checked on the same record.  This is the 2026-09-21 failure mode;
  * `test_summary_verdict_is_fail_when_a_measured_rung_misses_the_bar` -- all cells measured, one
    rung at 1800 s against the 1202 s bar: `VERDICT: FAIL`;
  * `test_summary_verdict_is_fail_when_a_rung_aborted_on_its_own_clock` -- a rung aborted at
    batch 0 leaves 4 of 6 cells measured and still reads `VERDICT: FAIL`; the same gap WITHOUT
    the abort record reads `INCOMPLETE (4/6 cells; error)`.
* `_write_failure` exercised directly (throwaway, not a committed test): its file ends
  `VERDICT: INCOMPLETE (0/27 cells; timeout)`.
* `_run_with_timeout` exercised directly in the shared env: `exit 3` with `capture=True` returns
  `(3, '', 'hi\n', 'boom\n')`; `sleep 60` at a 2 s budget returns `(None, 'killed after 2 s', '',
  '')` in 2.0 s; the non-capturing form still returns `(0, '', '', '')`.
* HASH CENSUS, before against after, with `scripts/sae_4a_lexlat_k2_census.py`.  The "before" run
  used a COPY of `src/speech_llm` with only `lexlat_k2_jobs.py` reverted to `HEAD` and asserted
  in-process that the module it imported came from that copy (the first attempt did not: sisyphus'
  settings load rewrites `sys.path`, which would have made the diff vacuous).  `live`
  (config/sae_4a_lexlat_k2.py, 12 jobs) and `official` (config/sae_4a_lexlat_k2_official.py,
  17 jobs) both diff to NOTHING, including `LexlatHLGBuildJob.rtX44PBJFNy1`,
  `LexlatK2ProbeJob.Tkl94t85j4pV`, `.5dP7W2NMFq4T`, `.R7QzD6vYBLD3`, `.DtWYToXTPh6w`.  The pack
  config still cannot be censused: `config_sae_4a_lexlat_k2_pack_v1.py:286` raises
  `PROBE_SEC_PER_SUBEPOCH is not stated`, so no training arm using `lexlat_k2_train` is in any
  graph and nothing there can move -- the census the reviewer used is the equivalent registration.
* `ruff check` on both files: clean; no new line above 100 characters (the 7 + 14 long lines are
  the pre-existing ones).

## Residuals and what this does NOT show

* On an INCOMPLETE run the per-rung TIME and MEMORY words are still printed as `PASS` / `FAIL`
  (`... 3.0 s/sub-epoch [...] PASS`), because the dispatch's rule names the per-rung `READ:` lines
  only.  A grep for the bare word `PASS` therefore still matches on such a file; a grep for
  `READ: PASS` or for `VERDICT: PASS` does not.
* The GPU-check timeout path and `_write_failure` have no committed test (they need a Job
  instance); their VERDICT strings are literals and were exercised by hand as above.
* Nothing was run on a GPU.  What the k2 leg costs against E1's bar is still unknown, and this
  change does not make any earlier `summary.txt` readable -- files written before this commit
  carry no VERDICT line at all.
* This report was written but NOT committed: the dispatch named only the `2025-10-speech-llm`
  repository and branch for committing.
