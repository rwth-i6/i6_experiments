# LexlatStepProfileJob.ItFHebFGK6nb — rerun (2nd) failure report, 2026-09-21

Job dir: work/speech_llm/sae/emc/lexlat_profile_jobs/LexlatStepProfileJob.ItFHebFGK6nb
Config: config/sae_4a_lexlat_profile.py
Manager pid 628195 exited (not restarted per instruction).

## Failure identity: NEW cause, NOT the same as first failure
First failure (output_failed1/log.run.1, 2026-09-21 11:29): `AssertionError: the timed step's loss
'blankfree_frames_per_sec' moved by 9.488e+01 relative` — a profiling-invariance assertion.

Second failure (this rerun, log.run.1:489, 17:06:22): the job died 3 seconds after start, before
reaching any GPU profiling code, at:
`recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lexlat_profile_jobs.py:489` ->
`self.returnn_config.write(self.out_config.get_path())` ->
`recipe/i6_core/returnn/config.py:199 ReturnnConfig._write_to_file` ->
`subprocess.check_call([self._black_path, file_path])` with `self._black_path = None` ->
`TypeError: expected str, bytes or os.PathLike object, not NoneType` (posixpath.dirname on
executable=None inside Popen._execute_child).

Cause: the `black` formatter binary was not found in the job's own conda execution environment
(`/e/project1/spell/wu24/env/conda/envs/speech_llm/...`), so `ReturnnConfig._black_path` resolved to
None and the write-time auto-format subprocess call crashed. This is a tooling/environment defect,
not the recipe logic that produced the first failure.

## Numbers
- sacct 1926511: State COMPLETED, Elapsed 00:01:16, ExitCode 0:0 (SLURM step itself exited cleanly;
  the Python task caught and logged the error internally).
- usage.run.1 (2nd run): used_time 0.0011 h (~4s), RSS 0.47 GB, host jpbo-085-15.
- output/ this run: only `returnn.config` (partial, pre-black-format) written; no summary.txt,
  no trace.json.gz, no per_frame.json this attempt (job died before reaching profiling).
- output_failed1/ (1st run) has full artifacts: summary.txt, profile.json, trace.json.gz (573 MB),
  per_frame.json — that run reached and completed a profiled batch before the assertion fired.

## summary.txt (from output_failed1, verbatim, only prior data available)
lexlat_20/ctrl_50_ep10  C=1024 lam_lex=1.000 epoch=10  --  per-COMPONENT profile of one lexicalised step
PARTIAL: written after a completed batch; the job had not returned.
sub-epoch 10: 57 batches, longest T at index 48; E1's plan [0, 1, 14, 15, 28, 29, 43, 44, 48]; profiled [0, 1]
peak GPU over the run: reserved 0.00 GiB (allocated 0.00 GiB)
gates: none

Not fixed (no rqmt/marker-rename/resubmit performed) per instruction: not relaunched, not cleared.
