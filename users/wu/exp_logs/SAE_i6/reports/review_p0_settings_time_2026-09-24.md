# Code review: settings.py time floor for ReturnnTrainingJob and matplotlib-base (SAE_i6 P0), 2026-09-24

Verdict: PASS_WITH_NOTES. Both deltas do what the dispatch intends. Neither changes a hash, a
partition, or the rqmt of any other task. There is one behavioural note: after a TIMEOUT the time
does not double. Every resubmit goes out at 72 h again. The implementer's report says the opposite.
The screen may launch.

## Delta 1: `settings.py:41-45` (`check_engine_limits`)

What I did:
- Diffed `settings.py` against `settings.py.orig_2026-09-24`.
- Diffed it against the compiled settings from 15:26 (`__pycache__/settings.cpython-311.pyc`, copied
  to the scratchpad). Only three names differ from that version:
  - `FFMPEG_PIN_ACCEPT` and the absolute `IMPORT_PATHS`, both already covered by the 16:50 launch review;
  - `check_engine_limits`.
  Everything else is byte-identical: `engine`, `worker_wrapper`, `file_caching` and `DEFAULT_ENVIRONMENT_SET`.
- Built the pre-change function by deleting lines 41-45 from the current body.
  - Checked it against the function in the 15:26 pyc. They agree on all 195 task rows.
- Built both graphs under the sae python with `sae/bin` first on PATH (`sis console --script`,
  `PYTHONDONTWRITEBYTECODE=1`). For every task and task id, computed
  `gs.engine().get_rqmt(task, id, update=False)` under the old function and under the new one.

Results:
- Screen (`config/sae_i6_p0_screen.py`): 75 jobs, 110 task rows. Exactly 1 row differs:
  `ReturnnTrainingJob.GiT88bxzoZbZ` run, time 11.5 -> 72.
  - Nothing else changed in that row: cpu 16, mem 64, gpu 1, gpu_mem 96, `-p gpu_48gb`.
  - `create_files` and `plot` are unchanged (engine short, 1 h).
- Full (`config/sae_i6_p0.py`): 139 jobs, 195 rows. Exactly 5 rows differ, all ReturnnTrainingJob run:
  - `GiT88bxzoZbZ`, `DvVfxf1LrCBi` and `jcKXbLMDk4hl`: 11.5 -> 72.
  - `Ac2eioZbRX7d`: 4 -> 72.
  - `Y1vbqR6KeJSx`: 3 -> 72.
  - In the full graph all 5 have module `i6_core.returnn.training`, and no subclass of it exists.
- Every `sbatch_args` value is identical to before, so no partition changed.
- ctrl_20 keeps the id `GiT88bxzoZbZ`. The time rqmt is not part of the hash.
- `PipelineJob.p4BOP5qZ6T1G`: its `tasks()` raises "Path not ready" under both functions. The lexicon
  download does not exist yet. For such a job the new code only runs `getattr` and `name()`, which
  have no side effects, and then the unchanged body.
- Identification. Every task in this sisyphus sets `_job` in `set_job` (`task.py:77-81`), and
  `name()` returns `_start` (`task.py:115`). The class and module are the same in the worker because
  the job is unpickled from `i6_core.returnn.training`.
- Isolated calls, none of which raise:
  - a bare Task with no `_job`, and `_job=None`;
  - a subclass (not floored), ReturnnForwardJobV2 run (not floored), and `plot` (not floored);
  - no time (floored to 72), time 200 (capped at 168), time 100 (kept at 100);
  - a pre-set `-p` (floored to 72, partition kept).
- Slurm limits: `gpu_48gb` MaxTime is 7-00:00:00, and the QoS has no MaxWall. A 72 h request is accepted.

Note N1, TIMEOUT: the time does not double.
- Where: `settings.py:45`, together with `sisyphus/sisyphus/engine.py:98-110`.
- Mechanism:
  - The submit log stores the rqmt after the limits are applied, so time = 72 (`engine.py:181,209-216`).
  - On a resubmit, `get_rqmt` keeps a logged value only when the first logged value equals the
    recipe value. 72 is not 11.5, so it starts again from the recipe's 11.5 h.
  - `update_engine_rqmt` doubles that to 23, and the floor raises it back to 72.
- Simulated with the real graph tasks, patching only the submit history and `{"out_of_time": True}`:
  - old: 11.5 -> 23 -> 46 -> 92 -> 168;
  - new: 72 -> 72 -> 72 -> 72 -> 72.
  - The same holds for all 5 trainings. p0 and gold phi before the change: 3 -> 6 -> ... and 4 -> 8 -> ....
- `impl_settings_time_2026-09-24.md`, section Notes, says "values above 72 h ... after update_rqmt ... are kept". That is false.
- No concrete failure follows:
  - Each resubmit gets 72 h and resumes.
  - `ReturnnTrainingJob.completed_fraction` (`i6_core/returnn/training.py:335`) makes resubmits that
    made progress not count toward MAX_SUBMIT_RETRIES (`task.py:399-411`).
  - The first 4 submits add up to 288 h here, against 172.5 h for 11.5/23/46/92 before the change.
- If escalation past 72 h is ever wanted, one option is to floor only when no submit history exists.
  Another is to raise the recipe time instead. Not needed for P0.

## Delta 2: matplotlib-base 3.11.2

- `environment.yml:56-57`: a comment line and `  - matplotlib-base=3.11.2` in the conda list, before `- pip:`.
  - `yaml.safe_load` parses the file: 25 conda specs, and the pip list is unchanged apart from the
    earlier librosa line.
- Env: `conda-meta/history` has one new transaction (16:51:29, `mamba install ... matplotlib-base`)
  with 19 `+` lines and no `-` lines.
  - In the sae python: matplotlib 3.11.2, numpy 2.4.6 (unchanged), python-dateutil 2.9.0.post0, pillow 12.3.0.
- Plot run: `i6_core.returnn.training.ReturnnTrainingJob.plot` was called unbound on a fake object
  pointing into the scratchpad.
  - It ran in a cleaned environment that copies DEFAULT_ENVIRONMENT_SET (sae/bin first, HOME kept),
    with MPLCONFIGDIR in the scratchpad.
  - Case a: the RETURNN 00171dfe format, i.e. `EpochData(learningRate=, error=)` with `:meta:` keys,
    `np.float64(...)`, `nan`, and an empty error dict at the last epoch. Result: 2 PNGs.
  - Case b: `train_score`, `dev_score` and `dev_error` keys plus `inf`, which exercises the twinx
    branch. Result: 2 PNGs.
  - Backend Agg; the module loads from the sae env.
- Nothing else reads matplotlib: no hit in the package, and none under `returnn/` at 00171dfe. So the
  install changes no other code path.

## Other
- Carried over, not new: `training/arms.py`, `inputs.py`, `data/vad.py` and `config/supervised_init.py`
  are still uncommitted. See launch review note 5: commit before the launch.
- Side effects:
  - Scratchpad files only.
  - `git status` refreshed the stat cache in `recipe/i6_core/.git/index`.
  - No bytecode was written, and nothing was written under `work/`.
