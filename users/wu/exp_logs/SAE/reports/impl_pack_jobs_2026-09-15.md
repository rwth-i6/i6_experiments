# SAE 4a -- PackedEmcTrainJob: four EMC arms per GH200 node (implementer report, 2026-09-15)

Dispatch: build a sisyphus job that runs FOUR EMC training arms on one GH200 node, one process per
GPU, so the exclusive-node flag (`settings.py:128-131`, unchanged) stops idling three of four GPUs.
Measured input: `analysis/out/emc_profile_step.b128.txt` -- one arm at `max_seqs 128` is SM 98 %,
1.67 s/step, 10.4 GiB peak of 96, i.e. one arm saturates one GPU -> four independent arms, not DDP.

Files written (both NEW, nothing else touched):

* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/pack_jobs.py`
* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_pack_jobs.py`

`emc_train_jobs.py` (live, another implementer), `lattice.py`, `s3_jobs.py`, `selection.py` and
`config_sae_4a_s3b_rate_v1.py` were read only.

## 1. What the job is

`PackedEmcTrainJob(arms={name: EmcArmSpec}, time_rqmt=..., mem_rqmt=..., cpu_rqmt=..., gpu_mem=...)`

* **rqmt**: `gpu = GPUS_PER_NODE = 4`, `cpu/mem/time = 4 x` one arm's. One arm's four numbers are
  READ OFF `emc_train_jobs.emc_training`'s signature by `inspect` (`_single_arm_rqmt_defaults`), not
  copied, so they cannot drift from the single-arm job; a stage that overrides one (the rate stage
  passes `time_rqmt=6.0`) passes it per arm and the job multiplies.
* **create_files**: for each arm, `ReturnnTrainingJob.create_returnn_config(...)` +
  `ReturnnConfig.write()` -- i6_core's own writing path -- into `output/<arm>/returnn.config`, plus
  `work/<arm>/rnn.sh` via `i6_core.util.create_executable`. `post_config["model"]` is set to that
  arm's own `output/<arm>/models/epoch`, exactly as `ReturnnTrainingJob.__init__` does after hashing.
* **run**: one `subprocess.Popen` per arm, `cwd=work/<arm>`, `CUDA_VISIBLE_DEVICES=<gpu index>`,
  `OMP/MKL_NUM_THREADS` = one arm's cpu, stdout+stderr to `output/<arm>/log.run.1`. Waits for all,
  writes each successful arm's marker and links its `learning_rates` into the output, then raises
  `ArmFailure` naming every failing arm with its return code and log path. A failing arm does NOT
  kill its siblings (their sub-epochs are worth the same, and a resume cannot get them back).
* **GPU assignment is by SORTED arm name**: a dict's sisyphus hash is order-independent
  (`sisyphus/hash.py:152`), so an order the hash cannot see must not decide which GPU an arm gets.
* The launcher is a free function `run_arms(Sequence[ArmRun])` with no sisyphus/RETURNN/GPU
  dependency -- that is what the CPU smoke test drives.

## 2. What the reads consume unchanged -- and what they do not

`job.arm(tag)` returns a view with the single-arm attribute names. Per arm, under `output/<arm>/`:

| pack output | single-arm equivalent | consumer |
|---|---|---|
| `out_checkpoints[arm][ep]` = `PtCheckpoint(models/epoch.%.3d.pt)` | `out_checkpoints[ep]` | `subepoch_reads(emc_checkpoint=)`, `theta_checkpoint`, `ExtractSubmoduleCheckpointJob(checkpoint=...path)` |
| `out_learning_rates[arm]` | `out_learning_rates` | `selection.UnsupervisedCheckpointSelectionJob(learning_rates=)` / `read_cv_scores` (`CV_LOSS_KEY`), `tk.register_output` |
| `out_model_dirs[arm]` | `out_model_dir` | `tk.register_output` |
| `out_returnn_config_files[arm]` | `out_returnn_config_file` | `tk.register_output` |
| `out_logs[arm]` = `log.run.1` | the job's `log.run.1` | `emc_train_jobs.read_efficiency_gate("<pack>/output/<arm>/log.run.1")` |

**NO adapter is needed by any read the rate config registers.** The epoch keys are
`ReturnnTrainingJob`'s own arithmetic (`save_interval` / `num_epochs` / `keep_epochs`), asserted
equal to the real job's in the test. `path_available` is mirrored from `ReturnnTrainingJob`, so a
sub-epoch's reads start while the pack still runs instead of waiting for all four arms.

Differences a config should know (none of them block a read):

1. `out_plot_se` / `out_plot_lr` are NOT exposed (no consumer in S3/S3b; a config that wants the
   curves plots them from `out_learning_rates[arm]`).
2. `log.run.1` is APPENDED to on a resume (a single-arm job would open `log.run.2`). RETURNN's own
   `./returnn.log` sits in `work/<arm>/returnn.log`, same lines.
3. The alias is per JOB, not per arm: a config calls `pack.add_alias(...)` once.

## 3. Hash

`PackedEmcTrainJob.hash` reads **only** `arms`; every rqmt is outside it, as in `ReturnnTrainingJob`.
`EmcArmSpec._sis_hash` is `ReturnnTrainingJob.hash`'s dict verbatim -- the FINALIZED `ReturnnConfig`,
`returnn_python_exe`, `returnn_root` -- so an arm's hash moves exactly when the single-arm job's
would, and `num_epochs` (post-config) does not move it.

Census, `scripts/sae_4a_cons_census.py`, BEFORE = the two new files moved out of the tree, AFTER =
as committed; diffs empty for all three graphs:

* `config_sae_4a_s3_v1`: **98 jobs before, 98 after, ids identical**
* `config/sae_4a_phase.py`: **1021 before, 1021 after, ids identical**
* `config_sae_4a_s3b_rate_v1`: **219 before, 219 after**, arms at
  `ReturnnTrainingJob.{WCh1fMyr88yu (lam1), DF6blPpto23t (lam3), axf11lrsap5i (lam10)}`

(The module is imported by no config, so this is the expected result; it is the check, not a claim.)

## 4. Tests (all pass)

Run from the setup dir; `black` must be on PATH (the sis venv's) or `ReturnnConfig.write` cannot
format:

    WS=$PWD; PATH=/e/project1/spell/wu24/env/sis_env/bin:$PATH \
      PYTHONPATH=$WS/tools/sisyphus:$WS/recipe:$WS/recipe/i6_models:$WS/recipe/returnn:$WS/recipe/2025-10-speech-llm/src \
      /e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python -m speech_llm.sae.emc.test_pack_jobs

* `test_stored_epochs` -- the kept epoch keys, including a `save_interval` that skips a kept epoch.
* `test_arm_run_smoke` -- two fake arms (a stub command, CPU only): per-arm log content, per-arm
  `CUDA_VISIBLE_DEVICES`, the `learning_rates` hard link into the output (`os.path.samefile`), the
  marker, one launch each.
* `test_failure_names_the_arm_and_skip_on_finished` -- a `rc 3` arm raises `ArmFailure` naming it
  with its rc and log path; the healthy sibling still finishes and keeps its outputs; the failing
  arm has no marker; on the rerun the finished arm is SKIPPED (its process is not started again,
  counted) and the failed one runs a second time.
* `test_config_equivalence` -- builds `config_sae_4a_s3b_rate_v1` with `emc_training` SPIED on, so
  the same `ReturnnConfig` object reaches the real single-arm `ReturnnTrainingJob` and the pack arm
  (no hand-built copy of the stage's arguments). The two WRITTEN config files for the lam3 arm are
  **byte-identical** once each job's own output prefix is normalised (`model = "<ARM_OUT>/models/
  epoch"`); the single-arm job is the reviewed `ReturnnTrainingJob.DF6blPpto23t`. Also: same epoch
  key set and checkpoint file names and `PtCheckpoint` type as the single-arm job, sorted-name GPU
  assignment, `rqmt` = `{gpu 4, cpu 4x16, mem 4x64, time 4x6.0, gpu_mem 96}`, hash unmoved by rqmt
  and moved by the arm set / an arm's name.

The config test adapts to `config_sae_4a_s3b_rate_v1.ACTIVE_ARMS` (currently `("lam3",)`, another
implementer's commit `8ff23fe`): it builds specs from the arms the config actually builds, requires
lam3 among them, and fills the pack to four with stand-in arms.

**No GPU work was run, nothing was launched, and no config wires the pack.** The module docstring
documents how a config packs arms, with the lam {1, 3, 10} rate arms + one filler as the example.

## 5. Open points for the orchestrator

1. **`time = 4 x` one arm's is the dispatch's rule, and the arms run CONCURRENTLY**, so the wall
   clock of the allocation is ONE arm's, not four. `settings.check_engine_limits` then clamps the
   request to 11.5 h (4 x 6 h = 24 h -> 11.5 h), so the pack will hit the cap and auto-resume at
   least once on an 8-sub-epoch stage -- which is what the per-arm markers are for, but it means the
   submitted time is NOT what the rule computes. If the intent was "one arm's time" the constructor
   argument is the only thing to change; I did not choose either, I implemented the dispatch's rule.
2. **A four-arm pack is one failure domain for the SCHEDULER**: any arm's crash fails the job (after
   the others finish), and the manager retries the whole job -- cheap, because finished arms are
   skipped, but the job's state in the console is per pack, not per arm.
3. **Filler arms are a config decision.** A pack of fewer than four arms still takes the node
   exclusively; the job allows 1..4 arms and says so. Which fourth arm the consistency stage packs
   is not chosen here.
4. Not implemented (not asked, no consumer): `info()` / `completed_fraction()` per arm for the
   console, and plotting.

Commit: `5994bd9` on `haotian_modality_matching_jupiter` in `recipe/2025-10-speech-llm` (two files,
staged explicitly, not pushed). This report is not committed (the dispatch names one repository).

## Round 2 (2026-09-15) -- the time rqmt is the longest arm's, not the sum

Coordinator correction: the arms run concurrently, so `rqmt["time"]` is now
`max over the arms of that arm's single-arm time x SHARED_NODE_TIME_FACTOR`, with
`SHARED_NODE_TIME_FACTOR = 1.1` documented in the module as the margin for four arms sharing one
node's host memory bandwidth, I/O paths and page cache (a wall-clock margin only: it enters no hash
and no measured quantity). `cpu` and `mem` stay `4 x` one arm's. `time_rqmt` now accepts one number
for every arm or a `{arm: hours}` mapping. This retires open point 1 of section 5: the rate stage's
6 h arms are submitted as 6.6 h, under the 11.5 h clamp, so the pack no longer hits the cap by
construction.

Tests re-run, all pass: `rqmt["time"] == 6.0 x 1.1` for the lam3 pack, and a second pack with
`{t_short: 2.0, t_long: 5.0}` gives `5.0 x 1.1` while cpu/mem/gpu stay `4 x 16 / 4 x 64 / 4`. That
check uses a FRESH arm set, because sisyphus hands back the same instance for an already-built hash
(a rqmt variant of an existing pack is that pack) -- which is also why the rqmt-not-in-hash
assertion is meaningful. Config equivalence, epoch keys, the CPU smoke and the resume/failure tests
are unchanged and still pass; the pack's id is unmoved at `PackedEmcTrainJob.yrssZh2YhBo4`.

Commit `3f5c805` on `haotian_modality_matching_jupiter` (pack_jobs.py plus the one assertion block
in test_pack_jobs.py, which the change invalidates; explicit paths, not pushed).
