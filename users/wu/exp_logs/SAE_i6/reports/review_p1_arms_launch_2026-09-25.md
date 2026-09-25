# Review: P1 Task B STAGE=arms launch (rt_r70, rt_r80 beside the running rt_r90), 2026-09-25 17:18

Reviewer: code-reviewer. Read-only: no manager was stopped or started, nothing was submitted or committed.
Inputs: `config/sae_i6_p1_ladder.py`, `settings.py`, `reports/impl_p1_arms_chunked_backward_2026-09-25.md`,
`reports/review_p1_arms_chunked_backward_2026-09-25.md`, `reports/launch_p1_rt_r90_probe_2026-09-25.md`,
`SAE_i6_P1.md` (Task B, G1.M, G1.R70, G1.L), the probe job dir, squeue/sinfo, /proc of the live managers.
S = `/u/hwu/setups/librispeech-960/2026-09-24-unsupervised`.

## Verdict: PASS_WITH_FIXES

The STAGE=arms graph is sound. rt_r90 resolves to the running `EexT85vdfx25`. rt_r70 and rt_r80 differ from it
only in the corrupted-phi checkpoint (plus their own model dir and r90's unhashed memory-log flag). All three
route to gpu_48gb with 72 h, 16 CPU, 64 GB. Nothing finished is rerun and nothing is shared with P0 unfinished.
The fix is operational, not in the config: the probe-stage watcher must be stopped and the new watcher must
carry `P1_LADDER_STAGE=arms`. Otherwise failures of rt_r70/rt_r80 go unseen (F1).

## Findings

F1 (MUST) `config/sae_i6_p1_ladder.py:155` (stage defaults to `probe`) with `~/.claude/skills/sis/sis_watch.sh:23`
(job list `log/sae_i6_p1_ladder.watch_jobs`, keyed on the config name only).
- The live watcher 1702163 (wrapper bash 1702161) runs with `P1_LADDER_STAGE=probe` on manager 1700738.
- When 1700738 is killed, it leaves its poll loop (`:158`) and rewrites the shared job list with the 87-job
  probe graph (`:256`).
- A new watcher started on the same config then reads a list without rt_r70/rt_r80. An `error.run.*` in
  `9lD2HS2Mzgcl` or `5CL3pQyyOvQt` is not seen until the next refresh, which happens only when some job finishes.
- The old watcher also keeps polling the probe graph. Once rt_r90 ends it can report STALLED or DONE for a graph
  that is no longer the one being managed.
- A new watcher (or any later restart or console) started without `P1_LADDER_STAGE=arms` silently builds the
  probe graph. A restarted manager without it would not schedule the rt_r70/rt_r80 reads.
- Fix: stop 1702163/1702161 before starting the new watcher. Start the new watcher with `P1_LADDER_STAGE=arms`,
  and record that env in State's watcher line (`SAE_i6_P1.md:18` still says `probe`).

F2 (SHOULD, brief only, not the config) - The dispatch says "about 20 sub-epochs, roughly 23 h" per arm. The graph
builds `num_epochs = 8`, with `learning_rates = [1e-05] + [1e-4]*7` and keep 1/2/4/8 (`ladder.NUM_SUBEPOCHS = 8`,
`ladder.py:385`). This matches the design (`SAE_i6_P1.md:68`, "N = 20 learning rates truncated to 8 sub-epochs",
and G1.L's 8-sub-epoch budget).
- The probe measured 1:10:25 of training plus about 3 min of dev per sub-epoch. That puts each arm at about
  10 h: rt_r90 ends around 01:45 and rt_r70/rt_r80 around 03:30 if they start at about 17:30.
- The 72 h limit covers this with a wide margin. Plan the next check on this time, not on 23 h.

No other finding.

## Check 1: job ids (dry graph build, `sis console -s`, sae python, S as cwd, no submit)

| stage | jobs | rt_r70 | rt_r80 | rt_r90 |
|---|---|---|---|---|
| probe | 87 | - | - | `ReturnnTrainingJob.EexT85vdfx25` |
| arms | 113 | `ReturnnTrainingJob.9lD2HS2Mzgcl` | `ReturnnTrainingJob.5CL3pQyyOvQt` | `ReturnnTrainingJob.EexT85vdfx25` |

- Every probe-stage job is in the arms graph. The arms stage adds exactly 26 jobs: 2 trainings (r70, r80), and
  for each of those two arms 4 `ExtractSubmoduleCheckpointJob`, 4 `ReturnnForwardJobV2` and 4 `BlankfreeGreedyPerJob`.
- The ids equal the implementer's and the first review's.
- rt_r90 is running as Slurm 4362041_1 on cn-506 under the name `...ReturnnTrainingJob.EexT85vdfx25.run`. The
  sisyphus Slurm engine matches tasks by that name (`simple_linux_utility_for_resource_management_engine.py:410-438`),
  so a new manager reads it as RUNNING and does not resubmit it.
- The job dir has no `error.*` marker, so the backgrounded manager meets no "Clear jobs" prompt.

## Check 2: the single delta per arm

Configs dumped from the in-memory arms graph (`ReturnnConfig.write`):
- arms r90 vs probe-stage r90: identical.
- arms r90 vs the probe's written `work/.../EexT85vdfx25/output/returnn.config`: identical.
- r70, r80 and r90 are byte-identical to the implementer's `analysis_out/p1_ladder_rt_{r70,r80,r90}_2026-09-25.returnn.config`.
- `diff` r70 vs r90: `reverse_checkpoint_path` (ruLnJFWyifwp vs uO8wkodbR2uh `/output/models/epoch.008.pt`), the
  job's own `model =` dir, and `torch_log_memory_usage = True` on r90 only.
- `diff` r80 vs r90: the same three lines (6IkAAuzcBwBR).
- `diff` r70 vs r80: the checkpoint and the model dir only.
- `torch_log_memory_usage` is logging only: RETURNN `engine.py:135,329-339,1614-1618` reads
  `max_memory_allocated` and prints it. It is in post_config, so it is not hashed. The per-step
  `lexlat_k2_pre_peak_allocated_gib` and `lexlat_k2_device_used_gib` monitors are on in all arms
  (`rt_chunked_backward.py:204-214`).
- Checkpoint to corruption chain, traced through the job input links:

| arm | fit | data | targets | corruption | realised rate |
|---|---|---|---|---|---|
| r70 | `ruLnJFWyifwp` | `SupervisedReverseDataJob.LO7eQJLYRc9M` | `PhoneTargetHdfJob.UgE8lffBU9f1` | `CorruptSeedGoldJob.aj2XQaDlED2E` | 0.699905 |
| r80 | `6IkAAuzcBwBR` | `pOT6A6WAq95V` | `OtACGK7q2d1R` | `H9GbhLPtIthV` | 0.800081 |
| r90 | `uO8wkodbR2uh` | `3qfqdoscYbJ7` | `Otjvrdt4P6Ya` | `wsKq52Dk68b4` | 0.899946 |

  The realised rates come from G1.F. All three fits are `finished` and have `epoch.008.pt`.
- Shared by all three arms: the flat init `FlatRecognizerInitJob.0J9d6wjrkRYH`; tau 2.0 x8; k2 rung 1000, on-set 1,
  ramp 3, lam 1; `lexlat_k2_chunk_seqs 4`; batch 88000 / max_seqs 128; no `random_seed`; the same RETURNN clone
  `CloneGitRepositoryJob.KQ3NuCaDE6QH` and sae python; the epilog `rt_chunked_backward import train_step` at the end.

## Check 3: resources

`settings.check_engine_limits`, applied to each arm's job-bound run task in the console:
`{'gpu': 1, 'cpu': 16, 'mem': 64.0, 'time': 72, 'gpu_mem': 96, 'sbatch_args': ['-p', 'gpu_48gb']}` for all three.
- The time limit is raised at `settings.py:394-395` before the `-p` early return (`:396-397`), so the explicit
  `-p gpu_48gb` from `config/sae_i6_p1_ladder.py:69` overrides `GPU_ROUTE_TRAIN = "gpu_32gb"` (`:399-400`).
- The resource request equals the probe's `submit_log.run`. gpu_mem 96 is not passed to Slurm on this path.
- L40S at 17:15: 7 GPUs free on cn-506..509 and none pending. The user holds 2 (P0 4346718_1 and rt_r90), so
  the arms make 4 of the QoS cap of 5.
- Per-chunk epilog with chunk_seqs 4: present in all three configs.

## Check 4: what else the arms stage schedules

- Only the 24 read jobs of rt_r70/rt_r80 besides the two trainings. rt_r90's ep1 read was already run by the
  probe manager: `ExtractSubmoduleCheckpointJob.ObmMhoMdxqqH`, `ReturnnForwardJobV2.pYn9JZHd401T` and
  `BlankfreeGreedyPerJob.zmpClyX2eupd` are finished. Its ep2/4/8 reads are unchanged from the probe graph.
- The reads start only when `epoch.00N.pt` exists (i6_core `path_available`). RETURNN writes the checkpoint
  atomically (`.tmp_write` then rename). Keep 1/2/4/8 protects those files from cleanup.
- Each read is wired to its own arm's checkpoint (dependency dump). They are the pre-registered reads
  (`SAE_i6_P1.md:72`), so none is premature.
- Routing of the reads: forwards (gpu 1, gpu_mem 24, 2 h) go flex to gpu_24gb/gpu_48gb and never to gpu_32gb or
  gpu_11gb; extract and PER are CPU jobs.
- The 12 fit-chain jobs are all finished; there is no fit rerun and no fits manager is live.
- P0 graph (164 ids, dry build now): 62 jobs are shared with the arms graph, all finished; no unfinished job is
  shared. P0 builds neither arm id, so no held-path construction can claim these hashes.
- `P1_LADDER_R100` and `P1_LADDER_SEED2` are unset in the probe manager's environment. Keep them unset.

## Check 5: imports

The arms stage runs the same `rt_arm` path as the probe, for three tags. The modules imported are the same:
`sae_i6_p1_fits`, `ladder`, `rt_chunked_backward`, `training.jobs`, `analysis.per`, `config.common` and `inputs`.
The RETURNN configs import `model.blankfree_model`, `model.train_step`, `model.param_groups` and
`reverse_model.rt_chunked_backward` from `S/recipe`.
- Since the probe launch (15:56), the only package change is the new test
  `tests/test_rt_chunked_backward_k2lat.py` (14a8042d7). The working tree is clean for the package.
- `config/sae_i6_p1_ladder.py` (15:14) and `settings.py` (12:21) are older than the probe launch.
- recipe/i6_core, recipe/returnn and sisyphus are unmodified since 15:50.
- rt_r70/rt_r80 therefore load the same code as rt_r90.

## Launch (from S)

1. Stop the probe-stage watcher (the orchestrator's background task): `kill 1702163 1702161`.
2. `kill 1700738`. Then `ps -eo pid,args | grep -E "m (-\S+ )*-r config/sae_i6_p1_ladder" | grep -vE "grep|bash -c"`
   must print nothing, and `squeue -h -j 4362041 -o "%i %T %N"` must still show `4362041_1 RUNNING cn-506`.
   Do not touch 1646677.
3. `(P1_LADDER_STAGE=arms PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH setsid nohup /work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis --log_level 30 m -r config/sae_i6_p1_ladder.py > log/sae_i6_p1_ladder.manager.log 2>&1 < /dev/null &)`
   - The pid from `ps` must be exactly one; write it to `log/sae_i6_p1_ladder.manager.pid`.
   - `tr '\0' '\n' < /proc/<pid>/environ | grep -E '^P1_LADDER|^PATH='` must show only `P1_LADDER_STAGE=arms`
     and a PATH that starts with `/work/asr4/hwu/conda/envs/sae/bin:`.
4. Watcher: `P1_LADDER_STAGE=arms SIS_LAUNCHER="/work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis" PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH bash ~/.claude/skills/sis/sis_watch.sh <pid> config/sae_i6_p1_ladder.py 60`.
5. Post-submit checks:
   - squeue shows `...EexT85vdfx25.run` once (4362041_1), and `...9lD2HS2Mzgcl.run` and `...5CL3pQyyOvQt.run`
     once each on gpu_48gb.
   - `diff work/i6_core/returnn/training/ReturnnTrainingJob.9lD2HS2Mzgcl/output/returnn.config analysis_out/p1_ladder_rt_r70_2026-09-25.returnn.config`
     is empty; so is the same diff for `5CL3pQyyOvQt` against `..._rt_r80_...`.
   - Each `submit_log.run` shows `['-p', 'gpu_48gb']` and time 72.
   - Each `log.run.1` contains `rt_chunked_backward: per-chunk k2 backward installed (chunk_seqs = 4)`.

Evidence was written to the session scratchpad (not durable). It is reproducible from S with the commands above
and `sis console config/sae_i6_p1_ladder.py -s -c ...` under `P1_LADDER_STAGE=arms` and `=probe`.
