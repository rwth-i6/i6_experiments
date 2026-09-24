# Review: full P0 launch (`config/sae_i6_p0.py`) after the cost screen, 2026-09-24 22:50

Verdict: PASS_WITH_NOTES. The full launch is safe to start as it is, right after the screen read, using
the sequence in section 3. No finding produces a wrong number, a wasted run or a double submission.
The notes cover three things: read forwards can queue behind this user's own gpu_48gb cap, the screen
manager's `-co` flag must not be carried over, and the pins to re-check just before the start.

Pins reviewed. Re-verify these just before the start; if any differs, this review does not cover it.
- settings.py: sha256 43585da6c5949df6 (the GPU-route round-3 version).
- config/sae_i6_p0.py: 5126c75719a2d73a. config/sae_i6_p0_screen.py: 690b1c3ebdb81f6e.
- Package: recipe/i6_experiments HEAD 551befc69 at review time, e23d8e886 at 22:51 (that commit touches only
  exp_logs/SAE_i6), with the package worktree equal to 51f4def2d.
- Graph ids: full 164, sha1 61a1419117f4; screen 75, sha1 db40328df82a. Both equal the values in the
  round-3 routing review.

Evidence: scratchpad `.../e99790e1-.../scratchpad/fl/`:
- full.txt and screen.txt: ids, per-task rqmt and Slurm options, registered outputs;
- cfg/ and cfg2/: every RETURNN config written through black;
- deps.py: the unfinished upstream jobs of each training.
All of it was built under the sae python with sae/bin first on PATH and PYTHONDONTWRITEBYTECODE=1.

## 1. Graph

- **Jobs.** The full graph has 164 jobs and the screen graph 75. All 75 screen ids are in the full graph.
- **Screen state.**
  - 52 screen jobs are finished.
  - 1 is running: PhonemizeWithSilJob.NpoY1pGJWNUJ, Slurm 4342305_1, the only job in squeue.
  - 22 are not set up yet, including ctrl_20 GiT88bxzoZbZ.
- **The 89 full-only jobs.** None of them has a job dir, so no stale state is left from an earlier
  attempt. No set-up graph job has a `hold` or `error.*` marker.
- **The live manager loaded the same graph.**
  - Manager 1583423 started at 18:32:36. The screen config dates from 15:47 and settings.py from 18:19.
  - No .py file under the package, i6_core, the pinned RETURNN or sisyphus changed after 18:32.
    The only hit is a zeyer file with a 2034 mtime, which is not imported.
  - Its in-memory graph is therefore the dumped one, and its PATH starts with sae/bin.
- **The six trainings.**

  | arm | job |
  |---|---|
  | ctrl_20 | GiT88bxzoZbZ |
  | ctrl_20_s1 | DvVfxf1LrCBi |
  | k2lat_20_ma3000 | jcKXbLMDk4hl |
  | ctrl_20_rc | llSFybyKXkbL |
  | gold phi | Ac2eioZbRX7d |
  | p0 | Y1vbqR6KeJSx |

- **Each arm differs from ctrl_20 in its intended delta only.** Apart from the job's own model path,
  the written configs differ from ctrl_20 as follows:
  - s1: the flat-init job (DMSwTLXT9MWG instead of 0J9d6wjrkRYH), `random_seed = 1` and
    `random_seed_offset: 1000`;
  - rc: `"sil_run_collapse": True`, nothing else;
  - k2: the `lexlat_k2_*` keys, with HLG avjHv1Xvjyqd.
  - All four arms share `num_epochs` 20, keep [1,4,10,20] with keep_last_n 1 and keep_best_n 0,
    batch 88000 / 128, and log_verbosity 5.
- **Reads, all registered.**
  - Greedy PER: 17 jobs, one per arm at epochs 1/4/10/20, plus p0 at its selected checkpoint.
  - Epoch 20: derangement and decode gaps for all 4 arms.
  - Paired deltas, 9 jobs:
    - k2lat − ctrl_20 at 20;
    - ctrl_20 − ctrl_20_s1 at 1/4/10/20;
    - rc − ctrl_20 at 1/4/10/20.

    `per_a` is the base and `per_b` the candidate (common.py:118).
  - JS rows: `p0` over three arms, with rows k2_vs_ctrl and ctrl_vs_s1; `p0_rc` with the row rc_vs_ctrl.
  - `learning_rates` of all 6 trainings. These carry dev l_tau, agg, reverse, the k2 monitors and
    gold phi's epoch-8 NLL.
  - phi_ep8.pt; p0's `selected_epoch` and its PER.
- **No G0.R1-R3 or G0.RC read is missing.**
  - The step-1 clauses are read from `returnn.log`: log_verbosity 5 prints the losses of every step.
  - G0.RC's log Z has no key of its own. The loss is −log Z_τ (lattice.py:1294), so "rc log Z <=
    ctrl" reads as step-1 `l_tau`(rc) >= `l_tau`(ctrl_20) on the same batch.

## 2. Resources

- **Every training's `run` task.**
  - 16 CPU, 64 GB, 1 GPU, gpu_mem 96, 72 h (`--time=4320`), `-p gpu_48gb`, untagged.
  - `create_files` and `plot` run on the desktop LocalEngine with 1 CPU and 1 GB.
  - gpu_mem is only a routing key: Slurm receives `--gres=gpu:1`. The screen's 40 GiB check is the real
    bound, and the reference k2 peak was 31.77 GiB.
- **gpu_48gb hardware.**
  - The partition is cn-506..509, each with 4 L40S, 96 CPUs and 500 GB, so 4 trainings fit on a node.
  - No higher-tier partition shares these nodes, so partition_prio preemption cannot requeue them.
- **Other tasks.**
  - 24 flexible forwards go to gpu_24gb (A10; the RTX 3090 cn-290 is drained) or to gpu_48gb.
  - 8 reverse-score forwards run on CPU with device=cpu.
  - The HLG job goes to gpu_32gb without a GPU. The other CPU jobs use the default partition.
  - No code path emits gpu_11gb or an A100 partition.
- **Only L40S, including resubmits.**
  - `engine.get_rqmt` (engine.py:103-107) rebuilds the rqmt from the recipe keys only, so `sbatch_args`
    is always recomputed.
  - Recomputing gives gpu_mem 96 > 24, hence `-p gpu_48gb` (settings.py:396). The routing code never
    sees a training.
  - `update_engine_rqmt` changes only time and mem.
  - A training cleared with `-co` gets a new dir and the same rule applies.
  - `./gpu_route --rebalance` excludes ReturnnTrainingJob.
  - One path is outside this review: a manual `scontrol update`.
- **Six trainings against the cap of 5 (gpu_48gb_qos gres/gpu=5).**
  - Slurm counts the cap on running jobs only, and no training consumes the output of a read or of
    another training (checked on the graph inputs). So there is no cycle and no deadlock; a waiting
    job only waits.
  - The order in which the trainings become runnable:
    - rc: at once, since it has the same inputs as ctrl_20;
    - s1: after FlatRecognizerInitJob, 1 h of CPU;
    - gold phi and p0: after the train-clean-100 MFA download (a local task), SeedGoldPhones (4 h) and
      the targets chain;
    - k2lat: last, after WordWindowReplay (4 h), KenLM, LexiconTrie (4 h) and HLG (6 h on gpu_32gb).
  - k2lat is therefore the 6th. It waits (QOSMaxGRESPerUser) only if gold phi and p0 are still running
    when HLG ends.
  - Separately, at 22:45 only 4 of the 16 L40S were idle, so a training may also wait on Resources.
- **Note: read forwards can queue behind the user's own trainings.**
  - Ties go to gpu_48gb (settings.py:221).
  - With 5 of the user's GPUs on gpu_48gb, the headroom there is 0. gpu_24gb currently scores 0 as
    well: at 22:45 `./gpu_route --cpu 4 --mem 24` gave gpu_24gb idle 2, fit 0, score 0.
  - A posterior forward then routes to gpu_48gb and stays QoS-pending until one of the trainings ends,
    which takes hours. It may also take that slot ahead of k2lat.
  - `--rebalance` leaves ties where they are.
  - The cost is delay only: the reads are not on any training's critical path, and their checkpoints
    are kept.

## 3. Switching managers

- **Local tasks survive a manager kill.**
  - The LocalEngine starts each task with `start_new_session=True` (sisyphus/localengine.py:123).
    SIGTERM to the manager therefore does not kill running local tasks.
  - The new manager re-adopts a live task from the pid in its usage file (`try_to_recover_task`,
    :309) and reports it as RUNNING.
  - Local tasks that were queued but not yet started die with the old manager and are submitted once
    by the new one.
  - The skill's statement "killing a manager kills its short jobs" does not hold for this checkout. The
    error is in the safe direction.
- **Slurm tasks are matched by job name in squeue** (slurm engine `task_state`, :410). A running
  ctrl_20 and any queued job read as running or queued, and are not resubmitted.
- **The only double-submit path is two live managers.** The skill's `ps` pattern with
  NAME=sae_i6_p0 also matches the `_screen` manager, so the pre-start check guards against it.
- **Do not copy `-co` from 1583423's command line (`m -co -r ...`).**
  - With `-co`, a training that is in error at the start would be moved aside together with its
    checkpoints.
  - Start plain.
  - A backgrounded manager dies with EOFError at "Clear jobs in <state> state?" (manager.py:558) when
    jobs are in error or interrupted_not_resumable. Such jobs must be handled first:
    - stateless jobs: `-co`;
    - a training: rename its `error.run.N` and `log.run.N`, never `-co`.

Sequence, from the setup dir:
1. `kill 1583627`. This is the screen watcher. Otherwise it wakes with a manager-exit verdict.
2. `kill 1583423`. Then
   `ps -eo pid,args | grep -E "m (-\S+ )*-r config/sae_i6_p0" | grep -vE "grep|bash -c"` must print
   nothing.
3. Take the skill's graph-aware status with `console config/sae_i6_p0.py` under the sae python.
   - Expect no error, interrupted_not_resumable or retry_error, and GiT88bxzoZbZ running or queued.
   - Otherwise follow the rule above.
4. `NAME=sae_i6_p0`, then
   `(PATH="/work/asr4/hwu/conda/envs/sae/bin:$PATH" setsid nohup /work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis --log_level 30 m -r "config/$NAME.py" > "log/$NAME.manager.log" 2>&1 < /dev/null &)`.
   - Then write the pid from `ps` to `log/$NAME.manager.pid`; it must be exactly one pid.
   - Check `tr '\0' '\n' < /proc/<pid>/environ | grep '^PATH=/work/asr4/hwu/conda/envs/sae/bin:'`.
5. Start the watcher on the new pid with `config/sae_i6_p0.py`, with SIS_LAUNCHER and PATH as in State.
6. After the first new create_files (llSFybyKXkbL or DvVfxf1LrCBi):
   - there must be no `error.create_files.*`;
   - squeue must show `ReturnnTrainingJob.GiT88bxzoZbZ.run` exactly once.

## 4. black and create_files

- **black path.** With sae/bin first on PATH, all 44 ReturnnConfig objects in the full graph carry
  `_black_path=/work/asr4/hwu/conda/envs/sae/bin/black` (black 26.3.1, shebang sae python3.11).
- **Configs written.**
  - The 6 training configs and their `rnn.sh` commands write through black.
  - The 32 forward configs go through `create_returnn_config(model_checkpoint, ..., device)`, as
    create_files does, and write through black.
  - 0 failures.
- **Regeneration is exact.** The regenerated configs of the 7 finished forwards are byte-identical
  (`cmp`) to their on-disk `output/returnn.config`.
- **Interpreters.** Every run command uses the sae python and the pinned RETURNN
  CloneGitRepositoryJob.KQ3NuCaDE6QH.

## 5. ctrl_20_rc runtime code

- 51f4def2d is an ancestor of HEAD (551befc69, and e23d8e886 at 22:51).
- `git diff 51f4def2d HEAD` touches only `exp_logs/SAE_i6/`: 0 other files, re-checked at 22:51.
- The package worktree equals 51f4def2d (`git diff --quiet`). It has no untracked files, and no ignored
  files other than `__pycache__`.
- The committed silfix diff matches the reviewed `/work/asr4/hwu/tmp_dev/silfix.patch` line for line.
  `config/sae_i6_p0.py` equals the reviewed `silfix_setup` copy.
- model/, training/ and analysis/ are therefore exactly the reviewed code. Keep them frozen until the P0
  trainings end, because RETURNN re-imports them live on every resume.

## Side effects

- Scratchpad files only. Nothing was started or stopped.
- `./gpu_route` made one read-only Slurm query.
- No bytecode was written in the setup.
