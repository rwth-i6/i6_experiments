# Availability-based GPU partition routing (2026-09-24)

Status: DONE_WITH_CONCERNS. Routing is in `settings.py` and the CLI is `./gpu_route`. This is the third
round: the original spec, then the burst fix, then the four review fixes
(`reports/review_gpu_route_2026-09-24.md`, findings 1-4). All 31 unit tests pass. The P0 graph keeps every
job id. I did not touch the live manager (pid 1554133): it keeps the old static routing until it is
restarted. Nothing was committed, launched or moved.

## Files (sha256 prefixes after round 3)

- `settings.py` (43585da6c5949df6, 459 lines; baseline 115 lines, reviewed round-2 version a8228ad5b2f0ed32).
  - Contains the routing block (`GPU_ROUTE_*`, `_gr_*`, `gpu_route_partition`).
  - In `check_engine_limits`, only the gpu_mem <= 24 branch changed. It now sets
    `-p gpu_route_partition(task, gpu, cpu, mem)` and `--comment=flex24`.
  - The 72 h training rule, the explicit `-p` passthrough, gpu_mem > 24 -> gpu_48gb and the CPU routing
    are unchanged.
  - Each round wrote `settings.py.new`, checked it (py_compile, sisyphus import, tests, graph build), then
    `mv`'d it over (an atomic rename).
- `gpu_route` (957c59f7209cff3a): stdlib-only CLI. It loads `settings.py` and contains `plan_rebalance`.
- `gpu_route_test.py` (2e9bb9c655034599): 31 unittest cases with mocked sinfo/squeue/sacct.
- `~/.claude/skills/sis/SKILL.md`, section `2026-09-24-unsupervised`: a 5-line entry, updated for round 3.

## Behaviour

Only a GPU task with gpu_mem <= 24 is routed. Everything else is as before.

- **Score per partition.** score(P) = min(fit - unplaced - competing, headroom), computed for the task's
  shape (gpu, cpu, mem). This is review fix 1.
  - **fit:** on each usable node of P, count how many jobs of this shape fit its idle GPUs, idle CPUs and
    unallocated memory, then multiply by the task's gpu count.
    - A node is usable when its state is in {idle, mixed, allocated, completing, planned} and has no `*`.
      Down, drain, maint and resv nodes therefore count nothing.
    - The data comes from one per-node query per refresh:
      `sinfo -N -O CPUsState,Memory,AllocMem,Gres,GresUsed`.
    - mem is rounded up to whole GB, the same way sisyphus passes `--mem`.
  - **competing:** GPUs of other users' pending jobs with reason Priority or Resources.
  - **headroom:** QoS cap (gpu_24gb 6, gpu_48gb 5) minus my PD/CF/R/CG GPUs and my reservations.
  - A tie goes to gpu_48gb.
  - The CLI also shows `idle`, the raw count of idle GPUs.
- **Reservations.** Each pick reserves the task's gpu/cpu/mem in the cached snapshot.
  - The reservation goes on the first node of P where the task fits. If none fits, it is "unplaced" on P and
    lowers fit.
  - It counts once per task (keyed by the submit_log path). It is dropped at the 60 s refresh.
- **Thread safety (review fix 2).** One `threading.Lock` covers cache refresh, pick and reservation.
  - A burst from the manager's 16 + 10 threads gets one sinfo+squeue pair and sequential reservations.
  - The lock is acquired once per call and never nested. The wait is capped at GPU_ROUTE_LOCK_WAIT_S = 10 s,
    after which the static rule applies.
  - Whoever holds the lock runs at most sinfo + squeue (+ sacct), each killed after 2 s. So the lock cannot
    deadlock or stall for more than about 6 s.
  - The manager's lock-free paths are unchanged. The CLI and tests call `_gr_snapshot` directly, single-threaded.
- **Queued jobs.** If the task's last Slurm id (from submit_log `engine_info`) is in the snapshot's queue, the
  task returns that partition and reserves nothing. This covers state checks of running or pending jobs, and
  it respects hand moves and rebalance moves.
- **Stickiness only for ReturnnTrainingJob (review fix 3).** Class and module are checked exactly, as in the
  72 h rule. The partition of the last Slurm job is taken, in this order:
  1. squeue (above);
  2. `sacct -X -P -j <last id> --format=Partition,State` (cached for good once the job has ended, else for 60 s);
  3. the host in the newest `usage.<task>.*` file;
  4. the last flexible `-p` in submit_log. This used to be the first `-p`.

  Other GPU tasks (forwards) route freely on every submit.
- **Rebalance (review fix 4).** Candidates are my PD jobs that meet all of these:
  - on a single flexible partition;
  - StdOut under `<setup>/work/` (or its realpath);
  - name or StdOut does not contain `ReturnnTrainingJob`.

  Tagging does not matter. A job may go to gpu_24gb only if it is tagged flex24 or its
  `<job dir>/submit_log.<task>` says gpu_mem <= 24. Unknown means never, so a 48 GB job is never sent to A10.
  All candidates are taken out of my usage, then placed one by one with their own gpus/NumCPUs/MinMemory
  shape, and each placement reserves before the next is planned. Running jobs are never touched.
  - Assumption: the brief says "WorkDir under work/". Slurm's WorkDir for these jobs is the setup dir itself,
    for this setup and equally for a sibling setup, while StdOut is `<setup>/work/<job>/engine/...`. So I
    filter on StdOut.
- Unchanged:
  - Slurm failure or timeout, which is cached for 60 s, falls back to static gpu_24gb.
  - `gpu_route_partition` never raises.
  - `sbatch --test-only --comment=flex24` is accepted.

## Checks

- `python3 gpu_route_test.py`: 31 OK, three times on 3.10 and once with the sae 3.11. Each review fix has
  tests:
  - Fix 1:
    - the live case: 2 idle A10s on nodes with 2 idle CPUs give fit 0, so a 4-CPU/64 GB task goes to
      gpu_48gb while a 1-CPU/4 GB task goes to gpu_24gb;
    - fit is limited by memory and CPUs;
    - reservations are node-level.
  - Fix 2:
    - 10 threads on a cold cache with a 0.2 s query give exactly 1 sinfo + 1 squeue, 10 reservations and the
      same 5/5 split as sequential routing;
    - a held lock plus a short wait gives the static rule and no query.
  - Fix 3:
    - a forward with a submit_log and usage on gpu_48gb routes freely, without sacct;
    - a queued (hand-moved) job keeps its queue partition and reserves nothing;
    - a training goes to the sacct partition, then the usage host when sacct fails, then the last `-p`;
    - sacct results for pending jobs expire, and those for ended jobs stay cached.
  - Fix 4:
    - only this setup's pending non-training jobs move;
    - another setup, a training, a running job and another user's job are untouched;
    - an untagged gpu_mem 24 job may move 48 -> 24;
    - a job with gpu_mem 48 or unknown never moves to 24.
  - Earlier coverage is kept: ties, QOS-pending jobs, down nodes, failure fallback and caching, the real
    2 s timeout, the 60 s cache, the CPU/-p/72 h rules, the burst split 3/3, and corrupt files.
- Import through `sisyphus.global_settings` worked, both with `SIS_GLOBAL_SETTINGS_FILE=settings.py.new` and
  after the swap.
- Graph build of `config/sae_i6_p0.py`. The config now has 164 jobs because ctrl_20_rc was added by
  another session. Scratchpad files: `route_base2.txt` (baseline settings) and `route_after3.txt`.
  - The ids sha1 61a1419117f432e2 is identical between the baseline settings and the new ones.
  - All rqmt except sbatch_args are identical.
  - Queued forwards keep their queue partition: 4 on gpu_48gb, NxXNQ8v87CCM on gpu_24gb.
  - The unsubmitted forwards go first to gpu_48gb (fit 4). Once the headroom there is used up (4/5 mine),
    they go to gpu_24gb.
  - The 6 trainings (gpu_mem 96) go to gpu_48gb, untagged.
- Live `./gpu_route --gpu 1 --cpu 4 --mem 64`, output after the swap:
  ```
  gpu_24gb: idle 2, fit 0, competing 0, mine 1/6, score 0
  gpu_48gb: idle 4, fit 4, competing 0, mine 4/5, score 1
  pick for 1 GPU, 4 CPU, 64 GB: gpu_48gb (tie -> gpu_48gb)
  ```
  `--rebalance --dry-run` would move 1: 4335683 gpu_24gb->gpu_48gb. That makes 5 on gpu_48gb, which is
  the cap. I moved nothing.

## Concerns / undetermined

1. Reservations are per process: the watcher's console reserves only in its own cache. A task whose
   submission is deferred keeps its reservation until the next refresh. An array task reserves once, not
   once per task id.
2. Once headroom on gpu_48gb is used up, flexible jobs go to gpu_24gb even when fit there is 0. Both
   scores are then <= 0, and the formula compares them without any wait estimate.
3. The worker's `get_rqmt` at job start can run one squeue/sinfo (and for trainings one sacct) on the
   compute node. This affects only `requested_resources` in the usage file.
4. Another user's pending job that lists both partitions counts as competing on both.
5. The `(mtime, size)` file cache could miss a same-second rewrite of equal size. This only matters for
   usage-host stickiness of trainings, where sacct comes first anyway.
