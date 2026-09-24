# Review: availability-based GPU partition routing (2026-09-24)

Verdict: PASS_WITH_NOTES (DONE_WITH_CONCERNS). The three parts of the criterion hold. Job ids are
unchanged. Routing of non-GPU tasks is unchanged. No path sends a task with gpu_mem > 24, or a
training, to the wrong GPU type. No Slurm failure can stall or crash the manager. Four findings remain.
Finding 1 is the most important: the "free" count does not measure real availability. Because of it,
new flexible forwards are routed right now to A10 GPUs that Slurm cannot schedule.

Reviewed files, pinned by sha256 prefix:
- settings.py a8228ad5b2f0ed32;
- gpu_route e699e663869a7260;
- gpu_route_test.py 98fb8705a874dd44;
- baseline settings: implementer scratchpad `settings_before_route.py` d8fc0f3bf1201983 (115 lines).

`diff` against the baseline shows only three changes: the routing block (lines 34-236), the
check_engine_limits docstring, and line 255 (`-p gpu_route_partition(...)`, `--comment=flex24`).

## Findings (most severe first)

1. **settings.py:141: "free" counts idle GPUs whether or not a job can use them.** Right now both free
   A10s sit on nodes with 30/32 CPUs allocated: cn-504 and cn-505, 1 idle GPU each. `sbatch --test-only`
   for 1 GPU on gpu_24gb gives these start estimates:
   - 1 CPU / 4 GB: 2026-10-08;
   - 4 CPU / 64 GB: 2026-10-10.

   The same 4-CPU request on gpu_48gb would start immediately on cn-508.

   The scores are gpu_24gb 2 against gpu_48gb 1, so the router picks gpu_24gb. The gpu_48gb headroom is
   5 − 4 because the 4 forwards the coordinator held and moved count as mine. In the current graph, the 4
   new ctrl_20_rc flexible forwards route 3 to gpu_24gb and 1 to gpu_48gb.

   `--rebalance` uses the same score, so it would not move these jobs. Stickiness then keeps a job that
   never started on its submit_log partition. This is the "strands jobs" failure. It is not a regression:
   the baseline sent every job with gpu_mem ≤ 24 to gpu_24gb. The cause is that gpu_route_partition does
   not receive cpu/mem, so it cannot check whether a job fits a node.

2. **settings.py:124-129 and 227-233: no lock, while the manager calls this code from thread pools.**
   - Where it runs: state updates use `for_all_nodes` with gs.GRAPH_WORKER=16. Submissions use
     `run_jobs` with gs.MANAGER_SUBMIT_WORKER=10.
   - Query count: when the cache has expired, every thread runs its own query.
     - In a simulated 16-thread state pass over 20 flexible tasks: 32 calls (16 sinfo + 16 squeue) in
       0.45 s, then 0 calls while the cache is valid.
     - In a 10-thread submit burst: 20 calls.
     - The design is 1 pair per 60 s.
   - Burst fix: each query replaces `_GR_CACHE` and drops the reservations made in earlier snapshots.
     - A 13-task burst with a cold cache split 11 to gpu_24gb and 2 to gpu_48gb, and only 1 reservation
       survived (3 of 3 trials).
     - With a warm cache it split 7:6, the same as the sequential reference.
   - When this happens: new jobs are routed only in the submit pool. The cache is cold whenever no
     flexible task was state-checked in the last 60 s, for example when read forwards become runnable
     after a long training. So the burst fix mostly does not work when it is needed.
   - Another possible effect (not observed): `_gr_scores` can iterate `resv` while another thread
     inserts into it. That RuntimeError is caught and the job gets the static gpu_24gb.
   - No stall: with the controller unreachable (bogus SLURM_CONF), 16 parallel sinfo calls were each
     killed at 2.0 s and the pass took 2.02 s. The failure is cached, so the next pass takes 0 s, and no
     sinfo process is left behind.

3. **settings.py:174-179: the submit_log fallback returns the FIRST flexible `-p` (`break`), not the
   previous attempt's.** submit_log never records a move made by hand or by rebalance.
   - When it matters: a moved job whose usage host cannot be mapped goes back to its original
     partition. That happens when the snapshot failed in the last 60 s, when the usage file is
     missing or unreadable, or when the job never started.
   - Jobs affected now:
     - 4335682/84/85/86 (khVKNe5qaTxE, UzlPyMEw1eR4, AjTxcYsLi4E6, vVoEfyB8FVzF): submit_log says
       gpu_24gb, but they are held on gpu_48gb.
     - 5z8bUupJ1f8j and QO3M1P9dOc2o: they ran on cn-508 and are now in error. Clearing the error
       moves their directory aside, so they then route fresh.
   - Once a job has run and the snapshot is available, the usage host wins, which is correct. All
     affected jobs are forwards.

4. **gpu_route:45: rebalance candidates include ANY of my untagged pending gpu_24gb jobs.** That covers
   jobs from the other 3 setups under /u/hwu/setups/librispeech-960, not only this one. There is also no
   check for resubmits. A training resubmit from another setup that ran on A10 and is pending on gpu_24gb
   would be moved to L40S. A tagged resubmit of any future flexible training (gpu_mem ≤ 24) could be moved
   in either direction.

   In this setup no training can be a candidate: all 6 ReturnnTrainingJob tasks (including ctrl_20_rc)
   request gpu_mem 96, so they are untagged on gpu_48gb. Jobs that "stay" only add reservations
   (`target == cur`) and can never move anything.

Minor points:
- `_GR_OK_NODE_STATES` strips state suffixes, so "idle~" (powered off) and "mixed-" (planned) count as free.
- Another user's job pending on both partitions is counted as competing on both (none exist now).
- GPUs requested per task (`tres-per-task`) are not counted.
- The output text "(tie -> gpu_48gb)" is a fixed label, not a sign of a tie. The pick logic is right:
  scores 1 vs 4 gave gpu_48gb, and scores 2 vs 1 gave gpu_24gb.

## Checks, with evidence (scratchpad `.../scratchpad/rv/`)

1. **Job ids and rqmt.** `dump.py` loads each config under the old and the new settings. It records
   every job id and, for each task id, the full `get_rqmt(update=False)` result and the Slurm
   `options()` line. Results:
   - P0: 139 jobs, sha1 578dd2d8…, identical under both settings.
   - Screen: 75 jobs, sha1 db40328d…, identical.
   - All rqmt keys of all 183 + 98 task ids are identical, except the sbatch_args of flexible GPU
     tasks (20 in P0, 11 in the screen, all ReturnnForwardJobV2 with gpu_mem 24).
   - Every non-flexible sbatch line is identical: CPU tasks, gpu_32gb, and the trainings'
     `-p gpu_48gb`.
   - After the ctrl_20_rc edit (config 18:12), P0 has 164 jobs. All 139 original ids are present
     and their rqmt lines are identical. The 25 new ids include 1 training with gpu_mem 96, which
     goes to gpu_48gb untagged. The screen still has 75 jobs with the same sha1.
2. **Sticky resubmits.**
   - TIMEOUT: the task becomes interrupted_resumable, then `submit` → `get_rqmt(update=True)` →
     check_engine_limits → sticky partition.
   - queue_error: once the job leaves the queue, the task becomes runnable or interrupted and is
     sticky from the usage host or the submit_log.
   - Cleared error: `_sis_move` moves the directory to `.cleared.NNNN`, and the task routes fresh as a
     new run.
   - Missing or garbled files: a missing submit_log gives a routed pick. A garbled submit_log gives
     None and then a scored pick (sisyphus's own `get_submit_history` would fail first). A garbled or
     unknown usage host falls back to the submit_log. A race in stat/listdir is caught and gives a
     scored pick.
   - Trainings are unaffected, because gpu_mem 96 never reaches the routing code.
3. **Slurm failures.** Timeout, CalledProcessError, OSError, empty sinfo and an unreachable controller
   all give the static rule, and `gpu_route_partition` never raises. The worker calls get_rqmt in the
   LoggingThread init. It queries Slurm only for a resubmit that has a usage file, which delays the start
   by up to 4 s. Query rate: at most 1 refresh per 60 s per process, that is every other 30 s manager
   loop, multiplied by the thread fan-out described in finding 2.
4. **Live counts** match a hand count of sinfo/squeue:
   - gpu_24gb: free 2 (cn-290 drained* excluded), competing 0 (every other pending job is
     QOSMaxGRESPerUser), mine 1/6.
   - gpu_48gb: free 4, competing 0, mine 4/5.

   No job is pending on more than one GPU partition. Other users' multi-partition jobs appear as one
   line with a comma-separated partition field.
5. **Rebalance:** only PD jobs are moved; running jobs never are. Untagged jobs never go to gpu_24gb.
   Jobs with gpu_mem > 24 are never candidates in this setup. The dry run before the hand move matched
   the manual trace (4 moves: 686, 685, 684, 682).
6. **Tests and sbatch:** 23/23 OK under both python 3.11 (sae) and python 3.10. `sbatch --test-only`
   accepts `--comment=flex24`.

## Delta review, round 3 (settings.py sha256 43585da6c5949df6, gpu_route 957c59f7209cff3a, tests 2e9bb9c655034599)

Verdict: PASS_WITH_NOTES. All four round-1 findings are fixed, and the criterion holds for the delta.
The routing block (settings.py lines 34-373) and `check_engine_limits` lines 375-400 were read in
full. Lines 1-33 and 401-459 of settings.py are byte-identical to the pre-routing baseline.

### What was checked, and the evidence

1. **Ids and non-GPU routing.** The configs were dumped under the pre-routing baseline settings and
   under the new settings.
   - sae_i6_p0: 164 jobs, ids sha1 61a14191. All 139 original ids are present.
   - sae_i6_p0_screen: 75 jobs, ids sha1 db40328d.
   - Every rqmt line and every non-flexible sbatch line is identical to the baseline.
   - The only differences are the sbatch_args of the ReturnnForwardJobV2 `run` tasks, which are the
     only flexible tasks (gpu_mem 24): 24 in P0 and 11 in the screen.
   - Every ReturnnTrainingJob still gets `-p gpu_48gb` (gpu_mem 96, line 396) and the 72 h rule.
2. **Fix 1, per-node fit.** Fit is computed per node from idle GPUs, idle CPUs (CPUsState) and
   Memory − AllocMem, in MB. The request memory is ceil(GB)·1024, which matches the Slurm engine's
   `--mem=%iG` (ceil). Checked against live sinfo at 18:27:
   - cn-504 and cn-505 each have 1 idle A10 but only 2 idle CPUs. The fit is 0 at 4 CPU and 2 at
     1 CPU / 4 GB, as `./gpu_route` printed.
   - gpu_48gb has no idle GPU; its fit is 0.
3. **Fix 2, one lock.** There is one module Lock, taken once, in `gpu_route_partition`
   (line 363), with a timeout of 10 s. Nothing under the lock calls back into routing, and the
   release is in `finally`. The lock cannot deadlock.
   - Concurrency simulation, ThreadPool 16, 20 flexible tasks, real Slurm: 1 sinfo, 1 squeue and
     1 sacct per flexible training per 60 s. A pass takes 0.36 s cold and 0.01 s warm, with 0 lock
     timeouts and no exceptions.
   - Burst of 13 tasks from ThreadPool 10: cold, warm and sequential give identical splits
     (6/7 at 4 CPU / 64 GB; 7/6 at 1 CPU / 4 GB), with 2 Slurm calls. A repeat call returns the
     same partition, so the state check and the submit agree.
   - Hanging Slurm, every call timing out after 2 s, with no flexible training (the P0/screen
     case): 2.0 s per 60 s. All picks fall back to static gpu_24gb, with no lock timeout.
4. **Fix 3, stickiness.**
   - A task whose last Slurm id is in my squeue (jobid or array id) keeps that partition and
     reserves nothing. Live, 4335683 (NxXNQ8v87CCM) and 4335685 (AjTxcYsLi4E6) resolve to gpu_48gb
     although their submit_log `-p` says gpu_24gb.
   - Otherwise only i6_core ReturnnTrainingJob tasks are sticky, in the order sacct, then usage
     host, then the last flexible `-p`. On real jobs sacct parsing gives gpu_48gb:
     - 4334397 and 4334398 are final and cached for good;
     - 4335683 is RUNNING and cached for 60 s.
   - TIMEOUT and queue_error keep the submit_log and resolve through sacct. A cleared error moves
     the job dir, so the training restarts from scratch and routes freely, which is correct.
   - Forwards route freely on resubmit.
5. **Fix 4, rebalance.** Candidates are my PD GPU jobs on a single flexible partition whose StdOut
   lies under `<setup>/work/` (or its realpath), excluding ReturnnTrainingJob by name or StdOut.
   A move to gpu_24gb requires the flex24 tag or submit_log gpu_mem ≤ 24.
   - Live, 4335686 is a candidate: StdOut is under work/, and gpu_mem 24 was read from
     submit_log.run through dirname(dirname(StdOut)).
   - The dry run moves 0. gpu_24gb fit is 0 at 4 CPU; gpu_48gb headroom is 1 once the candidate
     is excluded; the result is a tie, which stays on gpu_48gb.
   - Tests: 31/31 pass under python 3.11 (sae) and python 3.10.

### Notes (no failure in this setup's configs)

1. settings.py:363-364. When the lock wait times out, the call returns static gpu_24gb before
   `_gr_training_sticky` runs.
   - In simulation, 6 flexible trainings (gpu_mem ≤ 24) each needed sacct while sacct hung
     (2 s each, serial under the lock). The pass took 12 s, 4 threads hit the 10 s lock timeout,
     and 1 training got gpu_24gb instead of its sticky gpu_48gb.
   - This is latent: every training in sae_i6_p0 and the screen has gpu_mem 96 and never reaches
     this code.
2. settings.py:279/289. sacct runs under the lock, once per training id. Unknown, non-final or
   failed results are re-queried every 60 s.
   - The worst serial Slurm time per 60 s is 2 s (sinfo) + 2 s (squeue) + 2 s × (flexible
     trainings that need sacct).
   - For P0 this is at most 4 s per minute, and 2 s when sinfo itself hangs.
3. Live state differs from the dispatch. At 18:27 squeue shows 4335682-85 RUNNING and 4335686
   PD (Resources), all on gpu_48gb, with none held. The dispatch said all five were held.
4. The worker's LoggingThread (sisyphus/worker.py:62) also runs the router on the compute node.
   That costs 2 Slurm calls at job start and has no effect on placement.
