# L2-0 k2 pre-flight failure diagnosis (2026-09-23)
Status: DONE. Nothing relaunched or edited.

Only failed job in config/sae_4a_lexlat_v2_ladder_preflight.py: LadderK2PreflightJob.ZM3MD9viV7sM
Dir: work/speech_llm/sae/emc/blankfree_ladder_jobs/LadderK2PreflightJob.ZM3MD9viV7sM
Graph state interrupted_not_resumable (manager log 19:47). There is no error.run.1 marker. No upstream dependency failed.
Slurm 1971051_1 had state TIMEOUT after 02:01:15 (rqmt time 2 h). Step .0 FAILED with exit 127, CANCELLED DUE TO TIME LIMIT.

Cause: k2 int32 overflow in the first training step (a code/scale failure, and it fails the "no k2 overflow" criterion).
output/rt_r0/log.run.1:304  HLG on cuda:0: states 23,907,591, arcs 98,595,606
output/rt_r0/log.run.1:307  [F] k2/csrc/array.h:501 Array1<char>::Init Check failed: size >= 0 (-1885666895 vs. 0)
Stack: MultiGraphDenseIntersectPruned::PruneTimeRange -> Renumbering -> Array1::Init -> abort (signal 6). A core file of 4.6 GB was written at work/rt_r0/core.* (17:48).
RETURNN started at 17:46:23 and aborted at about 17:48:09, roughly 14 s after "Time to get first batch data". It completed 0 steps, and no checkpoint is in output/rt_r0/models.
Second defect (wrapper): after the child aborted, the parent job did not notice. It kept sampling nvidia-smi until the 2 h limit, so about 1 h58 of GPU time was wasted.

Measured: steps 0; no sec/step; startup about 1m46 to the crash. nvidia-smi peak on GPU0 was 63281 MiB (about 61.8 GiB, output/gpu_samples.csv col 4), and 43 MiB after the crash.
ABORT json: none in work/rt_r0/ or in the job. The root-level ./lexlat_k2_ABORT.json dates from Sep 21 and is a stale pytest artifact (hlg /tmp/pytest-...), unrelated.

Unchanged resubmit: NOT valid. It would overflow deterministically. The fix belongs to the implementer (intersect size/chunking/max_active) plus a fix for the wrapper's child-exit detection.
