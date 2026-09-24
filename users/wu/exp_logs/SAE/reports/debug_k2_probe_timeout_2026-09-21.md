# Debug: LexlatK2ProbeJob.Tkl94t85j4pV 2 h timeout (2026-09-21)

Status: DONE. Read-only diagnosis; nothing in the job dirs, recipe or env was changed.

## Verdict

The job was not slow, not thrashing memory and not stuck on graph load. The first k2 call of the
first timed cell (batch 0, 114 utterances x 257 recognizer frames, max_active 1000) died about
10 s after the HLG reached the GPU with a device-side memory fault inside
`k2.intersect_dense_pruned` (`log.run.1` line ~104: `[F] .../k2/csrc/eval.h:149 ... PropagateForward<32>
... Check failed: e == cudaSuccess (717 vs. 0) Error: operation not supported on global/shared
address space`). The k2 child process then never exited and the parent waited for it without a
timeout until Slurm cancelled the step at 2 h. Nothing was written because `partial.json` is only
written after a completed cell.

Root cause of the fault: a 32-bit integer overflow inside k2's pruned intersection, driven by the
per-state fan-out of the undeterminized escape HLG times the batch size. `intersect_dense_pruned`
expands EVERY out-arc of every active state before it prunes (`intersect_dense_pruned.cu`
`GetArcs` / `ai_lambda`, lines ~826-905) and keeps those `ArcInfo` records per frame for the
backward prune. The HLG has 1,129 states with more than 1,000 out-arcs, 77 with more than
10,000, and a maximum of 399,785 out-arcs from one state (the word-boundary states of the
undeterminized L.G: one arc per successor pronunciation). With 114 sequences that is roughly
1.6 M expanded arcs per sequence per frame; `PruneTimeRange` sums the arcs of its 30-frame
window into `int32_t tot_arcs` (`intersect_dense_pruned.cu:1383-1395`), the sum exceeds 2^31,
and the negative total is handed to `Array1` / `Arange`. On CPU k2's host-side check catches it;
on GPU (k2 built with `Sync kernels: False`) the negative sizes reach device kernels first and
surface as a sticky CUDA fault at whatever call next touches the driver, which is why the two
GPU jobs show it at two different lines of `PropagateForward<32>` (error 717 at `eval.h:149` in
this job; error 700 at `array.h:385` via `Renumbering::ComputeOld2New` in the sibling).

## Evidence (observed)

- `log.run.1` of Tkl94t85j4pV: child started 17:36:46-17:40; `HLG on NVIDIA GH200 120GB:
  {'states': 24736960, 'arcs': 143874554}` printed; then the `[F]` block; then nothing until
  `CANCELLED DUE TO TIME LIMIT` at 19:35. No Python traceback at all.
- `usage.run.1`: whole tree at 13.7 GB RSS, cpu 0.8 %, flat from 17:40 to 19:34
  (`out_of_memory: False`). The child's share stayed resident, so the child was alive and idle.
- Sibling `LexlatK2ProbeJob.R7QzD6vYBLD3` (official 3e-7 HLG, 9.8 M states / 81.6 M arcs), read
  only: same batch dump, `HLG on ...` at ~18:26:03, `[F] array.h:385 ... (700 vs. 0) an illegal
  memory access` inside `PropagateForward<32>` before 18:26:13, then idle at 0.6 % CPU since
  (will also burn to its 2 h limit at ~20:18). So the failure is independent of graph size.
- `output/gpu_check.txt`: k2 1.24.4.dev20260921+cuda12.6.torch2.7.1 passes on a 4-state FSA,
  1 x 6 x 3 emissions; the build report (`impl_k2_build_2026-09-21.md` sections "GPU check")
  states no larger GPU test was ever run.
- Input contract checked on CPU (all 9 dumped batches): `feat_lens` <= `log_q` time axis in every
  row, `log_q` finite and normalised, `(b, T, 40)`; HLG labels in [-1, 40] against 41 dense
  columns, properties `Valid|Nonempty|ArcSorted|EpsilonFree`, scores finite. Not an input bug.
- CPU reproduction under the same env (`/e/scratch/spell/wu24/envs/sae_k2/bin/python`), same
  HLG.pt, same `batch_00000.pt`, same beams / min_active 30 / max_active 1000 / temperature 2:
  - 2 utterances: passes, intersect 7.8 s, tot_scores -239.8 / -239.5, finite gradient.
  - 16 utterances: passes, 70.4 s, lattice 473,170 arcs, RSS 33.7 GiB.
  - 57 utterances: `[F] array.h:501 Array1<char>::Init ... size >= 0 (-1565110379 vs. 0)`.
  - 114 utterances (the job's batch): `[F] array.h:176 Array1<char>::Arange ... start >= 0
    (-81393326 vs. 0)` from `MultiGraphDenseIntersectPruned::PruneTimeRange` in k2's thread pool.
  - Fan-out of the failed HLG: mean out-degree 5.8, p99.9 108, p99.99 588, max 399,785;
    official HLG: max 403,419, 753 states > 1,000. Scripts and logs are in this session's
    scratchpad only (`cpu_repro.py`, `fanout.py`).

## Evidence (inferred)

- Why the child hung instead of dying: k2 prints the trace and then throws
  `std::runtime_error` (`log.h:176-213`, `K2_ABORT` unset). No traceback and no "terminate
  called" line ever appeared, so the process stalled during C++ unwinding on a CUDA context that
  had already taken a sticky fault (k2 array destructors release memory through the PyTorch
  allocator). The exact blocking call cannot be read off the logs. Both jobs also grew their VMS by
  ~80-100 GB after the fault (313 -> 393 GB; 311 -> 414 GB), consistent with the driver mapping
  device memory during fault handling; this is an observation, not a claim.
- The GPU threshold should match the CPU one (the overflow is host-side integer arithmetic in the
  same code path), i.e. somewhere between 16 and 57 sequences for this graph at max_active 1000.

## Owning layer

Primary: the probe configuration (batch of 114 sequences against a graph with ~400k-arc
word-boundary states) exceeding a k2 int32 limit in `intersect_dense_pruned`. The k2 build is not
broken (CPU and small GPU paths agree); the limit is upstream k2 code, present at the built
revision ec31d2c (2026-07-10). Upstream check: k2 issues #745 (illegal memory access in
intersect_dense_pruned, 2021, multi-GPU device mismatch) and #875 (hang in PropagateForward,
2021) are different settings; no issue found for this overflow or for aarch64/GH200/sm_90.
Secondary (why 2 h were lost): `lexlat_k2_jobs.py:556` `sp.run(cmd, env=_child_env())` has no
timeout (the HLG build's child call at line 201 has one), and `_child_env()` (line 51) does not
set `K2_ABORT`, so a k2 fatal does not terminate the child.

## What a fix would have to change

1. Keep the expanded arcs per 30-frame window under 2^31: run `intersect_dense_pruned` on chunks
   of the batch (<= 16 sequences verified on CPU; ~45 is the arithmetic ceiling for this graph) and
   sum the per-chunk totals, or reduce the word-boundary fan-out of the graph (the undeterminized
   escape L.G is what produces 400k-arc states; `max_active` does not help because it caps states,
   not arcs per state). Memory follows the same count: 16 sequences already held 33.7 GiB of frame
   data on CPU, so the 80 GiB bar has to be read per chunk.
2. `lexlat_k2_jobs.py:556`: give the child a timeout and kill its process group on expiry;
   `_child_env()`: set `K2_ABORT=1` so a k2 fatal aborts instead of unwinding.
3. Optional for the next GPU run: `K2_SYNC_KERNELS=1` in the child env names the faulting kernel
   directly if any GPU-only fault remains after (1).

The sibling job R7QzD6vYBLD3 is in the same state and will produce nothing; cancelling it saves
the remaining wall time.
