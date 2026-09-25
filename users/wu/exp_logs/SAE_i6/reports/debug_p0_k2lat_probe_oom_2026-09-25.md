# Debug: G0.K2M probe B (chunk 8) OOM on the V100, and probe A's silence (2026-09-25)

Status: DONE_WITH_CONCERNS. This was a read-only diagnosis. Nothing was cancelled, modified or resubmitted, and no Sisyphus job dir was touched.

Abbreviations:
- S = `/u/hwu/setups/librispeech-960/2026-09-24-unsupervised`
- P = `S/recipe/i6_experiments/users/wu/experiments/unsupervised_asr`
- B = `/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs8` (Slurm 4362705)
- A = `/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25` (Slurm 4362704)
- J = `S/work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl`

Both probes were seeded from J's `epoch.003.pt`, presented to RETURNN as epoch 7 (`seed_record.json`; `reports/launch_p0_k2lat_v100_probe_2026-09-25.md`:12). Both used `lexlat_k2_max_active` 3000 and a batch of 88000 features with at most 128 sequences. Step 0 of sub-epoch 8 is the long batch: 115 sequences by 764 frames.

## Verdict

The first real failure is B's main-thread forward `k2.intersect_dense_pruned` inside the per-chunk loop.
- Log: `B/slurm-4362705.out`:753.
- Python call site: `P/reverse_model/rt_chunked_backward.py`:258, in the loop at lines 252-277.
- k2 site: `MultiGraphDenseIntersectPruned::PropagateForward<32>`, frame #8 at log line 763. This is k2 `intersect_dense_pruned.cu`:945, which allocates one float per expanded arc of a single frame.
- The failed request was 118 MiB, which is 30.9 M expanded arcs in one frame across the 8 sequences of the chunk.

The failure belongs to the model code's runtime configuration: the per-chunk k2 leg at chunk 8 and max_active 3000 does not fit beside the bed on a 32 GB card. It depends on the data: the epoch-3 weights give flat posteriors and therefore dense lattices.

It is not a tooling regression. i6 k2 is a source build of commit ec31d2c, the same master commit as JUPITER's 1.24.4.dev20260921 (`reports/env_build_2026-09-24.md`:22).

The segfault at line 795 is downstream. In k2 source (observed), `Intersect()` submits `BackwardPass` to the global pool at `intersect_dense_pruned.cu`:208 and waits for it only at :238. The OOM thrown at :217 unwinds past that wait and destroys the intersector while the pool thread is still inside `PruneTimeRange` (log line 831). That is a use-after-free, and exit 139 follows at line 838.

## Q1. Where in the step (observed)

The sequence of events in B's log:
1. HLG print (line 751).
2. Stability read, which completed: median 0.0194 over 16 of 16 (line 752).
3. OOM (line 753).

The OOM is in the chunk loop's forward. The C++ stack at lines 763-765 runs from IntersectDensePruned through Function.apply up to Py_BytesMain, so it is on the main thread. That rules out several other candidates:
- **Stability read:** it completed and printed, and it catches its own errors (`P/model/lexlat_k2_train.py`:650).
- **Per-chunk backward:** it runs on the autograd device thread and never calls IntersectDensePruned.
- **H leg:** `k2.intersect_dense` runs after the loop.
- **float64 cycle lattice and optimizer:** neither is in the stack.

The log does not name the chunk index. Timing points to chunk 1:
- The last MEMORY line is at 16:47:52 and the crash at 16:48:13-14.
- In that window come the HLG load, the stability read (22 s or less in A for the same 32 single-sequence calls), and the leg.
- So the leg ran for only seconds before the OOM.

## Q2. What makes up the 29.96 GiB

Observed, from line 753:
- 29.96 GiB allocated.
- 1.34 GiB reserved but unallocated (fragmentation).
- About 0.37 GiB outside PyTorch (the CUDA context).
- 57 MiB free.

Inferred:
- **Bed resident during the leg:** 20.28 GiB or less. This is the whole forward-plus-backward peak of the ctrl_20 bed measured on this V100 (cn-32) at f88000_s128 (`S/analysis_out/v100_bench/v100/summary.txt`; 28.22 GiB reserved).
- **No measured bed peaks:** ctrl_20_s1 (DvVfxf1LrCBi), ctrl_20_rc (llSFybyKXkbL) and J before k2 log no `mem_usage` because `torch_log_memory_usage` is off. sacct records no GPU memory. J's pre-on-set path is the ctrl_20 bed, so the bench is the only bed number available.
- **HLG on the device:** about 3 GiB, computed from 24.9 M states and 103.2 M arcs (arcs 1.65 GB, aux_labels 0.41, raw score clone 0.41, row splits). Not measured.
- **Leaf, dense input and H graph:** under 0.1 GiB.
- **Therefore:** the k2 working set of chunk 8 at the failure point is at least about 6-7 GiB. It consists of per-frame ArcInfo records (16 B for each expanded arc, including arcs into pruned states), kept until their 30-frame window is pruned, plus per-frame transients.
- **Stability read:** it no longer occupied allocated memory. It runs under no_grad and its `finally` calls `empty_cache` (lexlat_k2_train.py:653-655).

i6 reference, observed. P1 rt_r90 (`ReturnnTrainingJob.EexT85vdfx25`) runs on an L40S with the same HLG, H leg and batch shape, on the per-chunk path at chunk 4 and max_active 1000.
- Its `lexlat_k2_pre_peak_allocated_gib` ranged from 16.8 to a maximum of 27.26 over 55 steps.
- Its lattice fell from 35,130 to about 300 arcs per frame over those steps.
- Its bed also carries a frozen reverse model.
- So the floor with small lattices on i6 is about 25-27 GiB. That leaves about 1-3 GiB under the 28 GiB bar for the k2 chunk.

## Q3. Probe A and J

**A cannot pass G0.K2M (inferred, strong).**
- k2 prunes per sequence: the beam and max_active apply per FSA (`P/model/lexlat_k2.py`:900-902). Chunk exactness is tested at chunk sizes 1, 2, 3 and 16 (`P/tests/test_rt_chunked_backward.py`:135).
- The resident memory outside the chunk is identical in A and B.
- So at B's failing frame, A holds B's arrays plus those of 8 more sequences: at least 30.08 GiB allocated, above 28.

**A's state (coordinator question).**
- Observed:
  - No output since the stability line (file mtime 16:46:41).
  - No `ep 8 train, step 0` line.
  - No Python exception, no k2 `[F]` line, no "terminate called" and no fault dump.
  - Still RUNNING at 17:05.
  - sstat fails for cn-32, and ssh to cn-32 fails host-key verification.
- Most likely code path: the per-chunk loop of `log_z_hlg_chunked_backward` in its first 16-sequence chunk. That is where B met the memory wall with half the sequences. The stability line is the last statement before it, and a completed step would have printed a step line.
- This is not legitimate compute. The same 16 sequences took 32 single-sequence k2 calls, 16 of them at max_active 10000, in 22 s or less, and J's whole step takes 43-51 s. A has now spent more than 20 minutes.
- It is also not a clean OOM. A main-thread OOM prints, as it did in B. A pool-thread OOM escapes `ThreadPool::ProcessTasks` uncaught (`thread_pool.cu`, no try/catch), which leads to std::terminate and an abort that would print.
- Two readings remain. The logs cannot tell them apart:
  - **(a) Thrashing at the capacity limit.** Each failed cudaMalloc makes PyTorch release its cache and retry, with a device sync each time. This needs sequences 8-15 to be light and A to be less fragmented than B.
  - **(b) A native stall without an error message.**
- JUPITER's int32 window-overflow hang (`exp_logs/SAE/reports/debug_k2_probe_timeout_2026-09-21.md`) is ruled out arithmetically. It needs more than 2^31 ArcInfo records (over 32 GiB) resident in one 30-frame window, so the V100 runs out of memory first.
- What would tell (needs a login on cn-32): `py-spy dump --native --pid <python pid>` twice, 60 s apart (a frozen stack means a stall, a moving one means thrashing), plus `nvidia-smi`. `scancel --signal=USR1` reaches the bash batch script and would probably end the job. Whether A is hung or thrashing does not change the fix or the gate. A's result cannot change G0.K2M.

**J (inferred; not measured on i6).**
- J's config has no per-chunk epilog, so at the on-set it runs the held path. That path is chunked at 16 (the default, or `$LEXLAT_K2_CHUNK_SEQS`) and holds every chunk until the outer backward. Its peak is therefore at least A's.
- With epoch-3 weights J would OOM at sub-epoch 8.
- The epoch-7 weights should give much sparser lattices. The indirect evidence:
  - **JUPITER same arm:** k2lat_20_ma3000, same on-set 8, same 88000/128 shape, held path. At the first on-set step, the leg's peak reserved was 17.4-21.9 GiB across all eight arms (`exp_logs/SAE/SAE_4A_lexlat.md`:492).
  - **JUPITER cold-start arms:** they showed 37,200 arcs per frame at step 0 and 1,400 by step 12, with 76.7-83.7 GB reserved peaks early (`SAE_4A_lexlat_v2.md`:128-137). Lattice size follows posterior sharpness by more than 20x.
  - **i6 rt_r90:** 35,130 fell to 1,463 arcs per frame in 10 steps.
- Caveats: those runs used GH200 reserved numbers, not V100 allocated. The i6 graph has 182,215 words against 151,731, so it has about 20% more fan-out at word boundaries. J's run on i6 is not JUPITER's.

## Q4. Smallest exact, hash-neutral change, ranked by evidence and exactness

1. **Per-chunk path with `LEXLAT_K2_CHUNK_SEQS=1` or `2`.**
   - The environment variable is read by `resolve_chunk_seqs` (`P/model/lexlat_k2.py`:907-925) and carried by no hash.
   - It is exact: pruning is per sequence, and the tests cover chunks 1 and 2.
   - Evidence that it fits the device: in both probes, the stability read on the same first 16 sequences made single-sequence calls at max_active 10000 (a larger per-call working set than 3000) on the same resident memory, and completed.
   - The i6 rt_r90 run's chunk-4 leg at max_active 1000 stayed at or under 27.26 GiB allocated, even at 35 k arcs per frame. Chunk 1 at 3000 has about the same working set as chunk 3 at 1000 (inferred).
   - Not shown:
     - that the peak stays at 28 GiB or under (the floor is about 25-27);
     - that it fits on sequences 16-114;
     - the stability read's own allocated peak. That peak counts toward G0.K2M through step 0's `pre_peak`, and no chunk knob changes it.
   - Cost: 115 k2 calls plus 115 backward calls per step; time to be measured.
   - J needs the per-chunk epilog applied, since the held path is not bounded by chunk size. The env var only reaches a worker started after it is set.
2. **Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.**
   - It is an unhashed environment variable, numerically neutral, and suggested by PyTorch's own OOM text.
   - It removes most of the 1.34 GiB of fragmentation and the risk of thrashing, but it does not lower allocated memory. It cannot meet the 28 GiB clause alone; use it together with (1).
3. **`empty_cache` after the stability read.** It is already present and has no effect on allocated memory.
4. **Not admissible:**
   - lower max_active, a smaller batch or a determinized HLG: each changes the loss or the hash;
   - the HLG on the CPU: not viable for the pruned intersection;
   - switching GPUs: excluded by the brief.

Next measurement: re-probe from the epoch-3 seed at chunk 2, and at chunk 1 if 2 fails, with `torch_log_memory_usage` and expandable_segments. The long step 0 decides it within about 5-10 minutes. Adding a per-chunk progress print (chunk index, `max_memory_allocated`) would make the next failure locatable.

Hazard for J: k2's exception-unsafety means an OOM inside the stability read's `intersect_dense_pruned` would not be contained by its `try/except`. Its orphaned pool task would segfault or hang the process, as in B. So "a diagnostic may not kill the arm" (lexlat_k2_train.py:605-609) does not hold for OOM. The read at max_active 10000 is the largest single call.

## Upstream

Searched k2-fsa/k2 issues:
- #475: the old intersection lookup table of states x minibatch; since replaced by the hash.
- #745: segfault in PruneTimeRange in multi-GPU training; a different trigger.
- #875: hang in PropagateForward; closed with no cause.

No issue was found for the exception-unsafety of `Intersect` (the OOM leading to the orphaned pool task). For PyTorch, expandable_segments is the documented fragmentation mitigation. The RETURNN pin checked out: no memory-history hooks, and `init_faulthandler` behaves as described.
