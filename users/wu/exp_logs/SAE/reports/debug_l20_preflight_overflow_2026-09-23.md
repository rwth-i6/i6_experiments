# Debug: L2-0 pre-flight k2 int32 overflow (LadderK2PreflightJob.ZM3MD9viV7sM, Slurm 1971051)

Verdict: DONE. The first real failure is k2's int32 `tot_arcs` in `PruneTimeRange`. It happened
inside the stability diagnostic's reference call (max_active 10000). The main leg at the
registered rung 1000 was never reached. Owning layer: model code (the training runtime's
chunking policy in `lexlat_k2_train.py::_stability` and `LexlatK2Spec.chunk_seqs`), against a
hard limit in k2. It is not a config error, not the launcher, not the environment, and not a
tooling regression.

## Evidence (observed)
Job dir: `work/speech_llm/sae/emc/blankfree_ladder_jobs/LadderK2PreflightJob.ZM3MD9viV7sM`

- `output/rt_r0/log.run.1:307`:
  `[F] k2/csrc/array.h:501 Array1<char>::Init Check failed: size >= 0 (-1885666895 vs. 0)`.
  - C++ stack: Renumbering, then MultiGraphDenseIntersectPruned::PruneTimeRange (k2 pool thread).
  - Python stack: lexlat_k2_train.py:542 `step` -> :620 `_stability` -> lexlat_k2.py:1038
    `chunked_tot_scores` -> k2/autograd.py:742.
- Core `work/rt_r0/core.jpbo-020-48.jupiter.internal.1410502`, analysed read-only with gdb.
  - Crashing thread: LWP 1410964 (k2 ThreadPool). The main thread was in cuMemCreate, still
    running the forward pass.
  - PruneTimeRange registers:
    - tot_arcs = 2,409,300,400. The Renumbering allocates tot_arcs + 1, which wraps to
      -1885666895.
    - tot_states = 82,479,248.
    - num_t = 20, so this is the FIRST window [0,20).
  - Intersector object: num_seqs_ 16, T_ 37 (longest sequence 36 recognizer frames),
    **max_active_ 10000**, search_beam 20, output_beam 8, min_active 30, hash 2^28 buckets.
  - Walking `frames_` gives per-frame expanded arcs (16 sequences) for frames 0..22:
    2.94M, 101.7M, 116.7M, 157.7M, 225.5M, 222.2M, 183.4M, 116.8M, 35.4M, 117.5M, 120.4M,
    76.4M, 136.2M, 141.0M, 93.8M, 147.8M, 156.6M, 110.4M, 32.7M, 114.2M | 117.6M, 73.7M,
    133.7M.
    - The sum over frames 0..19 is exactly 2,409,300,400 = tot_arcs.
    - The sum of states over frames 0..19 is exactly 82,479,248.
    - Per-frame states peak at 31.6M (frame 4), about 2.0M per sequence.
  - Every per-frame count is divisible by 16, so all 16 sequences had identical dynamics.
    This is what a uniform posterior predicts: FlatRecognizerInitJob zeroes the output logits.
- The rung-1000 call on the same 16 sequences COMPLETED. The loop order is
  (spec.max_active, ref), and the crashing object has max_active 10000.
- Anchor, a run that works: cold k2lat_20
  (`PackedBlankfreeTrainJob.euWIAvH0xO5p/output/k2lat_20`).
  - Same flat_init.pt (0J9d6wjrkRYH), graph cdcxYJMjiYj5, tau 2, rung 1000, reference 10000,
    chunk_seqs 16. The only difference is on-set 8.
  - Its stability read passed: median 0.0254 over 16 of 16 utterances.
  - Its step 0 has 46 lattice arcs per frame, lexlat_k2_sec 2.9, peak reserved 20-31 GiB.
  - supphi_k2lat (supervised theta, on-set 1) also passed, at 0.0029.
  - So the same code, graph and constants work once the posterior is peaked. What fails is a
    uniform posterior at k2 on-set.

## Q1: what overflows and at what scale
k2 `intersect_dense_pruned.cu::PruneTimeRange` does two things:
- It sums `frames_[t]->arcs.TotSize(2)` over the pruning window into `int32_t tot_arcs`.
- It builds `Renumbering renumber_arcs(c_, tot_arcs)` (Array1<char> of tot_arcs + 1).

The quantity is the number of EXPANDED out-arcs of all active states over the window, summed
over every sequence in the call. At the failure it was 16 sequences x 20 frames at rung 10000,
which is 150.6M arcs per sequence (7.0% of 2^31). The memory (about 2.4e9 x 20 B, roughly
45 GiB) matches the 61.8 GiB peak.

## Q2: chunking, and one utterance per chunk
- Chunking is active, in SEQUENCES: `chunked_tot_scores`, `CHUNK_SEQS = 16`.
- The stability read sets `n = min(chunk_seqs, stability_seqs, B)` (lexlat_k2_train.py:603) and
  calls with `chunk_seqs=n`. So its 16 utterances go into ONE call. The sample size and the chunk
  size are the same variable.
- The window is fixed at 30 frames (20 for the first). Under a uniform posterior the per-sequence
  dynamics are identical, so the window sum does not grow with utterance length.
- Observed per-sequence counts at rung 10000:
  - window [0,20): 1.51e8;
  - steady state from frame 9: about 7.0e6 arcs per frame, so a 30-frame window is about 2.1e8.
- Inferred:
  - Allowing for the last-5-frames no-decrease rule (current_min_active = max_active/2, beam not
    reduced), the worst case per sequence is about 2.8e8. That is below 2^31 by roughly 7-10x at
    rung 10000, and rung 1000 is at or below that.
  - So chunk 1 is safe. Chunk 2-4 is likely safe (4 x 2.8e8 = 1.1e9). Chunk 8 or more is not.
- Main leg at rung 1000 with chunk 16: NOT shown safe.
  - In frames 0-8 both rungs follow the same beam trajectory, since states exceed 10000 per
    sequence until frame 8. That is 1.16e9 arcs per 16 sequences before the rungs diverge.
  - The rung-1000 [10,37) window, 27 frames, passed. A full 30-frame steady window over 16
    utterances of up to 110 frames is unmeasured.
- Settling measurement: one pre-flight with chunk 1 (or 4) for both legs, logging per-call sizes.
  Or walk `frames_` of a rung-1000 run.

## Q3: k2 controls
- None bound the window. `prune_num_frames = 30` and `prune_shift = 20` are hard-coded in
  `Intersect()`.
- `intersect_dense_pruned` has no max_arcs. (`max_states` and `max_arcs` exist only on the
  unpruned `intersect_dense`, which is a different objective.)
- `max_active` is advisory: above it the beam is multiplied by 0.8 per frame (hard-coded). Here
  states exceeded 10000 per sequence in frames 1-7, so ANY max_active below about 15000 leaves
  frames 0-8 unchanged. Only search_beam (the initial and cap dynamic beam) would shrink them.
- The per-sequence invariance of beams and max_active is what makes sequence-chunking exact.

## Q4: ranked remedies
1. Implementation only, no constant change:
   - Decouple the stability read's sample (16 utterances) from its chunk size, running
     chunk_seqs 1 (or a measured safe value) inside `_stability`.
   - Set the main-leg chunk_seqs to 1-4, via `LEXLAT_K2_CHUNK_SEQS` or a spec field.
   - Both are hash-neutral and claimed bit-identical per sequence
     (test_chunking_is_bit_identical_to_one_call).
   - Caveat: the env var alone also shrinks the stability read to 1 utterance, because n is
     coupled. That silently changes the BINDING monitor (item 5), so decouple in code.
   - Step-time cost at a flat posterior is unmeasured; the A7 gate checks it.
2. Disclosed pruning change, needing an amendment to 9.3: a smaller search_beam (or an initial
   beam) for the first sub-epochs. Lowering max_active does not help (Q3).
3. Design change, needing an amendment to A4: k2 on-set 8, as in k2lat_20 (observed to work).
   This changes what R1 measures.

Also: the `_stability` docstring says any k2 failure is caught. That is false for a K2_CHECK in
the backward pool thread, which calls std::terminate and then SIGABRT (here, and in
5o9P5uidnoWv sup_k2lat). No Python guard can catch it.

## Q5: L2-1 stage 2 / L2-2
- A1 removed L2-1 stage 2, so no k2 runs in L2-1.
- L2-2 `dec_joint` (cold random theta, on-set 1, tau 2) has the same exposure: a near-uniform
  posterior. The same chunk fix applies.

## Q6: GPU held for about 2 h after the child died
- Location: `blankfree_ladder_jobs.py:445` `for line in proc.stdout:` (Popen at :442, PIPE).
  - The deadline check (:455) runs only when a line arrives. `proc.poll()` is only in `finally`
    (:459).
  - rnn.py died by SIGABRT, so no atexit cleanup ran. Descendants that inherited fd 1 (the
    RETURNN DataLoader workers; `Send signal SIGINT ... train worker proc` lines after the
    abort) kept the pipe open. No EOF and no line ever arrived, and the job ran to the Slurm
    limit.
- Second hang point: `proc.stdout.read()` in `finally` would also block. SIGTERM goes to the
  pid only, not to the process group.
- Contrast: pack_jobs.run_arms writes to a file and polls, and it caught rc -6 promptly.
- Which process held the pipe is inferred, not observed.

## Upstream
I searched the k2 issues:
- There is no report of the PruneTimeRange tot_arcs int32 overflow.
- Related: #928 (int32 overflow in hash buckets, intersect.cu, open, no fix) and #745 (a
  different crash in the same Renumbering path, multi-GPU).
- Installed: k2 1.24.4.dev20260921, source ec31d2c9. The int32 is structural (Array1 dim_ is
  int32).
