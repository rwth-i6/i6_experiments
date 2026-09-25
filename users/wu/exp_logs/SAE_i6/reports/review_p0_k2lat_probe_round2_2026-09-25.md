# Review: G0.K2M probe round 2 (chunk 2 and chunk 1), 2026-09-25

Role: code-reviewer, read-only. Nothing was submitted, and nothing in J, the setup or the probe dirs was changed.
The only things run were `read.py` on the round-1 dirs and on scratch fixtures (in the session scratchpad, `review_r2/`),
plus read-only `sinfo`, `sacct` and `squeue`. Checked against: the round-1 review (`reports/review_p0_k2lat_v100_probe_2026-09-25.md`),
the implementer report (`reports/impl_p0_k2lat_cs1_cs2_2026-09-25.md`), G0.K2M (`SAE_i6_P0.md`:144-150), RETURNN at the pin, and the package code.

## Verdict: PASS_WITH_FIXES

Both probes can launch. Each runs exactly round 1's B, with only the chunk size and the paths changed, and trains
sub-epoch 8 from J's latest checkpoint. Nothing blocks the launch. Two fixes change the outcome of a probe or how its
result is read:

- **SHOULD 1.** Give cs1 a longer time limit (for example `-t 6:00:00`); 3 h is marginal.
- **SHOULD 2.** Before anyone polls a live dir, `read.py` must stop printing PASS for a run that has not exited.

## Findings

**SHOULD 1: the 3 h limit is marginal for chunk 1, and a time-limit kill makes the run a FAIL (a wasted run).**
- Where: the cs1 command (`-t 3:00:00`) and `read.py`:208-209.
- What the code does per step (`rt_chunked_backward.py`:257-285): for each chunk it makes one `intersect_dense_pruned`
  call, one `get_tot_scores`, one `get_arc_post` (inside `_expected_words_one`, `lexlat_k2_train.py`:810) and one
  lattice `backward`, with a device sync in `int(value)`.
- There is no second intersection and no recompute: the backward uses the same chunk's lattice inside the loop. The
  brief's "again in the recompute" does not apply.
- At chunk 1 that is about 115-128 single-sequence passes per step, 57 steps in all (J's `ep 4` has 57 step lines),
  so about 7,000 per sub-epoch.
- Timing evidence:
  - The bed alone is 2427 s per sub-epoch (J's ep 4).
  - The stability read made 32 single-sequence no-grad calls on the longest batch (max_active 3000 and 10000) in
    22 s or less on this V100 (debug report, around line 43).
  - Allowing for the extra arc-posterior and backward work per sequence, the estimate is about 1.6-2.9 h for
    sub-epoch 8 plus startup and the dev pass. The upper end reaches the limit.
- If the limit is hit, `read.py` returns FAIL ("Slurm time limit"), even when every memory clause held. Chunk 1 is
  the variant most likely to fit in memory, so a round-3 relaunch would be needed.
- A longer limit costs nothing: the gpu_32gb partition limit is 7 days, and a job that finishes early releases its GPU.
- The memory answer does not depend on this: it appears at step 0 within minutes.
- Not shown: the actual step time. It is unmeasured.

**SHOULD 2: `read.py` prints PASS before the job has exited (the verdict does not match its own docstring, line 19).**
- Where: `read.py`:207-217. After `Epoch 8: Total train loss` has been written (`end_idx` set), the code checks the
  exit only when a non-zero `rnn exit` line exists.
- The failure: during the train checkpoint save and the dev pass there is no `rnn exit` line yet, so the verdict is
  PASS. The same happens after a time-limit kill or a scancel in that window, because `time_limit` and `cancelled`
  are tested only when `end_idx is None`.
- Reproduced on the implementer's PASS fixture:
  - with its `rnn exit 0` line removed, the verdict is PASS;
  - with a `DUE TO TIME LIMIT` line added in place of that line, the verdict is PASS.
- Why it matters: the k2 leg also runs in the dev pass at the same chunk size (`model/train_step.py`:153; the fixture
  shows `lexlat_k2` values in its dev step lines). An orchestrator who polls the live dir to beat the 19:30 deadline
  could apply a variant whose dev pass then OOMs or segfaults.
- Fix: PASS requires an `rnn exit 0` line. If that line is missing, or a time-limit or cancel line is present, the
  result is NOT_DECIDED or FAIL, whatever `end_idx` is.
- Only `read.py` changes, and the job calls it only at its end (`probe.sbatch`:64). It can therefore be fixed while
  the probes run. Until then, read a verdict only after `rnn exit 0` is in `slurm-*.out`.

## Items checked, no finding

1. **Configs.**
   - `diff` of each dir's `returnn.config` against B (`_cs8`) shows only the `model` line (169). The same holds for
     `config_delta.diff`.
   - The `dry_check.txt` files differ from B's only in the paths, `chunk_seqs` (2 and 1; the env route, with
     `stability_chunk_seqs` 1 unchanged), and the checkpoint's internal epoch/step (4/228 instead of 3/171).
   - The probe scripts other than `read.py` have mtimes of 16:22-16:28, before the round-1 launch at 16:44.
   - J's config sha256 is still `dff329b6...cdc`.
2. **Sub-epoch 8, seeding and isolation.** The dry check has 22 OK lines:
   - model epoch 7, start and final epoch 8, lr(8) 1e-4 (constant);
   - temperature 2.0, anchor 0, prior 1, lam 1/3 (active);
   - partition [3], strict state_dict load.
   Other details:
   - The model and opt sha256 equal J's `epoch.004.pt`/`.opt.pt` (`f649c354`/`21f39662`), identical in both dirs.
   - The lr file holds J's own entries 1-4 (entry 4 with its dev scores), placeholders for 5-7 (`train_`/`dev_loss_probe_placeholder`),
     and lr only for 8-20. It is identical in both dirs. Constant lr control means the placeholders only satisfy
     `_check_missing_eval`, as in round 1.
   - If a job starts after J writes `epoch.005`, `seed.py` picks 5. J writes the lr file's train score before the
     checkpoint (`engine.py`:641-655) and the dev score after it. The dev placeholder covers that gap.
   - Every write is inside PD (`seed.py`:127-128, 64-71), and `seed.py` refuses once J has epoch 8.
   - The new dirs hold no `slurm-*.out` and no ABORT marker, so round 1's stale-file caveat does not apply.
3. **`read.py` against G0.K2M.**
   - Peak = max(per-step max(`lexlat_k2_pre_peak_allocated_gib`, `mem_usage:cuda`), epoch-end alloc peak) > 28 means
     FAIL. `read.py` implements this exactly (:195-199). A fixture with epoch-end alloc 28.5 gives FAIL.
   - OOM, int32 and ABORT marker mean FAIL.
   - Stability: FAILED, a nan median, or 0 utterances means FAIL; a missing line means NOT_DECIDED.
     `RE_STAB_OK` matches the print in `lexlat_k2_train.py`:647.
   - The installed chunk size is compared with `dry_check.txt`. The installed line reaches `slurm-*.out` only from
     the RETURNN run, because the dry check's stdout is redirected (`probe.sbatch`:54).
   - Re-reads of round 1: B gives FAIL (rnn exit 139, 2 OOM lines, installed 8 = intended 8, stability 0.0194 over
     16 of 16). A gives NOT_DECIDED (cancelled, no step). Neither gives PASS.
   - The new dirs give NOT_DECIDED.
4. **Commands.**
   - The partition is gpu_32gb.
   - `-t`, `-J`, `--chdir` and `-o` override the `#SBATCH` lines.
   - `PROBE_DIR` and `PROBE_CHUNK_SEQS` are exported, since the default export is ALL (round 1), and reach RETURNN as
     `LEXLAT_K2_CHUNK_SEQS` (`probe.sbatch`:46-47).
   - The cwd guard (`probe.sbatch`:24) protects A's dir.
   - One gpu_32gb node was idle at 17:35, so both jobs should start together on the same seed. Compare the `seed:`
     lines in both `read.txt` files.

## Notes (no action needed for the launch)
- `mem_usage` and the epoch-end alloc peak are RETURNN's 1-decimal GB (`util/basic.py`:945, `prec=1`). A true peak
  in [28.0, 28.05) GiB therefore reads 28.0 and passes. `pre_peak` is exact.
- J's Slurm limit is 3 days, with 4.5 h used. Chunk 1 at about 2-3 h per sub-epoch over 13 k2 sub-epochs still fits,
  but the step-time cost against the banked 1.25x should be reported, as the gate asks.
- Not verified: the GPU run itself (memory, time) and the gate premise that an earlier theta is conservative.
