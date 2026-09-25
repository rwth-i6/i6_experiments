# Review: expandable_segments allocator install into P0 k2lat J (2026-09-25)

Verdict: **PASS.** No blocking findings. The four inserted lines are the only change. They take effect before torch is first imported, change neither the job hash nor the numerics, and the relaunch procedure resumes J from epoch.007 on a gpu_32gb V100. Some notes for the executor and the phase record follow at the end.

J = `work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl`. Inputs reviewed:
- `reports/impl_p0_k2lat_alloc_expandable_2026-09-25.md`
- `analysis_out/k2lat_alloc_expandable_install.sh`
- the chunk-2 precedent: `reports/launch_p0_k2lat_cs2_{install,resume}_2026-09-25.md`

Nothing was run on J and nothing was installed or submitted. The only thing executed was a CPU-only config-load test on a scratch copy.

## (c) Diff and hash
- J's live config sha is 3a53ad6c...92a2 (matches OLD_SHA).
- I rebuilt the expected file independently: `head -1` + the 4 lines + `tail -n +2`. Its sha is **aaf07df4...73a8d**, identical to the pinned NEW_SHA.
- `diff` shows exactly `1a2,5`: a comment, `import os`, `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` and a print. Nothing else changes.
- The installer refuses anything whose sha is not NEW_SHA, so the reviewed bytes are the installed bytes.
- `os` was already a config key (original line 4, `import os`), so no config key is added.
- `config.has("PYTORCH_CUDA_ALLOC_CONF")` is False, so RETURNN's own override path (`__main__.py:351-365`, which runs only when that config key exists) is not triggered.
- Hash: the Sisyphus hash comes from the job's parameters, not from the output file. create_files has finished (`finished.create_files.1`). `ReturnnTrainingJob.run` (`i6_core/returnn/training.py:387-404`) only runs `rnn.py <J/output/returnn.config>`, the same path as `work/rnn.sh`. It never rewrites the file, so the resumed run reads the installed file.

## (a) The setting takes effect
- The first line is still `#!rnn.py`, so RETURNN still treats the file as a Python config.
- Order in `returnn/__main__.py`: `init` → `init_config` → `Config.load_file` (config executed) → `init_backend_engine` (:478).
- The config's own torch imports come after the new lines: the model `__import__` at original line 14, `train_step` (:75), `param_groups` (:79) and the `rt_chunked_backward` epilog (:241).
- An independent test used the sae python with the pinned RETURNN, CPU only, running `init_config` on the expected file:
  - torch was not imported after `import returnn.__main__`, nor after `init_better_exchook`;
  - the env var was already `expandable_segments:True` at the first `import torch`;
  - `torch.cuda.is_initialized()` was False after the load;
  - `train_step` resolves to `rt_chunked_backward`.
- Nothing else sets the variable:
  - `settings.py` (DEFAULT_ENVIRONMENT_SET has only PYTORCH_KERNEL_CACHE_PATH; CLEANUP_ENVIRONMENT=True);
  - the i6_core run (sets only OMP/MKL threads);
  - the package, i6_core and config/ (grep finds nothing);
  - J's last env dump (log.run.1:73-76) has no PYTORCH_CUDA_ALLOC_CONF.
  - An assignment to `os.environ` overrides any inherited value anyway.
- Dataset worker processes unpickle the typed_dict (`config.py:64-80`). They do not re-execute the file, so the print appears once.

## (b) torch 2.7.1, V100, k2
- torch is 2.7.1 (conda-forge, cu126). libc10_cuda.so contains the ExpandableSegment asserts and the cuMemCreate/cuMemAddressReserve driver-API wrappers, and lacks the "not supported on this platform" warning, so driver-API support is compiled in.
- In v2.7.1 upstream (`CUDACachingAllocator.cpp`), the allocator does not query the device capability. A cuMemCreate error other than OOM raises through `C10_CUDA_DRIVER_CHECK`. There is **no silent fallback** to cudaMalloc: if VMM were unavailable, the run would fail loudly at the first allocation.
- Volta with driver 580 supports VMM. Not verified on a GPU here; the relaunch is the test.
- k2 build ec31d2c9: `GetCudaContext` is `PytorchCudaContext` (`k2/csrc/pytorch_context.cu:187,226,242,287`). It allocates with `c10::cuda::CUDACachingAllocator::get()->raw_allocate/raw_deallocate`, and moderngpu goes through the same context. The raw cudaMalloc is only in `default_context.cu`, which is not the context in use. So **k2 uses torch's allocator**, and expandable segments cover k2's lattices and the HLG as well.
- This is consistent with the OOM line: 31.48 GiB in process use against 31.10 GiB reserved by PyTorch leaves about 0.38 GiB non-PyTorch, which is about the CUDA context.
- Memory APIs in the training path:
  - `torch.cuda.empty_cache()` only in the once-per-sub-epoch stability read (`model/lexlat_k2_train.py:655`), which is supported: it unmaps free pages;
  - `reset_peak_memory_stats`, `max_memory_*` and `mem_get_info` in `rt_chunked_backward.py:170-207`, which are unaffected;
  - `lexlat_k2.py:1180` and `lattice.py:1417` are benchmark paths only.
- No CUDA graphs, torch.compile, MemPool or CUDA IPC are used; the DataLoader and MPD workers are CPU-only.

## (d) Relaunch procedure
1. Creating `J/hold` puts the job in STATE_HOLD before any other check (`sisyphus/job.py:701`). The installer refuses to run without it.
2. The installer checks squeue by full job name, the old sha and the new sha, and does an atomic mv.
3. **Only `J/error.run.1` needs removing.** The task has `tries=1`, so there is no log rotation, and there are no other markers (no ABORT json in J/work). After removal the task is started with no live Slurm job and a stale log, so Sisyphus gives it STATE_INTERRUPTED_RESUMABLE:
   - `completed_fraction` = 7/20 = 0.35;
   - one submit-history entry is at ≥ 0.35, against MAX_SUBMIT_RETRIES = 3.
   This is the same path the 19:14 resubmit took.
4. Once the hold is removed, the manager resubmits the job:
   - Routing: `check_engine_limits` sends every ReturnnTrainingJob run task to gpu_32gb. The manager (pid 1646677, started 12:54) predates the 18:31 settings.py edit. Its in-memory version already routed this job to `-p gpu_32gb` at 19:14 (submit_log.run line 2), which rules out a routing change.
   - Resume point: `epoch.007.pt` and `epoch.007.opt.pt` both exist. There is no `epoch.008*` or `*.tmp_write` file (RETURNN writes checkpoints through `.tmp_write` + rename, `engine.py:1278-1298`).
   - The epoch-8 entry in `learning_rates` holds only its learning rate. Epoch 7 has train and dev scores (dev l_tau 1.9188) and `global_train_step_end` 399.
   - So `_check_missing_eval` runs **no** dev eval this time, and the log should show load epoch.007 → "Starting training at epoch 8, global train step 399".
   - The laplace order is seeded by epoch, so steps 0-17 replay the OOM run's batches.
5. The step-5 checks are correct. Details in the notes below.

## (e) Alternatives
Expandable segments is the right single change.
- **Explicit empty_cache:** in v2.7.1, `malloc` already runs `release_available_cached_blocks` and then `release_cached_blocks` (the same thing `empty_cache` does) before raising OOM. The 5.92 GiB that was reserved but unallocated was therefore stuck in segments that also hold live blocks. An explicit `empty_cache` after the k2 backward could not have freed it.
- **max_split_size_mb:** only limits how blocks are split, and needs a constant that traces to nothing.
- **chunk 1:** targets the k2 leg, but the failing request was in the rate term (`rate_term.py:297` → `lattice.py:1263`).
- Expandable segments instead let the allocator unmap the physical pages of free holes and remap them at the end of a segment.
- Demand at the failure: 25.18 + 1.31 = 26.49 GiB, below the 27.22 GiB peak allocated at step 1 and below the roughly 31.35 GiB usable. Keep chunk 1 + expandable segments as the registered fallback.

## Notes (non-blocking)
1. The `alloc_conf:` line is printed before RETURNN's log starts. It is therefore in `log.run.1` (stdout), **not** in `work/returnn.log`. A stronger check is `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the env dump right after "PyTorch: 2.7.1". Neither line proves the allocator mode. Because there is no silent fallback, however, a run that allocates at all is using expandable segments.
2. To test "numerics unchanged", compare the replayed steps 0-16 with the OOM run's log (`J/engine/...run.4365260.1`), which has step 0 at l_tau 1.902 and lexlat_k2 0.424. Small differences from k2 atomics are possible. A "Last epoch model not yet evaluated" line would point to a learning_rates problem.
3. From sub-epoch 8 on, `lexlat_k2_peak_reserved_gib` and `_pre_peak_reserved_gib` measure mapped pages, so they are not comparable with epochs 1-7 or with the other arms. G0.K2M reads allocated memory, not reserved, so no gate moves; disclose it anyway.
4. Unlike the chunk-2 precedent, `config/sae_i6_p0.py` does not carry these lines. If create_files were ever re-run, the setting would be dropped silently. This is already disclosed as J-only in SAE_i6_P0.md.
