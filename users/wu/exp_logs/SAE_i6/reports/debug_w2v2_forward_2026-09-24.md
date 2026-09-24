# Debug: L15 wav2vec 2.0 forward jobs die with "DataLoader worker exited unexpectedly" (2026-09-24)

Status: DIAGNOSED.

## Verdict

The DataLoader worker does not crash on audio, memory, or /dev/shm. It fails while starting up,
before it reads any data, because it loads a different RETURNN than the main process.

- The main process runs the pinned RETURNN_ROOT:
  `work/i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn`, which is rwth-i6/returnn
  00171dfe2 (2026-05-18) plus the three local fixes.
- The generated `returnn.config` (line 43) runs
  `sys.path.insert(0, "/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/recipe")`.
- `recipe/` contains `recipe/returnn`, the manager's own RETURNN. It is a fresh clone of upstream
  master 5f752be49 (2026-09-22), taken 2026-09-24 12:56 according to its reflog.
- RETURNN starts DataLoader workers with `spawn_non_daemonic`. A spawned child copies the parent's
  `sys.path`, so `recipe` comes first and the child imports `returnn` from `recipe/returnn`.
- The parent pickles the global Config in the old format. Pinned `returnn/config.py:52-59` adds the
  Config itself to the pickler memo and sends `_self_memo_idx`.
- The child unpickles it with the new code, `recipe/returnn/returnn/config.py:93`. Upstream commit
  14db23e34 (2026-08-11, "Config: pickle self-reference via persistent_id") replaced memo seeding
  with `persistent_id`/`persistent_load`. The new code does not add the Config to the memo first, so
  memo references in the stream resolve to the wrong objects.
- Result: `importlib.import_module` receives a function as its module name, raises
  `AttributeError: 'function' object has no attribute 'startswith'`, and the worker exits with code 1.
- Owning layer: the setup or environment. Two RETURNN versions are importable, and the recipe copy
  shadows the pinned one in spawned subprocesses. Neither the dataset, the audio, the model, Slurm,
  nor the partition move is involved.
- What changed compared with the last run that worked: on JUPITER, one checkout served both the
  manager and the jobs (`returnn -> recipe/returnn`, 00171dfe plus the fixes; see
  `exp_logs/SAE/reports/impl_returnn_opt_load_fix_2026-09-21.md`). The i6 port added a separate
  CloneGitRepositoryJob root and filled `recipe/returnn` from upstream master.

## Evidence (observed)

1. Both jobs log the worker's own exception before the DataLoader RuntimeError:
   - `ReturnnForwardJobV2.5z8bUupJ1f8j/log.run.1:172-213`: the traceback runs through
     `recipe/returnn/returnn/util/multi_proc_non_daemonic_spawn.py:172` → `recipe/returnn/returnn/config.py:93`
     `Config.__setstate__` → `importlib/__init__.py:117` `name=<function importlib.import_module>` →
     `AttributeError: 'function' object has no attribute 'startswith'`. Line 213 reads
     `[PID 540635, PPID 540493] Error in NonDaemonicSpawnProcess._reconstruct_with_pre_init_func`.
   - Line 315: `DataLoader worker (pid 540635) exited unexpectedly with exit code 1`.
   - `QO3M1P9dOc2o/log.run.1:210-215, 317` shows the same thing (worker pid 540628).
   - The main-process frames (lines 326-337) are under `.../CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn/returnn/`.
     The two processes use two different RETURNN trees.
2. The unpickled state printed at log line 206 contains `'_self_memo_idx': 0`. Only the old
   `__getstate__` writes that key. The frame text at config.py:93 matches the new code in
   `recipe/returnn` exactly.
3. `git -C recipe/returnn log -S _SelfPersistentId` points to 14db23e34 (2026-08-11). The pinned commit
   00171dfe2 is 712 commits behind HEAD 5f752be49.
4. Desktop reproduction, CPU, sae python, writing only to scratch
   (`scratchpad/spawn/repro_spawn2.py`):
   - Setup: the pinned RETURNN root first on `sys.path`, the job's own `returnn.config` loaded as the
     global config, then one `NonDaemonicSpawnContext(process_pre_init_func=SubProcCopyGlobalConfigPreInitFunc())`
     child started.
   - As in the job: the child fails with the identical AttributeError at
     `recipe/returnn/returnn/config.py:93`, exit code 1.
   - Control (the only change is that `recipe` is moved to the end of `sys.path`): the child imports
     the pinned `returnn.config`, restores the config (`max_seqs=1`, `forward_callback` partial), and
     exits 0.
   - No GPU, Slurm, or cn-508 is involved, so the failure is deterministic.
5. Candidates ruled out (observed):
   - OOM: sacct 4334397/4334398 show COMPLETED 0:0, step MaxRSS about 1.78 GB against mem=64G, and
     `usage.run.1` has `out_of_memory: False`. The worker died from a Python exception with exit code
     1, not a signal.
   - Bus error or /dev/shm: there is no signal death, and the worker never reached data transfer.
   - Audio or input files: in-process iteration with the pinned RETURNN (`scratchpad/ds/ds_iter.py`,
     `use_cache_manager=False`) decoded seqs 0-5 of dev-clean (2703 seqs) and dev-other (2864 seqs).
     Shapes were plausible float32 with |x| < 0.71. Only 6 seqs per set were checked, not the whole
     set.
   - The partition move changed nothing at the failure point: ReqTRES is identical
     (cpu=4, mem=64G, gpu=1) and the failure reproduces without Slurm.

## Inferred, not verified

- Exact mechanism inside the unpickler: pickle protocol ≥4 MEMOIZE assigns `len(memo)`. The old
  pickler's memo starts with one entry (the Config) and the new unpickler's memo starts empty, so
  every `BINGET` returns the next object in the stream. The reproduction shows the crash; I did not
  disassemble the pickle stream.
- Pinning `recipe/returnn` to the same code should fix it. The control run shows that a child running
  the pinned code succeeds. I did not run a child that imports a pinned copy from the `recipe/returnn`
  path.

## Upstream

I searched for a RETURNN issue about pickle-format skew between two RETURNN versions across the
spawn boundary and found none. This is not an upstream bug: RETURNN assumes parent and child import
the same checkout. The relevant upstream change is commit 14db23e34 (2026-08-11).

## Blast radius

The five other L15 forward jobs pending on gpu_24gb (Slurm 4335682-4335686: train.shard0-3 and
seed_10h) have the same `returnn.config` line 43. They will fail the same way as soon as they start.
Any other RETURNN job in this setup that spawns subprocesses (DataLoader workers,
MultiProcDataset, etc.) and has `recipe` on its config `sys.path` is exposed too. Those children
would also miss the task_system `frombuffer` fix, because master still calls `numpy.fromstring`
according to default_tools.py.

## Fix

Preferred fix, no hash change: make `recipe/returnn` the same code as RETURNN_ROOT. Either check
out 00171dfe2 in `recipe/returnn` and apply
`recipe/i6_experiments/users/wu/experiments/unsupervised_asr/training/returnn_local_fixes.patch`, or
replace it with a symlink to the CloneGitRepositoryJob.KQ3NuCaDE6QH output. `recipe/returnn` is not
a hash input: the job's `returnn_root` is the CloneGitRepositoryJob with a hash_overwrite, and the
`sys.path` line is not hashed. So no job hash moves. The two errored jobs need only a cleared error
and a rerun; the existing `output/returnn.config` stays valid. Do this before the five pending
forward jobs start. The manager should keep working on 00171dfe, since the JUPITER manager ran on it
(inferred).

Alternatives, not preferred:
- `torch_dataloader_opts={"num_workers": 0}` in `post_config` in `w2v2/features.py:265`. This is
  unhashed, but it takes effect only if the job dirs are rebuilt, because create_files already wrote
  `returnn.config`. It also moves audio decoding into the GPU process and leaves other
  multiprocessing jobs exposed.
- Emitting `sys.path.append` instead of `insert(0, ...)` (`i6_experiments/common/setups/returnn_pytorch/serialization.py:137`).
  This is shared common code and changes the config text.
