# P0 k2lat: expandable_segments allocator installer for J (implementation, not run on J), 2026-09-25

J = work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl (V100 32 GB, gpu_32gb). The single delta is that PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set at the top of J/output/returnn.config. The edit is hash-neutral in the same way as the chunk-2 install: create_files has already finished, and only the file on disk changes.

## Script

`/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/k2lat_alloc_expandable_install.sh`

Location: the brief placed the chunk-2 installer under analysis/p0_k2lat_v100_probe/, but it is actually in analysis_out/ (`analysis_out/k2lat_perchunk_returnn_cs2_install.sh`). The new script sits next to it there.

The script does the following, in order:
1. It aborts unless `J/hold` exists. It also aborts if `squeue -u $USER` fails, or if squeue lists any job whose name contains `ReturnnTrainingJob.jcKXbLMDk4hl`.
2. It checks that the current sha256 is 3a53ad6c0ad127c66cbed724a37063e439ea7de287b523e24b92f2ca0a3792a2. If the file already has the new sha, it prints `already installed` and exits 0. It then backs up the config to `analysis_out/k2lat_alloc_expandable_returnn.config.orig`.
3. It inserts four lines after the `#!rnn.py` line (or after a coding line, if one exists) and writes the result to a temp file in J/output. It compiles the temp file with the sae python and checks that its sha is the pinned new sha. Only then does it `mv` the temp file over the config.
4. It re-checks the sha, prints the diff against the backup, prints `new sha256 ...` and prints `INSTALLED <timestamp>`.

The optional arguments `[JOB_DIR [BACKUP_DIR]]` exist only for the scratch test. With no arguments, the script targets J and analysis_out.

Current state of J (checked, not changed): there is no `hold` file, `error.run.1` exists, and no Slurm job for J is listed. The script therefore refuses until J/hold is created.

## Diff on the scratch copy and new sha

The real script was run on a copy of J's config in the scratchpad (`t1/ReturnnTrainingJob.jcKXbLMDk4hl`):
```
1a2,5
> # P0 k2lat: expandable CUDA allocator segments against fragmentation OOM; set before torch is imported (unhashed)
> import os
> os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
> print("alloc_conf:", os.environ["PYTORCH_CUDA_ALLOC_CONF"])
```
- New sha256: **aaf07df44339500181d159b273ac2423aabdc2625695b6c2aa599f0de4973a8d**. It is pinned in the script as NEW_SHA.
- The backup sha is 3a53ad6c..., which is the original.
- `python -c compile(...)` with the sae python returned `compile OK`.
- Refusal paths all gave rc 1 with the config unchanged:
  - no hold file;
  - wrong starting sha;
  - a job dir whose basename matches a RUNNING job (ReturnnTrainingJob.llSFybyKXkbL, Slurm 4359755).
- A rerun after the install printed `already installed` and returned rc 0.
- Live J/output/returnn.config still has sha 3a53ad6c... (untouched).

## RETURNN evidence (pinned checkout CloneGitRepositoryJob.KQ3NuCaDE6QH, commit 00171dfe2)

The checkout is dirty, but only in torch/engine.py, torch/updater.py and util/task_system.py. The files cited below are unmodified.

Call order in `returnn/__main__.py`:
1. `main` calls `init` (:743).
2. `init` calls `init_config` (:458), which calls `config.load_file` (:112). `returnn/config.py:82 load_file` executes the config with `custom_exec` (:125).
3. Only after that does `init_backend_engine()` run (:478). It contains RETURNN's own PYTORCH_CUDA_ALLOC_CONF handling (:365) and `BackendEngine.select_engine` (:367).
4. `init_engine()` runs last (:489).

On the torch side, the CUDA caching allocator reads the env var once, lazily: `torch/include/c10/cuda/CUDAAllocatorConfig.h:80-87` has `instance()` with a static `getenv("PYTORCH_CUDA_ALLOC_CONF")`. CUDA itself is initialized lazily through `torch/cuda/__init__.py:342 _lazy_init` → `:372 torch._C._cuda_init()`.

Empirical test (`scratchpad/load_test.py`, sae python, no GPU): the test ran RETURNN's own `init_config` on the installed copy, with an import hook that records the env var at the first `import torch`. Results:
- Before init_config, torch was not in sys.modules (importing `returnn.__main__` and `init_better_exchook` did not import torch).
- The config printed `alloc_conf: expandable_segments:True`.
- At the first `import torch`, the env var was already `expandable_segments:True`.
- After the load, `torch.cuda.is_initialized()` was False, and train_step and get_model were loaded.

The env var is therefore set before torch is even imported, so no `_set_allocator_settings` fallback is needed. In J's log, the value should appear twice: in the config print (before "RETURNN starting up") and in RETURNN's env dump (`diagnose_gpu.print_relevant_env_vars`, which prints PYTORCH_* vars after "PyTorch: 2.7.1").

## torch 2.7.1 support

- The sae env has torch 2.7.1 built with CUDA 12.6.
- `CUDAAllocatorConfig.h:25-33` returns false with the warning "expandable_segments not supported on this platform" only when the build lacks `PYTORCH_C10_DRIVER_API_SUPPORTED`. That warning string is absent from libc10_cuda.so, libtorch_cuda.so and libtorch_python.so. libc10_cuda.so does contain the ExpandableSegment code and the cuMemCreate/cuMemMap driver-API wrappers, so the build supports the feature on Linux.
- CUDA VMM has existed since CUDA 10.2 on Linux; NVIDIA mentions limitations only for Maxwell and older (https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/). V100 is Volta, and driver 580 is newer than CUDA 12.6 requires.
- Not verified on a V100: this node has no GPU and no job was submitted. If VMM were unavailable, the failure would be loud (a cuMemCreate/cuMemMap INTERNAL ASSERT at the first allocation), not silent.
- The DataLoader and MultiProcDataset workers are CPU-only. The CUDA-IPC restriction for expandable segments therefore does not apply.

## Not done / open

- The script was not run against J, and nothing was committed or submitted.
- RETURNN also supports a config key `PYTORCH_CUDA_ALLOC_CONF` (`__main__.py:352-365`), which it applies later but still before engine selection. The brief's top-of-config env var was used instead, because it takes effect earlier (before `import torch`).
