# Impl: pin `recipe/returnn` to the jobs' RETURNN_ROOT (2026-09-24)

Status: DONE_WITH_CONCERNS. The swap is done. No job id moved. The DataLoader-worker repro now exits 0.
The concerns are listed at the end: the live manager still has the old RETURNN in memory, and the link
target is a job output.

## What changed (one filesystem change, no code or config edits)

- `recipe/returnn` was a directory: a clean upstream master clone at 5f752be49d42e46aa87829243597267df27a0859
  (2026-09-22, `git status` clean). It is now `recipe/returnn.upstream-master`, unchanged.
- `recipe/returnn` is now a symlink to
  `/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/work/i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn`
  (realpath `/work/asr4/hwu/setups/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn`).
- The swap order was: create the link under a temporary name, `mv returnn returnn.upstream-master`, then
  `mv -T` the link into place. The gap was microseconds.

## RETURNN_ROOT (taken from the job files, not inferred)

- `ReturnnForwardJobV2.5z8bUupJ1f8j/info`: `PARAMETER: returnn_root: .../CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn`.
  Its `log.run.1:36` reports `version 1.20260518.222345+git.00171dfe2.dirty`. All three BlissToOggZipJobs
  use the same returnn_root.
- The checkout is at HEAD 00171dfe252eaabc71f6c1fa5a5a910c11a14788 (2026-05-18, "update better_exchook").
  Its working-tree diff modifies exactly 3 files: torch/engine.py, torch/updater.py and util/task_system.py.
  The +/- lines are identical to
  `unsupervised_asr/training/returnn_local_fixes.patch`, and they match the job's `patches` parameter.

## Job-id comparison (before the swap)

The script is `scratchpad/pin/dump_ids_pin.py`. It loads the config through `sisyphus.loader`, which is
the manager's code path, and dumps the sorted `_sis_id()`s. I built three graphs:
- the live tree (upstream code);
- scratch mirror `mirror_pinned`, where recipe/ is symlinks and returnn points to the pinned root;
- scratch mirror `mirror_upstream`, the control, where returnn points to the old clone.

The mirror settings.py is a copy with IMPORT_PATHS rewritten. The live settings.py was not touched.

| config | live | mirror_pinned | mirror_upstream |
|---|---|---|---|
| sae_i6_p0.py | 139, sha1 578dd2d8... | 139, identical list | 139, identical list |
| sae_i6_p0_screen.py | 75, sha1 db40328d... | 75, identical list | 75, identical list |

- The full graph's hash matches the earlier impl_gpu_route dump (578dd2d8...). All 75 screen ids are in
  the full graph.
- The graph build imports RETURNN: 115 `returnn.*` modules with the pinned code, 119 with upstream. The
  4 upstream-only modules are frontend.amp, frontend.packed, frontend._packed_backend and
  frontend.static_traceable. The build does not need them.
- Written config text: I wrote the `returnn_config` of all 31 RETURNN jobs in the full graph under both
  mirrors. After normalising the mirror path, the texts are byte-identical (`scratchpad/pin/cfg_*`).
- After the swap, I rebuilt both configs on the live tree. The ids are the same (139 and 75), and
  `returnn` is loaded from `recipe/returnn` → pinned.

## Verification after the swap

1. Debugger repro (`scratchpad/spawn/repro_spawn2.py as_job`, the job's own returnn.config, recipe first
   on sys.path):
   - For 5z8bUupJ1f8j, the child exits with code 0. It prints
     `returnn.config from .../recipe/returnn/returnn/config.py`, whose realpath is the pinned root. It
     restores `max_seqs=1` and the `L15FeatureCallback` partial with `expected_num_seqs=2703`.
     Log: `spawn/out3_as_job_pinned.log`.
   - The same holds for QO3M1P9dOc2o (`expected_num_seqs=2864`). Log: `spawn/out3_as_job_pinned_QO3.log`.
   - Before the swap, the same script failed with exit 1 (`spawn/out2_as_job.log`).
2. In-repo consumers (`scratchpad/pin/consumers.py`, sys.path in the sis worker order):
   - `returnn` resolves to the pinned root.
   - These symbols import from pinned files: every `from returnn... import` target used in
     unsupervised_asr and i6_core (SimpleHDFWriter, pformat, Dim/Tensor/TensorDict/batch_dim/single_step_dim,
     select_backend_torch, ForwardCallbackIface, CachedFile, log, init_dataset, OggZipDataset,
     Config/set_global_config, returnn.frontend). That is 16 of 16.
   - 21 of 22 modules that import returnn load cleanly: all unsupervised_asr ones, plus i6_core lib/hdf,
     returnn/compile, returnn/dataset and serialization_v2.
   - `i6_experiments.common.setups.returnn.serialization` fails on `No module named 'returnn_common'`.
     returnn_common is absent from recipe/, so this happens with either RETURNN, and P0 does not use this
     module (the graph builds without it).
   - A SimpleHDFWriter write and read round trip works. `i6_core.lib.hdf.get_returnn_simple_hdf_writer(None)`
     loads the pinned hdf.py.
   - BlissToOggZipJob runs `returnn_root/tools/...`, which is already the pinned root, and all 3 are finished.
3. Package CPU suite (`pytest tests/`, the same command and PYTHONPATH as the 18:05 pre-swap run in
   `scratchpad/run_suites.sh`):
   - 503 passed, 14 skipped, 12 xfailed, rc 0 (`scratchpad/pt_pin/suite.log`).
   - Each test has the same outcome as the pre-swap live-tree run (`pt_live/suite.log`, also 503/14/12).
   - The silfix worktree's 526 includes 23 tests that are not in the live tree.
   - `returnn` resolved to the pinned root in this run.

I did not touch manager 1554133 or the held jobs 4335682-86. Both were unchanged afterwards (Ssl,
JobHeldUser).

## Concerns (not fixed; outside this dispatch)

1. Manager 1554133 imported upstream RETURNN at 17:01 and keeps it in memory.
   - Any lazy `returnn.*` import it makes from now on resolves through `recipe/returnn` to the pinned
     files, which mixes the two versions inside one process.
   - The graph build imports everything it needs up front, and config text is identical under both
     versions. Mixing therefore looks unlikely to matter, but I have not verified that.
   - A manager restart at a convenient point removes the question.
2. The link target is a job output. If someone cleans or removes CloneGitRepositoryJob.KQ3NuCaDE6QH,
   `recipe/returnn` dangles.
3. The two errored forwards still need their error cleared and a rerun. That is the orchestrator's
   step and was not done here.

## Rollback (exact)

    cd /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/recipe
    unlink returnn && mv returnn.upstream-master returnn
    git -C returnn log -1 --format=%H   # expect 5f752be49d42e46aa87829243597267df27a0859

Scratch evidence dir: `/var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/e99790e1-d0a8-40e7-8845-f2cb4e1545a6/scratchpad/` (`pin/`, `spawn/`, `pt_pin/`).
