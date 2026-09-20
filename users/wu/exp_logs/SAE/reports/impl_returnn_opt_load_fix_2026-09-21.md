# Implementer report: RETURNN optimizer-checkpoint load blocks resume (2026-09-21)

## Bug
Resume of `PackedBlankfreeTrainJob.ks7CbtlvpcIL` arms `ctrl_100` / `odmprior_100` from `epoch.067`
died in `Engine._load_optimizer` -> `Updater.load_optimizer` ->
`torch.load(epoch.067.opt.pt)`: `UnpicklingError: Unsupported global: GLOBAL functools.partial was
not an allowed global by default`. Cause: torch >= 2.6 flipped the `torch.load` default to
`weights_only=True`; the optimizer checkpoint (written by this same job) contains `optimizer_opts`
with a `functools.partial` referencing `speech_llm`, so the restricted unpickler rejects it.

Traceback file paths (arm log lines 7186-7265 of
`.../ks7CbtlvpcIL/output/ctrl_100/log.run.1`) name the checkout actually imported:
`/e/project1/spell/wu24/2026-07-13_unsupervised/recipe/returnn` (the setup's `returnn` symlink
points there). That checkout already carried uncommitted local fixes in `returnn/torch/engine.py`
and `returnn/util/task_system.py`; neither was touched.

## Change (1 file, 1 call site, uncommitted)
`/e/project1/spell/wu24/2026-07-13_unsupervised/recipe/returnn/returnn/torch/updater.py:295`
(`Updater.load_optimizer`):

```
-        optimizer_state = torch.load(filename, map_location=self._device)
+        # This is our own optimizer checkpoint, so a full unpickle is intended. torch>=2.6 defaults
+        # to weights_only=True, which rejects e.g. functools.partial inside the optimizer state and
+        # breaks resume; older torch has no such kwarg.
+        try:
+            optimizer_state = torch.load(filename, map_location=self._device, weights_only=False)
+        except TypeError:
+            optimizer_state = torch.load(filename, map_location=self._device)
```

The checkout had **no** existing `weights_only` occurrence (`grep -rn weights_only` over
`returnn/` and `tools/` -> empty), so there was no local helper/convention to reuse; the guarded
`try/except TypeError` form named in the dispatch was used.

Scope check: the only other `torch.load` in the package is the model-checkpoint helper
`returnn/torch/engine.py:1793 _torch_load` (used at engine.py:984 model load and :1084 preload).
It is **not** an `.opt.pt` site and is left unchanged. It does not break here: the arm log shows
`Load model .../epoch.067.pt` succeeding at line 7119 before the optimizer failure, and a direct
`torch.load(epoch.067.pt, map_location="cpu")` with the torch default succeeds (keys
`effective_learning_rate, epoch, model, returnn_version, step` — all weights_only-safe). If a
future model checkpoint ever stores a non-tensor object, engine.py:1793 would fail the same way;
noted, not changed.

## Test (production env, production checkpoint)
Script: `<scratchpad>/test_opt_load.py`, run with the arm's own interpreter
`/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python` (from `work/ctrl_100/rnn.sh`) and the
arm config's `sys.path` (setup root, `recipe`, `recipe/2025-10-speech-llm/src`, `recipe/returnn`;
the full unpickle needs `speech_llm` importable, as it is in the real run). Target file:
`.../ks7CbtlvpcIL/output/ctrl_100/models/epoch.067.opt.pt`, `map_location="cpu"`.

```
torch version: 2.7.1
[1] unpatched behaviour, torch.load(map_location='cpu') with torch default:
    FAILED UnpicklingError: Weights only load failed. ...
[2] patched call site, torch.load(..., weights_only=False):
    SUCCESS, type dict, top-level keys:
    ['effective_learning_rate', 'epoch', 'optimizer', 'optimizer_class_name',
     'optimizer_opts', 'param_names', 'returnn_version', 'step']
    optimizer sub-keys: ['param_groups', 'state']  n_param_groups: 2  param_names: 17
[3] real patched code path, returnn.torch.updater.Updater.load_optimizer
    (module file: .../recipe/returnn/returnn/torch/updater.py), stub self with _device="cpu",
    matching param_groups/param_names:
    SUCCESS -> optimizer.load_state_dict received keys ['param_groups', 'state'],
    17 state entries.
```
`python -m py_compile returnn/torch/updater.py` passes; no line exceeds 120 chars.
Not tested: an end-to-end resumed training step (needs GPU + a job launch, out of scope).

## Effect on jobs in flight
All five jobs in `squeue -u wu24` have `WorkDir=/e/project1/spell/wu24/2026-07-13_unsupervised`, so
all import this same checkout via the setup's `returnn -> recipe/returnn` symlink:
- RUNNING, **unaffected until their own restart** (their python processes already imported the old
  `updater.py`; `run_arms` in pack_jobs.py launches all arm subprocesses at job start, so no new
  arm process appears inside a running pack job):
  1912407_1 (PackedBlankfreeTrainJob.YtvrSez8z9Wf), 1921103_1 (.5EIGJJ1MkcO9, prepro pack),
  1920366_1 (NeuralPhoneLmTrainJobV2.vkNGAOeLgNsy).
- PENDING, **will pick the fix up on their first start**: 1921334_[1] (budget pack node B,
  .reEI2Nd0S77A), 1921332_[1] (budget pack node C, .4QzmftNlbErt).
- Node A (.ks7CbtlvpcIL) has `error.run.1`; it picks the fix up only when that error is cleared and
  the manager resubmits — not done here.
The edit is a content change inside a path-referenced checkout (`returnn_root` is a `tk.Path`), so
it is sisyphus-hash-neutral; no job dir, marker or manager was touched.

## Notes / assumptions
- The dispatch's "do not edit anything under recipe/" was read as the recipe code trees
  (`recipe/i6_experiments`, `recipe/2025-10-speech-llm`); the RETURNN checkout lives at
  `recipe/returnn` and step 1-2 explicitly direct the fix there. Nothing else under `recipe/` was
  changed except this report file, whose path the dispatch fixed.
- Nothing committed anywhere; the returnn checkout keeps its pre-existing uncommitted diffs
  (`engine.py`, `task_system.py`) plus this one.
