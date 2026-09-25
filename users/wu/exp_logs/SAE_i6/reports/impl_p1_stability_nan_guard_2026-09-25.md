# impl: P1 rt arms, a non-finite stability read no longer stops the arm (2026-09-25)

Status: DONE. Nothing was committed and no cluster job was launched.
Brief: fix review F1 (`reports/review_p1_probe_memlog_2026-09-25.md`). A NaN `lexlat_k2_stability` stopped the arm
through RETURNN's `stop_on_nonfinite_train_score`.
In this report, P = `recipe/i6_experiments/users/wu/experiments/unsupervised_asr`.

## Summary

- **Where:** `P/reverse_model/rt_chunked_backward.py`, the per-chunk runtime's `step`. The ladder config is unchanged.
- **Change:** `lexlat_k2_stability` is emitted only when it is finite. A failed or empty read is omitted from
  the monitor dict.
  - RETURNN's non-finite check therefore never sees it.
  - That sub-epoch's `learning_rates` entry has every other `lexlat_k2_*` column but no `lexlat_k2_stability`.
    That absence is the missing read. It is never 0 and never a sentinel number.
  - The held read's print line (`... FAILED (...)`, or `median nan ... over 0 of n`) is untouched and stays in
    the job log.
- **Why this location:** the rt module builds the monitor dict itself, so the change is one `if` there.
  - The config/epilog route has nothing to hook at config level. RETURNN scans every accumulated loss, and it
    has no config option to exempt a key.
  - That route would therefore mean patching the Engine's train loop or wrapping `train_step`. Both are larger.

## Mechanism, verified

1. **The read.** `P/model/lexlat_k2_train.py:612-653`, `_stability`:
   - It sets `_stability_value = nan` before the read.
   - It keeps nan when no utterance has two finite totals (`if gaps:` at :645).
   - It catches any exception, prints `... FAILED ...` (:650-652) and returns nan.
   - The value is cached per sub-epoch (:612-614).
2. **The rt runtime.** It emitted `monitors["lexlat_k2_stability"] = float(stability)`, which was the pre-edit
   line 202 of `rt_chunked_backward.py`.
3. **The train step.** `P/model/train_step.py:209-211` marks every k2 monitor as
   `mark_as_loss(..., as_error=True, use_normalized_loss=False)`.
4. **RETURNN.**
   - `recipe/returnn/returnn/torch/engine.py:474-483` puts every entry of `train_ctx.losses` into `losses_dict`,
     `as_error` entries included.
   - `:527` accumulates it.
   - `:554-591`: `if self._stop_on_nonfinite_train_score: if any(isinf/isnan for v in accumulated_losses_dict)`
     leads to `debug_inf_nan`, then `raise Exception("Inf/nan score in step N.")`.
   - The flag defaults to True (`:141`), and the dumped r90 config sets it (line 140, as reported).
5. **Reproduced on CPU.** In a real RETURNN torch `Engine` run, the HELD runtime with a forced failure of the
   reference-rung read:
   - prints `Accumulated scores: {... 'lexlat_k2': 0.3045 ..., 'lexlat_k2_stability': nan ...}`;
   - then raises `Inf/nan score in step 0`;
   - even though every scored loss is finite.
   This is `test_nonfinite_stop_still_fires[held_stability_nan]`.

## Diff: `P/reverse_model/rt_chunked_backward.py` (+20 / -2; the file is untracked, so this is against the pre-edit text, 311 lines)

```diff
@@ -56,6 +56,19 @@
   leg, a POINT sample (not a peak) of the whole device, including the CUDA context and allocations
   outside the torch allocator.
 
+A NON-FINITE STABILITY READ IS OMITTED, NOT EMITTED (the one monitor whose key set differs from the
+held path's).  The held read catches its own failure and returns ``nan`` (``LexlatK2Runtime._stability``:
+"MAY NOT KILL THE ARM"), but the train step marks every monitor as an ``as_error`` loss and the rt
+config sets ``stop_on_nonfinite_train_score = True``; RETURNN's check (``returnn/torch/engine.py``,
+``if self._stop_on_nonfinite_train_score``) scans every accumulated loss, so a ``nan`` read would raise
+``Inf/nan score`` and end the arm.  Here ``lexlat_k2_stability`` is emitted only when it is finite; a
+failed or empty read leaves it out for that sub-epoch (the value is cached per sub-epoch, so it is
+absent from every step of it).  The read stays recoverable: the held read's print line (``stability at
+sub-epoch N: median nan ... over 0 of n`` or ``... FAILED (...)``) is in the job log, and that
+sub-epoch's ``learning_rates`` entry carries the other ``lexlat_k2_*`` columns but no
+``lexlat_k2_stability`` -- a missing read, never a number.  The non-finite stop still covers the total
+loss and every other loss and monitor.  No value, gradient or job hash moves.
+
 SELECTION.  Nothing in ``model/`` or ``training/`` imports this module.  The rt arms' RETURNN config
 rebinds ``train_step`` to :func:`train_step` here in its (unhashed) ``python_epilog``
 (:data:`EPILOG`, used by ``config/sae_i6_p1_ladder.py``), which swaps the model's runtime class on its
@@ -66,6 +79,7 @@
 
 from __future__ import annotations
 
+import math
 import time
 from typing import Dict, Optional, Tuple
 
@@ -199,8 +213,13 @@
                            ("lexlat_k2_pre_peak_reserved_gib", pre_reserved_gib),
                            ("lexlat_k2_device_used_gib", device_used_gib)):
             monitors[key] = torch.as_tensor(float(value), dtype=dtype, device=device)
-        monitors["lexlat_k2_stability"] = torch.as_tensor(
-            float(stability), dtype=dtype, device=device)
+        # the stability read is emitted only when it is a number: a failed or empty read (``nan``) is
+        # OMITTED, so RETURNN's non-finite stop (which scans every loss, ``as_error`` ones included)
+        # cannot end the arm on a diagnostic, and the sub-epoch's ``learning_rates`` entry has no
+        # ``lexlat_k2_stability`` column -- the Gate's missing read, never a number (module doc)
+        if math.isfinite(float(stability)):
+            monitors["lexlat_k2_stability"] = torch.as_tensor(
+                float(stability), dtype=dtype, device=device)
         # the term: the held expression, with z_hlg a constant here and z_h still on its graph (the
         # H leg's gradient reaches dense_in through it, as held); the HLG leg's gradient is the
         # accumulated leaf gradient, injected at the live dense tensor by a zero-valued addend
```

## Diff: `P/tests/test_rt_chunked_backward.py` (371 -> 524 lines; untracked)

- **Existing comparisons.** These now expect the rt path to OMIT a non-finite held stability read, and to emit a
  finite one equal to the held value. The fixture's rung equals the reference rung (10000), so the held read is
  always nan there. New helper `_held_keys`; the three places are:
  - `_compare`: the key order and the per-key loop;
  - `test_full_train_step_theta_and_phi`: the key list and the loop;
  - `test_chunked_backward_equals_held_cuda`: now also asserts the key list. NOT RUN (no GPU on this node).
- **New: a real RETURNN torch `Engine` on CPU.**
  - Setup: one sub-epoch, `stop_on_nonfinite_train_score=True`, Task12AX data to pace the steps (2 or more
    steps are asserted), `torch_dataloader_opts={"num_workers": 0}`. A forked loader worker segfaulted under
    pytest here; this option is test-harness only.
  - The train step runs the fixture leg on a fixed batch and marks the term and monitors exactly as
    `model/train_step.py` does. The reference rung is 20000, above the fixture's 10000, so the read really runs.
  - `test_nonfinite_stability_does_not_stop_training[failed|empty]`:
    - the reference-rung call is forced to raise `torch.cuda.OutOfMemoryError` (`failed`), or to return all-`-inf`
      totals (`empty`);
    - the sub-epoch completes;
    - the job output has `stability read at sub-epoch 1 FAILED (OutOfMemoryError` (or `median nan ... over 0 of 5`)
      and no `Inf/nan score`;
    - the epoch's error dict and the written `learning_rates` file have `train_loss_lexlat_k2_lam` and
      `..._term_mean` but no `lexlat_k2_stability`;
    - every logged value is finite.
  - `test_finite_stability_is_logged`: with a real read, `train_loss_lexlat_k2_stability` is present and equals the
    printed median. It is 0.0 on this fixture, where the two rungs agree.
  - `test_nonfinite_stop_still_fires[held_stability_nan|total_nan|other_monitor_nan]`: each raises
    `Inf/nan score in step 0`.
    - `held_stability_nan`: the F1 control.
    - `total_nan`: the scored term's value is nan (a constant NaN is added, so its gradient stays finite and
      RETURNN's debug re-run reaches its own raise).
    - `other_monitor_nan`: `lexlat_k2_term_mean` is set to nan.
  - The forced stability failure is active in all three, so the guard still fires with the omission in place.

The full test diff (212 lines) follows.

```diff
@@ -44,6 +44,15 @@
           "lexlat_k2_device_used_gib")
 #: the rt arms' ramp: on-set 1, ramp 3 -> lam 1/3, 2/3, 1 at sub-epochs 1, 2, 3
 ONSET, RAMP = 1, 3
+#: the stability read's monitor: the per-chunk path OMITS it when the held read is non-finite (a
+#: missing read), so RETURNN's non-finite stop cannot end the arm on it (rt_chunked_backward doc)
+STABILITY = "lexlat_k2_stability"
+
+
+def _held_keys(held_mon):
+    """The held path's monitor keys the per-chunk path emits: all of them, less a non-finite stability
+    read.  ``held_mon`` maps key -> float or tensor."""
+    return [k for k, v in held_mon.items() if not (k == STABILITY and not math.isfinite(float(v)))]
 
 
 @pytest.fixture(scope="module", autouse=True)
@@ -98,10 +107,10 @@
     assert mine["total"] == held["total"], (label, mine["total"], held["total"])
     assert set(MEMORY) <= set(mine["mon"]) and not set(MEMORY) & set(held["mon"]), label
     shared = [k for k in mine["mon"] if k not in MEMORY]
-    assert list(shared) == list(held["mon"]), label  # the column order RETURNN logs
+    assert list(shared) == _held_keys(held["mon"]), label  # the column order RETURNN logs
     mon_dev = 0.0
     for key, value in held["mon"].items():
-        if key in TIMING:
+        if key in TIMING or key not in _held_keys(held["mon"]):
             continue
         other = mine["mon"][key]
         both_nan = math.isnan(value) and math.isnan(other)
@@ -314,12 +323,13 @@
     held, mine = rec["held"], rec["mine"]
     assert "lexlat_k2" in held["losses"] and held["scale"]["lexlat_k2"] == pytest.approx(min(1.0, epoch / 3))
     assert mine["total"] == held["total"]
-    assert [k for k in mine["losses"] if k not in MEMORY] == list(held["losses"])
+    held_keys = _held_keys({k: v.sum() for k, v in held["losses"].items()})
+    assert [k for k in mine["losses"] if k not in MEMORY] == held_keys
     for key in MEMORY:  # the G1.M monitors: reported as errors, never part of the total
         assert mine["as_error"][key] and float(mine["losses"][key]) == 0.0, key
     skip = TIMING | {"blankfree_frames_per_sec"}
     for key, value in held["losses"].items():
-        if key in skip:
+        if key in skip or key not in held_keys:
             continue
         a, b = mine["losses"][key], value
         assert a.dtype == b.dtype and (torch.equal(a, b) or bool(torch.isnan(a).all() and torch.isnan(b).all())), (key, a, b)
@@ -341,6 +351,147 @@
 
 
 # ================================================================================================
+# RETURNN's non-finite stop (``stop_on_nonfinite_train_score = True``, the rt configs): a non-finite
+# stability read (a diagnostic) must not end the arm; a non-finite total loss or other monitor must
+# ================================================================================================
+#: a reference rung above the fixture's max_active 10000, so the stability read actually runs
+REF_RUNG = 20000
+#: the case the Engine's train step runs; module level, as RETURNN's own engine tests keep theirs
+_ENGINE_CASE = {}
+
+
+class _EngineLegModel(torch.nn.Module):
+    """RETURNN's ``get_model``: one parameter upstream of the fixture leg's ``log_q``."""
+
+    def __init__(self, **_kwargs):
+        super().__init__()
+        self.bias = torch.nn.Parameter(torch.zeros(O.N_PHONES, dtype=torch.float64))
+
+
+def _engine_train_step(*, model, extern_data, **_kwargs):
+    """The fixture leg on a fixed batch; the leg's term and monitors marked as the held train step
+    marks them (``model/train_step.py``: ``mark_as_loss(term, "lexlat_k2", scale=lam)`` and, per
+    monitor, ``mark_as_loss(torch.tensor([float(value)]), key, as_error=True,
+    use_normalized_loss=False)``).  ``extern_data`` only paces the steps."""
+    ctx = rf.get_run_ctx()
+    rt = _ENGINE_CASE["rt"]
+    epoch = int(ctx.epoch)
+    base, lens, retained, keep = _batch()
+    log_q = torch.log_softmax(base.to(torch.float64) + model.bias, dim=-1)
+    term, mon = rt.step(log_q, feat_lens=lens, retained=retained, keep=keep, epoch=epoch, temperature=1.0,
+                        cfg=bed_cfg(), global_step=int(ctx.step))
+    if _ENGINE_CASE.get("nan_total"):
+        # the scored term's VALUE is nan (so is the total loss); its gradient stays finite, so the
+        # parameters stay finite and RETURNN's debug re-run of the step reaches its own raise
+        term = term + torch.tensor(float("nan"), dtype=term.dtype)
+    if _ENGINE_CASE.get("nan_monitor"):
+        mon[_ENGINE_CASE["nan_monitor"]] = torch.tensor(float("nan"), dtype=torch.float64)
+    ctx.mark_as_loss(term, "lexlat_k2", scale=rt.lam(epoch), use_normalized_loss=False)
+    for key, value in mon.items():
+        ctx.mark_as_loss(torch.tensor([float(value)], device=log_q.device), key,
+                         as_error=True, use_normalized_loss=False)
+
+
+def _force_reference_rung(monkeypatch, mode):
+    """The stability read's REFERENCE-rung call (``max_active_states == REF_RUNG``; the training leg
+    runs at 10000) fails (``failed``: an OOM, caught by the held read) or scores no utterance
+    (``empty``: every total -inf).  Either way the held read returns ``nan``."""
+    orig = KT.K.chunked_tot_scores
+
+    def patched(*args, **kw):
+        if int(kw.get("max_active_states", -1)) != REF_RUNG:
+            return orig(*args, **kw)
+        if mode == "failed":
+            raise torch.cuda.OutOfMemoryError("forced: the reference rung does not fit")
+        tot, chunks, sec = orig(*args, **kw)
+        return torch.full_like(tot, -math.inf), chunks, sec
+
+    monkeypatch.setattr(KT.K, "chunked_tot_scores", patched)
+
+
+def _run_engine(tmp_path, rt, **case):
+    """One sub-epoch of RETURNN's torch Engine on CPU with the rt configs' non-finite stop.
+    :return: the sub-epoch's ``learning_rates`` error dict and the file's text"""
+    from returnn.config import Config, global_config_ctx
+    from returnn.datasets import init_dataset
+    from returnn.log import log
+    from returnn.torch.engine import Engine
+
+    if not getattr(log, "initialized", False):
+        log.initialize(verbosity=[3])
+    _ENGINE_CASE.clear()
+    _ENGINE_CASE.update(rt=rt, **case)
+    lr_file = str(tmp_path / "learning_rates")
+    config = Config(dict(
+        task="train", device="cpu", num_epochs=1, model=str(tmp_path / "epoch"),
+        learning_rate_file=lr_file, stop_on_nonfinite_train_score=True,
+        extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
+        get_model=_EngineLegModel, train_step=_engine_train_step, batch_size=500,
+        optimizer={"class": "adam"}, learning_rate=1e-3,
+        torch_dataloader_opts={"num_workers": 0}))  # a forked loader worker segfaults under pytest here
+    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 12, "name": "train"})
+    dataset.init_seq_order(epoch=1)
+    try:
+        with global_config_ctx(config):
+            engine = Engine(config=config)
+            engine.init_train_from_config(train_data=dataset)
+            engine.train()
+            error = dict(engine.learning_rate_control.epoch_data[1].error)
+            n_steps = engine.learning_rate_control.epoch_data[1].meta["epoch_num_train_steps"]
+    finally:
+        _ENGINE_CASE.clear()
+    assert n_steps >= 2, n_steps  # the cached read is emitted (or omitted) on more than one step
+    return error, open(lr_file).read()
+
+
+@pytest.mark.parametrize("mode", ["failed", "empty"])
+def test_nonfinite_stability_does_not_stop_training(fx, tmp_path, monkeypatch, capfd, mode):
+    """A failed / empty stability read: the sub-epoch completes, the read's print line says so, and
+    the learning-rate file has every other lexlat_k2 column but no lexlat_k2_stability (not 0)."""
+    _force_reference_rung(monkeypatch, mode)
+    rt = RC.install_chunked_backward(runtime(fx, chunk_seqs=2, onset=ONSET, ramp=RAMP,
+                                             stability_reference_max_active=REF_RUNG))
+    error, text = _run_engine(tmp_path, rt)
+    out = capfd.readouterr().out
+    want = ("stability read at sub-epoch 1 FAILED (OutOfMemoryError" if mode == "failed"
+            else "stability at sub-epoch 1: median nan nats per retained frame over 0 of 5")
+    assert want in out, out[-2000:]
+    assert "Inf/nan score" not in out
+    assert "train_loss_lexlat_k2_lam" in error and "train_loss_lexlat_k2_term_mean" in error
+    assert f"train_loss_{STABILITY}" not in error and STABILITY not in text, error
+    assert all(math.isfinite(v) for v in error.values()), error
+    print(f"[record] engine, {mode} stability read: sub-epoch 1 completed; learning_rates keys "
+          f"{sorted(k for k in error if 'lexlat_k2' in k)}")
+
+
+def test_finite_stability_is_logged(fx, tmp_path, capfd):
+    """A real read (reference rung above the arm's): the column is there, with the printed median."""
+    rt = RC.install_chunked_backward(runtime(fx, chunk_seqs=2, onset=ONSET, ramp=RAMP,
+                                             stability_reference_max_active=REF_RUNG))
+    error, text = _run_engine(tmp_path, rt)
+    out = capfd.readouterr().out
+    value = error[f"train_loss_{STABILITY}"]
+    assert math.isfinite(value) and value >= 0.0, value
+    assert f"stability at sub-epoch 1: median {value:.4f} nats per retained frame over 5 of 5" in out
+    assert f"train_loss_{STABILITY}" in text
+    print(f"[record] engine, real stability read: {STABILITY} = {value!r}")
+
+
+@pytest.mark.parametrize("case", ["held_stability_nan", "total_nan", "other_monitor_nan"])
+def test_nonfinite_stop_still_fires(fx, tmp_path, monkeypatch, case):
+    """The guard is unchanged for everything else: a NaN total loss or another NaN monitor still ends
+    the run; and the HELD runtime's NaN stability read does (the defect this module removes)."""
+    _force_reference_rung(monkeypatch, "failed")
+    kw = dict(chunk_seqs=2, onset=ONSET, ramp=RAMP, stability_reference_max_active=REF_RUNG)
+    held_rt = runtime(fx, **kw)
+    rt = held_rt if case == "held_stability_nan" else RC.install_chunked_backward(held_rt)
+    extra = {"total_nan": dict(nan_total=True),
+             "other_monitor_nan": dict(nan_monitor="lexlat_k2_term_mean")}.get(case, {})
+    with pytest.raises(Exception, match="Inf/nan score in step 0"):
+        _run_engine(tmp_path, rt, **extra)
+
+
+# ================================================================================================
 # GPU (the arms' device): the same comparison on CUDA
 # ================================================================================================
 @pytest.mark.gpu
@@ -354,11 +505,13 @@
     z_held, z_mine = _z_hlg(runtime(fx, chunk_seqs=chunk), base, lens, 2.0, device="cuda")
     assert torch.allclose(z_mine, z_held, atol=Z_TOL, rtol=0)
     assert mine["term"] == pytest.approx(held["term"], rel=1e-12, abs=1e-12)
+    # lexlat_k2_stability is the held read's NaN "no read" here (the fixture's max_active 10000 is
+    # not below the reference rung 10000): the per-chunk path omits it
+    assert [k for k in mine["mon"] if k not in MEMORY] == _held_keys(held["mon"])
     for key, value in held["mon"].items():
-        if key not in TIMING:
+        if key not in TIMING and key in _held_keys(held["mon"]):
             other = mine["mon"][key]
-            # equal, or both NaN (lexlat_k2_stability is the held read's NaN "no read" sentinel
-            # here: the fixture's max_active 10000 is not below the reference rung 10000)
+            # equal, or both NaN
             assert (math.isnan(value) and math.isnan(other)) or other == pytest.approx(
                 value, rel=1e-9, abs=1e-12), (key, other, value)
     for key in ("g_log_q", "g_bias"):
```

## Checks (CPU, this node; sae python; `PYTHONPATH=<setup>/recipe:<setup>/recipe/returnn:<setup>/sisyphus`)

- **`tests/test_rt_chunked_backward.py`:** 45 passed, 2 skipped. The skips are the two gpu cases. Before: 39
  passed, 2 skipped; the 6 new Engine cases account for the difference.
  - All 31 old `[record]` lines still show deviation 0.000e+00 (117 values).
  - The 3 Engine `[record]` lines are as described above.
- **`test_model_lexlat_k2` + `test_model_blankfree` + `test_reverse_ladder` + `test_training_config`:** 139 passed,
  2 skipped, 4 xfailed. This is unchanged from the last review.
- **Whole package suite, `-m "not gpu" tests`:** 591 passed, 11 skipped, 5 deselected, 12 xfailed.
  - It must run with `/work/asr4/hwu/conda/envs/sae/bin` on PATH.
  - Without it, `test_analysis_per.py::test_t2_6_posterior_dump_to_per_chain` fails: i6_core's `black` lookup
    returns None. This is environmental and unrelated; the test passes with PATH set.
- **Not run:** the GPU tests (no GPU here). The CUDA test's edit therefore has not been executed. If Launch A
  (the `-m gpu` parity job) runs after this change, it runs the edited test.

## Job ids (read-only `sis console -s`, sae python, setup dir; same method as the last review)

The ids were dumped before the edit and again after it, as sorted `_sis_path()` lists of `tk.sis_graph.jobs()`.

- **Probe** (`P1_LADDER_STAGE=probe`, `config/sae_i6_p1_ladder.py`): 87 jobs, before and after identical.
  Equal to the first column of `analysis_out/p1_ladder_jobs_probe_2026-09-25.tsv`. Contains
  `ReturnnTrainingJob.EexT85vdfx25`.
- **Arms** (`P1_LADDER_STAGE=arms`): 113 jobs, before and after identical. Equal to
  `analysis_out/p1_ladder_jobs_arms_2026-09-25.tsv`. Contains EexT85vdfx25 (r90), 9lD2HS2Mzgcl (r70) and
  5CL3pQyyOvQt (r80).
- **P0** (`config/sae_i6_p0.py`): 164 jobs, before and after identical. Equal to
  `analysis_out/g0g_p0_jobids_before_2026-09-25.txt`. `rt_chunked_backward` is not in `sys.modules` after the P0
  load.
- **Epilog.** The `EPILOG` text is not touched by the diff, so the rt arms' `returnn.config` is byte-unchanged.
  This is inferred from the diff; I did not re-dump the configs.

## Notes and open points

- **Reading the gate.** A sub-epoch whose `learning_rates` entry has `train_loss_lexlat_k2_lam` but no
  `train_loss_lexlat_k2_stability` is a missing read, and the job log carries the reason.
  - Any reader of the Gate's "median > 0.05 for 3 consecutive sub-epochs" column must treat a missing column as
    missing, not as 0.
  - I found no code reader of this key in `P/` (only the model module), `config/` or `analysis/`.
- **Held path.** It is unchanged. `model/` is frozen, and `LexlatK2Runtime.step` still emits the nan. Any arm that
  runs the held runtime under `stop_on_nonfinite_train_score=True` keeps F1. The rt arms and the probe load this
  module through the epilog.
- **Already-running jobs.** A job that imported the module before this edit keeps the old behaviour until its
  process restarts. The last review says the probe EexT85vdfx25 was running (manager pid 1646677); I did not check
  its state.
- **Commit.** The module and the test file are untracked, and nothing was committed. Per project rules, they are
  committed after the code review.
