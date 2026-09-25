# Implementer: w2vu2 GAN intermediate evaluation on CPU (2026-09-25)

**Status: DONE (round 3).** Round 3 fixes the reviewer's F1 and F2 (`reports/review_w2vu2_intermediate_eval_2026-09-25.md`). There are 180 updates per epoch, not 179, and the three epoch-end points move one save earlier. `tests/test_w2vu2_config.py` passes 13 of 13 (section 8). Sections 1-6 describe round 1. Their 179-based names, the valid N_train range and the checkpoint-name examples are superseded by section 8.

Brief: facts report `reports/facts_w2vu2_intermediate_eval_2026-09-25.md`. This is for SAE_i6 P0, gate G0.GAN.
Nothing was submitted, committed or restarted, and no job directory was touched. The GAN manager (pid 1786381)
was left alone. All graph loads used `sis console -s` (script mode, no manager) under the sae python.

## 1. What changed

One file changed: `P/config/w2vu2.py`, with 86 insertions and 19 deletions (full diff in section 6).
`P` = `recipe/i6_experiments/users/wu/experiments/unsupervised_asr`. No new module was needed.
`model/`, `training/` and `analysis/` are untouched. I only read `training.w2vu2_gan` for
`GAN_MAX_UPDATE` and `w2vu2_base_config`.

- **New constants.**
  - `INTERMEDIATE_EVAL_INTERVAL = 5000` (K, from the dispatch).
  - `GAN_TRAIN_UTTS = 28_539` and `GAN_BATCH_SIZE = 160`.
  - `GAN_UPDATES_PER_EPOCH = math.ceil(28539/160)`, which is 179. Its comment gives the valid N_train range
    [28,481, 28,640] and says that a wrong value stalls the eval rather than evaluating a wrong checkpoint.
  - `FORWARD_CPU_RQMT = FORWARD_RQMT` without `gpu_mem`, which gives time 2, mem 24 and cpu 4.
- **Graph-time asserts in `gan_1c`.** No job is added.
  - `EXPECTED_UTTS["train-clean-100"] == GAN_TRAIN_UTTS`.
  - `w2vu2_base_config()["dataset"]["batch_size"] == GAN_BATCH_SIZE`.
  - Per checkpoint: `U % 179 != 0`. fairseq names a save `checkpoint_E_U.pt` only when it is not at an epoch end.
- **Module-level `_checkpoint_file_exists(path)`** returns `os.path.isfile(path.get_path())` and is picklable.
  **`_intermediate_checkpoint(train, U)`** returns
  `tk.Path(f"checkpoints/checkpoint_{ceil(U/179)}_{U}.pt", creator=train, available=_checkpoint_file_exists)`.
- **Inner helper `dev_eval(fairseq_checkpoint, prefix)`.** This is the old per-seed block, verbatim, except
  that it passes `device="cpu", **FORWARD_CPU_RQMT`.
  - The `checkpoint_best` eval calls it with prefix `s{seed}`, so its aliases and outputs are unchanged.
  - The intermediate eval calls it with prefix `intermediate/s{seed}/u{U}` for U = 5000, 10000, ..., 150000.
- **Output paths.** `output/w2vu2/sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/intermediate/s{seed}/u{U}/{split}/per.{json,txt}`.
- **Aliases.**
  - Conversion: `.../gan_l15_sil0.5/intermediate/s{seed}/u{U}/generator`.
  - Forward: `sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/intermediate/s{seed}/u{U}/{split}/forward`.
  - PER: `.../intermediate/s{seed}/u{U}/per/{split}`.
  - All sit under the `w2vu2/` scope.
- **Return value.** `gan_1c` also returns `"intermediate": {seed: {U: {...}}}`.
- **Docstrings.** The Graph paragraph and the Resources paragraph now describe CPU dev forwards and the
  intermediate eval.

Assumption about naming: I used the existing per-seed layout (`<GAN_ARM>/.../s{seed}`). The dispatch's example
was `sae/1c/intermediate/seed{s}`.

**Pseudo-label forward on train-clean-100: kept on GPU.** It keeps `FORWARD_RQMT` and gpu_mem 40, routed to
gpu_48gb.
- My CPU estimate is about 15-20 min against its 2 h time_rqmt. This is not measured. It scales the facts
  report's own estimate (about 1 min of compute for both dev sets, about 10.7 h of audio) to 100 h,
  which gives about 10 min, and adds about 5 min to read about 31.6 GB at the measured 105 MB/s.
- An unmeasured estimate does not show that the job "clearly fits", and this forward produces labels, not a PER.

## 2. Hash identity (before vs after, same loader, `config/sae_i6_w2vu2.py`)

Graph size: 101 jobs before and 851 after. All 101 pre-change job ids are present after the change and
none is missing. No old job's aliases changed.

The table lists every job of the requested classes: the 6 trainings (5 GAN seeds and the CTC student), the
selection, the checkpoint_best conversions, forwards and PERs, the pseudo-label chain and the 1d jobs.

| class | alias (under w2vu2/ where scoped) | job id before | after |
|---|---|---|---|
| CtcPhoneDecodeJob | sae/1d/decode | 44O0I7VzpeFa | same |
| CtcWordDecodeJob | sae/1d/word_decode | 3qBkEbtXHIjA | same |
| FairseqAudioManifestJob | sae/1d/audio/dev-clean | YVplTwoHVRee | same |
| FairseqAudioManifestJob | sae/1d/audio/dev-other | klMuANMS97Da | same |
| FairseqAudioManifestJob | sae/1d/audio/train | W8JCwSUOv0ft | same |
| FairseqCtcDataJob | sae/1d/ctc_data | 0KhApVeH3BdE | same |
| FairseqHydraTrainingJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s0 | gY9AdQ2aYl3B | same |
| FairseqHydraTrainingJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s1 | qLeCqytt98iC | same |
| FairseqHydraTrainingJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s2 | fCcCEHaqxnyE | same |
| FairseqHydraTrainingJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s3 | K0RJEx1UUwcb | same |
| FairseqHydraTrainingJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s4 | RzVQARNBfUb2 | same |
| FairseqHydraTrainingJob | sae/1d/ctc_finetune | SWK00drxbXSC | same |
| FlashlightLexiconJob | sae/1d/flashlight_lexicon | Rv0Ya3z4O1nh | same |
| OggZipWordRefsJob | sae/1d/word_refs | EhCiozD1UYPE | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/s0/dev-clean/forward | 1yy9hBkxrpS5 | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/s0/dev-other/forward | d9mpykiqAEoT | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/s1/dev-clean/forward | 2yOQSKl7Wbjx | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/s1/dev-other/forward | RMWjdwXaLm80 | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/s2/dev-clean/forward | 9Awad7v1gumT | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/s2/dev-other/forward | WvoYsNRz3879 | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/s3/dev-clean/forward | TIWeqDOfVeLX | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/s3/dev-other/forward | 4skoR5bz7tz7 | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/s4/dev-clean/forward | UDMiItUSQSDy | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/s4/dev-other/forward | bGnjWf0fS5oY | same |
| ReturnnForwardJobV2 | sae/1c/gan/w2v2_lv60_l15/gan_l15_sil0.5/select/train/forward | zApR2FRnKz4N | same |
| ReturnnForwardJobV2 | sae/4a/feats/dev-clean/forward | 5z8bUupJ1f8j | same |
| ReturnnForwardJobV2 | sae/4a/feats/dev-other/forward | QO3M1P9dOc2o | same |
| ReturnnForwardJobV2 | sae/4a/feats/seed_10h/forward | AjTxcYsLi4E6 | same |
| ReturnnForwardJobV2 | sae/4a/feats/train.shard0/forward | NxXNQ8v87CCM | same |
| ReturnnForwardJobV2 | sae/4a/feats/train.shard1/forward | UzlPyMEw1eR4 | same |
| ReturnnForwardJobV2 | sae/4a/feats/train.shard2/forward | vVoEfyB8FVzF | same |
| ReturnnForwardJobV2 | sae/4a/feats/train.shard3/forward | khVKNe5qaTxE | same |
| W2vu2GanPerJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s0/per/dev-clean | UVsMUUgu31WE | same |
| W2vu2GanPerJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s0/per/dev-other | AUMRXcWcXERa | same |
| W2vu2GanPerJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s1/per/dev-clean | TgZItlLYillX | same |
| W2vu2GanPerJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s1/per/dev-other | h9XKPyml8rFP | same |
| W2vu2GanPerJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s2/per/dev-clean | tV1YXJC46iRx | same |
| W2vu2GanPerJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s2/per/dev-other | d8DnULfAnrmS | same |
| W2vu2GanPerJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s3/per/dev-clean | ZdQ9y9haDqG5 | same |
| W2vu2GanPerJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s3/per/dev-other | Lp27AWptL3L5 | same |
| W2vu2GanPerJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s4/per/dev-clean | HILUJWIofIWb | same |
| W2vu2GanPerJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s4/per/dev-other | QWJvRbh5trtz | same |
| W2vu2GanPseudoLabelJob | sae/1d/pseudo_labels | Abm4lL05dtNP | same |
| W2vu2GanSelectJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/select | 4Atr1P3Dotod | same |
| W2vu2GeneratorCheckpointJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s0/generator | RWXaJBwWdif9 | same |
| W2vu2GeneratorCheckpointJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s1/generator | T9XwTgrPGHew | same |
| W2vu2GeneratorCheckpointJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s2/generator | KOnVpY8P2jgf | same |
| W2vu2GeneratorCheckpointJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s3/generator | FOcgjhsbnJFB | same |
| W2vu2GeneratorCheckpointJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s4/generator | YO7ZJhuDgnQQ | same |
| W2vu2GeneratorCheckpointJob | sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/select/generator | Wc4gtXxOFpNs | same |

All 101 pre-change job ids present after the change: True; after = 851 jobs.

`device` is unhashed:
- It is absent from `config` and set in `post_config` (`"cpu"`) of the written RETURNN config.
- `sis_hash(create_returnn_config(device="cpu")) == sis_hash(create_returnn_config(device="gpu"))` is True.
- The existing checkpoint_best forward ids are the same before and after the change (for example `ReturnnForwardJobV2.d9mpykiqAEoT`, s0 dev-other).

## 3. New jobs

There are 750 new jobs. All their aliases contain `/intermediate/`, and there are 150 per seed.

| class | count |
|---|---|
| W2vu2GeneratorCheckpointJob | 150 |
| ReturnnForwardJobV2 | 300 |
| W2vu2GanPerJob | 300 |

There are 30 distinct checkpoint names per seed, from `checkpoint_28_5000.pt` and `checkpoint_56_10000.pt` to
`checkpoint_811_145000.pt` and `checkpoint_838_150000.pt`. There are 150 distinct checkpoint paths.

## 4. Path, pickle, routing

**Path hash.** The new Path's `sis_hash` equals that of
`train.out_checkpoint_dir.join_right("checkpoint_28_5000.pt")`, so `available` is not hashed.
`available()` is currently False because no training directory exists yet.

**Pickle round-trip** (seed-3 training `FairseqHydraTrainingJob.K0RJEx1UUwcb`, `checkpoints/checkpoint_28_5000.pt`):
- With the default `gs.INCLUDE_CREATOR_STATE=False` (the sisyphus default: creator dropped, absolute path kept):
  - The pickle is 378 bytes.
  - `get_path()` is equal after the round-trip.
  - `_available` unpickles to `...unsupervised_asr.config.w2vu2._checkpoint_file_exists`.
  - `available()` works and returns False.
- With `INCLUDE_CREATOR_STATE=True`:
  - The creator id and the relative path are equal after the round-trip.
  - `sis_hash` is equal.
  - `_available` is the same function object.

**Routing** (`settings.py check_engine_limits`):
- The new forward `ReturnnForwardJobV2.zV1XwTu8Gt7y` (s0, u5000, dev-other) has `run` rqmt
  `{cpu 4, mem 24, time 2, gpu 0}`, and it is the same after the check. No `sbatch_args` is set, so the job
  goes to Slurm's default partition. `sinfo` shows that default as `cpu_modern*`.
- The checkpoint_best forward `d9mpykiqAEoT` is identical.
- `create_files`, the conversion and the PER are mini tasks and run on the local 'short' engine.

## 5. Checks and results

- **Graph load before and after the change**: OK. Both loads used `sae/bin/python sisyphus/sis --log_level 30 console -s -c exec(...) config/sae_i6_w2vu2.py`, without a manager.
- **`pytest tests/test_w2vu2_config.py`** (sae env): 8 passed, 2 failed. Both failures follow from the requested change:
  - `test_job_types_and_counts`: `W2vu2GeneratorCheckpointJob` now has 156 jobs instead of 6. `ReturnnForwardJobV2` and `W2vu2GanPerJob` grow the same way.
  - `test_rqmt_is_production_except_training_gpu_mem`: it expects every generator forward to have `{gpu 1, gpu_mem 40}`. The 10 checkpoint_best dev forwards and the 300 new ones are now `{gpu 0}`.
  - The static and dynamic import-allowlist tests pass with the new `os` and `math` imports.
  - **Proposal**, for whoever may edit `tests/`: update `EXPECTED_COUNTS`, and split the forward rqmt assertion into dev forwards `{gpu 0, cpu 4, mem 24, time 2}` and the select/train forward `FORWARD_RQMT`.
- **The change taking effect**: not shown. It takes effect only when the manager is restarted with the same command. Loading the graph does not prove that the jobs run.

Things the spec left open, and caveats:
- **179 updates per epoch is not yet confirmed on disk.** Confirm it when seed 0's
  `output/checkpoints/checkpoint_6_1000.pt` appears. If it is wrong, all intermediate jobs wait forever and the
  fix changes only their hashes.
- **Pickled function reference.** `_checkpoint_file_exists` is pickled by reference into each new job's
  `job.save`. Renaming or moving it later breaks the unpickling of those files.
- **Other docstrings are out of scope.** The setup file `config/sae_i6_w2vu2.py` also describes the graph and
  does not mention the intermediate eval. The package README may too. I did not edit either.
- **Load on the local engine.** The 150 conversions and 300 PERs are mini tasks on the desktop's
  LocalEngine(cpus=4, mem=8). They run a few at a time as checkpoints appear.
- **Commit.** None was made. Per the project rules, the change is committed with the code-review report.

## 6. Diff

```diff
diff --git a/users/wu/experiments/unsupervised_asr/config/w2vu2.py b/users/wu/experiments/unsupervised_asr/config/w2vu2.py
index bc68b9fb7..829f88f0a 100644
--- a/users/wu/experiments/unsupervised_asr/config/w2vu2.py
+++ b/users/wu/experiments/unsupervised_asr/config/w2vu2.py
@@ -36,9 +36,16 @@ The graph (:func:`py`)
 4-gram KenLM (i6_core ``KenLMplzJob`` + ``CreateBinaryLMJob``), the fairseq feature data from the
 port's VAD-trimmed L15 stream (``data.w2vu2_features``), one i6_core ``FairseqHydraTrainingJob`` per
 seed and ``W2vu2GanSelectJob``.  Then per seed (production evaluated every seed) the generator is
-converted to a RETURNN checkpoint and decoded greedily by ``ReturnnForwardJobV2`` on dev-clean and
-dev-other, and scored by ``W2vu2GanPerJob`` (``analysis.w2vu2_gan_eval``).  The selected seed's
-generator decodes train-clean-100 into the pseudo-labels (``W2vu2GanPseudoLabelJob``).
+converted to a RETURNN checkpoint and decoded greedily by ``ReturnnForwardJobV2`` (on CPU) on
+dev-clean and dev-other, and scored by ``W2vu2GanPerJob`` (``analysis.w2vu2_gan_eval``).  The selected
+seed's generator decodes train-clean-100 into the pseudo-labels (``W2vu2GanPseudoLabelJob``).
+
+Intermediate evaluation (not in production): the same conversion, CPU forward and PER chain on each
+seed's update checkpoint ``checkpoint_<E>_<U>.pt`` every :data:`INTERMEDIATE_EVAL_INTERVAL` updates
+(:func:`_intermediate_checkpoint`).  Each point waits only for its own checkpoint file (a custom
+``available`` check, not the training's completion), so it runs while the training runs.  fairseq's
+in-training ``uer`` is no substitute: the GAN's ``valid`` split has no labels, so it logs 0.0.
+Outputs: ``<GAN_ARM>/intermediate/s<seed>/u<U>/<split>/per.{json,txt}``.
 
 1d (``training.w2vu2_ctc``, ``analysis.w2vu2_ctc_decode``): fairseq manifests of the ogg zips, the
 CTC data dir, the LV-60 fairseq checkpoint, the CTC fine-tune as i6_core ``FairseqHydraTrainingJob``,
@@ -76,8 +83,13 @@ unhashed exceptions:
   renames that constant, the trainings fall back silently to gpu_48gb (L40S): check that the first
   training's ``submit_log.run`` shows ``-p gpu_32gb``.
 
-The forwards and decodes keep production's ``gpu_mem`` 40 (:data:`FORWARD_RQMT`; the two CTC decodes'
-own default), which the i6 rule ``gpu_mem > 24`` sends to gpu_48gb (L40S 46 GB), as intended.  The
+The train-clean-100 pseudo-label forward and the two CTC decodes keep production's ``gpu_mem`` 40
+(:data:`FORWARD_RQMT`; the two CTC decodes' own default), which the i6 rule ``gpu_mem > 24`` sends to
+gpu_48gb (L40S 46 GB), as intended.  The dev-clean / dev-other generator forwards (per-seed
+``checkpoint_best.pt`` and intermediate) run on CPU (:data:`FORWARD_CPU_RQMT`: production's time, mem
+and cpu without a GPU; ``check_engine_limits`` leaves them on the default CPU partition), so every
+per-seed PER comes from the same device; ``device`` sits in the RETURNN ``post_config`` and is not
+hashed.  The
 GAN took 10.5 h and the CTC student 4.25 h on GH200 under an 11.5 h limit (i6's 72 h training floor
 also matches only ``ReturnnTrainingJob``); on V100 both will take longer and rely on
 ``FairseqHydraTrainingJob``'s resumable ``run`` task (fairseq resumes from ``checkpoint_last.pt``).
@@ -153,6 +165,8 @@ Run from the setup dir (sisyphus imports the module and calls :func:`py`)::
 from __future__ import annotations
 
 import contextlib
+import math
+import os
 from typing import Any, Dict, Iterator
 
 from sisyphus import gs, tk
@@ -166,7 +180,12 @@ __all__ = [
     "I6_TRAIN_GPU_MEM",
     "I6_TRAIN_PARTITION_SETTING",
     "FORWARD_RQMT",
+    "FORWARD_CPU_RQMT",
     "PHONE_DECODE_TIME",
+    "INTERMEDIATE_EVAL_INTERVAL",
+    "GAN_TRAIN_UTTS",
+    "GAN_BATCH_SIZE",
+    "GAN_UPDATES_PER_EPOCH",
     "gan_1c",
     "selftrain_1d",
     "py",
@@ -192,9 +211,22 @@ I6_TRAIN_GPU_MEM = 32
 I6_TRAIN_PARTITION_SETTING = "GPU_ROUTE_TRAIN"
 #: production's ``W2vu2PerEvalJob`` and ``GanPseudoLabelJob`` rqmt, for the RETURNN generator forwards
 FORWARD_RQMT = {"time_rqmt": 2, "mem_rqmt": 24, "cpu_rqmt": 4, "gpu_mem": 40}
+#: :data:`FORWARD_RQMT` without the GPU, for the dev-clean / dev-other generator forwards (``device="cpu"``)
+FORWARD_CPU_RQMT = {k: v for k, v in FORWARD_RQMT.items() if k != "gpu_mem"}
 #: production's dev-only ``Wav2Vec2CtcDecodeJob.qqKPLPBEt1K3`` time (``CtcPhoneDecodeJob`` defaults to
 #: 3 h, the train-including ``decode_all``'s)
 PHONE_DECODE_TIME = 2
+#: updates between two intermediate evaluations (K = 5000: 30 points per seed up to max_update 150000)
+INTERMEDIATE_EVAL_INTERVAL = 5000
+#: GAN training utterances (train-clean-100, ``data.librispeech.EXPECTED_UTTS``) and the yaml's
+#: ``dataset.batch_size``; both are asserted at graph time in :func:`gan_1c`
+GAN_TRAIN_UTTS = 28_539
+GAN_BATCH_SIZE = 160
+#: fairseq updates per GAN epoch, ceil(28,539 / 160) = 179; it fixes the epoch index E = ceil(U / 179) of
+#: ``checkpoint_<E>_<U>.pt``.  179 holds for any N_train in [28,481, 28,640] (178 * 160 + 1 .. 179 * 160).
+#: If it were wrong, the named files would never appear and the intermediate points would wait, not
+#: evaluate a wrong checkpoint.
+GAN_UPDATES_PER_EPOCH = math.ceil(GAN_TRAIN_UTTS / GAN_BATCH_SIZE)
 
 
 def _i6_train_rqmt(rqmt: Dict[str, Any]) -> Dict[str, Any]:
@@ -207,6 +239,26 @@ def _i6_train_rqmt(rqmt: Dict[str, Any]) -> Dict[str, Any]:
     return rqmt
 
 
+def _checkpoint_file_exists(path: tk.Path) -> bool:
+    """``available`` check of an intermediate checkpoint Path: the file exists.  fairseq writes
+    ``checkpoint_<E>_<U>.pt`` to ``.tmp`` and renames it, so the name never shows a partial file.
+    Module level: sisyphus pickles it with the Path (job.save); moving or renaming it breaks the
+    unpickling of existing job.save files."""
+    return os.path.isfile(path.get_path())
+
+
+def _intermediate_checkpoint(train, update: int) -> tk.Path:
+    """``checkpoints/checkpoint_<E>_<U>.pt`` of the fairseq GAN ``train`` at update ``update``
+    (E = ceil(U / :data:`GAN_UPDATES_PER_EPOCH`)), available as soon as the file exists.
+
+    The Path hash is (creator, path), as for ``out_checkpoint_dir.join_right(...)``; the ``available``
+    callable is not hashed.  fairseq names a save ``checkpoint_<E>_<U>.pt`` only when it is not at an
+    epoch end, hence the assert."""
+    assert update % GAN_UPDATES_PER_EPOCH != 0, (update, GAN_UPDATES_PER_EPOCH)
+    epoch = math.ceil(update / GAN_UPDATES_PER_EPOCH)
+    return tk.Path(f"checkpoints/checkpoint_{epoch}_{update}.pt", creator=train, available=_checkpoint_file_exists)
+
+
 @contextlib.contextmanager
 def _scoped() -> Iterator[None]:
     """Build jobs and register outputs under :data:`ALIAS_AND_OUTPUT_PREFIX`; restore afterwards."""
@@ -223,34 +275,48 @@ def gan_1c(inputs) -> Dict[str, Any]:
 
     :param inputs: ``inputs.get_inputs()``.
     :return: ``{"gan": W2vu2Gan, "eval": {seed: {"generator", split: {"forward", "per"}}},
+        "intermediate": {seed: {update: {"generator", split: {"forward", "per"}}}},
         "select": {"generator", "forward", "labels"}}``.
     """
     from ..analysis.w2vu2_gan_eval import W2vu2GanPerJob, W2vu2GanPseudoLabelJob, w2vu2_forward_job, \
         w2vu2_generator_checkpoint
     from ..data.librispeech import EXPECTED_UTTS
-    from ..training.w2vu2_gan import get_w2vu2_gan
+    from ..training.w2vu2_gan import GAN_MAX_UPDATE, get_w2vu2_gan, w2vu2_base_config
+
+    # the constants behind GAN_UPDATES_PER_EPOCH (intermediate checkpoint names)
+    assert EXPECTED_UTTS["train-clean-100"] == GAN_TRAIN_UTTS, (EXPECTED_UTTS["train-clean-100"], GAN_TRAIN_UTTS)
+    assert w2vu2_base_config()["dataset"]["batch_size"] == GAN_BATCH_SIZE, GAN_BATCH_SIZE
 
     gan = get_w2vu2_gan(seeds=PRODUCTION_SEEDS)
     text_dict = gan.text_data.out_dict
     arm = GAN_ARM.split("/", 2)[2]  # w2vu2_forward_job prefixes its alias with sae/1c/gan
 
-    evals: Dict[int, Dict[str, Any]] = {}
-    for seed, train in gan.trainings.items():
-        train.rqmt.update(_i6_train_rqmt(train.rqmt))  # unhashed (module docstring, Resources)
+    def dev_eval(fairseq_checkpoint: tk.Path, prefix: str) -> Dict[str, Any]:
+        """conversion, CPU forward and PER on :data:`EVAL_SPLITS`; aliases and outputs under ``prefix``
+        (relative to :data:`GAN_ARM`)"""
         conv, ckpt = w2vu2_generator_checkpoint(
-            fairseq_checkpoint=train.out_checkpoint_dir.join_right("checkpoint_best.pt"),
-            text_dict=text_dict, alias=f"{GAN_ARM}/s{seed}/generator")
-        evals[seed] = {"generator": conv}
+            fairseq_checkpoint=fairseq_checkpoint, text_dict=text_dict, alias=f"{GAN_ARM}/{prefix}/generator")
+        res = {"generator": conv}
         for split in EVAL_SPLITS:
             fwd = w2vu2_forward_job(
-                name=f"{arm}/s{seed}/{split}", checkpoint=ckpt, vocab=conv.out_vocab,
+                name=f"{arm}/{prefix}/{split}", checkpoint=ckpt, vocab=conv.out_vocab,
                 feature_hdfs=list(inputs.vad.out_feature_hdfs[split]), expected_num_seqs=EXPECTED_UTTS[split],
-                **FORWARD_RQMT)
+                device="cpu", **FORWARD_CPU_RQMT)
             per = W2vu2GanPerJob(hyps=fwd.out_files["hyps.json"], gold=inputs.gold, split=split)
-            per.add_alias(f"{GAN_ARM}/s{seed}/per/{split}")
-            tk.register_output(f"{GAN_ARM}/s{seed}/{split}/per.json", per.out_per)
-            tk.register_output(f"{GAN_ARM}/s{seed}/{split}/per.txt", per.out_report)
-            evals[seed][split] = {"forward": fwd, "per": per}
+            per.add_alias(f"{GAN_ARM}/{prefix}/per/{split}")
+            tk.register_output(f"{GAN_ARM}/{prefix}/{split}/per.json", per.out_per)
+            tk.register_output(f"{GAN_ARM}/{prefix}/{split}/per.txt", per.out_report)
+            res[split] = {"forward": fwd, "per": per}
+        return res
+
+    evals: Dict[int, Dict[str, Any]] = {}
+    intermediate: Dict[int, Dict[int, Dict[str, Any]]] = {}
+    for seed, train in gan.trainings.items():
+        train.rqmt.update(_i6_train_rqmt(train.rqmt))  # unhashed (module docstring, Resources)
+        evals[seed] = dev_eval(train.out_checkpoint_dir.join_right("checkpoint_best.pt"), f"s{seed}")
+        intermediate[seed] = {
+            update: dev_eval(_intermediate_checkpoint(train, update), f"intermediate/s{seed}/u{update}")
+            for update in range(INTERMEDIATE_EVAL_INTERVAL, GAN_MAX_UPDATE + 1, INTERMEDIATE_EVAL_INTERVAL)}
     tk.register_output(f"{GAN_ARM}/select/selection.json", gan.selection.out_selection)
 
     # the selected seed's generator labels train-clean-100 (production: GanPseudoLabelJob on s0)
@@ -263,7 +329,8 @@ def gan_1c(inputs) -> Dict[str, Any]:
     labels = W2vu2GanPseudoLabelJob(hyps=[fwd.out_files["hyps.json"]], expected_num_seqs=n_train)
     labels.add_alias(f"{SELFTRAIN_PREFIX}/pseudo_labels")
     tk.register_output(f"{SELFTRAIN_PREFIX}/pseudo_labels.json", labels.out_labels)
-    return {"gan": gan, "eval": evals, "select": {"generator": conv, "forward": fwd, "labels": labels}}
+    return {"gan": gan, "eval": evals, "intermediate": intermediate,
+            "select": {"generator": conv, "forward": fwd, "labels": labels}}
 
 
 def selftrain_1d(inputs, pseudo_labels: tk.Path) -> Dict[str, Any]:
```

Scratch artefacts (not project files): /var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/4186749c-0772-4a1a-a52d-b33a7b8fd5e1/scratchpad/{before,after,extra}.json, hash_dump.py, extra_checks.py, pickle2.py, pytest_after.log

## 7. Round 2 (coordinator follow-up): tests and setup docstring

**Files.**

- `P/tests/test_w2vu2_config.py`, +28/-5:
  - `EXPECTED_COUNTS` becomes conversions 6+150, forwards 11+300 and PERs 10+300.
  - New constant `DEV_FORWARD_RQMT = {gpu 0, mem 24, time 2, cpu 4}`.
  - `test_rqmt_is_production_except_training_gpu_mem` now asserts:
    - 311 forwards in total, of which 310 are dev forwards;
    - every dev forward has exactly `DEV_FORWARD_RQMT`;
    - the select/train forward has exactly `FORWARD_RQMT` (gpu 1, gpu_mem 40).
  - New `test_dev_forwards_of_best_and_intermediate_chains_run_on_cpu`:
    - seeds 0-4;
    - updates `range(5000, 150001, 5000)` per seed;
    - 310 forwards, each with `device == "cpu"` and `rqmt["gpu"] == 0`.
  - New `test_pseudo_label_forward_stays_on_gpu`: device "gpu", gpu 1, gpu_mem 40.
  - No other assertion was changed.
- `config/sae_i6_w2vu2.py` (setup-local; not in any repository, so it is recorded here):
  - Docstring only. It now says that the checkpoint_best forwards run on CPU and that the pseudo-label forward runs on GPU.
  - One new bullet covers the intermediate eval: per seed, conversion, CPU forward and PER on dev-clean and dev-other every 5000 updates (5000 to 150000, 30 points per seed), each starting when its checkpoint file exists.
  - Check: comparing the AST without the module docstring, the code is identical before and after (True).

**Pytest** (sae env: `python -m pytest -v tests/test_w2vu2_config.py` from `recipe/`, with `PYTHONPATH` set to recipe and sisyphus): **12 passed in 8.41 s**, 0 failed.

- test_job_types_and_counts PASSED
- test_production_seeds PASSED
- test_every_seed_is_evaluated_on_both_dev_splits PASSED
- test_pseudo_labels_come_from_the_selected_generator PASSED
- test_dev_forwards_of_best_and_intermediate_chains_run_on_cpu PASSED
- test_pseudo_label_forward_stays_on_gpu PASSED
- test_selftrain_wiring PASSED
- test_rqmt_is_production_except_training_gpu_mem PASSED
- test_i6_trainings_carry_the_settings_partition_and_nothing_else_does PASSED
- test_everything_is_scoped_under_w2vu2 PASSED
- test_config_imports_only_allowed_modules PASSED
- test_graph_build_imports_only_the_allowlist PASSED

Nothing was committed. Uncommitted in `recipe/i6_experiments`: `config/w2vu2.py` and `tests/test_w2vu2_config.py`.

### Test diff

```diff
diff --git a/users/wu/experiments/unsupervised_asr/tests/test_w2vu2_config.py b/users/wu/experiments/unsupervised_asr/tests/test_w2vu2_config.py
index d1081527e..eefb08497 100644
--- a/users/wu/experiments/unsupervised_asr/tests/test_w2vu2_config.py
+++ b/users/wu/experiments/unsupervised_asr/tests/test_w2vu2_config.py
@@ -31,10 +31,11 @@ EXPECTED_COUNTS = {
     # 5 GAN seeds + the 1d CTC student
     "FairseqHydraTrainingJob": 6,
     "W2vu2GanSelectJob": 1,
-    # report B: 5 seeds + the selected generator; 5 seeds x 2 dev splits + the train forward
-    "W2vu2GeneratorCheckpointJob": 6,
-    "ReturnnForwardJobV2": 11,
-    "W2vu2GanPerJob": 10,
+    # report B: 5 seeds + the selected generator; 5 seeds x 2 dev splits + the train forward;
+    # plus the intermediate eval: 5 seeds x 30 updates (5000 .. 150000), each on 2 dev splits
+    "W2vu2GeneratorCheckpointJob": 6 + 150,
+    "ReturnnForwardJobV2": 11 + 300,
+    "W2vu2GanPerJob": 10 + 300,
     "W2vu2GanPseudoLabelJob": 1,
     # report C
     "FairseqAudioManifestJob": 3,
@@ -49,6 +50,8 @@ EXPECTED_COUNTS = {
 GAN_RQMT = {"gpu": 1, "gpu_mem": 32, "mem": 100, "time": 11.5, "cpu": 8}
 CTC_RQMT = {"gpu": 4, "gpu_mem": 32, "mem": 60, "time": 11.5, "cpu": 16}
 FORWARD_RQMT = {"gpu": 1, "gpu_mem": 40, "mem": 24, "time": 2, "cpu": 4}
+#: the dev-clean / dev-other generator forwards (checkpoint_best and intermediate): FORWARD_RQMT on CPU
+DEV_FORWARD_RQMT = {"gpu": 0, "mem": 24, "time": 2, "cpu": 4}
 PHONE_DECODE_RQMT = {"gpu": 1, "gpu_mem": 40, "mem": 24, "time": 2, "cpu": 4}
 WORD_DECODE_RQMT = {"gpu": 1, "gpu_mem": 40, "mem": 64, "time": 11.5, "cpu": 8}
 
@@ -167,6 +170,26 @@ def test_pseudo_labels_come_from_the_selected_generator(built):
     assert labels.hyps == [sel["forward"].out_files["hyps.json"]] and labels.expected_num_seqs == 28539
 
 
+def test_dev_forwards_of_best_and_intermediate_chains_run_on_cpu(built):
+    """Every dev forward, of ``checkpoint_best.pt`` and of every intermediate checkpoint, uses device cpu."""
+    res, _, _ = built
+    evals, inter = res["1c"]["eval"], res["1c"]["intermediate"]
+    assert sorted(inter) == list(w2vu2.PRODUCTION_SEEDS)
+    forwards = [evals[s][split]["forward"] for s in evals for split in w2vu2.EVAL_SPLITS]
+    for seed, points in inter.items():
+        assert sorted(points) == list(range(5000, 150_001, 5000)), (seed, sorted(points))
+        forwards += [points[u][split]["forward"] for u in points for split in w2vu2.EVAL_SPLITS]
+    assert len(forwards) == 10 + 300
+    for fwd in forwards:
+        assert fwd.device == "cpu" and fwd.rqmt["gpu"] == 0, (fwd.device, fwd.rqmt)
+
+
+def test_pseudo_label_forward_stays_on_gpu(built):
+    res, _, _ = built
+    fwd = res["1c"]["select"]["forward"]
+    assert fwd.device == "gpu" and fwd.rqmt["gpu"] == 1 and fwd.rqmt["gpu_mem"] == 40, (fwd.device, fwd.rqmt)
+
+
 def test_selftrain_wiring(built):
     res, _, out = built
     from i6_experiments.users.wu.experiments.unsupervised_asr.lm.word_lm import official_4gram_arpa, official_lexicon
@@ -209,7 +232,11 @@ def test_rqmt_is_production_except_training_gpu_mem(built):
         assert job.rqmt == GAN_RQMT, job.rqmt
     assert res["1d"]["train"].rqmt == CTC_RQMT, res["1d"]["train"].rqmt
     forwards = _of(new, "ReturnnForwardJobV2")
-    assert forwards and all(j.rqmt == FORWARD_RQMT for j in forwards), [j.rqmt for j in forwards]
+    select_forward = res["1c"]["select"]["forward"]
+    dev_forwards = [j for j in forwards if j is not select_forward]
+    assert len(forwards) == 311 and len(dev_forwards) == 310
+    assert all(j.rqmt == DEV_FORWARD_RQMT for j in dev_forwards), [j.rqmt for j in dev_forwards]
+    assert select_forward.rqmt == FORWARD_RQMT, select_forward.rqmt
     assert res["1d"]["phone"].rqmt == PHONE_DECODE_RQMT
     assert res["1d"]["word"].rqmt == WORD_DECODE_RQMT
 
```

### Setup docstring diff (`config/sae_i6_w2vu2.py`)

```diff
--- /var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/4186749c-0772-4a1a-a52d-b33a7b8fd5e1/scratchpad/sae_i6_w2vu2.py.orig	2026-09-25 21:17:48.507938045 +0200
+++ config/sae_i6_w2vu2.py	2026-09-25 21:18:32.491672438 +0200
@@ -8,9 +8,12 @@
 
 * the shared port inputs (``inputs.get_inputs()``, unscoped, keeping their ``sae/4a/...`` aliases);
 * 1c: the p_sil 0.5 phone text and phone 4-gram, the fairseq feature data, five GAN seeds
-  (``FairseqHydraTrainingJob``), the ``weighted_lm_ppl`` selection, per seed the generator conversion,
-  RETURNN greedy forwards and PER on dev-clean / dev-other, the selected seed's train-clean-100
-  pseudo-labels;
+  (``FairseqHydraTrainingJob``), the ``weighted_lm_ppl`` selection, per seed the generator conversion of
+  ``checkpoint_best.pt``, RETURNN greedy forwards on CPU and PER on dev-clean / dev-other, the selected
+  seed's train-clean-100 pseudo-labels (forward on GPU);
+* 1c intermediate eval (not in production): per seed, the same conversion, CPU forward and PER chain on
+  dev-clean / dev-other for the update checkpoint every 5000 updates (5000 .. 150000, 30 points per seed),
+  each starting as soon as its checkpoint file exists (under ``.../gan_l15_sil0.5/intermediate/s<seed>/u<U>/``);
 * 1d: the fairseq CTC student (``FairseqHydraTrainingJob``, 4 GPUs) on those pseudo-labels, its viterbi
   phone PER and its lexicon + word 4-gram KenLM WER on dev-clean / dev-other.
 
```

## 8. Round 3 (review F1/F2): 180 updates per epoch, epoch-end points moved

**Files** (still uncommitted):

- **`P/config/w2vu2.py`**. Cumulative diff against HEAD: +122/-19.
  - `GAN_BATCH_SIZE_MULTIPLE = 8` (fairseq 0.12.2 `DatasetConfig.required_batch_size_multiple`, default 8, `configs.py:482`).
  - `GAN_UPDATES_PER_EPOCH = N//160 + (N%160 >= 8) + (N%160%8 > 0)`. For N = 28,539 this is 178 + 1 + 1 = 180, and a module-level `assert == 180` checks it.
  - The comment gives the new validity. 180 holds for N_train 28,537 to 28,543 around 28,539. Overall it holds for 160 values in [28,489, 28,800], but that set is not contiguous: for example, 28,536 and 28,544 give 179. I computed this with the dispatch's formula, not with fairseq itself.
  - New graph-time assert in `gan_1c`: the yaml sets no `required_batch_size_multiple`, or sets it to 8.
  - `INTERMEDIATE_EPOCH_END_SHIFT = 1000` (one `save_interval_updates`).
  - `intermediate_eval_updates(max_update)` gives every 5000 updates, with any epoch end moved 1000 earlier. For 150000 this replaces 45000, 90000 and 135000 with 44000, 89000 and 134000, leaving 30 points per seed. It asserts that the points are unique.
  - `gan_update_checkpoint_name(U)` returns `checkpoint_{ceil(U/180)}_{U}.pt` and keeps the assert `U % 180 != 0`. `_intermediate_checkpoint` uses it.
  - `gan_1c` iterates over `intermediate_eval_updates(GAN_MAX_UPDATE)`.
  - Docstring updated.
- **`P/tests/test_w2vu2_config.py`**. Cumulative diff: +51/-5.
  - New constant `INTERMEDIATE_UPDATES`: the 5000-grid with 45000/90000/135000 replaced by 44000/89000/134000.
  - The CPU-forward test now expects those updates per seed. It also checks that each point's checkpoint Path has the training as creator and the path `checkpoints/<gan_update_checkpoint_name(U)>`.
  - New `test_intermediate_checkpoint_names_follow_fairseq`, which checks:
    - there are 180 updates per epoch;
    - U = 148000 gives `checkpoint_823_148000.pt` (production s0) and U = 7000 gives `checkpoint_39_7000.pt`;
    - `intermediate_eval_updates(150000)` equals `INTERMEDIATE_UPDATES`;
    - no point is a multiple of 180;
    - `gan_update_checkpoint_name(45000)` raises.

**Checks.**

- **pytest**, sae env, `tests/test_w2vu2_config.py`: **13 passed in 9.67 s**, 0 failed.
- **Real graph**: `config/sae_i6_w2vu2.py` loaded with `sis console -s`, no manager.
  - **Job ids**: 101 before and 851 after. All 101 pre-change job ids are present, none is missing, and 0 aliases changed.
  - **New jobs**: 150 `W2vu2GeneratorCheckpointJob`, 300 `ReturnnForwardJobV2` and 300 `W2vu2GanPerJob`.
  - **Update points**: 30 distinct, `5000 ... 40000, 44000, 50000 ... 85000, 89000, 95000 ... 130000, 134000, 140000, 145000, 150000`.
  - **Names**: from `checkpoint_28_5000.pt` and `checkpoint_56_10000.pt` to `checkpoint_806_145000.pt` and `checkpoint_834_150000.pt`. That is 30 names per seed and 150 distinct paths.
  - **Path hash**: equals the `join_right` construction.
  - **Pickle round-trip**: the path is the same after the round-trip, and `_available` resolves to `config.w2vu2._checkpoint_file_exists`.
  - **Routing**: the new forward keeps `{cpu 4, mem 24, time 2, gpu 0}` after `check_engine_limits`. The checkpoint_best forward `d9mpykiqAEoT` is on device cpu, and the select/train forward on gpu.
  - **`device` is unhashed**: the config hash is the same for cpu and gpu.

**Still open.**

- 180 rests on the reviewer's `batch_by_size` run and on production's `checkpoint_823_148000.pt`. It is not yet confirmed on i6. `checkpoint_6_1000.pt` and `checkpoint_28_5000.pt` cannot tell 179 from 180, because both give the same epoch index. The first saves that can tell them apart are U = 7000 (`checkpoint_39_7000.pt` for 180 against `checkpoint_40_7000.pt` for 179), 9000 and 12000. The first grid point that differs is 25000. Check `checkpoint_39_7000.pt` in seed 0's `output/checkpoints/`.
- The graph asserts check the yaml's constants and `EXPECTED_UTTS`, not the GAN feature data's actual line count.

### Round 3 cumulative diffs (against HEAD)

```diff
diff --git a/users/wu/experiments/unsupervised_asr/config/w2vu2.py b/users/wu/experiments/unsupervised_asr/config/w2vu2.py
index bc68b9fb7..ac3a3ac78 100644
--- a/users/wu/experiments/unsupervised_asr/config/w2vu2.py
+++ b/users/wu/experiments/unsupervised_asr/config/w2vu2.py
@@ -36,9 +36,17 @@ The graph (:func:`py`)
 4-gram KenLM (i6_core ``KenLMplzJob`` + ``CreateBinaryLMJob``), the fairseq feature data from the
 port's VAD-trimmed L15 stream (``data.w2vu2_features``), one i6_core ``FairseqHydraTrainingJob`` per
 seed and ``W2vu2GanSelectJob``.  Then per seed (production evaluated every seed) the generator is
-converted to a RETURNN checkpoint and decoded greedily by ``ReturnnForwardJobV2`` on dev-clean and
-dev-other, and scored by ``W2vu2GanPerJob`` (``analysis.w2vu2_gan_eval``).  The selected seed's
-generator decodes train-clean-100 into the pseudo-labels (``W2vu2GanPseudoLabelJob``).
+converted to a RETURNN checkpoint and decoded greedily by ``ReturnnForwardJobV2`` (on CPU) on
+dev-clean and dev-other, and scored by ``W2vu2GanPerJob`` (``analysis.w2vu2_gan_eval``).  The selected
+seed's generator decodes train-clean-100 into the pseudo-labels (``W2vu2GanPseudoLabelJob``).
+
+Intermediate evaluation (not in production): the same conversion, CPU forward and PER chain on each
+seed's update checkpoint ``checkpoint_<E>_<U>.pt`` every :data:`INTERMEDIATE_EVAL_INTERVAL` updates, a
+point on an epoch end moved one save earlier (:func:`intermediate_eval_updates`,
+:func:`_intermediate_checkpoint`).  Each point waits only for its own checkpoint file (a custom
+``available`` check, not the training's completion), so it runs while the training runs.  fairseq's
+in-training ``uer`` is no substitute: the GAN's ``valid`` split has no labels, so it logs 0.0.
+Outputs: ``<GAN_ARM>/intermediate/s<seed>/u<U>/<split>/per.{json,txt}``.
 
 1d (``training.w2vu2_ctc``, ``analysis.w2vu2_ctc_decode``): fairseq manifests of the ogg zips, the
 CTC data dir, the LV-60 fairseq checkpoint, the CTC fine-tune as i6_core ``FairseqHydraTrainingJob``,
@@ -76,8 +84,13 @@ unhashed exceptions:
   renames that constant, the trainings fall back silently to gpu_48gb (L40S): check that the first
   training's ``submit_log.run`` shows ``-p gpu_32gb``.
 
-The forwards and decodes keep production's ``gpu_mem`` 40 (:data:`FORWARD_RQMT`; the two CTC decodes'
-own default), which the i6 rule ``gpu_mem > 24`` sends to gpu_48gb (L40S 46 GB), as intended.  The
+The train-clean-100 pseudo-label forward and the two CTC decodes keep production's ``gpu_mem`` 40
+(:data:`FORWARD_RQMT`; the two CTC decodes' own default), which the i6 rule ``gpu_mem > 24`` sends to
+gpu_48gb (L40S 46 GB), as intended.  The dev-clean / dev-other generator forwards (per-seed
+``checkpoint_best.pt`` and intermediate) run on CPU (:data:`FORWARD_CPU_RQMT`: production's time, mem
+and cpu without a GPU; ``check_engine_limits`` leaves them on the default CPU partition), so every
+per-seed PER comes from the same device; ``device`` sits in the RETURNN ``post_config`` and is not
+hashed.  The
 GAN took 10.5 h and the CTC student 4.25 h on GH200 under an 11.5 h limit (i6's 72 h training floor
 also matches only ``ReturnnTrainingJob``); on V100 both will take longer and rely on
 ``FairseqHydraTrainingJob``'s resumable ``run`` task (fairseq resumes from ``checkpoint_last.pt``).
@@ -153,6 +166,8 @@ Run from the setup dir (sisyphus imports the module and calls :func:`py`)::
 from __future__ import annotations
 
 import contextlib
+import math
+import os
 from typing import Any, Dict, Iterator
 
 from sisyphus import gs, tk
@@ -166,7 +181,16 @@ __all__ = [
     "I6_TRAIN_GPU_MEM",
     "I6_TRAIN_PARTITION_SETTING",
     "FORWARD_RQMT",
+    "FORWARD_CPU_RQMT",
     "PHONE_DECODE_TIME",
+    "INTERMEDIATE_EVAL_INTERVAL",
+    "GAN_TRAIN_UTTS",
+    "GAN_BATCH_SIZE",
+    "GAN_BATCH_SIZE_MULTIPLE",
+    "GAN_UPDATES_PER_EPOCH",
+    "INTERMEDIATE_EPOCH_END_SHIFT",
+    "intermediate_eval_updates",
+    "gan_update_checkpoint_name",
     "gan_1c",
     "selftrain_1d",
     "py",
@@ -192,9 +216,51 @@ I6_TRAIN_GPU_MEM = 32
 I6_TRAIN_PARTITION_SETTING = "GPU_ROUTE_TRAIN"
 #: production's ``W2vu2PerEvalJob`` and ``GanPseudoLabelJob`` rqmt, for the RETURNN generator forwards
 FORWARD_RQMT = {"time_rqmt": 2, "mem_rqmt": 24, "cpu_rqmt": 4, "gpu_mem": 40}
+#: :data:`FORWARD_RQMT` without the GPU, for the dev-clean / dev-other generator forwards (``device="cpu"``)
+FORWARD_CPU_RQMT = {k: v for k, v in FORWARD_RQMT.items() if k != "gpu_mem"}
 #: production's dev-only ``Wav2Vec2CtcDecodeJob.qqKPLPBEt1K3`` time (``CtcPhoneDecodeJob`` defaults to
 #: 3 h, the train-including ``decode_all``'s)
 PHONE_DECODE_TIME = 2
+#: updates between two intermediate evaluations (K = 5000: 30 points per seed up to max_update 150000)
+INTERMEDIATE_EVAL_INTERVAL = 5000
+#: a grid point on an epoch end moves this many updates earlier (one ``save_interval_updates``): fairseq
+#: writes no ``checkpoint_<E>_<U>.pt`` at an epoch end
+INTERMEDIATE_EPOCH_END_SHIFT = 1000
+#: GAN training utterances (train-clean-100, ``data.librispeech.EXPECTED_UTTS``), the yaml's
+#: ``dataset.batch_size`` and fairseq's ``dataset.required_batch_size_multiple`` (default 8, not set in
+#: the yaml); all three are asserted at graph time in :func:`gan_1c`
+GAN_TRAIN_UTTS = 28_539
+GAN_BATCH_SIZE = 160
+GAN_BATCH_SIZE_MULTIPLE = 8
+#: fairseq updates per GAN epoch.  ``batch_by_size`` (max_sentences 160, multiple 8) gives 178 full
+#: batches of 160 and splits the 59 left-over utterances into 56 + 3, so 180 = full batches
+#: + (rem >= 8) + (rem % 8 > 0) (checked by the review with fairseq 0.12.2's batch_by_size; production s0
+#: has ``checkpoint_823_148000.pt`` = ceil(148000 / 180)).  It fixes E = ceil(U / 180) of
+#: ``checkpoint_<E>_<U>.pt``.  180 holds for N_train 28,537 .. 28,543 around 28,539 (overall for 160
+#: values in [28,489, 28,800], not a contiguous range: e.g. 28,536 and 28,544 give 179).  If it were
+#: wrong, the named files would never appear and the intermediate points would wait, not evaluate a wrong
+#: checkpoint.
+GAN_UPDATES_PER_EPOCH = (GAN_TRAIN_UTTS // GAN_BATCH_SIZE
+                         + int(GAN_TRAIN_UTTS % GAN_BATCH_SIZE >= GAN_BATCH_SIZE_MULTIPLE)
+                         + int(GAN_TRAIN_UTTS % GAN_BATCH_SIZE % GAN_BATCH_SIZE_MULTIPLE > 0))
+assert GAN_UPDATES_PER_EPOCH == 180, GAN_UPDATES_PER_EPOCH
+
+
+def intermediate_eval_updates(max_update: int) -> tuple:
+    """The evaluated updates: every :data:`INTERMEDIATE_EVAL_INTERVAL` up to ``max_update``, a point on an
+    epoch end moved :data:`INTERMEDIATE_EPOCH_END_SHIFT` earlier (150000: 45000, 90000, 135000 -> 44000,
+    89000, 134000; 30 points)."""
+    out = tuple(u - INTERMEDIATE_EPOCH_END_SHIFT if u % GAN_UPDATES_PER_EPOCH == 0 else u
+                for u in range(INTERMEDIATE_EVAL_INTERVAL, max_update + 1, INTERMEDIATE_EVAL_INTERVAL))
+    assert len(set(out)) == len(out), out
+    return out
+
+
+def gan_update_checkpoint_name(update: int) -> str:
+    """fairseq's name of the save at ``update`` (not an epoch end): ``checkpoint_<E>_<U>.pt``,
+    E = ceil(U / :data:`GAN_UPDATES_PER_EPOCH`)."""
+    assert update % GAN_UPDATES_PER_EPOCH != 0, (update, GAN_UPDATES_PER_EPOCH)
+    return f"checkpoint_{math.ceil(update / GAN_UPDATES_PER_EPOCH)}_{update}.pt"
 
 
 def _i6_train_rqmt(rqmt: Dict[str, Any]) -> Dict[str, Any]:
@@ -207,6 +273,25 @@ def _i6_train_rqmt(rqmt: Dict[str, Any]) -> Dict[str, Any]:
     return rqmt
 
 
+def _checkpoint_file_exists(path: tk.Path) -> bool:
+    """``available`` check of an intermediate checkpoint Path: the file exists.  fairseq writes
+    ``checkpoint_<E>_<U>.pt`` to ``.tmp`` and renames it, so the name never shows a partial file.
+    Module level: sisyphus pickles it with the Path (job.save); moving or renaming it breaks the
+    unpickling of existing job.save files."""
+    return os.path.isfile(path.get_path())
+
+
+def _intermediate_checkpoint(train, update: int) -> tk.Path:
+    """``checkpoints/<gan_update_checkpoint_name(update)>`` of the fairseq GAN ``train``, available as soon
+    as the file exists.
+
+    The Path hash is (creator, path), as for ``out_checkpoint_dir.join_right(...)``; the ``available``
+    callable is not hashed.  fairseq names a save ``checkpoint_<E>_<U>.pt`` only when it is not at an
+    epoch end (asserted in :func:`gan_update_checkpoint_name`)."""
+    return tk.Path(f"checkpoints/{gan_update_checkpoint_name(update)}", creator=train,
+                   available=_checkpoint_file_exists)
+
+
 @contextlib.contextmanager
 def _scoped() -> Iterator[None]:
     """Build jobs and register outputs under :data:`ALIAS_AND_OUTPUT_PREFIX`; restore afterwards."""
@@ -223,34 +308,51 @@ def gan_1c(inputs) -> Dict[str, Any]:
 
     :param inputs: ``inputs.get_inputs()``.
     :return: ``{"gan": W2vu2Gan, "eval": {seed: {"generator", split: {"forward", "per"}}},
+        "intermediate": {seed: {update: {"generator", split: {"forward", "per"}}}},
         "select": {"generator", "forward", "labels"}}``.
     """
     from ..analysis.w2vu2_gan_eval import W2vu2GanPerJob, W2vu2GanPseudoLabelJob, w2vu2_forward_job, \
         w2vu2_generator_checkpoint
     from ..data.librispeech import EXPECTED_UTTS
-    from ..training.w2vu2_gan import get_w2vu2_gan
+    from ..training.w2vu2_gan import GAN_MAX_UPDATE, get_w2vu2_gan, w2vu2_base_config
+
+    # the constants behind GAN_UPDATES_PER_EPOCH (intermediate checkpoint names)
+    assert EXPECTED_UTTS["train-clean-100"] == GAN_TRAIN_UTTS, (EXPECTED_UTTS["train-clean-100"], GAN_TRAIN_UTTS)
+    dataset_cfg = w2vu2_base_config()["dataset"]
+    assert dataset_cfg["batch_size"] == GAN_BATCH_SIZE, GAN_BATCH_SIZE
+    # fairseq 0.12.2 DatasetConfig.required_batch_size_multiple defaults to 8
+    assert dataset_cfg.get("required_batch_size_multiple", 8) == GAN_BATCH_SIZE_MULTIPLE, dataset_cfg
 
     gan = get_w2vu2_gan(seeds=PRODUCTION_SEEDS)
     text_dict = gan.text_data.out_dict
     arm = GAN_ARM.split("/", 2)[2]  # w2vu2_forward_job prefixes its alias with sae/1c/gan
 
-    evals: Dict[int, Dict[str, Any]] = {}
-    for seed, train in gan.trainings.items():
-        train.rqmt.update(_i6_train_rqmt(train.rqmt))  # unhashed (module docstring, Resources)
+    def dev_eval(fairseq_checkpoint: tk.Path, prefix: str) -> Dict[str, Any]:
+        """conversion, CPU forward and PER on :data:`EVAL_SPLITS`; aliases and outputs under ``prefix``
+        (relative to :data:`GAN_ARM`)"""
         conv, ckpt = w2vu2_generator_checkpoint(
-            fairseq_checkpoint=train.out_checkpoint_dir.join_right("checkpoint_best.pt"),
-            text_dict=text_dict, alias=f"{GAN_ARM}/s{seed}/generator")
-        evals[seed] = {"generator": conv}
+            fairseq_checkpoint=fairseq_checkpoint, text_dict=text_dict, alias=f"{GAN_ARM}/{prefix}/generator")
+        res = {"generator": conv}
         for split in EVAL_SPLITS:
             fwd = w2vu2_forward_job(
-                name=f"{arm}/s{seed}/{split}", checkpoint=ckpt, vocab=conv.out_vocab,
+                name=f"{arm}/{prefix}/{split}", checkpoint=ckpt, vocab=conv.out_vocab,
                 feature_hdfs=list(inputs.vad.out_feature_hdfs[split]), expected_num_seqs=EXPECTED_UTTS[split],
-                **FORWARD_RQMT)
+                device="cpu", **FORWARD_CPU_RQMT)
             per = W2vu2GanPerJob(hyps=fwd.out_files["hyps.json"], gold=inputs.gold, split=split)
-            per.add_alias(f"{GAN_ARM}/s{seed}/per/{split}")
-            tk.register_output(f"{GAN_ARM}/s{seed}/{split}/per.json", per.out_per)
-            tk.register_output(f"{GAN_ARM}/s{seed}/{split}/per.txt", per.out_report)
-            evals[seed][split] = {"forward": fwd, "per": per}
+            per.add_alias(f"{GAN_ARM}/{prefix}/per/{split}")
+            tk.register_output(f"{GAN_ARM}/{prefix}/{split}/per.json", per.out_per)
+            tk.register_output(f"{GAN_ARM}/{prefix}/{split}/per.txt", per.out_report)
+            res[split] = {"forward": fwd, "per": per}
+        return res
+
+    evals: Dict[int, Dict[str, Any]] = {}
+    intermediate: Dict[int, Dict[int, Dict[str, Any]]] = {}
+    for seed, train in gan.trainings.items():
+        train.rqmt.update(_i6_train_rqmt(train.rqmt))  # unhashed (module docstring, Resources)
+        evals[seed] = dev_eval(train.out_checkpoint_dir.join_right("checkpoint_best.pt"), f"s{seed}")
+        intermediate[seed] = {
+            update: dev_eval(_intermediate_checkpoint(train, update), f"intermediate/s{seed}/u{update}")
+            for update in intermediate_eval_updates(GAN_MAX_UPDATE)}
     tk.register_output(f"{GAN_ARM}/select/selection.json", gan.selection.out_selection)
 
     # the selected seed's generator labels train-clean-100 (production: GanPseudoLabelJob on s0)
@@ -263,7 +365,8 @@ def gan_1c(inputs) -> Dict[str, Any]:
     labels = W2vu2GanPseudoLabelJob(hyps=[fwd.out_files["hyps.json"]], expected_num_seqs=n_train)
     labels.add_alias(f"{SELFTRAIN_PREFIX}/pseudo_labels")
     tk.register_output(f"{SELFTRAIN_PREFIX}/pseudo_labels.json", labels.out_labels)
-    return {"gan": gan, "eval": evals, "select": {"generator": conv, "forward": fwd, "labels": labels}}
+    return {"gan": gan, "eval": evals, "intermediate": intermediate,
+            "select": {"generator": conv, "forward": fwd, "labels": labels}}
 
 
 def selftrain_1d(inputs, pseudo_labels: tk.Path) -> Dict[str, Any]:
diff --git a/users/wu/experiments/unsupervised_asr/tests/test_w2vu2_config.py b/users/wu/experiments/unsupervised_asr/tests/test_w2vu2_config.py
index d1081527e..2e018265a 100644
--- a/users/wu/experiments/unsupervised_asr/tests/test_w2vu2_config.py
+++ b/users/wu/experiments/unsupervised_asr/tests/test_w2vu2_config.py
@@ -31,10 +31,11 @@ EXPECTED_COUNTS = {
     # 5 GAN seeds + the 1d CTC student
     "FairseqHydraTrainingJob": 6,
     "W2vu2GanSelectJob": 1,
-    # report B: 5 seeds + the selected generator; 5 seeds x 2 dev splits + the train forward
-    "W2vu2GeneratorCheckpointJob": 6,
-    "ReturnnForwardJobV2": 11,
-    "W2vu2GanPerJob": 10,
+    # report B: 5 seeds + the selected generator; 5 seeds x 2 dev splits + the train forward;
+    # plus the intermediate eval: 5 seeds x 30 updates (5000 .. 150000), each on 2 dev splits
+    "W2vu2GeneratorCheckpointJob": 6 + 150,
+    "ReturnnForwardJobV2": 11 + 300,
+    "W2vu2GanPerJob": 10 + 300,
     "W2vu2GanPseudoLabelJob": 1,
     # report C
     "FairseqAudioManifestJob": 3,
@@ -49,8 +50,14 @@ EXPECTED_COUNTS = {
 GAN_RQMT = {"gpu": 1, "gpu_mem": 32, "mem": 100, "time": 11.5, "cpu": 8}
 CTC_RQMT = {"gpu": 4, "gpu_mem": 32, "mem": 60, "time": 11.5, "cpu": 16}
 FORWARD_RQMT = {"gpu": 1, "gpu_mem": 40, "mem": 24, "time": 2, "cpu": 4}
+#: the dev-clean / dev-other generator forwards (checkpoint_best and intermediate): FORWARD_RQMT on CPU
+DEV_FORWARD_RQMT = {"gpu": 0, "mem": 24, "time": 2, "cpu": 4}
 PHONE_DECODE_RQMT = {"gpu": 1, "gpu_mem": 40, "mem": 24, "time": 2, "cpu": 4}
 WORD_DECODE_RQMT = {"gpu": 1, "gpu_mem": 40, "mem": 64, "time": 11.5, "cpu": 8}
+#: the intermediate-eval updates: every 5000, the epoch ends 45000 / 90000 / 135000 (180 updates per epoch)
+#: moved one save (1000 updates) earlier
+INTERMEDIATE_UPDATES = sorted({45000: 44000, 90000: 89000, 135000: 134000}.get(u, u)
+                              for u in range(5000, 150_001, 5000))
 
 
 @pytest.fixture
@@ -167,6 +174,41 @@ def test_pseudo_labels_come_from_the_selected_generator(built):
     assert labels.hyps == [sel["forward"].out_files["hyps.json"]] and labels.expected_num_seqs == 28539
 
 
+def test_dev_forwards_of_best_and_intermediate_chains_run_on_cpu(built):
+    """Every dev forward, of ``checkpoint_best.pt`` and of every intermediate checkpoint, uses device cpu."""
+    res, _, _ = built
+    evals, inter = res["1c"]["eval"], res["1c"]["intermediate"]
+    assert sorted(inter) == list(w2vu2.PRODUCTION_SEEDS)
+    forwards = [evals[s][split]["forward"] for s in evals for split in w2vu2.EVAL_SPLITS]
+    for seed, points in inter.items():
+        assert sorted(points) == INTERMEDIATE_UPDATES, (seed, sorted(points))
+        for u in points:
+            conv = points[u]["generator"]
+            assert conv.fairseq_checkpoint.creator is res["1c"]["gan"].trainings[seed]
+            assert conv.fairseq_checkpoint.path == f"checkpoints/{w2vu2.gan_update_checkpoint_name(u)}"
+        forwards += [points[u][split]["forward"] for u in points for split in w2vu2.EVAL_SPLITS]
+    assert len(forwards) == 10 + 300
+    for fwd in forwards:
+        assert fwd.device == "cpu" and fwd.rqmt["gpu"] == 0, (fwd.device, fwd.rqmt)
+
+
+def test_intermediate_checkpoint_names_follow_fairseq():
+    """180 updates per epoch (batch_by_size: 178 x 160 + 56 + 3); production's names; no epoch end."""
+    assert w2vu2.GAN_UPDATES_PER_EPOCH == 180
+    assert w2vu2.gan_update_checkpoint_name(148000) == "checkpoint_823_148000.pt"  # production s0 best
+    assert w2vu2.gan_update_checkpoint_name(7000) == "checkpoint_39_7000.pt"
+    assert list(w2vu2.intermediate_eval_updates(150_000)) == INTERMEDIATE_UPDATES
+    assert all(u % 180 for u in INTERMEDIATE_UPDATES)
+    with pytest.raises(AssertionError):
+        w2vu2.gan_update_checkpoint_name(45000)  # an epoch end: fairseq writes no update-named file
+
+
+def test_pseudo_label_forward_stays_on_gpu(built):
+    res, _, _ = built
+    fwd = res["1c"]["select"]["forward"]
+    assert fwd.device == "gpu" and fwd.rqmt["gpu"] == 1 and fwd.rqmt["gpu_mem"] == 40, (fwd.device, fwd.rqmt)
+
+
 def test_selftrain_wiring(built):
     res, _, out = built
     from i6_experiments.users.wu.experiments.unsupervised_asr.lm.word_lm import official_4gram_arpa, official_lexicon
@@ -209,7 +251,11 @@ def test_rqmt_is_production_except_training_gpu_mem(built):
         assert job.rqmt == GAN_RQMT, job.rqmt
     assert res["1d"]["train"].rqmt == CTC_RQMT, res["1d"]["train"].rqmt
     forwards = _of(new, "ReturnnForwardJobV2")
-    assert forwards and all(j.rqmt == FORWARD_RQMT for j in forwards), [j.rqmt for j in forwards]
+    select_forward = res["1c"]["select"]["forward"]
+    dev_forwards = [j for j in forwards if j is not select_forward]
+    assert len(forwards) == 311 and len(dev_forwards) == 310
+    assert all(j.rqmt == DEV_FORWARD_RQMT for j in dev_forwards), [j.rqmt for j in dev_forwards]
+    assert select_forward.rqmt == FORWARD_RQMT, select_forward.rqmt
     assert res["1d"]["phone"].rqmt == PHONE_DECODE_RQMT
     assert res["1d"]["word"].rqmt == WORD_DECODE_RQMT
 
```
