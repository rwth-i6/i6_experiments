# SAE — scaling the 2S GRPO loop from the 10 h seed to LibriSpeech 960 h

## State

Closed: no active experiment and no live run pointer. Findings are in the Conclusion section below;
work that reopens this phase restates its live gate here before producing a new result.

## Approach

**1. Scale the validated 10 h avunits loop to 960 h of unlabeled audio, assign-only.** Ground truth
pinned before designing: the existing state dumps and units cover the **10 h seed only** (8416
train+dev tags), and the 360/500 HF-ogg dirs did not exist, so the scale-up needs new ogg conversions
(104,014 + 148,688 utts) and new sharded state dumps over 281,241 train utterances. Design decisions,
each with the reason:

- **Label quarantine at conversion.** 360/500 transcripts are blanked to `""` when the per-subset ogg
  dirs are built, so the quarantined text never exists on disk; tc100 keeps gold text exactly as the
  frozen 10 h loop does (unread by the train step), dev keeps gold for evaluation only.
- **Per-subset dirs plus a concat job**, so ogg conversion runs once per subset, the merged dir is a
  cheap arrow rewrite, and each subset's state dump can start as soon as its own dir lands.
- **Units are assign-only, never refit.** `AssignUnitsJob` reconstructs PCA + MiniBatchKMeans from the
  seed quantizer's `quantizer.pkl` and calls the same per-utterance `quantize_utt`, so the arithmetic
  is bit-identical; the seed's own units win for the 8416 seed tags, so the loop keeps the exact
  validated reward targets.
- **In-train evals cut to 500-seq subsets.** Eval *frequency* is not configurable in this RETURNN
  torch engine — `eval_model()` runs unconditionally each sub-epoch and `_check_missing_eval` raises
  on missing epoch scores after a restart — so the only lever is subset size and set choice; this
  turns ~1.3 h of combined dev+devtrain eval per sub-epoch into ~10 min.
- **Schedule** `partition_epoch=10`, `num_epochs=20` (2 passes), warmup matched in **absolute steps**
  to the frozen 10 h arm (0.05066 sub-epochs ~ 712 steps).
- **One sisyphus graph** from conversions through recogs, so the manager orders everything and no
  raw-path waiter is needed.

**2. The joint-AR twin, built as a single-variable flip.** The seed-scale joint config imported the
frozen config's `baseline()`, which puts the frozen *train job* into the joint graph — harmless at
seed scale, but at 960 h the frozen job is walltime-killed ~20 times and two concurrent managers would
each see it resubmittable. The frozen module was therefore refactored additively into
`build_960h_train_setup()` = data pipeline + model args and no train job. Both train jobs were built
in one process and their serialized RETURNN configs diffed: the **only** difference is
`freeze_ar: True -> False` plus each job's own model output path.

| arm | steps/min | steps per sub-epoch | h per sub-epoch | fits the 11.5 h cap? |
|---|---|---|---|---|
| frozen `asmULNUedMNN` | 23.4 | 12784 | **9.1** | yes (2.4 h margin) |
| joint `YkGfwdTuKu7d` | 13.2 | 12784 | **16.1** | **no** |

## Conclusion

1. (1) Assign-only units are exact: the reconstructed quantizer re-assigned the full finished seed
   dump with **0 mismatching utterances and 0 mismatching frames** over 8416 utts / 924,607 frames,
   and the same check is wired in-graph rather than run once by hand.
2. (1) The `partition_epoch=10` schedule costs a full sub-epoch of throughput per allocation: at
   ~25 steps/min a sub-epoch's train plus its two evals ends ~11 h into an 11.5 h slot, so each
   allocation completes exactly **one** sub-epoch and loses the ~2 h of the next one — 20 allocations,
   ~9.6 days plus queue latency. Not fixable in flight (`partition_epoch` is hashed); recorded as the
   cost of this schedule.
3. (2) **The joint-AR arm was structurally unable to checkpoint and was deleted.** At 13.2 steps/min a
   sub-epoch needs 16.1 h against an 11.5 h cap, and the failure mode is a silent infinite loop rather
   than a crash: SIGTERM at the cap with no checkpoint -> `interrupted_resumable` -> restart from step
   0 -> repeat, at ~46 GPU-h per cycle. It had already completed one full wasted cycle. The built-in
   escape hatch does not apply — `update_engine_rqmt` doubles `time` on timeout but
   `check_engine_limits` clamps it back to 11.5 h.
4. (3) **Generalizable pre-launch check for any long loop:** `steps_per_sub_epoch /
   measured_steps_per_hour` must be < 11.5 *before* launch. Nothing in the graph or the config
   enforces it, and a job that never checkpoints looks exactly like a job running normally.
5. (2) No drift arbiter was built for the 960 h arms: at seed scale the theta_0-anchored drift probe
   was shown to underestimate an adapting reward (spearman and eta departed monotonically while WER
   improved), so a 960 h drift series would cost GPU for a number we already know how to misread.
   Trajectory WER is the verdict instrument.

## Catalog

`T/` = `work/i6_core/returnn/training/`.

| artifact | path |
|---|---|
| entry point + shims | `config/sae_2s_grpo_loop_avunits_960h.py`, `configs/config_sae_2s_grpo_loop_avunits_960h_v1.py` |
| **frozen 960 h loop (the arm that ran)** | `T/ReturnnTrainingJob.asmULNUedMNN` |
| joint 960 h twin (deleted; recipe config kept with an ABANDONED header) | was `T/ReturnnTrainingJob.YkGfwdTuKu7d` |
| ogg conversions (360 / 500 / merged) | `TransformAndMapHuggingFaceDatasetJob.{xYXH1u8T7RN3,HZmYeNZ1emya,1c6JQRMlzCyy}` |
| sharded state dumps (tc100 x2 / 360 x4 / 500 x6) | `AvStatesJob.{9HW4sHnnlOeJ,T4bOCrXoz0YG,jz66Ddj0Lppj}` |
| assigns + anchor compare + merge + pack | `AssignUnitsJob.{LWVLTd2LCPLP,uVhMo3LZWazF,D8jUwTsSKwUn}`, `CompareUnitsJob.eu4zbgNpfERy`, `MergeUnitsPklJob.Ux0V027RS7h8`, `PackUnitsJob.xaIQHKux4CSV` |
| seed quantizer the assigns reconstruct | `work/speech_llm/sae/quantize_states/QuantizeStatesJob.Zp4b9U9L3gSQ` |
| lazy memmap units attach (the 960 h store) | `prefix_lm/model/util/units_attach.py` — `AttachUnitsBySeqTagV2` + `PackUnitsJob` |
| commits (branch `haotian_modality_matching_jupiter`) | `cdc9d10` (loop line), `9305b39` (960 h scale-up + joint arm) |

**Two uncommitted local RETURNN fixes are load-bearing and must survive any checkout**
(`recipe/returnn` is not a hash input, so job identities are untouched by them): `task_system.py`
`numpy.fromstring` -> `numpy.frombuffer(...).copy().reshape(shape)`, needed because the 960 h config
carries the merged units as int16 **arrays** and numpy 2.4.6 removed binary-mode `fromstring`; and a
`torch/engine.py` guard skipping never-populated `extern_data` keys in the epoch-end padding
statistic, which otherwise divides by zero on the blanked `text` stream at the end of every sub-epoch.

The V1 units attach carried the 287 k-utterance dict inside the pickled global config (102 MB per
attach instance, re-sent to every spawned worker) and died at tree RSS ~545 GB; V2 is a packed memmap
store opened lazily and shared through the page cache — real-anchor verified bit-identical to V1 on
252 utterances / 35,862 frames across all five sources.

## Verifier feedback

None recorded.
