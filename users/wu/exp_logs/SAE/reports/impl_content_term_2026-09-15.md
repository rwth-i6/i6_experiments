# S3b-F: the label-free forward content term (Liu et al. 2022) -- implementation

2026-09-15, implementer.  Sources: design review `design_review_s3b_2026-09-15.md` (F2, item 5);
`SAE_4A.md` lines ~842-862 and `reports/lit_cold_start_collapse_2026-09-15.md` for Liu et al. 2022
(wav2vec-U 2.0, SLT), eq. (5) and Table 2 (unsupervised PER on LS dev-other: no term 15.9 +- 1.1,
K = 50 15.2, **K = 64 13.6 +- 0.9**, K = 100 14.8, K = 128 16.8, VQ 320x2 16.6 +- 2.2 -- 64 is the
measured optimum, 128 and the VQ target are WORSE than no term at all).

**Pre-registered reading rule** (carried in the docstrings of `content_term.py` and `mfcc_codes.py`,
memory `preregistration-lives-with-the-code`): `emc/content_ce` and `emc/content_acc` are MONITORS.
Take-off of the cold start is read on the phone rate, the vocabulary usage and the derangement
margin, NEVER on `L_tau` and never on the content CE.  The accuracy's baseline is the largest code's
share, which `MfccKMeansJob` prints beside it.

**Label-free by construction.**  The target is MFCC k-means over the SAME audio the L15 dump reads.
No transcript, alignment, lexicon or gold quantity enters any job, any constant or any test.

## Files written (owned by this dispatch)

| file | what |
|---|---|
| `src/speech_llm/sae/emc/content_term.py` (new) | masked per-frame CE + frame-accuracy monitor |
| `src/speech_llm/sae/emc/test_content_term.py` (new) | its tests + the head-off recognizer identity |
| `src/speech_llm/sae/emc/mfcc_codes.py` (new) | the MFCC / k-means / codes-HDF sisyphus jobs |
| `src/speech_llm/sae/emc/test_mfcc_codes.py` (new) | their tests |
| `src/speech_llm/sae/emc/recognizer.py` (modified) | optional tap + linear head of K outputs, default OFF |

### `content_term.py`

```python
CONTENT_CODES_KEY = "content_codes"
CONTENT_K = 64
class ContentStats(NamedTuple): ce; acc; frames; n_classes
def valid_frame_mask(lens, max_len, *, device=None) -> Tensor
def content_cross_entropy(*, logits, codes, lens, n_classes=None) -> Tuple[Tensor, ContentStats]
```

Conventions, all stated in the docstring: per valid frame MEAN (not the sum of eq. 5 -- a sum would
price long utterances higher and is not comparable across batches); padded frames excluded from both
the loss and the accuracy; `log_softmax` in float32 whatever the logits' dtype; a code outside
`[0, K)` on a VALID frame is refused, on a padded frame ignored; the accuracy is computed under
`no_grad`.

### `recognizer.py` -- the tap and the head

`CONTENT_TAPS = {0: "post_bn", 1: "pre_conv", 2: "mid"}` (tap 2 exists only for `n_layers = 2` and
`stride = 1`).  The constructor gained `content_k=None, content_layer=None` (both or neither), and
the head is constructed **last**, so with it off the global torch RNG draw sequence is unchanged --
`emc/bt_probe.constructed_phi_0` depends on that stream.  `forward` still returns ONLY the
log-probs (the fixed interface); `forward_with_content(feats, lens) -> (log_probs, content_logits)`
raises when there is no head.

Default-off is asserted, not assumed: `test_the_default_recognizer_is_unchanged_when_the_head_is_off`
pins the state_dict key list literally
(`bn.{bias,num_batches_tracked,running_mean,running_var,weight}, conv.weight, in_proj.{bias,weight}`)
and `test_the_head_changes_no_log_prob_and_no_pre_existing_parameter` builds head-on / head-off pairs
at the same seed over `n_layers in {1, 2}` and every valid tap: identical parameter VALUES,
`torch.equal(lp_off, lp_on)`, and the only extra keys are `content_head.{weight,bias}`.

### `mfcc_codes.py` -- the data side

**Audio path (label-free, stated as required).**  The same HuggingFace audio the L15 dump reads:
tc100 = `TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb/output/dataset`, split `train` (the dir
`AvStatesJob.Dsynh5MqmgjY` reads); dev = `TransformAndMapHuggingFaceDatasetJob.DlfDYnmTEkGZ/output/dataset`,
split `dev` (the dir `AvStatesJob.c4Ak1rACchRC` reads).  Raw waveform only.

**Frame grid -- the exact, tolerance-0 alignment rule.**  Not a resampling: the MFCC grid IS the
wav2vec2 conv front end's own.  With kernels (10,3,3,3,3,2,2) and strides (5,2,2,2,2,2,2) the
sequential length rule equals `T = floor((n_samples - 400) / 320) + 1` exactly (checked numerically
over `n in [400, 200000)`, 0 mismatches, and re-checked in `test_mfcc_codes` against the layer-by-layer
rule).  Frame `t` covers samples `[320t, 320t + 400)` = a 25 ms window at a 20 ms hop = 50 Hz.  So the
MFCC frame count EQUALS the L15 frame count per utterance by construction, and `MfccCodesHdfJob`
asserts equality at **tolerance 0** against the feature HDFs themselves (no truncation, no padding).

**MFCC definition** (numpy + scipy only -- memory `librosa-numba-cache-corrupt`; no librosa, no numba):
Hann periodic window over the 400-sample frame, rFFT at `n_fft = 512`, 40 HTK-mel triangular filters
over 20 Hz - 8 kHz with unit height, `log(max(e, 1e-10))`, `scipy.fft.dct(type=2, norm="ortho")`,
first 13 coefficients (C0 kept).  **Deltas: 2 orders, N = 2** regression
`sum_n n (c[t+n] - c[t-n]) / (2 sum_n n^2)` with replicated edges -> **39 dims**.  My call, as the
dispatch allows: Liu's own target is MFCC+deltas, the extra 26 dims cost nothing at k-means scale,
and a 13-dim static-only target would put more of the code identity on the absolute energy.

**k-means**: `MiniBatchKMeans(n_clusters=64, random_state=42, batch_size=4096, n_init=3, max_iter=100)`
-- the convention already in `sae/build_units.py:173-174`, seed 42.  Fit on a fixed-seed subset of
**tc100 train**: `KMEANS_FIT_UTTS = 2000` utterances drawn by a seeded permutation of the sorted tag
list, capped at `KMEANS_MAX_FIT_VECTORS = 1_000_000` frames (2000 utts x ~600 frames ~ 1.2 M, so the
cap binds and the fit sees ~1 M 39-dim vectors).  Features are globally standardized (mean/std of the
fit subset, `STD_FLOOR = 1e-5`) before the fit and with the SAME statistics at assignment -- the
`per_utt_standardize=False` convention of `sae/quantize_states.py`.  `MfccKMeansJob` prints the code
usage histogram, the empty-code count, the entropy and the **largest code's share**, which is the
prior baseline `emc/content_acc` has to beat.

**Jobs**: `MfccFeatureJob` (sharded, `run()` in-process -- no model is involved, so no
`ReturnnForwardJobV2`), `MfccKMeansJob`, `MfccCodesHdfJob` (writes one sparse int32 HDF per feature
shard, `dim = K`, `ndim = 1`, tags and order taken from the feature HDFs so the MetaDataset join by
seq tag is exact; splits: tc100 train, the CV subset and dev-other).  Few large HDFs, not one per
utterance (memory `inode-cost-is-fan-in-times-reruns`).

## UNDETERMINED -- the orchestrator must state it

`content_layer` has **no default** anywhere (module, job builder and config all refuse
`lam_content != 0` without it).  Which layer of a 1-2 layer convolutional generator is Liu's
"intermediate layer" is an experimental constant, not an implementation detail: tap 0 (`post_bn`,
the normalized input, 1024-d), tap 1 (`pre_conv`, after the residual input projection, 1024-d) or
tap 2 (`mid`, the hidden 512-d activation, `n_layers = 2` only).  With the standing
`n_layers = 1` recognizer only taps 0 and 1 exist.

Also left as a documented default, flippable at config level: global standardization of the MFCCs
before k-means (`standardize_features=True`).

## Checks run

* `speech_llm.sae.emc.test_content_term` -- PASS ("ALL emc.content_term TESTS PASSED"):
  CE = `-log p` at a one-hot (uniform -> `log K`, near-one-hot -> 0, random vs hand-computed);
  padding excluded from loss AND accuracy (garbage padded logits/codes change nothing);
  out-of-range valid code refused; fp16 logits -> float32 CE within 2e-2; gradient only through
  valid frames; flat (zeroed) head -> CE == `log K` to 1e-5; head-off recognizer bit-identical
  (both tests above); mid-tap refused at `n_layers = 1` and at `stride = 2`; half-stated head refused.
* `speech_llm.sae.emc.test_mfcc_codes` -- PASS ("ALL emc.mfcc_codes TESTS PASSED"):
  the frame rule against the layer-by-layer conv rule and the frame geometry; the delta regression on
  constant and ramp signals; **3 synthetic utterances end to end** through `mfcc_frames`, `mfcc_dump`
  and `MfccFeatureJob.run`/`collect` (1 s -> 49 frames, shapes `[T, 39]`, deterministic);
  **k-means determinism at seed 42** (same centroids twice, different at another seed) and
  `assign_codes` == sklearn's own `predict` on the same fit (memory
  `test-the-call-into-shared-primitives`); `MfccKMeansJob` + `MfccCodesHdfJob` end to end on a real
  RETURNN feature HDF -- codes equal `assign_codes` frame for frame -- and the tolerance-0 length
  mismatch path REFUSES rather than truncating.
* Regression of the neighbours that touch the recognizer: `test_recognizer`, `test_lattice`,
  `test_bt_probe`, `test_init_jobs`, `test_eval_jobs`, `test_rate_term` -- all PASS.
* **Hash census** (`scripts/sae_4a_cons_census.py`, BEFORE side produced from `git archive HEAD`):
  `config_sae_4a_s3_v1` **98 job ids identical**, `config/sae_4a_phase.py` **1021 job ids identical**
  (`diff` empty both graphs).
* The wiring diff below was applied to a scratch OVERLAY of the three off-limits files and exercised
  there (see "Checks of the proposed wiring").  Nothing was launched.

## The wiring I need -- UNIFIED DIFF, NOT APPLIED

`prefix_lm/model/train_steps/sae_emc.py`, `prefix_lm/model/definitions/sae_emc.py` and
`sae/emc/emc_train_jobs.py` belong to another implementer right now.  The diff below is against
their working tree AS IT STOOD when this report was written (commit `1f05ca9` + their then-current
edits), and is also saved beside this report as
`impl_content_term_2026-09-15.wiring.diff`.  **It no longer applies verbatim**: that implementer
edited `train_steps/sae_emc.py`, `definitions/sae_emc.py`, `emc_train_jobs.py` and `rate_term.py`
again while this work finished, so `git apply -p1` from `recipe/2025-10-speech-llm/` now fails on the
first two files (context drift, not a conflict of intent -- every hunk of mine is an insertion).
Re-apply the hunks by hand, or `git apply -3`.  Container keys, as briefed:
`lam_content = 0.0` (default), `content_k = 64`, `content_layer = <index>`, extern_data key
`content_codes`; loss = existing + `lam_content * CE`; monitors `emc/content_ce` and
`emc/content_acc` marked ONLY when `lam_content != 0`; **every key omitted at the default**, so no
existing hash moves.

Two consequences I found that the brief did not name, both in the diff:

1. `definitions/sae_emc._load_state` refuses MISSING keys, and no banked init (flat or CTC) contains
   a `content_head`.  It gains `allow_missing_prefix="content_head."`, used only when the head
   exists; every other missing key and every unexpected key stay hard errors.
2. The eval path: `emc_train_jobs.subepoch_reads` -> `eval_jobs.posterior_dump` builds a BARE
   `ConvRecognizer(**RECOGNIZER_NET_ARGS)`, while `ExtractSubmoduleCheckpointJob` slices
   `recognizer.*` and would hand it `content_head.*` as unexpected keys.  `subepoch_reads` gains a
   `net_args` passthrough (`posterior_dump` already has one); at `None` every existing read is
   unchanged.

### Checks of the proposed wiring (overlay, not applied to the real files)

* `py_compile` of all three patched files: OK.
* Against the overlay: `test_emc_train_jobs`, `test_consistency`, `test_rate_term` and
  `prefix_lm.model.train_steps.test_sae_emc` -- all PASS (the default-OFF path is unchanged).
* Positive smoke (scratch, not committed): one real `train_step` with `lam_content = 1.0` at taps 0
  and 1 -- `content` loss marked, `emc/content_ce` and `emc/content_acc` reported, gradient reaches
  `content_head.weight`; with `lam_content = 0` neither the loss nor the monitors appear at all.
  `build_emc_train_config` ON declares `extern_data["content_codes"]` and the third MetaDataset
  stream (`data_map: content_codes -> ("content", "data")`), writes `lam_content` and
  `recognizer_kwargs.content_k/content_layer`, and refuses: a scale without `content_layer`, a scale
  without the code HDFs, and code HDFs without the scale.  OFF writes none of it.
* Census with the overlay in front (`SAE_CENSUS_SRC`): `s3` **98** and `phase` **1021** ids identical
  to the live tree -- the proposed wiring is hash-neutral too.

```diff
--- a/src/speech_llm/prefix_lm/model/train_steps/sae_emc.py
+++ b/src/speech_llm/prefix_lm/model/train_steps/sae_emc.py
@@ -21,6 +21,17 @@
 columns to every other arm's ``learning_rates`` file, so ``lam_rate = 0`` is the loop exactly as it
 has always run, step for step.
 
+``lam_content != 0`` adds the LABEL-FREE forward content term of SAE_4A.md "S3b-F" (Liu et al.
+2022, wav2vec-U 2.0, eq. 5; ``emc.content_term``): the per-frame cross-entropy of a LINEAR softmax
+head, read off an intermediate layer of the recognizer, against the MFCC k-means code of the same
+frame (``emc.mfcc_codes``, K = 64 -- Liu's Table 2 measures 64 as the optimum and 128 as WORSE than
+no term).  The codes are computed from the audio alone; no transcript, alignment or gold quantity
+enters them.  The head exists only when the recognizer was built with ``content_k`` /
+``content_layer`` (``recognizer.py``: default OFF, and off it is the recognizer bit for bit), and
+the gradient reaches theta through the tap.  ``emc/content_ce`` and ``emc/content_acc`` are
+MONITORS: the take-off of this stage is read on the phone rate, the vocabulary usage and the
+derangement margin, NEVER on L_tau and never on the content CE (design review 2026-09-15, F2).
+
 ``lam_cons != 0`` adds the consistency term of SAE_4A.md "S3b-C" (``emc.consistency``): the
 per-frame KL(p_clean || p_aug) of the recognizer's UNTEMPERED posterior against its posterior on an
 augmented view of the same utterance, with p_clean detached (the teacher) so the gradient flows
@@ -83,6 +94,7 @@
     step_generator,
     warp_posterior,
 )
+from speech_llm.sae.emc.content_term import CONTENT_CODES_KEY, content_cross_entropy
 from speech_llm.sae.emc.lattice import build_segment_table, lattice_loss
 from speech_llm.sae.emc.rate_term import (
     DEFAULT_FD_BATCHED,
@@ -119,6 +131,15 @@
 ]
 
 
+# Reported ONLY when the content term is on (see the module docstring): the per-frame CE the term
+# optimizes and the frame accuracy against the MFCC codes.  Both are monitors, never a gate -- the
+# baseline the accuracy has to beat is the largest code's share, printed by MfccKMeansJob.
+CONTENT_MONITOR_KEYS = [
+    "emc/content_ce",
+    "emc/content_acc",
+]
+
+
 def cons_monitor_keys(views) -> List[str]:
     """Reported ONLY when the consistency term is on: one KL per VIEW, plus its frame count.
 
@@ -180,7 +201,15 @@
     temperature = model.temperature(epoch)
     anchor_weight = model.anchor_weight(epoch)
 
-    log_q = model.recognizer(feats, feat_lens)
+    # ONE forward.  With the content term on, the same pass also returns the head's logits off the
+    # intermediate tap, so the term costs a linear layer and nothing else; with it off this is
+    # ``model.recognizer(feats, feat_lens)``, the call it has always been.
+    lam_content = float(getattr(model, "lam_content", 0.0))
+    content_logits = None
+    if lam_content:
+        log_q, content_logits = model.recognizer.forward_with_content(feats, feat_lens)
+    else:
+        log_q = model.recognizer(feats, feat_lens)
     assert log_q.shape[:2] == feats.shape[:2], (
         f"the recognizer returned {log_q.shape[1]} frames for {feats.shape[1]} input frames; the "
         "lattice couples frame t of the recognizer to the reverse clock, so the recognizer must run "
@@ -189,6 +218,9 @@
     )
     if model.freeze_recognizer:
         log_q = log_q.detach()  # phi warm-up: theta must not move by any route
+        if content_logits is not None:
+            # the head is part of theta: the warm-up sub-epoch must not move it either
+            content_logits = content_logits.detach()
     log_q_init = None
     if anchor_weight or model.lam_selfdistill:
         if model.q_init is None:
@@ -352,6 +384,47 @@
         cons_keys = cons_monitor_keys(aug_log_p)
         assert set(cons_keys) == set(cons_values), (cons_keys, sorted(cons_values))
 
+    # -- the S3b-F content term (SAE_4A.md "S3b-F"), off by default ------------------------------
+    # ``getattr`` once more: a model built before the knob existed stays on exactly its old path.
+    # The target is a THIRD extern_data stream of int codes at the SAME 50 Hz clock as the features
+    # (emc.mfcc_codes writes it on the feature HDFs' own frame counts, asserted equal at tolerance
+    # 0), so the equality below is a re-assertion of that, not a new rule.
+    content_values = None
+    if lam_content:
+        if content_logits is None:
+            raise ValueError(
+                "lam_content != 0 but the recognizer has no content head; the model must be built "
+                "with recognizer_kwargs content_k / content_layer (recognizer.CONTENT_TAPS)"
+            )
+        codes_key = str(getattr(model, "content_codes_key", CONTENT_CODES_KEY))
+        if codes_key not in extern_data.data:
+            raise ValueError(
+                f"lam_content != 0 needs extern_data[{codes_key!r}], the label-free MFCC k-means "
+                "code of every frame (emc.mfcc_codes.MfccCodesHdfJob); the job that trains this "
+                "arm must declare it"
+            )
+        codes_ = extern_data[codes_key]
+        codes = codes_.raw_tensor.long()
+        code_lens = _lens(codes_)
+        if not torch.equal(code_lens, feat_lens):
+            bad = (code_lens != feat_lens).nonzero().flatten().tolist()[:4]
+            raise ValueError(
+                f"the content codes and the features disagree on the frame count of "
+                f"{int((code_lens != feat_lens).sum())} utterance(s) (rows {bad}: "
+                f"{code_lens[bad].tolist()} vs {feat_lens[bad].tolist()}). The codes are written on "
+                "the feature HDFs' own frame counts; there is no resampling rule."
+            )
+        content_ce, content_stats = content_cross_entropy(
+            logits=content_logits, codes=codes, lens=feat_lens,
+            n_classes=model.recognizer.content_k,
+        )
+        ctx.mark_as_loss(content_ce, "content", scale=lam_content, use_normalized_loss=False)
+        with torch.no_grad():
+            content_values = {
+                "emc/content_ce": float(content_stats.ce),
+                "emc/content_acc": float(content_stats.acc),
+            }
+
     # --- monitors. Every quantity is a per-arm magnitude in its own units (memory:
     # magnitudes-are-per-arm); only the rate is comparable to the text side.
     with torch.no_grad():
@@ -384,10 +457,13 @@
         values.update(rate_values)
     if cons_values is not None:
         values.update(cons_values)
+    if content_values is not None:
+        values.update(content_values)
     for key in (
         MONITOR_KEYS
         + (RATE_MONITOR_KEYS if rate_values is not None else [])
         + cons_keys
+        + (CONTENT_MONITOR_KEYS if content_values is not None else [])
     ):
         ctx.mark_as_loss(
             torch.tensor([values[key]], device=device), key.replace("/", "_"),
--- a/src/speech_llm/prefix_lm/model/definitions/sae_emc.py
+++ b/src/speech_llm/prefix_lm/model/definitions/sae_emc.py
@@ -54,7 +54,7 @@
 import torch
 from torch import nn
 
-from speech_llm.sae.emc import consistency
+from speech_llm.sae.emc import consistency, content_term
 from speech_llm.sae.emc.agg import AggConfig, AggLoss, text_target_counts
 from speech_llm.sae.emc.lattice import LatticeConfig, _use_matmul, build_prior_history
 from speech_llm.sae.emc.reverse import ReverseConfig, SegmentalReverseModel
@@ -81,14 +81,28 @@
     return float(seq[min(max(int(epoch), 1), len(seq)) - 1])
 
 
-def _load_state(module: nn.Module, path: str, *, name: str) -> None:
-    """Load a standalone checkpoint into ``module`` and refuse a silent miss."""
+def _load_state(
+    module: nn.Module, path: str, *, name: str, allow_missing_prefix: Optional[str] = None
+) -> None:
+    """Load a standalone checkpoint into ``module`` and refuse a silent miss.
+
+    ``allow_missing_prefix`` is the ONE exception: the S3b-F content head (``content_head.``) is a
+    randomly initialized readout that no banked init contains, so loading the flat / CTC init into a
+    head-ON recognizer must leave exactly those keys at their fresh values.  Every other missing key
+    -- and every unexpected key, always -- is still a hard error.
+    """
     state = torch.load(path, map_location="cpu", weights_only=False)
     sd = state["model"] if isinstance(state, dict) and "model" in state else state
     res = module.load_state_dict(sd, strict=False)
-    if res.missing_keys:
-        raise RuntimeError(f"{name}: {len(res.missing_keys)} key(s) not loaded from {path}, "
-                           f"e.g. {res.missing_keys[:4]}")
+    missing = list(res.missing_keys)
+    if allow_missing_prefix:
+        fresh = [k for k in missing if k.startswith(allow_missing_prefix)]
+        missing = [k for k in missing if not k.startswith(allow_missing_prefix)]
+        if fresh:
+            print(f"{name}: {len(fresh)} key(s) left at their fresh init: {sorted(fresh)}", flush=True)
+    if missing:
+        raise RuntimeError(f"{name}: {len(missing)} key(s) not loaded from {path}, "
+                           f"e.g. {missing[:4]}")
     if res.unexpected_keys:
         raise RuntimeError(f"{name}: {len(res.unexpected_keys)} unexpected key(s) in {path}, "
                            f"e.g. {res.unexpected_keys[:4]}")
@@ -145,6 +159,14 @@
         # the speed-perturbed L15 dump from ``cons_pert_features_key`` and warps its posterior back
         # onto the clean clock.  ``cons_specaug`` overrides the documented SpecAugment policy
         # (emc.consistency.SpecAugmentOpts); None = that policy, which is what the S3b-C arms run.
+        # S3b-F, the label-free forward content term (Liu et al. 2022 eq. 5; emc/content_term.py).
+        # ``lam_content`` defaults to 0, at which the train step marks neither the loss nor its
+        # monitors, and ``build_emc_train_config`` writes none of these keys.  Turning it on ALSO
+        # needs the recognizer's tap and head, i.e. ``recognizer_kwargs`` with ``content_k`` and
+        # ``content_layer`` (recognizer.CONTENT_TAPS), and the third extern_data stream named by
+        # ``content_codes_key``.
+        lam_content: float = 0.0,
+        content_codes_key: str = content_term.CONTENT_CODES_KEY,
         lam_cons: float = 0.0,
         cons_views: Sequence[str] = consistency.DEFAULT_VIEWS,
         cons_specaug: Optional[Dict] = None,
@@ -252,7 +274,14 @@
                 f"T = 704). Set lattice_checkpoint = S > 0, the D4 frame stride."
             )
         if recognizer_checkpoint_path:
-            _load_state(self.recognizer, recognizer_checkpoint_path, name="recognizer")
+            _load_state(
+                self.recognizer, recognizer_checkpoint_path, name="recognizer",
+                # the content head is not in any banked init; everything else must be there
+                allow_missing_prefix=(
+                    "content_head." if getattr(self.recognizer, "content_head", None) is not None
+                    else None
+                ),
+            )
         if reverse_checkpoint_path:
             _load_state(self.reverse, reverse_checkpoint_path, name="reverse")
 
@@ -309,6 +338,24 @@
                 raise ValueError("rate_fd_eps is the finite-difference step in b, in nats, > 0")
         # S3b-C.  The views and the mask policy are validated HERE, at job start: an unknown view or
         # a mistyped SpecAugment key must not reach the first step of a queued run.
+        # S3b-F.  Refused HERE, at job start: a scale without a head is a silent no-op, and a head
+        # without a scale is an untrained readout that still changes the parameter count.
+        self.lam_content = float(lam_content)
+        self.content_codes_key = str(content_codes_key)
+        # ``getattr``: a recognizer built before the tap existed (and the stub of the train-step
+        # tests) has no such attribute at all, and must stay on exactly its old path.
+        content_head = getattr(self.recognizer, "content_head", None)
+        if self.lam_content and content_head is None:
+            raise ValueError(
+                "lam_content != 0 needs the recognizer's content head: pass recognizer_kwargs "
+                "content_k (64, emc.content_term.CONTENT_K) and content_layer "
+                "(emc.recognizer.CONTENT_TAPS)"
+            )
+        if content_head is not None and not self.lam_content:
+            raise ValueError(
+                "the recognizer was built with a content head but lam_content = 0: the head would "
+                "be an untrained readout carried through every checkpoint"
+            )
         self.lam_cons = float(lam_cons)
         self.cons_views = tuple(cons_views)
         self.cons_pert_features_key = str(cons_pert_features_key)
--- a/src/speech_llm/sae/emc/emc_train_jobs.py
+++ b/src/speech_llm/sae/emc/emc_train_jobs.py
@@ -159,6 +159,23 @@
 # module is imported by the sisyphus manager, which has no torch, while ``consistency.py`` does;
 # ``test_consistency.test_the_job_builder_and_the_module_agree`` asserts the two strings are equal.
 EMC_CONS_PERT_KEY = "features_pert"
+# The scale of the LABEL-FREE forward CONTENT term (SAE_4A.md "S3b-F"; Liu et al. 2022 eq. 5; the
+# term is emc/content_term.py and its target is emc/mfcc_codes.py).  0.0 = off, which is every leg
+# up to S3b-C and is the model's own default, so ``build_emc_train_config`` writes none of its keys,
+# declares no third extern_data key and adds no third stream -- no existing config's hash moves.
+EMC_LAM_CONTENT = 0.0
+# K, the number of MFCC k-means codes.  Liu et al. 2022, Table 2 (unsupervised PER on LS dev-other):
+# no term 15.9 +- 1.1, K = 50 15.2, K = 64 13.6 +- 0.9, K = 100 14.8, K = 128 16.8 -- 64 is the
+# measured optimum and 128 is WORSE than no term, so this is a pinned constant, not a knob to sweep.
+EMC_CONTENT_K = 64
+# The recognizer layer the linear head reads (emc.recognizer.CONTENT_TAPS).  NO default: which tap
+# is Liu's "intermediate layer" in a 1-2 layer generator is a decision of the orchestrator, and a
+# silently chosen tap would be an unstated experimental constant.
+EMC_CONTENT_LAYER: Optional[int] = None
+# The extern_data / MetaDataset key of the code stream.  A literal here for the same reason
+# EMC_CONS_PERT_KEY is: this module is imported by the manager, which has no torch.
+# ``test_content_term`` asserts it equals ``content_term.CONTENT_CODES_KEY``.
+EMC_CONTENT_CODES_KEY = "content_codes"
 EMC_COUNT_EMA_DECAY = 0.99   # SAE_4A.md:104-105 (~100 steps of 64 utterances)
 EMC_BAND = 25                # W = 25 (SAE_4A.md:76)
 EMC_PRIOR_WEIGHT = 1.0       # beta = 1 (SAE_4A.md:134, 174)
@@ -505,6 +522,7 @@
     num_workers: int,
     buffer_seqs: int,
     pert_feature_hdfs: Optional[Sequence["tk.Path"]] = None,
+    content_code_hdfs: Optional[Sequence["tk.Path"]] = None,
 ) -> Dict[str, Any]:
     """A ``MetaDataset`` of (features, units), optionally wrapped in ``MultiProcDataset``.
 
@@ -538,6 +556,12 @@
     if pert_feature_hdfs:
         datasets["feats_pert"] = HDFDataset(files=list(pert_feature_hdfs))
         data_map[EMC_CONS_PERT_KEY] = ("feats_pert", "data")
+    if content_code_hdfs:
+        # The S3b-F code stream, looked up by tag exactly as the units are.  Its frame count EQUALS
+        # the clean one per utterance (MfccCodesHdfJob asserts it at tolerance 0 against these same
+        # feature HDFs), so it rides the features' clock and the train step re-asserts it.
+        datasets["content"] = HDFDataset(files=list(content_code_hdfs))
+        data_map[EMC_CONTENT_CODES_KEY] = ("content", "data")
     meta = MetaDataset(
         data_map=data_map,
         datasets=datasets,
@@ -588,6 +612,11 @@
     cons_specaug: Optional[Dict[str, Any]] = None,
     train_pert_feature_hdfs: Optional[Sequence["tk.Path"]] = None,
     dev_pert_feature_hdfs: Optional[Sequence["tk.Path"]] = None,
+    lam_content: float = EMC_LAM_CONTENT,
+    content_k: int = EMC_CONTENT_K,
+    content_layer: Optional[int] = EMC_CONTENT_LAYER,
+    train_content_code_hdfs: Optional[Sequence["tk.Path"]] = None,
+    dev_content_code_hdfs: Optional[Sequence["tk.Path"]] = None,
     count_ema_decay: float = EMC_COUNT_EMA_DECAY,
     band: int = EMC_BAND,
     prior_weight: float = EMC_PRIOR_WEIGHT,
@@ -710,6 +739,34 @@
             "perturbed feature hdfs were passed but no arm reads them: without the 'speed' view "
             "they would only be loaded and thrown away"
         )
+    # HASH-NEUTRAL BY OMISSION, a third time: with the content term OFF (the default) no key of it
+    # reaches ``model_args`` or ``recognizer_kwargs``, no third extern_data key is declared and no
+    # third stream is added, so every config built before it existed hashes exactly as it did.  With
+    # it on, the scale, the head and the codes are written TOGETHER: a scale without the head is a
+    # silent no-op, a head without codes cannot be trained, and the CV pass runs this same step.
+    if float(lam_content) != EMC_LAM_CONTENT:
+        assert content_layer is not None, (
+            "lam_content != 0 needs content_layer, the recognizer tap the linear head reads "
+            "(emc.recognizer.CONTENT_TAPS); there is no default -- the orchestrator states it"
+        )
+        assert int(content_k) > 1, content_k
+        assert train_content_code_hdfs and dev_content_code_hdfs, (
+            "the content term reads the label-free MFCC k-means codes of the SAME utterances on "
+            "BOTH datasets (the CV pass runs this train step too); pass train_content_code_hdfs / "
+            "dev_content_code_hdfs (emc.mfcc_codes.MfccCodesHdfJob)"
+        )
+        model_args["lam_content"] = float(lam_content)
+        # The head is part of theta's SHAPE, so it rides ``recognizer_kwargs`` -- which is also
+        # what ``subepoch_reads`` must hand the posterior dump for this arm.
+        recognizer_kwargs = dict(recognizer_kwargs)
+        recognizer_kwargs["content_k"] = int(content_k)
+        recognizer_kwargs["content_layer"] = int(content_layer)
+        model_args["recognizer_kwargs"] = recognizer_kwargs
+    else:
+        assert not train_content_code_hdfs and not dev_content_code_hdfs, (
+            "content code hdfs were passed but lam_content = 0: they would only be loaded and "
+            "thrown away"
+        )
     if reverse_kwargs:
         model_args["reverse_kwargs"] = dict(reverse_kwargs)
     if recognizer_checkpoint is not None:
@@ -737,12 +794,18 @@
             "shape": (None, recognizer_kwargs["in_dim"]),
             "dtype": "float16",
         }
+    if float(lam_content) != EMC_LAM_CONTENT:
+        # The S3b-F code stream: one sparse int code per frame, on the features' own clock.
+        extern_data[EMC_CONTENT_CODES_KEY] = {
+            "dim": int(content_k), "shape": (None,), "sparse": True, "dtype": "int32",
+        }
 
     train = _emc_dataset(
         feature_hdfs=train_feature_hdfs, units_hdfs=train_units_hdfs,
         partition_epoch=partition_epoch, seq_ordering="laplace:.1000", segments=train_segments,
         num_workers=num_workers, buffer_seqs=buffer_seqs,
         pert_feature_hdfs=train_pert_feature_hdfs,
+        content_code_hdfs=train_content_code_hdfs,
     )
     # The CV pass is a few hundred utterances once per sub-epoch: no worker processes, they would
     # cost more to spin up than the pass itself.
@@ -751,6 +814,7 @@
         partition_epoch=None, seq_ordering="sorted", segments=dev_segments,
         num_workers=0, buffer_seqs=0,
         pert_feature_hdfs=dev_pert_feature_hdfs,
+        content_code_hdfs=dev_content_code_hdfs,
     )
 
     config = {
@@ -1420,6 +1484,7 @@
     sctk_binary_path: Optional["tk.Path"] = None,
     baseline_sclite: Optional[Dict[str, Any]] = None,
     derangement: Optional[Dict[str, Any]] = None,
+    net_args: Optional[Dict[str, Any]] = None,
 ) -> Dict[str, Any]:
     """Build every read of ONE checkpoint.  Returns a dict of jobs; the CONFIG registers them.
 
@@ -1428,6 +1493,14 @@
     sclite word WER on each split, plus -- when ``baseline_sclite`` names the init (i) sclite job
     per split -- the paired WER delta against it.  With ``derangement``: the live-phi derangement
     gap (it needs the leg's REVERSE slice, so it is built here too).
+
+    ``net_args`` is the recognizer shape the theta slice must be loaded into.  It matters for
+    exactly one arm family: with the S3b-F content term on, the training checkpoint carries
+    ``recognizer.content_head.*``, the slice carries ``content_head.*``, and a BARE
+    ``ConvRecognizer(**RECOGNIZER_NET_ARGS)`` would refuse them as unexpected keys.  Such an arm
+    passes ``RECOGNIZER_NET_ARGS | {"content_k": ..., "content_layer": ...}`` here.  At ``None``
+    -- every arm up to S3b-C -- ``posterior_dump`` uses RECOGNIZER_NET_ARGS exactly as before, so
+    no existing read's hash moves (the head is unused by the forward, which returns log-probs only).
     """
     from speech_llm.sae.emc import eval_jobs, feature_dump
 
@@ -1442,6 +1515,7 @@
             feature_hdfs=feats_dev_hdfs[split],
             returnn_exe=returnn_exe,
             returnn_root=returnn_root,
+            net_args=net_args,
         )
         out[f"post_{split}"] = post
         hdf = post.out_files["posteriors.hdf"]
```

## What a config that turns it on must state

```python
build_emc_train_config(
    ...,
    lam_content=<scale>,          # e.g. 1.0; Liu's L_ss has no separate weight, it is summed in
    content_k=64,                 # pinned by Table 2, not a sweep knob
    content_layer=<index>,        # UNDETERMINED -- see above
    train_content_code_hdfs=<MfccCodesHdfJob.out_hdfs["train"]>,
    dev_content_code_hdfs=<MfccCodesHdfJob.out_hdfs["cv"]>,
)
# and, for the reads of that arm:
subepoch_reads(..., net_args={**RECOGNIZER_NET_ARGS, "content_k": 64, "content_layer": <index>})
```
