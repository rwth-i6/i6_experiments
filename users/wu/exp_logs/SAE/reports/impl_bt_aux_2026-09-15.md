# S3b-BT-aux: back translation as an auxiliary INSIDE the EMC training (implementation report)

Date 2026-09-15. Setup `/e/project1/spell/wu24/2026-07-13_unsupervised`, checkout
`recipe/2025-10-speech-llm` on `haotian_modality_matching_jupiter`, base commit `fb97abb`.
Status DONE. Nothing launched; `ACTIVE_ARMS` is unchanged (`("lam3",)`).

Commits (both stage explicit paths, no push):

* stage 1 `d40f708` -- `src/speech_llm/sae/emc/bt_aux.py` (new), `src/speech_llm/sae/emc/test_bt_aux.py`
  (new), `src/speech_llm/sae/emc/reverse.py` (insertion-only `unit_draw` argument).
* stage 2 `325e47c` -- the wiring, the held arms and their tests (files listed below).

## 1. What the term is

One BT step is: draw 128 sentences from T_phi (`TextToPhonemeJob.THKMON3k9LJQ/output/phon.txt.gz`,
one flat phone sequence per line, no word boundaries); under `no_grad`, run the LIVE reverse model
phi on them with a drawn speaker vector to get one codebook unit per phone by ARGMAX of phi's own
emission distribution and a SAMPLED duration per phone; collage a synthetic L15 stream out of REAL
frames of those same units; run the recognizer theta on it with the batch-norm running statistics
frozen; take CTC (blank 0, target = phone id + 1) against the phone sequence the sentence already
is. The loss is `per_utt.sum() / out_lens.sum()`.

LABEL-FREE by construction. The text is the unpaired corpus, never a transcript of the audio; the
audio side and the text side are never paired. `test_bt_aux` asserts by `inspect.signature` that no
function on the BT path takes a transcript / gold / alignment / reference argument.

phi RECEIVES NO GRADIENT from this pass: the units, durations and the speaker draw are produced
inside `torch.no_grad()` from the live phi, and the batch is detached before theta sees it. The
test asserts every phi parameter's `.grad is None` after a BT backward.

## 2. The alternation: two optimizer steps, never one graph

`accum_grad_multiple_step: 1`, so one `train_step` call is one optimizer step. The BT pass is a
different forward on different data, so it is a SEPARATE step:

* `build_emc_train_config` writes `config["torch_batching"] = CodeWrapper("bt_interleaved_batching")`
  and a `PartialImport` of `bt_aux.bt_interleaved_batching` carrying this config's own
  `batch_size` / `max_seqs` / `lam_bt` / `bt_ramp_epochs`. RETURNN calls a `torch_batching` callable
  as `cls(dataset, train=..., **fwd_compat)` and does NOT pass batch_size/max_seqs
  (`returnn/torch/data/pipeline.py:431-457`), which is why those ride the partial and cannot
  disagree with the config dict.
* `train = False` (the CV pass) gets RETURNN's plain `BatchingIterDataPipe`. The CV pass is
  untouched, so `dev_loss_emc` -- the unsupervised selection's own quantity -- still means what it
  meant.
* `train = True` wraps that same inner pipe: it yields an EMC batch, then ONE MARKER batch, then the
  next EMC batch. The EMC batches, their contents and their order are bit-identical to the plain
  pipe's. The marker is one sequence carrying a reserved seq tag (`__bt_aux_step__`) plus a
  length-1 zero array for EVERY declared `extern_data` key, because
  `raw_dict_to_extern_data` requires every template key (`torch/data/extern_data.py:53`); its meta
  fields (`seq_idx`, `epoch`, `num_seqs`, `complete_frac`) are copied from the batch it follows.
* `emc_train_jobs.train_step` tests the tag with a LOCAL literal (`EMC_BT_STEP_SEQ_TAG`) and imports
  `bt_aux` only after it matches, so a `lam_bt = 0` job -- including the arms training right now,
  which re-import the recipe tree on every resubmit -- never loads the module. On a match it calls
  `bt_aux.bt_train_step` and RETURNS: that step's run context holds the BT loss and its monitors and
  nothing of the EMC objective. On an EMC batch the BT loss does not exist at all. There is no
  summed loss, no shared graph and no gradient accumulation across the two.

STEP COUNT. The EMC steps per sub-epoch are unchanged; the BT steps are added beside them, so the
total RETURNN step count roughly DOUBLES from sub-epoch 2 on. The `learning_rates` schedule is
per SUB-EPOCH, so the learning rate seen by an EMC step is unchanged. `emc_sec_per_step` is measured
between consecutive entries of the EMC branch and therefore SPANS an EMC/BT PAIR once the ramp
opens; `emc_utts_per_sec` stays a statement about EMC batches only. Both are stated in
`train_step`'s docstring.

RAMP. The weight is ramped 0 -> lam_bt over sub-epochs 1-4 and held from 5
(`bt_aux.lam_bt_effective` delegates to `entropy_term.lam_ent_effective`). It is folded into the
LOSS VALUE and marked at `scale = 1.0`, because a BT step has no other optimized loss and RETURNN's
`total_loss()` SKIPS `scale == 0.0` losses (`frontend/run_ctx.py:416-425`), which would leave
`engine.py:505` with a Python float to call `.backward()` on. Belt and braces: while the ramped
weight is 0 the pipe emits NO marker batch at all, so sub-epoch 1 costs nothing.

## 3. Depth

`bt_depth = "full"` -- no detach, the whole recognizer gets the BT gradient.
`bt_depth = "output_only"` -- a forward pre-hook on `theta.out_logit_layer` detaches its input, i.e.
the content tap (the conv's input, after BN + the residual `in_proj` + dropout). At the standing
`n_layers = 1` `out_logit_layer` IS the output projection, so only the final conv is trained by the
BT step; `in_proj`, the BN affine parameters and everything below get exactly zero BT gradient. The
container REFUSES `output_only` at `n_layers != 1`, where that tap would be the 1x1 head above the
hidden conv and not the level the stage pre-registered. Test: under `output_only` every non-conv
parameter's gradient is zero and the conv's is not (conv 1.283e+03, everything else 0); under
`full` `in_proj` gets 1.969e+02. Batch-norm running statistics are bit-identical before and after a
BT forward.

## 4. Monitors

`emc/bt_ctc`, `emc/bt_synth_phones_per_s` (the BT batch's target phones per second of SYNTHESIZED
frames at the lattice's 50 Hz clock), `emc/bt_distinct_units` (distinct codebook units in the
batch), `emc/lam_bt_effective`. All four are `as_error=True`, so none enters the optimized total.

## 5. Frame pool -- a DISCLOSED DEVIATION from the probe

The probe (`bt_probe.py`) builds a FIXED collage pool of real L15 frames per unit over 2000 tc100
utterances. At the stage's shape that pool is ~2000 x ~635 frames x 1024 dims x fp16 = ~2.4 GiB,
past the coordinator's 2 GiB budget, and feeding it would add an HDF input the lam3-identical
datasets do not have (which would move the arm's hash for a reason unrelated to the term). So the
training-time pool is a ROLLING per-unit ring buffer (`bt_aux.FramePool`, `BT_POOL_FRAMES_PER_UNIT
= 64` frames per unit, fp16, `BT_POOL_INGEST_FRAMES = 2048` frames ingested per EMC step), filled
under `no_grad` from the EMC steps' OWN batches after each EMC step. A unit never yet seen falls
back to the pool's mean frame and is flagged. Consequences to read with the arms: the pool is
non-stationary early in a sub-epoch and its content depends on the seq order, so the BT batch is
reproducible only given the same data order (the draw itself is seeded per step through
`consistency.step_generator`).

The sentence pool is a seeded reservoir of `BT_SENTENCE_POOL = 100,000` sentences over the WHOLE
T_phi (39,630,169 lines), length-filtered to the probe's 20-200 phones, read once per process,
lazily, and cached (measured cost of the project's own reader: 4.71 s / 2 M lines, i.e. ~95 s once).

## 6. Hash neutrality

Everything is HASH-NEUTRAL BY OMISSION. At `lam_bt = 0` (the model's own default, every leg up to
S3b-F) none of the five `model_args` keys is written, no `torch_batching` entry is written and the
data pipeline is RETURNN's own -- so a job without the auxiliary cannot even PRODUCE a BT batch.
With it on, the five keys are written together (a scale without its corpus has nothing to translate;
without its ramp length it does not say what ran at sub-epoch 2; without its depth it does not say
which parameters it trained).

CENSUS (before = `git archive fb97abb src`, after = the working tree, script
`scripts/sae_4a_cons_census.py`):

| graph | job ids before | job ids after | diff |
|---|---|---|---|
| s3    | 98   | 98   | none |
| phase | 1021 | 1021 | none |
| rate  | 91   | 91   | none |

`ReturnnTrainingJob.DF6blPpto23t` (the launched `lam3` arm) is present and unmoved. The only
after-side change in the `rate` census is three new COMMENT lines reporting `bt_a` / `bt_b` /
`bt_c` as "not built".

BUILD CHECK: with `ACTIVE_ARMS = ("lam3", "bt_a", "bt_b", "bt_c")` in a scratch copy of `src`, the
graph builds and the three arms hash to `ReturnnTrainingJob.6G8CpcswqlTR` (bt_a),
`NSxRSjyBWp5g` (bt_b), `UjJpp4iQ4kkT` (bt_c); 192 jobs are ADDED and none removed, and `lam3` is
still `DF6blPpto23t`. That scratch copy was discarded; the live `ACTIVE_ARMS` is `("lam3",)`.

## 7. The three held arms

In `config_sae_4a_s3b_rate_v1.py`, all at `lam_rate = 3` and otherwise identical to `lam3`:

| tag | lam_bt | bt_depth | what it is for |
|---|---|---|---|
| `bt_a` | 0.1 | full        | the low dose |
| `bt_b` | 0.3 | full        | three times the dose -- the dose axis |
| `bt_c` | 0.3 | output_only | the control: is the gain a re-shaped representation or only a re-calibrated output layer? |

`bt_ramp_epochs = 4` and `bt_batch_sents = 128` stay at `emc_train_jobs`' documented defaults.
Their outputs land under `<base>/librispeech/sae_4a_s3b_bt/<tag>/ep<k>/<split>/`; the rate arms'
prefix is untouched. `BT_TEXT_PHN` reuses `hash_overwrite="sae1g_complete_tphi"`, the string
`config_sae_1g_h4_calibration_v1` already uses for that same artifact.

## 8. Files touched (stage 2)

* `src/speech_llm/prefix_lm/model/definitions/sae_emc.py` -- five container keys
  (`lam_bt`, `bt_depth`, `bt_ramp_epochs`, `bt_batch_sents`, `bt_text_path`) with off-default
  validation at job start (weight >= 0, known depth, ramp >= 1, sents >= 1, text path required and
  existing, `output_only` only at `n_layers = 1`, text without weight refused). The defaults are
  LITERALS, not imports of `bt_aux`: this module is re-imported live by every running job.
* `src/speech_llm/sae/emc/emc_train_jobs.py` -- the five constants and the reserved tag; the marker
  dispatch and the pool update in `train_step`; the optional BT `PartialImport` in `_serializer`;
  the builder arguments and the hash-neutral block writing the keys plus `torch_batching`.
* `.../librispeech/configs/config_sae_4a_s3b_rate_v1.py` -- `BT_TEXT_PHN`, `BT_ARMS`, `BT_PREFIX` /
  `BT_ARM_PREFIX`, the three tags in `LAM_RATE_ARMS`, the per-arm kwargs in the build loop.
* `src/speech_llm/sae/emc/test_emc_train_jobs.py` -- the builder's hash-neutrality test.
* `src/speech_llm/prefix_lm/model/train_steps/test_sae_emc.py` -- the two end-to-end wiring tests.

## 9. Checks run

* `speech_llm.sae.emc.test_bt_aux` -- 7 tests, all pass (conventions/ramp; text provenance, seeded
  reservoir and no gold argument; collage frames are real frames of their own unit, unseen-unit
  fallback, ring wrap; batch determinism, argmax == phi's own emission argmax, phi ungradiented;
  depth and frozen BN stats; the CTC convention; the interleaving pipe incl. sub-epoch 1 and CV).
* `speech_llm.sae.emc.test_emc_train_jobs` -- all pass, incl. the new
  `test_the_bt_auxiliary_is_hash_neutral_by_omission` (no key and no `torch_batching` at the
  default, five keys plus the interleaving partial when on, five refused mis-statements).
* `speech_llm.prefix_lm.model.train_steps.test_sae_emc` -- 11 tests, all pass, incl. the two new
  ones: at `lam_bt = 0` the wrapper's 18 losses/monitors are identical bit for bit between a model
  built without the keys and one built with them at their defaults (apart from the two wall-clock
  scalars), there is no `bt` key, no `emc_bt*` monitor and no frame pool, and every off-default
  mis-statement is refused at construction; at `lam_bt = 0.3` the EMC batch reports `l_tau` and no
  `bt`, the marker batch reports `bt` + the four monitors and NO EMC term (`optimized == ["bt"]`,
  scale 1.0), the BT backward gives theta a gradient (6.722e+02) and phi none, and at sub-epoch 1
  the effective weight and the loss value are both 0.
* Regression: `test_reverse`, `test_bt_probe`, `test_consistency`, `test_content_term` all pass.
* Census as in section 6; scratch build check as in section 6.

## 10. Not done

* GPU TIMING of a BT step against an EMC step was SKIPPED, as the coordinator allowed: it needs a
  GPU allocation and a run of the real shapes, which is more than the remaining budget. The
  expectation from the code alone -- the BT step has NO DP pass, it is one recognizer
  forward/backward on 128 synthetic utterances, against an EMC step that pays 3 DP calls at
  `rate_fd_mode = "central"` -- is that the BT step is much the cheaper of the two, but that is an
  expectation and not a measurement. It should be read off the first `bt_a` run's own log
  (`emc_sec_per_step` then spans a pair, so the BT cost is that value minus the lam3 arm's).
* The frame pool is the rolling one, not the probe's fixed pool (section 5).
