# Step 0b rerun: two longer-schedule neural phone LM fits (2026-09-20)

Implementer report.  Dispatch: add two more `NeuralPhoneLmTrainJob` instances and their two
`PriorGapAnalysisJob` reads to `config_sae_4a_prior_gap_v1`, leaving the banked Step 0b pair
untouched.

## What changed

* `recipe/2025-10-speech-llm/src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_prior_gap_v1.py`
  (workspace symlink `config/sae_4a_prior_gap.py`): a STEP 0B RERUN paragraph in the module
  docstring (why, what stays the same, the cosine/epoch tie, the GPU estimate); new constants
  `RERUN_EPOCHS = 30`, `ARCH_BASE`, `ARCH_L6W384`, `NEURAL_NAME_E30`, `NEURAL_NAME_L6W384_E30`; a
  private `_rerun_phone_lm(name, arch)` and the two public `neural_phone_lm_e30()` /
  `neural_phone_lm_l6w384_e30()` that construct the job with `n_layers`, `d_model`, `n_heads`,
  `d_ff` and `epochs` as explicit constructor arguments, add an alias
  (`sae/4a/prior_gap/<name>`) and register `model.pt`, `train_log.json`, `heldout.txt` under
  `<PREFIX>/<name>/`; `build()` adds the two train jobs and two `_analysis_job(..., neural_lm=
  <train>.out_model)` instances and returns all seven.  The existing `neural_phone_lm()` and the
  existing `_analysis_job(NEURAL_NAME, ...)` call are byte-identical; the only line removed in the
  whole diff is the old `return` of `build()`.
* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_neural_phone_lm.py`: one new test,
  `test_job_architecture_arguments_reach_the_built_model` (plus its module-docstring bullet (g)).

NO change was needed in `sae/emc/neural_phone_lm.py`: `NeuralPhoneLmTrainJob.__init__` already
takes `n_layers`, `d_model`, `n_heads`, `d_ff` and `epochs` as ordinary (hash-relevant, no
`__sis_hash_exclude__`) keyword arguments, so this is a config-level delta only and no existing
default moved.

## The arms

| arm | layers / width / heads / FFN | epochs | parameters | job |
| --- | --- | --- | --- | --- |
| banked | 4 / 256 / 4 / 1024 | 3 | 3,312,170 | `NeuralPhoneLmTrainJob.Iv6P6YVPNWmB` (finished) |
| (a) `neural_phone_lm_e30` | 4 / 256 / 4 / 1024 | 30 | 3,312,170 | `NeuralPhoneLmTrainJob.xObXEwRpvmzd` |
| (b) `neural_phone_lm_l6w384_e30` | 6 / 384 / 6 / 1536 | 30 | 10,876,458 | `NeuralPhoneLmTrainJob.pBozvj6c3l16` |

Head dim of (b) is 384 / 6 = 64, the banked 256 / 4.  Parameter counts are from
`n_parameters(build_model(job.config))` in the graph-load script, not an estimate.

Analysis instances (same inputs, window, KenLM orders, lexicon, version as the banked read; only
`neural_lm` differs):

* `ctrl_50_ep10/dev-other_neural_e30` -> `PriorGapAnalysisJob.5wNIQs2lpC5P`
* `ctrl_50_ep10/dev-other_neural_l6w384_e30` -> `PriorGapAnalysisJob.pg14aEYJyiva`

## Unchanged hashes (graph load from the setup dir)

`C.build()` under the setup's `settings.py` prints:

```
ctrl_50_ep10/dev-other                     PriorGapAnalysisJob.2RkbKYl0v1XK   (banked, unchanged)
neural_phone_lm                            NeuralPhoneLmTrainJob.Iv6P6YVPNWmB (banked, unchanged)
ctrl_50_ep10/dev-other_neural              PriorGapAnalysisJob.Gct95xZHe0zt   (banked, unchanged)
neural_phone_lm_e30                        NeuralPhoneLmTrainJob.xObXEwRpvmzd (new)
ctrl_50_ep10/dev-other_neural_e30          PriorGapAnalysisJob.5wNIQs2lpC5P   (new)
neural_phone_lm_l6w384_e30                 NeuralPhoneLmTrainJob.pBozvj6c3l16 (new)
ctrl_50_ep10/dev-other_neural_l6w384_e30   PriorGapAnalysisJob.pg14aEYJyiva   (new)
```

The three banked ids are the ones on disk (`work/speech_llm/sae/emc/neural_phone_lm/` and
`work/speech_llm/sae/emc/prior_gap/`); the four new ids have no job directory yet.  The script also
asserts each new analysis instance's `neural_lm` path is its own train job's `out_model`.

## What is held identical to the banked instance

Both arms pass only the architecture and `epochs`; every other argument stays at the job's default,
which is what the banked instance ran with: the `WINDOW_PHN` / `WORD_CORPUS` / `BLISS_LEXICON` /
`G2P_LEXICON` pins, the held split (`prior.HELD_STRIDE`, i % 101), 512 positions, dropout 0.1,
AdamW lr 5e-4 with 500 warmup steps, weight decay 0.01, grad clip 1.0, 32,768 tokens per batch,
sort window 8,192, seed 0, bf16 autocast, the 10,000-line train probe, the best-held-epoch
selection, the no-improvement abort (`held_improvement_nats` 1e-4) and `version=1`.

Caps: `rqmt {"gpu":1,"cpu":4,"mem":32,"time":4}` (the 4 h cap) and `max_seconds = 7200` (the 2 h
in-loop training cap) are both left at the defaults, i.e. at the banked instance's values -- the
dispatch's "same 4 h cap" and "everything else identical" both resolve to "do not touch them", so
no constant was chosen here.

Cosine: `train_phone_lm` builds one batch plan per planned epoch and sets `total_steps` to their
sum, and `cosine_lr` decays to zero at `total_steps`.  The schedule is therefore TIED to the epoch
count and stays tied: both arms decay to zero at the end of epoch 30.

## GPU time estimate

Banked run: 8,289 steps / 3 epochs in 124.8 s of training (43.0, 40.9, 40.9 s per epoch, i.e.
41 s/epoch) at 3.31 M parameters, plus 640.4 - 124.8 = 515.6 s (8.6 min) of corpus replay before
training.

* (a) same model: 30 x 41 s = 1,230 s (21 min) of training, ~29 min wall.
* (b) 10.88 M / 3.31 M = 3.28x the parameters; the attention term is negligible (81 tokens per line
  on average), so at parameter-proportional cost 3.28 x 41 = 135 s/epoch -> 4,040 s (67 min) of
  training, ~76 min wall.

Both are inside the 4 h `rqmt` (3.2x margin for (b)) and inside the 2 h in-loop training cap (1.8x
margin for (b)).  Caveat: the scaling is a parameter-count proportionality, not a measurement; the
banked 3.3 M model may have been partly launch-bound, in which case (b) is faster than this.  If
(b) were nevertheless more than ~1.8x slower than estimated, the in-loop cap would truncate it and
`train_log.json`'s `stopped_reason` would say so (the best epoch is still kept and saved).

## Checks run

* `pytest recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_neural_phone_lm.py` -- 13 passed
  (12 pre-existing + the new one).
* `pytest recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_prior_gap.py` -- 24 passed,
  including `test_the_banked_v2_instance_keeps_its_hash`, which loads this config.
* The graph-load script above (config imported and `build()` run from the setup dir) -- the seven
  ids printed, three banked assertions True, wiring assertion passed.

These check that the config loads, that the hashes are what they are and that the constructor
arguments reach the built model.  Nothing was launched, so no claim is made about the runs.

## Commit

`bf49fb7` on `haotian_modality_matching_jupiter` in `recipe/2025-10-speech-llm` (two files staged
by explicit path; `config_sae_1g_v1.py` and an untracked `config_sae_3e1_d6_swap_cont_v1.py`, other
people's live edits, were left alone).  Not pushed.  This report is not committed (it lives in the
i6_experiments checkout, which the dispatch did not name).
