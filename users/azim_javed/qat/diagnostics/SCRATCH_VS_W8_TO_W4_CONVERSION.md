# Scratch W4 versus W8-to-W4: measured differences beyond joiner sparsity

## Question and scope

The W8-to-W4 full-QAT 25-epoch fine-tune has 73.00% / 57.03% joiner zeros,
yet its reported full memristor greedy WER is 14.10654%. The scratch W4 model
has 86.68% / 74.28% joiner zeros and substantially poorer conversion results.
Global sparsity is not a sufficient explanation. This follow-up compares the
actual saved digital and programmed checkpoints, then predictor and joiner
computations with controlled inputs.

## Checkpoints and settings

| Model | Digital training job | Actual no-regrid conversion job |
| --- | --- | --- |
| Scratch W4 | `ReturnnTrainingJob.rvKceYv76piK`, epoch 2000 | `MemristorModelConversionJob.UJjbTmp4u6eI` |
| W8-to-W4 FT | `ReturnnTrainingJob.H2C74Y7GMY21`, epoch 500 | `MemristorModelConversionJob.OeIbbB3fpLE9` |

Both actual conversions preserve the embedding grid. The actual FT decode
uses ADC precision/range 4/8 throughout the model, positional settings 1/7,
normal programming, and prediction caching. The existing scratch no-regrid
30.18961% decode uses 4/8 for the joiner but 4/4 for the encoder and predictor.
Thus those corpus WERs are not a fully hardware-matched pair.

The local comparison below uses the same predictor hardware in both models
(4/8, with positional settings irrelevant to predictor evaluation) and varies
joiner range explicitly. Acoustic states are digital in the local joint-input
tests, isolating scorer effects from encoder conversion.

All expected converted encoder/scorer state entries are present in both
checkpoints; the 806 common digital/converted state values are unchanged.

## Large difference: the second prediction linear (`prediction_net.7`)

| Measurement | Scratch W4 | W8-to-W4 FT |
| --- | ---: | ---: |
| Nonzero quantized weights | 0.6025% | 6.3184% |
| Weight step | 0.771426 | 0.175680 |
| Historical observed absmax | 5.399985 | 1.229763 |
| Current floating absmax | 2.192389 | 1.140231 |
| Intended quantized weight RMS | 0.060075 | 0.048309 |
| Programmed resistance-reconstruction weight RMSE | 0.117369 | 0.030555 |
| Intended-zero-bit leakage weight RMS | 0.115785 | 0.026310 |

These reconstruction errors precede the I-V law, converters and read noise.
They characterize how programmed resistance differences represent intended
integer bitplanes, not the final layer-output error.

The scratch model's second predictor layer therefore has a specific large
programming-error floor, not merely a high percentage of zeros. Its step and
zero-bit leakage weight RMS are each about 4.4 times those of the fine-tune.
The fine-tune also has roughly ten times as many nonzero weights in this layer.

The difference is not uniformly better programming in every FT layer: its
first predictor layer has larger absolute programming error than scratch,
and the embeddings have similar absolute programming errors.

## Actual predictor forward: exhaustive over 185 histories

Both use their actual converted checkpoint, no regridding, identical ADC
precision/range 4/8 and replayed seed 17. Every possible single-token history,
including initial blank, is evaluated without the prediction cache.

| Full predictor-output measurement | Scratch W4 | W8-to-W4 FT |
| --- | ---: | ---: |
| Output RMSE, read noise disabled | 0.374629 | 0.258700 |
| RMSE / own digital output std, read noise disabled | 2.40252 | 0.90471 |
| Output RMSE, read noise enabled | 0.408983 | 0.276612 |
| RMSE / own digital output std, read noise enabled | 2.62284 | 0.96735 |

Neither predictor clips at range 8. The scratch predictor's conversion error
is larger in absolute units and relative to its digital output variation.
Normalizing by global output std is a useful scale comparison, not a claim
that all feature errors have equal semantic importance.

## Matched acoustic-input scorer controls

For each model, digital encoder states from the same 2.895-second real-audio
excerpt are crossed with 16 history IDs (256 pairs). Each model is compared
with its own digital computation. Read noise is disabled, prediction ADC range
is 8, joiner precision/range is 4/8. Positive/negative I-V equality is checked.
The factorized linear implementation matches the actual recognition module
on its validation rows with maximum absolute difference 0.0 for both layers.

| Final logit RMSE versus own digital model | Scratch W4 | W8-to-W4 FT |
| --- | ---: | ---: |
| Digital predictor + analog joiner | 0.969764 | 0.767490 |
| Analog predictor + digital joiner | 1.148722 | 0.734040 |
| Analog predictor + analog joiner | 1.697114 | 1.084257 |
| Same, removing only second-predictor-linear zero-bit leakage | 1.115651 | 0.972785 |

The last control leaves the embedding, first predictor linear and both joiner
linears programmed as saved. It equalizes the positive/negative resistance
states only where the second predictor linear's intended signed bit is zero.
This removes off-bit differential leakage while preserving programmed on-bits.
It reduces scratch logit RMSE by about 34%, compared with about 10% for FT.
For scratch, predictor-output RMSE on these 16 histories falls from 0.362902
to 0.120215; FT falls from 0.251484 to 0.173285.

## Supported conclusion and causal limit

The models are not functionally similar simply because both joiners are
sparse. There is a demonstrated difference in second-predictor programming
error, end-to-end prediction-state distortion and propagated logit error.
Both the predictor and joiner contribute to the scratch/FT difference.

These controls do not yet explain the complete corpus WER gap. They use one
programming cycle, a small digital-acoustic sample for joiner tests and one
read-noise seed for exhaustive history tests. Encoder conversion, actual
reference/search histories and margins are not covered by the local sample.
Some local argmax comparisons do not track RMSE monotonically; logit RMSE is
not a substitute for WER.

The narrow next corpus intervention would keep all other device states and
settings fixed while bypassing only the second prediction linear or removing
its off-bit leakage. The existing whole-predictor digital-offload option is a
broader alternative. Neither intervention was submitted by this diagnostic.

## Executed outputs

- `full_qat_w8_to_w4_sparsity.{json,csv,md}`
- `full_qat_recog_wiring_ft_w8_to_w4_no_regrid.json`
- `full_qat_recog_wiring_scratch_w4_no_regrid.json`
- `predictor_saturation_w8_to_w4_no_regrid.json`
- `predictor_saturation_scratch_w4_no_regrid.json`
- `full_qat_joiner_w8_to_w4_no_regrid.json`
- `full_qat_joiner_scratch_w4_no_regrid.json`

All files are under `diagnostics/`. Production recipes and simulator code
were not changed. `probe_full_qat_joiner.py` now accepts
`--predictor-adc-range` so the predictor settings can be matched explicitly.
