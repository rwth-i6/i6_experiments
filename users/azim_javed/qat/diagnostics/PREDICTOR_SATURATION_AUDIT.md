# Predictor saturation: full W4/A8 versus W8/A8

## Executed control

`probe_predictor_saturation.py` loads the existing cycle-1 converted checkpoint
with the actual recognition scorer and SynaptogenML clone. It evaluates every
one of the 185 possible single-token histories, including initial blank.
Prediction caching is disabled. It wraps the actual DAC/ADC methods and hooks
the prediction activation quantizers; clipping is counted before clamping, on
the rounded integer code that the implementation would use.

Tested predictor ADC ranges 4, 8, and 16 with precision fixed at 4, independently
with read noise disabled and enabled. Each range replays seed 17, so the read
noise draws match. Converter settings are replaced via dataclasses.replace,
which also recomputes the derived converter bounds.

## W4: no measured predictor saturation

With read noise enabled and baseline ADC precision/range 4/4:

| Stage | Max absolute input | Available bound | Clipped values |
| --- | ---: | ---: | ---: |
| Embedding ADC | 1.28045 | 8 | 0 / 142,080 |
| Prediction linear 1 DAC (normalized) | 0.43175 | 1 | 0 / 47,360 |
| Prediction linear 1 ADC | 2.53160 | 8 | 0 / 710,400 |
| First linear output activation quantizer | 0.76729 | 8.06250 | 0 / 118,400 |
| Prediction linear 2 DAC (normalized) | 0.57793 | 1 | 0 / 118,400 |
| Prediction linear 2 ADC | 1.53345 | 8 | 0 / 1,776,000 |
| Second linear output activation quantizer | 3.77823 | 29.12500 | 0 / 118,400 |

Embedding DAC one-hot values reach exactly 1 but do not exceed the supported
endpoint. With read noise disabled, no stage clips either.

Widening predictor ADC range from 4 to 8 or 16 produces **exactly identical W4
prediction states**, with and without read noise: maximum absolute output
difference is 0.0. Final predictor-output RMSE against the digital predictor is
0.386745 without read noise and 0.418639 with read noise, unchanged by range.

Thus predictor ADC/DAC or activation range saturation does not explain the W4
failure in this converted cycle. This is exhaustive over the finite history
inputs, but the read-noise experiment uses one seed, not every stochastic draw
or converted programming cycle.

The activation ranges are instead under-utilized: a wide historical range and
small signal can impose coarse quantization without clipping. That is a
different mechanism from range saturation and has not been isolated by this
range-only control.

## W8 control

W8's second prediction linear does show ADC clipping: 31,834 / 4,144,000 values
(0.7682%) with read noise, max absolute input 16.78904 against bound 8. Wider
range removes those events. Predictor-output RMSE changes only from 0.0860663
to 0.0859763, although some individual output values change by up to 0.21978.
Its other predictor stages do not clip in this test.

## Files

- `diagnostics/probe_predictor_saturation.py`
- `diagnostics/predictor_saturation_w4.json`
- `diagnostics/predictor_saturation_w8.json`

Run with base_env Python and the same checkpoint/clone arguments as
`audit_full_qat_recog_wiring.py`, plus `--output`. No workflow submission or
production configuration changes are involved.
