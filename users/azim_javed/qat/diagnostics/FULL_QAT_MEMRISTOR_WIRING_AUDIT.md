# Full W4/A8 memristor recognition wiring audit

## Scope and conclusion

Checked the full, from-scratch W4/A8 model at epoch 2000, conversion cycle 1,
against the corresponding W8/A8 control. This is the full-model memristor path,
including the embedding, prediction network, and joiner.

No checkpoint substitution, missing recognition state, wrong bit precision, or
predictor/joiner index mismatch was found. There is, however, a large measured
W4-specific programming sensitivity in the second prediction linear layer.
Embedding conversion also changes the trained quantization grid by default.
These findings do not establish a complete decomposition of corpus WER.

## Actual checkpoint and state checks

- W4 digital: `ReturnnTrainingJob.rvKceYv76piK/output/models/epoch.2000.pt`.
- W4 converted: `MemristorModelConversionJob.jnW3fxJoeLVD/output/memristor_converted_model.pt`.
- Both recognition encoder and scorer reference that converted checkpoint.
- The actual recognition log specifies W4/A8 and converter settings
  `8 4 4 0.6 8020.0`.
- Recognition SynaptogenML clone: `CloneGitRepositoryJob.pxBD5qpvkBp6/output/synaptogen_ml`.

The schema check constructs the actual recognition classes on the meta device:

| Check | W4 | W8 |
| --- | ---: | ---: |
| Required encoder state entries | 84,144 | 195,024 |
| Required scorer state entries | 2,318 | 5,366 |
| Missing required entries | 0 | 0 |
| Incompatible state shapes | 0 | 0 |
| Digital/conversion common entries changed | 0 / 806 | 0 / 806 |

Saved length-one input/output factors load into scalar parameters via PyTorch's
legacy scalar compatibility. They are not missing or incompatible parameters.
The four predictor/joiner linear biases are unchanged; their saved input factors
match the corresponding digital input-observer bounds.

## W4 predictor is unusually sparse on its trained grid

Layer indices include dropout and quantizer modules: `prediction_net.2` is the
first linear; `prediction_net.7` is the second linear.

| Measurement | W4 first linear | W4 second linear | W8 second linear |
| --- | ---: | ---: | ---: |
| Nonzero quantized weights | 13.3569% | 0.6025% | 97.5142% |
| Historical observed weight absmax | 0.284906 | 5.399985 | 0.794237 |
| Current floating weight absmax | 0.059258 | 2.192389 | 0.566874 |
| Quantized weight RMS | 0.014875 | 0.060075 | 0.102155 |
| Programmed weight RMSE | 0.007709 | 0.117482 | 0.035873 |
| Intended-zero bit leakage weight RMS | 0.006056 | 0.115876 | 0.014897 |

The second W4 linear's quantization step is 0.771426. Its upper signed-magnitude
bitplane is entirely unused, but simulated positive/negative off-state currents
still differ. Those differences enter the weighted bitplane sum. Leakage from
intended-zero bits is about 1.93 times the intended weight RMS in W4, versus
0.146 times in W8. These are programmed-state measurements, before ADC effects
or read noise; they should not be interpreted as output RMSE or WER.

## Matched-input causal control, executed for both precisions

Used 16 evenly spaced vocabulary IDs, including initial blank. Each model's own
digital embedding and first prediction layer generate inputs to its second
linear. Within each model, exactly those same inputs are used for all controls.
The actual recognition clone's `TiledMemristorLinear` executes each control with
ADC 4/4 and read noise disabled. Biases, converters, polynomial I-V laws, and
input/output factors are held fixed.

For the zero-bit control, positive/negative resistance states are set to their
average only where the intended signed bit is zero; programmed on-bits are
untouched. Positive/negative polynomial equality is asserted. This eliminates
off-bit differential leakage without replacing on-bit programming.

| Second prediction linear control | W4 output RMSE | W4 RMSE / reference std | W4 post-quant/tanh RMSE | W8 RMSE / reference std |
| --- | ---: | ---: | ---: | ---: |
| Saved programmed states | 0.337731 | 2.20424 | 0.303822 | 0.088725 |
| Only intended-zero bit leakage removed | 0.017107 | 0.11165 | 0.032601 | 0.038004 |
| Ideal resistance states on all bits | 0.077278 | 0.50437 | 0.097518 | 0.380280 |

Removing only zero-bit leakage reduces W4 pre-quantization output error by
19.74x. The effect survives the output quantizer and tanh. Thus W4's predictor
has a demonstrated analog failure on valid token-history inputs, independently
of acoustic encoder wiring, the joiner, or prediction-cache ordering.

Ideal resistance states are not an exact digital arithmetic control: the I-V
law and fixed current scaling remain, so ideal programming can introduce gain
error. It is not justified to infer that ideal programming alone fixes WER.

## Separate embedding grid change

`recipe/model_pipelines/qat_ffnn_transducer/pytorch_modules.py`,
`QATFFNNTransducerModel._convert_embedding`, replaces the saved observer range
with the current floating-weight absmax when `regrid_embedding=True` (default).
For W4:

- Trained/checkpoint weight step: 0.629741.
- Conversion weight step: 0.291681.
- Nonzero fraction: 14.2166% on the trained grid, 68.9717% on the converted grid.
- RMS difference between the two quantized embedding matrices: 0.247935.

This changes the function seen by the trained predictor before analog device
error is introduced. The existing `regrid_embedding=False` option preserves the
grid. The existing report's wider-joint-ADC no-regrid run has WER 30.18961%,
versus 36.06619% for the corresponding regridded configuration. These are
separate single-run measurements, not a paired-seed causal WER estimate.

The same report shows that keeping the W4 joiner digital improves greedy WER
from 37.43752% to 14.25637% (cycle aggregates). Therefore the predictor result
does not account for the entire full-offload degradation: joiner conversion is
also a substantial contributor.

## Reproduction and next discriminating decode

Diagnostic: `diagnostics/audit_full_qat_recog_wiring.py`. Pass `--digital`,
`--converted`, `--bits`, `--synaptogen-root`, `--output`, and `--probe-predictor`.
Use base_env Python from this setup. JSON results from the executed runs:

- `diagnostics/full_qat_recog_wiring_w4.json`
- `diagnostics/full_qat_recog_wiring_w8.json`

No production recipe or simulator code was changed. A useful next corpus
control is the existing `offload_prediction_network=False` option, keeping
joiner/encoder hardware settings and programmed cycle fixed. That isolates
prediction-network offload from the already executed digital-joiner ablation.
A selective second-prediction-linear digital bypass would isolate the newly
measured failure more narrowly, but is not an existing offload option.

## Follow-up: matched joiner inputs and predictor interaction

Executed `probe_full_qat_joiner.py` for W4 and W8. Inputs are digital encoder
states from the same 2.895-second real dev-other utterance excerpt, crossed
with 16 token histories (256 pairs). Within each model all controls use
identical inputs. Read noise is disabled. The factorized saved I-V/ADC
implementation matches the actual recognition module exactly on the two
validation rows for each joiner linear (max absolute difference 0.0).

This is a local arithmetic experiment, not a decode or a corpus WER estimate;
the histories are enumerated examples, not reference-aligned hypotheses.

| Second joiner linear, same digital inputs | W4 | W8 |
| --- | ---: | ---: |
| Partial ADC clipping fraction, precision/range 4/4 | 0.08755% | 10.2296% |
| Output RMSE, precision/range 4/4 | 0.530482 | 1.042930 |
| Output RMSE, precision/range 4/8 | 0.520057 | 0.396246 |

This explains why range widening is effective for W8 but does little for W4
on this sample. Neither first joiner linear clips. W4 second-layer error
persists without ADC rounding (RMSE 0.517805). Removing intended-zero-bit
leakage lowers its 4/8 RMSE to 0.165946.

W4 first-layer behavior is mixed: zero-bit leakage removal alone barely
changes its 4/4 RMSE (0.160185 to 0.156257). Combining leakage removal with
higher ADC precision lowers it to 0.034670. Higher precision alone does not
help that layer (RMSE 0.201103). These interacting errors caution against
claiming a single converter setting or programming change solves full WER.

With the same digital acoustic states, final logit RMSE is:

| Predictor / joiner | W4 | W8 |
| --- | ---: | ---: |
| Digital predictor / analog joiner, range 4 | 0.992092 | 1.302402 |
| Analog predictor / digital joiner | 1.347394 | 0.242857 |
| Analog predictor / analog joiner, range 4 | 1.861273 | 1.305306 |
| Analog predictor / analog joiner, range 8 | 1.857534 | 0.873144 |

This shows a substantially larger W4 predictor contribution and the remaining
joiner contribution on fixed acoustic inputs. It does not use a memristor
encoder, search, or read noise. Embedding regridding is one contributor, not
a sufficient explanation of the residual WER around 30% with regridding off.

Results: `diagnostics/full_qat_joiner_w4.json` and
`diagnostics/full_qat_joiner_w8.json`; saved matched inputs have suffix
`.inputs.pt`. The separate exhaustive predictor range test is documented in
`diagnostics/PREDICTOR_SATURATION_AUDIT.md`.

## Follow-up: ideal programming still changes the joiner function

`diagnostics/IDEAL_JOINER_GAIN_AUDIT.md` isolates a remaining gain mismatch.
Saved I-V coefficients and current scaling yield analytic small-signal gain
1.43413 for ideal resistance states; observed ideal W4 joiner gains are about
1.42. Correcting that gain before the ADC reduces second-linear RMSE from
1.59736 to 0.08436 on the fixed sample. Across both layers, final ideal logit
RMSE falls from 2.06127 to 0.81118, and to 0.27530 with higher ADC precision.
This is not a corpus recovery result, but demonstrates why ideal programming
alone is not equivalent to keeping the trained joiner digital.
