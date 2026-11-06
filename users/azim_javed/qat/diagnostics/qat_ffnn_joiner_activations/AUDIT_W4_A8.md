# W4/A8 joiner histogram audit (2026-10-15)

## Exact plot and verification

The supplied image is `plots_comparison_w4_a8/joiner_histograms.png`.
Its generator is `plot_histograms.py`. `audit_histograms.py` independently
recounted every saved scalar using the archived bin edges and independently
recomputed moments from per-channel sums and sums of squares. All histogram
counts, means, population standard deviations, ranges, and row counts match.
The two layers also have matching batch/row counts within each source record.

Source recordings:

- Digital: `ReturnnForwardJobV2.vHPgbrpoKPsl`, 458,516 rows.
- Memristor: `ReturnnForwardJobV2.DByLm046ialY`, 460,515 rows.
- Both: 2,864 appended records, one empty, widths 1,024 and 185.

Both configurations use W4/A8, greedy dev-other recognition, zero ILM scale,
and zero blank penalty. The conversion input is the exact digital checkpoint
`ReturnnTrainingJob.rvKceYv76piK/output/models/epoch.2000.pt`.
The memristor scorer config has converters `8 4 4 0.6 8020.0`.

The original figure uses raw counts, shared bins **within** each layer,
independent panel axes, and hidden y ticks. The 0.44% row-count difference
does not explain the large layer-1 shape change. The density-normalized version
with visible y ticks is `audit_w4_a8/joiner_histograms_density.png`.

## Layer 1 genuinely broadens

| Statistic | Digital | Memristor |
| --- | ---: | ---: |
| Pooled standard deviation | 0.669657 | 1.050199 |
| Fraction with absolute output below 0.25 | 56.466% | 25.287% |
| Median channel standard deviation | 0.288013 | 0.764011 |
| Mean within-channel variance | 0.357190 | 1.013815 |
| Variance of channel means | 0.091250 | 0.089102 |

1,021/1,024 channels have greater standard deviation in the memristor recording.
Thus this is predominantly broader variation within channels, not a wider
spread of channel offsets. These are comparisons of marginal statistics over
independent searches, not paired errors.

The digital checkpoint has 13.322% nonzero quantized layer-1 weights, but **no
entirely zero weight rows**. No recorded digital channel is constant. The tall
blue peak is a concentration of many low-variance channels, not an artifact
from thousands of exactly zero/dead outputs. Weight sparsity alone does not
establish the cause of that concentration or of the memristor broadening.

## What the ADC actually clips

`recipe/model_pipelines/common/pyscorers.py:224-237` registers forward hooks on
`joint_net[2]` and `[7]`. These capture the complete linear-module outputs,
including model-unit rescaling and bias, before the following activation
quantizers. Layer 1 is also before tanh.

The recognition job imports SynaptogenML from
`CloneGitRepositoryJob.pxBD5qpvkBp6/output/synaptogen_ml`.
In that clone:

- `synaptogen_ml/memristor_modules/memristor.py:289-290` subtracts the positive
  and negative array outputs before ADC conversion.
- `memristor.py:301-306,335-347` defines the ADC grid.
- `linear.py:327-357` applies ADC conversion separately to each input tile and
  signed-magnitude weight bit-plane, then sums tiles and weights planes,
  rescales, and adds bias.

For precision 4 / range 4, the implementation is:

```text
u = differential_current * 8020
q(u) = clamp(round(16*u), -128, 128) / 16

y = output_factor * sum_input_tiles(4*q(u_bit2) + 2*q(u_bit1) + q(u_bit0)) + bias
```

The ADC step is 1/16 and the endpoints are +/-8 in internal scaled-current
units. "4/4" does not mean a conventional four-total-bit ADC or a final
linear output clipped to +/-4, +/-8, or +/-16 model units. Here it provides
257 signed codes, inclusive of both endpoints.

Even direct final-output hard clipping would leave the interior unchanged and
pile tail mass at the clipping boundaries; it would not necessarily sharpen
the central peak. Here the plot is further downstream of multiple partial
clipping operations. Unequal clipping can disturb cancellations among tiles
and planes. Programming errors and read noise enter before these operations,
and encoder/prediction input changes enter upstream. A broader reconstructed
distribution is therefore compatible with partial ADC saturation.

## Layer 2's apparent narrowing is mostly a pooling effect

| Statistic | Digital | Memristor |
| --- | ---: | ---: |
| Pooled standard deviation, all 185 channels | 3.942118 | 3.511710 |
| Mean within-token variance | 5.375625 | 6.262120 |
| Variance of token means | 10.164672 | 6.069988 |
| Pooled standard deviation, ordinary 182 tokens | 2.792595 | 2.783431 |
| Mean `</s>` logit | -26.125190 | -20.103363 |
| Mean `<unk>` logit | -26.326544 | -20.560224 |
| Mean blank logit | 14.041460 | 9.464583 |

The identity is `pooled variance = mean(within-channel variance) +
variance(channel means)`, since every recorded row includes every channel.
155/185 channels have greater memristor standard deviation. The aggregate
narrowing is caused by reduced separation between token means, strongly
influenced by the three special channels. Excluding those channels leaves
almost identical pooled standard deviations. This is not evidence that each
token's output was compressed.

## Causal limit and the decisive measurement

These are full-system independent searches. Encoder outputs, prediction-network
outputs, and visited token histories can all differ. The memristor scorer also
uses prediction caching (`fixed-context-py-cache`), whereas the digital scorer
uses `fixed-context-py`. In the analog model, caching preserves a sampled
prediction output rather than redrawing it on every occurrence of a token.

The saved dictionaries contain **only** the two reconstructed outputs. They do
not contain joiner inputs, contexts, or pre-ADC partial currents. Consequently
these recordings cannot determine ADC clipping rates or identify how much
broadening comes from programming, read noise, ADC effects, or changed inputs.
The earlier `../JOINER_CONVERSION_AUDIT.md` reports substantial W4 programming
error, including a synthetic no-read-noise control; that supports a mechanism
for broadening but is not a causal decomposition of these corpus histograms.

The decisive follow-up is to save a small real-input sample and evaluate both
joiners on exactly those inputs, recording pre-ADC currents and actual clamp
events per tile/plane. Compare fixed programmed states with read noise on/off,
ADC 4/4 versus wider/unclipped range, and ideal programming. Keep layer-1
post-quantization/post-tanh values and token-conditioned layer-2 outputs as
separate measurement points. A final-output histogram alone cannot substitute
for those controls.

## Reproduce the executed audit

From `training/qat`:

```bash
/work/smt4/azim.javed/envs/base_env/bin/python \
  diagnostics/qat_ffnn_joiner_activations/audit_histograms.py \
  --comparison diagnostics/qat_ffnn_joiner_activations/plots_comparison_w4_a8 \
  --digital-checkpoint /work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.rvKceYv76piK/output/models/epoch.2000.pt \
  --output-dir diagnostics/qat_ffnn_joiner_activations/audit_w4_a8
```

Exact numerical results are in `audit_w4_a8/audit.json`.
