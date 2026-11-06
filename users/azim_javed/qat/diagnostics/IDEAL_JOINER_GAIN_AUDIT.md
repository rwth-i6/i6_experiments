# Why ideal joiner programming is not a digital-equivalent control

## Scope

`joint_correction_settings.ideal_programming=True` with
`joint_correction_layer=3` sets ideal resistance states for both joiner linears.
It does not idealize the encoder/predictor, remove read noise or converters,
replace the I-V polynomials, or recalibrate output factors. The digital-joiner
ablation therefore differs from the ideal-programming ablation in more than
programming errors.

Earlier emphasis on programmed off-bit leakage alone was insufficient to
explain failure with ideal joiner programming. The following control isolates
an additional gain mismatch directly.

## Analytic gain from the actual saved I-V coefficients

An ideal signed bit has current proportional to:

```text
I_difference(v) = poly_low(v) - poly_high(v)
scaled_current = 8020 * I_difference(v)
v = 0.6 * normalized_input
```

For digital-equivalent small-signal behavior the normalized-input gain should
be 1. The saved coefficients instead give:

```text
g0 = 8020 * 0.6 * (poly_low[1] - poly_high[1])
   = 1.434128642
```

This is the derivative at zero, not a fit to the diagnostic outputs. It is
the same for both tested joiner layers and both bit precisions. All saved
positive/negative polynomials and all tile/plane polynomials are checked for
equality before applying the shared per-layer calibration.

For the saved nonideal W4 programming, reconstructed resistance-weight gain
is approximately 0.682. That attenuation and the I-V gain approximately
compensate in aggregate: 0.682 * 1.434 is about 0.978. Ideal programming removes
the attenuation without removing the I-V gain. This numerical relationship
does not establish the original calibration intent.

## Executed matched-input controls

Uses the same digital acoustic/prediction inputs as `probe_full_qat_joiner.py`:
256 frame/history pairs from a 2.895-second real-audio excerpt. All controls
disable read noise and use ADC range 8 (no clipping on these inputs). Biases
and activation grids remain unchanged.

The calibrated control divides the physical current scaling by the analytic
gain **before** ADC conversion. It does not scale the bias. The original I-V
nonlinearity is retained. A separate ideal-linear-I-V control replaces the
current law with normalized voltage, retaining the DAC and ADC.

W4 measurements, ADC precision 4:

| Layer | Ideal programming signal gain | Ideal RMSE | Ideal + slope calibration RMSE |
| --- | ---: | ---: | ---: |
| First joiner linear | 1.42101 | 0.304288 | 0.155847 |
| Second joiner linear | 1.42354 | 1.597363 | 0.084360 |

With ADC precision 8, calibrated first/second linear RMSE falls to
0.022141 / 0.082261. The second linear also retains some nonlinear-I-V error:
ideal linear I-V gives RMSE 0.035404 at precision 8.

Final W4 logits after both joiner layers and their activation quantizers:

| Control | Logit RMSE versus digital |
| --- | ---: |
| Saved nonideal programming, ADC precision 4 | 0.983973 |
| Ideal programming, ADC precision 4 | 2.061272 |
| Ideal programming + analytic gain calibration, precision 4 | 0.811178 |
| Ideal programming + analytic gain calibration, precision 8 | 0.275302 |
| Ideal programming + linear I-V, precision 8 | 0.260439 |

The W8 control shows the same gain issue: analytic calibration reduces final
ideal-programming logit RMSE from 1.573995 to 0.306356 at precision 4, and to
0.261307 at precision 8. Thus gain mismatch is shared simulator behavior,
not a W4-specific wiring error. W4 has additional predictor sensitivity and
stronger finite-resolution sensitivity in the first joiner layer.

These are layer/logit controls on a fixed sample, not corpus WER measurements.
They establish that ideal resistance programming does not reproduce the
trained joiner, and explain why failure to recover with that setting does not
rule out analog joiner arithmetic as the cause. They do not establish which
calibration settings recover full recognition performance.

## Reproduce

Run `diagnostics/probe_ideal_joiner_gain.py` with base_env Python. Arguments:
`--digital`, `--converted`, `--inputs`, `--bits`, `--output`.
The saved `.inputs.pt` files are from `probe_full_qat_joiner.py`.

Results:

- `diagnostics/ideal_joiner_gain_w4.json`
- `diagnostics/ideal_joiner_gain_w8.json`

No production hardware settings, conversion jobs, or simulator files were
modified; calibration and ideal I-V changes exist only inside this diagnostic.
