# W4/A8 versus W8/A8 joint-network diagnosis

W8 works with widened ADC range and is the required control. A discrepancy in
shared code is not, by itself, an explanation of W4-specific behavior. No
production model, hardware settings, or shared simulator code was changed.

## Report evidence (dev-other, lexicon-free)

From `report_digital.txt` and `memristor_runs_aggregated_report.txt`:

| Configuration | W4/A8 WER | W8/A8 WER |
| --- | ---: | ---: |
| Digital | 8.2% | 6.4% |
| Memristor, default joint ADC 4 precision / 4 range | 37.44% | 44.73% |
| Joint L1 4/4, L2 4/8 | 37.18% | 12.15% |
| Both joint layers 4/8 | 36.07% | not used in this comparison |
| Entire joint network kept digital | 14.26% | not used in this comparison |
| Only second joint linear kept digital | 29.91% | not used in this comparison |

The widened-range rows above are single-cycle measurements, while the default
and offload ablations aggregate multiple cycles. Keeping both joint layers
digital is considerably better than keeping only layer 2 digital.

## Code trace

- `recipe/experiments/librispeech/recognition/memristor/qat_ffnn_transducer_bpe.py`
  sends separate joint settings to the label scorer.
- `recipe/model_pipelines/qat_ffnn_transducer/mem_inited/label_scorer_config.py`
  serializes both settings.
- The scorer in `mem_inited/pytorch_modules.py` selects them separately for
  `joint_net[2]` and `joint_net[7]`.
- Conversion splits signed weight magnitudes into `bits - 1` planes. Forward
  reconstruction uses the corresponding powers of two; no W4-only indexing
  discrepancy was found.
- W4's saved input/output factors match the factors derived from its digital
  observer buffers.
- DAC is signed symmetric but uses a 128-level normalization versus QAT's 127.
  This shared grid mismatch needs a matched W4/W8 sensitivity check before it
  can be blamed for this failure.

## W4 checkpoint probe already executed

Uses existing cycle-1 conversion, digital epoch 2000, and the SynaptogenML clone
referenced by the generated recognition config. The inputs are synthetic;
these are arithmetic checks, not WER measurements. The digital reference uses
the same effective DAC inputs to isolate downstream errors.

| Measurement | Joint layer 1 | Joint layer 2 |
| --- | ---: | ---: |
| Weight quantization scale | 0.199826 | 0.199622 |
| Nonzero digital quantized weights | 13.32% | 25.72% |
| Intended digital weight RMS | 0.07818 | 0.10705 |
| Programmed weight RMS error from resistance states | 0.03898 | 0.04541 |
| Programmed error / digital weight RMS | 49.86% | 42.42% |
| Output RMSE, current simulator, ADC 4/8 | 3.35574 | 0.89525 |
| Output RMSE, read noise disabled, ADC 4/8 | 3.17426 | 0.87602 |
| No-read-noise error / reference output std | 57.03% | 36.12% |

Zero-valued signed bits have approximately 0.0326–0.0328 RMS differential
resistance leakage per plane. Programmed weight error above uses
`sum(2**bit * (r_negative - r_positive))` versus intended signed quantized
integers, multiplied by the saved digital weight scale. It is independent of
ADC quantization, read noise, and sampled input values.

This is strong evidence of a W4 error floor that extra ADC range cannot remove.
It does not yet establish why W8 tolerates conversion: W8's corresponding
scale, sparsity, relative programming error, and ideal-programming result are
still needed. Programming variability is intentionally enabled in this run
(`correction_settings=None`), so a large programming error is not automatically
a bitplane-conversion bug.

## Separate simulator inconsistencies

In the actual recognition clone's `synaptogen_ml/memristor_modules/memristor.py`:

- Line 138 sets `self.e = np.exp(1)` and line 191 uses it as electron charge in
  shot noise. The reference `synaptogen.py` uses `1.602176634e-19`.
- Line 135 sets bandwidth to `1e-8`; reference `Iread()` defaults to `1e8`.

These inconsistencies affect both bit widths. They are not established causes
of the W4/W8 difference. W4's large error persists with read noise disabled.
The probe changes these constants only in its own process, as diagnostic
controls, and does not propose a global change to the successful W8 setup.

## Remaining GPU control

From `training/qat` on a GPU node:

```bash
bash diagnostics/run_joiner_conversion_gpu.sh both
```

Runs W4 and W8 sequentially and saves
`diagnostics/joiner_conversion_w4_gpu.log` and
`diagnostics/joiner_conversion_w8_gpu.log`.
Checkpoint metadata loading is on CPU and can take time, especially for W8.
The layer forwards and bitplane comparisons run on CUDA.

Each model runs with ADC 4/8 on the same inputs within that model:

1. Current simulator.
2. Read noise disabled.
3. Physical electron charge at current bandwidth.
4. Physical electron charge at reference bandwidth.
5. Ideal programmed resistance states, read noise disabled, preserving DAC,
   ADC, I-V polynomials, conversion factors, and bias.

The models use their own weights and grids, so compare normalized errors rather
than assuming their raw output tensors should match. The ideal-programming
control tests whether removing device-state errors removes W4's arithmetic
failure. No corpus decoding or Sisyphus manager is launched.

The CPU W4 checks passed. The updated ideal-programming control and CUDA path
await execution; a W8 metadata-only attempt timed out while loading metadata.
