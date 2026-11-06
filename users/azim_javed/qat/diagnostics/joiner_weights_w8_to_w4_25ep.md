# Pre-quantization joiner weight distributions

Measured from stored floating-point parameters, before fake quantization.
Near-zero percentages include exact zeros; thresholds are strict absolute values.

| Model | Joiner layer | Exact zero % | abs(w) < 0.001 % | abs(w) < 0.01 % | Median abs(w) | RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W4_finetuned_from_W8_25ep | 1 | 0.0000 | 0.40 | 4.05 | 0.129294 | 0.217646 |
| W4_finetuned_from_W8_25ep | 2 | 0.0000 | 0.57 | 5.96 | 0.038835 | 0.058446 |

## Quantization zero bins

Saved observer grids, without regridding. These are separate from exact floating-point zeros.

| Model | Joiner layer | Zero-bin half-width | Post-quantization zero % |
| --- | ---: | ---: | ---: |
| W4_finetuned_from_W8_25ep | 1 | 0.176765 | 73.00 |
| W4_finetuned_from_W8_25ep | 2 | 0.038942 | 57.03 |

## Checkpoints

- W4_finetuned_from_W8_25ep: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.H2C74Y7GMY21/output/models/epoch.500.pt`

Absolute thresholds compare raw weight units; models can have different learned scales.
The zero-bin table shows why exact floating-point zeros and quantized zeros differ.
