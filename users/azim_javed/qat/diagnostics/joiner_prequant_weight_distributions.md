# Pre-quantization joiner weight distributions

Measured from stored floating-point parameters, before fake quantization.
Near-zero percentages include exact zeros; thresholds are strict absolute values.

| Model | Joiner layer | Exact zero % | abs(w) < 0.001 % | abs(w) < 0.01 % | Median abs(w) | RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP_baseline | 1 | 0.0000 | 1.09 | 10.96 | 0.051204 | 0.096060 |
| FP_baseline | 2 | 0.0000 | 1.58 | 16.03 | 0.033434 | 0.054976 |
| W4_from_scratch | 1 | 0.0000 | 0.80 | 7.93 | 0.065191 | 0.084885 |
| W4_from_scratch | 2 | 0.0000 | 0.43 | 4.19 | 0.095177 | 0.108137 |
| W8_from_scratch | 1 | 0.0000 | 1.06 | 11.01 | 0.051136 | 0.097617 |
| W8_from_scratch | 2 | 0.0000 | 1.60 | 16.01 | 0.033360 | 0.054972 |
| W4_finetuned_25ep | 1 | 0.0000 | 0.38 | 3.70 | 0.138978 | 0.180924 |
| W4_finetuned_25ep | 2 | 0.0000 | 0.63 | 6.60 | 0.034097 | 0.062618 |

## Quantization zero bins

Saved observer grids, without regridding. These are separate from exact floating-point zeros.

| Model | Joiner layer | Zero-bin half-width | Post-quantization zero % |
| --- | ---: | ---: | ---: |
| W4_from_scratch | 1 | 0.099913 | 86.68 |
| W4_from_scratch | 2 | 0.099811 | 74.28 |
| W8_from_scratch | 1 | 0.005762 | 6.32 |
| W8_from_scratch | 2 | 0.001664 | 2.66 |
| W4_finetuned_25ep | 1 | 0.186965 | 76.98 |
| W4_finetuned_25ep | 2 | 0.034054 | 47.33 |

## Checkpoints

- FP_baseline: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.KVsfmDJ2gkuo/output/models/epoch.2000.pt`
- W4_from_scratch: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.rvKceYv76piK/output/models/epoch.2000.pt`
- W8_from_scratch: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.s9sXAl53j9fG/output/models/epoch.2000.pt`
- W4_finetuned_25ep: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.QDAkmoDhtjK5/output/models/epoch.500.pt`

Absolute thresholds compare raw weight units; models can have different learned scales.
The zero-bin table shows why exact floating-point zeros and quantized zeros differ.
