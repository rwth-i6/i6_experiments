# W4/A8 fine-tuning sparsity diagnostic

Sparsity is measured after quantizing weights on their saved observer grids.
Percentages below are parameter-count weighted, not averages over layers.

## Encoder comparison

| Model | Epoch (subepoch) | Quantized layers | Nonzero % | Zero % | Bit 0 active % | Bit 1 active % | Bit 2 active % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_qat_ft25_fp_to_w4 | 500 | 120 | 49.162 | 50.838 | 41.564 | 8.419 | 0.086 |
| full_qat_scratch_w4 | 2000 | 120 | 44.233 | 55.767 | 39.824 | 4.664 | 0.013 |

## full_qat_ft25_fp_to_w4

Checkpoint: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.QDAkmoDhtjK5/output/models/epoch.500.pt`

| Group | Layers | Nonzero % | Layers below 10% nonzero | Empty highest bitplane |
| --- | ---: | ---: | ---: | ---: |
| encoder_ff | 48 | 51.214 | 4 | 0 |
| encoder_conv | 36 | 55.103 | 0 | 0 |
| encoder_mhsa | 36 | 38.960 | 0 | 0 |
| encoder_output | 1 | 55.314 | 0 | 0 |
| token_embedding | 1 | 38.066 | 0 | 0 |
| predictor | 2 | 9.116 | 1 | 1 |
| prediction_output | 1 | 92.839 | 0 | 0 |
| joiner | 2 | 27.123 | 0 | 0 |

### Quantized embedding, predictor and joiner

| Layer | Nonzero % | Weight scale | Observer/current absmax | Quantized RMS | Highest bit active % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `token_embedding` | 38.066 | 0.126088 | 1.104 | 0.085563 | 0.07601 |
| `prediction_net.2` | 14.967 | 0.098832 | 4.421 | 0.038240 | 0.00000 |
| `prediction_net.7` | 6.775 | 0.201188 | 1.217 | 0.054251 | 0.01001 |
| `joint_net.2` | 23.021 | 0.373930 | 1.021 | 0.185961 | 0.00068 |
| `joint_net.7` | 52.669 | 0.068108 | 1.078 | 0.065345 | 0.28294 |

### Ten sparsest encoder layers

| Layer | Nonzero % | Observer/current absmax | Zero output rows |
| --- | ---: | ---: | ---: |
| `conformer.module_list.10.module_list.0.linear_out` | 5.232 | 1.055 | 0/512 |
| `conformer.module_list.10.module_list.0.linear_ff` | 7.566 | 1.026 | 0/2048 |
| `conformer.module_list.11.module_list.0.linear_out` | 8.853 | 1.036 | 0/512 |
| `conformer.module_list.9.module_list.0.linear_out` | 9.532 | 1.052 | 0/512 |
| `conformer.module_list.11.module_list.3.linear_out` | 17.930 | 1.016 | 0/512 |
| `conformer.module_list.0.module_list.2.mhsa.qkv_proj` | 19.532 | 1.047 | 0/1536 |
| `conformer.module_list.3.module_list.2.mhsa.linear_pos` | 19.621 | 1.140 | 0/512 |
| `conformer.module_list.1.module_list.1.depthwise_conv` | 22.322 | 1.025 | 0/512 |
| `conformer.module_list.2.module_list.1.depthwise_conv` | 23.841 | 1.050 | 0/512 |
| `conformer.module_list.3.module_list.1.depthwise_conv` | 23.948 | 1.040 | 0/512 |

## full_qat_scratch_w4

Checkpoint: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.rvKceYv76piK/output/models/epoch.2000.pt`

| Group | Layers | Nonzero % | Layers below 10% nonzero | Empty highest bitplane |
| --- | ---: | ---: | ---: | ---: |
| encoder_ff | 48 | 46.686 | 0 | 2 |
| encoder_conv | 36 | 49.570 | 0 | 0 |
| encoder_mhsa | 36 | 33.116 | 0 | 1 |
| encoder_output | 1 | 54.147 | 0 | 1 |
| token_embedding | 1 | 14.217 | 0 | 1 |
| predictor | 2 | 4.247 | 1 | 2 |
| prediction_output | 1 | 92.877 | 0 | 0 |
| joiner | 2 | 15.038 | 0 | 0 |

### Quantized embedding, predictor and joiner

| Layer | Nonzero % | Weight scale | Observer/current absmax | Quantized RMS | Highest bit active % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `token_embedding` | 14.217 | 0.629741 | 2.159 | 0.240126 | 0.00000 |
| `prediction_net.2` | 13.357 | 0.040701 | 4.808 | 0.014875 | 0.00000 |
| `prediction_net.7` | 0.603 | 0.771426 | 2.463 | 0.060075 | 0.00000 |
| `joint_net.2` | 13.322 | 0.199826 | 1.158 | 0.078184 | 0.00526 |
| `joint_net.7` | 25.724 | 0.199622 | 1.110 | 0.107046 | 0.01478 |

### Ten sparsest encoder layers

| Layer | Nonzero % | Observer/current absmax | Zero output rows |
| --- | ---: | ---: | ---: |
| `conformer.module_list.11.module_list.0.linear_out` | 12.802 | 1.700 | 0/512 |
| `conformer.module_list.11.module_list.2.mhsa.linear_pos` | 16.098 | 1.522 | 0/512 |
| `conformer.module_list.2.module_list.2.mhsa.linear_pos` | 17.124 | 1.651 | 0/512 |
| `conformer.module_list.1.module_list.2.mhsa.qkv_proj` | 17.254 | 1.144 | 0/1536 |
| `conformer.module_list.5.module_list.2.mhsa.qkv_proj` | 17.533 | 1.435 | 0/1536 |
| `conformer.module_list.1.module_list.2.mhsa.linear_pos` | 17.991 | 2.019 | 0/512 |
| `conformer.module_list.11.module_list.1.depthwise_conv` | 18.819 | 1.424 | 0/512 |
| `conformer.module_list.11.module_list.1.pointwise_conv2` | 19.104 | 1.308 | 0/512 |
| `conformer.module_list.11.module_list.3.linear_out` | 19.224 | 1.195 | 0/512 |
| `conformer.module_list.4.module_list.2.mhsa.qkv_proj` | 19.532 | 1.377 | 0/1536 |

## Interpretation

This measures numerical sparsity, not explicit pruning or memristor programming error.
An active bitplane percentage counts weights with that magnitude bit set, independently of sign.
The CSV has every quantized layer; JSON additionally includes code histograms and bitplane counts.
The current-range regrid column is a hypothetical comparison only, not a recommended model change.
Saved observer ranges are held fixed, matching conversion; the JSON flags layers whose current
weights exceed those ranges and would expand a lazily initialized eval observer.
