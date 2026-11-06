# W4/A8 fine-tuning sparsity diagnostic

Sparsity is measured after quantizing weights on their saved observer grids.
Percentages below are parameter-count weighted, not averages over layers.

## Encoder comparison

| Model | Epoch (subepoch) | Quantized layers | Nonzero % | Zero % | Bit 0 active % | Bit 1 active % | Bit 2 active % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_qat_ft25_w8_to_w4 | 500 | 120 | 45.682 | 54.318 | 39.523 | 6.754 | 0.054 |
| full_qat_scratch_w4 | 2000 | 120 | 44.233 | 55.767 | 39.824 | 4.664 | 0.013 |

## full_qat_ft25_w8_to_w4

Checkpoint: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.H2C74Y7GMY21/output/models/epoch.500.pt`

| Group | Layers | Nonzero % | Layers below 10% nonzero | Empty highest bitplane |
| --- | ---: | ---: | ---: | ---: |
| encoder_ff | 48 | 46.770 | 4 | 0 |
| encoder_conv | 36 | 52.820 | 0 | 0 |
| encoder_mhsa | 36 | 37.830 | 0 | 0 |
| encoder_output | 1 | 42.821 | 0 | 0 |
| token_embedding | 1 | 12.158 | 0 | 1 |
| predictor | 2 | 7.185 | 2 | 1 |
| prediction_output | 1 | 92.877 | 0 | 0 |
| joiner | 2 | 29.211 | 0 | 0 |

### Quantized embedding, predictor and joiner

| Layer | Nonzero % | Weight scale | Observer/current absmax | Quantized RMS | Highest bit active % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `token_embedding` | 12.158 | 0.629025 | 2.568 | 0.220905 | 0.00000 |
| `prediction_net.2` | 9.351 | 0.135462 | 3.451 | 0.041431 | 0.00000 |
| `prediction_net.7` | 6.318 | 0.175680 | 1.079 | 0.048309 | 0.01147 |
| `joint_net.2` | 27.002 | 0.353530 | 1.059 | 0.216701 | 0.01009 |
| `joint_net.7` | 42.967 | 0.077884 | 1.052 | 0.061038 | 0.07549 |

### Ten sparsest encoder layers

| Layer | Nonzero % | Observer/current absmax | Zero output rows |
| --- | ---: | ---: | ---: |
| `conformer.module_list.8.module_list.0.linear_out` | 2.765 | 1.204 | 0/512 |
| `conformer.module_list.10.module_list.0.linear_out` | 2.790 | 1.018 | 0/512 |
| `conformer.module_list.9.module_list.0.linear_out` | 3.181 | 1.060 | 0/512 |
| `conformer.module_list.11.module_list.3.linear_out` | 9.322 | 1.024 | 0/512 |
| `conformer.module_list.11.module_list.0.linear_out` | 11.446 | 1.018 | 0/512 |
| `conformer.module_list.3.module_list.2.mhsa.linear_pos` | 17.450 | 1.072 | 0/512 |
| `conformer.module_list.11.module_list.2.mhsa.linear_pos` | 19.338 | 1.310 | 0/512 |
| `conformer.module_list.11.module_list.1.pointwise_conv2` | 21.532 | 1.050 | 0/512 |
| `conformer.module_list.2.module_list.2.mhsa.linear_pos` | 21.655 | 1.582 | 0/512 |
| `conformer.module_list.0.module_list.2.mhsa.linear_pos` | 22.315 | 1.856 | 0/512 |

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
