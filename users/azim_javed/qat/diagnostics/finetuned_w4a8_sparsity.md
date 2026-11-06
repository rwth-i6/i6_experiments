# W4/A8 fine-tuning sparsity diagnostic

Sparsity is measured after quantizing weights on their saved observer grids.
Percentages below are parameter-count weighted, not averages over layers.

## Encoder comparison

| Model | Epoch (subepoch) | Quantized layers | Nonzero % | Zero % | Bit 0 active % | Bit 1 active % | Bit 2 active % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ft25_fp_to_w4 | 500 | 120 | 48.977 | 51.023 | 41.423 | 8.373 | 0.085 |
| ft50_fp_to_w4 | 1000 | 120 | 43.878 | 56.122 | 38.726 | 5.562 | 0.034 |
| ft50_w8_to_w4 | 1000 | 120 | 43.483 | 56.517 | 38.541 | 5.321 | 0.030 |
| scratch_encoder_w4 | 2000 | 120 | 41.822 | 58.178 | 37.815 | 4.242 | 0.012 |
| scratch_full_w4 | 2000 | 120 | 44.233 | 55.767 | 39.824 | 4.664 | 0.013 |

## ft25_fp_to_w4

Checkpoint: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.oRvQEgaIpBis/output/models/epoch.500.pt`

| Group | Layers | Nonzero % | Layers below 10% nonzero | Empty highest bitplane |
| --- | ---: | ---: | ---: | ---: |
| encoder_ff | 48 | 50.860 | 4 | 0 |
| encoder_conv | 36 | 55.152 | 0 | 0 |
| encoder_mhsa | 36 | 39.172 | 0 | 0 |

### Ten sparsest encoder layers

| Layer | Nonzero % | Observer/current absmax | Zero output rows |
| --- | ---: | ---: | ---: |
| `conformer.module_list.10.module_list.0.linear_out` | 5.236 | 1.055 | 0/512 |
| `conformer.module_list.10.module_list.0.linear_ff` | 7.113 | 1.027 | 1/2048 |
| `conformer.module_list.11.module_list.0.linear_out` | 8.588 | 1.029 | 0/512 |
| `conformer.module_list.9.module_list.0.linear_out` | 9.477 | 1.054 | 0/512 |
| `conformer.module_list.11.module_list.3.linear_out` | 16.091 | 1.018 | 0/512 |
| `conformer.module_list.0.module_list.2.mhsa.qkv_proj` | 18.421 | 1.036 | 0/1536 |
| `conformer.module_list.3.module_list.2.mhsa.linear_pos` | 20.614 | 1.135 | 0/512 |
| `conformer.module_list.1.module_list.1.depthwise_conv` | 21.988 | 1.034 | 0/512 |
| `conformer.module_list.3.module_list.1.depthwise_conv` | 23.003 | 1.051 | 0/512 |
| `conformer.module_list.2.module_list.1.depthwise_conv` | 24.937 | 1.037 | 0/512 |

Embedding/predictor/joiner weights are floating point in this model:
- `token_embedding`: 0.5405% exact zeros (not W4 sparsity).
- `prediction_net.1`: 0.0000% exact zeros (not W4 sparsity).
- `prediction_net.4`: 0.0000% exact zeros (not W4 sparsity).
- `joint_net.1`: 0.0000% exact zeros (not W4 sparsity).
- `joint_net.4`: 0.0000% exact zeros (not W4 sparsity).

## ft50_fp_to_w4

Checkpoint: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.zUCmiFH9CKC3/output/models/epoch.1000.pt`

| Group | Layers | Nonzero % | Layers below 10% nonzero | Empty highest bitplane |
| --- | ---: | ---: | ---: | ---: |
| encoder_ff | 48 | 45.737 | 4 | 0 |
| encoder_conv | 36 | 49.786 | 0 | 0 |
| encoder_mhsa | 36 | 34.314 | 0 | 0 |

### Ten sparsest encoder layers

| Layer | Nonzero % | Observer/current absmax | Zero output rows |
| --- | ---: | ---: | ---: |
| `conformer.module_list.10.module_list.0.linear_out` | 4.553 | 1.096 | 0/512 |
| `conformer.module_list.9.module_list.0.linear_out` | 4.760 | 1.152 | 0/512 |
| `conformer.module_list.10.module_list.0.linear_ff` | 6.246 | 1.075 | 4/2048 |
| `conformer.module_list.11.module_list.0.linear_out` | 9.759 | 1.232 | 0/512 |
| `conformer.module_list.0.module_list.2.mhsa.qkv_proj` | 13.491 | 1.081 | 0/1536 |
| `conformer.module_list.11.module_list.3.linear_out` | 14.330 | 1.118 | 0/512 |
| `conformer.module_list.3.module_list.2.mhsa.linear_pos` | 18.309 | 1.297 | 0/512 |
| `conformer.module_list.1.module_list.1.depthwise_conv` | 20.010 | 1.164 | 0/512 |
| `conformer.module_list.3.module_list.2.mhsa.qkv_proj` | 21.382 | 1.217 | 0/1536 |
| `conformer.module_list.11.module_list.1.pointwise_conv2` | 21.679 | 1.083 | 0/512 |

Embedding/predictor/joiner weights are floating point in this model:
- `token_embedding`: 0.5405% exact zeros (not W4 sparsity).
- `prediction_net.1`: 0.0000% exact zeros (not W4 sparsity).
- `prediction_net.4`: 0.0000% exact zeros (not W4 sparsity).
- `joint_net.1`: 0.0000% exact zeros (not W4 sparsity).
- `joint_net.4`: 0.0000% exact zeros (not W4 sparsity).

## ft50_w8_to_w4

Checkpoint: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.CeKXLOB9QBnH/output/models/epoch.1000.pt`

| Group | Layers | Nonzero % | Layers below 10% nonzero | Empty highest bitplane |
| --- | ---: | ---: | ---: | ---: |
| encoder_ff | 48 | 45.862 | 3 | 0 |
| encoder_conv | 36 | 47.503 | 0 | 0 |
| encoder_mhsa | 36 | 33.410 | 0 | 0 |

### Ten sparsest encoder layers

| Layer | Nonzero % | Observer/current absmax | Zero output rows |
| --- | ---: | ---: | ---: |
| `conformer.module_list.10.module_list.0.linear_out` | 3.449 | 1.482 | 0/512 |
| `conformer.module_list.9.module_list.0.linear_out` | 6.846 | 1.185 | 0/512 |
| `conformer.module_list.10.module_list.0.linear_ff` | 7.876 | 1.113 | 2/2048 |
| `conformer.module_list.0.module_list.2.mhsa.qkv_proj` | 10.818 | 1.124 | 4/1536 |
| `conformer.module_list.11.module_list.0.linear_out` | 12.348 | 1.089 | 0/512 |
| `conformer.module_list.11.module_list.3.linear_out` | 15.776 | 1.115 | 0/512 |
| `conformer.module_list.3.module_list.2.mhsa.linear_pos` | 19.232 | 1.218 | 0/512 |
| `conformer.module_list.9.module_list.1.pointwise_conv2` | 19.728 | 1.473 | 0/512 |
| `conformer.module_list.5.module_list.2.mhsa.qkv_proj` | 20.175 | 1.198 | 0/1536 |
| `conformer.module_list.1.module_list.1.depthwise_conv` | 20.363 | 1.137 | 0/512 |

Embedding/predictor/joiner weights are floating point in this model:
- `token_embedding`: 0.5405% exact zeros (not W4 sparsity).
- `prediction_net.1`: 0.0000% exact zeros (not W4 sparsity).
- `prediction_net.4`: 0.0000% exact zeros (not W4 sparsity).
- `joint_net.1`: 0.0000% exact zeros (not W4 sparsity).
- `joint_net.4`: 0.0000% exact zeros (not W4 sparsity).

## scratch_encoder_w4

Checkpoint: `/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.Zc829L8iOl23/output/models/epoch.2000.pt`

| Group | Layers | Nonzero % | Layers below 10% nonzero | Empty highest bitplane |
| --- | ---: | ---: | ---: | ---: |
| encoder_ff | 48 | 43.101 | 2 | 0 |
| encoder_conv | 36 | 47.456 | 0 | 0 |
| encoder_mhsa | 36 | 34.283 | 0 | 0 |

### Ten sparsest encoder layers

| Layer | Nonzero % | Observer/current absmax | Zero output rows |
| --- | ---: | ---: | ---: |
| `conformer.module_list.10.module_list.0.linear_out` | 2.037 | 1.162 | 0/512 |
| `conformer.module_list.11.module_list.3.linear_out` | 5.612 | 1.075 | 0/512 |
| `conformer.module_list.9.module_list.0.linear_out` | 10.352 | 1.113 | 0/512 |
| `conformer.module_list.11.module_list.0.linear_out` | 10.419 | 1.133 | 0/512 |
| `conformer.module_list.11.module_list.1.pointwise_conv2` | 11.435 | 1.208 | 0/512 |
| `conformer.module_list.3.module_list.2.mhsa.linear_pos` | 14.889 | 1.191 | 0/512 |
| `conformer.module_list.1.module_list.2.mhsa.linear_pos` | 15.981 | 1.377 | 0/512 |
| `conformer.module_list.10.module_list.0.linear_ff` | 16.389 | 1.246 | 1/2048 |
| `conformer.module_list.2.module_list.2.mhsa.linear_pos` | 16.817 | 1.853 | 0/512 |
| `conformer.module_list.10.module_list.2.mhsa.linear_pos` | 17.150 | 1.927 | 0/512 |

Embedding/predictor/joiner weights are floating point in this model:
- `token_embedding`: 0.5405% exact zeros (not W4 sparsity).
- `prediction_net.1`: 0.0000% exact zeros (not W4 sparsity).
- `prediction_net.4`: 0.0000% exact zeros (not W4 sparsity).
- `joint_net.1`: 0.0000% exact zeros (not W4 sparsity).
- `joint_net.4`: 0.0000% exact zeros (not W4 sparsity).

## scratch_full_w4

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
