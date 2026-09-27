# Back-Translation Probe Results (2026-09-15)

## Summary Table: Per-Arm Verdicts

| Arm | rho_hz | best_PER | ANY-round takes-off | FINAL-round takes-off |
|-----|--------|---------|--------------------|-----------------------|
| warm_phi/renderer | 9.6619373279 | 1.3706450408152377 | False | False |
| warm_phi/collage | 9.6619373279 | 0.31419730032545484 | True | True |
| phi_0/collage_spk_run | 9.6619373279 | 0.8969748706183642 | False | False |
| phi_0/centroid | 9.6619373279 | 1.5062156538441018 | False | False |
| warm_phi/collage_spk_run | 9.6619373279 | 0.33436482953636026 | True | True |
| phi_0/collage | 9.6619373279 | 0.8668836365576482 | False | False |
| phi_0/renderer | 9.6619373279 | 1.2463853171850825 | False | False |
| warm_phi/centroid | 9.6619373279 | 0.6947126927386225 | True | False |

## Pre-registered Reading (from SAE_4A.md)

**Takes-off criterion:**
- dev-other greedy PER < 0.80
- AND emitted phone rate (SIL excluded) within [0.6, 1.5] x rho
- AND speaker-matched derangement gap > 0 with 95% CI excluding 0

**ANY-round verdict:** Takes-off in any of rounds 0-3

**FINAL-round verdict:** Takes-off at round 3 only

## Detailed Results by Arm

### phi_0/centroid

**rho_hz:** 9.6619373279

**Pre-registered Verdicts:**
- ANY-round takes-off: False
- FINAL-round takes-off: False
- best_per: 1.5062156538441018

**Round-by-Round Data:**

| Round | PER | n_utts | emitted_per_sec (SIL excl) | rate_in_band | gap (CI 95%) | feasible_frac | distinct_strings | CTC_loss (first→last) |
|-------|-----|--------|----------------------------|--------------|-------------|---------------|------------------|----------------------|
| 0 | 1.5062 | 300 | 15.6094 | False | -0.0638 [-0.0984, -0.0332] | 0.0 | 300 | 3.2988→0.2965 |
| 1 | 1.9091 | 300 | 20.2517 | False | -0.0661 [-0.1313, -0.0154] | 0.961 | 300 | 1.3509→0.8817 |
| 2 | 1.8213 | 300 | 19.5116 | False | -0.0473 [-0.0900, -0.0112] | 0.7595 | 300 | 1.3292→1.0915 |
| 3 | 1.7559 | 300 | 18.7120 | False | -0.0529 [-0.0997, -0.0140] | 0.7885 | 300 | 1.1012→1.0604 |

### phi_0/collage

**rho_hz:** 9.6619373279

**Pre-registered Verdicts:**
- ANY-round takes-off: False
- FINAL-round takes-off: False
- best_per: 0.8668836365576482

**Round-by-Round Data:**

| Round | PER | n_utts | emitted_per_sec (SIL excl) | rate_in_band | gap (CI 95%) | feasible_frac | distinct_strings | CTC_loss (first→last) |
|-------|-----|--------|----------------------------|--------------|-------------|---------------|------------------|----------------------|
| 0 | 0.8669 | 300 | 3.6953 | False | -0.0001 [-0.0032, 0.0028] | 0.0 | 299 | 3.2988→0.3254 |
| 1 | 0.8706 | 300 | 3.2133 | False | 0.0057 [-0.0273, 0.0341] | 0.9305 | 298 | 0.3214→0.3161 |
| 2 | 0.8855 | 300 | 2.5565 | False | -0.0202 [-0.0590, 0.0159] | 0.95 | 282 | 0.3432→0.3349 |
| 3 | 0.8752 | 300 | 2.6842 | False | -0.0314 [-0.0557, -0.0057] | 0.9345 | 296 | 0.3223→0.3064 |

### phi_0/collage_spk_run

**rho_hz:** 9.6619373279

**Pre-registered Verdicts:**
- ANY-round takes-off: False
- FINAL-round takes-off: False
- best_per: 0.8969748706183642

**Round-by-Round Data:**

| Round | PER | n_utts | emitted_per_sec (SIL excl) | rate_in_band | gap (CI 95%) | feasible_frac | distinct_strings | CTC_loss (first→last) |
|-------|-----|--------|----------------------------|--------------|-------------|---------------|------------------|----------------------|
| 0 | 0.9474 | 300 | 1.7362 | False | 0.0034 [-0.0128, 0.0232] | 0.0 | 200 | 3.2988→0.4165 |
| 1 | 0.8970 | 300 | 2.3299 | False | -0.0128 [-0.0449, 0.0190] | 0.329 | 290 | 0.4010→0.3841 |
| 2 | 0.9135 | 300 | 1.8681 | False | -0.0456 [-0.1030, 0.0045] | 0.564 | 268 | 0.5923→0.3814 |
| 3 | 0.9057 | 300 | 1.7486 | False | -0.0294 [-0.0812, 0.0205] | 0.441 | 280 | 0.3575→0.3444 |

### phi_0/renderer

**rho_hz:** 9.6619373279

**Pre-registered Verdicts:**
- ANY-round takes-off: False
- FINAL-round takes-off: False
- best_per: 1.2463853171850825

**Round-by-Round Data:**

| Round | PER | n_utts | emitted_per_sec (SIL excl) | rate_in_band | gap (CI 95%) | feasible_frac | distinct_strings | CTC_loss (first→last) |
|-------|-----|--------|----------------------------|--------------|-------------|---------------|------------------|----------------------|
| 0 | 1.2464 | 300 | 13.7625 | True | -0.0131 [-0.0289, 0.0026] | 0.0 | 300 | 3.2988→0.3985 |
| 1 | 1.9907 | 300 | 21.8596 | False | -0.1987 [-0.2868, -0.1231] | 1.0 | 300 | 7.5388→0.9275 |
| 2 | 1.7479 | 300 | 19.2437 | False | 0.0214 [-0.0048, 0.0476] | 0.859 | 300 | 1.4864→1.4786 |
| 3 | 1.7231 | 300 | 18.9742 | False | 0.0076 [-0.0183, 0.0337] | 0.9505 | 300 | 1.5449→1.3756 |

### warm_phi/centroid

**rho_hz:** 9.6619373279

**Pre-registered Verdicts:**
- ANY-round takes-off: True
- FINAL-round takes-off: False
- best_per: 0.6947126927386225

**Round-by-Round Data:**

| Round | PER | n_utts | emitted_per_sec (SIL excl) | rate_in_band | gap (CI 95%) | feasible_frac | distinct_strings | CTC_loss (first→last) |
|-------|-----|--------|----------------------------|--------------|-------------|---------------|------------------|----------------------|
| 0 | 0.6947 | 300 | 9.6167 | True | 1.5146 [1.3543, 1.6920] | 0.0 | 300 | 3.2837→0.1501 |
| 1 | 0.8140 | 300 | 11.5758 | True | 1.3193 [1.2246, 1.4209] | 0.999 | 300 | 0.6966→0.4081 |
| 2 | 0.9622 | 300 | 13.3694 | True | 1.2968 [1.2047, 1.3920] | 1.0 | 300 | 0.5432→0.5137 |
| 3 | 1.0582 | 300 | 14.1798 | True | 1.2505 [1.1674, 1.3350] | 1.0 | 300 | 0.4359→0.5157 |

### warm_phi/collage

**rho_hz:** 9.6619373279

**Pre-registered Verdicts:**
- ANY-round takes-off: True
- FINAL-round takes-off: True
- best_per: 0.31419730032545484

**Round-by-Round Data:**

| Round | PER | n_utts | emitted_per_sec (SIL excl) | rate_in_band | gap (CI 95%) | feasible_frac | distinct_strings | CTC_loss (first→last) |
|-------|-----|--------|----------------------------|--------------|-------------|---------------|------------------|----------------------|
| 0 | 0.3630 | 300 | 7.7387 | True | 2.7393 [2.5882, 2.9043] | 0.0 | 300 | 3.2837→0.2226 |
| 1 | 0.3271 | 300 | 7.7237 | True | 2.8464 [2.7385, 2.9607] | 0.9995 | 299 | 0.3479→0.2498 |
| 2 | 0.3235 | 300 | 7.9932 | True | 3.3130 [3.1706, 3.4608] | 1.0 | 299 | 0.3518→0.1795 |
| 3 | 0.3142 | 300 | 8.4778 | True | 3.5518 [3.4023, 3.7068] | 1.0 | 300 | 0.1667→0.2100 |

### warm_phi/collage_spk_run

**rho_hz:** 9.6619373279

**Pre-registered Verdicts:**
- ANY-round takes-off: True
- FINAL-round takes-off: True
- best_per: 0.33436482953636026

**Round-by-Round Data:**

| Round | PER | n_utts | emitted_per_sec (SIL excl) | rate_in_band | gap (CI 95%) | feasible_frac | distinct_strings | CTC_loss (first→last) |
|-------|-----|--------|----------------------------|--------------|-------------|---------------|------------------|----------------------|
| 0 | 0.3866 | 300 | 7.3814 | True | 2.5905 [2.3940, 2.7981] | 0.0 | 300 | 3.2837→0.2700 |
| 1 | 0.3344 | 300 | 7.8096 | True | 2.9011 [2.8027, 3.0044] | 0.996 | 300 | 0.5541→0.2516 |
| 2 | 0.3362 | 300 | 7.8629 | True | 3.2443 [3.1046, 3.3908] | 1.0 | 299 | 0.2773→0.2603 |
| 3 | 0.3391 | 300 | 8.0677 | True | 3.4979 [3.3544, 3.6500] | 1.0 | 300 | 0.3122→0.3463 |

### warm_phi/renderer

**rho_hz:** 9.6619373279

**Pre-registered Verdicts:**
- ANY-round takes-off: False
- FINAL-round takes-off: False
- best_per: 1.3706450408152377

**Round-by-Round Data:**

| Round | PER | n_utts | emitted_per_sec (SIL excl) | rate_in_band | gap (CI 95%) | feasible_frac | distinct_strings | CTC_loss (first→last) |
|-------|-----|--------|----------------------------|--------------|-------------|---------------|------------------|----------------------|
| 0 | 1.3706 | 300 | 16.1964 | False | 0.5224 [0.4478, 0.6049] | 0.0 | 300 | 3.2837→0.1806 |
| 1 | 1.8717 | 300 | 20.7218 | False | 0.0504 [0.0042, 0.0952] | 1.0 | 300 | 1.7961→0.9181 |
| 2 | 2.0851 | 300 | 22.7663 | False | -0.0723 [-0.1159, -0.0316] | 0.957 | 300 | 1.6684→1.3878 |
| 3 | 2.1320 | 300 | 23.1443 | False | -0.0110 [-0.0600, 0.0464] | 0.619 | 300 | 1.5371→1.5003 |
