# S3b-R lam3 Training Job DF6blPpto23t: Metrics Extract

**Report Date:** 2026-09-15  
**Setup:** /e/project1/spell/wu24/2026-07-13_unsupervised  
**Training Job:** work/i6_core/returnn/training/ReturnnTrainingJob.DF6blPpto23t  
**SLURM Job ID:** 1818726  
**Training Status:** RUNNING (epoch 2, step 28, 49% complete as of this report)

---

## Training Metrics from work/learning_rates

### Epoch 1 (COMPLETE)

| Metric | Value | Source |
|--------|-------|--------|
| dev_loss_emc | 1.2292395035425823 | learning_rates epoch 1 |
| dev_loss_emc_agg | 1.2292395035425823 | learning_rates epoch 1 |
| dev_loss_emc_expected_phone_rate_hz | 10.914304733276367 | learning_rates epoch 1 |
| dev_loss_emc_phone_rate | 15.859437624613443 | learning_rates epoch 1 |
| dev_loss_emc_rate_dp_calls | 1.6666666666666667 | learning_rates epoch 1 |
| dev_loss_emc_rate_fd_check | 4.3740825276472606e-05 | learning_rates epoch 1 |
| dev_loss_l_tau | -0.22186102966467539 | learning_rates epoch 1 |
| train_loss_emc | 1.4301161018468567 | learning_rates epoch 1 |
| train_loss_emc_phone_rate | 16.271853867223708 | learning_rates epoch 1 |
| train_loss_emc_rate_dp_calls | 2.0 | learning_rates epoch 1 |
| train_loss_emc_rate_fd_check | 4.5813011125974914e-05 | learning_rates epoch 1 |

### Epochs 2-8 (IN PROGRESS)
- **Epochs 2-8:** error dicts are empty in learning_rates (no eval metrics yet, training still running)

---

## Key Metric Definitions

**dev_loss_emc_rate_dp_calls:**  
Reports whether the two central-difference tilted passes for the rate term's gradient ran stacked (1 DP call) or sequentially (2 DP calls) at this step. Value 1.667 (epoch 1) indicates the batch budget was sometimes not sufficient for stacking.  
*Source: sae_emc.py line marking "DP CALLS the term's gradient cost this step"*

**dev_loss_emc_rate_fd_check:**  
A pooled relative deviation monitor: sum_kept |E_fd - E[N]| / max(sum_kept E[N], 1e-3). Checks consistency between the finite-difference gradient estimate (tau * dlogZ/db) and the direct expected token count. This is a monitor, never a gate.  
*Source: rate_term.py, "pre-registered convention"*

**dev_loss_emc_phone_rate & dev_loss_emc_expected_phone_rate_hz:**  
Expected non-SIL token rate (phones/s) under the posterior. Phone rate = 15.859 /s; expected token rate (assuming 50 Hz frame rate) = 10.914 tokens/s.

---

## READ Jobs: GreedyPerJob and DecodeStatsJob Results

### Epoch 1 - dev-other (COMPLETE)

**GreedyPerJob output (per.txt):**
```
greedy PER, split=dev-other, utts=2864, convention (S+D+I)/N over the split
PER = 0.867827  (S 61807 / D 91796 / I 241 / N 177275)
PER macro (mean over utterances) = 0.882551
emitted phones/s = 4.6588 (with SIL 7.5439); reference phones/s = 9.6347
frames = 919980 (5.111 h at 50.0 Hz)
```

**DecodeStatsJob outputs:**
- **emitted phone rate (per s):** 4.658796930368052  
  (from: output/exp2025_11_06_speech_llms/librispeech/sae_4a_s3b_rate/lam3/ep1/dev-other/phone_rate)
- **distinct decoded strings:** 2864  
  (from: output/exp2025_11_06_speech_llms/librispeech/sae_4a_s3b_rate/lam3/ep1/dev-other/distinct_strings)

**Epoch 1 - dev-clean:** Files exist in output directory, not extracted in this report.

---

## READ Jobs: S3DerangementGapJob (Gap reads)

**Epochs 4 and 8 (gated epochs):** No gap job results yet (training is currently in epoch 2).  
These will be available once training reaches epochs 4 and 8.

---

## Execution Timeline

| Item | Value |
|------|-------|
| SLURM elapsed time | 00:08:57 |
| SLURM requested time | 6.0 h |
| Current training epoch | 2 |
| Current training step | 28 / ~59 steps per epoch |
| Training progress | ~49% complete (epoch 2) |
| Epoch 1 completed | ~18 minutes from job start |
| Epoch 2 reads completed | ~23 minutes from job start |

---

## Summary of Missing Reads

- **Epochs 2-3:** No dev metrics in learning_rates yet; READ jobs created but awaiting results
- **Epochs 4-8:** No directories exist; training in progress
- **Gap (S3DerangementGapJob):** Will appear at epochs 4 and 8 only; not yet launched

---

**Full paths referenced:**
- Training job work dir: /e/project1/spell/wu24/2026-07-13_unsupervised/work/i6_core/returnn/training/ReturnnTrainingJob.DF6blPpto23t/
- Learning rates: work/learning_rates
- Output directory: /e/project1/spell/wu24/2026-07-13_unsupervised/output/exp2025_11_06_speech_llms/librispeech/sae_4a_s3b_rate/lam3/

