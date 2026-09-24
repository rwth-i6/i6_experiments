# NodeA Epoch 1 Extraction - 2026-09-20

Job dir: `/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL`

## Arm: ctrl_50

### Most Recent ~5 Step Values (Epoch 19)

| Step | blankfree_l_tau_per_frame | l_tau | agg | rate | blankfree_temperature | blankfree_frames_per_sec | blankfree_phone_rate_original_hz | blankfree_reverse_per_frame | blankfree_prior_per_token |
|------|---------------------------|-------|-----|----|------------------------|--------------------------|----------------------------------|---------------------------|-------------------------|
| 26 | 1.795 | 1.795 | 1.294 | 0.010 | 2.000 | 6.589e+03 | 9.255 | -3.127 | -3.464 |
| 27 | 1.784 | 1.784 | 1.294 | 0.016 | 2.000 | 6.592e+03 | 8.984 | -3.104 | -3.479 |
| 28 | 1.784 | 1.784 | 1.295 | 0.017 | 2.000 | 6.603e+03 | 8.973 | -3.101 | -3.462 |
| 29 | 1.786 | 1.786 | 1.296 | 0.019 | 2.000 | 6.417e+03 | 8.935 | -3.104 | -3.455 |
| 30 | 1.797 | 1.797 | 1.296 | 0.026 | 2.000 | 5.542e+03 | 8.974 | -3.121 | -3.461 |

### Training Progress

- **Current Epoch**: 19
- **Current Step**: 30 (within epoch 19)
- **Total Global Step**: 1056 (start of epoch 19 was step 1026, so 1026+30=1056)
- **Steps per Epoch**: 57 (constant across all epochs)
- **Total Elapsed Since Job Start**: 3.176 hours (based on used_time in usage.run.1)
- **Epoch 19 Elapsed So Far**: 0:05:29 (5 minutes 29 seconds)
- **Wall Time per Finished Sub-epoch (Epochs 1-18)**: Approximately 616.86 seconds (10 minutes 17 seconds) per epoch
  - Calculation: (3.176 hours - 5:29) / 18 epochs = 11103.4 sec / 18 = 616.86 sec/epoch

### Batch Shape Information

From most recent step (step 30):
- **max_seqs**: 128 (number of sequences per batch)
- **max_size:time:var-unk:features**: 500 (max time dimension for features)
- **max_size:time:var-unk:units**: 500 (max time dimension for units, same as features)
- **max_size:time:var-unk:original_length**: 1

From recent steps (26-30):
- max_seqs varies: 128 (steps 26,27,28,30), 128 (step 29)
- max_size:time varies: 654 (step 26), 620 (step 27), 594 (step 28), 560 (step 29), 500 (step 30)

### GPU Memory

- **GPU Device**: cuda device 1: NVIDIA GH200 120GB
- **Total GPU 1 Memory**: 95.0 GB
- **Free Memory at Job Start**: 94.4 GB (99%)

### Step Time Analysis

- Step 26: 11.783 sec/step
- Step 27: 11.193 sec/step
- Step 28: 10.708 sec/step
- Step 29: 10.114 sec/step
- Step 30: 9.031 sec/step

Average for last 5 steps: ~10.57 sec/step

### Learning Rates

From learning_rates file (final epoch summary for epoch 19):
- epoch 19: learningRate=0.0001, error={}
- All measured metrics in epoch 19 final averaged values shown in learning_rates file

## Comparison Note

Arm odmprior_50 was NOT selected for detailed reporting as its returnn.log (1.29 MB) is larger than ctrl_50 (896 KB), indicating it has done more computation and is not equally cheap.

## Data Sources

- returnn.log: `/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/work/ctrl_50/returnn.log`
- learning_rates: `/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/work/ctrl_50/learning_rates`
- usage.run.1: `/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/usage.run.1`

