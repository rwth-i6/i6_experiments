# rt_r90 probe, sub-epoch 1 extraction (2026-09-25)

Job dir: `/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/work/i6_core/returnn/training/ReturnnTrainingJob.EexT85vdfx25`
(alias `alias/sae_i6/p1/ladder/rt_r90/training`). Read `log.run.1` (only log.run.* present),
`work/learning_rates`. Sub-epoch 1 = steps 0..56 (lines 623-761 in log.run.1), before the
`epoch.001.pt` save at line 766.

## 1. Steps and memory maxima (sub-epoch 1, 57 steps: step 0..56)
- steps in sub-epoch 1: 57 (step 0 through step 56)
- max `lexlat_k2_pre_peak_allocated_gib`: 27.256, at step 6 (line 647)
- max step's `mem_usage:cuda`: 27.0GB, at step 2 (line 641) [next highest: 26.9GB step 6 line 647; 26.5GB step 4 line 645]
- max `lexlat_k2_device_used_gib`: 41.311, first reached at step 54 (line 759), also 41.311 at step 55 (760) and step 56 (761)
- device-level used memory from `torch.cuda.mem_get_info`-style line: not printed per-step in this log; the only device-level readings found are the `lexlat_k2_device_used_gib` field above and the end-of-epoch `Memory usage (cuda): alloc cur 4.1GB alloc peak 13.2GB reserved cur 40.8GB reserved peak 40.8GB` (line right after Epoch 1 total loss line, before "Save model under ...")
- context only, max `lexlat_k2_peak_reserved_gib`: 40.789, at steps 54, 55, 56 (lines 759-761)

## 2. Error/anomaly grep over full log.run.1
- "out of memory": 0 occurrences
- "OutOfMemory": 0 occurrences
- "int32": 0 occurrences
- "ABORT": 0 occurrences
- "Traceback": 0 occurrences
- "nan": 0 occurrences
- "inf" as a non-finite score value: 0 occurrences (only "INFO"/"info" substrings present, excluded)

## 3. lexlat_k2_ABORT.json
- `find <job_dir> -iname "*ABORT*"` returned nothing: file does not exist anywhere in the job dir.

## 4. Stability line and learning_rates epoch-1 entry
- Verbatim (log.run.1, line 621): `lexlat_k2: stability at sub-epoch 1: median 0.0826 nats per retained frame over 16 of 16 utterances (|log Z(1000) - log Z(10000)|)`
- `work/learning_rates`, epoch 1 entry: yes, holds `'train_loss_lexlat_k2_stability': 0.08260878920555115,`

## 5. Step time and wall time
- sec/step values collected from all 57 "ep 1 train, step N, ... sec/step" lines
- median sec/step: 78.113
- max sec/step: 103.545 (step 0, line 623)
- sub-epoch 1 wall time: from `Starting training at epoch 1, global train step 0` (line 246, timestamp context at [2026-09-25 15:56:59,678] "Run time: 0:00:10") through `Epoch 1: Trained 57 steps, 1:10:25 elapsed` (log.run.1 line 762) to the `Save model under .../epoch.001.pt` line (line 766). RETURNN-reported elapsed at the epoch summary: `1:10:25`. Checkpoint file `output/models/epoch.001.pt` mtime: 17:07 (per job-dir listing); job start Slurm/log timestamp: 2026-09-25 15:56:49 (line 21, "Start Job").

## 6. Epoch-1 train scores, verbatim (log.run.1, line 764)
`Epoch 1: Total train loss: l_tau 2.537 agg 1.483 rate 0.019 lexlat_k2 0.563 blankfree_l_tau_per_frame 2.537 blankfree_agg_kl_unigram 0.319 blankfree_agg_kl_bigram 1.165 blankfree_reverse_per_frame -5.019 blankfree_prior_per_token -2.560 blankfree_phone_rate_original_hz 8.979 blankfree_phone_rate_retained_hz 10.506 blankfree_expected_phone_rate_hz 8.627 blankfree_expected_tokens 114.151 blankfree_z_zero_frac 0.000e+00 blankfree_rate_fd_check 1.782e-05 blankfree_rate_dp_calls 1.982 blankfree_temperature 2.000 blankfree_frames_per_sec 890.466 lexlat_k2_lam 0.333 lexlat_k2_term_mean -0.563 lexlat_k2_n_empty 0.000e+00 lexlat_k2_empty_frac 0.000e+00 lexlat_k2_lattice_arcs_per_frame 2.125e+03 lexlat_k2_lattice_states_per_frame 778.293 lexlat_k2_expected_words 29.312 lexlat_k2_expected_escape_words 3.157e-05 lexlat_k2_sec 16.240 lexlat_k2_peak_reserved_gib 40.205 lexlat_k2_pre_peak_allocated_gib 24.564 lexlat_k2_pre_peak_reserved_gib 40.030 lexlat_k2_device_used_gib 40.726 lexlat_k2_stability 0.083`

Also immediately preceding line 764: `Epoch 1: Trained 57 steps, 1:10:25 elapsed (99.8% computing time, features: 7.3%, original_length: 0.0%, units: 7.3% padding)`, and a `Learning-rate-control: error key 'train_loss_agg' from {...}` dict line with the same values plus full-precision floats (see log.run.1 for exact line).

## 7. First step of sub-epoch 2
MISSING (searched: log.run.1 for "^ep 2 train, step 0," and "Starting training at epoch 2" — neither pattern found; job was still running at time of read, no sub-epoch 2 step logged yet)
