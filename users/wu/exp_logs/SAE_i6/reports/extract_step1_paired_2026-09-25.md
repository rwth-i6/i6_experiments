# Extract: step-0/1 fields, paired reads, PER jsons (2026-09-25)

Job dirs (readlink -f of the 4 aliases):
- ctrl_20: /work/asr4/hwu/setups/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/i6_core/returnn/training/ReturnnTrainingJob.GiT88bxzoZbZ
- ctrl_20_s1: .../ReturnnTrainingJob.DvVfxf1LrCBi
- ctrl_20_rc: .../ReturnnTrainingJob.llSFybyKXkbL
- k2lat_20_ma3000: .../ReturnnTrainingJob.jcKXbLMDk4hl

## 1. "ep 1 train, step 0" line, verbatim, per run

ctrl_20 (log.run.1:670) and k2lat (log.run.1:967) have IDENTICAL verbatim lines:
`ep 1 train, step 0, l_tau -0.352, agg 1.789, rate 0.069, blankfree_l_tau_per_frame -0.352, blankfree_agg_kl_unigram 0.403, blankfree_agg_kl_bigram 1.386, blankfree_reverse_per_frame -7.209, blankfree_prior_per_token -5.637, blankfree_phone_rate_original_hz 12.400, blankfree_phone_rate_retained_hz 15.091, blankfree_expected_phone_rate_hz 11.874, blankfree_expected_tokens 63.854, blankfree_z_zero_frac 0.000e+00, blankfree_rate_fd_check 7.701e-06, blankfree_rate_dp_calls 2.000, blankfree_temperature 8.000, blankfree_frames_per_sec 0.000e+00, num_seqs 128, max_size:time:var-unk:features 330, max_size:time:var-unk:units 330, max_size:time:var-unk:original_length 1, <sec/step differs>, elapsed <differs>, exp. remaining <differs>, complete 1.87%`
  - ctrl_20: 37.787 sec/step, elapsed 0:00:46, exp. remaining 0:41:07
  - k2lat: 25.741 sec/step, elapsed 0:23:42, exp. remaining 20:45:45
  - Both preceded by identical line "ep 1 train num_seqs: 7066" (ctrl_20 log.run.1:669 area; k2lat log.run.1 same text, un-numbered here).
  - No log_z/logZ field and no lexlat/k2 field on this line for either run (checked: only "lexlat"/"k2" mentions in k2lat's log are earlier resource-loading lines, e.g. log.run.1:49, :353, unrelated to the step-0 metrics line).

ctrl_20_s1 and ctrl_20_rc: **MISSING** (searched: work/i6_core/returnn/training/ReturnnTrainingJob.{DvVfxf1LrCBi,llSFybyKXkbL}/log.run.1, the only log file present in either job dir — no log.run.2/3). Both logs begin at "start epoch 4 global train step 171 ..." (s1: log.run.1:295; rc: similarly) — i.e. these jobs were resumed starting already at sub-epoch 4; no "ep 1 train, step 0" line, and no earlier log.run.N exists in these job directories to search. Their first present step-0 lines are "ep 4 train, step 0" (s1: log.run.1:795; rc: log.run.1:832).
  - s1 ep4 step0: l_tau 2.675, blankfree_prior_per_token -2.876, blankfree_expected_tokens 131.994, num_seqs 111, max_size:time:var-unk:features/units 791, no log_z/lexlat/k2 fields.
  - rc ep4 step0: l_tau 2.682, blankfree_prior_per_token -2.841, blankfree_expected_tokens 132.637, num_seqs 109, max_size:time:var-unk:features/units 803, no log_z/lexlat/k2 fields.
  - No field with "log Z"/"logZ"/"log_z" found anywhere in s1's or rc's log.run.1 (grepped).

GPU for k2lat: log.run.1 lines 75/346: "Tesla V100-SXM3-32GB" (Slurm partition gpu_32gb, log.run.1:8).

Sequence tags/sizes at step 0: **not logged** for any of the four runs — no "seq_tag"/"sequence tag" string found in ctrl_20's or k2lat's log.run.1 (grepped, zero hits). Cannot compare identity across runs.

## 2. ctrl_20 vs k2lat step-1 numeric diffs
All numeric fields on the "ep 1 train, step 0" line are IDENTICAL between ctrl_20 and k2lat (abs diff 0, rel diff 0) except the two throughput/wallclock fields, which differ by machine/queue conditions, not by data: sec/step 37.787 (ctrl_20) vs 25.741 (k2lat), diff -12.046 (-31.88% rel to ctrl_20); elapsed and exp.remaining differ (wallclock, not a metric).

## 3. dev-other greedy PER json (ep4)
ctrl_20_s1: /u/.../output/sae/4a/ctrl_20_s1/ep4/dev-other/per.json -> resolves to i6_experiments/.../analysis/per/BlankfreeGreedyPerJob.sa5M4QGL1ALP/output/per.json
  per 0.8778056691580877, sub 128130, del 16185, ins 11298, reference_phones 177275
ctrl_20_rc: /u/.../output/sae/4a/ctrl_20_rc/ep4/dev-other/per.json -> resolves to .../BlankfreeGreedyPerJob.yyfOivL459Dr/output/per.json
  per 0.884360456917219, sub 127820, del 17557, ins 11398, reference_phones 177275

## 4. Paired reads (dev-other, from paired_per.json / summary.txt)
Sign convention (verbatim from json "convention" field): delta_per = (sum d_B - sum d_A)/sum N; A=baseline, B=candidate; NEGATIVE = B better; "refines" iff 95% CI of delta_per lies entirely below 0. CI: speaker-clustered bootstrap, 2000 resamples, seed 0, [2.5,97.5] pct.

ctrl_20_rc_vs_ctrl_20/ep1 (A=ctrl_20_rc? NOTE baseline_name field = "ctrl_20" -> A=ctrl_20, B=ctrl_20_rc despite dir name order):
  delta_per +0.0004907629389366841, CI95 [0.0, 0.0010233799670083222], utterances 2864, speakers 33, cluster "speaker", excludes_zero false, refines false, macro +0.000543537384980407 CI [-1.5212119630947571e-05, 0.0011376621886974726]

ctrl_20_rc_vs_ctrl_20/ep4: delta_per +0.007175292624453489, CI95 [0.00461019907563515, 0.009762420684984272], utterances 2864, speakers 33, excludes_zero true, refines false, macro +0.0031921098586428324 CI [-0.00036157292071579815, 0.006637704856262321]

ctrl_20_vs_ctrl_20_s1/ep1 (baseline_name "ctrl_20_s1" -> A=ctrl_20_s1, B=ctrl_20): delta_per +0.0031984205330701787, CI95 [0.00037366257193294596, 0.005913072774067159], utterances 2864, speakers 33, excludes_zero true, refines false, macro +0.004237414450214632 CI [0.0015043894507891949, 0.006977412149757765]

ctrl_20_vs_ctrl_20_s1/ep4: delta_per -0.000620504865322169, CI95 [-0.003683075402070022, 0.0024538880413389353], utterances 2864, speakers 33, excludes_zero false, refines false, macro +0.0035818806774688488 CI [-0.0009496950299201861, 0.00810875143943274]

Paths (job dirs): PairedPerDeltaJob.X3oIpq9LvT8G (rc_vs_ctrl/ep1), .dJygEbSwG8QV (rc_vs_ctrl/ep4), .fo78CC19PVd6 (ctrl_vs_s1/ep1), .JJWsDSp9fHbC (ctrl_vs_s1/ep4), all under i6_experiments/users/wu/experiments/unsupervised_asr/analysis/paired/

## 5. "emitted greedy rate (/s)"
MISSING as a named field (searched: learning_rates files of ctrl_20/s1/rc job dirs' work/learning_rates, and both per.json files). learning_rates only contains dev_loss_blankfree_phone_rate_original_hz / _retained_hz / _expected_phone_rate_hz per epoch (Hz, not "/s" emitted-rate). per.json contains phone_rate_original_hz and phone_rate_retained_hz (s1 ep4: 9.369116719928694 / 11.034597535605696 Hz; rc ep4: 9.299984782277875 / 10.953176508241318 Hz). learning_rates epoch-1/epoch-4 learningRate values (not a "rate/s"): s1 & rc both ep1 learningRate=0.0001, ep4 learningRate=7.75e-05, error={} in the short-form repr (per-metric values are elsewhere in the file's long dict, not reproduced here).
