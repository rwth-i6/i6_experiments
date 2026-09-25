# Extract: ctrl_20 ep20 gate G0.R1 (2026-09-25)

ctrl_20 job (resolved): S/work/i6_core/returnn/training/ReturnnTrainingJob.GiT88bxzoZbZ

## 1. ep20 dev-other greedy PER
File: S/alias/sae/4a/blankfree/ctrl_20/ep20/dev-other/per -> readlink -f ->
/work/asr4/hwu/setups/.../analysis/per/BlankfreeGreedyPerJob.LbLZ8pK3RIm9/output/per.json (identical content also at .../decode_stats.json)
- per = 0.886233253419828
- sub = 129496
- del = 18234
- ins = 9377
- reference_phones = 177275
- emitted rate: field "phone_rate_original_hz" = 9.153351159807823 (/s). (At ep1 this same field read 3.2551794604230526, matching the brief's "3.255 /s".) A second field "phone_rate_retained_hz" = 10.780476876300208 also present in same file.

## 2. Gaps (ep20, dev-other)
derangement_gap -> BlankfreeDerangementGapJob.3gZNfM04VHa8/output/derangement_gap.json:
  gap = 4.234650811801392, matched_utterances=500, selected=500, feasible=500, masked_frames=130684, own_logp_per_frame=-3.4134671617950283, deranged_logp_per_frame=-7.64811797359642

decode_gap -> BlankfreeDecodeGapJob.C2mhWfaciCef/output/decode_gap.json:
  registered_decoded_gap_per_frame=4.234650811801392, n_boot=2000, seed=0, selected=500, decoded_feasible=500, matched_decoded=500, utterances=500, speakers=33, cluster="speaker", masked_frames=130684.0, gold_tokens=30690.0, reference_gap_tolerance=0.001, frac_decoded_above_gold=1.0, n_decoded_above_gold=500

derangement_gap_items -> ReverseGapItemsJob.TngR92EpUvxY/output/items.json: split=dev-other, with_gold=False, selected=500, decoded_feasible=500, n_items=1000
decode_gap_items -> ReverseGapItemsJob.QrX8dZR2QAFn/output/items.json: split=dev-other, with_gold=True, selected=500, decoded_feasible=500, n_items=2000

## 3. learning_rates epoch 20 (dev scores), S/.../work/learning_rates line 914 block
dev_loss_agg = 1.3825716177622478
dev_loss_blankfree_agg_kl_bigram = 1.2565797567367554
dev_loss_blankfree_agg_kl_unigram = 0.12599187592665353
dev_loss_blankfree_expected_phone_rate_hz = 9.123482386271158
dev_loss_blankfree_expected_tokens = 127.35457611083984
dev_loss_blankfree_frames_per_sec = 4229.133219401042
dev_loss_blankfree_l_tau_per_frame = 1.8019438187281291
dev_loss_blankfree_phone_rate_original_hz = 9.535022417704264
dev_loss_blankfree_phone_rate_retained_hz = 10.962497393290201
dev_loss_blankfree_prior_per_token = -3.337722142537435
dev_loss_blankfree_rate_dp_calls = 1.6666666666666667
dev_loss_blankfree_rate_fd_check = 7.559902163241834e-06
dev_loss_blankfree_reverse_per_frame = -3.20613161722819
dev_loss_blankfree_temperature = 2.0
(":meta:" keys also at line 914 block: epoch_num_train_steps=57, global_train_step=1083, global_train_step_end=1140, epoch_train_time_secs=3241, device="NVIDIA L40S", hostname="cn-508")

epoch_train_time_secs: epoch18=3246, epoch19=3247, epoch20=3241
global_train_step_end epoch20 = 1140

## 4. Checkpoint / log
output/models/epoch.020.pt exists: yes, 7189939 bytes, mtime Sep 25 18:46
log.run.1 last 3 lines:
  warnings.warn('resource_tracker: %r: %s' % (name, e))
  [2026-09-25 18:49:29,637] INFO: Max resources: Run time: 18:56:09 CPU: 115.3% RSS: 12.16GB VMS: 125.37GB
  [2026-09-25 18:49:29,638] INFO: Job finished successfully

## 5. ep10 comparison arms (finished) and paired reads
ctrl_20_s1 ep10 per.json (S/output/sae/4a/ctrl_20_s1/ep10/dev-other/per.json): per=0.8632632914962629, sub=124577, del=20210, ins=8248, reference_phones=177275
ctrl_20_rc ep10 per.json (S/output/sae/4a/ctrl_20_rc/ep10/dev-other/per.json): per=0.8954844168664504, sub=131149, del=18239, ins=9359, reference_phones=177275

Paired: ctrl_20_vs_ctrl_20_s1/ep10 -> PairedPerDeltaJob.BVDlYxEdQj6c/output/paired_per.json
  baseline_name="ctrl_20_s1" (A=ctrl_20_s1, B=ctrl_20); per_a=0.8632632914962629, per_b=0.8846537864899168
  delta_per=0.021390494993653864, CI95=[0.017921122438658708, 0.02478154354830998], utterances=2864, speakers=33, refines=false

Paired: ctrl_20_rc_vs_ctrl_20/ep10 -> PairedPerDeltaJob.NjpEZQICzSK0/output/paired_per.json
  baseline_name="ctrl_20" (A=ctrl_20, B=ctrl_20_rc); per_a=0.8846537864899168, per_b=0.8954844168664504
  delta_per=0.010830630376533645, CI95=[0.008483352085394486, 0.01337128769381804], utterances=2864, speakers=33, refines=false

Per convention (S/reports/extract_step1_paired_2026-09-25.md sec.4): delta_per=(sum d_B - sum d_A)/sum N; negative=B better.
