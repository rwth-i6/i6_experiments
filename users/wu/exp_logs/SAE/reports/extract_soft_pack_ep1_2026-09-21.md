# PackedBlankfreeTrainJob.MXKoywbfon8O State Report

Extract as of 2026-09-21 18:08:25.

## Job Overview
- Job ID: work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O
- Start time: 2026-09-21 17:06:29 UTC
- Pack config: 4 arms on one 4-GPU node, N=20 sub-epochs, 601 s/sub-epoch target
- Peak RSS: 52.03179931640625 GB

## Per-Arm Progress

### sf_20
- Current sub-epoch: 4 (completed)
- Step-1 loss (train_loss_agg): 1.792
- Epoch 1 loss (dev_loss_agg): 1.5167909065882366 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/sf_20/learning_rates:15`
- Wall seconds per completed sub-epoch (average): 717 s (ratio to 601 s: 1.193)
  - Epoch times: 729, 714, 715, 710 seconds - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/sf_20/learning_rates:8,82,156,230`
- Epoch 4 dev monitor columns:
  - sf_reward_mean: -0.5098998149236044 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/sf_20/learning_rates:260`
  - blankfree_phone_rate_retained_hz: 10.908450444539389 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/sf_20/learning_rates:245`
  - blankfree_expected_phone_rate_hz: 9.071449915568033 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/sf_20/learning_rates:240`
  - blankfree_temperature: 2.0 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/sf_20/learning_rates:250`
  - sf_lam: 0.045393798500299454 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/sf_20/learning_rates:257`

### soft_20
- Current sub-epoch: 4 (completed)
- Step-1 loss (train_loss_agg): NOT READ
- Epoch 1 loss (dev_loss_agg): 1.4302053451538086 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20/learning_rates:15`
- Wall seconds per completed sub-epoch (average): 848.5 s (ratio to 601 s: 1.412)
  - Epoch times: 856, 847, 847, 844 seconds - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20/learning_rates:8,74,140,206`
- Epoch 4 dev monitor columns:
  - soft_reward_mean: -0.45077858368555707 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20/learning_rates:235`
  - blankfree_phone_rate_retained_hz: 10.411775271097818 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20/learning_rates:221`
  - blankfree_expected_phone_rate_hz: 8.802952766418457 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20/learning_rates:216`
  - blankfree_temperature: 2.0 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20/learning_rates:226`
  - soft_lam: 0.32663899660110474 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20/learning_rates:233`

### soft_20_s1
- Current sub-epoch: 4 (completed)
- Step-1 loss (train_loss_agg): NOT READ
- Epoch 1 loss (dev_loss_agg): 1.4350388447443645 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20_s1/learning_rates:15`
- Wall seconds per completed sub-epoch (average): 851.75 s (ratio to 601 s: 1.418)
  - Epoch times: 862, 851, 845, 849 seconds - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20_s1/learning_rates:8,74,140,206`
- Epoch 4 dev monitor columns:
  - soft_reward_mean: -0.4083094596862793 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20_s1/learning_rates:235`
  - blankfree_phone_rate_retained_hz: 10.613077799479166 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20_s1/learning_rates:221`
  - blankfree_expected_phone_rate_hz: 8.977310180664062 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20_s1/learning_rates:216`
  - blankfree_temperature: 2.0 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20_s1/learning_rates:226`
  - soft_lam: 0.32663899660110474 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/soft_20_s1/learning_rates:233`

### softshuf_20
- Current sub-epoch: 4 (completed)
- Step-1 loss (train_loss_agg): NOT READ
- Epoch 1 loss (dev_loss_agg): 1.520421067873637 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/softshuf_20/learning_rates:15`
- Wall seconds per completed sub-epoch (average): 848.5 s (ratio to 601 s: 1.412)
  - Epoch times: 857, 846, 847, 844 seconds - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/softshuf_20/learning_rates:8,74,140,206`
- Epoch 4 dev monitor columns:
  - soft_reward_mean: -1.799160083134969 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/softshuf_20/learning_rates:235`
  - blankfree_phone_rate_retained_hz: 10.185827255249023 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/softshuf_20/learning_rates:221`
  - blankfree_expected_phone_rate_hz: 8.635418891906738 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/softshuf_20/learning_rates:216`
  - blankfree_temperature: 2.0 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/softshuf_20/learning_rates:226`
  - soft_lam: 0.32663899660110474 - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/work/softshuf_20/learning_rates:233`

## Evaluation Jobs Status
- ep1 forward jobs (ReturnnForwardJobV2): COMPLETE (posteriors generated)
- ep1 WER/PER summary files: NOT FOUND

## Peak Memory
- Max RSS: 52.03179931640625 GB - `/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O/usage.run.1:4`
- GPU memory: NOT LOGGED
