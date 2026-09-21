# Extract: soft_pack gaps and paired PER, 2026-09-21

All prior_gap and paired jobs checked (sf_20, soft_20, soft_20_s1, softshuf_20 at ep1/ep4/ep10,
plus paired against ctrl_20 / ctrl_20_s1) trace through ExtractSubmoduleCheckpointJob to
PackedBlankfreeTrainJob.MXKoywbfon8O (the live pack, started 2026-09-21 17:06 UTC) -> LIVE, not stale.
Job dirs are under /e/scratch/spell/wu24/2026-07-13_unsupervised/work/... (prior_gap, blankfree_eval_jobs,
i6_core/returnn/forward) and /e/project1/spell/wu24/2026-07-13_unsupervised/work/... (eval_jobs, emc_train_jobs).

## prior_gap (dev-other, ep10) - `like_for_like` table (gold minus private, SIL-free), NATS per token

Job dirs (work/speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.<hash>), finished.tar.gz mtime, checkpoint chain -> MXKoywbfon8O:
- sf_20: PriorGapAnalysisJob.SRAdp38vlvTg, finished 2026-09-21 19:24:36; via BlankfreeGreedyPerJob.eQVw2XrcOWXg -> ReturnnForwardJobV2.ZV8ROhl6N0jE -> ExtractSubmoduleCheckpointJob.HfqaLearwVgS -> MXKoywbfon8O. LIVE.
- soft_20: PriorGapAnalysisJob.APsPQcXqWfqm, finished 2026-09-21 19:51:39; via BlankfreeGreedyPerJob.ZzEhEJ2ghmA7 -> RFJV2.5wpWXkTFjlrF -> ExtractSubmoduleCheckpointJob.JB86mmt5XUAA -> MXKoywbfon8O. LIVE.
- soft_20_s1: PriorGapAnalysisJob.Nb61mFcWMMjA, finished 2026-09-21 19:51:39; via BlankfreeGreedyPerJob.59MMKl1PFa6G -> RFJV2.WDIpb3Nzussg -> ExtractSubmoduleCheckpointJob.wbbob6cRlX9W -> MXKoywbfon8O. LIVE.
- softshuf_20: PriorGapAnalysisJob.kAjDAogYtvAi, finished 2026-09-21 19:51:39; via BlankfreeGreedyPerJob.aP10M9b5T6bF -> RFJV2.C4LXm5pTyXwi -> ExtractSubmoduleCheckpointJob.x60M5OyO0iKC -> MXKoywbfon8O. LIVE.

Row labels as written; gap column header = "gap /token (paired)"; gold/private columns = "gold /token" / "private /token" (NATS, per-token, sentence-start context, NO end-of-sentence term).

| row (label as written) | sf_20 gold/private/gap | soft_20 gold/private/gap | soft_20_s1 gold/private/gap | softshuf_20 gold/private/gap |
|---|---|---|---|---|
| unigram (Witten-Bell, live) | -3.5035/-3.6775/0.1699 | -3.5035/-3.6347/0.1245 | -3.5035/-3.5630/0.0422 | -3.5035/-3.5355/0.0211 |
| bigram (Witten-Bell, live) | -3.2502/-3.7117/0.4305 | -3.2502/-3.7722/0.5126 | -3.2502/-3.6320/0.3489 | -3.2502/-3.6474/0.3686 |
| trigram (Witten-Bell, live) | -3.1992/-4.4093/1.1466 | -3.1992/-4.5312/1.2954 | -3.1992/-4.3098/1.0553 | -3.1992/-4.3192/1.0657 |
| 4-gram (KenLM, mKN) | -2.7862/-4.3151/1.4603 | -2.7862/-4.4246/1.5917 | -2.7862/-4.2669/1.4186 | -2.7862/-4.3191/1.4754 |
| 6-gram (KenLM, mKN) | -2.5742/-4.1496/1.4914 | -2.5742/-4.2365/1.6002 | -2.5742/-4.1813/1.5369 | -2.5742/-4.1845/1.5406 |
| lexicon+word3gram STRICT (subset) | -2.1450/-4.5560/2.4026 (n=1005) | -2.1450/-4.7710/2.6259 (n=928) | -2.1450/-4.5372/2.4118 (n=436) | -2.1450/-4.6670/2.5117 (n=664) |
| lexicon+word3gram+escape word | -2.1945/-4.3992/2.2365 | -2.1945/-4.4433/2.3130 | -2.1945/-4.4119/2.2941 | -2.1945/-4.3916/2.2688 |

All four arms' decision-table outcome (as written) = "no_row_of_the_table_fires". Only ep10/dev-other prior_gap files exist for these arms (no ep1/ep4 prior_gap).

## paired PER (delta_per = B minus A; negative = B better), from summary.txt

Every PairedPerDeltaJob's BlankfreeGreedyPerJob traces to an ExtractSubmoduleCheckpointJob whose input is MXKoywbfon8O, for ep1, ep4, AND ep10 alike (checked jobs: A8ndOEpJ7b4G/m5rBiIkGl9mg/jBzDIEhM2jPV/b328Eh3bSIfB/j70QWN8e76xv/3v63CVXDRUdp/rCyzp5DFCeKo/iBpqNMidqr4l/g883E5BN1Obs/4eWaDmuAIRFV/PDfOxn9NiGBA and their forward-job ancestors). All rows below = LIVE (MXKoywbfon8O).

sf_20_vs_ctrl_20:
- ep1 dev-clean: PairedPerDeltaJob.A8ndOEpJ7b4G, delta_per=+0.005071 [+0.002683,+0.007427], PER A(ctrl)=0.846564 -> B(sf_20)=0.851635
- ep1 dev-other: PairedPerDeltaJob.iNstrKa79RG4, delta_per=+0.007508 [+0.004808,+0.010336], PER A=0.855439 -> B=0.862947
- ep4 dev-clean: PairedPerDeltaJob.xP0MNQTHlVNq, delta_per=+0.036056 [+0.031947,+0.040772], PER A=0.857331 -> B=0.893387
- ep4 dev-other: PairedPerDeltaJob.K55L49OmEyHM, delta_per=+0.035064 [+0.031481,+0.039134], PER A=0.875059 -> B=0.910123
- ep10 dev-clean: PairedPerDeltaJob.m5rBiIkGl9mg, delta_per=+0.025795 [+0.023557,+0.028095], PER A=0.856551 -> B=0.882346
- ep10 dev-other: PairedPerDeltaJob.jBzDIEhM2jPV, delta_per=+0.030597 [+0.028117,+0.033037], PER A=0.868876 -> B=0.899473

soft_20_vs_ctrl_20:
- ep1 dev-clean: PairedPerDeltaJob.b328Eh3bSIfB, delta_per=+0.021731 [+0.018616,+0.025134], PER A=0.846564 -> B=0.868294
- ep1 dev-other: PairedPerDeltaJob.3hnlDCFUWvU6, delta_per=+0.020601 [+0.017678,+0.023761], PER A=0.855439 -> B=0.876040
- ep4 dev-clean: PairedPerDeltaJob.NhJ1SAmg59ZG, delta_per=+0.001937 [-0.001025,+0.005122], PER A=0.857331 -> B=0.859268
- ep4 dev-other: PairedPerDeltaJob.e9kvqfLZSbId, delta_per=-0.000446 [-0.003119,+0.002491], PER A=0.875059 -> B=0.874613
- ep10 dev-clean: PairedPerDeltaJob.aztMe79J02aj, delta_per=+0.009605 [+0.007709,+0.011537], PER A=0.856551 -> B=0.866156
- ep10 dev-other: PairedPerDeltaJob.j70QWN8e76xv, delta_per=+0.010063 [+0.007488,+0.012549], PER A=0.868876 -> B=0.878940

softshuf_20_vs_ctrl_20:
- ep1 dev-clean: PairedPerDeltaJob.t7E7kvtLsc8C, delta_per=+0.081800 [+0.077468,+0.086105], PER A=0.846564 -> B=0.928363
- ep1 dev-other: PairedPerDeltaJob.4eWaDmuAIRFV, delta_per=+0.071048 [+0.067287,+0.075140], PER A=0.855439 -> B=0.926487
- ep4 dev-clean: PairedPerDeltaJob.2pCUnlMKzHws, delta_per=-0.003217 [-0.006228,-0.000084], PER A=0.857331 -> B=0.854114
- ep4 dev-other: PairedPerDeltaJob.DsOsckolEF1y, delta_per=+0.000677 [-0.002299,+0.004015], PER A=0.875059 -> B=0.875735
- ep10 dev-clean: PairedPerDeltaJob.FbDwuxMQe4cD, delta_per=+0.007679 [+0.005497,+0.010009], PER A=0.856551 -> B=0.864230
- ep10 dev-other: PairedPerDeltaJob.PDfOxn9NiGBA, delta_per=+0.012083 [+0.009458,+0.014637], PER A=0.868876 -> B=0.880959

soft_20_s1_vs_ctrl_20_s1:
- ep1 dev-clean: PairedPerDeltaJob.3v63CVXDRUdp, delta_per=+0.021927 [+0.019673,+0.024245], PER A=0.845577 -> B=0.867504
- ep1 dev-other: PairedPerDeltaJob.IdlPiSSC8eNR, delta_per=+0.020917 [+0.017819,+0.024298], PER A=0.851671 -> B=0.872588
- ep4 dev-clean: PairedPerDeltaJob.70qNhdrnSxmH, delta_per=+0.009652 [+0.007356,+0.011945], PER A=0.855565 -> B=0.865217
- ep4 dev-other: PairedPerDeltaJob.Ry8HavLdFbZd, delta_per=+0.009855 [+0.006910,+0.012718], PER A=0.872943 -> B=0.882798
- ep10 dev-clean: PairedPerDeltaJob.yE7yNDjwa7hi, delta_per=+0.006321 [+0.004423,+0.008274], PER A=0.864752 -> B=0.871073
- ep10 dev-other: PairedPerDeltaJob.rCyzp5DFCeKo, delta_per=+0.009928 [+0.006695,+0.013186], PER A=0.880203 -> B=0.890131

sf_20_vs_soft_20 (ep1/4/10, dev-clean/dev-other) and soft_20_vs_softshuf_20 (ep1/4/10, dev-clean/dev-other) also present; values in raw tool output, all traced LIVE to MXKoywbfon8O (same job set as above). Not re-copied here for length; see setup dir paths listed at top.

## MXKoywbfon8O per-arm learning_rates (work/<arm>/learning_rates)

Latest completed sub-epoch (last non-empty error{} block): sf_20 = epoch 14; soft_20 = epoch 12; soft_20_s1 = epoch 12; softshuf_20 = epoch 12. LR schedule dict itself lists placeholders up to epoch 20 (learningRate only, error={}), not yet reached.

- sf_20 epoch 14: dev_loss_agg=1.6823749542236328; dev_loss_sf_reward_mean=-0.514888217051824; dev_loss_sf_reward_mean_sample=-0.3466938336690267; train_loss_sf_reward_mean=-0.4859245733210915; train_loss_sf_reward_mean_sample=-0.36962456086225676
- soft_20 epoch 12: dev_loss_agg=1.4085910320281982; dev_loss_soft_reward_mean=-0.5900722742080688; train_loss_soft_reward_mean=-0.5704649875038549
- soft_20_s1 epoch 12: dev_loss_agg=1.3520103693008423; dev_loss_soft_reward_mean=-0.4980478286743164; train_loss_soft_reward_mean=-0.5158935759151191
- softshuf_20 epoch 12: dev_loss_agg=1.6003495454788208; dev_loss_soft_reward_mean=-2.02537206808726; train_loss_soft_reward_mean=-2.0711935152087295

No prior_gap.md exists for ctrl_20 / ctrl_20_s1 themselves (MISSING - searched output/exp2025_11_06_speech_llms/librispeech/sae_4a_soft_pack/prior_gap/, only sf_20, soft_20, soft_20_s1, softshuf_20 present; ctrl arms are "frozen from an earlier pack" per task brief and only appear as the paired-PER comparison side A).
