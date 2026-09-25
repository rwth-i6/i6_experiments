# L2-2 bridge read for stage-2 key arms (A18 (c)), G4a.L2.4 — extraction

Source report:
`output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_em/a16b_keyinit/keys/tau4/l22/read/report.txt`
(real: `work/speech_llm/sae/emc/l22_bridge_read_jobs/BridgeReadJob.HWuOTb0Tefo1/output/report.txt`,
resolved dir `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/l22_bridge_read_jobs/BridgeReadJob.HWuOTb0Tefo1`)
and `bridge_read.json` in the same dir.

## 1. Verdicts / convention (report.txt, RULE + IMPLEMENTATION blocks and final lines)
- Convention verbatim printed at top of report.txt (RULE, A6, A9, coordinator rulings 2026-09-24) — see file head.
- `G4a.L2.4 (dec_joint vs cold_ctl): LOWER -- OBJECTIVE ONLY` (report.txt, last line)
- Reported, deciding nothing (same class rule applied to the other arms):
  - `dec_distil    LOWER -- OBJECTIVE ONLY`
  - `dec_frz       LOWER -- OBJECTIVE ONLY`
  - `dec_joint_s2  LOWER -- OBJECTIVE ONLY`
  (report.txt, "reported, deciding nothing" block)
- CODE BROKEN/OBJECTIVE ONLY rule as printed: "CODE BROKEN = LOWER AND the arm's dev-other greedy PER < 0.50 at any kept epoch (1,2,4,8); OBJECTIVE ONLY = LOWER otherwise" (report.txt, IMPLEMENTATION item 6).

## 2. ep8 delta dec_joint − cold_ctl, CI, B, margin, n
- `dec_joint - cold_ctl: -0.160859 [-0.170965, -0.148819]  |delta| > margin  agg point -2.382924  total point -0.401151` (report.txt, PAIRED block)
- identity band `B = |mean(L_dec_joint - L_dec_joint_s2)| = 0.003807; margin max(B, 0.01) = 0.010000` (report.txt, line above PAIRED rows)
- `PAIRED over 285 of 285 utterances kept in every cell, 164 speakers, 2000 resamples, seed 0` (report.txt)
- n utterances = 285 (also stated "285 CV-holdout utterances in 3 batches, identical across cells"); n speakers = 164.

## 3. Per-term contrasts vs cold_ctl (report.txt CELLS + PAIRED blocks; l_tau/lexlat_k2/rate paired, agg point-only per rule item 7)
- dec_joint: l_tau 1.803999, lexlat_k2 0.318848, rate 0.032241, agg 0.803917 (cell value; cold_ctl agg 3.186841); paired delta (l_tau+lexlat_k2+3·rate) = -0.160859 [-0.170965,-0.148819]; agg point contrast -2.382924
- dec_distil: l_tau 1.802410, lexlat_k2 0.316596, rate 0.032238, agg 0.785149; paired delta -0.165360 [-0.174913,-0.153480]; agg point -2.401693
- dec_frz: l_tau 1.815203, lexlat_k2 0.333049, rate 0.041883, agg 0.701243; paired delta -0.106334 [-0.119648,-0.087861]; agg point -2.485598
- dec_joint_s2: l_tau 1.800459, lexlat_k2 0.327088, rate 0.029835, agg 0.849517; paired delta -0.164667 [-0.174264,-0.153435]; agg point -2.337324
- cold_ctl (baseline values): l_tau 2.087983, lexlat_k2 0.235907, rate 0.019513, agg 3.186841
(all from report.txt CELLS and PAIRED blocks)

## 4. Dev-other greedy PER at ep1/2/4/8 (report.txt BESIDE block; class column shown, no separate "chance band" numeric printed — only class labels LIFT/PARTIAL/NO LIFT per A4 referenced in RULE)
| arm | ep1 | ep2 | ep4 | ep8 | class@ep8 |
|---|---|---|---|---|---|
| dec_joint | 0.8622 | 0.8523 | 0.8401 | 0.8427 | NO LIFT |
| dec_distil | 0.8570 | 0.8535 | 0.8444 | 0.8429 | NO LIFT |
| dec_frz | 0.8612 | 0.8565 | 0.8495 | 0.8464 | NO LIFT |
| dec_joint_s2 | 0.8625 | 0.8497 | 0.8433 | 0.8431 | NO LIFT |
| cold_ctl | 0.8881 | 0.8864 | 0.8554 | 0.8487 | NO LIFT |
No separate numeric "chance band" or "D10e's bands" values are printed in report.txt; only the RULE text references them and A4's classes (LIFT < 0.50, PARTIAL < 0.8164) are given as thresholds (report.txt IMPLEMENTATION item 8).

## 5. phi generative PER direct/Hungarian/NMI at kept epochs (report.txt, "phi's generative PER on the D4 dev-other set" block)
- dec_joint: ep1 0.8582/0.8532/0.0774; ep2 0.8575/0.8434/0.0785; ep4 0.8554/0.8436/0.0795; ep8 0.8584/0.8551/0.0773
- dec_distil: ep1 0.8582/0.8613/0.0774; ep2 0.8571/0.8521/0.0782; ep4 0.8554/0.8462/0.0789; ep8 0.8587/0.8453/0.0775
- dec_frz: ep1 0.8582/0.8613/0.0774; ep2 0.8582/0.8613/0.0774; ep4 0.8582/0.8613/0.0774; ep8 0.8582/0.8613/0.0774
- dec_joint_s2: ep1 0.8584/0.8535/0.0781; ep2 0.8575/0.8515/0.0773; ep4 0.8564/0.8511/0.0780; ep8 0.8586/0.8508/0.0779
- cold_ctl: ep1 0.9375/0.9022/0.3993; ep2 0.8262/0.8259/0.1425; ep4 0.8382/0.8422/0.0922; ep8 0.8498/0.8519/0.0811

## 6. Lift-test verdict / gold-key lift reference
No A14 (i)-form lift-test verdict line and no same-line gold-key lift reference is printed anywhere in report.txt or bridge_read.json. Only per-epoch class labels (LIFT/PARTIAL/NO LIFT, all NO LIFT here) appear in the BESIDE PER table.
MISSING (searched: report.txt full text, bridge_read.json full text — grep for "lift", "A14", "gold" found no separate verdict/reference line).

## 7. Arm list, phi paths, pack job, SLURM id
Arms in this bridge: dec_joint, dec_distil, dec_frz, dec_joint_s2 (bridged pack), plus baseline cold_ctl (reused from L2-0).
- All four bridge arms are initialized from ONE selected phi: the S-best stage-2 key arm "rank1" at sub-epoch 48, selected via `SelectedCheckpointJob.PAV7erhRs8wq` output
  `output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_em/a16b_keyinit/keys/tau4/l22/selected_checkpoint/selected.json`
  (real: `/e/scratch/.../work/speech_llm/sae/emc/a18_selected_checkpoint_jobs/SelectedCheckpointJob.PAV7erhRs8wq/output/selected.json`):
  - selected = "rank1", verdict "KEY BASIN"
  - checkpoint = `/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.G0Vzzokj5PQC/output/rank1/models/epoch.048.pt`
  - selection_json = `work/speech_llm/sae/emc/keyarms_read_jobs/KeyArmsReadJob.STcxhF0w4kpq/output/keyinit_arms.json`, key ["comparison","best_arm"] = "rank1"
- cold_ctl baseline = L2-0's `PackedBlankfreeTrainJob.UdhhxiGIMBob` arm `cold_ctl`, checkpoint `.../output/cold_ctl/models/epoch.008.pt` (report.txt DISCLOSED BASELINE DIFFERENCE section; also seen via job.save strings of BridgeReadJob.HWuOTb0Tefo1).
- Bridge pack job (dec_joint/dec_distil/dec_frz/dec_joint_s2 ep8 checkpoints): `PackedBlankfreeTrainJob.SCtv4DjzFQ50`
  (work dir `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.SCtv4DjzFQ50`;
  alias `sae/4a/lexlat_v2/em/a16b_keyinit/keys/tau4/l22/pack/training`)
  - checkpoints referenced in BridgeReadJob.HWuOTb0Tefo1/job.save (strings dump): dec_distil, dec_frz, dec_joint, dec_joint_s2 all at `.../PackedBlankfreeTrainJob.SCtv4DjzFQ50/output/<arm>/models/epoch.008.pt`
  - SLURM id found via `engine/` dir of this job: `speech_llm.sae.emc.blankfree_pack_jobs.PackedBlankfreeTrainJob.SCtv4DjzFQ50.run.2007225.1` → SLURM job id **2007225**
  - BridgeReadJob.HWuOTb0Tefo1 itself (the L2-2 reader) has no engine/ dir with a SLURM id found in a quick search of its files (info/submit_log/log.run.1 contained no "slurm"/"jobid" hits) — MISSING (searched: BridgeReadJob.HWuOTb0Tefo1/info, submit_log.run, log.run.1).

## 8. Warnings / NaN / missing-epoch notes
report.txt prints an explicit `identity: PASS` line after all 20 IDENTITY rows (l_tau, lexlat_k2 PASS; agg, rate "printed, not checked"), each with `diff +0.000000`. No VOID, NaN, or missing-epoch marker appears anywhere in report.txt or bridge_read.json (grepped both files). No other warnings printed.
