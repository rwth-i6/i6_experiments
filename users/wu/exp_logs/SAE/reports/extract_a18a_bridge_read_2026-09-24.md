# extract: A18 (a) bridge read, 2026-09-24

DONE.

## 1. Pack status
Pack PackedBlankfreeTrainJob.63nbj6Jgiegj FINISHED (finished.tar.gz present) at
/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.63nbj6Jgiegj
All four arms have models/epoch.001.pt, epoch.002.pt, epoch.004.pt, epoch.008.pt (+epoch.008.opt.pt), i.e. all completed sub-epochs 1/2/4/8 (kept grid). No NaN, no traceback, no exception found in any arm's log.run.1 (grep -iE "nan|error|traceback|exception"); the only "error" hits are RETURNN's routine "Learning-rate-control: error key ..." score-dict prints and "Cache manager: Error occurred, using local file" (benign infra fallback), not job errors.
ep8 checkpoint paths:
- dec_joint: .../output/dec_joint/models/epoch.008.pt
- dec_distil: .../output/dec_distil/models/epoch.008.pt
- dec_frz: .../output/dec_frz/models/epoch.008.pt
- dec_joint_s2: .../output/dec_joint_s2/models/epoch.008.pt
(all under the pack path above)

## 2. Reader verdict
Path: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/l22_bridge_read_jobs/BridgeReadJob.v6SONiAZL7rH/output/report.txt (and bridge_read.json)
Printed verdict string: "G4a.L2.4 (dec_joint vs cold_ctl): LOWER -- OBJECTIVE ONLY"
bridge_read.json verdict field: {'class': 'LOWER', 'flag': 'OBJECTIVE ONLY'}
dec_joint - cold_ctl: delta -0.115829, interval [-0.125691, -0.103458], |delta| > margin
identity band B = |mean(L_dec_joint - L_dec_joint_s2)| = 0.009338; margin = max(B, 0.01) = 0.010000
Per-term contrasts (point, per cell, not intervalled beyond the total): 
- dec_joint: l_tau 1.848370, agg 1.078690, rate 0.028315, lexlat_k2 0.331078, total 2.372262 (kept 285)
- cold_ctl: l_tau 2.087983, agg 3.186841, rate 0.019513, lexlat_k2 0.235907, total 2.701114 (kept 285)
- dec_joint agg point contrast vs cold_ctl: -2.108152; total point contrast: -0.328852
Other arms vs cold_ctl (reported, deciding nothing): dec_distil delta -0.107139 [-0.116958, -0.095885] LOWER OBJECTIVE ONLY (agg point -2.111655, total point -0.320438); dec_frz delta -0.044921 [-0.055303, -0.031795] LOWER OBJECTIVE ONLY (agg point -2.270521, total point -0.276967); dec_joint_s2 delta -0.106491 [-0.116282, -0.094954] LOWER OBJECTIVE ONLY (agg point -2.099546, total point -0.319628).
Identity check (all PASS, tolerance 0.005, diffs all +0.000000/-0.000000): l_tau and lexlat_k2 checked for every arm+cold_ctl; agg and rate printed but not checked.

## 3. Dev-other greedy PER (ep1/ep2/ep4/ep8, class@ep8 all NO LIFT)
- dec_joint:    0.8615 / 0.8536 / 0.8464 / 0.8420
- dec_distil:   0.8537 / 0.8526 / 0.8471 / 0.8461
- dec_frz:      0.8623 / 0.8585 / 0.8432 / 0.8463
- dec_joint_s2: 0.8629 / 0.8549 / 0.8426 / 0.8460
- cold_ctl:     0.8881 / 0.8864 / 0.8554 / 0.8487

## 4. Generative PER (direct / Hungarian / NMI(symbol,phone)) on D4 dev-other, kept epochs
- dec_joint:    ep1 0.8568/0.8564/0.0773  ep2 0.8561/0.8556/0.0786  ep4 0.8551/0.8522/0.0768  ep8 0.8558/0.8526/0.0779
- dec_distil:   ep1 0.8566/0.8587/0.0761  ep2 0.8554/0.8519/0.0796  ep4 0.8543/0.8537/0.0787  ep8 0.8552/0.8584/0.0769
- dec_frz:      ep1 0.8566/0.8587/0.0761  ep2 0.8566/0.8587/0.0761  ep4 0.8566/0.8587/0.0761  ep8 0.8566/0.8587/0.0761
- dec_joint_s2: ep1 0.8574/0.8574/0.0766  ep2 0.8583/0.8528/0.0767  ep4 0.8572/0.8525/0.0772  ep8 0.8564/0.8553/0.0761
- cold_ctl:     ep1 0.9375/0.9022/0.3993  ep2 0.8262/0.8259/0.1425  ep4 0.8382/0.8422/0.0922  ep8 0.8498/0.8519/0.0811

dec_frz ep1 (= em_s13's own generative PER, phi frozen, unchanged across ep1/2/4/8 per report): direct 0.8566, Hungarian 0.8587, NMI(symbol,phone) 0.0761. (All four dec_frz epochs print identically: 0.8566/0.8587/0.0761.)

## 5. Recompute vs copy, set sizes
The reader recomputed the totals: report.txt states "IMPLEMENTATION" step 1 runs one crosseval_jobs forward per arm/baseline at ep8 (D16 eval path, training-step loss code, no update) on the shared CV holdout, then an "IDENTITY" check compares each cell's recomputed dev_loss_l_tau and dev_loss_lexlat_k2 against the run's own logged values (tolerance 0.005); all pass with diff +0.000000/-0.000000. It did not simply copy training-log totals -- it recomputed via forward passes and cross-checked against the logged values.
Set sizes: 285 CV-holdout utterances in 3 batches, identical across cells; PAIRED over 285 of 285 utterances kept in every cell, across 164 speakers, 2000 bootstrap resamples, seed 0.

DISCLOSED (from report.txt): cold_ctl is L2-0's reused cold_ctl arm (PackedBlankfreeTrainJob.UdhhxiGIMBob), differing from dec_joint only in phi's init (random init, no durinit prior applied to phi rows) per A9/A6.
