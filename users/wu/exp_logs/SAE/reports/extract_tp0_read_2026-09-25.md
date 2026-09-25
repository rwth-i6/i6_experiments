# TP0 read extraction (2026-09-25)

Reader job dir: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/rename_tp0_jobs/Tp0ReadJob.t7gzZhB2fWtn
(workspace: work/speech_llm/sae/emc/rename_tp0_jobs/Tp0ReadJob.t7gzZhB2fWtn, symlink of same)
Output files: .../output/report.txt, .../output/tp0.json

Pack job (checkpoints): /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.21tww6QQK0tH
Checkpoint files (per arm, output/<arm>/models/epoch.0NN.pt; ep0 = each arm's init phi, not from this pack):
- full_s1: output/full_s1/models/epoch.{001..012}.pt (ep4=004, ep8=008, ep12=012)
- full_s2: output/full_s2/models/epoch.{001..012}.pt
- 5pair_s1: output/5pair_s1/models/epoch.{001..012}.pt
- control: output/control/models/epoch.{001..012}.pt
Init (ep0) checkpoints: full_s1/full_s2/5pair_s1 = AN-2's PhiDerangeJob outputs (ids QRIjZnmG0boo / DLoD5WXxaOx6 / PhNxF9aMQxAJ, out_checkpoint); control = gold-key phi PhiFromKeyInitJob.f0jaGuiJVe6A out_checkpoint.

## Verbatim reader VERDICT
"VERDICT: EM LOCKED (DELTA = identity(12) - identity(0); DELTA >= 0.2 on [] (EM RENAMES needs >= 2 of 3); DELTA <= 0.05 on ['5pair_s1', 'full_s1', 'full_s2'] (EM LOCKED needs >= 2 of 3))"
"TP0 OUTCOME: EM LOCKED" (printed at top and bottom of report.txt)
control drift (identity(12) - identity(0), not counted): -0.2202

## KEY IDENTITY (frame, A20 weights) and BEST 1:1 RENAMING IDENTITY, per arm, id0/id4/id8/id12, DELTA=id12-id0
arm        id0     id4     id8    id12    DELTA     1:1_0   1:1_4   1:1_8  1:1_12
full_s1    0.0777  0.1059  0.0996  0.1016  +0.0239   0.9998  0.5973  0.5650  0.5541
full_s2    0.0778  0.0928  0.0917  0.0903  +0.0125   0.9999  0.5546  0.5364  0.5177
5pair_s1   0.5931  0.6266  0.5981  0.5880  -0.0051   1.0000  0.7014  0.6786  0.6501
control*   1.0000  0.8169  0.7929  0.7798  -0.2202   1.0000  0.8169  0.7929  0.7798
(* control reported beside, not counted in verdict)

identity(12) - identity(0) delta per deranged arm (verbatim column "DELTA" above): full_s1 +0.0239; full_s2 +0.0125; 5pair_s1 -0.0051.

## S (held-out tau=1 NLL/frame, nats) on the common 260-tag set (excluded: none); own-set S(n) in parens
full_s1   ep0: 5.11291 (5.11291,260)  ep4: 3.48714 (3.48714,260)  ep8: 3.42172 (3.42172,260)  ep12: 3.38725 (3.38725,260)
full_s2   ep0: 5.08628 (5.08628,260)  ep4: 3.45688 (3.45688,260)  ep8: 3.39278 (3.39278,260)  ep12: 3.37038 (3.37038,260)
5pair_s1  ep0: 4.93332 (4.93332,260)  ep4: 3.36462 (3.36462,260)  ep8: 3.30832 (3.30832,260)  ep12: 3.29222 (3.29222,260)
control   ep0: 4.57305 (4.57305,260)  ep4: 3.33975 (3.33975,260)  ep8: 3.27941 (3.27941,260)  ep12: 3.25229 (3.25229,260)

## Generative PER on D4 dev-other (direct / Hungarian / NMI(symbol,phone))
full_s1   ep0: 0.9533/0.9697/0.0526  ep4: 0.8582/0.8712/0.0746  ep8: 0.8560/0.8685/0.0767  ep12: 0.8541/0.8659/0.0799
full_s2   ep0: 0.9489/0.9751/0.0608  ep4: 0.8575/0.8632/0.0716  ep8: 0.8557/0.8664/0.0767  ep12: 0.8549/0.8652/0.0762
5pair_s1  ep0: 0.6597/0.5765/0.4519  ep4: 0.4607/0.4598/0.6087  ep8: 0.4917/0.4941/0.5741  ep12: 0.5019/0.5034/0.5625
control   ep0: 0.3271/0.3848/0.7398  ep4: 0.2811/0.3302/0.7531  ep8: 0.3129/0.3600/0.7294  ep12: 0.3358/0.3793/0.7061

## A15-F measures (OWN of 40, CLM/UNCL claimed/unclaimed gold types, DUP duplicated gold types(symbols), R4 direct/emis)
full_s1   ep0  OWN 3 CLM 38 UNCL 2 DUP 1(3) R4 0.0710/0.6123 [AH:3]
full_s1   ep4  OWN 2 CLM 35 UNCL 5 DUP 5(10) R4 0.1040/0.4710 [AY:2,DH:2,F:2,N:2,T:2]
full_s1   ep8  OWN 2 CLM 30 UNCL 10 DUP 9(19) R4 0.0999/0.4441 [AY:2,DH:2,ER:2,F:2,K:2,N:2,P:2,R:2,T:3]
full_s1   ep12 OWN 2 CLM 32 UNCL 8 DUP 7(15) R4 0.1040/0.4316 [AY:2,DH:2,F:2,N:2,P:2,R:2,T:3]
full_s2   ep0  OWN 3 CLM 38 UNCL 2 DUP 1(3) R4 0.0706/0.6123 [AH:3]
full_s2   ep4  OWN 1 CLM 31 UNCL 9 DUP 8(17) R4 0.0841/0.4180 [CH:2,D:3,K:2,M:2,N:2,R:2,S:2,T:2]
full_s2   ep8  OWN 1 CLM 31 UNCL 9 DUP 9(18) R4 0.0830/0.4196 [CH:2,D:2,HH:2,IY:2,K:2,N:2,R:2,S:2,T:2]
full_s2   ep12 OWN 1 CLM 32 UNCL 8 DUP 7(15) R4 0.0836/0.4054 [D:2,HH:3,IY:2,N:2,R:2,S:2,T:2]
5pair_s1  ep0  OWN 30 CLM 38 UNCL 2 DUP 1(3) R4 0.3872/0.6123 [AH:3]
5pair_s1  ep4  OWN 32 CLM 34 UNCL 6 DUP 6(12) R4 0.4444/0.4853 [K:2,L:2,M:2,N:2,OW:2,P:2]
5pair_s1  ep8  OWN 31 CLM 34 UNCL 6 DUP 6(12) R4 0.4319/0.4855 [EH:2,L:2,M:2,N:2,OW:2,T:2]
5pair_s1  ep12 OWN 31 CLM 34 UNCL 6 DUP 6(12) R4 0.4294/0.4767 [EH:2,L:2,M:2,N:2,OW:2,T:2]
control   ep0  OWN 40 CLM 38 UNCL 2 DUP 1(3) R4 0.6123/0.6123 [AH:3]
control   ep4  OWN 38 CLM 38 UNCL 2 DUP 2(4) R4 0.5729/0.5730 [AA:2,AW:2]
control   ep8  OWN 38 CLM 37 UNCL 3 DUP 3(6) R4 0.5591/0.5591 [AE:2,AW:2,K:2]
control   ep12 OWN 38 CLM 37 UNCL 3 DUP 3(6) R4 0.5569/0.5569 [AE:2,AW:2,K:2]

## Conventions printed by the reader (verbatim, abridged)
- KEY(u) = argmax_s m(u|s) pi(s) (ties to lowest id); IDENTITY = agreement(KEY, gold key, w)["frame_identity"], w = unit_frame_counts over train.segments (A20's weights, total asserted 15,275,716); m = PhiContentJob m_phi (D4 dev-other etas, duration-weighted); pi = expected symbol frame share of AN-4 forward (unit_symbol.npz, A13's 260 disjoint CV-holdout utterances possible in it).
- BEST 1:1 RENAMING = oracle_rename_1to1(KEY, gold key, w): Hungarian assignment (maximise) on 40x40 w-weighted KEY-symbol x gold-symbol table (A20 variant (b)); partition-only measure.
- DELTA(arm) = IDENTITY(12) - IDENTITY(0).
- VERDICT rule (3 deranged arms): EM RENAMES iff DELTA>=0.2 on >=2/3; EM LOCKED iff DELTA<=0.05 (signed) on >=2/3; else PARTIAL.
- S: A14(ii)'s reader, per-utterance nll_tau1_per_frame of CV-holdout marginal (genmarg.json: ReturnnForwardJobV2, null recognizer, bed's trigram lattice, float64, tau=1); mean over the common 260-tag set possible in all 16 marginals (size/exclusions printed); own-set S over its own possible 260 tags printed beside.
- PER: GenDecodeReportJob report.json of dev-other (D4, 500 utterances) posterior decode: direct PER (per_identity), Hungarian PER (per_hungarian), NMI(symbol,phone).
- A15-F: rename_an5_jobs.a15f_measures of checkpoint's content.json.
- Recipe: A17(ii)'s tau=1 recipe (config_sae_4a_lexlat_v2_a17_v1.drift_config: A14(ii) restart config, seed 1, 12 sub-epochs, tau=1 every sub-epoch, constant phi lr 3e-3, same data, null recognizer + trigram prior).
- Epoch 0 = each arm's init phi (gold-key phi, deranged or not); ARMS: one four-GPU PackedBlankfreeTrainJob; gold-key phi = PhiFromKeyInitJob.f0jaGuiJVe6A on GoldUnitKeyJob.sLnMRRd2qO0t.
