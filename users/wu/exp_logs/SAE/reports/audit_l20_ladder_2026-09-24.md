# Audit: L2-0 competence ladder read (2026-09-24)

Verdict: CONFIRMED_WITH_NOTES. The extracted numbers are right and comparable. The registered reads
below follow from them. Two of the notes limit what the ladder licenses (N1 and N2).

Scope: packs WX41NC734WLo (P), mZaZk7Ptt5Sg (R1) and UdhhxiGIMBob (R2); reader EktNvNSRrXKj; config
`config_sae_4a_lexlat_v2_ladder_v1.py`; rules from SAE_4A_lexlat_v2.md (L2-0, A4, A5, A7, A13).
Everything below was re-read from primary artifacts under `work/` and `output/`, not from
`reports/extract_l20_ladder_read_2026-09-24.md`. Read-only work: one CPU torch.load of five small
checkpoints to compare tensors.

## 1. Theta init, phi per arm, trainable parameters: CONFIRMED

Theta init, R1 and R2:
- Every R arm's `output/<arm>/returnn.config` sets `recognizer_checkpoint_path` =
  `FlatRecognizerInitJob.0J9d6wjrkRYH/flat_init.pt` (seed 0). The one exception is rt_r0_s2, which
  uses `.DMSwTLXT9MWG` (seed 1), with `random_seed = 1` and `random_seed_offset 1000`.
- `FlatRecognizerInitJob` zeroes only the output conv. Every other layer keeps torch's default init,
  and the job asserts that the logit spread is below 1e-6.
- No R arm config references p0 (`SupervisedRecognizerExportJob.YtuVg6ZYuK0P`).
- Config diffs: D10e `supphi_k2lat` against rt_r0 differs only in the recognizer path and in
  `lexlat_k2_chunk_seqs: 4`. rt_r0 against the other R arms differs only in the reverse path (none for
  cold_ctl), and for rt_r0_s2 in the flat file and the seeds. Node P against D10e differs only in
  the reverse path.

Step 0 of the training logs:
- Every seed-0 R arm reads `lexlat_k2` 1.053 with stability 0.081, identical across arms. That is
  the signature of one uniform theta, since k2 reads theta only. rt_r0_s2 reads 1.049 (different
  batch order).
- Every node P arm reads 0.500 with stability 0.003, which is the trained p0.
- Every arm logs "Starting training at epoch 1, global train step 0" and "learning_rates does not
  exist yet", so there were no resumes.
- Each arm ran on its own GPU and in its own work dir, and exited with rc 0 (pack engine logs).

Theta is learned, not loaded:
- In rt_r0, `lexlat_k2` falls from 1.053 at step 0 to 0.469 at step 14 and 0.308 at step 56.
- Cosine of rt_r0's ep1 conv weight with p0's conv is 0.833. For rt_r100 it is 0.046, and for
  cold_ctl 0.048.
- p0 itself is flat0 after 24 supervised steps (export stats: epoch 1, step 24). The largest
  |flat0 - p0| on any trained tensor is 3.2e-4, and p0's in_proj equals flat0's (cosine 1.000).
  That is ancestry: p0 descends from flat0. It is not leakage into the R arms.
- rt_r0_s2 starts from an in_proj orthogonal to p0's (cosine 0.000) and still reaches 0.1786.

Phi per arm (reverse-init job inputs):
- gold = `16v7R6ztSq1u`
- r30 = `HpmOCSCklRsB` (from Corrupt `xsTnpYUCPdlV`)
- r50 = `QGovsj3absf2` (`c4FmMx93HuK5`)
- r70 = `OLIUO3BGy0ug` (`mnzDC7XJX00r`)
- r100 = `Ud3O2Pa3Lotp` (`LCFfke9OsUqO`)
- permphi = `jjc51BwoW48b` (from Permute `skZmTlTbW6Fc`)
- cold_ctl: no reverse checkpoint. Its step-0 reverse term is -6.92 nats per frame, against -3.35
  to -5.04 for the fitted phis.

Corruption reports (train part, 2821 utterances):

| rho | realised substitution rate | corrupted-string PER |
|---|---|---|
| 0.3 | 0.300004 | 0.299843 |
| 0.5 | 0.500036 | 0.499064 |
| 0.7 | 0.699901 | 0.695282 |
| 1.0 | 1.000000 | 0.902755 |

- There are 0 collapses at every rho.
- Permutation: 0 fixed points, all 351,312 tokens changed.

Trainable: every arm logs "theta 5 tensors x1, phi 12 tensors x30", so both models train.

## 2. PER numbers: CONFIRMED

The chain per.txt -> BlankfreeGreedyPerJob -> ReturnnForwardJobV2 -> ExtractSubmoduleCheckpointJob
(prefix `recognizer.`) -> the arm's own `epoch.008.pt` (step 456) was traced for rt_r0, rt_r0_s2,
rt_r100, cold_ctl, rt_perm and corrphi_r30. The PER is theta-only greedy on features, and units are
not used.

Dev-other greedy PER at ep8 (S+D+I)/N, N = 177275, 2864 utterances (per.json):

| arm | PER at ep8 |
|---|---|
| rt_r0 | 0.177634 |
| rt_r0_s2 | 0.178576 |
| rt_r30 | 0.182643 |
| rt_r50 | 0.186326 |
| rt_r70 | 0.190901 |
| rt_r100 | 0.855169 |
| cold_ctl | 0.848698 |
| rt_perm | 0.865508 |
| corrphi_r30 | 0.181673 |
| corrphi_r50 | 0.182355 |
| corrphi_r70 | 0.188696 |
| corrphi_r100 | 0.852963 |

Comparability:
- p0's 0.189446 comes from `BlankfreeGreedyPerJob.kL6MFqeFIfCy`, with the same gold
  (`GoldPhonesJob.ZGSp0hxyd2YP`), the same dev-other features and the same N.
- The chance band uses the same job class, gold and N: k2lat_20 ep20 0.843469, and k2lat_20_x60
  ep21-60 0.840-0.849.
- All 24 decodes at ep1 and ep8 have distinct md5 hashes, so no two arms produced identical output.

## 3. Leakage: none beyond the disclosed seed labels

- Theta training data: features, units and original_length only, on train-clean-100 (28,539
  utterances) minus the 285-utterance CV holdout (`CvHoldoutSplitJob.PfpCPQRCfIAk`). The dev set
  inside training is that CV holdout. Dev-other enters only the evaluation.
- Features are wav2vec2-large-lv60 layer 15 (AvStatesJob, av_checkpoint None: self-supervised, no
  transcripts). Units are k-means (PackUnitsJob).
- The prior (`PhoneNgramPriorJob.RtzbESkOedsT`) is a seed-0 sample of the phonemized
  librispeech-lm-norm.
- The HLG (`cdcxYJMjiYj5`) is built from `LexiconTrieBuildJob.rlMsnTBSZXsB`: a word 3-gram on
  librispeech-lm-norm; the official librispeech-lexicon plus G2P for the LM corpus's OOV words.
- The OpenSLR LM corpus excludes books whose titles match dev/test books (a convention check only).
- The phi fit set is the SeedGoldPhonesJob MFA alignments of 2849 train-clean-100 utterances (2821
  fit, 28 held out). This is disclosed.
- Caveat: those 2821 seed utterances are inside theta's training stream. For rt_r0 this makes the
  run 10 h of pseudo-labelling through phi. That is the disclosed design, not a leak.

## 4. Registered reads, applied here

A4 at ep8:
- rt_r0, rt_r0_s2, rt_r30, rt_r50 and rt_r70 read LIFT (all below 0.50).
- rt_r100, cold_ctl and rt_perm read NO LIFT (all at or above 0.8164).
- The ladder is monotone: 0.1776 < 0.1826 < 0.1863 < 0.1909 < 0.8552. So rho*_lift = 0.7.

A7: the rt_r0 condition is met, since both seeds read LIFT. The SIGNAL half is still pending from
L2-1. L2-2 is therefore not yet funded.

Node P, D10e bands at ep8: r30, r50 and r70 read HOLD; r100 reads COLLAPSE. rho* = 0.7.

Node P, D14 rule. M_w = max(0.010, B_warm) = 0.010. B_warm is 0.00185 at ep4
(`PairedPerDeltaJob.cKduNKrfo5Jd`) and 0.00105 at ep8 (`3rNu5Z5YLwuT`).
- ep8: r30 -0.0078 [-0.0132, -0.0026], PRESERVES. r50 -0.0071, PRESERVES. r70 -0.0008,
  PRESERVES. r100 +0.6635, DEGRADES.
- ep4: all four DEGRADE (+0.0147, +0.0115, +0.0216, +0.6716).
- The paired job's `refines=True` for r30 and r50 at ep8 is its interval-only flag. It is not the
  D14 class, which is PRESERVES.

A5 bar, 260 set primary. Statistic (a) = tau 1 NLL, utterance mean, from the reader's report.txt:

| phi | gold | r30 | r50 | r70 | r100 | phi_c | permphi |
|---|---|---|---|---|---|---|---|
| (a) | 3.474 | 3.719 | 3.959 | 4.411 | 4.690 | 3.499 | 4.075 |

(a):
- Monotone in rho: yes.
- Separates LIFT from NO LIFT: no. permphi (NO LIFT) at 4.075 sits inside the LIFT range, between
  r50 and r70.
- phi_c on the non-lifting side: no (3.499, next to gold).
- permphi on its own arm's side: no.
- The pooled and F_2 variants fail the same clauses. F_2 puts phi_c at 3.130, below gold's 3.148.

Statistic (b):

| phi | gold | r30 | r50 | r70 | r100 | phi_c | permphi |
|---|---|---|---|---|---|---|---|
| (b) | +4.302 | +2.525 | +1.907 | +1.077 | +0.638 | +4.555 | +4.101 |

(b):
- Monotone: yes.
- Separates: no. permphi at +4.101 is above r30 through r70.
- phi_c: no (above gold).
- permphi: no.

The 285 set gives the same conclusions: every value is within 0.015 of its 260 value, and the same
clauses fail.

cold_ctl's random phi has no statistic (UNDETERMINED in the config). This does not change the
verdict, because other clauses already fail.

G4a.L2.3 = CANNOT_TELL (NO SEPARATING STATISTIC).

## 5. Artifact check

- The mechanism (theta pulled toward phi's generative posterior under the joint EMC objective) is
  the registered one. L2-2's dec_joint uses the same objective, so the effect is by design, not a
  bug.
- There are no duplicate outputs and no cross-arm checkpoint loads.
- The fast lift (0.18 within sub-epoch 1, 57 steps at lr 1e-5) is plausible: p0 reached 0.1894
  in 24 supervised steps from the same flat init, because theta is a one-layer readout.

N1, the frame of rho*_lift:
- The corruption substitutes symbols independently of the acoustics. The fitted emission is then
  roughly (1 - rho) x true + rho x marginal, which preserves the true ranking at any rho below 1.
- So rho*_lift = 0.7 bounds only this benign family. The transition lies somewhere between 0.7 and
  1.0 and was not probed.
- Structured-error phis (permphi, phi_c) do not lift. "About 30 % correct labels suffice" must not
  be read across to an EM phi.

N2, the label-free statistics: (a) and (b) rate phi_c as at least as competent as gold, and place
permphi among the lifting phis. The statistics the ladder was meant to calibrate do not track
lifting.

N3, minor: the registered text says "a fraction rho ... run-collapse may shorten". The
implementation substitutes exactly round(rho x n) per utterance with no collapse. Realised rates
match rho, so this is not decisive.

## Sources (convention check only)

- https://arxiv.org/pdf/2012.03411 (MLS paper, describing LibriSpeech LM-corpus filtering of
  dev/test books)
- https://www.danielpovey.com/files/2015_icassp_librispeech.pdf
