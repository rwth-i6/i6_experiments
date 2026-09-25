# Audit: A17 (i) and A17 (ii) reads (2026-09-24)

Status: CONFIRMED_WITH_CORRECTIONS. Both verdicts hold as registered. The corrections concern how A17 (ii) is recorded, not what it reads.

Registered text read: SAE_4A_lexlat_v2.md lines 285-306 (A17 (i), (ii), including line 306 on the spread), line 62 (A4 bands), lines 341-342 (A18 (d); the brief's 343-345 now falls on the A18 build-choice lines), 641-669 (A14 (ii) read), 715-735 (A14 (i) read).

## A17 (i): BASIN SUFFICIENT, confirmed

Recomputed the dev-other greedy PER at ep8 from each arm's `greedy_phones.json` against `GoldPhonesJob.ZGSp0hxyd2YP/output/gold.json` (dev-other). All 2,864 utterances were scored, none were missing, and the denominator is 177,275 reference phones. The results match `per.json` and `a17_lift.json` exactly.

| arm | errors | PER ep8 | band | margin to the relevant bar |
|---|---|---|---|---|
| gold-EM (`BlankfreeGreedyPerJob.lnBncUyLK9LE`) | 35,532 | 0.2004 | LIFT | 0.300 below 0.50 |
| r30-EM (`CcGCFp2OWhSG`) | 35,587 | 0.2007 | LIFT | 0.299 below 0.50 |
| r70-EM (`WbGJ6P5xrKuI`) | 64,167 | 0.3620 | LIFT | 0.138 below 0.50 |
| r100-EM (`33orW7QAlQzl`) | 148,715 | 0.8389 | NO LIFT | 0.0225 above 0.8164 |

The rule holds: gold-EM and r30-EM lift, and r100-EM reads NO LIFT, so the result is BASIN SUFFICIENT. It is not VOID.

Provenance and frame:
- **Chain of custody.** Each PER job reads a posteriors forward, which reads `ExtractSubmoduleCheckpointJob` (prefix `recognizer.`), which reads `PackedBlankfreeTrainJob.T18RrTNTdg65/output/<arm>/models/epoch.008.pt`.
- **Phi inits.** Each arm's `reverse_checkpoint_path` is an extraction (prefix `reverse.`) of an A14 (ii) sub-epoch-48 checkpoint:
  - gold: `fTBdXD0SwBaA` (`PMxbuDHd9fOA`);
  - r30: `T2V8nn5obzj9` (`ROzbAnfsAQLP`);
  - r70: `MU6Q3RanOQqF` (`g4yq6c7rOdcG`);
  - r100: `lR4CfDiHyAvH` (`7fETF1FEOo0A`).
  These are the same run hashes that A14 (ii)'s `A14ObjectiveFloorReadJob.PiYQ1OCFD4ot` decodes at sub-epoch 48 (aliases `l21_a14_floor_<init>_s01_ep48`). The ep0 phi reports reuse A14 (ii)'s sub-epoch-48 reports and give 0.3529 / 0.3941 / 0.4949 / 0.8601 direct PER, which matches the registered 0.353 / 0.394 / 0.495 / 0.860.
- **Phi was loaded and trained.** The gold-EM phi reads 0.339 at pack ep1 (`GenDecodeReportJob.yXtk9HBGcVbE`, taken from pack epoch.001) and 0.232 at ep8. A random phi would read about 0.86.
- **Recipe.** The resolved `returnn.config` of the A17 gold arm was diffed against all four A14 (i) arms (`PackedBlankfreeTrainJob.e9xZa5ElF16P`). The only differences are `reverse_checkpoint_path` and the model output path. The four A17 arms differ from each other in the same two lines only.
  - The config has `num_epochs = 8`, `reverse_lr_multiplier` 30, and no freeze flag, so both models train.
  - All four arms log "Finished training at epoch 8, global train step 456", the same as A14 (i). No log contains nan, nonfinite or Traceback, and `stop_on_nonfinite_train_score = True`.
- **Code state.** No commit touched `model/definitions/sae_blankfree.py`, `train_steps/sae_blankfree.py` or `sae/emc` after 12:09. A14 (i) started at 18:58 and A17 (i) at 19:17. The working tree was clean on those paths.
- **Caveat, known and not a correction.** The basin phis kept supervised MFA durations (A14 (ii) audit; State line 11). So BASIN SUFFICIENT is shown for phis that carry MFA durations. A17 (iii) tests whether segmentation carries the lift.
- **Robustness.** r100-EM's NO LIFT margin (0.0225) is of the order of the 0.01-0.03 same-config PER spread recorded for the joint bed. The A14 (i) content-free EM phis landed at 0.820-0.849. A rerun crossing into PARTIAL, which would make the read VOID, is unlikely but not excluded. The gold-EM and r30-EM margins are about 0.30.

## A17 (ii): OBJECTIVE DRIFT by the letter, margin +0.0019

Recomputed from the gold arm's diagnostics. The GenDecodeReportJob counts were checked, and PER was recomputed independently from `decode_raw.json` (SIL dropped) against gold on D4: 500 utterances, 29,690 reference phones.
- PER(0) = 5,738 / 29,690 = 0.19326 (`GenDecodeReportJob.KgnN187NUtKd` on forward `WPtevtMemFK9`, checkpoint `BlankfreeSupervisedReverseInitJob.16v7R6ztSq1u/epoch.008.pt`).
- PER(12) = 8,756 / 29,690 = 0.29491 (`JJbP7TjuCk60` on forward `4ieB7Y5feLS3`, checkpoint `PhiFirstProbeTrainingJob.nVpD2O3xpfcJ/epoch.012.pt`).
- Drift against the registered 0.193: +0.10191. That is 0.0019 above the 0.10 bar, or 57 phone errors.
- Drift against the measured PER(0): +0.10165. That is 0.0017 above the bar, or 49 errors.
- For comparison, A14 (ii) gold PER(12) is 0.32779 (`JUmW3uaPaQru`), a drift of +0.1348. A17 minus A14 at matched sub-epochs is -0.0335, -0.0334 and -0.0329 at 4, 8 and 12. So the tau = 4 sub-epoch accounts for about 0.033 of A14 (ii)'s 0.135, and about 0.10 remains at tau = 1.

Recipe and frame:
- **Resolved configs.** `nVpD2O3xpfcJ` (gold) and `YtsRkvAl7Kl8` (r70) were diffed against A14 (ii)'s `fTBdXD0SwBaA` and `MU6Q3RanOQqF`. The only differences are:
  - `temperature_schedule` = [1.0] x 12, against A14's [4.0, 1.0 x 47];
  - `num_epochs` 12 against 48;
  - the learning_rates list truncated to its first 12 entries (the prefix is unchanged);
  - the keep list and the model path.
- **Unchanged from A14 (ii).** The init (`16v7R6ztSq1u` ep8), the durations handling, freeze_recognizer/null_recognizer, lam and the prior are all identical.
- **Logs.** `probe.txt` shows temperature 1.0000 at sub-epoch 1, where A14 (ii) shows 4.0000. The runs completed sub-epochs 1-12 at 57 steps each, `l_tau_all_finite` is True, and `log_events.jsonl` ends in run_end.
- **Decode configs.** The ep0 and ep12 decode configs are identical apart from the checkpoint path and the name.
- **PER(0) is the init phi.** It is the same GenDecodeReportJob that A14 (ii) uses at sub-epoch 0, run on the init checkpoint.

Readings that could move the verdict:
- **Direct vs Hungarian.** The registered text (lines 299-303) does not fix which PER is read. The reader uses Hungarian and prints direct. For gold, the Hungarian map is the identity (40 of 40 symbols), and the sub/del/ins counts are identical at 0, 4, 8 and 12. The choice has no effect.
- **PER(0) as registered (0.193) vs measured (0.19326).** Both give a drift of 0.10 or more. No effect.
- **Set size.** The auditor's own calculation (not a banked number) is a paired bootstrap of the drift over the 500 D4 utterances:
  - by utterance: SE 0.0024, 95 % interval [0.0970, 0.1062], with 75 % of resamples at 0.10 or more;
  - by speaker cluster (33 speakers): SE 0.0031, interval [0.0954, 0.1078], with 70 % at 0.10 or more.
  So OBJECTIVE DRIFT versus MIXED is not resolved beyond the sampling error of the evaluation set. ANNEALING-DOMINATED (below 0.05) lies more than 15 SE away.
- **Same-config spread.** No same-config repeat of the gold tau = 1 run exists. The A14 (ii) sub-epoch-0 value is not a repeat, because it is the identical job (`KgnN187NUtKd`) reused by both reads. Line 306's "same-config PER spread (0.01-0.03)" is the joint-recognizer bed's figure. For this phi-first stage-1 bed, the only rerun evidence is on S: identity bands of 2.7e-5 (G4a.L2.2 wave) and 3.4e-5 (A10). No PER repeat exists. A seed repeat of the gold arm would be needed to size run-to-run PER spread here.
- **Consequences.** Only ANNEALING-DOMINATED carries a registered consequence (line 305). OBJECTIVE DRIFT and MIXED lead to the same decisions: no tau = 1 amendment to A16 (b) stage 2, and no A18 (d) pack. The boundary sensitivity therefore changes no decision, only the label.

## A18 (d) condition: not met

The condition is A17 (i) gold-EM NO LIFT AND A17 (ii) ANNEALING-DOMINATED. Both clauses fail, each by a wide margin:
- gold-EM reads 0.2004, which is LIFT;
- the drift is 0.102, far above the 0.05 bar.

## Corrections for the record

1. Record A17 (ii) as "OBJECTIVE DRIFT (drift +0.102 against the 0.10 bar, margin 0.002 = 57 of 29,690 phones; MIXED is within the D4 sampling interval; not ANNEALING-DOMINATED under any reading)". Do not record it as a clean OBJECTIVE DRIFT.
2. Line 306's 0.01-0.03 spread is not measured on the phi-first stage-1 bed. Cite it as the joint bed's figure, or measure a gold-arm seed repeat.
3. The registered text does not fix direct vs Hungarian. The choice is immaterial for gold, but future drift clauses should name the PER.
4. Recording note for A17 (i): the r100 control's NO LIFT margin is 0.0225, and the basin phis carry MFA durations (pending A17 (iii)).
