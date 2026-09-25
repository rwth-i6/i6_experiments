# Audit: GOLD KEY REACHES BASIN (A16 (b) stage-2 control) and A17 (iii), keyinit pack ge1MKcAPmZIV (2026-09-24)

Verdict: CONFIRMED_WITH_CORRECTIONS. Both registered verdicts re-derive exactly from the per-utterance genmarg jsons.
The comparison holds: same set, same S, same pairing, same A10 files as A14 (ii). All four inits carry exactly A9's
durinit duration logits, not MFA-fitted ones. The corrections concern how LABELS SUFFICE should be read and how thin
its deciding clause is; neither verdict changes.

## 1. Re-derived numbers (my recomputation, not copied from the readers)
Method: I loaded the sub-epoch-48 `genmarg.json` of each arm and of the six A10 restarts (paths taken from the
readers' `info` kwargs) and computed S = mean over tags of `per_utterance[t].nll_tau1_per_frame`. T is the 260-set
tags (`CvDisjointSegmentsJob.PvgJ79Qc1Nro/output/disjoint.segments`) that are possible in every compared json. The
script is in the audit scratchpad (s_recompute.py).
- T = 260 of 260, with no tag excluded.
- A10 at 48: durinit_s01 3.29903 (S_min), uniform_s02 3.32244, uniform_s01 3.34756, durfrz_s01 3.36665,
  durinit_s02 3.37032, durfrz_s02 3.39956. The unrounded bar is 3.2890300.
- gold_key: S48 = 3.20704. That is 0.08199 below 3.28903. S_c - S_min = -0.09199, lower on 227/260 utterances.
  **GOLD KEY REACHES BASIN** holds.
- g_dur: S48 = 3.22387, 0.06513 below 3.289 (lower than durinit_s01 on 211/260).
- r70_dur: S48 = 3.26568, 0.02332 below 3.289 (lower on 170/260).
- r30_dur (reported only): 3.21151.
- **LABELS SUFFICE** holds: G-dur < 3.289 and r70-dur < 3.289.
- A14 (ii) re-derived with the same code: gold 3.21602, r30 3.20982, r70 3.27068, r100 3.37793. These match the doc.
- Paired against the A14 (ii) arm with the same emission source: G-dur - A14 gold = +0.00785 (lower on 103/260);
  r30-dur - A14 r30 = +0.00169; r70-dur - A14 r70 = -0.00500; gold_key - A14 gold = -0.00898.
- The readers' reports and jsons agree with all of these values to 5 decimals.
  - Reader files: `keyinit_read_jobs/KeyInitControlReadJob.JH7zzrX3Egfm/output/{report.txt,keyinit_control.json}`
    and `keyinit_read_jobs/A17SegmentationReadJob.lg4fTnb7deS8/output/{report.txt,a17iii.json}`.

## 2. Comparability (same set, same S, same pairing)
- Both readers import `_s`, `_possible_in` and `_check_cv` from `blankfree_a14_jobs`, so the code is the same as the
  A14 (ii) reader.
- The `a10_marginals`, `disjoint_segments` and `a10_diagnostics` inputs are identical, path for path, to
  those of `A14ObjectiveFloorReadJob.PiYQ1OCFD4ot`.
- The genmarg `settings` block of all 4 arms x 49 sub-epochs equals A14 gold ep48's settings exactly, which in turn
  equal A10 durinit_s01's. The block covers the null recognizer, temperatures [1,2], the trigram prior RtzbESkOedsT,
  blankfree lattice d_min 2, float64, cv_holdout, and shuffle_seed None.
- Every read at sub-epochs 1..48 loads `ge1MKcAPmZIV/output/<arm>/models/epoch.0NN.pt` for its own arm and epoch.
  Every read at sub-epoch 0 loads the arm's init `phi.pt`. I checked the forward configs of all 4 x 49 marginals and
  decodes.
- Bar: the control uses the paired 3.28903. The A17 reader uses a fixed 3.289 (`SEGMENTATION_BAR`), which is not
  "the same unrounded bar" as the doc (line 276) says. This is a minor wording correction and is immaterial here,
  because the smallest margin is 0.023.

## 3. Recipe and durations (the MFA-leak check)
- Configs: the arm returnn.configs differ from A10 durinit s1 (`PhiFirstProbeTrainingJob.tDzmBsvX73X5`) only in
  one place. `reverse_duration_prior` and `mode "init"` are replaced by `reverse_checkpoint_path`. Against A14 gold
  (`fTBdXD0SwBaA`) only the checkpoint path differs, and the four arms differ from each other only in that path.
  - Mode "init" has no training-time effect beyond construction (grep of sae_blankfree.py).
  - Each arm log shows a single "Starting training at epoch 1, global train step 0".
- Durations: I loaded the init checkpoints myself.
  - `dur_logits` is bit-equal to `durinit_logits(cfg, 4.413787)` in all four inits, from the same prior.json
    (`BlankfreeDurationPriorMeanJob.ReQtJKYpZgsN`) that A10 durinit uses.
  - E[d] is 4.4138 for every phone and 26.0 for SIL. This is identical to the A10 durinit s1 start log line.
  - The MFA-fitted source durations were replaced. Before the swap, phone E[d] ranged 2.80-8.80 (gold),
    3.28-7.46 (r30) and 3.59-6.41 (r70); the maximum |logit - durinit| was 6.4-6.7.
  - **No MFA duration leak.**
- Emission source:
  - G-dur, r30-dur and r70-dur come from `BlankfreeSupervisedReverseInitJob` 16v7R6ztSq1u, HpmOCSCklRsB and
    OLIUO3BGy0ug. These are the A14 (ii) gold/r30/r70 inits; I confirmed them against the A14 reader's ep0 forwards.
  - All 11 non-duration tensors are bit-identical to the source.
- Gold key: `PhiFromKeyInitJob.f0jaGuiJVe6A` takes only two inputs, `key.json` and unsupervised unit frame counts.
  - `key.json` holds 500 ints, each unit's majority phone on the 2821 L2-0 fit utterances; frame purity is 0.644,
    and OY and ZH are empty.
  - The unit frame counts come over 28,254 train utterances, with smoothing 0.1.
  - emb_dur, emb_pos and eta are zeroed, so the emission is identical in every duration/position cell (JS <= 2.2e-12;
    I measured 0.0). The durations are durinit.
  - It is type-level only.

## 4. Correction to the reading of A17 (iii): swapping the durations does not remove segmentation
- The source phis' emission heads depend on the duration-bucket and position-in-segment cells, and they were fitted
  on MFA segments.
  - I measured the frame-weighted JS between each cell and the type marginal on 256 etas: a mean of 0.206 nats per
    phone type for the G-dur init and 0.162 for the r70-dur init; the gold_key init has 0.
  - The Viterbi boundaries at sub-epoch 0 confirm it. G-dur has F1 0.834 against A14 gold's 0.839, so the swap barely
    changed the init segmentation.
- As registered, the arms are defined by the duration swap, so the verdict stands. It licenses "the A14 (ii) basin did
  not depend on MFA-fitted duration logits".
- The cleaner no-segmentation evidence is the gold-key control. It has durinit durations and cell-independent
  emissions, and it reaches the basin by a larger margin (0.082).

## 5. Margins
- G-dur clears the bar by 0.065 and gold_key by 0.082.
- The clause separating LABELS SUFFICE from PARTIAL-LABELS NEED SEGMENTS (which would hold the key arms) is r70-dur
  alone, at 0.023 below the bar on one seed.
  - No same-init seed repeat exists for phi EM.
  - The six A10 random restarts span 0.10 in S.
  - The rule has no noise clause, so the verdict holds, but the not-held consequence rests on this single thin margin.
  - A14 r70 (MFA durations) sat 0.018 below the bar, which is consistent.

## 6. Generative PER on dev-other (direct / Hungarian / NMI), re-read from the GenDecodeReportJob report.json files
All reads cover 500 utterances and 29,690 reference phones, the same as A14.

| arm | ep0 | ep4 | ep12 | ep48 |
|---|---|---|---|---|
| gold_key | .327/.385/.740 | .284/.334/.747 | .322/.367/.717 | .346/.391/.695 |
| g_dur | .197/.197/.836 | .297/.297/.751 | .324/.324/.730 | .345/.345/.710 |
| r30_dur | .248/.248/.803 | .327/.327/.725 | .350/.350/.705 | .368/.368/.685 |
| r70_dur | .605/.622/.452 | .430/.453/.609 | .435/.459/.606 | .442/.466/.606 |
| A14 gold | .193/.193/.835 | .300/.300/.745 | .328/.328/.722 | .353/.353/.703 |
| A14 r30 | .240/.240/.801 | .350/.350/.705 | .383/.383/.682 | .394/.394/.671 |
| A14 r70 | .608/.628/.410 | .477/.477/.564 | .482/.482/.572 | .495/.508/.566 |
| A14 r100 | .826/.830/.112 | .846/.850/.081 | .852/.852/.078 | .860/.861/.070 |

- S and PER agree between basins: every arm below the bar ends at PER 0.34-0.47, against r100 and the A10 restarts
  in the chance band.
- They disagree inside the basin:
  - gold_key has the lowest S but the same PER as g_dur.
  - r30_dur has a lower S than g_dur but a higher PER.
  - Along the trajectory, S falls while PER rises: g_dur +0.148, r30_dur +0.120, and gold_key +0.062 after ep4.
    r70_dur is the exception; its PER improves.
- For gold_key, the Hungarian PER is above the direct PER (0.391 against 0.346), so the Hungarian map is not the
  identity there. Two symbols are empty in the key.
- The durinit arms end 0.008-0.053 lower in PER than their MFA-duration A14 twins. There is one seed and no PER spread,
  so this is descriptive only.

## 7. Boundaries against MFA (SegmentationBoundaryJob.g52gIGtUHpkr; 260/260 covered, 20 ms tolerance, MFA 11.904 Hz)
- F1 at ep48:

  | arm | F1 |
  |---|---|
  | gold_key | 0.799 |
  | g_dur | 0.798 |
  | r30_dur | 0.791 |
  | r70_dur | 0.776 |
  | A14 gold | 0.800 |
  | A14 r30 | 0.785 |
  | A14 r70 | 0.777 |
  | A14 r100 | 0.660 |

- At ep48 all basin arms run at 10.02-10.53 Hz with OS between -0.12 and -0.16; A14 r100 runs at 9.13 Hz.
- At ep0, gold_key has F1 0.813 at 12.51 Hz (OS +0.05), and r70_dur/A14 r70 have 0.531/0.523.
- The segmentation converges to the same quality whatever the duration init.
- Consistency checks: F1, OS and the R-value recompute from the json P and R, every report line matches the json,
  and there are no skipped utterances.

## 8. Integrity around the 22:16 restart
- g52g:
  - The old manager submitted it at 22:15:38. Its inputs were complete by then; the last one was the gold_key ep48
    CV decode, which finished at 22:15:36.
  - It ran as one worker (pid 2465778) from 22:15:39 to 22:16:11 and wrote its outputs once, at 22:16:11. The archive
    holds one log.run.1, one usage.run.1 and one finished.run.1, ending "Job finished successfully".
  - The new manager (config loaded 22:16:06) logged "changed, recovering it anyway". In sisyphus
    `localengine.try_to_recover_task`, this means the running pid's cmdline differs from the call the new manager
    would issue, which fits the restarted python environment. It adopted that pid; it did not start a second run.
- lg4f: pid 2466538 ran from 22:15:55 to 22:16:04 in a single run. Its inputs were final by 22:15:45.
- Control JH7z: pid 2466537 ran from 22:15:52 to 22:15:55.
- The output symlinks resolve to these job dirs.
- No partial or double write was found.
- `uRzbIh0EfQXG`, missing in the extraction, is A18 (b)'s DurinitBasinLiftReadJob, which is still waiting. It is
  unrelated to these reads.
