# Audit: A14 (i) lift read and the A18 hold-rule clause for em_s13 (2026-09-24)

Status: CONFIRMED_WITH_CORRECTIONS. The A14 (i) verdict EM PHI DOES NOT LIFT is correct: I re-derived it from the decode and gold files. The hold-rule clause is met by a wide margin when em_s13's S at its final checkpoint (sub-epoch 12) is compared with the tested arms' S at sub-epoch 48. But the registered clause asks for "the candidate's S at 48", and em_s13 has no sub-epoch-48 checkpoint, so that value does not exist. The clause can only be applied with an interpretation, and that interpretation should be recorded as one. If the comparison is made at the same sub-epoch instead (12 against 12), the clause fails, but only narrowly and within noise.

## 1. A14 (i) verdict, re-derived

Chain checked for every arm at ep8: `A14LiftReadJob.GAKp6IJ5pA3l` reads `BlankfreeGreedyPerJob`, which reads `ReturnnForwardJobV2`, which loads `ExtractSubmoduleCheckpointJob` (prefix `recognizer.`), which comes from `PackedBlankfreeTrainJob.e9xZa5ElF16P/output/<arm>/models/epoch.008.pt`. The aliases and arm names agree at every step. The cold_ctl baseline comes from `PackedBlankfreeTrainJob.UdhhxiGIMBob/output/cold_ctl/models/epoch.008.pt`.

I recomputed the ep8 PER independently: the Levenshtein distance between each arm's `greedy_phones.json` and `GoldPhonesJob.ZGSp0hxyd2YP/output/gold.json["dev-other"]`, summed over utterances and divided by the number of reference phones. I did not use the per.json summary numbers. Scoring set: 2864 of 2864 dev-other utterances, 177,275 reference phones, no SIL in either the gold or the hypotheses. That is the full dev-other.

| arm | errors | PER ep8 (mine) | reader | class (LIFT < 0.50, PARTIAL < 0.8164) | distance above PARTIAL |
|---|---|---|---|---|---|
| rt_em_durinit_s1 | 148870 | 0.839769 | 0.839769 | NO LIFT | 0.0234 |
| rt_em_durinit_s2 | 145374 | 0.820048 | 0.820048 | NO LIFT | 0.0036 |
| rt_em_durfrz_s1 | 149319 | 0.842302 | 0.842302 | NO LIFT | 0.0259 |
| rt_em_durfrz_s2 | 145476 | 0.820623 | 0.820623 | NO LIFT | 0.0042 |
| cold_ctl (beside) | 150453 | 0.848698 | 0.848698 | - | - |

All four arms read NO LIFT, so the verdict is **EM PHI DOES NOT LIFT**, as registered. For all 20 per.json files (ep1/2/4/8 for each arm and for cold_ctl), (S+D+I)/N reproduces the printed PER exactly.

Phi inits: the four `reverse.` extractions (c5WVHZvlSuhF, sPCV5PXQWpJJ, H7YC2Iub8bOh, vM4xvAHEdVM0) come from the `epoch.048.pt` of PhiFirstProbeTrainingJob tDzmBsvX73X5 (ext/durinit_s01), KWQGSa1BfXon (durinit_s02), SxUAdNXSRuNy (durfrz_s01) and vH8KrqqxQtte (durfrz_s02). Each is recorded at epoch 48, step 2736, 12 keys. The load is strict: `_load_state` raises on any missing or unexpected key.

Recipe: each arm's `returnn.config` differs from L2-0 R2's `rt_r70` only in `reverse_checkpoint_path` and the output path. It differs from `cold_ctl` only by adding that path. The four arms differ from each other only in the phi path, so they share the seed, data order and theta init (`FlatRecognizerInitJob.0J9d6wjrkRYH`). The k2 block, LR schedule [1e-5, 1e-4 x 7] and reverse LR multiplier of 30 are the same as in L2-0.

Completion: every arm has 8 epochs in `learning_rates` with finite losses, 57 steps per epoch, and "Finished training at epoch 8, global train step 456" in its log. The logs have 0 nan, 0 inf and 0 tracebacks, and `stop_on_nonfinite_train_score` is set. The checkpoints epoch.001/002/004/008 are present.

## 2. Hold-rule clause (A18): em_s13 against the best A14 (i) arm

S = the utterance mean of `nll_tau1 / frames` from each read's `genmarg.json`. The wave read and the A10 reads have identical `settings` and `conventions` blocks: null recognizer, trigram prior, float64, blank-free lattice with d_min 2. All reads are finite on all 285 utterances. The 260 set is `CvDisjointSegmentsJob.PvgJ79Qc1Nro/output/disjoint.segments` (260 unique tags, a subset of the 285).

| phi | S, 285 set | S, 260 set | source |
|---|---|---|---|
| A10 durinit_s1 @48 (best arm) | 3.30403 | 3.29903 | ReturnnForwardJobV2.M9II4LBlD7rC |
| A10 durinit_s2 @48 | 3.37600 | 3.37032 | xjnwm8MDWHni |
| A10 durfrz_s1 @48 | 3.37201 | 3.36665 | u9ya1iu8rxOY |
| A10 durfrz_s2 @48 | 3.40582 | 3.39956 | Onpcw1Cj4DHr |
| em_s13 @12 (wave, selected; epoch.012.pt of KCj5mptWgBqb) | 3.38630 | 3.38099 | i1PBNWq9yDsD |
| A10 durinit_s1 @12 | 3.39810 | 3.39171 | 6BtjG9YXfDIu |
| A10 durinit_s2 @12 | 3.44519 | 3.44038 | JetuyTq5zCBb |

These match the doc's A10 table (260 set: 3.2990, 3.3703, 3.3667, 3.3996) and the wave's 3.3863.

- **em_s13 on the 260 set:** no job prints this value. GenMargSelectionJob reads the 285 set only. I computed it from em_s13's per-utterance genmarg.json, which contains all 260 tags: 3.38099. Under the project's derived-statistic rule, a registered reader would have to print it before it is recorded.
- **Is the 285-set comparison valid?** Yes. The A10 restarts and the wave restarts all train on `CvHoldoutSplitJob.PfpCPQRCfIAk/output/train.segments`, and I checked that it shares 0 utterances with the 285 `cv.segments`. A13's overlap concerns only the ladder phis, which were fitted on the 2821-utterance split. The set choice moves the difference by 0.0003.
- **Clause, as-sent reading** (em_s13's sent checkpoint at 12 against the tested phis at 48): S(em_s13) - S(best arm) = **+0.0820 on the 260 set** (+0.0823 on the 285 set). The paired bootstrap 95 % CI is [+0.072, +0.092], and em_s13 has the lower S on only 15 % of utterances. The candidate is not below the best arm. It is above it by 0.082, so the clause is met: em_s13 would need to be 0.092 lower to break it. em_s13 lies inside the range of the tested phis. It is worse than 3 of the 4, and better only than durfrz_s2, by 0.019.
- **Frame problem.** The registered text asks for "the candidate's S at 48". The wave ran 12 sub-epochs and sends epoch.012.pt, so em_s13 has no S at 48. The as-sent reading above is an interpretation. It is the one that matches the rule's purpose ("really sure it cannot lift" concerns the phi that would actually be sent), but it is not the literal text.
- **Matched-sub-epoch reading** (12 against 12): em_s13 - durinit_s1@12 = **-0.0107 on the 260 set** (-0.0118 on the 285 set). The paired CI is [-0.0210, -0.0004] on the 260 set, and em_s13 is lower on 55 % of utterances. That is below by more than 0.01, so the clause fails, but only by 0.0007 (260) or 0.0018 (285), and the CI does not separate the difference from the 0.01 floor.
- **Projected-to-48 reading:** this would need em_s13 trained to 48. It cannot be decided from these files.

## 3. Same line

The wave arm's config (em_s13, pack KCj5mptWgBqb) and A10 durinit_s01's config (tDzmBsvX73X5) differ only in these respects:
- the number of epochs (12 against 48), with the tau list [4, 1, ...] and the constant LR list 1e-4 truncated to match;
- the seed (13 against 1);
- `cleanup_old_models` and the output path;
- one memory-logging flag.

Duration prior mode, prior, data, CV filter, lexicon and loss settings are identical. As an empirical check, wave em_s01 at 12 and A10 durinit_s1 at 12 agree per utterance: the mean difference is -3.9e-6 and the largest is 7.5e-4. The wave is therefore A10's durinit recipe stopped at sub-epoch 12. em_s02 against durinit_s2@12 differs by 4.2e-4 on average, which is run-to-run drift.

Same line therefore holds for the two durinit arms, including the best-S arm. The durfrz arms are the same EM recipe with the phone-duration rows frozen during EM. A18 (a) itself names A14 (i) as the same-line read, and all four arms read NO LIFT.

## 4. Things that bear on the verdict or the clause

1. **The frame issue in section 2.** Whether em_s13's 4-GPU joint run is withheld depends on how "S at 48" is read for a candidate trained for 12 sub-epochs. As-sent: withheld, by a wide margin. Matched length: not withheld, by a margin smaller than its noise. The reading should be recorded as an interpretation of A18 and not presented as the literal clause.
2. **NO LIFT is close to PARTIAL.** Two arms are 0.0036 and 0.0042 PER above the PARTIAL bar, which is 647 and 747 phone errors out of 177,275. Each arm is one run. Project memory records a same-config spread of 0.01-0.03 PER by ep4 on this bed, which I did not re-verify here. So the verdict is correct as a registered point read, but one rerun could plausibly land an arm in PARTIAL. All four paired rows against cold_ctl are negative with CIs that exclude 0 (-0.0064 to -0.0287). The EM phis therefore do pull theta, weakly. They do not pull it as far as PARTIAL. "Does not lift" means a failed band. It does not show that no lift is possible, and that matters for a rule whose ground is "really sure it cannot lift".
3. Not verdict-changing: the joint dev_loss_agg reaches its minimum at ep4-5 and rises by ep8 in all four arms, and PER rises from ep4 to ep8 in three of them. A14 (i) registers ep8, so this is reported only.

## Files

- Reader: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_a14_jobs/A14LiftReadJob.GAKp6IJ5pA3l/output/{a14_lift.json,report.txt}
- Pack: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.e9xZa5ElF16P/output/<arm>/{returnn.config,learning_rates,log.run.1,models/}
- ep8 PER jobs: BlankfreeGreedyPerJob PjieUNGdqc1e, jgAbDU3u51S9, l7PM0KoyzBql, ypUbUAlGUy7p, and sQ3EqBbi53lJ (cold_ctl), under work/speech_llm/sae/emc/blankfree_eval_jobs/
- S reads: work/i6_core/returnn/forward/ReturnnForwardJobV2.{M9II4LBlD7rC,xjnwm8MDWHni,u9ya1iu8rxOY,Onpcw1Cj4DHr,6BtjG9YXfDIu,JetuyTq5zCBb,i1PBNWq9yDsD}/output/genmarg.json; wave inputs in GenMargSelectionJob.dPjElzqvLUYt/output/selection.json
- 260 set: work/speech_llm/sae/emc/blankfree_a13_disjoint/CvDisjointSegmentsJob.PvgJ79Qc1Nro/output/disjoint.segments
- Scratch recompute scripts (not part of the project): /tmp/claude-34349/-e-project1-spell-wu24-2026-07-13-unsupervised/a2d26b34-6240-4bb2-b34b-ab87c54fe915/scratchpad/{per.py,s.py,s2.py}
