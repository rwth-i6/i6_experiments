# Audit: G4a.L2.2 wave read (GenMargSelectionJob.dPjElzqvLUYt), 2026-09-24

Status: CONFIRMED_WITH_CORRECTIONS. The corrections are to the report text only. They change neither the verdict nor the selected restart.

## Verdict re-derived

I recomputed everything from the 28 per-utterance `genmarg.json` files and the 16 `gendecode.json` files that the job read (paths are in `selection.json` under "inputs"). I did not use the job's summary numbers.

- S = the utterance mean of `nll_tau1 / frames` over the 285 CV-holdout utterances. All 28 json files have the same 285 tags, which equal `CvHoldoutSplitJob.PfpCPQRCfIAk/output/cv.segments`. That file shares no utterance with train.segments. No utterance is impossible in any file.
- Restarts (real holdout): the lowest S is em_s13 at 3.386298, then em_s01 at 3.398101 and em_s04 at 3.399038. The highest is em_s02 at 3.444774.
- A8 emitted rate: I recounted the non-SIL segments of each decode and divided by the summed orig_length, pooled over all 285 utterances. The range is 7.047 Hz (em_s02) to 7.581 Hz (em_s13). All 16 restarts are inside [5.80, 14.49] Hz, so none is VOID and all 16 are eligible.
- Nulls (permuted holdout): null_s01 5.715990, null_s02 5.713272, null_s03 5.714981, null_s04 5.714854. The best null is null_s02 and the null range is 0.002718.
- Identity band: em_s01 against em_s01_rerun differs by 1.5e-6, and em_s02 against em_s02_rerun by 2.73e-5. The band is 2.73e-5.
- Margin = max(0.002718, 0.0000273, 0.01) = 0.01 nats per frame.
- Gap (selected log-likelihood minus best null log-likelihood) = 5.713272 - 3.386298 = 2.3270. This is far above 0.01, so the gate reads SIGNAL.
- Paired per utterance, em_s13 has a lower S than null_s02 on 285 of 285 utterances. The smallest per-utterance difference is 0.70 nats per frame.
- Report-only phi_c comparison: the best phi_c restart is phic_s01 at 3.344147. S(phi_c) - S(selected) = -0.0422, which reads NOT BEYOND. The selected random restart is worse than phi_c by more than the margin, and is better on only 31% of utterances.
- Robustness to the per-frame convention: with pooled per-frame NLL (sum of NLL / sum of frames), em_s13 is still selected (3.384405), the gap is 2.3294 and phi_c is still NOT BEYOND (-0.0489).
- Selection margin: em_s13 beats em_s01 by 0.0118 on average (pooled: 0.0087), and is better on 55.8% of paired utterances. The ordering is about 400 times the identity band, so it is not nondeterminism. It is still a thin lead, and the rule is a plain argmin.

## The six checks

1. **Verdict follows from the per-restart S values.** Yes. Every S, rate, range, band, margin, gap and reading above matches `selection.json` and `report.txt` to the printed precision.
2. **Nulls on the permuted holdout, restarts on the real holdout, same utterances.**
   - The null jsons have `shuffle_seed` 0. The real reads (restarts, phi_c, reruns, null_real) have None.
   - Every forward job reads the same real `BlankfreeVadHdfJob.SAjz8y1cT06g` shards and the same `GenMargSampleJob.b7aFZd9Tse5X` segment list.
   - I read both HDFs directly. For all 285 holdout utterances, the permuted HDF the nulls trained on (`PermutedUnitsHdfJob.FrvsC6NaeTJu`, seed 0, unit granularity) equals the real units reordered by `RandomState(0 ^ crc32(tag)).permutation(n)`. That is the permutation `_setup` applies in the forward. So the gated null holdout is exactly the nulls' own permuted holdout.
   - Frame counts and orig_length match per utterance across all 28 jsons.
3. **Identity band pairs the right runs.** Yes.
   - The `returnn.config` of em_s01_rerun and em_s02_rerun (pack kNy7Ggnl3nCt) differ from em_s01 and em_s02 (pack 9rwHpJQ4MXAD) only in the model output path.
   - The rerun keeps its original's seed: `random_seed` = `random_seed_offset` = 1 and 2 respectively, so the data order is the same (A8).
4. **The VOID rate is A8's emitted rate with A8's pooling.** Yes. It is `gendecode.json` `emitted_nonsil_rate.pooled_hz` = 50 x the non-SIL segment count of the tau = 1 Viterbi decode / the sum of original 50 Hz frames, pooled over the whole holdout. I reproduced it exactly from the segments. The expected rate appears only as report-only columns.
5. **Every arm ran the wave setting and sub-epoch count, and S was read at the final sub-epoch.** Yes.
   - All 24 arms: `num_epochs` 12, tau [4, then 1 x 11], 12 learning rates, 57 steps x 12 sub-epochs, a single start, 12 dev passes, no nan or inf in the logs.
   - The 16 restarts, 4 nulls and 2 reruns use `reverse_duration_prior_mode` "init" (durinit).
   - The 2 phi_c arms have no prior keys and load `ExtractSubmoduleCheckpointJob.rI5m4yzQACDD`. That matches A9 ("phi_c arms keep phi_c's own durations").
   - The nulls train on the permuted units. Everyone else trains on the real units.
   - All 28 scored jsons load epoch.012.pt at step 684.
   - durinit at 12 sub-epochs is what `PhiFirstA10ReadJob.DcfCsZNq1ucr/output/report.txt` states ("VERDICT: WAVE SETTING durinit, 12 sub-epochs (K* 12)"). I did not re-derive the A10 read itself.
6. **Checkpoint a downstream consumer takes.** `build_wave_bridge` builds `SelectedCheckpointJob` with key ("selected",) over `out_checkpoints[12]` of the 16 restarts. That resolves to `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.KCj5mptWgBqb/output/em_s13/models/epoch.012.pt` (sha256 12d9917e...c89d5, mtime 04:45 on 09-24, before the scoring). This is the same file the genmarg and decode reads loaded. No `SelectedCheckpointJob` exists on disk yet.

## Corrections and flags (none changes the verdict or the selection)

- **Wrong funding rule in the report.** `report.txt` quotes A7 as ending "at least one rt_r0 seed reading LIFT". It also prints "L2-2 may be funded if at least one rt_r0 seed reads LIFT". The registered A7 (line 65) says "LIFT or PARTIAL". The job's quote of A7 is out of date, and the funding sentence understates the rule.
- **No verdict check on the bridge.** `SelectedCheckpointJob` does not check the verdict, and `build_wave_bridge` does not check the rt_r0 half of A7's funding rule. That is harmless here because the verdict is SIGNAL. Whether A18 (a) replaces the rt_r0 condition is outside the line ranges I read.
- **What SIGNAL means.** The null is beaten by 2.33 nats per frame on every utterance. A7 itself records that "any segmental fit beats the within-utterance shuffle". SIGNAL therefore licenses spending on L2-2. It says nothing about phonetic content. The report-only reading points the other way: the selected random restart is 0.042 nats per frame worse than the phi_c-initialised restarts.
- **The four nulls share one permutation.** All four use shuffle seed 0 and differ only in init seed and data order. This matches the registered "the structure-destroyed corpus" (singular), but the null range contains no permutation variance. It does not matter here: the margin is the 0.01 floor, and the gap is about 230 times that.
- **Data frame not checked against the doc.** The L2-1 recipe text (line 391) says "one fixed sub-epoch ... repeated for 4 passes". What ran is the bed's stream at `partition_epoch` 4 (a full train-clean-100 pass every 4 sub-epochs). The code docstring attributes this to A3, which is outside my assigned line ranges. A10's "48 sub-epochs (12 passes)" is consistent with 4 sub-epochs per pass. The setup is the same for every arm, so the comparison stays fair.

## Files

- `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_genmarg_jobs/GenMargSelectionJob.dPjElzqvLUYt/output/{selection.json,report.txt}`
- The per-arm reads are the `ReturnnForwardJobV2.*` jobs listed in `selection.json` "inputs" (for example em_s13: i1PBNWq9yDsD genmarg, bLyIfnZ6zIwM gendecode; null_s02 permuted: uh4bSnceNS3m).
- Packs: 9rwHpJQ4MXAD, hgKo8LqPWFN4, 0XsleDRiS1DO, KCj5mptWgBqb (restarts), T5fmwEeVo3Wq (nulls), kNy7Ggnl3nCt (phi_c and reruns), under `.../emc/blankfree_pack_jobs/`.
- Null corpus: `.../emc/blankfree_phifirst_jobs/PermutedUnitsHdfJob.FrvsC6NaeTJu`.
- Recompute scripts (scratch, not part of the project): `/tmp/claude-34349/-e-project1-spell-wu24-2026-07-13-unsupervised/a2d26b34-6240-4bb2-b34b-ab87c54fe915/scratchpad/{recompute,rates,permcheck,cfgs,paired}.py`
