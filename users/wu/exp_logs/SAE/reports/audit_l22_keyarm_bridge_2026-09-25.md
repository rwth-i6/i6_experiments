# Audit: A18 (c) L2-2 bridge read on the stage-2 key arm, G4a.L2.4 (dec_joint on rank1 against cold_ctl), 2026-09-25

**CONFIRMED_WITH_CORRECTIONS.** I re-derived every number the claim names from the per-utterance forward outputs and from the
dev-other posteriors, and all of them reproduce exactly. The registered read holds: G4a.L2.4 = LOWER -- OBJECTIVE ONLY. It
is not CODE BROKEN, because greedy PER on dev-other is at least 0.8401 for every arm at every kept epoch. The lift reading
(A4 bands at ep8) is NO LIFT for all five runs. The corrections are about the frame and the wording of the record. None of
them changes a number or a class.

This was a read-only audit: nothing was edited, launched or cleared. The scripts are in the session scratchpad
(`/tmp/claude-34349/-e-project1-spell-wu24-2026-07-13-unsupervised/a2d26b34-6240-4bb2-b34b-ab87c54fe915/scratchpad/{paired,phicmp,per,per_dj,per_all}.py`).
They read artifacts only and ran on the login node (CPU, bounded).

Inputs audited:
- Reader: `work/speech_llm/sae/emc/l22_bridge_read_jobs/BridgeReadJob.HWuOTb0Tefo1/output/{report.txt,bridge_read.json}`.
- Extraction: `reports/extract_l22_keyarm_bridge_2026-09-25.md`.
- Pack: `PackedBlankfreeTrainJob.SCtv4DjzFQ50`, SLURM 2007225 (engine file inside `finished.tar.gz`), node jpbo-117-05.
  It ran from 06:43 to 08:38 on 2026-09-25, and all four arms finished with rc 0.
- Baseline: `PackedBlankfreeTrainJob.UdhhxiGIMBob/output/cold_ctl`.
- Gate text: `SAE_4A_lexlat_v2.md` line 47 (2026-09-23 14:02), A6 at line 60 (2026-09-23 14:36), A18 (c) at lines
  336-339 (2026-09-24 11:04), and the build choices at lines 343-349 (2026-09-24 11:40); dates from `git blame`. All of
  them predate the result (reader finished 2026-09-25 08:42).

## 1. The paired delta, the interval and B

Sources: the `per_utterance.json` files of the five crosseval forwards under `work/i6_core/returnn/forward/`:
- dec_joint `FWISF5ppTqFT`;
- dec_distil `mDYEPqKA4F1Z`;
- dec_frz `SZUkgvvO0Et2`;
- dec_joint_s2 `s0QEgj3xC6CU`;
- cold_ctl `sVxuayfU6CYi`, which is the same forward that A18 (a) used, reused rather than rerun.

Each forward loads its run's `epoch.008.pt` for both theta and phi. A diff of the five `returnn.config` files shows only the
two checkpoint lines and the cell name differ. So every cell is read at one setting: tau 2.0, lam 1 / 0.1 / 3 / 1, max_active
1000, and HLG `cdcxYJMjiYj5`.

My recomputation takes L = l_tau + lexlat_k2 + 3 x rate per utterance, each term per retained frame. All 285 CV-holdout
utterances are kept in every cell. Retained frames are identical across cells. Speakers are clustered by the first field of
the utterance ID, which gives 164 speakers, the same as `s1a_job._clusters_by_speaker`.

| contrast vs cold_ctl | delta L | 95 % CI (seed 0, 2000) | CI (seed 7, 5000) | utts lower | frame-weighted delta |
|---|---|---|---|---|---|
| dec_joint | **-0.160859** | **[-0.170965, -0.148819]** | [-0.170907, -0.149893] | 272/285 | -0.1648 |
| dec_distil | -0.165360 | [-0.174913, -0.153480] | [-0.175255, -0.154696] | 274/285 | |
| dec_frz | -0.106334 | [-0.119648, -0.087861] | [-0.120019, -0.089207] | 262/285 | -0.1092 |
| dec_joint_s2 | -0.164667 | [-0.174264, -0.153435] | [-0.174241, -0.154361] | 272/285 | |

- B = |mean(L_dec_joint - L_dec_joint_s2)| = **0.003807**, so the margin is max(B, 0.01) = 0.01.
- The interval lies wholly below 0, and |delta| is about 16 times the margin. The class is **LOWER**. It is also LOWER
  under a second bootstrap seed and under frame weighting.
- Every value matches `bridge_read.json` to 6 decimals.
- The reader's identity check (each cell against its run's logged ep8 dev l_tau and lexlat_k2) prints diff +0.000000 for all
  cells.
- The registered gate row (line 47) put agg inside the total. A6 superseded that and made agg a point contrast only. With
  0.1 x agg included, the point contrast is -0.401, so the class would be the same either way.

## 2. Initial phi, frozen phi, and what differs from cold_ctl

**dec_joint's initial phi is rank1's epoch.048.**
- `SelectedCheckpointJob.PAV7erhRs8wq/output/selected.json` reads selected "rank1", with verdict KEY BASIN and candidates
  rank1-4. Its `model.pt` is a link to `PackedBlankfreeTrainJob.G0Vzzokj5PQC/output/rank1/models/epoch.048.pt`.
- The selection key is `KeyArmsReadJob.STcxhF0w4kpq` `comparison.best_arm`. The S values at 48 on the 260 set are rank1
  3.27275, rank2 3.29550, rank3 3.28529 and rank4 3.33540, so rank1 is the S-best arm, as A18 (c) requires.
- `ExtractSubmoduleCheckpointJob.r7tE5wxMIVzw` has prefix `reverse.`, 12 keys, epoch 48 and step 2736. It is the only input
  of the pack. I loaded both files: the extract equals rank1's reverse block with a max-abs difference of **0.0**. Against
  rank2-4 the difference is 4.4-5.0.
- All four arm configs set `reverse_checkpoint_path` to this extract.

**Phi across the kept epochs** (max-abs difference from the init phi):
- dec_frz: **0.0 at ep1, 2, 4 and 8**, so phi is unchanged.
- dec_distil: 0.0 at ep1, as its phi-frozen distil sub-epoch requires; then 0.19, 0.55 and 1.26.
- dec_joint: 0.021, 0.244, 0.610 and 1.247 at ep1, 2, 4 and 8.
- dec_joint_s2: 0.022, 0.245, 0.594 and 1.278.
- dec_frz's genmarg report at every kept epoch (0.8582 / 0.8613 / 0.0774) equals rank1's own sub-epoch-48 numbers in the
  key-arms read.

**Arms against each other** (config diffs):
- dec_frz adds `freeze_reverse: True`.
- dec_distil swaps in the `sae_blankfree_distil` train step (distil_subepochs 1, distil_tau 2.0).
- dec_joint_s2 uses flat init `DMSwTLXT9MWG`, random_seed 1 and a seed offset of 1000.
- Each difference is what it should be.

**cold_ctl**
- It is the **same job as A18 (a)'s**, `PackedBlankfreeTrainJob.UdhhxiGIMBob/output/cold_ctl`, run 2026-09-23 from 22:17.
  It is read through the same crosseval forward `sVxuayfU6CYi` and the same four greedy-PER jobs (`MRzJUdkdFw0i`,
  `X85IkX4ih0rj`, `RQR5XVWnbe8g`, `sQ3EqBbi53lJ`).
- A diff of its `returnn.config` against dec_joint's shows exactly one line besides the model path: dec_joint's
  `reverse_checkpoint_path`. The flat init `0J9d6wjrkRYH`, the seed, tau [2.0] x 8, lr [1e-5, 1e-4 x 7], the reverse lr
  multiplier of 30, the k2 block, and the prior, eta and HLG paths are all equal.
- Both runs used RETURNN `00171dfe.dirty`.
- No training-path file changed between the two runs: `definitions/sae_blankfree.py` and `lexlat_k2_train.py` were last
  modified 09-23 20:17, the `train_steps/sae_blankfree.py` step 09-23 15:07, `crosseval_jobs.py` 09-23 11:33 and
  `reverse.py` 09-15. `git status` shows these files clean. Only `sae_blankfree_distil.py` (09-24 11:43) is newer, and only
  dec_distil uses it.
- **cold_ctl does not carry durinit.** Its config sets no `reverse_duration_prior`, and `reverse.py:188` initialises
  `dur_logits` to zeros, which gives uniform durations.
- rank1's phi was built with durinit: `PhiFromKeyInitJob.86DqydG62mal` takes `BlankfreeDurationPriorMeanJob.ReQtJKYpZgsN`,
  and 48 sub-epochs of EM then trained the durations.
- So dec_joint and cold_ctl differ in phi's emissions **and** its durations. See correction 1.

## 3. Dev-other greedy PER, recomputed from the posteriors

I took the argmax of each forward's `posteriors.hdf`, collapsed repeats, dropped SIL, and ran my own Levenshtein against
`GoldPhonesJob.ZGSp0hxyd2YP` dev-other. That covers **all 2,864 utterances and 177,275 reference phones**. The symbol order
equals `prior.PHONES`.
- I traced each of the 20 PER jobs through its forward and its `recognizer.` extract (`extract.stats.txt`) to the right arm
  and epoch checkpoint.
- None of the ep8 or crosseval forwards ran on the faulty jpbo-028-30. They ran on jpbo-005-16, jpbo-012-33 and others.

| run | ep1 | ep2 | ep4 | ep8 | class at ep8 |
|---|---|---|---|---|---|
| dec_joint | 0.862237 | 0.852348 | 0.840141 | **0.842736** (149,396 errors) | NO LIFT |
| dec_distil | 0.857007 | 0.853454 | 0.844411 | 0.842860 | NO LIFT |
| dec_frz | 0.861249 | 0.856466 | 0.849494 | 0.846352 | NO LIFT |
| dec_joint_s2 | 0.862502 | 0.849691 | 0.843317 | 0.843063 | NO LIFT |
| cold_ctl | 0.888089 | 0.886374 | 0.855400 | **0.848698** (150,453 errors) | NO LIFT |

- All 20 values equal the jobs' `per.json` and the reader's table.
- The minimum over all arms and epochs is 0.8401 (dec_joint at ep4). So the claim "at least 0.84 at every kept epoch" holds.
  CODE BROKEN (< 0.50) misses by 0.34, and no arm reaches PARTIAL (< 0.8164).
- Reported beside only: the paired ep8 PER, dec_joint minus cold_ctl, is -0.0060 [-0.0097, -0.0023] (33 speakers, ratio
  of sums). A18 (a) found -0.0067 for em_s13. Both are about 0.006 in the chance band and carry no lift meaning.

## 4. Labels in selection or training

**None found.**
- The pack's training inputs are:
  - the train-stream feats, units and orig_length HDFs, with the CV split `CvHoldoutSplitJob.PfpCPQRCfIAk`;
  - the phone n-gram prior `RtzbESkOedsT`;
  - the speaker eta table;
  - the flat recognizer init;
  - the rank1 extract;
  - the HLG and the lexicon trie.
- No gold, alignment or transcript path appears in any arm's config. dec_distil's target is the stopped-gradient
  generative posterior, which is label-free.
- The epoch is fixed at ep8 by registration and not selected.
- The phi's lineage is label-free:
  - The key is stage 1's rank1, `cluster_centroid_s01_warm`: Ward clustering on unit centroids, deciphered and searched
    on train J, and selected by held-out J. The stage-1 audit confirms that KeySearchJob has no gold input.
  - The phi built from it (`PhiFromKeyInitJob.86DqydG62mal`) takes the unit HDFs, the durinit prior, the eta table and the
    key.
  - Stage 2 selected on held-out S only.
  - Gold enters only the dev-other PER and genPER reports.
- Standing, not new here:
  - The speaker IDs (eta) and the prior are unchanged campaign inputs under the Constraints.
  - The 260-utterance set is defined by the ladder phis' fit-set membership (A13). That uses no transcript content.

## 5. Corrections: what the record must carry, and what the claim may not say

1. **The baseline is the relaxed one, again.**
   - A9 (line 63) says "L2-2's cold_ctl baseline takes the wave's duration setting, so its single delta stays phi's
     emissions."
   - The reused cold_ctl has uniform duration logits. rank1 carries durations that were initialised by durinit and then
     trained by EM.
   - The A18 build choice (line 347, fixed before any result) re-classed this as part of phi's init, which A6 allows to
     differ. report.txt discloses it.
   - The record should call it "a pre-result A18 build choice relaxing A9", not "only phi's init differs, per A9/A6".
   - How much of the -0.161 the durations carry is unmeasured. Finding out would need a durinit cold_ctl at L2-2
     constants, which these files do not contain.
   - The relaxation cannot make the result CODE BROKEN, because the PER clause fails by 0.34 on its own.
   - This is the same correction A18 (a) received (`reports/audit_a18a_bridge_read_2026-09-24.md` §4.1).
2. **Record LOWER with its decomposition.** These are paired term contrasts against cold_ctl, scaled, with 95 % CIs:

   | arm | l_tau | lexlat_k2 | 3 x rate |
   |---|---|---|---|
   | dec_joint | -0.2836 [-0.2939, -0.2710] | +0.0807 [+0.0769, +0.0847] | +0.0421 [+0.0337, +0.0506] |
   | dec_frz | -0.2744 [-0.2865, -0.2574] | +0.0957 [+0.0916, +0.0999] | +0.0724 [+0.0637, +0.0815] |

   - The whole drop is l_tau. The word-lexicon term and the rate term both get worse in every arm.
   - The frozen rank1 phi alone gives 97 % of dec_joint's l_tau drop.
   - Joint minus frozen is -0.0545 [-0.0631, -0.0474].
   - Under A6, OBJECTIVE ONLY is "no claim". LOWER must not be read as a better fit to the lexicon or as content.
3. **The lift test's reading must be stated explicitly.**
   - A18 (c) says dec_joint carries A14 (i)'s registered lift test, but the reader prints no lift-test verdict line. Only
     the per-run A4 class appears.
   - The reading follows from A4's bands: dec_joint 0.8427 at ep8 reads NO LIFT, and dec_joint_s2 0.8431 agrees. So the
     S-best key-arm phi does not lift a random theta under this recipe.
   - This is one phi with two theta seeds, not A14 (i)'s four-phi pack. The key line's other three arms were not bridged.
4. **Scope.** The read licenses not funding further joint runs from this phi, and it closes A18 (c) as NO LIFT. It does not
   license the following:
   - "the key line cannot yield a lifting phi" (one S-best phi, and KEY BASIN rests on a 0.016 margin that is below the
     A10 seed spread);
   - "S or J are wrong";
   - any statement that the lower objective reflects phonetic content.
   - Descriptively, the key phi lowers the objective more than the A18 (a) wave phi did against the same cold_ctl cells
     (-0.161 against -0.116), while ep8 PER is unchanged (0.8427 against 0.8420). A lower objective in this bed does not
     track lift.
5. **Minor.**
   - The ">= 0.84" claim holds only just (0.8401). The class rule uses 0.50 and 0.8164, so this has no consequence.
   - B again comes from one theta-seed pair on the same phi. It covers theta-init noise only, as registered, and it does
     not matter at a margin of 16 times.

## Bottom line

The G4a.L2.4 read for A18 (c) is LOWER -- OBJECTIVE ONLY, and all five runs read NO LIFT at ep8. The numbers are exact.
The comparison is like-for-like: one setting, the same 285 utterances, the same cold_ctl cells as A18 (a), and a single
config delta. The one exception is the disclosed relaxation of A9's durinit on the baseline.
