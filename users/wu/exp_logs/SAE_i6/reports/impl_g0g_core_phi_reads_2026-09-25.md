# impl G0.G core phi reads (2026-09-25): fixes 1, 2, 4 done; fix 3 deferred

## Outcome

DONE_WITH_CONCERNS. Round 1 of this dispatch returned BLOCKED: `reverse_model/phi_first.py` is in
the live P0 import closure (`config/sae_i6_p0.py` -> `inputs.get_inputs()` line 123 ->
`phi_first.duration_prior_json`). The orchestrator chose option 2, so fix 3 (the wave defaults in
`phi_first.py`) is deferred until the trainings end. `phi_first.py` was not edited.

Fixes 1, 2 and 4 are implemented. Changes are confined to `reverse_model/genmarg.py`,
`reverse_model/ladder.py`, two new test files and the new setup-local `config/sae_i6_g0g.py`.
- The P0 graph is unchanged: all 164 job ids, the list of loaded package modules and the list of
  job classes are identical to the round-1 baseline.
- 92 CPU tests pass: the new tests plus the directly affected existing ones.
- Nothing was launched, submitted or committed, and `work/` was not touched.
- The concerns are the launch-time facts in "Exact launch" and the UNDETERMINED list. None has been
  measured on real data yet: no job ran.

## Files touched (package: `recipe/i6_experiments/users/wu/experiments/unsupervised_asr`)

- `reverse_model/genmarg.py` (+456 / -15)
  - `genmarg_reads` stays label-free. It still raises on `"dev-other"`, `gap` and `report`.
  - It also reads `CV_DISJOINT = "cv_holdout_disjoint"` through a new optional
    `disjoint_segments` argument. That read asserts no count.
  - `eval_dataset` takes an optional `stream`. For `"dev-other"` it swaps the three sub-datasets'
    `files` to the dev-other VAD stream and keeps every other option of the bed's dev dataset.
  - New LABEL-USING, REPORT-ONLY section:
    - pure helpers: `align` (same costs, ties and backtrace as `analysis.per.edit_counts`; checked
      against it), `corpus_per`, `hungarian_map`, `relabel`, `nmi_bits`, `segment_frame_labels`,
      `gold_frame_labels`, `duration_means`;
    - `UniformPhonePriorJob`;
    - `DevOtherPhoneReadJob`;
    - the builder `dev_other_reads`.
  - The module docstring carries the quarantine statement.
- `reverse_model/ladder.py` (+87 / -7): new `disjoint_holdout`, new
  `DisjointHoldoutSegmentsJob` (derives the set from the fit sets' segment files and writes
  `disjoint.segments`, `overlap.segments` and `counts.json`), and a changed `competence_reads`.
  - `competence_reads` now reads `CV_DISJOINT` by default, with `cv_holdout` (285) beside it.
  - A new `fit_segments` argument defaults to `[get_seed_inputs().seed_inputs["train_segments"]]`.
    Every ladder fit and the gold phi fit on that set.
  - A new `datasets` argument was added.
  - The frozen `config/lexlat_v2.ladder()` keeps working unchanged: it still indexes
    `jobs["cv_holdout"]`.
- `tests/test_reverse_genmarg_devother.py` (new, 305 lines, 16 tests).
- `tests/test_reverse_ladder_disjoint.py` (new, 113 lines, 4 tests).
- Setup dir: `config/sae_i6_g0g.py` (new). `py()` does the following:
  - gets the gold phi from `supervised_init.gold_phi(inputs, seed)["checkpoint"]`;
  - builds the bed from `phi_first.reads_bed(inputs.data)["bed"]` (no wave constant is read);
  - calls `dev_other_reads` with both priors and the frame-NMI inputs (the VAD raw-index HDFs and
    `get_mfa_alignments("dev")`);
  - sets the forward jobs' `rqmt["sbatch_args"] = ["-p", "gpu_32gb"]` (rqmt is not hashed);
  - registers outputs under `sae_i6/g0g/gold_phi/dev-other/...`.

## Import closure and job ids

- `genmarg.py` and `ladder.py` are not loaded by the P0 graph. The round-1 module list
  (`analysis_out/g0g_p0_graph_modules_2026-09-25.txt`) is identical after the change.
- P0 job ids: a read-only `sis console config/sae_i6_p0.py -s` after the change gave 164 jobs. The
  diff against `analysis_out/g0g_p0_jobids_before_2026-09-25.txt` is EMPTY; the after-change list is
  `analysis_out/g0g_p0_jobids_after_2026-09-25.txt`.
- The frozen ladder's 285 reads are unchanged. The 18 `cv_holdout` sample, marginal and decode job
  ids of `lexlat_v2.ladder()`, built from a `git archive HEAD` copy and from the edited tree, are
  identical. Files: `analysis_out/g0g_ladder285_jobids_{head,after}_2026-09-25.txt`.
- The g0g graph has 64 jobs (`analysis_out/g0g_entry_jobids_2026-09-25.txt`). 58 of them are P0
  jobs, including the gold phi training `ReturnnTrainingJob.Ac2eioZbRX7d`, so the gold phi is the
  P0 one. The 6 new jobs are:
  - `GenMargSampleJob.sMYB3BL8ppHJ` (D4)
  - trigram: `ReturnnForwardJobV2.RFaMyWRYUUQN`, `DevOtherPhoneReadJob.4eT8Gj5YOGWO`
  - uniform: `UniformPhonePriorJob.n7Yd9EbtVi13`, `ReturnnForwardJobV2.4hVXB1LWkw5p`,
    `DevOtherPhoneReadJob.Vk8VRJVfKaZd`
  - The two forward jobs have rqmt `{'gpu': 1, 'cpu': 4, 'mem': 32, 'time': 2, 'sbatch_args': ['-p', 'gpu_32gb']}`.
  - The two report jobs have rqmt cpu 2 / mem 16 / time 2.

## Tests (CPU, sae python, PYTHONPATH recipe:recipe/returnn:sisyphus)

- `analysis_out/g0g_tests_affected_2026-09-25.txt`: **92 passed, 0 failed, 0 skipped**. The run
  covered `test_reverse_genmarg.py`, `test_reverse_ladder.py`, `test_config_graph.py`,
  `test_reverse_genmarg_decode.py`, `test_reverse_phi_first.py` and the two new files.
- The new tests check the following against hand-computed oracles:
  - the D4 sample on a fixture: tags with empty gold are excluded; seed-0 order `[c, d, b]`;
  - `align` agrees with `edit_counts` on 300 random pairs;
  - the Hungarian map on two hand fixtures, including the case where the single DELETE slot goes to
    a non-SIL symbol;
  - permutation recovery gives Hungarian PER 0;
  - a gold-like decode gives identity map, direct = Hungarian = 0, and token NMI 1;
  - NMI is 1 when perfect, 0 when independent, and matches a hand 2x2 (I, H and NMI);
  - frame rasterisation;
  - E[d] of a zero-logit table: 13.5 for phones, 26 for SIL;
  - the uniform prior npz: -log 40 everywhere, rows sum to 1;
  - an end-to-end `DevOtherPhoneReadJob.run` on a fixture with an HDF, a parquet and a checkpoint;
  - the builder graph: dev-other files, the filter, `expected_utterances` 500, and a uniform decode
    whose only model-arg delta is `prior_npz_path`;
  - quarantine: `genmarg_reads` refuses dev-other, and `GenMargSelectionJob` refuses a dev-other
    decode record.
- The 260 set derived from the shipped id lists:
  - The package's `CvHoldoutSplitJob` runs on the four `train_clean_100_shard*_ids.txt` files
    (28,539 ids), giving 285 CV items.
  - It runs on `seed_10h_ids.txt` (2,849 ids), giving 2,821 train and 28 cv.
  - `DisjointHoldoutSegmentsJob` then gives an overlap of **25**: both the union and the
    intersection of the fit sets give 25.
  - It gives **260** disjoint items, with no seed-train item among them.

## UNDETERMINED (choices made, each needs the orchestrator's acceptance)

1. **Uniform prior.** I used a PhoneNgramPrior npz with every conditional equal to 1/40, at the
   bed's prior_weight of 1.0. Each token therefore pays log 40.
   - The alternative is prior_weight 0 (no prior term and no per-token cost). That would change the
     decode's token count.
   - The source's R2 definition ("under a uniform phone prior") does not say which.
   - `bed_from_train_config` asserts prior_weight == 1.0, which favours the npz reading.
2. **Hungarian convention.**
   - Source: the `private_code.hungarian_labelling` precedent
     (`SAE/reports/impl_private_code_2026-09-20.md`).
   - The RAW decode (SIL kept) is aligned to the SIL-free gold by identity-label edit distance.
   - A 40x40 `linear_sum_assignment(maximize)` runs over 40 symbols x (39 phones + one zero-gain
     DELETE column).
   - The mapped PER is re-scored with a fresh alignment, with no re-collapse.
   - Property (the A15-E weakness): for a decode with no correct labels, the edge SILs make shifted
     alignments tie, so the map can miss a true permutation. For the gold phi, matches anchor the
     alignment, and the map is expected to be the identity.
3. **Direct PER token convention.** The decode's segment tokens with SIL dropped, not run-collapsed:
   segments are real tokens. The job also reports `adjacent_repeats` and `direct_per_collapsed`, so
   any effect of a collapse is visible. Impossible utterances are excluded from the PER and are
   counted in `per_impossible_as_deletions`.
4. **NMI.**
   - Both variants are computed: token-level (aligned pairs of that identity alignment, 40 x 39) and
     frame-level (40 x 40, with SIL a gold label).
   - The source's A10 NMI was most likely token-level. Frozen `SAE_4A_lexlat_v2.md` line 551 says
     "GenDecodeReportJob aligns decoded tokens to gold by identity-label edit alignment before its
     Hungarian step, so its map and NMI depend on the labels".
   - The frame rasterisation convention is also my choice: MFA canonical phone at the retained
     frame's centre (raw_index + 0.5)/50 s, and SIL where no phone covers it.
5. **E[d].** The primary value is phi's duration-table mean (`duration_law` at `ReverseConfig()`):
   the mean over the 39 phones, and SIL separately. The decode's mean segment duration is reported
   beside it. Ref line 131 ("durfrz is held at 4.41") points to the table. The A9 start values
   (uniform 12.8, where the zero table gives 13.5; durinit 4.09 where the law gives 4.41) do not fit
   a pure table mean.
6. **The 260 set** = the holdout minus the UNION of the fit sets. The gate says "shared by every
   ladder phi", which is the intersection. With the single fit set the two are equal (25), and the
   job records both counts.
7. **Out-of-scope wording.** The brief puts "GenDecodeReportJob" out of scope, but fix 1 asks for
   direct and Hungarian PER, NMI and E[d]. So I wrote a new job, `DevOtherPhoneReadJob`, that
   covers exactly those reads. Statistic (b) and `sil_run_collapse` are not included.
8. **Resources.** The decode is routed to gpu_32gb. Its memory on a V100 is not measured.
   `FORWARD_TIME_H` = 2 h and `FORWARD_MEM_GB` = 32 GB are carried over from the port's CV reads,
   where the reference ran on a GH200.

## Exact launch the reviewer must check

From the setup dir, under the sae python, in a separate manager:
`PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH /work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis m config/sae_i6_g0g.py`

- Single intended delta: the 6 new jobs listed above, on top of the P0 gold phi
  (`ReturnnTrainingJob.Ac2eioZbRX7d`, epoch 8).
- **Concern:** the g0g graph contains that P0 training job. It was not in `squeue` at 2026-09-25
  (only four other trainings were running), and I did not look in `work/`.
  - If it has not finished, a g0g manager would submit or run it. That would be a second manager
    acting on a P0 job whose release the P0 launch order controls.
  - Launch g0g only once `Ac2eioZbRX7d` epoch 8 exists, or decide explicitly otherwise.
- Output: `sae_i6/g0g/gold_phi/dev-other/{trigram,uniform}/phone_read.{json,txt}` plus
  `gendecode.json`, and `.../dev-other/sample.json`.
- Gate read: trigram `hungarian.per` against 0.193 +- 0.02. Report only: `direct.per`, the uniform
  `hungarian.per` against 0.276, `token_nmi`, `frame_nmi` and `e_d_table`.

## Proposals (not implemented)

- After the unfreeze, `config/lexlat_v2.ladder()` should also register the `cv_holdout_disjoint`
  outputs and the split job's `counts.json`. The frozen config registers only the 285 reads.
- Fix 3 (the `phi_first.py` wave defaults) should be done after the four trainings end, with a P0
  job-id diff.
