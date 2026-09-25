# Audit: A14 (ii) objective-floor read (reader A14ObjectiveFloorReadJob.PiYQ1OCFD4ot), 2026-09-24

Status: CONFIRMED_WITH_CORRECTIONS. The PHONETIC BASIN LOWER verdict holds as the rule was registered.
The corrections are to what the plan text says and to what the verdict licenses. No number changes.

Registration: SAE_4A_lexlat_v2.md, A14 (ii), lines 136-142 (real file
recipe/i6_experiments/users/wu/exp_logs/SAE/SAE_4A_lexlat_v2.md). S_g is the gold-init restart's S at
sub-epoch 48 on the 260 set. S_min is 3.2990. PHONETIC BASIN LOWER requires S_g < S_min - 0.01 AND a
gold-init Hungarian PER at 48 below 0.50.

## Re-derived numbers (independent of the reader: raw genmarg.json and report.json, same 260 tags)
- S_g = 3.21602: mean of per-utterance `nll_tau1_per_frame` over the 260 tags of
  CvDisjointSegmentsJob.PvgJ79Qc1Nro, taken from ReturnnForwardJobV2.c6fZOrfZj97b/output/genmarg.json.
  All 260 tags are possible. Frame-pooled, it is 3.21008.
- The six A10 restarts at 48 (ReturnnForwardJobV2.{EJe8yYQesSpG, xeXSsYZwOW9d, M9II4LBlD7rC, xjnwm8MDWHni,
  u9ya1iu8rxOY, Onpcw1Cj4DHr}) read uniform s1 3.34756, uniform s2 3.32244, durinit s1 3.29903,
  durinit s2 3.37032, durfrz s1 3.36665, durfrz s2 3.39956.
  - S_min = 3.29903 (durinit_s01). It equals the A10 diagnostics json
    (PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO, a13 block, 260 of 260 paired).
- S_g - S_min = -0.08301. Frame-pooled, it is -0.0844. The bar S_min - 0.01 = 3.28903 is met.
- Paired per utterance, gold@48 minus durinit_s01@48:
  - mean -0.0830, with speaker-clustered bootstrap 95% CI [-0.094, -0.072] (159 speakers, 2000 draws);
  - 222 of 260 utterances are lower.
  - Gold@48 is lower than every A10 restart, by 0.083 to 0.184.
- H = 0.35295, from GenDecodeReportJob.n2EIsGktK1Zy/output/report.json: per_hungarian sub 5648, del 4152,
  ins 679, over 29,690 reference phones, which is 10479/29690. This is below 0.50.
- Gold before any EM (ep0, JhiT3D0MyfPg) = 3.47351 on the 260 set. This matches the stated 3.4735.

## Check 1. Provenance: holds
- S_g comes from ReturnnForwardJobV2.c6fZOrfZj97b (name l21_a14_floor_gold_s01_ep48, dataset cv_holdout).
  Its phi_checkpoint is PhiFirstProbeTrainingJob.fTBdXD0SwBaA/output/models/epoch.048.pt, with alias
  sae/4a/lexlat_v2/em/a14/floor/floor_gold_s01/training.
- That training's resolved returnn.config has reverse_checkpoint_path =
  BlankfreeSupervisedReverseInitJob.16v7R6ztSq1u/output/models/epoch.008.pt.
  - Its alias is sae/4a/supervised_goldphi/phi_init.
  - Its gold_json is SeedGoldPhonesJob.zii9E9tvr51e, i.e. uncorrupted. r30, r70 and r100 use
    CorruptSeedGoldJob inputs and train as T2V8nn5obzj9, MU6Q3RanOQqF and lR4CfDiHyAvH.
- The engine log lists 16v7R6ztSq1u epoch.008 as an input. Step 0 of sub-epoch 1 reads
  blankfree_reverse_per_frame -4.019, against -6.770 for the random-init A10 durinit_s01
  (tDzmBsvX73X5). So a fitted phi was loaded.
- H comes from GenDecodeReportJob.n2EIsGktK1Zy, which reads ReturnnForwardJobV2.kEJ39HAHSTgS. That job
  decodes the same epoch.048 checkpoint on dev-other with 500 utterances.
- This closes the extractor's item 6, "training jobs not found": they are reachable through the forward
  configs.

## Check 2. S computed identically on both sides: holds
- The gold@48 and durinit_s01@48 forward returnn.config files differ only in phi_checkpoint and name.
  Everything else is identical:
  - the dataset (GenMargSampleJob.b7aFZd9Tse5X, 285 CV);
  - the prior (RtzbESkOedsT, trigram);
  - the eta table (U4etvcSpsQi4);
  - the lattice settings (d_min 2, D 25, D_sil 50, float64);
  - the null recognizer and shuffle_seed None.
- The reader's `_s` is an unweighted utterance mean of nll_tau1_per_frame over one common tag set, and the
  A10 diagnostics use the same convention. It reproduces the registered 3.29903 exactly.
- 3.29903 is the lowest of the six at sub-epoch 48 on the 260 set, and it belongs to durinit_s01.
- The genmarg code (blankfree_genmarg_jobs.py) was last changed 2026-09-23 15:58, before both forwards.

## Check 3. Recipe equality: holds, with one disclosed deviation missing from the plan text
- `diff` of the resolved training configs (fTBdXD0SwBaA against A10 durinit_s01 tDzmBsvX73X5) shows only
  two differences: `reverse_checkpoint_path` replaces the two keys `reverse_duration_prior` and
  `reverse_duration_prior_mode: init`, and the model path differs.
- Everything else is identical:
  - tau [4, 1 x 47] and 48 sub-epochs;
  - partition_epoch 4 on train.segments of CvHoldoutSplitJob.PfpCPQRCfIAk, 57 steps per sub-epoch;
  - lr 1e-4 x reverse multiplier 30 = 3e-3, clip 5, Adam;
  - random_seed 1, so the data order matches the S_min restart;
  - batch 88k, lam_agg, rate constants.
- The model code sae_blankfree.py (mtime 2026-09-23 20:17) predates both runs, and both runs use RETURNN
  00171dfe.dirty.
- CORRECTION 1: the plan text says "the A10 recipe verbatim (durinit, ...)". The run does not apply
  durinit.
  - The model asserts that durinit and a loaded phi exclude each other, so the gold restart keeps the
    gold fit's supervised durations.
  - This follows from "phi initialised from the L2-0 fits" and was disclosed in
    reports/impl_a14_wave_2026-09-24.md (concern 1). FLOOR_SEED = 1 is concern 2.
  - Neither is recorded in the A14 block.

## Check 4. PER as in A10's diagnostics: holds
- A10 durinit_s01@48 comes from GenDecodeReportJob.XVSb8nJ3v7Y7, which reads jo705Eo2sLqd.
- Its forward config differs from the gold@48 one (kEJ39HAHSTgS) only in phi_checkpoint and name.
  Both use:
  - the same 500-utterance dev-other sample (GenMargSampleJob.zXJjnNqU7kTa);
  - the same GoldPhonesJob.ZGSp0hxyd2YP and units hdf;
  - the same GenDecodeReportJob class, with no code change between the runs.
- The Hungarian map in n2EIsGktK1Zy's report.json is the identity on all 39 phones, with SIL -> DELETE.
  So direct PER equals Hungarian PER because the assignments are identical (the same S/D/I counts).
- This is meaningful: the S-preferred gold-init solution keeps the symbol-to-phone identity and is not a
  relabelled code.

## Check 5. Fairness and alternative explanations
- Held-out overlap: none that matters.
  - The 260 set is a subset of the EM CV (285), shares 0 utterances with the EM train stream (28,254), and
    shares 0 with the gold fit set (2,821).
  - One utterance of the 260 set sits in the gold fit's own 28-utterance CV. The fit takes its last epoch
    (8), with no selection on that CV, so this is negligible.
- Speakers: symmetric.
  - All 159 speakers of the 260 set are in the gold fit set (250 speakers).
  - They are also all in the EM train stream (251 speakers).
  - Both phi families use the same eta table.
- Supervised pre-fit, durations and rate (r100 control): r100 is fitted on length-preserving strings with
  random identities, then run through the same EM.
  - It ends at 3.37793, which is +0.079 above S_min, CI [+0.069, +0.089], 44/260 lower.
  - Its PER is 0.861, in the chance band.
  - So a supervised pre-fit together with gold token counts and durations does not get below S_min. The
    gap follows phone content.
  - The endpoint ordering follows init content: r30 3.2098, gold 3.2160, r70 3.2707 (Hungarian 0.508),
    A10 3.299-3.400, r100 3.378.
- Compute: the gold-init restart is below S_min - 0.01 by sub-epoch 3 (3.27946), against 48 sub-epochs
  for A10.
  - The A10 restarts still drift about 0.001 per sub-epoch at 48: durinit_s01 on 285 falls 3.3182 -> 3.3040
    from sub-epoch 36 to 48.
  - Closing 0.083 at that rate would take about 70 more sub-epochs. This is not measured, and neither
    side is converged.
- Size of the margin:
  - The registered margin is 0.01, A7's floor. The identity band from the A10 exact reruns is 3.4e-5.
  - The spread of the six random restarts is 0.1005 (range), with sd 0.036 and mean 3.351.
  - The seed spread within durinit is 0.071. The gap from S_min to the second-lowest restart is 0.023.
  - S_g sits 0.083 below the minimum, 3.7 sd below the mean of the six.
  - So the margin is of the same order as the seed-to-seed spread. It is not a multiple of it.
  - The gold-init restart has one seed, so its own seed spread is unmeasured.
  - With 6 draws, the lower tail of random-init S is not characterised.
  - Among the random restarts, lower S does not come with lower PER: the lowest, durinit_s01, has
    Hungarian 0.8615.

## Check 6. What the verdict licenses
- LICENSED: under the matched stage-1 objective, data, schedule and seed, an EM run started in the
  phonetic basin ends 0.083 nats/frame below the best of six random-init restarts. It still decodes at
  PER 0.353 with identity labels.
  - So S does not prefer the non-phonetic fixed points A10 found.
  - A10's failure is, at least in part, that random-init EM does not reach a lower-S phonetic region.
  - This supports the A16 (b) search class in NEXT over an objective change, as registered.
  - r70 (partial content, Hungarian 0.51) also ends below S_min - 0.01, so partial phonetic content
    already beats every random restart.
- NOT LICENSED (CORRECTION 2, to be carried with the verdict):
  - (a) That S's global minimum is phonetic. Only 6 random and 4 supervised starts were compared.
  - (b) That S rewards phonetic accuracy inside the basin. EM from gold lowers S by 0.257 (3.4735 ->
    3.2160) while PER rises 0.193 -> 0.353, and it rises steadily from sub-epoch 4 to 48. r30 ends
    below gold, 0.0062 lower (CI [0.0003, 0.0124]), with worse PER (0.394 against 0.353). So a search
    that reaches the basin should be expected to land near PER 0.35-0.39, not at gold's 0.19.
  - (c) That search moves (split, merge, reassign) from a random init can reach the basin. The basin was
    entered only by a supervised init.
  - (d) That longer random-init EM could not close the gap. Not measured.
  - (e) Anything about a second seed of the gold init.

Single strongest reason: the paired, same-code, same-config contrast gold@48 minus durinit_s01@48 is
-0.083 (CI [-0.094, -0.072]), and the content-free r100 control run through the identical recipe does
not get below S_min (+0.079). So the gap tracks phone content, not the pre-fit or the durations.
