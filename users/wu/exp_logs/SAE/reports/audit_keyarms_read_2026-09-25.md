# Audit: A16 (b) stage 2 key-arms read (KEY BASIN), 2026-09-25

Verdict: CONFIRMED_WITH_CORRECTIONS. KEY BASIN is what the registered rule gives, and I re-derived it from the
per-utterance files. The corrections are about reporting and scope. The verdict rests on S alone. All four key
arms decode in the chance band at sub-epoch 48, the same profile as the A10 random-init restarts. So KEY BASIN
here does not show that EM from the keys reached the phonetic basin that A14 (ii) described (PER 0.35-0.50 at 48).
Nothing was edited, launched or rerun. My check script is at
/tmp/claude-34349/-e-project1-spell-wu24-2026-07-13-unsupervised/a2d26b34-6240-4bb2-b34b-ab87c54fe915/scratchpad/recompute.py.

## 1. S at 48, recomputed from the per-utterance output
- Inputs, taken from the reader's `info` kwargs:
  - For each arm, the sub-epoch-48 CV-holdout genmarg.json: rank1 `ReturnnForwardJobV2.wS3wnqSYxAwx`, rank2
    `iyjKPo8bqQYQ`, rank3 `dNODJGfOwgAp` and rank4 `YtyzT64Xu5Ry`. Each json's `load` is
    `PackedBlankfreeTrainJob.G0Vzzokj5PQC/output/rankN/models/epoch.048.pt`, with epoch 48, step 2736 and
    strict loading.
  - The six A10 restarts at 48: `EJe8yYQesSpG`, `xeXSsYZwOW9d`, `M9II4LBlD7rC` (durinit_s01, from
    `PhiFirstProbeTrainingJob.tDzmBsvX73X5` epoch.048), `xjnwm8MDWHni`, `u9ya1iu8rxOY` and `Onpcw1Cj4DHr`.
  - The 260 set: `CvDisjointSegmentsJob.PvgJ79Qc1Nro/output/disjoint.segments` (260 unique tags).
- Checks:
  - All 10 jsons have identical `settings`: null recognizer, prior RtzbESkOedsT, float64, the blankfree lattice
    with d_min 2, and the real stream (shuffle_seed None).
  - The per-utterance frame counts are identical across all 10 jsons.
  - No 260-set tag is missing, impossible or non-finite in any json. The paired set therefore has 260 of 260 tags.
- S is the utterance mean of `nll_tau1_per_frame` over the 260 tags. This is A14 (ii)'s convention,
  `blankfree_a14_jobs._s`.

| arm | S at 48 (260) | pooled per frame (report only) | minus S_min, paired, speaker-clustered 95 % CI (159 speakers) | utterances lower |
|---|---|---|---|---|
| rank1 | 3.272750 | 3.26689 | -0.0263 [-0.0366, -0.0154] | 169/260 |
| rank2 | 3.295495 | 3.28999 | -0.0035 [-0.0140, +0.0069] | 134/260 |
| rank3 | 3.285286 | 3.27952 | -0.0137 [-0.0255, -0.0022] | 149/260 |
| rank4 | 3.335398 | 3.33381 | +0.0364 [+0.0263, +0.0466] | 85/260 |

- S_min is 3.299030 (durinit_s01), the lowest of the six A10 restarts at 48 on the same tags. The registered
  value in `PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO` is also 3.29903 over 260 tags.
- Best arm by S: rank1. Its key is `cluster_centroid_s01_warm`, with held-out J -4.5525.
- 3.27275 < 3.289, so the verdict is **KEY BASIN**. It also holds against the unrounded bar 3.28903, which the
  control reader uses. The keyarms reader uses the rounded 3.289, the registered text. No arm falls between the
  two bars, so the choice does not matter.
- The rule is registered. SAE_4A_lexlat_v2.md line 279 reads "KEY BASIN if the best key arm by S at 48 (260 set,
  paired as in A14 (ii)) has S < 3.289". 3.289 is S_min - 0.01, where 0.01 is A7's floor and S_min is the A14
  (ii) value.
- The pairing follows A14 (ii): one tag set, made of the tags possible in every compared json (the four arms and
  the six A10 restarts), as in `keyinit_read_jobs._paired`. Since all 260 tags are possible, any pairing variant
  gives the same numbers.
- An independent code path agrees to 5 decimals: `PhiFirstA10DiagnosticsDisjointJob.nyCwi94aepC0` (rank1..4
  s01).
- The ep48 forwards ran on jpbo-012-47, not on the faulty jpbo-028-30.

## 2. Recipe, checkpoint, coverage
- Config diff of each arm's returnn.config against A10 durinit_s01 (`tDzmBsvX73X5`):
  - The only difference is `reverse_duration_prior` + mode "init", which becomes
    `reverse_checkpoint_path = PhiFromKeyInitJob.{86DqydG62mal, AiESERYYJThQ, G2XQwRvYHjR8, 9G5tdIFuDzis}`.
    Beyond that, only the model dir differs, plus the unhashed torch_log_memory_usage.
  - Everything else is the same: random_seed 1, 48 sub-epochs, temperature schedule [4, 1, 1, ...], the Adam
    param groups, clip 5 and the data.
  - So the arms are the A10 recipe verbatim except for the init.
- The inits:
  - They read `KeySearchSelectJob.g9wsznNnqmyO` selected_1..4 in rank order: J -4.5525, -4.5647, -4.5671,
    -4.5755.
  - Their parameters equal those of the gold-key control's `f0jaGuiJVe6A`: smoothing 0.1, seed 0, preact 2.0
    and the same duration prior.
  - phi_from_key.json records E[d] = 4.4138 frames for every phone type, which is durinit.
- Tau: this is the unamended form. A17 (ii) read OBJECTIVE DRIFT, so no tau = 1 amendment applies, and the job is
  named "tau 4 then 1".
- Training:
  - The pack log reads "Finished training at epoch 48, global train step 2736" for all four arms, with 0 "nan"
    lines.
  - 48 checkpoints are kept per arm.
- The read is at checkpoint epoch 48, with all 260 utterances scored and paired.

## 3. Margins set against measured spread (context only; the rule is not rewritten)
- Margins below the bar: rank1 0.0163 and rank3 0.0037. Against S_min, rank1's paired CI lies entirely below
  -0.01. rank3's CI straddles -0.01, so rank3 cannot be told from the bar on sampling alone.
- The same recipe with only the seed changed (A10 seed pairs at 48, 260 set, paired; the seed moves both the
  init and the data order):
  - durinit: +0.0713 [+0.061, +0.081];
  - durfrz: +0.0329;
  - uniform: -0.0251;
  - range over the six restarts 0.1005, sd 0.036.
  rank1's 0.016 margin is below every one of these. Each key arm has a single seed. The spread under a data-order
  change alone, with a fixed key init, is unmeasured.
- Spread across the key arms (same seed, different key): range 0.063, sd 0.027. The key-arm mean is 3.2972,
  above the bar by 0.008 and 0.002 below S_min. The six A10 restarts average 3.3509.
- Max-of-4 selection:
  - rank1 sits about 0.9 sd below the key-arm mean, which is what the minimum of 4 draws gives.
  - The family mean does not clear the bar; the best arm does.
  - The bar is itself a minimum over 6 draws, so the extreme-against-extreme comparison does not favour KEY BASIN
    by construction.
  - Selecting and testing on the same 260 set adds little: the paired sampling SE is about 0.005, and rank3 -
    rank1 is +0.0125 [+0.003, +0.022].
- Jitter between checkpoints: over sub-epochs 40-48, S ranges 0.0065 (rank1) and 0.0060 (rank3).
  - rank1 is below 3.289 at sub-epoch 28 and at every sub-epoch from 30 to 48.
  - rank3 is below it only at 40, 45 and 48, so its below-bar status at 48 depends on the checkpoint.
- Reference points on the same 260 set: the basin arms end at 3.207-3.271. These are gold_key 3.207, G-dur 3.224,
  r30-dur 3.212, r70-dur 3.266, and A14 (ii) gold 3.216, r30 3.210, r70 3.271. rank1's 3.2728 sits at r70's
  level in S.

## 4. Generative PER at 48 (D4 dev-other, 500 utterances, 0 impossible)
- Sources:
  - `GenDecodeReportJob` report.json: rank1 `O2rjYiJs4lH7`, rank2 `j1xUDxnQYyIc`, rank3 `A4MLmX9FAdid`, rank4
    `nu04h62DdFzX`. Each decode loads the rank's epoch.048.pt.
  - The same values are printed in KeyArmsReadJob report.txt and in nyCwi94aepC0 report.txt.

| arm | direct PER | Hungarian PER | NMI(symbol, phone) | identity hits, ep0 -> ep48 |
|---|---|---|---|---|
| rank1 | 0.8582 | 0.8613 | 0.0774 | 5062 -> 4486 |
| rank2 | 0.8469 | 0.7975 (BELOW the band) | 0.0986 | 5650 -> 4886 |
| rank3 | 0.8490 | 0.8416 | 0.0962 | 5111 -> 4884 |
| rank4 | 0.8715 | 0.8734 | 0.0677 | 4897 -> 4402 |

- Identity hits = reference phones (29690) - substitutions - deletions, derived by me from the printed counts.
- For comparison at 48:
  - A10 restarts: direct 0.832-0.856, Hungarian 0.838-0.862, NMI 0.079-0.113.
  - Basin arms: gold_key 0.346 / 0.391 / 0.695, r70-dur 0.442 / 0.466 / 0.606, A14 r70 0.495.
- E[d] at 48: phones 5.66-5.91 and SIL 6.7-8.2, against A10 durinit's 6.0-6.2 and 6.6-7.3.
- Direct PER falls from ep0 to ep48 only because insertions drop. Identity-label hits fall on all four arms.
- Caveat (A15): these decode-based measures depend on the labels. A phi that is phonetic up to renaming can still
  read chance here (permphi read Hungarian 0.819). So they cannot rule out mislabelled content. AN-5, the A15-F
  measures that the registration names and no job has yet computed, owns that question.

## 5. Operating point; what KEY BASIN licenses
- Operating point: phi alone under the null recognizer, sub-epoch 48, seed 1, tau 4 then 1, durinit. S is read
  on the 260 disjoint CV-holdout utterances. One seed per key.
- Licensed:
  - The S-best key arm (rank1) ends 0.026 below every A10 random-init restart in held-out S [-0.037, -0.015],
    beyond the 0.01 floor.
  - The registered consequence: rank1 goes to L2-2, whose dec_joint carries the lift test. A18 (c) sends it there
    whatever KEY BASIN reads, so the verdict changes no funding.
- Not licensed:
  - That the key arms are in the phonetic basin of A14 (ii). That read also required Hungarian PER < 0.50; these
    arms read 0.80-0.87. The S-only rule, by construction, answers "lower than random-init EM by 0.01", not
    "phonetic".
  - Anything about names or renaming (AN-5).
  - Any lift (the L2-2 read).
  - Robustness to the seed: the margin is below the same-recipe seed spread.
  - A per-arm KEY BASIN. The registered verdict concerns the best arm only. AN-5's precedence clause, "KEY BASIN
    (S at 48 < 3.289) on the same arm", applied per arm, holds for rank1 and rank3. rank3's 0.0037 margin is
    within checkpoint jitter and within its CI.

## Corrections to the extraction and the claim
1. log.run.1 exists. It is inside `KeyArmsReadJob.STcxhF0w4kpq/finished.tar.gz`, because the job was cleaned. It
   contains no warning, error or traceback lines, and its printed text equals report.txt.
2. The claim "phi EM from the found keys reaches the phonetic basin" goes beyond the read. What KEY BASIN shows
   is lower S than any A10 restart. genPER at 48 puts every key arm in the chance band, indistinguishable from
   A10.
3. The A15-F measures that the registration lists under "reported" are not computed for the key arms (the review
   finding, owed by AN-5).
