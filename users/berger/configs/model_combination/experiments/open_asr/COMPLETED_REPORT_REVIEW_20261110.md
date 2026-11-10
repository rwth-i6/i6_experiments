# Completed model-combination report review — 2026-11-10

## Evidence and scope

Protocol `f4212424889ee5d3`: exact configurations selected jointly on LibriSpeech dev-other and Loquacious dev.short. This review contains 232 of 234 expected family/dataset cells (18 families, 13 corpora or subsets). Only SPGISpeech Parakeet + Speech-Qwen 1.7B one-pass and Parakeet + Speech-Qwen 0.6B two-pass remain unavailable. Parents and overlapping Loquacious subsets are never pooled. No new inference was run.

The original controlled-comparison report is a frozen pre-evaluation snapshot; its “323 missing” is historical, not the current missing count. The selected-evaluation overview still waits for all dependencies. This review combines its completed per-condition score artifacts with exact reused rows in the original snapshot. The full 232-row table is `completed-selected-results.tsv`, with score and segment paths.

For the 1.7B comparisons below, raw-reference file hashes, normalizer revisions, scoring-unit IDs and aggregate edit counts were checked. Paired intervals use the existing identifier-derived cluster policy, 10,000 replicates and seed 20261103, retaining hypothesis-dependent compound-normalization denominators. They are exploratory, unadjusted 95% intervals, not confirmatory evidence on untouched data. The standalone minimum is recomputed in each bootstrap replicate. Audio provenance was additionally checked for the recomputed complementarity pairs.

## Frozen 1.7B conditions

WER (%). G = Granite CTC; P = Parakeet RNN-T; S = standalone Speech-Qwen 1.7B. Fusion columns use the same Speech-Qwen checkpoint. “1” and “2” mean one-pass and two-pass. These are frozen development choices, not per-corpus winners.

| Dataset | G control | P control | S | G+S (1) | P+S (1) | P+S (2) |
|---|---:|---:|---:|---:|---:|---:|
| ami-cleaned | 7.363 | 13.295 | 8.367 | 7.062 | 10.699 | 10.182 |
| earnings22-cleaned | 8.471 | 8.882 | 5.652 | 7.556 | 7.483 | 7.952 |
| gigaspeech-cleaned | 8.523 | 8.169 | 7.289 | 7.928 | 7.526 | 7.449 |
| librispeech-test-clean | 1.457 | 1.089 | 1.213 | 1.343 | 0.993 | 0.989 |
| librispeech-test-other | 2.596 | 2.099 | 2.862 | 2.360 | 1.962 | 1.987 |
| loquacious-test | 4.936 | 4.353 | 4.973 | 4.561 | 4.163 | 4.191 |
| loquacious-test.commonvoice | 6.737 | 5.713 | 6.633 | 6.041 | 5.455 | 5.388 |
| loquacious-test.librispeech | 2.581 | 2.098 | 2.846 | 2.340 | 1.956 | 1.984 |
| loquacious-test.voxpopuli | 5.637 | 5.018 | 5.804 | 5.430 | 5.070 | 5.179 |
| loquacious-test.yodas | 7.818 | 8.053 | 6.743 | 7.042 | 6.913 | 6.869 |
| monsoon-en-in | 4.286 | 4.421 | 3.425 | 3.355 | 3.279 | 3.254 |
| spgispeech | 1.841 | 2.679 | 2.569 | 1.691 | pending | 2.537 |
| voxpopuli-cleaned | 4.635 | 3.776 | 2.852 | 3.741 | 3.105 | 3.004 |

The P control is the dev-selected `zero-beam32` AM-only search. Correction after checking the paired native artifacts: native Parakeet on AMI is 14.339%, not 9.426%; 9.426% is Speech-Qwen 0.6B. The beam control improves to 13.295%. The 1.7B one-pass/two-pass fusions improve further to 10.699%/10.182%, but remain worse than standalone Speech-Qwen 1.7B at 8.367%. Granite + Speech-Qwen reaches 7.062%, close to fixed native Cohere at 7.035%. There is no aggregate AMI beam-search degradation demonstrated by these results.

## What has changed in the conclusions

1. Speech-Qwen 1.7B generally improves fusion over 0.6B and is usually stronger than text-Qwen at the same AM. For Parakeet/test-other: text 0.6B 2.142%, text 1.7B 2.129%, Speech 0.6B 1.980%, Speech 1.7B 1.962%, against AM-only 2.099%. Granite/test-other: text 1.7B 2.524%, Speech 1.7B 2.360%, against 2.596%. These are selected-system comparisons, not a parameter-count-only causal ablation.

2. Fusion can improve on a stronger Speech-Qwen component. Monsoon: Parakeet 4.421%, Speech-Qwen 3.425%, one-pass 3.279%, two-pass 3.254%. The two-pass gain is 0.171 points / about 5.0% relative to the better constituent. This contradicts the strict version of “AM-led fusion only helps when the AM is already stronger.” It does not establish that the AM proposal bottleneck is absent.

3. Stronger Speech-Qwen exposes clear failures too. On Earnings22, 5.652% standalone is much better than 7.483% one-pass or 7.952% two-pass. On GigaSpeech, standalone 7.289% beats 7.526% and 7.449%. VoxPopuli is closer: 2.852% versus 3.105% and 3.004%. Loquacious Yodas is also worse with fusion than standalone Speech-Qwen, whereas CommonVoice and LibriSpeech benefit. The Loquacious VoxPopuli subset regresses slightly against its already stronger Parakeet control.

4. Two-pass is competitive and cheaper at the measured operating points. Test-other: one-pass 1.962% / RTF 1.293 versus two-pass 1.987% / RTF 0.504. Loquacious test: 4.163% / 1.142 versus 4.191% / 0.551. Monsoon: 3.279% / 1.341 versus 3.254% / 0.695. One-pass is not uniformly more accurate; two-pass is better on GigaSpeech, VoxPopuli, AMI and CommonVoice. RTFs include the recorded generation/scoring work; historical runs are not hardware-matched, so these are operating-point observations, not controlled speedup claims. The dev RTF constraint does not guarantee evaluation RTF <1.5: 1.7B one-pass reaches 2.429 on Earnings22 and 1.590 on VoxPopuli.

## Paired uncertainty: Parakeet + Speech-Qwen 1.7B

ΔWER against the better frozen standalone constituent; negative is better. Earnings22 has only six independent recordings and AMI 16 meetings, so their intervals are especially fragile.

| Dataset | One-pass Δ [95% CI] | Two-pass Δ [95% CI] |
|---|---:|---:|
| librispeech-test-clean | -0.096 [-0.138, -0.055] | -0.100 [-0.147, -0.055] |
| librispeech-test-other | -0.136 [-0.199, -0.073] | -0.112 [-0.185, -0.038] |
| loquacious-test | -0.190 [-0.261, -0.121] | -0.162 [-0.233, -0.092] |
| monsoon-en-in | -0.146 [-0.294, -0.001] | -0.171 [-0.316, -0.023] |
| gigaspeech-cleaned | +0.237 [+0.146, +0.335] | +0.160 [+0.082, +0.244] |
| voxpopuli-cleaned | +0.253 [+0.037, +0.473] | +0.151 [-0.064, +0.368] |
| earnings22-cleaned | +1.831 [+0.929, +2.826] | +2.299 [+1.201, +3.515] |
| ami-cleaned | +2.331 [+1.901, +2.728] | +1.814 [+1.396, +2.196] |

## Complementarity and ranking

The exact dev-selected Parakeet/Speech-Qwen 1.7B standalone-selection oracle is 1.533% on dev-other (better standalone 2.036%) and 3.236% on dev.short (better standalone 3.951%). Current one-pass fusion reaches 1.897% / 3.795%; there is still complementary information. This reference-dependent per-scoring-unit selector is neither deployable nor a lower bound on fusion. In particular, zero recording-selector headroom on Earnings22 does not mean there are no complementary word errors.

For the current 50-best Speech-Qwen 1.7B rescoring protocols:

| AM / dev set | First pass | Selected | Pool oracle | Coverage gaps vs Qwen | Ranking gaps vs Qwen | Pool-correct units missed |
|---|---:|---:|---:|---:|---:|---:|
| Granite CTC / librispeech-dev-other | 2.631 | 2.491 | 1.337 | 170 | 151 | 345 |
| Parakeet RNN-T / librispeech-dev-other | 2.071 | 1.935 | 0.795 | 34 | 93 | 368 |
| Granite CTC / loquacious-dev.short | 4.705 | 4.263 | 2.478 | 145 | 195 | 434 |
| Parakeet RNN-T / loquacious-dev.short | 3.989 | 3.811 | 1.964 | 44 | 142 | 439 |

Coverage gap = even the best AM-pool candidate has more edits than standalone Qwen. Ranking gap = the pool can match/beat Qwen but the selected candidate does not. These count scoring units, not word errors, and are measured against Qwen, not every possible correct transcript. Parakeet has 93 versus 34 ranking/coverage gaps on dev-other and 142 versus 44 on dev.short. Thus changing search direction is not the only plausible next step. Granite has a more substantial coverage limitation (170/145 coverage gaps).

Pure Speech-Qwen ranking of Parakeet candidates gives 2.213% / 4.255%, worse than combined rescoring 1.935% / 3.811%. Raising the Speech-Qwen weight globally therefore cannot be assumed to solve the ranking gap. Word reward also overturns some better auxiliary choices (34/84 cases). Parakeet pools contain only 8.7 / 11.6 distinct text candidates per utterance on average despite an advertised 50-best cap, versus 36.1 / 34.9 for Granite. Larger N alone is not evidence of useful diversity.

Concrete coverage example: dev-other `3660-172183-0026`: standalone Speech-Qwen gets GERAINT, while the selected Parakeet candidate and pool oracle have GERANE. Concrete ranking example: dev.short `20180704-0900-PLENARY-17-en_20180704-15:50:40_2`: standalone Speech-Qwen and the pool oracle say “institutional framework”; selection adds “Greek institutional framework”. The reference says “interinstitutional framework”, so even the pool oracle is imperfect, but its error count is lower. Its first-pass cost is 12.739 versus 9.214 for selection; Speech NLL 50.470 versus 56.291 and the -1/word reward does not compensate under the frozen scale 0.3.

## Rare words and duration

“Unseen” here means absent from the LibriSpeech LM-training text, not absent from pretrained-model training or necessarily OOV for any decoder. On the shared normalized reference grid:

| Dev set | AM | Unseen accuracy alone | + Speech-Qwen 1.7B one-pass | Common (>100) accuracy alone → fusion |
|---|---|---:|---:|---:|
| librispeech-dev-other | Granite CTC | 35.26 | 43.16 | 98.52 → 98.51 |
| librispeech-dev-other | Parakeet RNN-T | 44.74 | 47.63 | 99.01 → 99.10 |
| loquacious-dev.short | Granite CTC | 67.59 | 72.74 | 97.16 → 97.34 |
| loquacious-dev.short | Parakeet RNN-T | 72.91 | 74.18 | 97.81 → 97.91 |

Rare/unseen words therefore show larger percentage-point accuracy gains than the aggregate WER suggests, especially for Granite. This is exploratory and the dev-other unseen bin contains only 380 reference words; dev.short contains 1,185. Neighboring common-word accuracy does not consistently improve: for Parakeet/dev-other it moves 97.08% → 96.93%. Evaluate surrounding errors as well as target-word recall.

Duration bins do not yet show a persuasive long-form trend. Most development segments are shorter than 20 seconds; each dev set has only seven 30–60-second units. Grouped scoring over input chunks does not test unsegmented long-form recognition. Proper names are not labeled independently, and frequency is not a proper-name label.

## Reporting defect found and corrected

`analyze_pair()` returned the final loop variable `stats` for every fusion label, rather than `fusion_stats[label]`. WERs, raw segments, standalone-pair complementarity/oracles, candidate-ranking reports and stratified reports are unaffected. Multi-fusion unit and word correction/regression counters in the old complementarity summaries must be regenerated. A test with two distinct fusions fails before the fix and passes afterwards; the focused complementarity/diagnostics suite passes all 14 tests. The correction is only in the isolated `berger_dev` worktree; the live runtime remains untouched. At branch cutover the source-versioned CPU analysis jobs will invalidate their old outputs without rerunning ASR inference.

`corrected-current-complementarity.json` separately recomputes both 1.7B fusion methods on seven completed evaluation corpora, with reference/audio provenance checks. It confirms simultaneous corrections and regressions: Parakeet two-pass on GigaSpeech corrects 2,355 reference-word errors from Speech-Qwen but introduces 3,032; on Monsoon it corrects 409 and introduces 338. These word counts exclude units with unequal normalized references and exclude insertions; the JSON records both exclusions and insertion changes.

## Recommended next work, in order

1. Run a small calibration study on the existing development candidate pools, with a separate validation partition. Focus on EOS/length terms, confidence/disagreement-based score balance, and retained first-pass LM costs where present. Compare to the frozen global scale/reward baseline. Do not fit using test labels or pretend the reference oracle is a usable ranking rule. Pure Speech-Qwen ranking is a required negative control.

2. Finish the comparison against native and frozen beam controls, and audit the report interpretations. The initial AMI degradation claim was a reading error, not an experimental finding; no extra AMI search experiment is motivated by it.

3. Pilot a consistently scored union pool. Prioritize Granite, where coverage gaps are larger, and inspect representative Qwen-strong corpora before committing to Speech-LLM-led search. Every union candidate needs a comparable AM and auxiliary score; do not use a zero or missing AM cost for Qwen-generated hypotheses. If better candidates win after consistent scoring, prioritize proposal/search direction. If they lose, prioritize calibration/objective changes.

4. Proceed with the small controlled-noise pilot and company-side AppTek preparation. Keep common segmentation and settings fixed initially. Reserve natural long-form claims for real long-input experiments. No extra large model sweep or three-model fusion is justified by this review alone.

For a paper the defensible result is conditional benefit and failure modes, with competitive two-pass/cost controls—not universal superiority of one-pass fusion. The last two SPGISpeech cells need not block these pilots.

## Follow-up: AMI native/beam error inspection

The two scores use the same raw reference file and all 7,715 IDs match. Native Parakeet has I/D/S = 950/6,557/4,288; the beam control has 1,298/5,033/4,605. Beam search removes 1,524 deletions while adding 348 insertions and 317 substitutions: 859 fewer errors overall (14.339% → 13.295%). It improves 1,001 units and worsens 481; 6,233 tie in edit count. Native-correct/beam-wrong units number 157; the reverse numbers 324.

Newly introduced errors include dropping the repeated “I AM” in `AMI_EN2002a_H02_FEO072_0084151_0084417` and replacing “WHAT WE ARE DOING IS FINE” with “WE ARE DOING SO” in `AMI_EN2002a_H03_MEE071_0064862_0064969`. These regressions coexist with a larger aggregate reduction in deletions. They do not establish a broken search path or an overall beam-search regression.

The 9.426% overview value was the minimum of the Parakeet and Speech-Qwen 0.6B scores, not the Parakeet score. The source snapshot and current score/segment artifacts agree; this was a reading mistake, not changed evidence. Calibration, union-pool, noise and long-form studies remain optional follow-ups. None is required merely to complete the present descriptive comparison; mechanistic claims about proposal bottlenecks would require additional evidence.
