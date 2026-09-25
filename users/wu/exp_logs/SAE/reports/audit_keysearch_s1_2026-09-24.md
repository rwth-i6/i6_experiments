# Audit: A16 (b) stage 1 key search. Did J lead the search toward the gold key? (2026-09-24)

**Verdict against the extraction (`reports/extract_keysearch_s1_2026-09-24.md`): CONFIRMED_WITH_CORRECTIONS.**
Its numbers are right. One claim is circular, and one sentence is wrong and then corrected in place (listed at the end).

**Answer to the question.** The highest-J keys the search found lie away from gold. All 124 label-free unit-level final keys have
held-out J above J(gold) = -4.8180, by 0.070 to 0.266 nats/frame. Their frame-weighted identity agreement with gold is 0.02-0.21,
and their many-to-one agreement is 0.45-0.58. Every final key beats gold on the emission term (+0.08 to +0.26), and the emission
term does not depend on the symbol names. On the LM term the final keys tie gold (mean -0.002; 65 of 124 score below gold).
This is "clusters with wrong names" again, now with J above gold.

Sources (paths under `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/`):
- S = `key_search_jobs/KeySearchJob.AzM1NoHpOnFJ/output/search.json` and `search_progress.log`.
- SEL = `key_search_jobs/KeySearchSelectJob.g9wsznNnqmyO/output/{selection.json,table.txt,selected_1..4.json}`.
- AG1 = `key_search_jobs/KeyAgreementReportJob.Xk3Wxu9kJ38t/output/table.{txt,json}`. The alias `agreement/table.txt` points here.
- F0 = `unit_key_jobs/KeyFloorReadJob.DgDbciHlq2wY/output/table.{txt,json}`.
- AG0 = `key_search_jobs/KeyAgreementReportJob.HZxcl9qgT1im/output/table.txt`. Its stage-0 rows are identical to the same rows in AG1.
- Code: `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/{key_search_jobs.py,unit_key.py}`. Config:
  `.../configs/config_sae_4a_lexlat_v2_keysearch_s1_v1.py`.
- AUDITOR-SIDE numbers are marked [ad hoc]. They are read-only recomputations from the keys and the train-side unit frame counts
  (`unit_key.unit_frame_counts`, 15,275,716 frames, the same as the AG1 header). No registered reader produced them, so they are
  not for citation in the project documents. They inform only whether a reader is worth commissioning.

## 1. Validity
- Status counts (S `status_counts`, repeated in SEL): real 196 ok, 0 void_on_rate, 0 failed. Null 172 ok, 8 void_on_rate, 0 failed.
  `loop_error` is None, `failed_runs` is empty, and the progress log contains no error or traceback line. Job wall time was 13,025 s.
- The 8 VOID-ON-RATE runs are all null runs, `decipher_context_s01..s08`. They stopped on the stall rule after 100-136 proposals,
  at 5.671-5.685 Hz. The context-cluster null therefore contributes no unit-level runs.
- 15 runs were TRUNCATED, all on the null corpus: 9 `cluster_centroid` runs and 6 `relabel` runs. They completed 19-21 sweeps
  in 5,389-5,748 s of wall time.
  - No real run was truncated or repaired. All 124 real unit-level runs report `converged` True.
  - The truncation touches only the null's report-only J rise.
- Selection is by held-out J only and uses no labels:
  - `KeySearchSelectJob.select` sorts the eligible real unit-level finals by `-J_final.heldout.J`. Eligible means status ok and
    held-out rate in band. The selection removes duplicate keys.
  - The select job's only input is S. Its selection report lists 124 candidates, 124 eligible, no duplicates and no void keys.
  - KeySearchJob's inputs contain no gold key and no alignment. Its inputs are the unit HDFs, the segment lists, the prior, the
    duration prior, the 7 A10/A11 argmax keys and the codebook centroids. The search.json field `labels` holds the Ward cluster
    labels of the 500 units, not transcripts.
  - Gold enters only the report-only AG1.
- Gold J on the same held-out set: **the extraction's "reproduction" is circular.**
  - The AG1 gold row (-4.8180) is read from F0's table.json, which is one of the `key_tables` inputs. The stage-1 code never scored
    gold.
  - Comparability holds anyway. Both stages call `unit_key.corpus_counts` and `unit_key.j_terms(heldout=True)` on the same held-out
    file (`CvDisjointSegmentsJob.PvgJ79Qc1Nro/output/disjoint.segments`, 260 utterances, 137,933 frames) and the same train
    segments, prior, duration prior and unit HDFs.
  - The stage-1 job's own J_start for the 27 keys the two stages share (20 random keys and 7 argmax keys; identical key arrays)
    matches F0 exactly. The maximum absolute difference is 0 over J, lm, emis, dur and rate, on both the train and the held-out side.
  - The per-frame denominator is the same for every key (137,933 held-out frames).
- The selected keys are inside the rate band [5.80, 14.49] Hz. Held-out rates are 9.195, 9.678, 9.281 and 9.494 Hz; train-side
  rates are 9.161, 9.649, 9.244 and 9.450 Hz. Gold's held-out rate is 9.470 Hz.
- Frame: every constant (64 random, 7 argmax, 16 cluster and 7 relabel starts; 20 annealing sweeps down to T0/1000; ICM of at most
  50 sweeps; warm start at T0/100; top 4; the A2 band; the A13 held-out set) is registered in SAE_4A_lexlat_v2.md lines 229-270
  or in the module's STAGE-1 DECISIONS, and each was fixed before any result.
  - The held-out set is also the set on which the four keys were selected. Winner's curse is negligible here: train-side J is
    within 0.001-0.008 of held-out J for the top 4.

## 2. Held-out J (sources: S, SEL, F0)
| key | J_ho | lm | emis | dur |
|---|---|---|---|---|
| sel 1 cluster_centroid_s01_warm | -4.5525 | -0.9484 | -3.1290 | -0.4750 |
| sel 2 cluster_centroid_s04_warm | -4.5647 | -0.9483 | -3.1364 | -0.4800 |
| sel 3 cluster_context_s01_warm | -4.5671 | -0.9457 | -3.1450 | -0.4764 |
| sel 4 cluster_context_s04_warm | -4.5755 | -0.9754 | -3.1223 | -0.4779 |
| gold | -4.8180 | -0.9897 | -3.3468 | -0.4815 |
| best A10 argmax key (a10_durfrz_s01) | -4.9714 | -1.2283 | -3.2667 | -0.4763 |
| K30 mean over 5 seeds | -5.6097 | | | |

- Advantage of each selected key over gold:
  - Total: +0.266, +0.253, +0.251 and +0.243.
  - LM term: +0.041, +0.041, +0.044 and +0.014.
  - Emission term: +0.218, +0.210, +0.202 and +0.225, which is 82-93 % of the total.
  - Duration term: +0.007, +0.002, +0.005 and +0.004.
- Across all 124 finals, the difference from gold is: J +0.070 to +0.266; LM -0.105 to +0.071 (mean -0.002); emission +0.082 to
  +0.255 (mean +0.158); duration -0.009 to +0.019.
- The emission term is exactly invariant to renaming the symbols. Absorption depends only on the run structure, and the per-symbol
  D_k touches only the LM and duration terms. Only the LM and duration terms see the names, and on them the final keys tie gold.
- [ad hoc] Where the emission gain comes from:
  - The deterministic part H(S) - H(U) differs from gold's by only +0.028 to +0.072.
  - The remainder comes from d_min absorption. Gold relabels 14.3 % of held-out frames; the selected keys relabel 9.9-11.3 %.
    Gold's symbol stream flickers more, which fits the gold key's own frame purity of 0.644 (GoldUnitKeyJob meta).
- Best null (run-permuted corpus, SEL): random_s51 at -5.8543. Real minus null by rank: 1.302, 1.292, 1.290 and 1.282. This is a
  different corpus, so it is a rise statistic, not a key comparison.
- Relabel maps: the best of 8 decipherments of each A10/A11 partition reaches held-out J of -4.903 to -5.046 (SEL), all below
  gold. At a fixed EM partition, re-naming alone did not reach J(gold). Only the unit-level moves took keys above it.

## 3. Agreement with the gold key
Source: AG1, frame-weighted; identity / many-to-one (m2o).
- Selected keys: 0.114/0.564, 0.143/0.579, 0.124/0.489 and 0.065/0.515. At type level: 0.134/0.572, 0.152/0.588, 0.152/0.500
  and 0.096/0.532. [ad hoc] I recomputed all four exactly.
- A10/A11 argmax keys before the search: 0.072-0.136 (mean 0.110) / 0.460-0.505 (mean 0.486).
- The same keys after the search:
  - Full schedule, mean over 7: 0.096/0.501.
  - Warm schedule, mean over 7: 0.119/0.506.
  - Example, a10_durfrz_s01: 0.079/0.497 at J -4.971 before; 0.064/0.522 at J -4.642 (full) and 0.071/0.505 at J -4.619 (warm)
    after.
  - J rose by 0.33-0.43 with no gain in identity.
- Relabel starts: 0.080/0.466 before; 0.087/0.500 after (full) and 0.102/0.510 after (warm).
- Cluster keys, before and after (full, warm):
  - Centroid: 0.082/0.522 before; 0.098/0.523 and 0.091/0.552 after.
  - Context: 0.091/0.459 before; 0.095/0.492 and 0.095/0.514 after.
- Random keys: 0.026/0.223 before (n=64; stage 0's 20 random keys read 0.026/0.225); 0.098/0.487 after. The best random-start key,
  random_s12, reaches J -4.590 at 0.202/0.519.
- Ladder (printed in AG1 and AG0): K30 0.696/0.713, K70 0.297/0.352, K100 0.000/0.222. r30 reads 0.864/0.864 at J -4.822.
- Baselines: chance identity is 1/40 = 0.025, which matches the random keys. The m2o chance level is empirical: about 0.22 for both
  the random keys and K100.
- Reading:
  - The final keys' m2o (0.45-0.58) lies between K70 and K30.
  - Their identity (about 0.10) is about what a 90 %-reassigned key would have. That point is interpolated between K70 and K100;
    no 90 % ladder key was run.
  - Their J is 0.86-1.06 above K30's (1.03-1.06 for the four selected keys) and 0.07-0.27 above gold's.
  - AG1 prints Spearman(J, frame identity) = 0.548 over all 346 keys and 0.476 without the random keys, descriptive only.
  - [ad hoc] Within the 124 finals, Spearman(J, identity) = 0.06. The ordering is driven by start (chance) versus searched keys,
    not by anything within the searched keys.

## 4. Do the start families converge to similar keys?
- No job prints pairwise agreement between keys, so that is **not available from the artifacts**.
- What is printed: similar J across families. The best key per family is cluster_centroid -4.5525, cluster_context -4.5671,
  relabel -4.5780, argmax -4.5860 and random -4.5903. All 124 finals fall in [-4.748, -4.552], and the family ranges overlap.
- [ad hoc, not for citation] Pairwise frame-weighted identity among the finals:
  - Within a family, 0.122 (n=2,438 pairs). Across families, 0.111 (n=5,188 pairs; maximum 0.325).
  - Full/warm twins from the same start: 0.223.
  - The four selected keys among themselves: 0.055-0.189.
  - Symmetric m2o is about 0.55. NMI is 0.67, against 0.38 for pairs of random start keys and 0.60 for the finals against gold.
- So the families converge to similar J and share partition structure, but they do not converge to one key. The finals agree with
  each other about as little as they agree with gold, which points to a broad plateau of many distinct keys above J(gold). A reader
  printing these pairwise statistics would make this citable.

## 5. What the search licenses, what it does not, and what would change the reading
- Licensed:
  - Gold is not J's maximiser on this held-out set. At least 124 label-free keys score higher, all far from gold.
  - The advantage over gold sits entirely on the name-invariant emission term (partition quality and smoothness of the symbol
    stream), with the LM term at parity with gold. So J at its found optima does not favour the gold names over wrong ones.
  - J does carry some key information: from random starts, the search lifts identity from 0.026 to 0.098 and m2o from 0.22 to
    0.49. That gain stops at about the A10 level. Climbing J beyond it (argmax keys +0.33 to +0.43) buys no agreement.
  - Stage 1 selected the top 4 by J, so its four keys should be expected to carry that property. Their identity is 0.07-0.14 and
    their m2o 0.49-0.58, no better than the A10 argmax keys on identity and only modestly better on m2o.
- Not licensed:
  - That J's global maximum is far from gold. The search gives lower bounds on max J. No search started at or near gold (for example
    at r30, identity 0.86, J -4.822, or at K30), so whether a near-gold key exceeds -4.55 is unknown.
  - That the four keys are "the" optimum. They are four of many distinct near-equal keys.
  - Anything about stage 2's S-EM outcome or its PER, which is the real quantity.
  - That J is blind. It separates random keys from searched keys.
- What would change the reading:
  - A disclosed label-using analysis search started at gold, and at r30 or K30, under the same schedule. If it stays near gold
    (identity above 0.8) and reaches J at or above -4.55, J's maximum may lie near gold and stage 1's starts never reached that
    basin. If it drifts to identity near 0.1, J's maximum lies away from gold by construction.
  - An oracle-name check: J of each selected key's partition, and of each A10 partition, under the gold-majority (m2o) name map,
    compared with its searched names. This shows whether J's LM term prefers the gold names on a fixed partition.
  - Stage 2's PER for the four key-seeded S-EM runs against an argmax-key seed on the same items.
    - A win there would show the four keys to be useful seeds.
    - It would not make J track gold.

## Corrections to the extraction
1. Section 4, "Same-scale reproduction check": AG1's gold J is copied from F0, not reproduced. The valid comparability evidence is
   the exact match on the 27 shared keys (section 1 above).
2. Section 4, first sentence ("No real key's final J_ho beats gold's -4.818") is wrong. The extraction corrects it in the next
   sentence only for the top key. In fact all 124 real finals beat gold.
3. Section 2, rank 1: "J_tr start -4.9548" is a stray value. The train-side start is -4.9675 and the held-out start -4.9553.
4. Section 5 lists 10 of the 14 argmax finals. The omitted four are durfrz_s02 (-4.7108; identity/m2o 0.137/0.485),
   durfrz_s02_warm (-4.6763; 0.177/0.507), durinit_s02_warm (-4.7007; 0.126/0.519) and uniform_s01_warm (-4.7484; 0.132/0.504).
   They do not change the reading.
5. Truncated wall time is 5.39-5.75 ks, not "about 5.5-5.7 ks". This is immaterial.
