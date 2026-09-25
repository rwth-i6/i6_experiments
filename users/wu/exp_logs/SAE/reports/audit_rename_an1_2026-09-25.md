# Audit: SAE_4A_rename AN-1, the key-level objective screen (2026-09-25)

Verdict: **CONFIRMED_WITH_CORRECTIONS**.

The job's numbers are right: I recomputed them independently and they match to about 1e-15. The comparison sets are the registered ones, and the rules are applied correctly. The five verdicts follow: NOT GOLD FIRST under V1, V2 and V12, and NOT VISIBLE under V2 and V12. None of the disclosed build decisions flips a verdict under any reading I tried.

The correction is about how far V12 is from GOLD FIRST. That distance depends on one disclosed build decision: V2's occupancy term is divided by all retained frames T, not by the non-SIL frames. With T as the divisor, every SIL frame scores 0 while a non-SIL frame scores about -3.1 to -3.5, so the term rewards keys for putting frames on SIL. Every key that beats gold under V12 has a higher SIL share than gold.
- Job's reading: V12 misses by -0.058, and gold ranks 6th of 188.
- Non-SIL-frame divisor, the other reading the registered formula admits: gold ranks 1st of 188 with a margin of +0.0076. That is still NOT GOLD FIRST, because it does not exceed the 0.01 floor, but it misses by only 0.0024.

## Artifacts read
- Job `work/speech_llm/sae/emc/key_objective_screen_jobs/KeyObjectiveScreenJob.erCoRAmZID5a`: output/report.txt, output/table.json and log.run.1.
  - It ran on jpbl-s02-03 on the local "short" engine, from 00:51:08 to 00:54:28 on 2026-09-25, and finished successfully.
  - Every input path in log.run.1 is a registered input: the VAD HDFs SAjz8y1cT06g, the train side CvHoldoutSplitJob.PfpCPQRCfIAk, A13's disjoint.segments PvgJ79Qc1Nro, the prior RtzbESkOedsT, the duration prior ReQtJKYpZgsN, A20 QD3S2xcmg0Wt, stage 1 AzM1NoHpOnFJ with g9wsznNnqmyO, and stage 0 DgDbciHlq2wY with the gold key sLnMRRd2qO0t.
- Code `key_objective_screen_jobs.py` at commit 7d550c1d (00:49:15). The working tree matches the commit. I also read its test and the J reference in `unit_key.py`.
- Registration: SAE_4A_rename.md, sections "Shared inputs", AN-1 and the decision table, at commit b8774d3d7 (00:31:40). That commit precedes both the code and the run, and the file is unchanged since. I also read design review F4, amendment A6 and the AN-1 cost note.
- Build report `reports/impl_rename_an1_2026-09-25.md`, exec report `reports/exec_rename_an1_2026-09-25.md`, and the prior audit `reports/audit_a20_oracle_names_2026-09-24.md`.

## Independent recomputation
My implementation does not import `unit_key` or the screen job. Its parts:
- its own HDF reader;
- a vectorised d_min absorption across the whole corpus, checked against a literal per-utterance loop of the docstring rule. The loop agrees on every held-out utterance for all 218 real-corpus rows, and on the full train side for gold and selected_1;
- its own D_k token split, and trigram indexing [h2, h1, s] from (BOS, BOS);
- its own max-entropy duration law (bisection; lambda -0.3459 at mean 4.4138 frames; SIL uniform on [2, 50]);
- its own absorbed and unabsorbed emission tables, and pi_LM taken from prior.npz `log_uni`, restricted to the 39 non-SIL symbols and renormalised.

Keys were read from their source files: gold, the stage-0 ladder keys, A20's keys/, the stage-1 search.json finals and the KeySearchSelectJob picks. I regenerated the derangements from the stated recipe, the first fixed-point-free `default_rng(seed).permutation(39)` for seeds 1-3. All 218 real-corpus keys equal the job's keys. Every derangement fixes SIL and has no fixed point on the symbols in use.

Agreement with the job, over all 218 rows, train and held out, on lm, emis, dur, emis_v1, occ and J_V0/V1/V2/V12: max |diff| 3.6e-15.
- Baseline, J = V0: my V0 equals A20's table.json on all 132 (key x variant x side) values (max 8.9e-16). It equals stage 1's J_final on all 124 real finals, both sides (1.8e-15). Gold held out is -4.818029, stage 0's value.
- E9 split: my train emis_v1 equals the job's H(S) - H(U) for gold and the 4 selected keys (for example gold -2.749873), and my train emis equals the job's EMIS (gold -3.353004).

Held out, 260 utterances, 137,933 frames, nats per frame. Selected keys are stage 1's ranks 1-4: cluster_centroid_s01/s04_warm and cluster_context_s01/s04_warm.

| row | lm | emis | emis_v1 | occ (job) | dur | V0 | V1 | V2 | V12 |
|---|---|---|---|---|---|---|---|---|---|
| gold | -0.9897 | -3.3468 | -2.7411 | -3.1139 | -0.4815 | -4.8180 | -4.2123 | -7.9319 | -7.3262 |
| selected_1 (a) | -0.9484 | -3.1290 | -2.7154 | -3.2568 | -0.4750 | -4.5525 | -4.1388 | -7.8093 | -7.3956 |
| selected_2 (a) | -0.9483 | -3.1364 | -2.7067 | -3.2895 | -0.4800 | -4.5647 | -4.1349 | -7.8542 | -7.4245 |
| selected_3 (a) | -0.9457 | -3.1450 | -2.6743 | -3.3168 | -0.4764 | -4.5671 | -4.0965 | -7.8838 | -7.4132 |
| selected_4 (a) | -0.9754 | -3.1223 | -2.6724 | -3.2999 | -0.4779 | -4.5755 | -4.1257 | -7.8754 | -7.4256 |

## Rule application (my numbers; identical to the job's)
Competitors are the 124 stage-1 real unit-level finals plus the 63 A20 rows whose key is not gold, 187 in all (173 distinct keys). This list equals the job's. The three A20 gold rows are excluded because they are the gold key itself. Null finals enter no reading.

| V | J_V(gold) | best competitor | margin | gold's rank of 188 | rows >= gold | ladder (seed means) | reading |
|---|---|---|---|---|---|---|---|
| V0 (reference) | -4.8180 | selected_1 (a) -4.5525 | -0.2656 | 136 | 135 | monotone | not read |
| V1 | -4.2123 | relabel_a10_uniform_s02_ep48 -4.0737 | -0.1387 | 134 | 133 | monotone | NOT GOLD FIRST |
| V2 | -7.9319 | selected_1 (a) -7.8093 | -0.1227 | 48 | 47 | monotone | NOT GOLD FIRST |
| V12 | -7.3262 | random_s63 -7.2686 | -0.0576 | 6 | 5 (4 distinct keys) | monotone | NOT GOLD FIRST |

To reach GOLD FIRST, gold would need to gain 0.149 under V1, 0.133 under V2 and 0.068 under V12. The ladder holds on seed means under every variant. Taken seed by seed it breaks under V1: K70_s5 -4.8164 is below K100_s2 -4.7636. That does not matter, because the margin clause already fails.

NAMES VISIBLE, J_V(b) - J_V(a) on the 4 selected keys:
- V2 and V12: -0.8346, -0.6165, -0.5566 and -0.5843. None of the four exceeds 0.01, so both read NOT VISIBLE.
- V12 equals V2 here, because the V1 emission is name-free.
- Random derangements of (a) cost -1.12 to -1.81.
- So the repaired objectives do see names. They prefer the found names to the oracle 1:1 names, and by more than J does (V0: -0.42 to -0.57): the occupancy term adds a further 0.13-0.26 against the oracle names.

## Build decisions: can any of them flip a verdict?
1. **V2's divisor (disclosed: T = all retained frames).** The registered formula "(1/T) sum_t log pi_LM(s_t | non-SIL) over non-SIL frames" does not fix T. The job's reading is J's own per-frame divisor, which is literal and additive to J. Its side effect is that a SIL frame contributes 0, against about -3.1 to -3.5 for a non-SIL frame, so the term rewards SIL share. The other reading divides by the non-SIL frames, which is a pure cross-entropy of the key's non-SIL occupancy against pi_LM. F4's "occupancy = the key's non-SIL segment share" fits that reading.
   - Gold's held-out SIL frame share (absorbed) is 0.0744.
   - The four distinct keys above gold under V12 have shares of 0.083-0.093: random_s63 0.093, argmax_a10_uniform_s02_ep48 and its _warm 0.087, relabel_a10_uniform_s02_ep48_warm 0.083.
   - Over the distinct competitors, the correlation of (J_V12 - gold) with SIL share is 0.35 under the job's divisor and 0.15 under the non-SIL divisor.
   - Under the non-SIL divisor:
     - V12: gold -7.5765 ranks 1st of 188, with margin +0.0076 over cluster_centroid_s04. NOT GOLD FIRST, by 0.0024. The ladder holds on seed means but not seed by seed (K70_s5 -8.7835 is below K100_s1 -8.7675).
     - V2: margin -0.1702, rank 44.

   This is the closest any variant comes to GOLD FIRST. It does not flip the verdict.
2. **Absorbed or unabsorbed string for OCC (disclosed: absorbed, and V12 reuses it).**
   - Unabsorbed string, divisor T: V2 margin -0.1537 (rank 61), V12 -0.0820 (rank 11).
   - Unabsorbed string, non-SIL divisor: V2 -0.1916 (rank 59), V12 -0.0186 (rank 3).

   The job's absorbed choice is the more favourable one for gold. No flip.
3. **A20 rows identical to gold excluded.** This is correct: gold cannot beat itself by more than 0.01. Including those rows would make GOLD FIRST impossible under every V. The exclusion can only help gold, and gold still fails. No flip.
4. **Full derangements, seeds 1-3.** "Shared inputs" lists 1-pair, 5-pair and full; AN-1 does not name one. These rows are reported beside and never read, so no flip is possible.
5. **V3 dropped.** The registration allows this when the estimate is over one day. I cannot check the implementer's estimate from these files. The consequence is that "no GOLD FIRST under any V" covers V1, V2 and V12 only.

NAMES VISIBLE is 0/4 under all four OCC readings (d from -0.56 to -0.85), so it cannot flip.

## Frame
- Same held-out set (260, A13 disjoint), same metric (held-out J per frame) and same J path as A20, stage 0 and stage 1. The equality is exact, so the V0 baseline is comparable by construction.
- Every constant traces to the registration or the J reference: the 0.01 floor (A7), seeds 1-3, d_min 2, D_k 25/50, alpha 1e-3, the prior, the duration prior and the train side.
- "Stage-1 finals" means the 124 unit-level finals, as in E7 and the stage-1 read. The 72 class-level decipher runs are intermediate starts and were not scored. Adding competitors can only make GOLD FIRST harder, so leaving them out cannot bias the reading toward NOT GOLD FIRST.
- The GOLD FIRST ladder is read on seed means. The registration does not specify this, and it matters only when the margin clause passes.

## Corrections to carry into the record
- Record V12 as "NOT GOLD FIRST, margin -0.058 (gold 6th of 188) under the built divisor; gold 1st of 188 but +0.0076 < 0.01 under the non-SIL-frame divisor". It is not a clear miss.
- The V2 term as built is not SIL-neutral: it rewards SIL frames. The rows beating gold under V12 all carry more SIL than gold. Any later use of V2 or V12, including in a TP-C search, should fix the divisor first. The search would otherwise have a direct incentive to inflate SIL.
- Under the decision table as written, AN-1 brings no TP-C: no variant reads GOLD FIRST, and V3 was not tested. The near miss under the alternative divisor is a fact for the orchestrator to weigh. It is not a pass.

## Scratch (session files, not kept)
`/tmp/claude-34349/-e-project1-spell-wu24-2026-07-13-unsupervised/a2d26b34-6240-4bb2-b34b-ab87c54fe915/scratchpad/an1_audit/`: score.py (scorer), scores.jsonl, full.log, subset.log, analyse.py with analyse.log and readings.json, extra.py (SIL-share and divisor sensitivity), and baseline.py (V0 against A20, stage 1 and stage 0).
