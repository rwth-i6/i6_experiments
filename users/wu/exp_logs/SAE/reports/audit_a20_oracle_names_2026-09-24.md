# Audit: A20 oracle-name J check (2026-09-24)

Verdict: **CONFIRMED_WITH_CORRECTIONS**. The registered reading is **NAME-BLIND**: J(b) - J(a) <= 0.01 on 4 of the 4 selected keys. The numbers and the rule application are right, and I re-derived them independently. The correction is to the interpretation. The trigram term is not blind to names. It strongly prefers the found names over the oracle 1:1 names, by 0.42-0.57 nats per frame. On the gold partition it penalises scrambled names by 0.69-1.09. The name gap therefore comes from the found partitions (splits and merges), on which no 1:1 naming is close to right. It is not missing name signal in the trigram.

## Artifacts read
- Job `work/speech_llm/sae/emc/key_oracle_names_jobs/KeyOracleNameReadJob.QD3S2xcmg0Wt` (resolves to /e/scratch/...). I read output/table.txt, table.json, keys/*.json and log.run.1. It ran on the login node through the local "short" engine: submit_log shows engine_info jpbl-s02-03, wall 70 s. That matches the registered cost.
- Code `key_oracle_names_jobs.py` at commit 6e6156ca (23:38:14). The registration commit 59121ec0a (23:26:52) precedes both the code and the run (23:41), and the A20 bullet has not changed since. The uncommitted diff of SAE_4A_lexlat_v2.md touches only State.
- Build report `reports/impl_a20_oracle_names_2026-09-24.md`.
- The J reference in `unit_key.py` and `key_search_jobs._j_pair`, and the stage-1 move set in `key_search.py`.
- Stage-1 inputs: `KeySearchJob.AzM1NoHpOnFJ` search.json, `KeySearchSelectJob.g9wsznNnqmyO`, `KeyAgreementReportJob.Xk3Wxu9kJ38t`, and stage-0 `KeyFloorReadJob.DgDbciHlq2wY` (tarball log).
- My independent recomputation: scratchpad scripts `audit_a20.py`, `audit_a20_b.py` and inline controls. Their logs are `audit_run.log`, `audit_b.log` and `audit_c.log` under /tmp/claude-34349/.../scratchpad/. These are session scratch files and are not kept.

## Re-derived numbers (held-out, 260 utterances, 137,933 frames; nats per frame)
My implementation does not call unit_key's J code. It has its own HDF reader, a literal loop implementation of the absorption rule (it agrees with a vectorised version on every held-out utterance), its own D_k splitting, its own max-entropy duration law solved by bisection (mean 4.4138 frames; SIL uniform on [2, 50]), and its own trigram indexing [h2, h1, s] with a BOS, BOS start. It reproduces every job row to 5 decimals on both the train and held-out sides.

| key | J(a) | J(b) | d = J(b)-J(a) | lm d | emis d | dur d | J(c)-J(a) |
|---|---|---|---|---|---|---|---|
| selected_1 | -4.55248 | -5.12585 | -0.5734 | -0.5734 | 0 | 0 | -0.5767 |
| selected_2 | -4.56470 | -5.00065 | -0.4359 | -0.4359 | 0 | 0 | -0.4742 |
| selected_3 | -4.56709 | -4.98891 | -0.4218 | -0.4218 | 0 | 0 | -0.5742 |
| selected_4 | -4.57550 | -5.01036 | -0.4349 | -0.4349 | 0 | 0 | -0.4894 |
| gold | -4.81803 | -4.81803 | 0 | | | | 0 |

Rule: 0 of 4 above 0.01 and 4 of 4 at or below it, so **NAME-BLIND**. The reading is not sensitive to the margin: every d is at least 0.42 below it.

## Checks
1. **J as in stages 0-1: holds.**
   - The job calls the same `_j_pair` on the same input files as stage 0's KeyFloorReadJob and stage 1's KeySearchJob: the units and orig_length HDFs, the train segments `CvHoldoutSplitJob.PfpCPQRCfIAk` (28,254 utterances), A13's `disjoint.segments` (260 utterances), the prior `PhoneNgramPriorJob.RtzbESkOedsT` and the duration prior `ReQtJKYpZgsN`.
   - alpha is 1e-3 and d_min is 2. The rate band is printed and not applied, and no row is VOID.
   - The in-job same-scale check gives |dJ| = 0.0 on all 22 keys. My recomputation matches too.
2. **(b) is the agreement-maximising 1:1 map and leaves the partition unchanged: holds.**
   - My own linear_sum_assignment on the 40x40 frame-weighted table (train-side unit frame counts, which are KeyAgreementReportJob's weights) gives exactly the job's (b) key for all 4 keys and for gold. No pairwise swap improves the assignment (2-opt gain 0). The (c) keys also match.
   - (b) renames 37, 36, 34 and 39 symbols. perm[SIL] = SIL in all four.
   - Emission, absorbed and relabelled frames, token count, non-SIL tokens and overlong segments are identical under (a) and (b). My recomputed tri counts of (b) equal the relabelled tri counts of (a).
3. **Terms: lm carries all of J(b)-J(a); dur and rate are unchanged.** Because SIL maps to SIL, neither the SIL duration row nor D_k = 50 moves.
4. **Could a label-free search have found (b)? No, because J ranks it lower.**
   - All 4 selected runs are untruncated and converged. Their final T=0 unit sweep and swap sweep (all 780 name pairs) moved nothing, so (a) is a pairwise-swap local optimum of train J, and (b) is 0.42-0.57 below it.
   - This is neither the SIL duration term nor an absorption artifact (point 2).
   - The lm loss is not confined to leftover names. Symbols that (b) names correctly lose 0.09-0.29 of it. Even trigrams whose three positions are all correctly named (11-39 % of tokens) lose 0.000-0.103.
   - *Unregistered:* a steepest-ascent name-swap climb from (b) on train lm (SIL fixed, label-free) lowers identity from 0.39-0.47 to 0.13-0.26. It stops at lm -0.98 to -1.08, not at (a).
   - *Unregistered control:* on the gold partition, random non-SIL name permutations cost 0.69-1.09 of lm, so the trigram does see names. Gold names are, however, not its exact local optimum: the best single swap (JH<->W) gains +0.010 held-out and +0.009 train.
5. **(c) (22-26 symbols):**
   - The emission term changes by -0.453, -0.473, -0.467 and -0.407; lm by -0.135, -0.013, -0.122 and -0.093; dur by +0.012 to +0.015.
   - The symbol frame entropy H(S) falls by 0.46-0.60 under (c).
6. **Identity after (b):**
   - (a) 0.114, 0.143, 0.124, 0.065; (b) 0.465, 0.473, 0.394, 0.428; many-to-one 0.564, 0.579, 0.489, 0.515.
   - Wrong names account for about 0.28-0.36 of identity. The remaining 0.09-0.10 up to many-to-one is frame mass that no 1:1 naming can label correctly.
   - Duplicates: 10-15 gold phones are each the majority of 2-4 symbols (T up to 4, N up to 4, SIL 2-3).
   - Merges: 14-18 phones are no symbol's majority; B, CH, G, JH, M, OY, P, TH, UH and ZH are among them in all four keys.
   - Under (b), 15-22 symbols, holding 25-47 % of the frames, carry leftover names. Each key's extra SIL cluster is named V, Z or ER in (a) and UH, D, Z or IY in (b).

## Unregistered: the emission term against H(S) - H(U) (train side)
- The train emission is exactly H(S') - H(U,S'), where S' is the absorbed symbol. It equals H(S) - H(U) only without absorption. H(U) = 6.0751.
- Found keys (a): H(S) - H(U) is -2.72, -2.71, -2.68 and -2.68, and the emission is -3.128 to -3.149.
- Gold: H(S) - H(U) is -2.750 and the emission is -3.353.
- H(S) - H(U) therefore separates the found keys from gold by only 0.03-0.07. The 0.20-0.23 emission margin over gold divides as follows:
  - 0.10-0.17 from a lower absorption uncertainty H(S'|U): 0.42-0.49 for the found keys against 0.59 for gold. The relabelled frames are 9.3-10.6 % against 13.6 %.
  - 0.06-0.10 from a higher H(S').
- The name-free reward that favours these partitions is thus mostly the d_min absorption term, with symbol entropy second.

## Frame
- The keys are stage 1's selected ranks 1-4, as registered: cluster_centroid_s01/s04_warm and cluster_context_s01/s04_warm.
- The registration says 6 argmax finals and the job reports 7 (six A10 and A11 real_c_s16). These rows are report only, the deviation is disclosed, and it does not change the verdict.
- The oracle's agreement is key-level and frame-weighted by train-side unit counts, before absorption. That matches "frame agreement with the gold key" and KeyAgreementReportJob.

## Corrections to carry into the record
- NAME-BLIND must not be glossed as "the trigram term does not see names". On these partitions J *penalises* the gold-agreeing 1:1 names by 0.42-0.57, entirely through lm, and a label-free rename climb moves away from gold.
- "A rename step would not help" is supported: (a) is already swap-optimal and J(b) < J(a).
- "The cost work goes to the objective's name signal" goes further than the evidence. The (b) test cannot separate a name-signal deficit from a partition deficit, because no 1:1 naming of these partitions is near-right (identity at most 0.47). The unregistered entropy split points at the emission term's absorption and entropy reward as what selects these partitions.
