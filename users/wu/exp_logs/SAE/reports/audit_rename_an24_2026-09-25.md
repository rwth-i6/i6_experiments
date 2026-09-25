# Audit: AN-2 and AN-4 reads (SAE_4A_rename.md), 2026-09-25

Verdict: CONFIRMED_WITH_CORRECTIONS. Every number and every reading reproduces from the per-utterance files. The
corrections concern claim scope and the wording of decision-table row 2, the form and pairing sensitivity of P2,
the fragility of P3 on the gold-key trajectory, one rule-text omission in the AN-4 job, and an open precedence
between rows 2 and 6.

## Artifacts checked
- Jobs (resolved): `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/rename_an24_jobs/`
  - `An2ReadJob.91eeK6mtb6UH/output/{report.txt, an2.json, per_utterance.tsv}`
  - `An4ReadJob.c75KVY92wtjP/output/{report.txt, an4.json, per_utterance.tsv}`
- Code: `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/rename_an24_jobs.py`, `phi_from_key.py`, `lattice.py`.
- Config: `recipe/2025-10-speech-llm/src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_rename_an24_v1.py`.
- Rules: `recipe/i6_experiments/users/wu/exp_logs/SAE/SAE_4A_rename.md` (R1 at 92-118, AN-2 at 170-183, AN-4 at
  197-215, AN-5 at 216-221, decision table at 277-287 in the 390-line version read at the end of the audit).
- Git history of the rules: decision table unchanged since b8774d3d7; AN-2 unchanged since e9ed4ec64 (R1); AN-4
  changed only by the R1 scope bullet (08b33244f, 01:43:44). Both reports were written later (02:00:37 and
  02:00:48), so both jobs ran after the final rule text was committed.
- Reports: `reports/extract_rename_an24_2026-09-25.md` (matches the job outputs),
  `reports/review_rename_an24_launch_2026-09-25.md`, `reports/impl_rename_an24_2026-09-25.md`.
- Recomputation scripts (scratch, not part of the project): `an2_recompute.py`, `ab_check.py` and
  `an4_recompute.py` in this session's scratchpad. They read only per_utterance.tsv, the per-column score json and
  the phi checkpoints.

## Check 1: registered rule text against the job's printed rules
- AN-2: the printed rules (H2, SEES / PREFERS / NAME-BLIND / MIXED, KEEPS / FAVOURS) match the registered text word
  for word. The implementation follows it: sees = d < min(rand) - 0.01; prefers = d > 0.01;
  blind = min(rand) <= d <= max(rand); every clause that fires is reported. The registered text does not make the
  clauses exclusive, so reporting both SEES and PREFERS is correct under it.
- AN-4: the printed P1a/P1b/P1c/H3'/P2/P3 match the registered text, but the job does not print the R1 scope
  bullet (plain form only; P2 on the plain term; rate through P1c). The job's behaviour conforms to the bullet:
  every AN-4 term comes from the plain lambda = 1 +/- 0.05 difference. This is a rendering omission, not a
  deviation.
- Two operationalisations are the implementer's, not the registration's (listed as open choices in the impl
  report):
  - P2 "paired S gap of the same phis" is computed on group means (mean over finals minus mean over basin, for both
    D_LM and D_S).
  - P3 "as PER rises" is computed on the endpoints 0 -> 48.
  Both are sensitivity-checked below.

## Check 2: the numbers are the quantities the rules name
Recomputed from per_utterance.tsv and the score json files. Every AN-2 value matches the job exactly (difference
0); every AN-4 value matches to 2.2e-15. Reproduced at lambda = 1: min over the finals 3.299030 (durinit_s01), max
over the basin 3.223870 (g_dur ep48), and S 3.207044 for the gold-key basin arm.

- S_1: held-out, tau = 1, null recognizer (log_q = 0), trigram-history lattice, d_min 2, float64. It is the
  utterance mean of nll_tau1_per_frame over the 260 disjoint CV-holdout tags (`CvDisjointSegmentsJob.PvgJ79Qc1Nro`).
  The count is 260 of 260 in every column and the excluded list is []. The scoring settings are identical across
  all 44 columns at every (lambda, form); only h_lm_json differs by design.
- Full derangements (H2): fixed-point-free by `check_derangement`. The pool is R1's: the 37 non-SIL symbols that
  hold at least one unit in the gold key (`key_pool`). All 37 move and SIL stays. There are 3 seeds.
  `PhiDerangeJob` sets row k to source row g[k] for both emission and duration (checked with torch.equal).
  - S_1(gold key) = 4.57305. The derangements are worse by +0.5399, +0.5132 and +0.5673, against a 0.1 bar, on
    all 3 seeds. H2 REFUTED.
- The phis compared, and what a rename permutes:
  - (a): `PhiFromKeyInitJob` on `KeyOracleNameReadJob.QD3S2xcmg0Wt/output/keys/selected_{r}__a.json`.
  - (b): the same job on `selected_{r}__b.json`.
  - Random renames: `PhiDerangeJob` full, seeds 1-3, over the 39 non-SIL symbols, applied to (a).
  - A rename permutes which LM symbol each emission/duration row answers to. The table is
    0.9 * ML(key counts) + 0.1 / 500, and the durations are durinit, so every non-SIL duration row is identical.
- (b) against (a) is like-for-like. The (b) table equals (a)'s rows permuted (max difference 0.0). Duration rows and
  every other parameter are identical, SIL is fixed, both use smoothing 0.1, and neither has an empty symbol. Only
  the names the LM sees differ.
  - Caveat: (b) keeps 1-5 non-SIL fixed points relative to (a), while the random renames keep none. This biases d
    toward SEES. Scaled linearly by the share of symbols moved (at most 5 of 39 fixed), the random band would still
    sit near 0.51 or above, against d <= 0.35, so SEES does not depend on it.

AN-2 name clauses (lambda = 1, where the plain and rate-neutral forms coincide because the offset is
(lambda - 1) / lambda * H_LM = 0):

| Key | d = S_1(b) - S_1(a) | Random renames minus (a) |
|---|---|---|
| 1 | +0.3506 | +0.6243, +0.6669, +0.6776 |
| 2 | +0.2508 | +0.6243, +0.6120, +0.7436 |
| 3 | +0.2916 | +0.6183, +0.6464, +0.7205 |
| 4 | +0.2784 | +0.6643, +0.5866, +0.6823 |

SEES NAMES fires on 4 of 4, PREFERS FOUND NAMES on 4 of 4, and NAME-BLIND on 0 of 4. S separates the oracle 1:1
names from random renames (J did not: A20 read NAME-BLIND), but it still ranks the found names first.

- Basin lead = min over the six finals minus max over the durinit basin set (gold-key, G-dur and r30-dur at 48), as
  registered.
  - Rate-neutral form (R1): leads +0.07516, +0.16558, +0.25966 and +0.27174 at lambda = 1, 2, 4.4 and 10. LAMBDA
    KEEPS THE BASIN; FAVOURS FINALS does not fire.
  - Plain form, reported beside: +0.075, +0.081, -0.053 and -0.481, which would read NEITHER.
  - The rate-neutral form matches R1. Every prior entry is shifted by (lambda - 1) / lambda * H_LM, SIL included,
    and the lattice multiplies by lambda. The lattice's `_prior_term` is linear with no renormalisation and there
    is no EOS term.
  - H_LM = 2.2571083839547077 nats per token, from `TrigramLogLossJob.853KJaAjFUKb`: the counted lines of the bed
    prior `PhoneNgramPriorJob.RtzbESkOedsT` (1,000,000 lines, 81,559,944 tokens); the held-lines value is 2.25770.
- AN-4 terms (plain form only, lambda = 1 +/- 0.05):
  - LM per frame = (log Z_1.05 - log Z_0.95) / (0.1 T);
  - channel = E_q[sum of G] / T, emission plus duration;
  - H = -S - channel - LM;
  - rate = 50 * sum E[N_nonSIL] / sum of the original frames;
  - LM per phone = sum of LM_u, including the cost of the SIL tokens, divided by sum E[N_nonSIL].
  This matches the registered E_q[log P_LM] and the R1 scope bullet. AN-4 used the plain form only. No utterance
  was excluded.
- P2 uses the S gap of the same phis, on group means: D_LM = -0.02157 and D_S = +0.13679, a ratio of 0.158, so P2
  is True. The remainder of D_S decomposes as D_channel = -0.1244 and D_H = +0.0092.

## AN-4 readings (recomputed)
- P1a is False. The finals' channel term is -3.049 to -2.918 against the basin's -2.877 to -2.860: the finals are
  lower, not higher.
- P1b is True. LM per phone is -3.987 to -3.809 for the finals against -3.163 to -3.124 for the basin.
- P1c is True. The rate is 6.854-7.562 Hz for the finals against 8.416-8.549 Hz for the basin.
- Not H3', because P1a fails.
- P2 is True as computed, but it depends on the reading:
  - Per pair (6 x 3 = 18 finals-basin pairs), 17 of 18 pass. The exception is durinit_s01 against g_dur:
    |D_LM| = 0.0395 against half the D_S of 0.0376. An "every pair" reading would fail P2; the registered text does
    not choose.
  - Under the rate-neutral derivative (not registered, out of scope by the bullet), D_LM would be -0.0945 against
    half the D_S of 0.068, and P2 would fail. P1b would stay True in either form.
  - The bullet's "constant per phone" is not exact: the SIL to non-SIL token ratio ranges 0.041-0.066 across phis.
    P1b's margin (above 0.6 nats per phone) makes this immaterial.
  - P2 feeds no decision-table row.
- P3 HOLDS by the endpoint rule on all three basin trajectories.
  - gold-key: the endpoint dPER is only +0.0059, and its Hungarian PER is not monotone (0.3848, 0.3336, 0.3670,
    0.3907 at 0, 4, 12 and 48).
  - g_dur and r30_dur rise monotonically (+0.148 and +0.120).
  - A step-wise reading also holds: on every step where PER rises, channel rises and LM per phone falls.
  - PER comes from the dev-other decode, while the terms come from the 260 CV set.
  - P3 feeds no decision-table row.

## Check 3: decision-table rows
- Row 1 (TP-A1) does not fire. It needs AN-3 reads, and AN-3 was dropped after AN-0 DEAD.
- Row 2 FIRES: S PREFERS FOUND NAMES with H2 REFUTED.
  - Rename levers are dropped (TP-A1 and TP0's lambda arms).
  - TP-A2 is NOT brought, because its condition "AN-4 P1a and P1c hold" fails on P1a.
  - TP-B and the coarse-to-fine inventory are brought.
- Row 3 (TP-A2) does not fire, because P1a is False.
- Row 4 (TP-C) does not fire: AN-1 read NOT GOLD FIRST.
- Row 5 does not fire, because AN-3 was dropped.
- Row 6 (AN-5 EM RENAMES with KEY BASIN leads to "stage 2 is the route; no TP") stays OPEN pending AN-5. If it
  fires it contradicts row 2's proposal (TP-B plus inventory against no TP), and the table has no precedence rule.
  This must be settled before AN-5 is read, not after.
- Row 7 does not fire: H2 is REFUTED and row 2 fires.

## Check 4: operating point and corrections
Operating point measured:
- The names and H2 read at lambda = 1, where both forms coincide. The basin lead is read at lambda 1, 2, 4.4 and 10
  on the rate-neutral form.
- The phis:
  - the gold-key phi `PhiFromKeyInitJob.f0jaGuiJVe6A` at sub-epoch 0;
  - stage-1 (a) `r5Ufhm7mnjRK`, `3MQ5edH5NjWN`, `BjYtniPxft7g`, `giteWpZkrsfg`;
  - stage-1 (b) `g94ahzsobJF9`, `v52TpKaLnB6X`, `mcRBteAZe71Y`, `mQ6EkgTnA1B8`, all key-built at sub-epoch 0
    (cell-free emissions, durinit durations);
  - the finals: the six A10 `PhiFirstProbeTrainingJob` checkpoints at epoch 48;
  - the basin: `PackedBlankfreeTrainJob.ge1MKcAPmZIV` gold_key, g_dur and r30_dur at epoch 48.
- The corpus is the 260 disjoint CV-holdout utterances.

Corrections a claim needs:
1. Row 2's wording "on the partitions EM reaches" overstates the evidence. The read concerns the four stage-1
   key-search partitions as key-built phis before any EM, not EM-refit phis or the A10 finals. Suggested wording:
   "S, like J, ranks the found (wrong) names above the oracle 1:1 names on the four stage-1 key partitions, scored
   as key-built phis before EM."
2. The found names were climbed on J (train side), and the key-built phi's S is J's objective with a soft
   channel. PREFERS therefore largely re-expresses A20 through S. It is not independent evidence about what EM does
   from those phis, which is AN-5's question.
3. The oracle names (b) are the best 1:1 naming of a many-to-one partition (identity 0.39-0.47, not 1.0).
   "Prefers found names" means over that naming, not over gold.
4. Selection on the same set: the top 4 keys were selected by held-out J on the same 260 utterances (see
   `SAE_4A_lexlat_v2.md`, A16 (b) and A20: identical J values, "260 set"). This favours (a) on this set and pushes d
   toward PREFERS. The size is not measured. To flip PREFERS it would have to exceed 0.24 per frame on 2 of 4 keys,
   which is more than the entire J range over all 124 finals (-4.55 to -4.75).
5. P2 is True only on group means and the plain derivative. It should be reported as "P2 holds on group means; 17
   of 18 pairs; fails under the rate-neutral derivative". It gates nothing.
6. The gold-key trajectory's P3 endpoint rise (+0.006) is within any plausible decode spread, and its path is
   non-monotone. "P3 HOLDS on all three" rests on g_dur and r30_dur. It gates nothing.
7. The AN-4 job's printed rules omit the R1 scope bullet. The behaviour conforms, but the report is not
   self-describing.
8. Row 2 against row 6 precedence is unregistered. Fix it before AN-5 reports, as an amendment dated before the
   AN-5 result.

## Single strongest reason for the verdict
The per-utterance recomputation reproduces every AN-2 and AN-4 value exactly on the registered quantities. The
comparisons are like-for-like: (b) against (a) is a pure row permutation, and the same 260 tags and scoring
settings are used in all 44 columns. The readings under the registered text are H2 REFUTED; SEES and PREFERS
(4 of 4 each); KEEPS THE BASIN; P1a False, P1b True, P1c True; not H3'; P2 True; P3 HOLDS. Row 2 fires, with
TP-A2 excluded. What needs correcting is the scope row 2 claims (pre-EM key-built partitions) and the open
precedence with row 6, not the numbers.
