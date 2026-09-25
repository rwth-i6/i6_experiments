# Audit: SAE_4A_rename AN-1 V3, J with a 4-gram / 5-gram LM term (2026-09-25)

Verdict: **CONFIRMED_WITH_CORRECTIONS**. The corrections are minor and none of them touches a number or a verdict.

I re-derived every number in the claim with independent code. My code counts n-grams on the uniform-sample window, runs its own
interpolated Witten-Bell recursion, and tokenises the held-out side with a literal per-utterance loop. Keys were read from their
source files. My LM terms equal the job's on all 218 real-corpus rows at orders 3, 4 and 5 (max |diff| 8.9e-16). The readings
therefore hold:
- V3a (4-gram): NOT GOLD FIRST, margin -0.2570, gold ranks 136th of 188.
- V3b (5-gram): NOT GOLD FIRST, margin -0.2453, gold ranks 136th of 188.
- V0 (the reference J): margin -0.2656, gold ranks 136th of 188.
- NAMES VISIBLE: 0 of 4 selected keys pass under V3a and under V3b.

Under the decision table, no row fires, so TP-C is not brought. The corrections:
1. The executor report says the job ran on the gpupack SLURM engine. That is false: it ran on the login node (see "Other findings").
2. The record should state the licensed scope, given below.
3. SAE_4A_rename.md lines 450 and 554 still say that V3 was not built.

## Artifacts read
- Job `work/speech_llm/sae/emc/key_objective_v3_jobs/KeyObjectiveV3Job.N7q9hgRBDrFw`: output/report.txt, output/table.json,
  log.run.1, submit_log.run and usage.run.1. The job ran from 10:45:01 to 10:51:54 on 2026-09-25 and finished successfully.
- Code at commit 1add18a8 (10:43:07) in recipe/2025-10-speech-llm: `prior_high_order.py`, `key_objective_v3_jobs.py`, and the
  unchanged `prior.py`, `unit_key.py`, `key_objective_screen_jobs.py` and `key_search_jobs._j_pair`. The working tree matches the
  commit, and every file's mtime is before the run.
- The registration, SAE_4A_rename.md, bullet "AN-1 V3", was committed in e7062ff66 at 10:26:34, before the code and before the
  run. It has not changed since. The only later edit (c9641e31d, 10:30) touched State. I also read AN-1's readings (lines
  162-177), the decision table (lines 321-339) and TP-C (line 307).
- Reports: `reports/impl_rename_an1v3_2026-09-25.md`, `reports/exec_rename_an1v3_launch_2026-09-25.md`, and the reference
  audit `reports/audit_rename_an1_2026-09-25.md`, together with that audit's independent scores (scratchpad
  `an1_audit/scores.jsonl`).

## 1. Independent recomputation of the LM terms
I wrote a script that imports nothing from speech_llm. It works as follows:
- It reads the first 1,010,000 lines of `SampleLinesJob.orN768ARKwlt/output/text.phn.gz`. Lines whose index is a multiple of
  101 are held out, and the rest are counted. Each counted line is padded with 4 BOS.
- It counts orders 2-5 sparsely (`np.unique` over encoded n-grams). This gives 1,000,000 lines and 81,559,944 tokens counted,
  and 10,000 held-out lines. The 5-gram model has 2,356,634 distinct 5-grams over 408,617 contexts.
- The estimator is interpolated Witten-Bell, applied recursively per queried token:
  - p1 = (c + T0/40) / (N0 + T0);
  - p_n = (c(h,w) + T(h) p_{n-1}) / (N(h) + T(h)) for a seen context;
  - p_n = p_{n-1} for an unseen context.
- Fit check: my orders 1-3 equal the banked `PhoneNgramPriorJob.RtzbESkOedsT/prior.npz` (log_uni, log_bi, log_tri), with
  max |diff| 0, 0 and 0. So my estimator is the trigram's estimator, and extending it to orders 4 and 5 is the same recursion.
- I also read the job's estimator line by line. `fit_log_tables` calls `prior._witten_bell` on row blocks, and row r of order
  n backs off to row r mod 41^(n-2) of order n-1 (the oldest symbol dropped). The tables are indexed h1 + 41 h2 + ...
  + 41^(n-2) h_{n-1}, and `token_log_probs` uses the same order. This matches `from_counts`' np.tile rule at order 3.
- Tokenisation: a literal per-utterance loop over the 260 held-out utterances of A13's disjoint.segments (137,933 frames):
  - runs;
  - d_min = 2 absorption (the nearest preceding long run, else the nearest following one);
  - collapse of equal neighbours;
  - D_k split (25 frames for a phone, 50 for SIL).
  The history is BOS-padded at each utterance start, with no EOS term.
- Keys: gold, the K30/K70/K100 x 5 ladder keys, A20's keys/ directory, the KeySearchSelectJob picks and the stage-1
  search.json unit finals. The derangements were rebuilt by AN-1's recipe. All 218 real-corpus keys equal the keys in the
  job's table.json.

Result: over all 218 real rows, held out, max |my lm_n - job lm_n| is 4.4e-16 (order 3), 4.4e-16 (order 4) and 8.9e-16
(order 5). The job's order-3 term also equals the AN-1 audit's independent lm on all 218 rows (max |diff| 0), and its emis and
dur equal that audit's to 2.2e-16.

Sample rows, held out, in nats per frame (mine; identical to the job's to 1e-15):

| row | lm3 | lm4 | lm5 | emis | dur | J_V3a | J_V3b |
|---|---|---|---|---|---|---|---|
| gold | -0.9897 | -1.1571 | -1.2901 | -3.3468 | -0.4815 | -4.9854 | -5.1184 |
| selected_1 (a) | -0.9484 | -1.1243 | -1.2761 | -3.1290 | -0.4750 | -4.7284 | -4.8802 |
| selected_1 (b) | -1.5218 | -1.7581 | -1.8604 | -3.1290 | -0.4750 | -5.3622 | -5.4645 |
| selected_3 (a) | -0.9457 | -1.1106 | -1.2517 | -3.1450 | -0.4764 | -4.7320 | -4.8731 |
| K30_s1 | -1.3991 | -1.6479 | -1.7820 | -3.7141 | -0.4889 | -5.8510 | -5.9851 |

## 2. Is the higher-order fit sensible?
On the window's held-out text it is. Perplexity falls at every order, and the 5-gram is better than the 4-gram:

| order | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| held-out perplexity (mine = job's) | 26.7269 | 14.1426 | 9.5611 | 7.2015 | 5.9841 |
| nats per token | 3.286 | 2.649 | 2.258 | 1.974 | 1.789 |
| held-out tokens whose n-gram is unseen | - | 0 | 0.003 % | 0.10 % | 1.06 % |

There is no sign of over-fitting or a smoothing defect on text. Orders 1-3 equal `prior.stats.txt` exactly.

On the key-induced token strings that J scores, however, the picture reverses:

| string (held out) | nats/token o1 | o2 | o3 | o4 | o5 | unseen 4-grams | unseen 5-grams |
|---|---|---|---|---|---|---|---|
| gold | 3.41 | 3.61 | 4.26 | 4.98 | 5.55 | 27 % | 51 % |
| selected_1 (a) | 3.49 | 3.60 | 4.24 | 5.02 | 5.70 | 24 % | 61 % |
| selected_1 (b) | 3.74 | 4.99 | 6.80 | 7.86 | 8.31 | 63 % | 86 % |
| K100_s1 | 4.12 | 6.10 | 8.23 | 9.25 | 9.57 | 82 % | 97 % |

For comparison, uniform over 40 symbols is 3.69 nats per token. Gold's own string gets worse with every order from 2 up, and it
is worse than uniform from order 3 on. Half of its 5-grams never occur in 81.6 M tokens of text (1 % for real text). A
higher-order LM mostly backs off on these strings, and it penalises every key's string more sharply. So the V3 read measures a
sharper LM on strings far from its training domain. It is not a test of whether a higher-order LM can recognise correct names.

## 3. Order-3 reproduction, the competitor set and the exclusion
- I recomputed the reproduction from table.json: max |J_3 - J_V0|, |lm3 - lm| is 8.88e-16 over 652 (row x side) comparisons.
  J_V0 and its lm, emis and dur equal AN-1's banked table.json (erCoRAmZID5a) exactly (max |diff| 0) on all 326 rows and both
  sides. The keys equal AN-1's on all 326 rows.
- The competitor list equals AN-1's, in order: 187 rows, made up of the 124 stage-1 unit-level real finals and the 63 A20 rows
  whose key is not gold. My own rule gives the same set.
- Excluded as identical to gold: a20/gold__a, __b and __c, the same as AN-1.
- The stage-1 class-level runs (72 real) and the null finals do not compete, as in AN-1.

## 4. Verdict logic against the registration
My re-derived readings:

| V | J_V(gold) | best competitor | margin | rank of 188 | ladder (seed means) | reading |
|---|---|---|---|---|---|---|
| V0 | -4.8180 | selected_1 (a) -4.5525 | -0.2656 | 136 | monotone | reference |
| V3a | -4.9854 | selected_1 (a) -4.7284 | -0.2570 | 136 | monotone | NOT GOLD FIRST |
| V3b | -5.1184 | selected_3 (a) -4.8731 | -0.2453 | 136 | monotone | NOT GOLD FIRST |

The ladder also holds seed by seed under all three variants, so the seed-mean convention decides nothing here.

J_V(b) - J_V(a) on the 4 selected keys:
- V3a: -0.6338, -0.4844, -0.4878 and -0.4883.
- V3b: -0.5843, -0.4661, -0.4590 and -0.4585.

0 of 4 exceed 0.01, so NOT VISIBLE under both. The claim "between -0.63 and -0.46" is right after rounding.

The rules are applied as registered: margin > 0.01 over every competitor AND gold > K30 > K70 > K100, and NAMES VISIBLE on at
least 3 of 4 keys. The decision-table row "AN-1 GOLD FIRST under some V" does not fire, so TP-C is not brought. The report
does not claim more; its header says the job "decides only whether TP-C is brought".

The licensed statement for the record: "With J's trigram LM term replaced by a 4-gram or 5-gram of the same estimator and
window, at weight 1 and on the same absorbed, split token strings, held out on the 260 set, gold is not first (margin -0.257 /
-0.245, 136th of 188), and the found names are preferred to the oracle 1:1 names on all 4 selected keys. TP-C is not brought."

Not licensed:
- "no higher-order LM can see names";
- "a higher-order LM in S, in the E-step or on phi-posterior strings would not help";
- any statement at another LM weight or on cleaner strings.

## 5. Other findings a reader should know
- **(b) - (a) is entirely the LM term, by construction.** Its emis + dur part is exactly 0.0 on all 4 keys. The reason:
  - I checked from the key files that on every selected key (b) is a bijection of (a)'s 40 used symbols, with SIL mapped to
    SIL.
  - Runs, absorption and collapse depend only on whether neighbouring symbols are equal.
  - D_k depends only on SIL versus phone, so the tokens are unchanged.
  - The held-out emission is the train table with its columns permuted, which leaves each frame's value unchanged.
  - The 39 phone duration rows are one law, and the SIL row is fixed.

  The same holds for the non-SIL derangements. So NAMES VISIBLE under any LM-only variant is a pure LM comparison. The (a)
  names came out of a stage-1 search that maximised J with the trigram. The 4-gram widens the preference for them (-0.48 to
  -0.63, against -0.42 to -0.57 under the trigram). The oracle names still beat all 3 random renames on 4 of 4 keys at every
  order, so the LM does separate names; it just prefers the found ones.
- **Gold's deficit is an emission fact, which caps what any LM-only swap can do at weight 1.**
  - Against the best competitor, the margin splits into lm -0.041 / -0.033 / -0.038, emis -0.218 / -0.218 / -0.202 and dur
    -0.007 / -0.007 / -0.005 (V0 / V3a / V3b).
  - 146 of the 187 competitors have emis + dur at or above gold's.
  - Higher orders do improve gold's LM standing: the number of competitors with an LM term at or above gold's goes 69 -> 50
    -> 40. The rank stays at 136.
  - To clear every competitor by 0.01, gold would need a further LM-term gain of +0.276 (V0), +0.267 (V3a) or +0.255 (V3b)
    nats per frame.

  AN-1's term table already showed this before V3 was built. The negative was therefore close to foreclosed by the emission
  term. It is still the registered read.
- **Per-frame normalisation and token rate.** Per token, gold beats selected_1 (a) at orders 4 and 5 (4.98 against 5.02 and
  5.55 against 5.70 nats). Per frame it loses, because its string carries more tokens per frame (0.232 against 0.224). This is
  J's registered convention.
- **Window.** The job's input and table.json `lm_fit.corpus` are `SampleLinesJob.orN768ARKwlt`. That job is a seeded uniform
  sample, `sorted(random.Random(0).sample(range(39,630,169), 1,010,000))`. Its sentence-initial phones are spread (DH 0.167,
  HH 0.130), where the alphabetical head gave AH 0.74 (`PhoneNgramPriorJob.TRPE0D5nF3bh`). The refit equals the banked
  uniform-window prior exactly.
- **Engine (correction to the executor report).** submit_log.run records engine 'short', engine_name 'local' and host
  jpbl-s02-03, and usage.run.1 records the same host. The job ran on the login node, not on gpupack, and held no booster node.
- The implementer relabelled one render string after its test run. The banked run used the committed file (mtime 10:42:51),
  and its output rendered correctly.
- SAE_4A_rename.md still says "V3 was not built" in the Proposal's "Not tested" list (line 450) and in AN-1's Results (line
  554). Both need updating when V3 is recorded.

## Scratch (session files, not kept)
`/tmp/claude-34349/-e-project1-spell-wu24-2026-07-13-unsupervised/a2d26b34-6240-4bb2-b34b-ab87c54fe915/scratchpad/an1v3_audit/`:
- lm_audit.py and lm_audit.log / lm_audit.json: the fit, the perplexities and all 218 rows;
- analyse.py and analyse.log: the readings from my LM terms plus the AN-1 audit's emis and dur;
- diag.py and diag.log / diag.json: per-token nats and unseen shares on the key strings.
