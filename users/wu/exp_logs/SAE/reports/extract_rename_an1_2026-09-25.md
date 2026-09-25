# AN-1 extraction, SAE_4A_rename
Job dir: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/key_objective_screen_jobs/KeyObjectiveScreenJob.erCoRAmZID5a
(alias: work/speech_llm/sae/emc/key_objective_screen_jobs/KeyObjectiveScreenJob.erCoRAmZID5a under the setup dir; source alias/sae/4a/rename/an1/objective_screen)
Files: output/report.txt (524 lines), output/table.json, log.run.1

## 1. VERDICT lines verbatim (report.txt lines 517-521)
VERDICT V1 (GOLD FIRST rule): NOT GOLD FIRST
VERDICT V2 (GOLD FIRST rule): NOT GOLD FIRST
VERDICT V12 (GOLD FIRST rule): NOT GOLD FIRST
VERDICT V2 (NAMES VISIBLE rule): NOT VISIBLE
VERDICT V12 (NAMES VISIBLE rule): NOT VISIBLE

Log.run.1 line 56 summary: "V1: NOT GOLD FIRST (margin -0.1387); V2: NOT GOLD FIRST (margin -0.1227); V12: NOT GOLD FIRST (margin -0.0576); V2: NOT VISIBLE; V12: NOT VISIBLE; wall 198 s, peak RSS 1.96 GB"

### READING RULES (printed verbatim, report.txt lines 6-19)
READING RULES (registered 2026-09-25, SAE_4A_rename.md AN-1, fixed before any result), on held-out
J_V in nats per frame, margin 0.01 (A7's floor):
  GOLD FIRST under V  if  J_V(gold) exceeds every stage-1 real final and every A20 variant by more than
    0.01, AND gold > K30 > K70 > K100 (seed means, strict).  A20 rows whose key equals the gold key
    (all 500 units) are not competitors of gold; they are listed.  Read for V1, V2 and V12; V0 (= J)
    is printed as the reference.
  NAMES VISIBLE under V  (V2 and V12 only)  if  J_V(b) - J_V(a) > 0.01 on at least 3 of the 4 selected
    keys; otherwise NOT VISIBLE.  The random-rename rows J_V(derangement) - J_V(a) are reported beside,
    never read.  V0 and V1 differences are printed, not read (V1's emission is name-free).
  The stage-1 null keys are scored on their own corpus and enter no reading.  A key's VOID flag (rate
  band [5.80, 14.49] Hz) is printed, not applied; the rules read J_V as computed.
  Consequence (decision table): GOLD FIRST under some V brings TP-C beside the other proposals; TP-C then
  carries stage 0 / 1's own gates again under J_V.

V3 line (report.txt line 21): "V3: DROPPED: a 4-gram / 5-gram prior needs a new prior fit (prior.py fits
orders 1-3) on the uniform-sample window with its own job, launch and review before this read could run,
and a new LM term (the key scorer indexes a trigram table); estimated above one day end to end, so not
built (AN-1: built only under one day)"

## 2. GOLD FIRST block (held-out J_V; report.txt lines 25-29), per variant
Header: "GOLD FIRST (held-out J_V; competitors = the 124 stage-1 real finals + A20 variants whose key is
not gold; 3 A20 rows identical to gold excluded: a20/gold__a, a20/gold__b, a20/gold__c)"

V0   J_V(gold) -4.8180   best competitor a20/selected_1__a -4.5525   margin -0.2656 (<= 0.01)
     competitors at or above gold 135/187; ladder K30 -5.6097 K70 -6.3335 K100 -6.5597 monotone True
     -> reference (J), not read
V1   J_V(gold) -4.2123   best competitor real/relabel_a10_uniform_s02_ep48 -4.0737   margin -0.1387 (<= 0.01)
     competitors at or above gold 133/187; ladder K30 -4.4822 K70 -4.7348 K100 -4.8280 monotone True
     -> NOT GOLD FIRST
V2   J_V(gold) -7.9319   best competitor a20/selected_1__a -7.8093   margin -0.1227 (<= 0.01)
     competitors at or above gold 47/187; ladder K30 -8.9447 K70 -10.0884 K100 -10.5523 monotone True
     -> NOT GOLD FIRST
V12  J_V(gold) -7.3262   best competitor real/random_s63 -7.2686   margin -0.0576 (<= 0.01)
     competitors at or above gold 5/187; ladder K30 -7.8172 K70 -8.4897 K100 -8.8205 monotone True
     -> NOT GOLD FIRST

NOTE: "competitors at or above gold X/187" and "rank of gold among real finals (how many finals beat
gold)" are two different counts. The report.txt text prints only the X/187 combined count (real finals +
A20 variants, denominator 187); it does NOT separately print a real-finals-only beat-count. Extractor did
not compute this separately, per no-Python-analysis-in-session constraint; MISSING as a standalone number
(searched: report.txt full text, table.json not parsed further) -- the X/187 figure above is the closest
printed value and is reported verbatim.

Ladders K30/K70/K100 (seed means) per variant, from the same lines above:
  V0:  K30 -5.6097  K70 -6.3335  K100 -6.5597
  V1:  K30 -4.4822  K70 -4.7348  K100 -4.8280
  V2:  K30 -8.9447  K70 -10.0884  K100 -10.5523
  V12: K30 -7.8172  K70 -8.4897  K100 -8.8205

Best/worst stage-1 real final per variant is not printed as a separate explicit "best/worst real final"
line distinct from the "best competitor" (which is over reals+A20 combined); best competitor lines are
given above. Per-row J_V values for all 124 stage-1 real finals + 5 K30/K70/K100 seeds are in the ALL ROWS
table (report.txt lines 188-407); see table.json for full machine-readable values (193815 lines, not
individually enumerated here).

### Random derangements (3 seeds per selected key) -- mean/max, from ALL ROWS "der/" rows (lines 272-283)
(held-out J_V0 values shown; der rows = J_V for the 3 non-SIL derangement seeds per selected key)
  der/selected_1__a__s1 J_V0=-5.2980 s2=-5.5396 s3=-5.4357  -> mean=-5.4244, max=-5.2980
  der/selected_2__a__s1 J_V0=-5.4472 s2=-5.5963 s3=-5.6294  -> mean=-5.5576, max=-5.4472
  der/selected_3__a__s1 J_V0=-5.4296 s2=-5.5787 s3=-5.5501  -> mean=-5.5195, max=-5.4296
  der/selected_4__a__s1 J_V0=-5.5363 s2=-5.5046 s3=-5.3448  -> mean=-5.4619, max=-5.3448
(V1/V2/V12 der-row J_V columns are also printed per row in lines 272-283 of report.txt; not separately
summarized here for brevity -- see file for J_V1/J_V2/J_V12 columns on same der/ rows.)

The margin/rank numbers directly reported by the job's own reading rule are the "GOLD FIRST" block lines
above (item 2); those are the values registered as decisive by AN-1's reading rules.

## 3. NAMES VISIBLE block (report.txt lines 31-35)
Header: "NAMES VISIBLE (held-out J_V(b) - J_V(a) on the 4 selected keys; random renames J_V(der s1..s3) - J_V(a) beside)"

V0   sel1 -0.5734 [rand -0.7455, -0.9871, -0.8833]; sel2 -0.4359 [rand -0.8825, -1.0316, -1.0647];
     sel3 -0.4218 [rand -0.8625, -1.0116, -0.9831]; sel4 -0.4349 [rand -0.9608, -0.9291, -0.7693]
     printed, not read (J)
V1   sel1 -0.5734 [rand -0.7455, -0.9871, -0.8833]; sel2 -0.4359 [rand -0.8825, -1.0316, -1.0647];
     sel3 -0.4218 [rand -0.8625, -1.0116, -0.9831]; sel4 -0.4349 [rand -0.9608, -0.9291, -0.7693]
     printed, not read (V1's emission is name-free: equals J's)
V2   sel1 -0.8346 [rand -1.1482, -1.6786, -1.3772]; sel2 -0.6165 [rand -1.3998, -1.8101, -1.5622];
     sel3 -0.5566 [rand -1.4207, -1.6785, -1.3943]; sel4 -0.5843 [rand -1.4035, -1.5645, -1.1173]
     0/4 above 0.01 -> NOT VISIBLE
V12  sel1 -0.8346 [rand -1.1482, -1.6786, -1.3772]; sel2 -0.6165 [rand -1.3998, -1.8101, -1.5622];
     sel3 -0.5566 [rand -1.4207, -1.6785, -1.3943]; sel4 -0.5843 [rand -1.4035, -1.5645, -1.1173]
     0/4 above 0.01 -> NOT VISIBLE

## 4. Term decomposition (ALL ROWS table, report.txt lines 188-190, 206, 209, 212, 215)
Columns: J_V0  J_V1  J_V2  J_V12  lm  emis  emis_v1  occ  dur  rate  void  frame_id

gold                    J0=-4.8180 J1=-4.2123 J2=-7.9319 J12=-7.3262  lm=-0.9897 emis=-3.3468 emis_v1=-2.7411 occ=-3.1139 dur=-0.4815  rate=9.47 void=ok frame_id=1.0000
a20/selected_1__a       J0=-4.5525 J1=-4.1388 J2=-7.8093 J12=-7.3956  lm=-0.9484 emis=-3.1290 emis_v1=-2.7154 occ=-3.2568 dur=-0.4750  rate=9.20 void=ok frame_id=0.1137
a20/selected_2__a       J0=-4.5647 J1=-4.1349 J2=-7.8542 J12=-7.4245  lm=-0.9483 emis=-3.1364 emis_v1=-2.7067 occ=-3.2895 dur=-0.4800  rate=9.68 void=ok frame_id=0.1433
a20/selected_3__a       J0=-4.5671 J1=-4.0965 J2=-7.8838 J12=-7.4132  lm=-0.9457 emis=-3.1450 emis_v1=-2.6743 occ=-3.3168 dur=-0.4764  rate=9.28 void=ok frame_id=0.1237
a20/selected_4__a       J0=-4.5755 J1=-4.1257 J2=-7.8754 J12=-7.4256  lm=-0.9754 emis=-3.1223 emis_v1=-2.6724 occ=-3.2999 dur=-0.4779  rate=9.49 void=ok frame_id=0.0653

(occ = the occupancy term column; emis is with absorption, emis_v1 is without/name-free absorption per
docstring "V1's emission is name-free"; dur = duration term.)

## 5. TRAIN-SIDE EMISSION SPLIT (E9) (report.txt lines 37-52)
Convention line printed verbatim (line 37):
"TRAIN-SIDE EMISSION SPLIT (E9), nats per train frame.  Convention: EMIS = H(S') - H(U,S') = [H(S) - H(U)]
+ [H(S') - H(S)] - H(S'|U); S = k(U) unabsorbed, S' absorbed; EMIS_V1 = H(S) - H(U); ABSORPTION = EMIS -
(H(S) - H(U)).  Gaps: key minus gold."
H(U) = 6.0751 (line 38)

Columns: key | EMIS H(S)-H(U) ABSORB H(S) H(Sab) H(Sab|U) | dEMIS d(HS-HU) dABSORB dH(Sab) -dH(Sab|U)

gold                     EMIS=-3.3530 H(S)-H(U)=-2.7499 ABSORB=-0.6031 H(S)=3.3252 H(Sab)=3.3144 H(Sab|U)=0.5923 | dEMIS=+0.0000 d(HS-HU)=+0.0000 dABSORB=+0.0000 dH(Sab)=+0.0000 -dH(Sab|U)=-0.0000
a20/selected_1__a       EMIS=-3.1279 H(S)-H(U)=-2.7223 ABSORB=-0.4057 H(S)=3.3529 H(Sab)=3.3703 H(Sab|U)=0.4231 | dEMIS=+0.2251 d(HS-HU)=+0.0276 dABSORB=+0.1975 dH(Sab)=+0.0559 -dH(Sab|U)=+0.1692
a20/selected_2__a       EMIS=-3.1346 H(S)-H(U)=-2.7134 ABSORB=-0.4212 H(S)=3.3617 H(Sab)=3.3749 H(Sab|U)=0.4343 | dEMIS=+0.2184 d(HS-HU)=+0.0365 dABSORB=+0.1819 dH(Sab)=+0.0605 -dH(Sab|U)=+0.1579
a20/selected_3__a       EMIS=-3.1490 H(S)-H(U)=-2.6811 ABSORB=-0.4679 H(S)=3.3940 H(Sab)=3.4180 H(Sab|U)=0.4919 | dEMIS=+0.2040 d(HS-HU)=+0.0688 dABSORB=+0.1352 dH(Sab)=+0.1037 -dH(Sab|U)=+0.1003
a20/selected_4__a       EMIS=-3.1218 H(S)-H(U)=-2.6778 ABSORB=-0.4440 H(S)=3.3974 H(Sab)=3.4123 H(Sab|U)=0.4589 | dEMIS=+0.2312 d(HS-HU)=+0.0721 dABSORB=+0.1591 dH(Sab)=+0.0979 -dH(Sab|U)=+0.1333

a20/gold__a (identical to gold, listed not a competitor):
EMIS=-3.3530 H(S)-H(U)=-2.7499 ABSORB=-0.6031 H(S)=3.3252 H(Sab)=3.3144 H(Sab|U)=0.5923 | all deltas 0

## 6. Same-scale check line (report.txt line 23) and VOID/skip lines
"same-scale check (V0 = J): 628 comparisons (J, lm, emis, dur; train and held out) against A20
table.json / stage-1 search.json / stage-0 table.json, max |diff| 0.00e+00 <= 1e-06; J(gold) held out
rounds to -4.8180: True"

Counts of keys scored (report.txt line 3): "rows: 124 stage-1 real finals, 108 stage-1 null finals
(run-permuted corpus, null_seed 0), 22 A20 keys x (a, b, c), gold + K30/K70/K100 x 5 seeds, 4 selected
keys x 3 non-SIL derangements (seeds [1, 2, 3]); selected runs ['cluster_centroid_s01_warm',
'cluster_centroid_s04_warm', 'cluster_context_s01_warm', 'cluster_context_s04_warm']"

VOID/skip lines: every row in the ALL ROWS table (lines 188-515) has void column = "ok" for all rows
shown (gold, K30/K70/K100, a20/*, der/*, real/*, null/*); no row has a VOID/skip flag set. No explicit
"VOID" or "skip" text string found elsewhere in report.txt via inspection of lines 1-524. Docstring notes
(line 16-17): "A key's VOID flag (rate band [5.80, 14.49] Hz) is printed, not applied" -- but no row's
void column actually reads other than "ok" in this dump (searched: report.txt lines 188-515, void column
values all "ok").

Held-out set: 260 utterances (137933 frames), A13 disjoint set; train side: 28254 utterances
(15275716 frames); add-alpha 0.001 (report.txt line 2).
