# AN-2 / AN-4 extraction (2026-09-25)

Sources:
- AN2: work/speech_llm/sae/emc/rename_an24_jobs/An2ReadJob.91eeK6mtb6UH/output/report.txt
- AN4: work/speech_llm/sae/emc/rename_an24_jobs/An4ReadJob.c75KVY92wtjP/output/report.txt

## 1. Reading rules (verbatim, cited by line range)
- H2: see report.txt "H2 REFUTED if S_1(gold-key) < S_1(full derangement) - 0.1 on all 3 seeds; H2 SUPPORTED otherwise" (AN2 READING RULES block).
- S SEES NAMES / PREFERS FOUND NAMES / NAME-BLIND: "S SEES NAMES ON FOUND PARTITIONS if S_1(b) - S_1(a) is below the minimum over the 3 random renames minus 0.01 on at least 3 of 4 keys; S PREFERS FOUND NAMES if S_1(b) > S_1(a) + 0.01 on at least 3 of 4; S NAME-BLIND ON FOUND PARTITIONS if S_1(b) lies inside the random-rename band on at least 3 of 4; else MIXED."
- LAMBDA KEEPS THE BASIN / FAVOURS FINALS: "KEEPS iff lead > 0 at 1, 2, 4.4 and 10; FAVOURS FINALS iff lead is strictly decreasing over 1<2<4.4<10 and lead(10)<0; NEITHER otherwise."
- P1a/P1b/P1c: "P1a: every final has a higher channel term per frame. P1b: every final has a lower LM term per phone. P1c: every final has a lower expected non-SIL rate."
- H3' (private code): "P1a and P1c with P1b failing (the finals win the LM per phone) is the private-code outcome H3'; it contraindicates TP-A1 and TP-B swaps."
- P2: "the LM term per frame differs between finals and basin by less than half the paired S gap of the same phis" -> "P2 iff |D_LM| < 0.5 x |D_S|."
- P3: "HOLDS on a trajectory iff PER(48) > PER(0), channel(48) > channel(0) and LM per phone(48) < LM per phone(0); FAILS iff PER rose and either term moved the other way; NOT TESTABLE iff PER did not rise. P3 HOLDS iff it holds on all three; FAILS iff it fails on any; NOT TESTABLE otherwise."

## 2. AN-2 numbers
H_LM = 2.257108 nats/token (counted lines, 81559944 tokens; held lines 2.257698).

H2 (lambda=1): S_1(gold_key) = 4.57305. gold_key/full derangements: full_s1 5.11291 (+0.5399), full_s2 5.08628 (+0.5132), full_s3 5.14031 (+0.5673) vs threshold +0.1 -> True/True/True -> H2 REFUTED.

Found-partition names (lambda=1), d=S1(b)-S1(a), random-rename band:
- key1: a=4.39305, b=4.74367, d=+0.3506, randoms +0.6243/+0.6669/+0.6776
- key2: a=4.42365, b=4.67443, d=+0.2508, randoms +0.6243/+0.6120/+0.7436
- key3: a=4.41255, b=4.70419, d=+0.2916, randoms +0.6183/+0.6464/+0.7205
- key4: a=4.40882, b=4.68719, d=+0.2784, randoms +0.6643/+0.5866/+0.6823
counts: sees 4/4, prefers found 4/4, blind 0/4.

Basin lead per lambda, rate-neutral (R1): lambda1 +0.07516 [+0.0635,+0.0873] (min finals durinit_s01 3.29903, max basin g_dur_ep48 3.22387); lambda2 +0.16558 [+0.1532,+0.1781]; lambda4.4 +0.25966 [+0.2432,+0.2757]; lambda10 +0.27174 [+0.2489,+0.2947]. KEEPS=True, FAVOURS FINALS=False.
Plain form beside: lambda1 +0.07516; lambda2 +0.08111; lambda4.4 -0.05254; lambda10 -0.48123 -> "would read NEITHER (beside only)".

## 3. AN-4, six finals + durinit basin at 48 (chan/frame, LM/frame, LM/phone, H(q), non-SIL rate Hz, SIL share)
Finals: uniform_s01 chan -2.99805 LM -0.63223 LM/phone -3.9026 H 0.28272 rate 6.927 SIL 0.0698; uniform_s02 chan -2.97918 LM -0.62357 LM/phone -3.8134 H 0.28031 rate 6.976 SIL 0.0714; durinit_s01 chan -2.91752 LM -0.66267 LM/phone -3.9688 H 0.28115 rate 7.132 SIL 0.0595; durinit_s02 chan -3.01610 LM -0.63902 LM/phone -3.9867 H 0.28480 rate 6.854 SIL 0.0659; durfrz_s01 chan -2.98164 LM -0.67337 LM/phone -3.8087 H 0.28836 rate 7.562 SIL 0.0612; durfrz_s02 chan -3.04909 LM -0.64255 LM/phone -3.8516 H 0.29208 rate 7.132 SIL 0.0679.
Basin: gold_key chan -2.85979 LM -0.62080 LM/phone -3.1604 H 0.27355 rate 8.416 SIL 0.0769; g_dur chan -2.87719 LM -0.62316 LM/phone -3.1237 H 0.27648 rate 8.549 SIL 0.0483; r30_dur chan -2.86058 LM -0.62804 LM/phone -3.1631 H 0.27712 rate 8.525 SIL 0.0539.

P1a: finals channel range [-3.04909,-2.91752] vs basin [-2.87719,-2.85979] -> False.
P1b: finals LM/phone [-3.9867,-3.8087] vs basin [-3.1631,-3.1237] -> True (finals win outright: False).
P1c: finals rate [6.854,7.562] vs basin [8.416,8.549] Hz -> True.
Private code: False.
P2: D_LM/frame (finals-basin) -0.0216 [-0.0265,-0.0167]; D_S +0.1368 [+0.1290,+0.1452]; |D_LM|<0.5|D_S| -> True.

P3 trajectories (0->48, Hungarian PER):
- keyinit/gold_key: dPER +0.0059, dchan +1.30377, dLM/phone -0.1421 -> HOLDS
- keyinit/g_dur: dPER +0.1476, dchan +0.33121, dLM/phone -0.3262 -> HOLDS
- keyinit/r30_dur: dPER +0.1202, dchan +0.64963, dLM/phone -0.3581 -> HOLDS
-> P3 HOLDS (all three)

## 4. Generative PER lines (dev-other, Hungarian PER given, from report.txt "=== the six finals ===" / "=== the durinit basin set ===")
Finals (direct/Hung/NMI): uniform_s01 0.8487/0.8607/0.0860; uniform_s02 0.8325/0.8436/0.1024; durinit_s01 0.8533/0.8615/0.0788; durinit_s02 0.8315/0.8380/0.1131; durfrz_s01 0.8564/0.8570/0.0784; durfrz_s02 0.8392/0.8478/0.0984.
Basin at 48: gold_key 0.3457/0.3907/0.6945; g_dur 0.3447/0.3447/0.7099; r30_dur 0.3681/0.3681/0.6847.
