# A19 trigram-only lift ladder read - extraction

Job: A19TriLadderReadJob.5ny4LPLFrIXw at
/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_a19_jobs/A19TriLadderReadJob.5ny4LPLFrIXw
Outputs: output/a19_triladder.json, output/report.txt

## 1. Verdict
VERDICT: SAME LADDER
rho*_tri = 0.7, monotone = True
READING-SENSITIVE: no (all 4 verb-reading combinations give rho_star 0.7 / SAME LADDER)

## 2. Per-arm PER (dev-other greedy)
arm       ep1     ep2     ep4     ep8     class(ep8)
tri_r30   0.2029445776336201  0.2817656183895078  0.2982202792271894  0.2676632350867297  LIFT
tri_r50   0.20980397687209137 0.2773882386123255  0.2999407699901283  0.263065858130024   LIFT
tri_r70   0.33445212240868705 0.36797066704273024 0.3535298265406854  0.3292511634466225  LIFT
tri_r100  0.8306303765336341  0.8987956564659427  0.9030997038499506  0.8981356649273727  NO LIFT

## 3. Paired rows tri_rX minus rt_rX (A=rt baseline, B=tri candidate), M=0.010
rung  ep  delta_tri_minus_rt        95% CI                              class
r30   1   0.016821322803553806      [0.015575803979159739, 0.017952383355306165]  K2 HELPS
r30   8   0.08502044845578904       [0.08064236301826411, 0.08986847108510637]    K2 HELPS
r50   1   0.019607953744182754      [0.01798429336028143, 0.021173947615447306]   K2 HELPS
r50   8   0.07673952898039768       [0.07320152582306266, 0.08049533824321987]    K2 HELPS
r70   1   0.08114511352418557       [0.07580294368120695, 0.08657951414126029]    K2 HELPS
r70   8   0.13835002115357495       [0.1328093367834937, 0.14435004377590413]     K2 HELPS
r100  1   -0.013820335636722691     [-0.01676089637162953, -0.011183988097379043] TRIGRAM ENOUGH
r100  8   0.0429671414469045        [0.036177556440887305, 0.050602819898021026]  K2 HELPS
(utterances = 2864 for all rows)

## 4. Phi generative PER at ep8 (D4 dev-other, 500 utterances)
arm       per_direct           per_hungarian         nmi_symbol_phone
tri_r30   0.2598518019535197   0.2598518019535197    0.7875479894629059
tri_r50   0.256517345907713    0.256517345907713     0.7837326491712496
tri_r70   0.3163691478612327   0.35661839003031326   0.7248942344695689
tri_r100  0.8734590771303469   0.8759514988211519    0.06776366783874871

## 5. rt_rX reference (ep8 PER) used for pairing, source: PairedPerDeltaJob's "per_rt" field (L2-0's banked decode)
r30_ep8:  per_rt = 0.18264278663094063 (r30_ep1: 0.18612325483006628)
r50_ep8:  per_rt = 0.1863263291496263  (r50_ep1: 0.19019602312790862)
r70_ep8:  per_rt = 0.19090114229304753 (r70_ep1: 0.25330700888450147)
r100_ep8: per_rt = 0.8551685234804682 (r100_ep1: 0.8444507121703568)
(exact rt_rX source PairedPerDeltaJob hashes: not read in this pass -> MISSING (searched: a19_triladder.json paired_vs_rt fields only, no separate job hash reference field in json))

## 6. Warnings / NaN / checkpoints
Checkpoints epoch.001.pt / epoch.002.pt / epoch.004.pt / epoch.008.pt confirmed present for all 4 arms (tri_r30, tri_r50, tri_r70, tri_r100) under
/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.DzrmcjOQ4I3r/output/<arm>/models/.
grep -icE "nan|inf" on all 4 log.run.1 files = 0 (no NaN/inf occurrences). grep -icE "warn|error" = 26 hits in each of the 4 log.run.1 files (not individually inspected for content in this bounded pass).
