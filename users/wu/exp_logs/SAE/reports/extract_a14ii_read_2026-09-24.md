# A14(ii) reader PiYQ1OCFD4ot extraction

Job dir: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_a14_jobs/A14ObjectiveFloorReadJob.PiYQ1OCFD4ot
(workspace alias: work/speech_llm/sae/emc/blankfree_a14_jobs/A14ObjectiveFloorReadJob.PiYQ1OCFD4ot)
Files: output/report.txt, output/a14_floor.json, info

## 1. Verdict and rule
VERDICT: PHONETIC BASIN LOWER
Rule (verbatim from report.txt lines 3-17): PHONETIC BASIN LOWER iff S_g < S_min - 0.01 AND gold-init Hungarian PER at 48 < 0.50; NON-PHONETIC PREFERRED iff S_g > S_min + 0.01 OR H >= 0.83; TIE otherwise.

## 2. S_g, S_min, difference (sub-epoch 48, 260 set, paired 260/260, excluded [])
S_g (gold init) = 3.21602 (json: 3.216019510880995)
S_min (lowest A10 restart) = 3.29903 [durinit_s01] (json: 3.299029975355225)
S_g - S_min = -0.08301 (json: -0.08301046447423); margin stated 0.01
registered S_min (A10 diagnostics, its own 260 paired tags) = 3.29903 [durinit_s01] (matches)
All six A10 restarts on paired set: uniform_s01 3.34756, uniform_s02 3.32244, durinit_s01 3.29903, durinit_s02 3.37032, durfrz_s01 3.36665, durfrz_s02 3.39956

## 3. Gold-init PER at 48
Hungarian PER (dev-other D4) = 0.3529 (json hungarian_per_gold_init: 0.3529471202425059)
Direct PER (ep48, trajectories) = 0.3529 (equal to Hungarian at ep48)

## 4. r30/r70/r100 at 48 and ep0 (report only)
r30  ep48: S260=3.20982 (285: 3.21680); PER direct 0.3941, Hungarian 0.3941, NMI 0.6709
r30  ep0:  S260=3.71916 (285: 3.71978); PER direct 0.2404, Hungarian 0.2404, NMI 0.8013
r70  ep48: S260=3.27068 (285: 3.27546); PER direct 0.4949, Hungarian 0.5076, NMI 0.5655
r70  ep0:  S260=4.41132 (285: 4.40844); PER direct 0.6083, Hungarian 0.6282, NMI 0.4101
r100 ep48: S260=3.37793 (285: 3.38314); PER direct 0.8601, Hungarian 0.8612, NMI 0.0704
r100 ep0:  S260=4.69008 (285: 4.68612); PER direct 0.8261, Hungarian 0.8304, NMI 0.1124
(PER/Hungarian/NMI also printed at diagnostics sub-epochs 4,8,...,44 for all inits in report.txt; gold ep4-44 in report.txt lines 52-100 similarly for r30/r70/r100 lines 101-250.)

## 5. Diagnostics job input
Reader's actual input: PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO (grep of info file; input dir entry `speech_llm_sae_emc_blankfree_a13_disjoint_PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO`).
The job PhiFirstA10DiagnosticsDisjointJob.8S8gTgbRFKU6 exists on disk at work/speech_llm/sae/emc/blankfree_a13_disjoint/PhiFirstA10DiagnosticsDisjointJob.8S8gTgbRFKU6 (and a .cleared.0001 copy) but is NOT among this reader's inputs (0 hits for "8S8gTgbRFKU6" in info file) — different hash from the one actually consumed (G2NeV8oNr4tO).
S_min (registered) and the paired 260-set A10 values come from PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO's a13 block (per IMPLEMENTATION note in report.txt).
Gold-init PER at 48 comes from GenDecodeReportJob.n2EIsGktK1Zy/output/report.json (per_hungarian.per; the sub-epoch-48 entry of the 'gold' reports dict in info, PARAMETER: reports).

## 6. Training job dirs
Not found: the four one-GPU restart training jobs (gold, r30, r70, r100) are NOT direct inputs of this reader. The reader's only inputs are ReturnnForwardJobV2 (452 dirs, decode/forward dumps) and GenDecodeReportJob outputs plus PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO; no ReturnnTrainingJob or similarly-named training-job class/hash appears in info or input/. MISSING (searched: info file for "Train"/job-class patterns, input/ directory listing — only forward/report/diagnostics job classes present). The gold init job is referenced only by its L2-0 fit name/hash "16v7R6ztSq1u" in the rule text, not as a job path in this reader's info/input.

Report path: /e/project1/spell/wu24/2026-07-13_unsupervised/reports/extract_a14ii_read_2026-09-24.md
