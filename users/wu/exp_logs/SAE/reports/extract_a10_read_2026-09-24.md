# A10 read extraction (2026-09-24)

## Output files read
- work/speech_llm/sae/emc/blankfree_phifirst_a10_read/PhiFirstA10ReadJob.DcfCsZNq1ucr/output/report.txt (resolves to /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_phifirst_a10_read/PhiFirstA10ReadJob.DcfCsZNq1ucr/output/report.txt)
- work/speech_llm/sae/emc/blankfree_phifirst_a10_read/PhiFirstA10ReadJob.DcfCsZNq1ucr/output/a10_read.json (present, not opened; report.txt is the printed reader output)
- work/speech_llm/sae/emc/blankfree_a13_disjoint/PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO/output/report.txt (resolves to /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_a13_disjoint/PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO/output/report.txt)
- work/speech_llm/sae/emc/blankfree_a13_disjoint/PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO/output/a10_diagnostics.json (present, not opened)

Note: the `work/` symlink in the setup dir resolves to /e/scratch/spell/wu24/2026-07-13_unsupervised/work. A separate, distinct `work/` directory exists at /e/scratch/spell/wu24/project-relocated/2026-07-13_unsupervised/work but does not contain these A10 jobs.

## 1. Reader verdicts (verbatim, PhiFirstA10ReadJob report.txt)

Line 66: `setting durinit (candidate): K* K* 12 (wave count if chosen 12); rate in band at 48: {'1': True, '2': True}`
Line 165: `setting durfrz (candidate): K* K* 10 (wave count if chosen 12); rate in band at 48: {'1': True, '2': True}`
Line 264: `setting uniform (report only): K* K* 10 (wave count if chosen 12); rate in band at 48: {'1': True, '2': True}`

Line 363: `comparison at sub-epoch 12 (largest K* among the candidates with a K*; paired over 285 utt): durinit 3.42165, durfrz 3.45707`

Line 389: `max |dS|: {'where': 'durinit/s2/ep4', 'value': 3.449235842678533e-05}`

Line 390: `VERDICT: WAVE SETTING durinit, 12 sub-epochs (K* 12)`
Line 391: `WAVE_DURATION_SETTING = 'durinit'; WAVE_NUM_SUBEPOCHS = 12 (nulls and phi_c arms run the same count)`

No NOT CONVERGED / NO WAVE SETTING / TIE / CANNOT_TELL line is printed; the file ends with the VERDICT/WAVE_DURATION_SETTING lines above.

## 2. Per-restart S and emitted-rate trajectory (every 4th sub-epoch 4..48 + gain at last few)

All S values are "S (285 utt)" (285-utterance CV holdout, own possible utterances), as printed in report.txt.

### durinit, seed 1
ep4: S 3.68303, gain 0.28502, emitted 7.835 Hz
ep8: S 3.46069, gain 0.01640, emitted 7.475 Hz
ep12: S 3.39810, gain 0.00534 ok True, emitted 7.321 Hz
ep16: S 3.37397, gain 0.00457, emitted 7.265 Hz
ep20: S 3.35890, gain -0.00086, emitted 7.249 Hz
ep24: S 3.34073, gain 0.00652, emitted 7.229 Hz
ep28: S 3.32628, gain 0.00736, emitted 7.223 Hz
ep32: S 3.31995, gain 0.00377, emitted 7.173 Hz
ep36: S 3.31821, gain 0.00011, emitted 7.168 Hz
ep40: S 3.30952, gain 0.00873, emitted 7.154 Hz
ep44: S 3.31044, gain -0.00016, emitted 7.156 Hz
ep48 (final): S 3.30403, gain 0.00639 ok True, emitted 7.134 Hz
Last few gains (ep44-48): -0.00016, -0.00009 (ep45), 0.00161 (ep46), -0.00148 (ep47), 0.00639 (ep48)

### durinit, seed 2
ep4: S 3.70926, gain 0.27875, emitted 7.473 Hz
ep8: S 3.48960, gain 0.01549, emitted 7.154 Hz
ep12: S 3.44519, gain 0.00889 ok True, emitted 7.055 Hz
ep16: S 3.42384, gain 0.00130, emitted 6.992 Hz
ep20: S 3.41193, gain 0.00280, emitted 6.954 Hz
ep24: S 3.40414, gain 0.00136, emitted 6.903 Hz
ep28: S 3.39503, gain 0.00119, emitted 6.869 Hz
ep32: S 3.39371, gain -0.00217, emitted 6.874 Hz
ep36: S 3.38232, gain 0.00748, emitted 6.862 Hz
ep40: S 3.37928, gain 0.00179, emitted 6.834 Hz
ep44: S 3.37290, gain 0.00438, emitted 6.851 Hz
ep48 (final): S 3.37600, gain -0.00174, emitted 6.821 Hz
Last few gains (ep44-48): 0.00438, -0.00089 (ep45), -0.00273 (ep46), 0.00225 (ep47), -0.00174 (ep48)

### durfrz, seed 1
ep4: S 3.71740, gain 0.29915, emitted 7.587 Hz
ep8: S 3.50029, gain 0.01562, emitted 7.563 Hz
ep12: S 3.44490, gain 0.00283 ok True, emitted 7.589 Hz
ep16: S 3.41764, gain 0.00470, emitted 7.544 Hz
ep20: S 3.40619, gain -0.00121, emitted 7.584 Hz
ep24: S 3.39723, gain 0.00533, emitted 7.618 Hz
ep28: S 3.38877, gain 0.00063, emitted 7.624 Hz
ep32: S 3.38570, gain -0.00030, emitted 7.627 Hz
ep36: S 3.38341, gain -0.00064, emitted 7.600 Hz
ep40: S 3.37502, gain 0.00676, emitted 7.596 Hz
ep44: S 3.37381, gain 0.00208, emitted 7.584 Hz
ep48 (final): S 3.37201, gain 0.00236, emitted 7.573 Hz
Last few gains (ep44-48): 0.00208, -0.00081 (ep45), -0.00011 (ep46), 0.00036 (ep47), 0.00236 (ep48)

### durfrz, seed 2
ep4: S 3.73945, gain 0.29611, emitted 7.417 Hz
ep8: S 3.51287, gain 0.01745, emitted 7.257 Hz
ep12: S 3.46924, gain 0.00846 ok True, emitted 7.246 Hz
ep16: S 3.44989, gain 0.00338, emitted 7.217 Hz
ep20: S 3.43846, gain 0.00369, emitted 7.201 Hz
ep24: S 3.42647, gain 0.00470, emitted 7.139 Hz
ep28: S 3.42421, gain 0.00191, emitted 7.137 Hz
ep32: S 3.42019, gain -0.00131, emitted 7.135 Hz
ep36: S 3.41218, gain 0.00252, emitted 7.122 Hz
ep40: S 3.40883, gain 0.00283, emitted 7.141 Hz
ep44: S 3.40424, gain 0.00512, emitted 7.118 Hz
ep48 (final): S 3.40582, gain 0.00400, emitted 7.114 Hz
Last few gains (ep44-48): 0.00512, -0.00309 (ep45), -0.00049 (ep46), -0.00201 (ep47), 0.00400 (ep48)

### uniform (report only), seed 1
ep4: S 3.75955, gain 0.31440, emitted 6.527 Hz
ep8: S 3.50297, gain 0.02514, emitted 6.713 Hz
ep12: S 3.43765, gain 0.00704 ok True, emitted 6.820 Hz
ep16: S 3.41226, gain 0.01182, emitted 6.893 Hz
ep20: S 3.39712, gain 0.00389, emitted 6.907 Hz
ep24: S 3.38619, gain 0.00508, emitted 6.921 Hz
ep28: S 3.37562, gain 0.00268, emitted 6.949 Hz
ep32: S 3.37066, gain 0.00036, emitted 6.967 Hz
ep36: S 3.36747, gain -0.00144, emitted 6.944 Hz
ep40: S 3.35612, gain 0.00896, emitted 6.927 Hz
ep44: S 3.35787, gain -0.00006, emitted 6.943 Hz
ep48 (final): S 3.35049, gain 0.00446, emitted 6.914 Hz
Last few gains (ep44-48): -0.00006, 0.00024 (ep45), -0.00012 (ep46), 0.00280 (ep47), 0.00446 (ep48)

### uniform (report only), seed 2
ep4: S 3.70946, gain 0.33290, emitted 6.743 Hz
ep8: S 3.44786, gain 0.01990, emitted 6.940 Hz
ep12: S 3.38858, gain 0.00623 ok True, emitted 6.932 Hz
ep16: S 3.36623, gain 0.00044, emitted 6.919 Hz
ep20: S 3.35432, gain -0.00031, emitted 6.973 Hz
ep24: S 3.34422, gain 0.00412, emitted 6.950 Hz
ep28: S 3.34626, gain -0.00579, emitted 6.942 Hz
ep32: S 3.33663, gain 0.00050, emitted 6.967 Hz
ep36: S 3.33156, gain 0.00277, emitted 6.963 Hz
ep40: S 3.33042, gain 0.00068, emitted 6.957 Hz
ep44: S 3.32448, gain 0.00465, emitted 6.958 Hz
ep48 (final): S 3.32768, gain -0.00376, emitted 6.971 Hz
Last few gains (ep44-48): 0.00465, -0.00113 (ep45), 0.00112 (ep46), 0.00057 (ep47), -0.00376 (ep48)

## 3. Identity band: sub-epochs 1-4, extension vs probe (as printed, report.txt lines 364-389)

durinit/s1/ep1: dS -0.000000 (285 utt); rate ext 1.360 / probe 1.360 Hz
durinit/s1/ep2: dS -0.000001 (285 utt); rate ext 6.718 / probe 6.718 Hz
durinit/s1/ep3: dS -0.000014 (285 utt); rate ext 7.694 / probe 7.694 Hz
durinit/s1/ep4: dS -0.000024 (285 utt); rate ext 7.835 / probe 7.835 Hz
durinit/s2/ep1: dS 0.000000 (285 utt); rate ext 1.413 / probe 1.413 Hz
durinit/s2/ep2: dS -0.000000 (285 utt); rate ext 6.739 / probe 6.739 Hz
durinit/s2/ep3: dS 0.000001 (285 utt); rate ext 7.495 / probe 7.495 Hz
durinit/s2/ep4: dS 0.000034 (285 utt); rate ext 7.473 / probe 7.471 Hz
durfrz/s1/ep1: dS -0.000000 (285 utt); rate ext 1.307 / probe 1.307 Hz
durfrz/s1/ep2: dS 0.000000 (285 utt); rate ext 6.677 / probe 6.677 Hz
durfrz/s1/ep3: dS -0.000002 (285 utt); rate ext 7.669 / probe 7.669 Hz
durfrz/s1/ep4: dS -0.000006 (285 utt); rate ext 7.587 / probe 7.588 Hz
durfrz/s2/ep1: dS 0.000000 (285 utt); rate ext 1.373 / probe 1.373 Hz
durfrz/s2/ep2: dS -0.000000 (285 utt); rate ext 6.758 / probe 6.759 Hz
durfrz/s2/ep3: dS -0.000010 (285 utt); rate ext 7.464 / probe 7.465 Hz
durfrz/s2/ep4: dS -0.000001 (285 utt); rate ext 7.417 / probe 7.416 Hz
uniform/s1/ep1: dS 0.000000 (285 utt); rate ext 0.978 / probe 0.978 Hz
uniform/s1/ep2: dS -0.000000 (285 utt); rate ext 4.272 / probe 4.272 Hz
uniform/s1/ep3: dS -0.000000 (285 utt); rate ext 6.170 / probe 6.170 Hz
uniform/s1/ep4: dS -0.000004 (285 utt); rate ext 6.527 / probe 6.527 Hz
uniform/s2/ep1: dS -0.000000 (285 utt); rate ext 0.931 / probe 0.931 Hz
uniform/s2/ep2: dS 0.000000 (285 utt); rate ext 4.566 / probe 4.566 Hz
uniform/s2/ep3: dS 0.000009 (285 utt); rate ext 6.340 / probe 6.340 Hz
uniform/s2/ep4: dS -0.000023 (285 utt); rate ext 6.743 / probe 6.743 Hz

max |dS|: {'where': 'durinit/s2/ep4', 'value': 3.449235842678533e-05}

## 4. Diagnostics (PhiFirstA10DiagnosticsDisjointJob report.txt): every 4 sub-epochs, direct PER, Hungarian PER, NMI(symbol,phone), E[d], 500-utt dev-other set

Chance band for NMI-adjacent quantity printed: PER direct/Hungarian labelled "(IN)" against chance band [0.83, 0.91] per the docstring text (line 34).

### uniform seed 1
ep4: PER direct 0.8401 (IN), Hungarian 0.8476 (IN), NMI(symbol,phone) 0.0926; S 3.75955; E[d] 9.952
ep8: PER direct 0.8419 (IN), Hungarian 0.8520 (IN), NMI 0.0899; S 3.50297; E[d] 7.903
ep12: PER direct 0.8437 (IN), Hungarian 0.8529 (IN), NMI 0.0914; S 3.43765; E[d] 7.110
ep16: PER direct 0.8437 (IN), Hungarian 0.8525 (IN), NMI 0.0901; S 3.41226; E[d] 6.703
ep20: PER direct 0.8439 (IN), Hungarian 0.8533 (IN), NMI 0.0884; S 3.39712; E[d] 6.504
ep24: PER direct 0.8442 (IN), Hungarian 0.8507 (IN), NMI 0.0882; S 3.38619; E[d] 6.393
ep28: PER direct 0.8473 (IN), Hungarian 0.8528 (IN), NMI 0.0857; S 3.37562; E[d] 6.315
ep32: PER direct 0.8467 (IN), Hungarian 0.8527 (IN), NMI 0.0871; S 3.37066; E[d] 6.261
ep36: PER direct 0.8470 (IN), Hungarian 0.8572 (IN), NMI 0.0871; S 3.36747; E[d] 6.231
ep40: PER direct 0.8484 (IN), Hungarian 0.8605 (IN), NMI 0.0847; S 3.35612; E[d] 6.198
ep44: PER direct 0.8486 (IN), Hungarian 0.8606 (IN), NMI 0.0851; S 3.35787; E[d] 6.189
ep48: PER direct 0.8487 (IN), Hungarian 0.8607 (IN), NMI 0.0860; S 3.35049; E[d] 6.187

### uniform seed 2
ep4: PER direct 0.8381 (IN), Hungarian 0.8485 (IN), NMI 0.0969; S 3.70946; E[d] 9.936
ep8: PER direct 0.8323 (IN), Hungarian 0.8426 (IN), NMI 0.1038; S 3.44786; E[d] 7.847
ep12: PER direct 0.8315 (IN), Hungarian 0.8383 (IN), NMI 0.1022; S 3.38858; E[d] 7.079
ep16: PER direct 0.8313 (IN), Hungarian 0.8340 (IN), NMI 0.1020; S 3.36623; E[d] 6.771
ep20: PER direct 0.8308 (IN), Hungarian 0.8344 (IN), NMI 0.1024; S 3.35432; E[d] 6.599
ep24: PER direct 0.8316 (IN), Hungarian 0.8400 (IN), NMI 0.1021; S 3.34422; E[d] 6.492
ep28: PER direct 0.8321 (IN), Hungarian 0.8361 (IN), NMI 0.1023; S 3.34626; E[d] 6.426
ep32: PER direct 0.8330 (IN), Hungarian 0.8410 (IN), NMI 0.1020; S 3.33663; E[d] 6.380
ep36: PER direct 0.8317 (IN), Hungarian 0.8403 (IN), NMI 0.1020; S 3.33156; E[d] 6.349
ep40: PER direct 0.8314 (IN), Hungarian 0.8434 (IN), NMI 0.1019; S 3.33042; E[d] 6.335
ep44: PER direct 0.8321 (IN), Hungarian 0.8437 (IN), NMI 0.1017; S 3.32448; E[d] 6.310
ep48: PER direct 0.8325 (IN), Hungarian 0.8436 (IN), NMI 0.1024; S 3.32768; E[d] 6.296

### durinit seed 1
ep4: PER direct 0.8624 (IN), Hungarian 0.8690 (IN), NMI 0.0675; S 3.68303; E[d] 4.856
ep8: PER direct 0.8564 (IN), Hungarian 0.8538 (IN), NMI 0.0766; S 3.46069; E[d] 5.329
ep12: PER direct 0.8533 (IN), Hungarian 0.8627 (IN), NMI 0.0785; S 3.39810; E[d] 5.616
ep16: PER direct 0.8524 (IN), Hungarian 0.8572 (IN), NMI 0.0803; S 3.37397; E[d] 5.769
ep20: PER direct 0.8523 (IN), Hungarian 0.8620 (IN), NMI 0.0804; S 3.35890; E[d] 5.842
ep24: PER direct 0.8525 (IN), Hungarian 0.8621 (IN), NMI 0.0794; S 3.34073; E[d] 5.889
ep28: PER direct 0.8531 (IN), Hungarian 0.8631 (IN), NMI 0.0810; S 3.32628; E[d] 5.928
ep32: PER direct 0.8534 (IN), Hungarian 0.8623 (IN), NMI 0.0801; S 3.31995; E[d] 5.956
ep36: PER direct 0.8534 (IN), Hungarian 0.8621 (IN), NMI 0.0805; S 3.31821; E[d] 5.976
ep40: PER direct 0.8535 (IN), Hungarian 0.8616 (IN), NMI 0.0785; S 3.30952; E[d] 5.989
ep44: PER direct 0.8526 (IN), Hungarian 0.8613 (IN), NMI 0.0802; S 3.31044; E[d] 6.004
ep48: PER direct 0.8533 (IN), Hungarian 0.8615 (IN), NMI 0.0788; S 3.30403; E[d] 6.012

### durinit seed 2
ep4: PER direct 0.8327 (IN), Hungarian 0.8449 (IN), NMI 0.1047; S 3.70926; E[d] 4.937
ep8: PER direct 0.8318 (IN), Hungarian 0.8421 (IN), NMI 0.1109; S 3.48960; E[d] 5.464
ep12: PER direct 0.8308 (IN), Hungarian 0.8404 (IN), NMI 0.1118; S 3.44519; E[d] 5.717
ep16: PER direct 0.8315 (IN), Hungarian 0.8407 (IN), NMI 0.1145; S 3.42384; E[d] 5.839
ep20: PER direct 0.8307 (IN), Hungarian 0.8394 (IN), NMI 0.1161; S 3.41193; E[d] 5.922
ep24: PER direct 0.8310 (IN), Hungarian 0.8411 (IN), NMI 0.1134; S 3.40414; E[d] 5.992
ep28: PER direct 0.8307 (IN), Hungarian 0.8403 (IN), NMI 0.1137; S 3.39503; E[d] 6.047
ep32: PER direct 0.8301 (IN), Hungarian 0.8415 (IN), NMI 0.1150; S 3.39371; E[d] 6.087
ep36: PER direct 0.8304 (IN), Hungarian 0.8405 (IN), NMI 0.1147; S 3.38232; E[d] 6.117
ep40: PER direct 0.8303 (IN), Hungarian 0.8383 (IN), NMI 0.1158; S 3.37928; E[d] 6.151
ep44: PER direct 0.8309 (IN), Hungarian 0.8380 (IN), NMI 0.1139; S 3.37290; E[d] 6.178
ep48: PER direct 0.8315 (IN), Hungarian 0.8380 (IN), NMI 0.1131; S 3.37600; E[d] 6.205

### durfrz seed 1
ep4: PER direct 0.8541 (IN), Hungarian 0.8617 (IN), NMI 0.0730; S 3.71740; E[d] 4.414
ep8: PER direct 0.8571 (IN), Hungarian 0.8570 (IN), NMI 0.0755; S 3.50029; E[d] 4.414
ep12: PER direct 0.8558 (IN), Hungarian 0.8598 (IN), NMI 0.0771; S 3.44490; E[d] 4.414
ep16: PER direct 0.8555 (IN), Hungarian 0.8576 (IN), NMI 0.0772; S 3.41764; E[d] 4.414
ep20: PER direct 0.8564 (IN), Hungarian 0.8563 (IN), NMI 0.0785; S 3.40619; E[d] 4.414
ep24: PER direct 0.8575 (IN), Hungarian 0.8599 (IN), NMI 0.0778; S 3.39723; E[d] 4.414
ep28: PER direct 0.8579 (IN), Hungarian 0.8563 (IN), NMI 0.0785; S 3.38877; E[d] 4.414
ep32: PER direct 0.8577 (IN), Hungarian 0.8560 (IN), NMI 0.0778; S 3.38570; E[d] 4.414
ep36: PER direct 0.8564 (IN), Hungarian 0.8549 (IN), NMI 0.0800; S 3.38341; E[d] 4.414
ep40: PER direct 0.8560 (IN), Hungarian 0.8555 (IN), NMI 0.0788; S 3.37502; E[d] 4.414
ep44: PER direct 0.8567 (IN), Hungarian 0.8553 (IN), NMI 0.0799; S 3.37381; E[d] 4.414
ep48: PER direct 0.8564 (IN), Hungarian 0.8570 (IN), NMI 0.0784; S 3.37201; E[d] 4.414

### durfrz seed 2
ep4: PER direct 0.8393 (IN), Hungarian 0.8515 (IN), NMI 0.0957; S 3.73945; E[d] 4.414
ep8: PER direct 0.8362 (IN), Hungarian 0.8483 (IN), NMI 0.1015; S 3.51287; E[d] 4.414
ep12: PER direct 0.8386 (IN), Hungarian 0.8419 (IN), NMI 0.1013; S 3.46924; E[d] 4.414
ep16: PER direct 0.8369 (IN), Hungarian 0.8408 (IN), NMI 0.1024; S 3.44989; E[d] 4.414
ep20: PER direct 0.8390 (IN), Hungarian 0.8429 (IN), NMI 0.1022; S 3.43846; E[d] 4.414
ep24: PER direct 0.8384 (IN), Hungarian 0.8418 (IN), NMI 0.1004; S 3.42647; E[d] 4.414
ep28: PER direct 0.8378 (IN), Hungarian 0.8414 (IN), NMI 0.1015; S 3.42421; E[d] 4.414
ep32: PER direct 0.8387 (IN), Hungarian 0.8431 (IN), NMI 0.1023; S 3.42019; E[d] 4.414
ep36: PER direct 0.8379 (IN), Hungarian 0.8487 (IN), NMI 0.1009; S 3.41218; E[d] 4.414
ep40: PER direct 0.8385 (IN), Hungarian 0.8489 (IN), NMI 0.0985; S 3.40883; E[d] 4.414
ep44: PER direct 0.8390 (IN), Hungarian 0.8496 (IN), NMI 0.0989; S 3.40424; E[d] 4.414
ep48: PER direct 0.8392 (IN), Hungarian 0.8478 (IN), NMI 0.0984; S 3.40582; E[d] 4.414

### Reference S values (as printed)
- reference S (CV holdout, all 285, own possible utterances, SECONDARY, diagnostics report line 38): gold 3.47024 (285 utt), r100 4.68612 (285 utt)
- reference S (CV holdout, A13 260-set PRIMARY / 285-set SECONDARY, diagnostics report line 22): gold 3.47351 (260 PRIMARY) / 3.47024 (285 SECONDARY); r100 4.69008 (260 PRIMARY) / 4.68612 (285 SECONDARY)

### A13 260-disjoint-set (PRIMARY) S trajectory, every 4th sub-epoch, with 285-set (SECONDARY) in parentheses (diagnostics report lines 23-28)
uniform seed 1: ep4 3.75554 (3.75955); ep8 3.50021 (3.50297); ep12 3.43416 (3.43765); ep16 3.40870 (3.41226); ep20 3.39333 (3.39712); ep24 3.38268 (3.38619); ep28 3.37192 (3.37562); ep32 3.36691 (3.37066); ep36 3.36462 (3.36747); ep40 3.35351 (3.35612); ep44 3.35548 (3.35787); ep48 3.34756 (3.35049)
uniform seed 2: ep4 3.70472 (3.70946); ep8 3.44405 (3.44786); ep12 3.38450 (3.38858); ep16 3.36182 (3.36623); ep20 3.34990 (3.35432); ep24 3.33953 (3.34422); ep28 3.34135 (3.34626); ep32 3.33185 (3.33663); ep36 3.32687 (3.33156); ep40 3.32565 (3.33042); ep44 3.31984 (3.32448); ep48 3.32244 (3.32768)
durinit seed 1: ep4 3.67687 (3.68303); ep8 3.45331 (3.46069); ep12 3.39171 (3.39810); ep16 3.36749 (3.37397); ep20 3.35238 (3.35890); ep24 3.33486 (3.34073); ep28 3.32097 (3.32628); ep32 3.31434 (3.31995); ep36 3.31260 (3.31821); ep40 3.30499 (3.30952); ep44 3.30551 (3.31044); ep48 3.29903 (3.30403)
durinit seed 2: ep4 3.70562 (3.70926); ep8 3.48415 (3.48960); ep12 3.44038 (3.44519); ep16 3.41859 (3.42384); ep20 3.40694 (3.41193); ep24 3.39921 (3.40414); ep28 3.39024 (3.39503); ep32 3.38886 (3.39371); ep36 3.37687 (3.38232); ep40 3.37396 (3.37928); ep44 3.36756 (3.37290); ep48 3.37032 (3.37600)
durfrz seed 1: ep4 3.71092 (3.71740); ep8 3.49413 (3.50029); ep12 3.43879 (3.44490); ep16 3.41227 (3.41764); ep20 3.40050 (3.40619); ep24 3.39151 (3.39723); ep28 3.38345 (3.38877); ep32 3.38021 (3.38570); ep36 3.37801 (3.38341); ep40 3.36967 (3.37502); ep44 3.36852 (3.37381); ep48 3.36665 (3.37201)
durfrz seed 2: ep4 3.73649 (3.73945); ep8 3.50769 (3.51287); ep12 3.46400 (3.46924); ep16 3.44404 (3.44989); ep20 3.43303 (3.43846); ep24 3.42087 (3.42647); ep28 3.41845 (3.42421); ep32 3.41355 (3.42019); ep36 3.40571 (3.41218); ep40 3.40268 (3.40883); ep44 3.39778 (3.40424); ep48 3.39956 (3.40582)

set disjoint260: S paired over 260 of 260 (possible in every reference and restart json); excluded []
set all285: S paired over 285 of 285; excluded []
