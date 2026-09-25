# Extraction: AN-0 RenameEmStepJob.sXyLgUqfPBPv

Job dir (workspace path): /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/rename_emstep_jobs/RenameEmStepJob.sXyLgUqfPBPv
Resolved (readlink -f): /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/rename_emstep_jobs/RenameEmStepJob.sXyLgUqfPBPv
Alias: alias/sae/4a/rename/an0

## 1. VERDICT / READING lines

No literal lines beginning "VERDICT" or "READING" occur in log.run.1 (grep found none). report.txt (and log.run.1, which
embeds the same report) instead carries these outcome lines, verbatim:

```
AN-0 OUTCOME: AN-0 DEAD
```
(printed twice in report.txt: once near the top, once again at the end before the RULE text.)

The printed reading rule (verbatim, from report.txt / log.run.1, "Applied as:" paragraph):
"Applied as: a cell is one (lambda, form); VALID(cell) iff the undeployed gold-key row's identity_after >= 0.95 in
that cell.  AN-0 reads the gold-key 5-pair seed-1 row; the outcomes are tested in the rule's order: 'AN-0 OPEN' if
restore >= 0.05 in some VALID cell with lambda <= 4.4 (lambda 1 included; its cell is shared by both forms); else
'AN-0 DEAD' if at least one of the four cells with lambda in {2, 4.4} (both forms) is VALID and restore < 0.05 in
every VALID one of them; else 'AN-0 CANNOT TELL' (then none of those four cells is VALID).  When OPEN holds and none
of the lambda-{2, 4.4} cells is VALID, the CANNOT TELL clause also holds; the outcome stays OPEN and the detail
prints both flags.  lambda = 10 cells are printed and enter no outcome.  AN-3: ... [rest of paragraph covers AN-3
reading rules, not applicable to this AN-0 run; full text reproduced below in section 6]."

Full verdict-detail JSON (verbatim, printed under "verdict detail:"):
```json
{"row": "gold_key__5pair_s1", "cells": {"plain/1": {"form": "plain", "lambda": 1.0, "restore": 0.0, "rho": -0.001574721603884277, "valid": true, "gold_identity_after": 0.9963711029977252, "report_only": false}, "plain/2": {"form": "plain", "lambda": 2.0, "restore": 0.0018689794966075567, "rho": 0.00029425789272330416, "valid": true, "gold_identity_after": 0.9925101383136476, "report_only": false}, "plain/4.4": {"form": "plain", "lambda": 4.4, "restore": 7.050406016974916e-05, "rho": -0.01601581228663851, "valid": true, "gold_identity_after": 0.9888153851511772, "report_only": false}, "plain/10": {"form": "plain", "lambda": 10.0, "restore": 0.0, "rho": -0.2935162580922557, "valid": false, "gold_identity_after": 0.5817096233001452, "report_only": true}, "rate_neutral/1": {"form": "rate_neutral", "lambda": 1.0, "restore": 0.0, "rho": -0.001574721603884277, "valid": true, "gold_identity_after": 0.9963711029977252, "report_only": false}, "rate_neutral/2": {"form": "rate_neutral", "lambda": 2.0, "restore": 0.0031830913850453882, "rho": 0.0031830913850453557, "valid": true, "gold_identity_after": 0.9949542790661989, "report_only": false}, "rate_neutral/4.4": {"form": "rate_neutral", "lambda": 4.4, "restore": 0.005340567996943646, "rho": 0.0053405679969436015, "valid": true, "gold_identity_after": 0.9833956064645349, "report_only": false}, "rate_neutral/10": {"form": "rate_neutral", "lambda": 10.0, "restore": 0.02094160430843307, "rho": -0.030216521438340527, "valid": false, "gold_identity_after": 0.8982037241331273, "report_only": true}}, "open_cells": [], "dead_set_valid": ["plain/2", "plain/4.4", "rate_neutral/2", "rate_neutral/4.4"], "open_clause": false, "dead_clause": true, "cannot_tell_clause": false, "threshold": 0.05, "valid_min_gold_identity": 0.95, "outcome": "AN-0 DEAD"}
```

Gold-row guard line (verbatim):
```
gold-row guard (VALID iff the undeployed gold-key row's identity after >= 0.95): plain/1 VALID, plain/2 VALID, plain/4.4 VALID, plain/10 not valid, rate_neutral/1 VALID, rate_neutral/2 VALID, rate_neutral/4.4 VALID, rate_neutral/10 not valid
```

## 2. Table (verbatim from report.txt, both rows: gold_key = undeployed gold-key guard row; gold_key__5pair_s1 = deranged row)

Columns: form, lambda | restore  id_bef  id_aft  rho | m2o_bef m2o_aft | tok/fr SIL | S1_bef S1_aft | re-keyed | moved back | valid

```
row                      form         lambda | restore  id bef  id aft      rho | m2o bef m2o aft | tok/fr    SIL |   S1 bef   S1 aft | re-keyed | moved back | valid
gold_key                 plain             1 |  0.0000  1.0000  0.9964  -0.0036 |  1.0000  0.9964 | 0.2356 0.0803 |  4.56323  4.25083 |        3 | - | VALID
gold_key                 plain             2 |  0.0000  1.0000  0.9925  -0.0075 |  1.0000  0.9925 | 0.2161 0.0852 |  4.56323  4.25655 |        5 | - | VALID
gold_key                 plain           4.4 |  0.0000  1.0000  0.9888  -0.0112 |  1.0000  0.9888 | 0.1618 0.1067 |  4.56323  4.38189 |       10 | - | VALID
gold_key                 plain            10 |  0.0000  1.0000  0.5817  -0.4183 |  1.0000  0.5817 | 0.0704 0.3113 |  4.56323  4.86011 |      204 | - | no (report only)
gold_key                 rate_neutral      1 |  0.0000  1.0000  0.9964  -0.0036 |  1.0000  0.9964 | 0.2356 0.0803 |  4.56323  4.25083 |        3 | - | VALID (shared with plain/1)
gold_key                 rate_neutral      2 |  0.0000  1.0000  0.9950  -0.0050 |  1.0000  0.9950 | 0.2357 0.0844 |  4.56323  4.25686 |        4 | - | VALID
gold_key                 rate_neutral    4.4 |  0.0000  1.0000  0.9834  -0.0166 |  1.0000  0.9834 | 0.2524 0.1105 |  4.56323  4.32451 |        8 | - | VALID
gold_key                 rate_neutral     10 |  0.0000  1.0000  0.8982  -0.1018 |  1.0000  0.8982 | 0.3252 0.2011 |  4.56323  4.58013 |       51 | - | no (report only)
gold_key__5pair_s1       plain             1 |  0.0000  0.5931  0.5916  -0.0016 |  1.0000  0.9863 | 0.2312 0.0786 |  4.88554  4.49803 |        8 | 0.019 | VALID
gold_key__5pair_s1       plain             2 |  0.0019  0.5931  0.5934  +0.0003 |  1.0000  0.9771 | 0.2000 0.0858 |  4.88554  4.50057 |       13 | 0.029 | VALID
gold_key__5pair_s1       plain           4.4 |  0.0001  0.5931  0.5771  -0.0160 |  0.9999  0.9527 | 0.1367 0.1190 |  4.88554  4.62234 |       35 | 0.033 | VALID
gold_key__5pair_s1       plain            10 |  0.0000  0.5931  0.2996  -0.2935 |  0.9999  0.4362 | 0.0627 0.3403 |  4.88554  5.00949 |      263 | 0.023 | no (report only)
gold_key__5pair_s1       rate_neutral      1 |  0.0000  0.5931  0.5916  -0.0016 |  1.0000  0.9863 | 0.2312 0.0786 |  4.88554  4.49803 |        8 | 0.019 | VALID (shared with plain/1)
gold_key__5pair_s1       rate_neutral      2 |  0.0032  0.5931  0.5963  +0.0032 |  1.0000  0.9829 | 0.2318 0.0834 |  4.88554  4.50871 |        9 | 0.031 | VALID
gold_key__5pair_s1       rate_neutral    4.4 |  0.0053  0.5931  0.5985  +0.0053 |  1.0000  0.9733 | 0.2528 0.1184 |  4.88554  4.58172 |       15 | 0.045 | VALID
gold_key__5pair_s1       rate_neutral     10 |  0.0209  0.5932  0.5629  -0.0302 |  0.9999  0.7396 | 0.3287 0.2187 |  4.88554  4.80466 |      115 | 0.037 | no (report only)
```

Note: the printed table columns do not literally separate "key identity before/after" and "gold-row identity after" as
distinct named columns; "id bef"/"id aft" is the row's own key identity (before/after); for the gold_key row this IS
the gold-row identity after (guard target); for the deranged row (gold_key__5pair_s1) the gold-row identity after is
given only in the "verdict detail" JSON's "gold_identity_after" field per cell (section 1), not as a separate column
in this table.

## 3. H_LM, derangement, utterances, smoothing (verbatim from report.txt)

```
H_LM = 2.257108 nats per token (the frozen trigram on its own 1000000 counted lines, 81559944 tokens; train ppl 9.5554; /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/prior/PhoneNgramPriorJob.RtzbESkOedsT/output/prior.npz)
```

Derangement pool (verbatim):
```
derangement pool (non-SIL symbols holding units in the gold key, 37): ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z']
```

Drawn derangement pairs (verbatim, from the "gold_key__5pair_s1:" line):
```
gold_key__5pair_s1: drawn ['AH<->T', 'AO<->N', 'K<->OW', 'M<->Y', 'P<->S']; moved gold frame share 0.4068
```

Utterances used (verbatim):
```
300 train utterances (A12 selection, seed 0); tau 1.0; E-step emission 0.9 table + 0.1 uniform; type-level M-step (pseudo-count 0.001); identity / many-to-one / restore frame-weighted by train unit counts against the gold key (/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/unit_key_jobs/GoldUnitKeyJob.sLnMRRd2qO0t/output/key.json)
```
This matches the spec's "300 fixed train utterances" — 300 is the printed count, no discrepancy noted in the report
(no separate "utterances actually used" / "Z=0 skipped" count is printed elsewhere in report.txt or log.run.1; only
the S_1 line below gives a separate held-out count).

S_1 utterance count (verbatim):
```
S_1 paired over 260 of 260 utterances (A13 260 set; excluded [])
```

Smoothing used: "E-step emission 0.9 table + 0.1 uniform" (A12's smoothing), and per the CONVENTION text: "the E-step
sees 0.81 ML + 0.19 / 500 (both as registered)" because the key phis' tables already carry the key conversion's own
0.1 smoothing.

## 4. Runtime, node, times, warnings/errors

usage.run.1 (verbatim JSON):
```
{'current': {'cpu': 100.0, 'rss': 1.33221435546875, 'vms': 233.18798828125},
 'current_time': 'Fri Sep 25 01:44:29 2026',
 'host': 'jpbo-054-42.jupiter.internal',
 'max': {'cpu': 101.3, 'rss': 1.3331298828125, 'vms': 233.18896484375},
 'out_of_memory': False,
 'pid': 691980,
 'requested_resources': {'cpu': 4,
                         'engine': 'gpupack',
                         'gpu': 1,
                         'mem': 24.0,
                         'sbatch_args': ['-A', 'spell', '-p', 'booster', '--exclusive'],
                         'time': 0.5},
 'used_time': 0.059354865815904406,
 'user': ('wu24',)}
```

submit_log.run (verbatim):
```
([1], {'cpu': 4, 'mem': 24.0, 'time': 0.5, 'gpu': 1, 'engine': 'gpupack', 'sbatch_args': ['-A', 'spell', '-p', 'booster', '--exclusive'], 'engine_info': [([1], 'gpupack-buffered')], 'engine_name': 'gpupack', 'completed_fraction': None})
```
gpupack: the submit_log's engine is "gpupack" (buffered), no separate gpupack job dir found under this job's
`engine/` subdirectory (it contains only `speech_llm.sae.emc.rename_emstep_jobs.RenameEmStepJob.sXyLgUqfPBPv.run.2005126.1`;
no further gpupack-specific job log was located).

Node (from log.run.1 line 1 and GPU lines):
```
jpbo-054-42.jupiter.internal
CUDA_VISIBLE_DEVICES=0
GPU 0: NVIDIA GH200 120GB (UUID: GPU-c65f9b25-72ef-3d04-e9a6-c8010e693479)
GPU 1: NVIDIA GH200 120GB (UUID: GPU-5c538311-75cb-5ff6-1877-37e89a9eb818)
GPU 2: NVIDIA GH200 120GB (UUID: GPU-e8ae8fa4-b755-d659-8a1a-1a709e812a23)
```
(3 GPUs listed in the node banner; CUDA_VISIBLE_DEVICES=0 restricts the job to GPU 0; --exclusive requested in
submit_log implies whole-node allocation per the OverSubscribe=EXCLUSIVE booster convention, but the job used gpu:1.)

Start time (log.run.1 line 7):
```
[2026-09-25 01:40:55,850] INFO: Start Job: Job<alias/sae/4a/rename/an0 work/speech_llm/sae/emc/rename_emstep_jobs/RenameEmStepJob.sXyLgUqfPBPv> Task: run
```

Finish time (log.run.1, last lines):
```
[2026-09-25 01:44:29,525] INFO: Max resources: Run time: 0:03:33 CPU: 101.3% RSS: 1.33GB VMS: 233.19GB
[2026-09-25 01:44:29,527] INFO: Job finished successfully
```

Runtime printed in report.txt: `time 192.3 s; peak GPU 14.144063472747803 GiB; peak RSS 1.26 GB`
(Run time per log.run.1's "Max resources" line: 0:03:33 = 213 s, wall clock 01:40:55.850 to 01:44:29.527 = 213.7 s;
the report.txt-printed "time 192.3 s" is the job's internal compute-only timer, distinct from the wall/manager-tracked
run time — both values reported as printed, no reconciliation performed here.)

Warnings/errors: `grep -in "warn|error|traceback|exception"` on log.run.1 returned no matches. finished and
finished.run.1 are present (0 bytes each, standard sisyphus completion markers).

## 5. Output file paths

- /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/rename_emstep_jobs/RenameEmStepJob.sXyLgUqfPBPv/output/report.txt
- /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/rename_emstep_jobs/RenameEmStepJob.sXyLgUqfPBPv/output/rename_emstep.json

(workspace-view equivalents, same job, symlinked path:
/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/rename_emstep_jobs/RenameEmStepJob.sXyLgUqfPBPv/output/report.txt
/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/rename_emstep_jobs/RenameEmStepJob.sXyLgUqfPBPv/output/rename_emstep.json)
