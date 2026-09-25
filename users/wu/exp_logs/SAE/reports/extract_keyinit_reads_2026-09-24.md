# Keyinit pack (PackedBlankfreeTrainJob.ge1MKcAPmZIV) - extracted facts, 2026-09-24

Job dir: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ge1MKcAPmZIV

## 1. Pack status
- All 4 arms (g_dur, gold_key, r30_dur, r70_dur) have finished.tar.gz present at job root (job finished).
- Each arm's output/<arm>/models/ contains epoch.048.pt and epoch.048.opt.pt (all 48 sub-epochs present, checkpoints epoch.031..048 visible, earlier epochs presumably cleaned/rotated but 48 present for all four).
- grep of log.run.1 for nan/error/traceback in each arm: only "Cache manager: Error occurred, using local file" (benign, repeated) and "Learning-rate-control: error key 'train_loss_agg'/'dev_loss_agg' from {...}" (a harmless LR-control key-not-found message, not a NaN/crash) appear in all 4 arms. No traceback, no NaN value found in any of the 4 logs.
- Checkpoint paths (epoch 48):
  - g_dur: .../PackedBlankfreeTrainJob.ge1MKcAPmZIV/output/g_dur/models/epoch.048.pt
  - gold_key: .../PackedBlankfreeTrainJob.ge1MKcAPmZIV/output/gold_key/models/epoch.048.pt
  - r30_dur: .../PackedBlankfreeTrainJob.ge1MKcAPmZIV/output/r30_dur/models/epoch.048.pt
  - r70_dur: .../PackedBlankfreeTrainJob.ge1MKcAPmZIV/output/r70_dur/models/epoch.048.pt

## uRzbIh0EfQXG
MISSING (searched: entire /e/scratch/spell/wu24 tree, entire /e/project1/spell/wu24 tree, and grep -r for the string under .../work/speech_llm/sae/emc/ — no file, directory or job with this hash exists anywhere on disk).

## Reader jobs actually found in this graph (both finished, both carry S + genmarg reads)
1. KeyInitControlReadJob.JH7zzrX3Egfm
   - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/keyinit_read_jobs/KeyInitControlReadJob.JH7zzrX3Egfm
   - finished (finished.tar.gz present); output: keyinit_control.json, report.txt
   - Covers the gold_key control arm; GOLD KEY REACHES BASIN read.
2. A17SegmentationReadJob.lg4fTnb7deS8
   - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/keyinit_read_jobs/A17SegmentationReadJob.lg4fTnb7deS8
   - finished (finished.tar.gz present); output: a17iii.json, report.txt
   - Covers g_dur, r30_dur, r70_dur arms; A17 (iii) segmentation-need read.
3. SegmentationBoundaryJob.g52gIGtUHpkr (report-only, boundary P/R/F1/rate/OS/R-value; not a decision job)
   - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/keyinit_read_jobs/SegmentationBoundaryJob.g52gIGtUHpkr
   - finished (finished.tar.gz present); output: boundaries.json, report.txt

## 2. Per-arm S at sub-epoch 48 on the 260 set, bar, verdicts

Bar used by both readers: 3.289 (= S_min - 0.01, S_min = 3.29903 [durinit_s01], registered A10 diagnostics 260-tag paired value). Expected bar given in brief (3.28903) matches to stated precision (report prints "3.289"; the underlying arithmetic in KeyInitControlReadJob's report line gives "S_c - S_min ... (bar S_min - 0.01 = 3.28903)").

- gold_key (control): S_c(48) = 3.20704 (paired 260/260, excluded []). S_min = 3.29903 [durinit_s01]. S_c - S_min = -0.09199. Bar = S_min - 0.01 = 3.28903.
  Verdict (KeyInitControlReadJob report.txt): "VERDICT: GOLD KEY REACHES BASIN"
  Path: .../KeyInitControlReadJob.JH7zzrX3Egfm/output/report.txt (also keyinit_control.json)

- g_dur: S(48) = 3.22387 (paired 260/260, excluded [])
- r30_dur: S(48) = 3.21151 (paired 260/260, excluded []) — "r30_dur reported only", not a verdict input
- r70_dur: S(48) = 3.26568 (paired 260/260, excluded [])
  paired S_min (report only) = 3.29903 [durinit_s01]; registered S_min = 3.29903 [durinit_s01, 260 tags]
  Verdict (A17SegmentationReadJob report.txt): "VERDICT: LABELS SUFFICE"
  (rule: SEGMENTATION-CARRIED if g_dur S>=3.289; else LABELS SUFFICE if r70_dur S<3.289; else PARTIAL-LABELS NEED SEGMENTS. g_dur=3.22387<3.289 and r70_dur=3.26568<3.289 -> LABELS SUFFICE)
  Path: .../A17SegmentationReadJob.lg4fTnb7deS8/output/report.txt (also a17iii.json)

## 3. Beside-reads (report only, never gating)

Generative PER (direct, Hungarian) + NMI(symbol,phone) at sub-epochs 0, 4, 12, 48 — all present in both reader report.txt trajectories:
- gold_key: ep0 direct 0.3271 / Hung 0.3848 / NMI 0.7398; ep4 0.2842/0.3336/0.7465; ep12 0.3216/0.3670/0.7173; ep48 0.3457/0.3907/0.6945
- g_dur: ep0 0.1971/0.1971/0.8361; ep4 0.2973/0.2973/0.7508; ep12 0.3241/0.3241/0.7298; ep48 0.3447/0.3447/0.7099
- r30_dur: ep0 0.2479/0.2479/0.8025; ep4 0.3273/0.3273/0.7248; ep12 0.3500/0.3500/0.7053; ep48 0.3681/0.3681/0.6847
- r70_dur: ep0 0.6053/0.6219/0.4515; ep4 0.4300/0.4528/0.6093; ep12 0.4350/0.4591/0.6061; ep48 0.4420/0.4662/0.6058
(NMI column here = "NMI(symbol, phone)" as printed; no separate genmarg-direct/Hungarian/NMI json path beyond the report — these figures are read from GenDecodeReportJob outputs cited in the reader's IMPLEMENTATION section; report.txt is the on-disk artifact carrying them.)

Nothing found "pending" for these reads: both reader jobs are finished and both report.txt files already contain the full 0..48 trajectory with PER/NMI at the diagnostics sub-epochs (0,4,8,...,48, including the registered 0/4/12/48 checkpoints — note ep12 is present as a listed diagnostics sub-epoch). SLURM 1992872 (sacct): State = COMPLETED (00:02:34), all env sub-steps COMPLETED.

Boundary P/R/F1/rate(Hz)/OS/R-value against MFA, 260 set, 20 ms tolerance (SegmentationBoundaryJob report.txt), all present, none pending:
- MFA reference: 32579 boundaries on 260 utterances (137933 retained frames); MFA segment rate 11.904 Hz
- a16b_gold_key: ep0 P0.7929 R0.8337 F1 0.8128 rate 12.510 Hz OS 0.0514 R-value 0.8360; ep48 P0.8514 R0.7525 F1 0.7989 rate 10.532 Hz OS -0.1161 R-value 0.8168
- a17iii_g_dur: ep0 P0.8668 R0.8042 F1 0.8343 rate 11.051 OS -0.0722 R-value 0.8520; ep48 P0.8528 R0.7499 F1 0.7981 rate 10.479 OS -0.1206 R-value 0.8154
- a17iii_r30_dur: ep0 P0.8425 R0.7410 F1 0.7885 rate 10.481 OS -0.1205 R-value 0.8082; ep48 P0.8471 R0.7415 F1 0.7908 rate 10.432 OS -0.1247 R-value 0.8092
- a17iii_r70_dur: ep0 P0.6978 R0.4282 F1 0.5307 rate 7.341 OS -0.3864 R-value 0.5894; ep48 P0.8436 R0.7185 F1 0.7760 rate 10.153 OS -0.1483 R-value 0.7938
- a14ii_gold: ep0 P0.8656 R0.8141 F1 0.8390 rate 11.201 OS -0.0595 R-value 0.8577; ep48 P0.8571 R0.7497 F1 0.7998 rate 10.424 OS -0.1253 R-value 0.8158
- a14ii_r30: ep0 P0.8349 R0.7509 F1 0.7906 rate 10.716 OS -0.1006 R-value 0.8132; ep48 P0.8448 R0.7323 F1 0.7845 rate 10.332 OS -0.1331 R-value 0.8029
- a14ii_r70: ep0 P0.6577 R0.4345 F1 0.5233 rate 7.896 OS -0.3394 R-value 0.5903; ep48 P0.8509 R0.7153 F1 0.7772 rate 10.022 OS -0.1594 R-value 0.7925
- a14ii_r100: ep0 P0.6168 R0.3734 F1 0.4652 rate 7.243 OS -0.3947 R-value 0.5477; ep48 P0.7608 R0.5821 F1 0.6595 rate 9.131 OS -0.2348 R-value 0.6956
Path: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/keyinit_read_jobs/SegmentationBoundaryJob.g52gIGtUHpkr/output/report.txt

## 4. Recompute vs copy; set size

The readers' IMPLEMENTATION section (verbatim in both report.txt files) states: "S(json, tags) = the mean over tags of nll_tau1_per_frame (A14 (ii)'s convention) ... read on T = the 260-set tags possible ... in the [arm's] sub-epoch-48 json". Source code (keyinit_read_jobs.py) computes S from the `marginals` job inputs (per-epoch marginals JSONs) at run() time — i.e. the reader RECOMPUTES S itself from the marginals jsons; it does not copy an S value out of the training logs. No "copied from training log" statement found (searched report.txt and json for "recomput|copied|copy"; none matched — the recompute is established by the IMPLEMENTATION text + code reading marginals directly).

Set size: 260 (both readers report "paired over 260 of 260 (excluded [])" at sub-epoch 48 for all arms) — matches expected 260 utterances.
