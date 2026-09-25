# A18 (b) durinit basin lift extraction — 2026-09-25

## 1. VERDICT / READING lines, rule, bands, per-arm table (verbatim from report.txt)

Job: work/speech_llm/sae/emc/a18_keyinit_lift_jobs/DurinitBasinLiftReadJob.uRzbIh0EfQXG
Resolved path: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/a18_keyinit_lift_jobs/DurinitBasinLiftReadJob.uRzbIh0EfQXG

Title: "A18 (b): does the durinit keyinit basin lift a random theta?  DISCLOSED ANALYSIS ONLY.  L2-1 A18 (b) keyinit lift (disclosed analysis)"

RULE (verbatim):
"The keyinit pack (ge1MKcAPmZIV: gold-key control, G-dur, r30-dur, r70-dur), disclosed analysis only.
   All four sub-epoch-48 phis go to one four-arm lift pack in A14 (i)'s form, as A17 (i): no selection,
   launched when the keyinit pack finishes.  Read per arm in A4's bands.  DURINIT BASIN LIFTS if G-dur or
   r30-dur reads LIFT or PARTIAL; DURINIT BASIN DOES NOT LIFT if G-dur, r30-dur and r70-dur all read NO
   LIFT.  Reported beside: the jointly trained phi's generative PER at ep8, and paired rows against A17
   (i)'s same-init arm with MFA durations (G-dur against gold-EM, r30-dur against r30-EM, r70-dur against
   r70-EM).  These rows ask the segmentation question at the lift level."
  A4's bands: LIFT = dev-other greedy PER < 0.50 at ep8, PARTIAL < 0.8164, else NO LIFT.

IMPLEMENTATION (verbatim, key points):
* Class per arm from the arm's ep8 dev-other greedy PER (per.json), A17 (i)'s lift_class.
  * The gold-key control is read and classed like the others and enters neither clause.
  * The case the rule does not cover -- G-dur and r30-dur NO LIFT while r70-dur lifts -- reads MIXED
    (coordinator ruling 2026-09-24: reported, no consequence); a missing ep8 PER prints CANNOT_TELL.
  * The paired rows are NOT read here: depends on lift pack's reads only (and keyinit arms' own
    sub-epoch-48 phi reports for ep0), never on an A17 job. Rows are their own eval_jobs.PairedPerDeltaJob
    jobs (paired_per.json / summary.txt, A17 (i)'s arm = A, durinit arm = B, delta = PER(B) - PER(A) over
    the 2864 dev-other utterances, 95% speaker-clustered bootstrap), built only behind
    config_sae_4a_lexlat_v2_a18_v1.A18B_PAIRED_ROWS; never gating.
  * The jointly trained phi (report only): GenDecodeReportJob on the D4 dev-other set, direct / Hungarian
    PER and NMI(symbol, phone), at every kept epoch given; ep0 = the init phi (keyinit arm's sub-epoch 48).

Per-arm dev-other greedy PER table (verbatim):
dev-other greedy PER            ep     1  ep     2  ep     4  ep     8   class@ep8
  gold_key                        0.2752    0.2581    0.2351    0.2078   LIFT
  g_dur                           0.2749    0.2371    0.2321    0.1931   LIFT
  r30_dur                         0.2901    0.2638    0.2283    0.2059   LIFT
  r70_dur                         0.3546    0.3446    0.2783    0.2275   LIFT

paired rows line (verbatim): "paired rows at ep8 against A17 (i)'s same-init arm: not read here (their own
PairedPerDeltaJob summaries, g_dur vs a17_gold_em, r30_dur vs a17_r30_em, r70_dur vs a17_r70_em)"

beside (report only) block, jointly trained phi genmarg decode of D4 dev-other, direct/Hungarian PER/NMI
(verbatim, ep0 = keyinit sub-epoch 48 init phi):
  gold_key phi    ep0 0.3457/0.3907/0.6945  ep1 0.3299/0.3771/0.7062  ep2 0.3000/0.3493/0.7330  ep4 0.2686/0.3214/0.7609  ep8 0.2577/0.3104/0.7709
  g_dur phi       ep0 0.3447/0.3447/0.7099  ep1 0.3308/0.3308/0.7200  ep2 0.2734/0.2734/0.7692  ep4 0.2458/0.2458/0.7973  ep8 0.2215/0.2215/0.8103
  r30_dur phi     ep0 0.3681/0.3681/0.6847  ep1 0.3496/0.3496/0.7027  ep2 0.3100/0.3100/0.7373  ep4 0.2615/0.2615/0.7750  ep8 0.2435/0.2435/0.7869
  r70_dur phi     ep0 0.4420/0.4662/0.6058  ep1 0.4255/0.4526/0.6206  ep2 0.3828/0.4138/0.6632  ep4 0.3261/0.3701/0.7058  ep8 0.2838/0.3349/0.7412

VERDICT (verbatim): "DURINIT BASIN LIFTS"

Bands in JSON: lift_per = 0.5, partial_per = 0.8164. No separate "spread B" field present in
a18_keyinit_lift.json or report.txt — MISSING (searched: output/a18_keyinit_lift.json, output/report.txt).

## 2. Per-arm reading for a18_gold_key specifically (verbatim, from JSON "arms.gold_key")
per: {"1": 0.27517698491044984, "2": 0.2580736144408405, "4": 0.2350980115639543, "8": 0.2078127203497391}
class_ep8: "LIFT"
(Note: report.txt row label is "gold_key"; there is no separate arm literally named "a18_gold_key" in
this job's arms dict — the arm key is "gold_key".)

## 3. Paired rows (G-dur vs gold-EM, r30-dur vs r30-EM, r70-dur vs r70-EM)
Not present in this job. Per the report's own text: "paired rows ... not read here (their own
PairedPerDeltaJob summaries, g_dur vs a17_gold_em, r30_dur vs a17_r30_em, r70_dur vs a17_r70_em)" and
"built only behind config_sae_4a_lexlat_v2_a18_v1.A18B_PAIRED_ROWS; never gating."
MISSING (searched: work/speech_llm/sae/emc/ for PairedPerDeltaJob dirs named with a17_gold_em / a17_r30_em
/ a17_r70_em pairing to g_dur/r30_dur/r70_dur; none found matching this specific A18(b) pairing under the
job dirs inspected). No eval_jobs.PairedPerDeltaJob output located for this specific comparison.

## 4. Jointly trained phi generative PER at ep8 per arm (direct, Hungarian, NMI) — from GenDecodeReportJobs
(values as embedded in DurinitBasinLiftReadJob's own JSON output/a18_keyinit_lift.json, which reads these
per-epoch GenDecodeReportJob outputs; cross-checked against GenDecodeReportJob.wGgGstGW3XpG output, which
matches the r30_dur ep8 row exactly)
  gold_key ep8: per_direct=0.2576625126305153, per_hungarian=0.3104412260020209, nmi_symbol_phone=0.7709204955210871
  g_dur    ep8: per_direct=0.221488716739643,  per_hungarian=0.221488716739643,  nmi_symbol_phone=0.8103468755419565
  r30_dur  ep8: per_direct=0.24348265409228698,per_hungarian=0.24348265409228698,nmi_symbol_phone=0.7868500524553578
  r70_dur  ep8: per_direct=0.2837992590097676, per_hungarian=0.3349275850454699, nmi_symbol_phone=0.7411775400219944
GenDecodeReportJob.wGgGstGW3XpG output/report.json (path:
/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_genmarg_jobs/GenDecodeReportJob.wGgGstGW3XpG/output/)
matches r30_dur ep8 exactly: nmi_symbol_phone.nmi=0.7868500524553578, per_identity.per=0.24348265409228698,
per_hungarian.per=0.24348265409228698 (per_identity == per_hungarian here, band "BELOW").
Note: job bbrCBu3GYtJH named in the brief resolves to a BlankfreeGreedyPerJob (not a GenDecodeReportJob):
/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_eval_jobs/BlankfreeGreedyPerJob.bbrCBu3GYtJH
The 16 individual GenDecodeReportJob dirs were not each opened; values above are taken from the
DurinitBasinLiftReadJob's aggregated JSON (its stated source) plus one direct cross-check (wGgGstGW3XpG).

## 5. Rerun of ReturnnForwardJobV2.GTAGKejuQTxD
Path: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_core/returnn/forward/ReturnnForwardJobV2.GTAGKejuQTxD
- First attempt failed on node jpbo-028-30.jupiter.internal (files: error.run.1.failed_jpbo-028-30,
  log.run.1.failed_jpbo-028-30, submit_log.run.failed_jpbo-028-30, usage.run.1.failed_jpbo-028-30).
- Successful rerun ran on node jpbo-099-02.jupiter.internal (usage.run.1: 'host':
  'jpbo-099-02.jupiter.internal'; log.run.1 Uname node='jpbo-099-02.jupiter.internal').
- Run time (from log.run.1): "Forward 5 steps, 0:00:23 elapsed (68.6% computing time)"; "elapsed:
  0:00:24.7959"; final "Max resources: Run time: 0:00:34 CPU: 98.5% RSS: 3.41GB VMS: 260.85GB". usage.run.1
  used_time field = 0.009590082433488634 (hours, i.e. ~34.5s).
- Job input: phi_checkpoint PackedBlankfreeTrainJob.ZUZypSQn7qc0/output/a18_r30_dur/models/epoch.008.pt
  (epoch 8), i.e. this is the a18_r30_dur ep8 forward.
- "re-add check": no line literally containing "re-add"/"readd" was found in log.run.1. The closest
  printed check is: "gendecode l21_a18b_a18_r30_dur_ep8 on dev-other: 500 / 500 decoded, 0 impossible;
  every live path valid; max(log_w - log Z_1) = -11.356478167756222" — i.e. all 500 utterances decoded, 0
  impossible, every live path valid. Whether this is "the re-add check" referenced in the brief is
  MISSING/CANNOT_TELL (searched: log.run.1, info, job.save for "re-add"/"readd"/"add-back").
- Job finished successfully: "[...] INFO: Job finished successfully" in log.run.1; finished.run.1 present.

## 6. S values at sub-epoch 48 on the 260 set (hold rule vs 3.19704)
None found. The hold rule (SAE_4A_lexlat_v2.md line 15, verbatim): "apply the A18 (c) hold rule at once,
which the review's condition requires: if (b)'s gold-key arm reads NO LIFT and S_cand >= 3.19704 on the
same tags, the implementer sets BRIDGE_KEYARMS False ... Re-enable only if S_cand < 3.19704." The
gold-key arm's class_ep8 in this job is "LIFT", not "NO LIFT", so per the rule's own stated trigger
condition the S_cand comparison does not apply here. No S_cand value was found printed anywhere in
work/speech_llm/sae/emc/ (grepped for "S_cand"/"s_cand": only unrelated PairedWerDeltaJob per_utterance.json
files matched, none containing an S_cand field) or in keyinit_read_jobs/ outputs (KeyInitControlReadJob.JH7zzrX3Egfm,
A17SegmentationReadJob.lg4fTnb7deS8). MISSING (searched: work/speech_llm/sae/emc/ grep -rl "S_cand"/"s_cand";
work/speech_llm/sae/emc/keyinit_read_jobs/*/output/*).

## 7. Output paths
- DurinitBasinLiftReadJob.uRzbIh0EfQXG:
  /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/a18_keyinit_lift_jobs/DurinitBasinLiftReadJob.uRzbIh0EfQXG/output/report.txt
  /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/a18_keyinit_lift_jobs/DurinitBasinLiftReadJob.uRzbIh0EfQXG/output/a18_keyinit_lift.json
- PackedBlankfreeTrainJob.ZUZypSQn7qc0:
  /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ZUZypSQn7qc0/
- GenDecodeReportJob.wGgGstGW3XpG:
  /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_genmarg_jobs/GenDecodeReportJob.wGgGstGW3XpG/output/
- BlankfreeGreedyPerJob.bbrCBu3GYtJH (not a GenDecodeReportJob, see item 4 note):
  /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_eval_jobs/BlankfreeGreedyPerJob.bbrCBu3GYtJH/
- ReturnnForwardJobV2.GTAGKejuQTxD:
  /e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_core/returnn/forward/ReturnnForwardJobV2.GTAGKejuQTxD/
- All 16 GenDecodeReportJob dirs live under:
  /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_genmarg_jobs/ (not individually
  enumerated/opened beyond wGgGstGW3XpG; the lift job's own JSON is the source for the per-arm ep8 table above)
