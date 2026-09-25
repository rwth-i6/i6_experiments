# Code review: G1.G core phi reads (genmarg dev-other PER/Hungarian/NMI/E[d], ladder 260 set), 2026-09-25

Reviewer: code-reviewer. Scope: uncommitted `reverse_model/genmarg.py` and `reverse_model/ladder.py`, new tests
`tests/test_reverse_genmarg_devother.py` (16) and `tests/test_reverse_ladder_disjoint.py` (4), setup-local
`config/sae_i6_g0g.py`. Implementer report: `reports/impl_g0g_core_phi_reads_2026-09-25.md`. Nothing edited or launched.

## Verdict: PASS (no must-fix). Recommendations below are non-blocking.

## CHECK 1: conventions against JUPITER
The JUPITER source of GenDecodeReportJob is not on i6; verdicts rest on the frozen logs and reports in
`recipe/i6_experiments/users/wu/exp_logs/SAE/`.

- (a) Uniform prior (npz of -log 40 at prior_weight 1, `genmarg.py:814` UniformPhonePriorJob) vs weight 0: UNKNOWN.
  No JUPITER text says which. At weight 1 every emitted token pays log 40 extra relative to weight 0, which
  changes the insertion/deletion trade-off of the Viterbi decode, so the two readings give different R2.
  R2 (banked 0.276) is report-only, so no gate number depends on it. Fix if resolution wanted: take the A15 R2
  code from the JUPITER orchestrator, or add a report-only prior_weight-0 arm (the assert at `genmarg.py:501`
  pins weight 1.0 for all decodes, so that arm needs a code change).
- (b) Hungarian map (`genmarg.py:701`): MATCH with the campaign precedent. 40x40 linear_sum_assignment
  (maximize) plus one zero-gain DELETE column, counts from the identity-label alignment of the raw decode (SIL
  kept), relabelled string re-aligned fresh: `reports/impl_private_code_2026-09-20.md:66-108`,
  `reports/audit_private_code_2026-09-20.md:26`, and `SAE_4A_lexlat_v2.md:551` ("aligns decoded tokens to gold
  by identity-label edit alignment before its Hungarian step"). GenDecodeReportJob's own code: not seen.
- (c) Direct PER (SIL dropped, no repeat collapse, `genmarg.py:689`): MATCH with the banked scoring convention
  (`analysis/per.py` edit_counts; impl_private_code:103-107). The collapsed variant is reported beside it.
- (d) NMI: token-level over aligned pairs MATCH (lexlat_v2:551: map and NMI depend on labels, i.e. token-level).
  Frame NMI (MFA parquet, frame centre (i+0.5)/50) is an extra with no JUPITER counterpart; report-only.
- (e) E[d] (`genmarg.py:782`, duration_law at ReverseConfig(), unweighted mean over 39 phones, SIL separate):
  table-mean MATCH (lexlat_v2:436,455 "durfrz is held at 4.41"). Type weighting UNKNOWN; report-only. Runs on the
  real gold phi (Ac2eioZbRX7d epoch.008.pt): phones 5.0966, SIL 4.9991.
- (f) D4 sample (`genmarg.py:1018`, sorted non-empty-gold tags, RandomState(0).permutation, first 500): MATCH
  (`reports/audit_lam3_gate_2026-09-15.md:55,67-68`: eligible 2863, 500 selected). i6 gold: 2864 tags, 2863
  non-empty, empty tag 1651-136854-0012 as on JUPITER (audit_prior_gap_2026-09-20.md:32); 177,275 phones total.
- (g) 260 set (`ladder.py:512` holdout minus union of fit sets): MATCH (lexlat_v2:121-127 A13: all ladder fits
  on the 2821 split, so union = intersection). On disk: CvHoldoutSplitJob.mIdi9Dy1b4Xy cv 285, intersect
  zYcc8EJsvdfV train 2821 = 25, disjoint 260.

## CHECK 2: jobs, tests, leakage, single delta
- Trigram decode ReturnnForwardJobV2.RFaMyWRYUUQN and uniform decode 4hVXB1LWkw5p differ only in
  prior_npz_path (PhoneNgramPriorJob.qJxXHgXLe31S vs UniformPhonePriorJob.n7Yd9EbtVi13). Both: phi from
  ReturnnTrainingJob.Ac2eioZbRX7d epoch 8, prior_weight 1.0, eta SpeakerEtaJob.Zoat6qkUPL8Q, dev-other VAD HDFs
  BlankfreeVadHdfJob.RLrgIh6lFv9m, seq_list_filter GenMargSampleJob.sMYB3BL8ppHJ, sorted, batch features 88000,
  max_seqs 128, expected 500.
- The trigram is the bed's i6 prior (qJxXHgXLe31S), i.e. the trigram the gate names under the disclosed P0
  deviation (i6 full-text prior, not JUPITER's truncated RtzbESkOedsT).
- Uniform npz loads through PhoneNgramPrior.load; the trigram history reads log_tri only (no tri_counts needed).
- Eta covers all 2864 dev-other tags; feats/units/orig_length/raw_index HDFs carry the same 2864 tags.
- Leakage: the gold phi is fitted on the 2821 seed train-clean-100 items; dev-other is disjoint. Labels enter
  only the report job. The gap/report quarantine in genmarg_reads is kept and tested.
- Test oracles (hand-computed alignments, Hungarian with DELETE column, NMI, disjoint split) checked by reading.

## CHECK 3: P0 import closure
- `sis console config/sae_i6_p0.py -s` (read-only): 164 ids, identical to
  `analysis_out/g0g_p0_jobids_before_2026-09-25.txt`. Loaded modules identical to
  `analysis_out/g0g_p0_graph_modules_2026-09-25.txt`; genmarg and ladder are not in the P0 closure;
  `reverse_model/__init__.py` imports nothing; phi_first.py unmodified (git status: only genmarg.py, ladder.py).
- No P0 returnn.config references genmarg or ladder.

## CHECK 4: second manager on config/sae_i6_g0g.py
- 64 ids, identical to `analysis_out/g0g_entry_jobids_2026-09-25.txt`; 58 shared with P0, ALL finished in
  `work/` (incl. ReturnnTrainingJob.Ac2eioZbRX7d, finished Sep 25 01:43). No unfinished shared job, so the
  second manager submits nothing the P0 manager (pid 1646677) could also submit. 6 new jobs absent from work/.
- Resources: gpu 1, cpu 4, mem 32, time 2 h, `-p gpu_32gb` (config line 44; settings.check_engine_limits
  respects an explicit -p). gpu_32gb = V100, sm_70, which the k2 build covers.
- 2 h plausibility: D4 is about 136k retained frames (781,125 frames / 2864 utt x 500). V100 training steps
  run about 52 s per 60k frames forward+backward (SAE_i6_P0.md:304); a forward-only decode at 2.3x the frames
  should fit in 2 h. Not measured; V100 memory at 88000 features / float64 also not measured.

## CHECK 5: tests (sae python, CPU, PYTHONPATH=recipe:recipe/returnn:sisyphus)
test_reverse_genmarg_devother, test_reverse_ladder_disjoint, test_reverse_genmarg, test_reverse_ladder,
test_config_graph, test_reverse_genmarg_decode, test_reverse_phi_first: 92 passed, 0 failed, 0 skipped
(20 new tests collected; the implementer's file lists 19, from an earlier stage).

## Non-blocking recommendations
1. (a) is UNKNOWN; if R2 is to be compared with 0.276, resolve it from the JUPITER source or add a weight-0 arm.
2. The gate's +/-0.02 on R1 0.193 is attributed to the phi refit only; R1 also carries the i6 trigram prior,
   the i6 eta and the i6 audio (all disclosed P0 deviations). Name them when the gate is read.
3. The implementer's launch command is `sis m config/sae_i6_g0g.py` without `-r`, while the P0 manager runs
   `m -r`; without -r the manager waits for a console confirmation before submitting.
4. `config/sae_i6_g0g.py:1` docstring and PREFIX say G0.G / SAE_i6_P0.md / `sae_i6/g0g`, while the dispatch
   gate is G1.G in SAE_i6_P1.md. Output paths only; no effect on numbers.
