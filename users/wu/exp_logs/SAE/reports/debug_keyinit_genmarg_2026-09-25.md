# Debug: genmarg decode a18_r30_dur ep8 (ReturnnForwardJobV2.GTAGKejuQTxD), 2026-09-25

Status: DONE_WITH_CONCERNS. The fault cannot be reproduced and sits in the environment (runtime) layer.
The code is correct and the phi is well formed. The mechanism on the failing GPU cannot be
determined from these logs.

## Failure
- Job: work/i6_core/returnn/forward/ReturnnForwardJobV2.GTAGKejuQTxD. It decodes arm a18_r30_dur at
  epoch 8 from PackedBlankfreeTrainJob.ZUZypSQn7qc0 (output/a18_r30_dur/models/epoch.008.pt); the
  sample is GenMargSampleJob.zXJjnNqU7kTa.
- Where it ran: jpbo-028-30, CUDA_VISIBLE_DEVICES=1 (GH200 device 1), SLURM 2004639 through the
  gpupack engine, with cwd /e/project1/spell/wu24/2026-06-17_ssl/tmp/ere27alf. RETURNN
  1.20260518.222345+git.00171dfe.dirty. It started at 2026-09-25 01:01:15 and died at 01:01:46.
- log.run.1 line 107: `AssertionError: ('8254-84205-0073', -3613454734492735.0, -306.4832884073266)`.
  The assert is check (1) in `generative_decode` at
  recipe/2025-10-speech-llm/src/speech_llm/sae/emc/blankfree_genmarg_jobs.py:508:
  `assert abs(readd[i] - lw[i]) <= _READD_RTOL * max(1.0, abs(lw[i]))`, with _READD_RTOL = 1e-8.
- The assert fired in forward step 0 (the first batch: 128 utterances, units shape (128,136)) at batch
  index 12. Rows 0 to 11 of the same batch passed checks (1) and (2).
- In the failing tuple, the max-plus value lw = -306.4832884073266 is correct. The re-added path
  weight readd = -3.6e15 is the wrong value.

## Q1: root cause
Observed:
1. Code and config unchanged. The decode code (blankfree_genmarg_jobs.py, soft_scorer.py,
   lattice.py, reverse.py) was last modified on 09-23, 09-21 and 09-19. That is before the run and
   before the 15 lift-pack decodes that succeeded, so there is no tooling regression.
2. Exact reproduction on the login GH200 (jpbl-s02-03). I used the job's own returnn.config
   (identical to the job's output/returnn.config per diff), the same RETURNN checkout and version,
   the same conda env and the same checkpoint. The scratch dirs are in the scratchpad (rnnrun,
   rnnrun2, rnnrun3).
   - All 3 runs exit 0. Each has 500/500 decoded, 0 impossible, and all checks pass (log:
     "every live path valid; max(log_w - log Z_1) = -11.356478167756222").
   - The outputs of the 3 runs are identical.
3. Utterance-level reproduction (repro.py). I rebuilt the job's batch exactly (the same 128 tags in
   the same order, sorted by length with HDF-order tie-break) and ran it on CUDA.
   - lw matches the job's value bit for bit for 8254-84205-0073 (-306.4832884073266). It also
     matches utterances 0 to 13.
   - readd for 8254-84205-0073 is -306.4832884073266, equal to lw.
   - 20 of 20 repeats (loop.py) give identical results.
4. The value is unreachable from correct tables. On this batch:
   - A seg_table cell is either masked (<= -1e20 after scaling, because NEG_INF = -1e30) or legal
     (>= -1013).
   - prior_term lies in [-35.5, 0], and log_q is 0 (the null recognizer).
   - So a sum of per-token terms is either > about -1e5 or < -1e20. The value -3.6e15 lies in
     neither range.
   - It is also not a single-bit flip of any per-token term or of the correct total (checked in
     tokens.py).
   - This refutes the masked-component hypothesis: a path through a masked cell would read about
     -1e29 to -1e30, not -3.6e15.
5. No phi anomaly. For all four arms, dur_logits [40,50] are finite. The legal duration log-probs
   are >= -13.2, and there are no extreme parameters (durs.py). a18_r30_dur has no duration or
   transition entry that the other arms lack.
6. Pack-mates on the same node reproduce exactly. The ep8 decodes of gold_key (6dI3l4J7CoBB,
   GPU3), g_dur (VJnNjKM4mCPH, GPU2) and r70_dur (0PxK9UgOfqFn, GPU2) ran on jpbo-028-30 in the
   same pack. When rerun on the login GH200, their gendecode.json and decode_raw.json are
   byte-identical to the finished outputs.

Inferred:
- Something produced a corrupted readd on jpbo-028-30 GPU1 during this one run. Candidates are
  corruption in the walk's backpointer/argmax tensors or in the gather inside
  `segment_conditionals`, since lw itself came out right. That points to transient hardware or
  runtime corruption on that device (environment layer).
- It is not a decode bug and not a re-add bug: the same code on the same inputs gives
  readd == lw, deterministically.
- The mechanism cannot be determined from these logs. The following would tell:
  - node-side XID/ECC records for jpbo-028-30 GPU1 around 2026-09-25 01:01 (admin or
    `nvidia-smi -q -d ECC,PAGE_RETIREMENT` on that node);
  - one rerun pinned to that node and GPU.
- Upstream: I searched PyTorch, RETURNN and NVIDIA for silent wrong values in gather/argmax on
  GH200. PyTorch #196258 (expandable_segments plus empty_cache across devices) does not apply: this
  is a single device, and empty_cache is only called in bench(). No matching issue found.

## Q2: exposure of other genmarg forwards
- Lift pack (alias/sae/4a/lexlat_v2/em/a18_keyinit_lift/phi_diag/*): 15 of 16 decode jobs are
  finished and 15 of 16 GenDecodeReportJobs are finished. The exceptions are GTAGKejuQTxD (error)
  and GenDecodeReportJob.wGgGstGW3XpG (waiting on it, no dir).
- Each finished decode passed the same check battery (1) and (2) for all 500 utterances. The three
  ep8 decodes that shared the node with GTAGK reproduce byte for byte (Q1 item 6). Nothing
  indicates that any finished output is corrupt.
- All 16 BlankfreeGreedyPerJob per jobs of the pack (alias/sae/4a/blankfree/sae_4a_lexlat_v2_a18b/*
  /ep{1,2,4,8}/dev-other/per) are finished.
- ytNqcrGIw7tA and u1hlrBFgkw8M are keyinit-pack (ge1MKcAPmZIV) r30_dur seed-1 cv_holdout decodes
  at sub-epochs 18 and 41. They finished on 09-24 at 20:39 and 21:10. They are not runnable and not
  part of this pack.
- The executor report reports/exec_keyinit_error_2026-09-25.md has stale claims:
  - its "4 finished GenDecodeReportJobs" are unrelated 09-24 ladder reads;
  - ytNq, u1hl, PiYQ1OCFD4ot and G2NeV8oNr4tO are finished, not runnable or waiting.

## Q3: dependencies
- The manager's final state was error 1 (GTAGKejuQTxD) and waiting 2:
  - GenDecodeReportJob.wGgGstGW3XpG, the r30_dur ep8 report, which needs GTAGK;
  - DurinitBasinLiftReadJob uRzbIh0EfQXG, the lift read.
- The lift read is one job over ARMS=(gold_key, g_dur, r30_dur, r70_dur).
  - It asserts that epoch 8 is present in both pers[arm] and phi_reports[arm] for every arm, so it
    needs all 16 reports, including wGgGstGW3XpG.
  - Its LIFT / PARTIAL / NO LIFT verdict uses only the ep8 greedy PERs; the phi reports are
    report-only.
- The gold-key arm's own verdict inputs are finished: BlankfreeGreedyPerJob.bbrCBu3GYtJH per.json
  and the PairedPerDelta rows. But the registered reader for the gold-key arm cannot run until GTAGK
  and wGgG finish. So the A18 (b) read for a18_gold_key depends on the one errored job and on the
  unsubmitted report behind it, and on nothing else.
- Reading the gold-key band straight off per.json would be a derived statistic outside the
  registered reader. Per the project rules, that is not a read.

## Q4: fix spec and hashes
- No code or config change. This is an unchanged resubmit of GTAGKejuQTxD in the same hash dir,
  after an infrastructure failure, which under the project rules needs no new code review.
  1. Keep the crash evidence: copy log.run.1 and engine/ out of the job dir.
  2. Whoever is authorized clears the errored job and restarts the manager. Per the ops note, this is
     restart_manager_clear.sh, not just a console clear.
  3. If the gpupack engine allows it, exclude jpbo-028-30, or at least GPU1 there.
  4. wGgGstGW3XpG and uRzbIh0EfQXG then run unchanged.
  5. If the assert repeats on a different node, re-dispatch the debugger: that would falsify this
     verdict.
- Hash correction to the premise: blankfree_genmarg_jobs.py defines no `__sis_version__` (grep
  finds none). The only file-sha stamp in sae/emc is BlankfreeCostProfileJob in
  blankfree_train_jobs.py:274.
  - An edit to blankfree_genmarg_jobs.py therefore moves no Job hash: not GenMargSampleJob,
    GenerativeDecodeGapJob, GenMargSelectionJob, GenDecodeReportJob or the forward.
  - The forward's config refers to the module only by import string.
  - A hash-preserving code change is possible, but none is warranted: the code is correct. An edit
    would also silently change the behaviour behind already-finished hash dirs.
- If a runtime guard is wanted later, for example recomputing readd once on a mismatch before
  asserting, put it in a new module and register it deliberately.

## Artifacts (scratchpad)
/tmp/claude-34349/-e-project1-spell-wu24-2026-07-13-unsupervised/a2d26b34-6240-4bb2-b34b-ab87c54fe915/scratchpad/
- rnnrun, rnnrun2, rnnrun3: exact reruns of GTAGK.
- rnnrun_gk, rnnrun_g_dur, rnnrun_r70_dur: pack-mate reruns, byte-identical.
- repro.py, loop.py, tokens.py, durs.py, job_tags.txt.
