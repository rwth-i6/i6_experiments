# Review: T3 read.py rewrite against the registered T3 reading (2026-09-25)

Verdict: PASS_WITH_FIXES. The rewritten `analysis/prior_t3/read.py` implements the registered reading
(`SAE_i6_P0.md`, "### G0.R1 ctrl_20 at ep20", bullet "T3 reading") for the following:
- the control tolerances and the same-batch test;
- the VOID path;
- the outcome order and bounds;
- PART A MISSING;
- the log parsing.

Three narrow defects remain. None needs compute, and each can be fixed after Part B, because read.py only
reads. Rerun it by hand after a fix (probe.sbatch:81 runs it only once).

Inputs:
- the criterion `SAE_i6_P0.md:430-443`;
- `reports/review_p0_prior_t3_2026-09-25.md` (findings 1-3);
- `reports/impl_p0_prior_t3_read_fix_2026-09-25.md`;
- `analysis/prior_t3/{read.py, common.py, probe.sbatch, part_a_prior.py:391-443, run_part_a.sh}`;
- ctrl_20 `ReturnnTrainingJob.GiT88bxzoZbZ` (`log.run.1:669-670`, `work/returnn.log`, `output/returnn.config`);
- the earlier probes' `returnn.log` and Slurm stdout under `/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs{1,2}`;
- the scratch original `read.py.orig`.

## Findings

1. **`read.py:151-152` (docstring `:29`): PARTIAL also catches cases that are not "between".** The
   registered text is: "anything between is reported as a partial share". `classify` returns PARTIAL for
   everything that is neither PRIOR nor NOT PRIOR. That includes two shapes of result that do not lie
   between "not explained" and "owed to the prior":
   - a share of the wrong sign, where arm ii is above -5.632 and the swap moves prior per token away from
     JUPITER;
   - an overshoot, where arm ii is below -5.662.

   Tested:
   - Arm ii at -5.620 prints `T3 PARTIAL ...: partial share; ... share +0.017 = -85% of the +0.020 gap`.
   - Arm ii at -5.680 prints `T3 PARTIAL ... share -0.043 = 215%`.
   - Arm ii at -5.663 prints PARTIAL at 130 %.

   The verdict word the user is told is "partial share" for a result the registration does not cover. This
   is reachable: the Part A prior is expected to be only JUPITER-like, so its step-0 value can land on
   either side.

   Fix: keep the order and the registered bounds. For share > +0.005 or arm ii < -5.662, print the outcome
   as outside the registered outcomes, naming the wrong sign or the overshoot, and do not print "partial
   share". The orchestrator reads that case; the registration is not amended.
2. **`read.py:121` (with `part_a_prior.py:412`): `[JUPITER prior]` is looser than the registered ppl.**
   - The registration says: JUPITER's prior "only if it gives EXACTLY ... ppl 9.561056344".
   - read.py takes the summary's `exact` flag, which is |got - ref| <= 1e-9 x ref, about 9.6e-9 absolute. That
     is about 19 times the 5e-10 of agreement at 9 decimals.
   - Tested: a summary with ppl 9.561056350 and exact counts (diff 5.9e-9, so `exact` is True) prints
     `T3 PRIOR [JUPITER prior]`, although 9.561056350 != 9.561056344. The registered label is JUPITER-like.
   - When this can happen: a few rare-word pronunciations differ at equal length, so lines, tokens counted
     and held tokens stay exact while ppl moves in the ninth decimal.
   - Fix: in read.py, require |got - 9.561056344| < 5e-10 for held_ppl_order3. The reference check at
     `:118` already uses that bound.
3. **`read.py:133-134`: the printed fraction's denominator changed, which no finding asked for.**
   - `read.py.orig` computed the gap as arm i - JUPITER. The rewrite uses ctrl_20 (L40S) - JUPITER, while
     the share (arm ii - arm i) is measured on the V100.
   - The class is unaffected: the registration defines no fraction. The percentage on the verdict line is
     affected.
   - Tested: arm i -5.642 with arm ii -5.657 prints "PRIOR ... 75% of the gap", although arm ii lands exactly
     on JUPITER. Arm i -5.632 with arm ii -5.657 prints 125 %. The control allows +-0.005, which is +-25
     points.
   - Fix: state the denominator. Print the fraction against arm i - JUPITER, or print both, labelled.

## Checked and sound

- **Control** (`:219-251`, `CONTROL_TOL :53`).
  - It uses +-0.002 / +-0.005 / +-1.0 on `l_tau`, `blankfree_prior_per_token` and
    `blankfree_expected_tokens` against the values parsed from ctrl_20 `log.run.1:670`. Those values are
    -0.352 / -5.637 / 63.854, the registered numbers.
  - The bounds are inclusive with EPS 1e-9.
  - Tested on both sides of each bound, all correct: l_tau -0.354 and -0.350 pass, -0.355 and -0.349 fail;
    prior per token -5.642 and -5.632 pass, -5.643 and -5.631 fail; tokens 64.854 and 62.854 pass, 64.855
    and 62.853 fail.
  - Same batch: num_seqs (128) and max frames (330) must be string-equal. num_seqs 127 or max frames 331
    gives VOID.
  - The other fields that differ are printed and not gated.
- **VOID path** (`:263-266`): it returns before any share or Part A line. It takes precedence over
  NOT_READY and INVALID.
- **Outcome order and bounds** (`:136-152`), with arm i at -5.637:
  - -5.662 and -5.652 give PRIOR; -5.663 and -5.651 give PARTIAL.
  - -5.642 and -5.632 give NOT PRIOR; -5.643 gives PARTIAL.
  - The PRIOR and NOT PRIOR conditions cannot both hold while the control passes: the gap between the two
    ranges is at least 0.010.
- **Part A label** (`:99-127`).
  - It needs all four core quantities, and each summary reference must equal the registered value.
  - A missing quantity, or lines_out off by 1, gives `JUPITER-like prior; DIFF <q>`.
  - A missing summary gives `PART A MISSING` and no verdict (`:272-274`).
- **Parsing.**
  - The prefix is `ep 1 train, step 0,`; `ep 10` cannot match it.
  - The arm dirs `PROBE_ROOT/{i_i6prior,ii_partA}` equal probe.sbatch `ROOT`/`ARMS`, in order i, ii.
  - A RETURNN `returnn.log` has the same unprefixed step lines as stdout (checked on the earlier probes).
  - ctrl_20's `work/returnn.log` step-0 line is byte-identical to `log.run.1:670`.
  - The `rnn.out` fallback was tested.
  - CONFIG requires ctrl_20 -> arm i to differ in the model line only, and arm i -> arm ii in the prior path
    only.
- **Partial logs.** None of these crashes or prints a share verdict:
  - no step-0 line in either arm, or in both: NOT_READY;
  - arm ii line truncated: INVALID (BATCH FAIL);
  - arm i line truncated: VOID;
  - arm ii config missing: NOT_READY.

  Two cases are unreachable in the pipeline. RETURNN writes each step line in a single record, so a
  truncated line cannot occur in practice. A truncated `partA_summary.json` raises JSONDecodeError and
  prints no verdict; Part B runs `afterok` on Part A, and the summary is written in one `json.dump` at the
  end.
- **Fixtures and real paths.**
  - The implementer's harness reproduces 12 of 12 into a fresh fixture root.
  - 40 of my own cases gave the results above: `scratchpad/review_t3read/t_review.{py,out}`.
  - read.py on the real paths prints `PART A MISSING` and `T3 NOT_READY: {CONFIG, BATCH, CONTROL
    NOT_READY}; no share verdict`, rc 0. At that time, Part A 4365697 was running and Part B 4365700 was
    pending on its dependency.
- **Scope.** The reported diff equals `diff read.py.orig read.py`. No other file in `analysis/prior_t3`
  changed after the first review: the latest mtime of the others is 19:37, and the review is 19:51.
- **Part A.** Job 4365697 is a full run, not a smoke run. Its scan read 40,418,261 lines and kept
  39,630,169.

Nothing was edited. All test fixtures were written under the session scratchpad.

## Round 2 (2026-09-25, read.py installed 20:04:06)

Verdict: PASS. Each of the three findings is fixed as the orchestrator decided. No registered bound moved,
and nothing else changed.

- **Scope.** I ran `diff` between the round-1 file (identical to the file reviewed above) and the installed
  read.py.
  - The only changes are in the docstring, `part_a_label` (`got_ok`, DIFF text) and `classify`. This equals
    the diff in the "Fix round 2" section of the implementation report.
  - The following are unchanged: `CONTROL_TOL` (:59), `PRIOR_TOL` and `SHARE_NULL_TOL` 0.005 (:61-62),
    `EPS` (:64), `PART_A_CORE` (:66-67), `main()` and every other file in `analysis/prior_t3`.
  - Nothing parses the verdict string in `read.txt`; probe.sbatch:82 only tails it.
- **Finding 1 is fixed.** PRIOR and NOT PRIOR are tested first, with unchanged conditions. After that:
  - arm ii < -5.662 gives `OUTSIDE REGISTERED OUTCOMES: overshoot`;
  - share x (-5.657 - arm i) < 0 gives `OUTSIDE REGISTERED OUTCOMES: wrong sign`;
  - PARTIAL is what remains, which is exactly -5.652 < arm ii < arm i - 0.005, the region between the two
    bands.

  The OUTSIDE lines contain no "%", "owed" or "partial share" (0 of 7 lines).
- **Finding 2 is fixed.** `got_ok` (:126) requires the three counts to be equal and |ppl - 9.561056344| < 5e-10.
  - 9.561056350 and 9.5610563446 give JUPITER-like.
  - JUPITER's full 9.561056344111362 and 9.5610563436 give `[JUPITER prior]`.
- **Finding 3 is fixed.** The percentage is share / (-5.657 - arm i).
  - Arm i -5.642 with arm ii -5.657 prints 100 %; arm i -5.632 with arm ii -5.657 prints 100 %.
  - Arm i -5.642 with arm ii -5.648 prints 40 %.
- **Tests.** I reran my 40 cases and added 9, 49 in all (`scratchpad/review_t3read/t_review2.py`, output
  `t_review_r2.out`).
  - The control, VOID, NOT_READY and INVALID outcomes are unchanged.
  - The only changes are the intended ones: -5.663 and -5.680 now give overshoot, and -5.631 and -5.620 now
    give wrong sign.
  - The truncated summary JSON still raises JSONDecodeError and prints no verdict. As before, this cannot
    happen in the pipeline.
  - On the real paths, read.py prints `T3 NOT_READY`, rc 0.
- **The display line (`read.py:113-116`; the line that prints EXACT is 116): yes, it can mislead.** When
  5e-10 < |ppl - 9.561056344| <= about 9.6e-9, it prints `held_ppl_order3 ... EXACT`, while the verdict says
  `[JUPITER-like prior; DIFF held_ppl_order3]`. Tested with 9.561056350 and 9.5610563446.
  - The line directly below it (`PART A CORE NOT EXACT: DIFF held_ppl_order3 ...`) and the verdict label are
    correct.
  - The reverse cannot happen: read.py's pass implies Part A's flag.
  - The counts cannot disagree, because both tests use equality.
  - `partA_summary.txt` from Part A (`part_a_prior.py:441-443`) says "ALL EXACT" in the same window.
