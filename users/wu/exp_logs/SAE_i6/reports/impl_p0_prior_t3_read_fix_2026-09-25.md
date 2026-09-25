# Implementation: T3 read.py fixed to the registered T3 reading (2026-09-25)

Status: DONE. `analysis/prior_t3/read.py` now implements the T3 reading registered in `SAE_i6_P0.md`
(Results, "### G0.R1 ctrl_20 at ep20", bullet "T3 reading"), and fixes findings 1-3 of
`reports/review_p0_prior_t3_2026-09-25.md`. It was tested on fabricated fixtures for every branch.
No other file was touched: common.py, run_part_a.sh, part_a_prior.py, build_config.py and probe.sbatch are
unchanged. They are used by the running Part A and Part B jobs. The new read.py was installed by an atomic
rename, with mode 644 as before. probe.sbatch:81 runs read.py at the end of Part B, so `read.txt` will come
from the fixed version.

## What changed (read.py only)

- **Control (finding 2).** Arm i passes if two conditions hold:
  - It is on ctrl_20's batch: num_seqs and max frames are string-equal.
  - l_tau, prior per token and expected tokens are within +-0.002 / +-0.005 / +-1.0 of ctrl_20's
    `log.run.1:670`. The bounds are inclusive, with a float slack of 1e-9.

  Every other field that differs is printed as "information, not gated". A failure prints `CONTROL FAIL`
  and `T3 VOID`, with no share verdict. The existing BATCH (both arms plus the epoch num_seqs line) and
  CONFIG gates are kept unchanged. If either fails, the read prints `T3 INVALID/NOT_READY`, with no verdict.
- **Outcome (finding 1).** `classify()` takes share = pp(arm ii) - pp(arm i) and applies the first rule
  that matches:
  1. PRIOR: arm ii is within -5.657 +-0.005.
  2. NOT PRIOR: |share| <= 0.005. The audio or the port explains the miss (debugger).
  3. PARTIAL: anything else. The read prints the share and its fraction of the observed gap.

  The gap is ctrl_20 - JUPITER = +0.020, computed from the log. The fraction is share / -gap, the share of
  the gap that the prior closes.

  If PRIOR and |share| <= 0.005 hold at once, a FLAG line prints both facts. That case cannot occur while
  CONTROL passes: arm i lies in [-5.642, -5.632], so |share| >= 0.010.

  The STEP1_TOL WITHIN/OUTSIDE clauses (+-0.01, l_tau, tokens +-3 %) and their import are removed.
- **Part A label (finding 3).** The verdict carries `[JUPITER prior]` only if `partA_summary.json` exists
  and lines_out, tokens_counted, held_raw_tokens and held_ppl_order3 are all `exact` in it. Each summary
  reference must also equal the value registered in read.py (39,630,169; 81,559,944; 808,146;
  9.561056344 to within 5e-10). Otherwise the label is `[JUPITER-like prior; DIFF <fields>]`, and a
  `PART A CORE NOT EXACT` line gives got and diff. A missing summary prints `PART A MISSING`, with no verdict.
- **Resolution note.** The output states that each value is printed at 3 decimals (+-0.0005), so a
  difference of two values is +-0.001.

New constants, all inside read.py: CONTROL_TOL, PRIOR_TOL, SHARE_NULL_TOL, EPS, PART_A_CORE and
RESOLUTION_NOTE.

## Assumptions

- **Part A exactness.** It uses the summary's own `exact` flag. That flag is ppl within 1e-9 relative, which
  is the criterion the review accepted. read.py adds a check that the summary's reference equals the
  registered value. It does not re-derive exactness at 9 decimals.
- **Boundaries.** All three ranges are inclusive: "within +-" and "<=".

## Checks

The harness is `$SC/t_read.py` (SC = scratch dir `.../scratchpad/t3read_fix`). It imports the installed
read.py and redirects `arm_dir` and `PART_A_SUMMARY_JSON` to fixtures under `$SC/fx/<case>/`, which it
builds as follows:
- The arm configs are ctrl_20's config with the model line changed, and in arm ii the prior path changed.
- The arm logs are ctrl_20's line 670 with fields edited.
- The Part A summaries are fabricated.

ctrl_20's log and config are only read. The run gave 12 of 12 expected outcomes, rc 0. Separately, a run of
read.py on the real paths now prints `PART A MISSING` and `T3 NOT_READY`, rc 0, because Part A and Part B
are still running. These checks exercise the read logic only. No T3 number exists yet.

### Test output (installed read.py)
```
=== control_fail_ltau: OK (expected 'T3 VOID')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
    control fails on: l_tau
  CONTROL FAIL
  CONTROL FAIL
  T3 VOID: arm i does not reproduce ctrl_20's step 0 within the registered tolerances; no share verdict
=== control_fail_batch: OK (expected 'T3 VOID')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH FAIL
    other fields that differ (information, not gated): none
    control fails on: max_size:time:var-unk:features
  CONTROL FAIL
  CONTROL FAIL
  T3 VOID: arm i does not reproduce ctrl_20's step 0 within the registered tolerances; no share verdict
=== control_pass_info_diffs: OK (expected 'T3 PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): agg: ctrl_20 1.789 vs arm i 1.790, blankfree_rate_fd_check: ctrl_20 7.701e-06 vs arm i 7.702e-06
  CONTROL PASS
  T3 PRIOR [JUPITER prior]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.642, arm ii -5.655, JUPITER -5.657; share (arm ii - arm i) -0.013 = 65% of the +0.020 gap (ctrl_20 -5.637 - JUPITER); residual arm ii - JUPITER +0.002; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== prior_partA_exact: OK (expected 'T3 PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER prior]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.637, arm ii -5.655, JUPITER -5.657; share (arm ii - arm i) -0.018 = 90% of the +0.020 gap (ctrl_20 -5.637 - JUPITER); residual arm ii - JUPITER +0.002; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== prior_edge_-5.652: OK (expected 'T3 PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER prior]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.637, arm ii -5.652, JUPITER -5.657; share (arm ii - arm i) -0.015 = 75% of the +0.020 gap (ctrl_20 -5.637 - JUPITER); residual arm ii - JUPITER +0.005; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== not_prior: OK (expected 'T3 NOT PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 NOT PRIOR [JUPITER prior]: |share| <= 0.005 and arm ii outside -5.657 +-0.005: the prior does not explain the miss; the audio or the port does (debugger); arm i -5.637, arm ii -5.640, JUPITER -5.657; share (arm ii - arm i) -0.003 = 15% of the +0.020 gap (ctrl_20 -5.637 - JUPITER); residual arm ii - JUPITER +0.017; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== not_prior_edge_share_-0.005: OK (expected 'T3 NOT PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 NOT PRIOR [JUPITER prior]: |share| <= 0.005 and arm ii outside -5.657 +-0.005: the prior does not explain the miss; the audio or the port does (debugger); arm i -5.637, arm ii -5.642, JUPITER -5.657; share (arm ii - arm i) -0.005 = 25% of the +0.020 gap (ctrl_20 -5.637 - JUPITER); residual arm ii - JUPITER +0.015; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== partial_review_example_partA_inexact: OK (expected 'T3 PARTIAL [JUPITER-like prior; DIFF tokens_counted, held_ppl_order3]')
  CONFIG PASS
  PART A CORE NOT EXACT: DIFF tokens_counted (got 81559000, diff -944); held_ppl_order3 (got 9.5601, diff -0.0009563441113620286)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PARTIAL [JUPITER-like prior; DIFF tokens_counted, held_ppl_order3]: arm ii outside -5.657 +-0.005 and |share| > 0.005: partial share; arm i -5.637, arm ii -5.650, JUPITER -5.657; share (arm ii - arm i) -0.013 = 65% of the +0.020 gap (ctrl_20 -5.637 - JUPITER); residual arm ii - JUPITER +0.007; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== prior_partA_inexact: OK (expected 'T3 PRIOR [JUPITER-like prior; DIFF tokens_counted, held_ppl_order3]')
  CONFIG PASS
  PART A CORE NOT EXACT: DIFF tokens_counted (got 81559000, diff -944); held_ppl_order3 (got 9.5601, diff -0.0009563441113620286)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER-like prior; DIFF tokens_counted, held_ppl_order3]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.637, arm ii -5.657, JUPITER -5.657; share (arm ii - arm i) -0.020 = 100% of the +0.020 gap (ctrl_20 -5.637 - JUPITER); residual arm ii - JUPITER +0.000; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== partA_missing: OK (expected 'PART A MISSING')
  CONFIG PASS
  PART A MISSING
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  PART A MISSING: fx/partA_missing/partA_summary.json does not exist; no share verdict
=== classify, both facts at once (arm i -5.655, arm ii -5.657):
  PRIOR; FLAG: arm ii is within -5.657 +-0.005 AND |share| 0.002 <= 0.005 at once; this is impossible with arm i near -5.637 (arm i is -5.655); the registered order gives PRIOR
=== classify, review example (arm i -5.637, arm ii -5.650): PARTIAL; arm ii outside -5.657 +-0.005 and |share| > 0.005: partial share; arm i -5.637, arm ii -5.650, JUPITER -5.657; share (arm ii - arm i) -0.013 = 65% of the +0.020 gap (ctrl_20 -5.637 - JUPITER); residual arm ii - JUPITER +0.007

TOTAL UNEXPECTED: 0
```

### Diff (original read.py -> installed)
```diff
--- /var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/4186749c-0772-4a1a-a52d-b33a7b8fd5e1/scratchpad/t3read_fix/../read.py.orig	2026-09-25 19:52:41.588855645 +0200
+++ /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/prior_t3/read.py	2026-09-25 19:54:32.109188279 +0200
@@ -1,23 +1,35 @@
 #!/work/asr4/hwu/conda/envs/sae/bin/python
 """T3 read (Part B, with the Part A prior's stats).  Read-only.
 
+Implements the T3 reading registered in ``SAE_i6_P0.md`` (Results, "### G0.R1 ctrl_20 at ep20", bullet
+"T3 reading"), as fixed by ``reports/review_p0_prior_t3_2026-09-25.md`` findings 1-3.
+
 Prints:
   * ctrl_20's step-0 line (``log.run.1:670``) and each arm's first ``ep 1 train, step 0,`` line, verbatim
     (from the arm's ``returnn.log``, else ``rnn.out``), plus the ``ep 1 train num_seqs`` line and the GPU;
   * the config diffs ctrl_20 -> arm i and arm i -> arm ii (the latter must be the prior path only);
   * the batch (num_seqs, max frames) of ctrl_20 and both arms;
-  * the Part A checks (lines kept, tokens counted / held, ppl3 against JUPITER);
+  * the Part A checks (lines kept, tokens counted / held, ppl3 against JUPITER) and the PART A CORE label;
   * l_tau, prior per token and expected tokens for ctrl_20, both arms and JUPITER's banked step 1.
 
 Verdict lines:
-  CONTROL  PASS iff arm i's step-0 line equals ctrl_20's in every field except wall-clock (sec/step,
-           elapsed, exp. remaining), i.e. the V100 re-run reproduces ctrl_20's step 0 as printed.
-  BATCH    PASS iff ctrl_20 and both arms have the same num_seqs, max frames and epoch num_seqs.
   CONFIG   PASS iff diff(arm i, arm ii) is exactly the prior_npz_path line, and diff(ctrl_20, arm i) exactly
            the model line.
-  T3       only when all three pass (else INVALID, or NOT_READY if a line is missing): the prior's share of
-           the step-0 prior-per-token gap, share = arm ii - arm i, against the gap ctrl_20 - JUPITER; and
-           whether arm ii lies within JUPITER's step-1 values at the G0.R1 audio-label tolerances.
+  BATCH    PASS iff ctrl_20 and both arms have the same num_seqs, max frames and epoch num_seqs.
+  CONTROL  PASS iff arm i is on ctrl_20's batch (num_seqs and max frames equal) and its l_tau, prior per token
+           and expected tokens are within +-0.002 / +-0.005 / +-1.0 of ctrl_20's step-0 line (the registered
+           cross-hardware tolerances).  Other fields that differ are printed as information, not gated.
+           CONTROL FAIL -> "T3 VOID", no share verdict.
+  PART A   "JUPITER prior" iff partA_summary.json exists and the four core checks (lines_out, tokens_counted,
+           held_raw_tokens, held_ppl_order3) are all EXACT against the registered values; else "JUPITER-like
+           prior" with the DIFF fields.  Missing summary -> "PART A MISSING", no verdict.
+  T3       share = prior per token (arm ii) - prior per token (arm i); first match wins:
+             PRIOR      arm ii within -5.657 +-0.005 (the miss is owed to the prior);
+             NOT PRIOR  |share| <= 0.005 (the prior does not explain it: audio or port, debugger);
+             PARTIAL    otherwise, with the share and its fraction of the observed ctrl_20 - JUPITER gap.
+           If PRIOR and |share| <= 0.005 hold at once, both facts are printed and flagged.
+
+All step-line values are printed by RETURNN at 3 decimals, so each carries +-0.0005 rounding.
 
 Usage: read.py
 """
@@ -30,11 +42,25 @@
 
 sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 from common import (ARMS, CTRL20_CONFIG, CTRL20_LOG, CTRL20_STEP0_LINENO, JUPITER_STEP1,  # noqa: E402
-                    PART_A_SUMMARY_JSON, STEP0_PREFIX, STEP1_TOL, arm_dir)
+                    PART_A_SUMMARY_JSON, STEP0_PREFIX, arm_dir)
 
 WALLCLOCK = re.compile(r"^(?:[\d.]+ sec/step|elapsed \S+|exp\. remaining \S+)$")
 KEYS = ("l_tau", "blankfree_prior_per_token", "blankfree_expected_tokens")
 BATCH = ("num_seqs", "max_size:time:var-unk:features")
+PP = "blankfree_prior_per_token"
+
+#: registered control tolerances, arm i vs ctrl_20's step-0 line (SAE_i6_P0.md, T3 reading, "Control")
+CONTROL_TOL = {"l_tau": 0.002, "blankfree_prior_per_token": 0.005, "blankfree_expected_tokens": 1.0}
+#: registered outcome thresholds on prior per token (SAE_i6_P0.md, T3 reading, "The prior's share")
+PRIOR_TOL = 0.005        # arm ii within JUPITER's -5.657 +- this -> PRIOR
+SHARE_NULL_TOL = 0.005   # |share| <= this -> NOT PRIOR
+#: float slack for comparing 3-decimal printed values against the (inclusive) tolerances
+EPS = 1e-9
+#: registered Part A core values (SAE_i6_P0.md, T3 reading, "Part A"); ppl3 registered at 9 decimals
+PART_A_CORE = {"lines_out": 39_630_169, "tokens_counted": 81_559_944, "held_raw_tokens": 808_146,
+               "held_ppl_order3": 9.561056344}
+RESOLUTION_NOTE = ("resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of "
+                   "two of them is +-0.001")
 
 
 def fields(line: str) -> dict:
@@ -66,6 +92,66 @@
     return "".join(d), body
 
 
+def within(a: float, b: float, tol: float) -> bool:
+    return abs(a - b) <= tol + EPS
+
+
+def part_a_label():
+    """(state, label, lines): state MISSING / EXACT / INEXACT; label for the verdict line."""
+    if not os.path.exists(PART_A_SUMMARY_JSON):
+        return "MISSING", None, [f"  NO {PART_A_SUMMARY_JSON}", "PART A MISSING"]
+    pa = json.load(open(PART_A_SUMMARY_JSON))
+    lines = []
+    by_q = {c["quantity"]: c for c in pa.get("checks", [])}
+    for c in pa.get("checks", []):
+        if c["quantity"] in ("lines_out", "sil_tokens", "tokens", "tokens_counted", "held_raw_tokens",
+                             "held_ppl_order3"):
+            lines.append(f"  {c['quantity']:18s} {c['got']!r:>22} vs JUPITER {c['jupiter']!r:>22}  "
+                         f"{'EXACT' if c['exact'] else 'DIFF ' + repr(c['diff'])}")
+    lines.append(f"  prior.npz {pa.get('prior_npz')} sha256 {str(pa.get('prior_npz_sha256'))[:16]}")
+    bad = []
+    for q, reg in PART_A_CORE.items():
+        c = by_q.get(q)
+        if c is None:
+            bad.append(f"{q} (missing from summary)")
+            continue
+        ref_ok = (c["jupiter"] == reg) if isinstance(reg, int) else (abs(c["jupiter"] - reg) < 5e-10)
+        if not ref_ok:
+            bad.append(f"{q} (summary reference {c['jupiter']!r} != registered {reg!r})")
+        elif not c["exact"]:
+            bad.append(f"{q} (got {c['got']!r}, diff {c['diff']!r})")
+    if bad:
+        lines.append("PART A CORE NOT EXACT: DIFF " + "; ".join(bad))
+        return "INEXACT", "JUPITER-like prior; DIFF " + ", ".join(b.split()[0] for b in bad), lines
+    lines.append("PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)")
+    return "EXACT", "JUPITER prior", lines
+
+
+def classify(pp_ctrl: float, pp_i: float, pp_ii: float, jup: float) -> tuple:
+    """Registered T3 outcome on prior per token: (outcome, text, flag or None)."""
+    share = pp_ii - pp_i
+    gap = pp_ctrl - jup                  # the observed G0.R1 gap, ctrl_20 - JUPITER (+0.020)
+    frac = share / -gap if gap else float("nan")   # fraction of the gap the prior closes
+    resid = pp_ii - jup
+    is_prior = within(pp_ii, jup, PRIOR_TOL)
+    is_null = abs(share) <= SHARE_NULL_TOL + EPS
+    nums = (f"arm i {pp_i:.3f}, arm ii {pp_ii:.3f}, JUPITER {jup:.3f}; share (arm ii - arm i) {share:+.3f} "
+            f"= {frac:.0%} of the {gap:+.3f} gap (ctrl_20 {pp_ctrl:.3f} - JUPITER); residual arm ii - JUPITER "
+            f"{resid:+.3f}")
+    flag = None
+    if is_prior and is_null:
+        flag = (f"FLAG: arm ii is within {jup:.3f} +-{PRIOR_TOL} AND |share| {abs(share):.3f} <= {SHARE_NULL_TOL} "
+                f"at once; this is impossible with arm i near {pp_ctrl:.3f} (arm i is {pp_i:.3f}); the "
+                f"registered order gives PRIOR")
+    if is_prior:
+        return "PRIOR", (f"arm ii within {jup:.3f} +-{PRIOR_TOL}: the miss is owed to the prior; " + nums), flag
+    if is_null:
+        return "NOT PRIOR", (f"|share| <= {SHARE_NULL_TOL} and arm ii outside {jup:.3f} +-{PRIOR_TOL}: the prior "
+                             f"does not explain the miss; the audio or the port does (debugger); " + nums), flag
+    return "PARTIAL", (f"arm ii outside {jup:.3f} +-{PRIOR_TOL} and |share| > {SHARE_NULL_TOL}: partial share; "
+                       + nums), flag
+
+
 def main():
     verdicts = {}
     # ---- ctrl_20 reference
@@ -113,16 +199,8 @@
 
     # ---- Part A
     print("\n== Part A prior (JUPITER emulation) ==")
-    if os.path.exists(PART_A_SUMMARY_JSON):
-        pa = json.load(open(PART_A_SUMMARY_JSON))
-        for c in pa["checks"]:
-            if c["quantity"] in ("lines_out", "sil_tokens", "tokens", "tokens_counted", "held_raw_tokens",
-                                 "held_ppl_order3"):
-                print(f"  {c['quantity']:18s} {c['got']!r:>22} vs JUPITER {c['jupiter']!r:>22}  "
-                      f"{'EXACT' if c['exact'] else 'DIFF ' + repr(c['diff'])}")
-        print(f"  prior.npz {pa['prior_npz']} sha256 {pa['prior_npz_sha256'][:16]}")
-    else:
-        print(f"  NO {PART_A_SUMMARY_JSON}")
+    pa_state, pa_label, pa_lines = part_a_label()
+    print("\n".join(pa_lines))
 
     # ---- batch and control
     print("\n== batch ==")
@@ -134,15 +212,43 @@
         same = all(rows[a].get(k) == rows["ctrl_20"].get(k) for a in ARMS for k in BATCH)
         same_ep = all(ep_lines[a] == ep_lines["ctrl_20"] for a in ARMS)
         verdicts["BATCH"] = "PASS" if (same and same_ep) else "FAIL"
-        a0 = list(ARMS)[0]
-        mism = sorted(k for k in set(rows["ctrl_20"]) | set(rows[a0]) if rows["ctrl_20"].get(k) != rows[a0].get(k))
-        verdicts["CONTROL"] = "PASS" if not mism else "FAIL"
-        if mism:
-            print("  control mismatches: " + ", ".join(f"{k}: ctrl_20 {rows['ctrl_20'].get(k)} vs "
-                                                      f"{rows[a0].get(k)}" for k in mism))
     else:
-        verdicts["BATCH"] = verdicts["CONTROL"] = "NOT_READY"
-    print(f"BATCH {verdicts['BATCH']}\nCONTROL {verdicts['CONTROL']}")
+        verdicts["BATCH"] = "NOT_READY"
+    print(f"BATCH {verdicts['BATCH']}")
+
+    print("\n== control (arm i vs ctrl_20 step 0, registered tolerances) ==")
+    a0 = list(ARMS)[0]
+    if a0 in rows:
+        c, r0 = rows["ctrl_20"], rows[a0]
+        fails = []
+        for k in BATCH:
+            ok = r0.get(k) == c.get(k)
+            print(f"  {k:32s} ctrl_20 {c.get(k)!s:>10} arm i {r0.get(k)!s:>10}  must be equal: "
+                  f"{'OK' if ok else 'DIFFERENT'}")
+            if not ok:
+                fails.append(k)
+        for k, tol in CONTROL_TOL.items():
+            try:
+                vc, va = float(c[k]), float(r0[k])
+                ok = within(va, vc, tol)
+                d = f"{va - vc:+.3f}"
+            except (KeyError, ValueError):
+                ok, d = False, "missing"
+            print(f"  {k:32s} ctrl_20 {c.get(k)!s:>10} arm i {r0.get(k)!s:>10}  diff {d} (tol +-{tol:g}): "
+                  f"{'OK' if ok else 'OUTSIDE'}")
+            if not ok:
+                fails.append(k)
+        other = sorted(k for k in set(c) | set(r0)
+                       if k not in CONTROL_TOL and k not in BATCH and c.get(k) != r0.get(k))
+        print("  other fields that differ (information, not gated): "
+              + (", ".join(f"{k}: ctrl_20 {c.get(k)} vs arm i {r0.get(k)}" for k in other) if other else "none"))
+        print(f"  {RESOLUTION_NOTE}")
+        verdicts["CONTROL"] = "FAIL" if fails else "PASS"
+        if fails:
+            print(f"  control fails on: {', '.join(fails)}")
+    else:
+        verdicts["CONTROL"] = "NOT_READY"
+    print(f"CONTROL {verdicts['CONTROL']}")
 
     # ---- the three step-1 numbers
     print("\n== step 0 (= step 1) numbers ==")
@@ -150,30 +256,32 @@
     print(f"  {'':28s}" + "".join(f"{n:>14s}" for n in names) + f"{'JUPITER':>14s}")
     for k in KEYS:
         print(f"  {k:28s}" + "".join(f"{rows[n].get(k, '-'):>14s}" for n in names) + f"{JUPITER_STEP1[k]:>14.3f}")
+    print(f"  {RESOLUTION_NOTE}")
 
-    if all(verdicts.get(v) == "PASS" for v in ("CONFIG", "BATCH", "CONTROL")):
-        a_i, a_ii = list(ARMS)
-        pp = {n: float(rows[n]["blankfree_prior_per_token"]) for n in ("ctrl_20", a_i, a_ii)}
-        jup = JUPITER_STEP1["blankfree_prior_per_token"]
-        gap = pp[a_i] - jup
-        share = pp[a_ii] - pp[a_i]
-        resid = pp[a_ii] - jup
-        print(f"\n  prior per token: gap arm i - JUPITER = {gap:+.3f}; prior share arm ii - arm i = {share:+.3f}; "
-              f"residual arm ii - JUPITER = {resid:+.3f}")
-        within = []
-        for k in KEYS:
-            kind, tol = STEP1_TOL[k]
-            v, ref_v = float(rows[a_ii][k]), JUPITER_STEP1[k]
-            d = abs(v - ref_v) if kind == "abs" else abs(v - ref_v) / abs(ref_v)
-            within.append(f"{k} {v:.3f} vs {ref_v:.3f} +-{tol:g}{' rel' if kind == 'rel' else ''}: "
-                          f"{'WITHIN' if d <= tol else 'OUTSIDE'}")
-            print(f"  arm ii {within[-1]}")
-        print(f"\nT3 VALID: prior share {share:+.3f} of the {gap:+.3f} prior-per-token gap; residual {resid:+.3f}; "
-              + "; ".join(w.split(':')[0].split()[0] + ' ' + w.split(': ')[-1] for w in within))
-    else:
-        bad = {k: v for k, v in verdicts.items() if v != "PASS"}
+    # ---- verdict
+    print()
+    if verdicts["CONTROL"] == "FAIL":
+        print("CONTROL FAIL\nT3 VOID: arm i does not reproduce ctrl_20's step 0 within the registered tolerances; "
+              "no share verdict")
+        return
+    bad = {k: v for k, v in verdicts.items() if v != "PASS"}
+    if bad:
         state = "NOT_READY" if all(v == "NOT_READY" for v in bad.values()) else "INVALID"
-        print(f"\nT3 {state}: {bad}")
+        print(f"T3 {state}: {bad}; no share verdict")
+        return
+    if pa_state == "MISSING":
+        print(f"PART A MISSING: {PART_A_SUMMARY_JSON} does not exist; no share verdict")
+        return
+    a_i, a_ii = list(ARMS)
+    try:
+        pp = {n: float(rows[n][PP]) for n in ("ctrl_20", a_i, a_ii)}
+    except (KeyError, ValueError):
+        print(f"T3 INVALID: {PP} missing or unparsable in a step-0 line; no share verdict")
+        return
+    outcome, text, flag = classify(pp["ctrl_20"], pp[a_i], pp[a_ii], JUPITER_STEP1[PP])
+    if flag:
+        print(flag)
+    print(f"T3 {outcome} [{pa_label}]: {text}; {RESOLUTION_NOTE}")
 
 
 if __name__ == "__main__":
```

## Fix round 2

Status: DONE_WITH_CONCERNS. This round applies the orchestrator's decisions on the three findings in
`reports/review_p0_prior_t3_read_fix_2026-09-25.md`. Only `analysis/prior_t3/read.py` changed. It was
installed by an atomic rename (copy to `.read.py.new`, then `chmod 644` and `mv`) at 20:04:06, while Part B
4365700 was still PENDING on its dependency. Before the swap, `cmp` confirmed that the installed file was
still the round-1 version.

### What changed
1. `classify` (finding 1). PRIOR and NOT PRIOR are checked first, as before. After them:
   - arm ii < -5.657 - 0.005 (below -5.662) prints `T3 OUTSIDE REGISTERED OUTCOMES: overshoot`;
   - a share opposite in sign to arm i's own gap (share x (JUPITER - arm i) < 0), with |share| > 0.005
     already implied because NOT PRIOR did not match, prints `T3 OUTSIDE REGISTERED OUTCOMES: wrong sign`;
   - everything else is PARTIAL. What remains is exactly this: the share has the gap's sign and
     -5.662 < arm ii.

   The two outside lines print arm i, arm ii, JUPITER, the share and the residual. They carry no
   percentage and no "owed" or "partial share" wording. The Part A label stays in brackets after the
   outcome, as on every T3 line.
2. `part_a_label` (finding 2). read.py now judges the core checks itself and no longer uses Part A's
   `exact` flag:
   - the three counts must satisfy `got == registered`;
   - held_ppl_order3 must satisfy `|got - 9.561056344| < 5e-10`.

   Otherwise the label is `JUPITER-like prior; DIFF <q>`. The DIFF detail now prints got, the registered
   value and the difference. The existing check that the summary reference equals the registered value
   is unchanged.
3. Percentage (finding 3). The share is divided by arm i's own gap, JUPITER - arm i, i.e.
   -5.657 - arm i. It is labelled "= N% of arm i's own gap (JUPITER - arm i = X)". The ctrl_20 - JUPITER
   gap is no longer printed; ctrl_20's value still appears in the FLAG text. The docstring was updated to
   match.

### Assumptions
- The overshoot test is written literally as arm ii < JUPITER - 0.005. It matches "on the far side" only
  while arm i is above JUPITER. That always holds when classify runs, because CONTROL PASS puts arm i in
  [-5.642, -5.632].
- The outside lines keep only the share and the residual. The percentage was dropped from them because on
  those lines it would read as an attribution.

### Concern (not changed; outside the delta)
The information line for each check, `read.py:109-110`, still shows Part A's own flag. With ppl
9.561056350, it prints `held_ppl_order3 9.56105635 vs JUPITER 9.561056344111362  EXACT`, and on the next
line `PART A CORE NOT EXACT: DIFF held_ppl_order3 ...`. The verdict label is correct (JUPITER-like), but
the display line can mislead a reader. A possible fix is to relabel that column "Part A flag", or to
compute it with read.py's own test. This is for the orchestrator to decide.

### Checks
- The harness is `$SC2/t_read.py`, where SC2 is `.../scratchpad/t3read_fix2`. It is the round-1 harness
  plus 11 cases and two stricter tests:
  - the outside lines must contain no `%`, "owed" or "partial share";
  - the percentage case must print "100% of arm i's own gap (JUPITER - arm i = -0.015)".
- It was run against the scratch copy and then against the installed file, with a fresh fixture root.
  Both runs gave 21 of 21 cases as expected, plus the two classify checks, with TOTAL UNEXPECTED 0 and
  rc 0.
- read.py on the real paths gives `PART A MISSING` and `T3 NOT_READY: {CONFIG, BATCH, CONTROL
  NOT_READY}; no share verdict`, rc 0. At that point Part A 4365697 is running and Part B is pending.
- These checks exercise the read logic only. No T3 number exists yet.

### Test output (installed read.py)
```
=== control_fail_ltau: OK (expected 'T3 VOID')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
    control fails on: l_tau
  CONTROL FAIL
  CONTROL FAIL
  T3 VOID: arm i does not reproduce ctrl_20's step 0 within the registered tolerances; no share verdict
=== control_fail_batch: OK (expected 'T3 VOID')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH FAIL
    other fields that differ (information, not gated): none
    control fails on: max_size:time:var-unk:features
  CONTROL FAIL
  CONTROL FAIL
  T3 VOID: arm i does not reproduce ctrl_20's step 0 within the registered tolerances; no share verdict
=== control_pass_info_diffs: OK (expected 'T3 PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): agg: ctrl_20 1.789 vs arm i 1.790, blankfree_rate_fd_check: ctrl_20 7.701e-06 vs arm i 7.702e-06
  CONTROL PASS
  T3 PRIOR [JUPITER prior]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.642, arm ii -5.655, JUPITER -5.657; share (arm ii - arm i) -0.013 = 87% of arm i's own gap (JUPITER - arm i = -0.015); residual arm ii - JUPITER +0.002; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== prior_partA_exact: OK (expected 'T3 PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER prior]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.637, arm ii -5.655, JUPITER -5.657; share (arm ii - arm i) -0.018 = 90% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.002; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== prior_edge_-5.652: OK (expected 'T3 PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER prior]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.637, arm ii -5.652, JUPITER -5.657; share (arm ii - arm i) -0.015 = 75% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.005; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== not_prior: OK (expected 'T3 NOT PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 NOT PRIOR [JUPITER prior]: |share| <= 0.005 and arm ii outside -5.657 +-0.005: the prior does not explain the miss; the audio or the port does (debugger); arm i -5.637, arm ii -5.640, JUPITER -5.657; share (arm ii - arm i) -0.003 = 15% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.017; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== not_prior_edge_share_-0.005: OK (expected 'T3 NOT PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 NOT PRIOR [JUPITER prior]: |share| <= 0.005 and arm ii outside -5.657 +-0.005: the prior does not explain the miss; the audio or the port does (debugger); arm i -5.637, arm ii -5.642, JUPITER -5.657; share (arm ii - arm i) -0.005 = 25% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.015; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== partial_review_example_partA_inexact: OK (expected 'T3 PARTIAL [JUPITER-like prior; DIFF tokens_counted, held_ppl_order3]')
  CONFIG PASS
  PART A CORE NOT EXACT: DIFF tokens_counted (got 81559000, registered 81559944, diff -944); held_ppl_order3 (got 9.5601, registered 9.561056344, diff -0.000956344000000442)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PARTIAL [JUPITER-like prior; DIFF tokens_counted, held_ppl_order3]: arm ii strictly between the bands (-5.662 < arm ii, on the -5.657 side of arm i) and |share| > 0.005: partial share; arm i -5.637, arm ii -5.650, JUPITER -5.657; share (arm ii - arm i) -0.013 = 65% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.007; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== prior_partA_inexact: OK (expected 'T3 PRIOR [JUPITER-like prior; DIFF tokens_counted, held_ppl_order3]')
  CONFIG PASS
  PART A CORE NOT EXACT: DIFF tokens_counted (got 81559000, registered 81559944, diff -944); held_ppl_order3 (got 9.5601, registered 9.561056344, diff -0.000956344000000442)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER-like prior; DIFF tokens_counted, held_ppl_order3]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.637, arm ii -5.657, JUPITER -5.657; share (arm ii - arm i) -0.020 = 100% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.000; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== partA_missing: OK (expected 'PART A MISSING')
  CONFIG PASS
  PART A MISSING
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  PART A MISSING: /var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/4186749c-0772-4a1a-a52d-b33a7b8fd5e1/scratchpad/t3read_fix2/fx/partA_missing/partA_summary.json does not exist; no share verdict
=== wrong_sign_-5.620: OK (expected 'T3 OUTSIDE REGISTERED OUTCOMES: wrong sign [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 OUTSIDE REGISTERED OUTCOMES: wrong sign [JUPITER prior]: share opposite in sign to arm i's own gap (JUPITER - arm i = -0.020) and |share| > 0.005; arm i -5.637, arm ii -5.620, JUPITER -5.657; share (arm ii - arm i) +0.017; residual arm ii - JUPITER +0.037; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== wrong_sign_edge_-5.631: OK (expected 'T3 OUTSIDE REGISTERED OUTCOMES: wrong sign [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 OUTSIDE REGISTERED OUTCOMES: wrong sign [JUPITER prior]: share opposite in sign to arm i's own gap (JUPITER - arm i = -0.020) and |share| > 0.005; arm i -5.637, arm ii -5.631, JUPITER -5.657; share (arm ii - arm i) +0.006; residual arm ii - JUPITER +0.026; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== overshoot_-5.670: OK (expected 'T3 OUTSIDE REGISTERED OUTCOMES: overshoot [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 OUTSIDE REGISTERED OUTCOMES: overshoot [JUPITER prior]: arm ii below -5.662; arm i -5.637, arm ii -5.670, JUPITER -5.657; share (arm ii - arm i) -0.033; residual arm ii - JUPITER -0.013; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== overshoot_edge_-5.663: OK (expected 'T3 OUTSIDE REGISTERED OUTCOMES: overshoot [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 OUTSIDE REGISTERED OUTCOMES: overshoot [JUPITER prior]: arm ii below -5.662; arm i -5.637, arm ii -5.663, JUPITER -5.657; share (arm ii - arm i) -0.026; residual arm ii - JUPITER -0.006; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== prior_edge_-5.662: OK (expected 'T3 PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER prior]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.637, arm ii -5.662, JUPITER -5.657; share (arm ii - arm i) -0.025 = 125% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER -0.005; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== partial_edge_-5.643: OK (expected 'T3 PARTIAL [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PARTIAL [JUPITER prior]: arm ii strictly between the bands (-5.662 < arm ii, on the -5.657 side of arm i) and |share| > 0.005: partial share; arm i -5.637, arm ii -5.643, JUPITER -5.657; share (arm ii - arm i) -0.006 = 30% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.014; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== partial_edge_-5.651: OK (expected 'T3 PARTIAL [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PARTIAL [JUPITER prior]: arm ii strictly between the bands (-5.662 < arm ii, on the -5.657 side of arm i) and |share| > 0.005: partial share; arm i -5.637, arm ii -5.651, JUPITER -5.657; share (arm ii - arm i) -0.014 = 70% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.006; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== ppl_9.561056350_flag_exact: OK (expected 'T3 PRIOR [JUPITER-like prior; DIFF held_ppl_order3]')
  CONFIG PASS
  PART A CORE NOT EXACT: DIFF held_ppl_order3 (got 9.56105635, registered 9.561056344, diff 5.999998720085387e-09)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER-like prior; DIFF held_ppl_order3]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.637, arm ii -5.655, JUPITER -5.657; share (arm ii - arm i) -0.018 = 90% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.002; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== ppl_9.5610563444_flag_exact: OK (expected 'T3 PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER prior]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.637, arm ii -5.655, JUPITER -5.657; share (arm ii - arm i) -0.018 = 90% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.002; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== lines_out_off_by_1_flag_exact: OK (expected 'T3 PRIOR [JUPITER-like prior; DIFF lines_out]')
  CONFIG PASS
  PART A CORE NOT EXACT: DIFF lines_out (got 39630170, registered 39630169, diff 1)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER-like prior; DIFF lines_out]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.637, arm ii -5.655, JUPITER -5.657; share (arm ii - arm i) -0.018 = 90% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.002; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== pct_arm_i_-5.642_on_jupiter: OK (expected 'T3 PRIOR [JUPITER prior]')
  CONFIG PASS
  PART A CORE ALL EXACT (lines_out, tokens_counted, held_raw_tokens, held_ppl_order3)
  BATCH PASS
    other fields that differ (information, not gated): none
  CONTROL PASS
  T3 PRIOR [JUPITER prior]: arm ii within -5.657 +-0.005: the miss is owed to the prior; arm i -5.642, arm ii -5.657, JUPITER -5.657; share (arm ii - arm i) -0.015 = 100% of arm i's own gap (JUPITER - arm i = -0.015); residual arm ii - JUPITER +0.000; resolution: RETURNN prints these values at 3 decimals, so each is +-0.0005; a difference of two of them is +-0.001
=== classify, both facts at once (arm i -5.655, arm ii -5.657):
  PRIOR; FLAG: arm ii is within -5.657 +-0.005 AND |share| 0.002 <= 0.005 at once; this is impossible with arm i near -5.637 (arm i is -5.655); the registered order gives PRIOR
=== classify, review example (arm i -5.637, arm ii -5.650): PARTIAL; arm ii strictly between the bands (-5.662 < arm ii, on the -5.657 side of arm i) and |share| > 0.005: partial share; arm i -5.637, arm ii -5.650, JUPITER -5.657; share (arm ii - arm i) -0.013 = 65% of arm i's own gap (JUPITER - arm i = -0.020); residual arm ii - JUPITER +0.007

TOTAL UNEXPECTED: 0
```

### Diff (round-1 read.py -> installed)
```diff
--- /var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/4186749c-0772-4a1a-a52d-b33a7b8fd5e1/scratchpad/t3read_fix2/read.py.round1	2026-09-25 19:54:32.109188279 +0200
+++ /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/prior_t3/read.py	2026-09-25 20:04:06.695826649 +0200
@@ -21,12 +21,18 @@
            cross-hardware tolerances).  Other fields that differ are printed as information, not gated.
            CONTROL FAIL -> "T3 VOID", no share verdict.
   PART A   "JUPITER prior" iff partA_summary.json exists and the four core checks (lines_out, tokens_counted,
-           held_raw_tokens, held_ppl_order3) are all EXACT against the registered values; else "JUPITER-like
-           prior" with the DIFF fields.  Missing summary -> "PART A MISSING", no verdict.
+           held_raw_tokens, held_ppl_order3) are all EXACT against the registered values, judged here and not
+           by Part A's own ``exact`` flag: the three counts equal, and |ppl - 9.561056344| < 5e-10 (equal at
+           the banked 9 decimals); else "JUPITER-like prior" with the DIFF fields.  Missing summary ->
+           "PART A MISSING", no verdict.
   T3       share = prior per token (arm ii) - prior per token (arm i); first match wins:
              PRIOR      arm ii within -5.657 +-0.005 (the miss is owed to the prior);
              NOT PRIOR  |share| <= 0.005 (the prior does not explain it: audio or port, debugger);
-             PARTIAL    otherwise, with the share and its fraction of the observed ctrl_20 - JUPITER gap.
+             OUTSIDE REGISTERED OUTCOMES: overshoot    arm ii below -5.662; share printed, no attribution;
+             OUTSIDE REGISTERED OUTCOMES: wrong sign   share opposite in sign to arm i's own gap
+                        (-5.657 - arm i) and |share| > 0.005; share printed, no attribution;
+             PARTIAL    share has the sign of arm i's own gap and -5.662 < arm ii (strictly between the two
+                        bands), with the share and its percentage of arm i's own gap (-5.657 - arm i).
            If PRIOR and |share| <= 0.005 hold at once, both facts are printed and flagged.
 
 All step-line values are printed by RETURNN at 3 decimals, so each carries +-0.0005 rounding.
@@ -116,10 +122,12 @@
             bad.append(f"{q} (missing from summary)")
             continue
         ref_ok = (c["jupiter"] == reg) if isinstance(reg, int) else (abs(c["jupiter"] - reg) < 5e-10)
+        # judged here against the registered value, not by Part A's own (relative 1e-9) ``exact`` flag
+        got_ok = (c["got"] == reg) if isinstance(reg, int) else (abs(c["got"] - reg) < 5e-10)
         if not ref_ok:
             bad.append(f"{q} (summary reference {c['jupiter']!r} != registered {reg!r})")
-        elif not c["exact"]:
-            bad.append(f"{q} (got {c['got']!r}, diff {c['diff']!r})")
+        elif not got_ok:
+            bad.append(f"{q} (got {c['got']!r}, registered {reg!r}, diff {c['got'] - reg!r})")
     if bad:
         lines.append("PART A CORE NOT EXACT: DIFF " + "; ".join(bad))
         return "INEXACT", "JUPITER-like prior; DIFF " + ", ".join(b.split()[0] for b in bad), lines
@@ -130,14 +138,14 @@
 def classify(pp_ctrl: float, pp_i: float, pp_ii: float, jup: float) -> tuple:
     """Registered T3 outcome on prior per token: (outcome, text, flag or None)."""
     share = pp_ii - pp_i
-    gap = pp_ctrl - jup                  # the observed G0.R1 gap, ctrl_20 - JUPITER (+0.020)
-    frac = share / -gap if gap else float("nan")   # fraction of the gap the prior closes
+    gap_i = jup - pp_i                   # arm i's own gap, JUPITER - arm i (both on this hardware's batch)
+    frac = share / gap_i if gap_i else float("nan")   # share as a fraction of arm i's own gap
     resid = pp_ii - jup
     is_prior = within(pp_ii, jup, PRIOR_TOL)
     is_null = abs(share) <= SHARE_NULL_TOL + EPS
-    nums = (f"arm i {pp_i:.3f}, arm ii {pp_ii:.3f}, JUPITER {jup:.3f}; share (arm ii - arm i) {share:+.3f} "
-            f"= {frac:.0%} of the {gap:+.3f} gap (ctrl_20 {pp_ctrl:.3f} - JUPITER); residual arm ii - JUPITER "
-            f"{resid:+.3f}")
+    base = (f"arm i {pp_i:.3f}, arm ii {pp_ii:.3f}, JUPITER {jup:.3f}; share (arm ii - arm i) {share:+.3f}")
+    tail = f"; residual arm ii - JUPITER {resid:+.3f}"
+    nums = (base + f" = {frac:.0%} of arm i's own gap (JUPITER - arm i = {gap_i:+.3f})" + tail)
     flag = None
     if is_prior and is_null:
         flag = (f"FLAG: arm ii is within {jup:.3f} +-{PRIOR_TOL} AND |share| {abs(share):.3f} <= {SHARE_NULL_TOL} "
@@ -148,8 +156,16 @@
     if is_null:
         return "NOT PRIOR", (f"|share| <= {SHARE_NULL_TOL} and arm ii outside {jup:.3f} +-{PRIOR_TOL}: the prior "
                              f"does not explain the miss; the audio or the port does (debugger); " + nums), flag
-    return "PARTIAL", (f"arm ii outside {jup:.3f} +-{PRIOR_TOL} and |share| > {SHARE_NULL_TOL}: partial share; "
-                       + nums), flag
+    # outside the registered outcomes: printed with the share and no attribution
+    if pp_ii < jup - PRIOR_TOL:
+        return "OUTSIDE REGISTERED OUTCOMES: overshoot", (
+            f"arm ii below {jup - PRIOR_TOL:.3f}; " + base + tail), flag
+    if share * gap_i < 0:
+        return "OUTSIDE REGISTERED OUTCOMES: wrong sign", (
+            f"share opposite in sign to arm i's own gap (JUPITER - arm i = {gap_i:+.3f}) and |share| > "
+            f"{SHARE_NULL_TOL}; " + base + tail), flag
+    return "PARTIAL", (f"arm ii strictly between the bands ({jup - PRIOR_TOL:.3f} < arm ii, on the {jup:.3f} side "
+                       f"of arm i) and |share| > {SHARE_NULL_TOL}: partial share; " + nums), flag
 
 
 def main():
```
