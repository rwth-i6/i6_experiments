# Launch review: SAE_4A_rename AN-0 (RenameEmStepJob.aPi62dEywaMl) -- 2026-09-25

VERDICT: APPROVE_WITH_CONDITIONS. The code does what the registration says: no silent no-op, the right
key, no label in E/M. But the registered read at lambda = 10 is confounded. At lambda 10 the scaled prior
collapses the segmentation, so an "AN-0 DEAD" is close to predetermined and would not support the rule's
own conclusion. The orchestrator has to decide this before launch, because amending a gate is legitimate
only before its result exists.

Inputs reviewed: commit 4a086c81 (rename_emstep_jobs.py, test_rename_emstep.py,
config_sae_4a_rename_an0_v1.py), the shim config/sae_4a_rename_an0.py, SAE_4A_rename.md l.79-133, and
blankfree_emtable.py (TableReverseModel, e_step_batch, m_step). Also read: blankfree_genmarg_jobs._setup,
lattice._prior_term/_forward_ctx, SaeBlankfreeModelV1.prior_weight_at, phi_from_key.key_table/load_key,
phi_relabel_s.relabel_state, key_search_jobs.agreement, unit_key.unit_frame_counts, and
key_oracle_names_jobs l.20-38. The build report is reports/impl_rename_an0_2026-09-25.md; the design review
is reports/design_review_rename_2026-09-25.md (F1).

Re-run by the reviewer (login GH200; outputs in the session scratchpad only; none of this is the
registered row):
- test --part unit: PASS.
- test --part graph: PASS. The graph has 5 jobs, 4 of them finished. The only unfinished one is
  RenameEmStepJob.aPi62dEywaMl, routed to gpupack with 1 GPU, 4 CPU, 24 GB, 0.5 h and slot width 1.
- test --part real (24 utterances, 5-pair seed 2): PASS.
- A 24-utterance probe on the undeployed gold-key row at lambda 1, 2, 4.4 and 10.
- A check that the 260 S tags are all in the units shards: 260 of 260. They share no tag with the
  28,254 train segments or with the 300-utterance subset.

## Findings (most severe first)

1. rename_emstep_jobs.py:120 and :307-310 (spec l.89-93). AN-0 reads rho at lambda 10 only. At lambda 10
   the operator collapses the segmentation, whatever the names. In the reviewer's 24-utterance probe on the
   undeployed gold-key row, prior_weight scaled the per-token prior cost, which acts as a token penalty:

   | lambda | expected tokens per frame | pi(SIL) | gold identity after the step |
   |---|---|---|---|
   | 1 | 0.236 | 0.090 | 0.980 |
   | 2 | 0.215 | 0.095 | 0.973 |
   | 4.4 | 0.159 | 0.121 | 0.955 |
   | 10 | 0.067 | 0.336 | 0.518 |

   At lambda 10, 235 of 500 units change key, against 12 at lambda 1. In the implementer's seed-2
   deranged row, rho(10) = -0.308 and the mean share moved back to the correct name is 0.013.

   The collapse happens inside each utterance, so it should persist at 300 utterances. The job's own
   reference row will show it. rho(10) on the read row is then dominated by about -0.3 to -0.5 of
   collateral loss, and "AN-0 DEAD" follows whether or not any swapped name can move. The rule text says
   "no LM-weighted E-step moves a whole symbol". It would then be printed and applied to a regime that
   lies outside the proposal's own range: "no label-free lambda0 above 4.4" (l.105). The design review
   (F1) predicted "partial movement at 4.4, and a return at 10" and did not model this token-count effect.
   The implementation follows the registration; the defect is in the registered pre-check.

   Options, all pre-result and for the orchestrator to choose:
   (a) Add lambda 2 and 4.4 to AN-0. That costs 2 more E-steps, about 1 GPU-minute. It needs
       LAMBDAS["AN-0"] and AN0_RULE changed, since the job asserts both.
   (b) Pre-register an interpretability guard, for example: DEAD is not read if the undeployed gold row
       loses more than a stated amount of identity at the read lambda.
   (c) Launch as registered and record now that a DEAD read under the collapse does not license dropping
       TP-A1.

2. rename_emstep_jobs.py:200-203, with config l.90 (seed 1). The registered seed-1 5-pair derangement is
   AH<->T, AO<->M, K<->OY, N<->Y, P<->S. The gold key has no units on OY, so PhiFromKey gives OY a
   uniform row (1/500). One of the "5 name pairs" therefore moves K's rows onto an empty name that is also
   rare under the LM, and leaves a uniform row at K. This is not a swap of two real phones:
   - Moving K back has a lower channel cost. Under the uniform row K's frames score 0.002 per frame,
     against 0.00038 for a swapped real phone, about 1.7 nats per frame easier.
   - The wrong name OY carries a far larger LM penalty than a real phone would.
   - K's frame share is 0.0269, more than half of the 0.05 bar.
   The real test skipped this case (test_rename_emstep.py:219 asserts no empty symbol moves) and ran
   seed 2, so the configuration of the registered row was never exercised. Its reference check
   "gold - deranged identity = moved share" will not be exact on this row.

   The key-convention side effect is small. The uniform row at K takes a unit from its own row in
   key_before for at most 0.0003 / 0.0007 / 0.0015 / 0.0025 of train weight at pi(K) = 0.03 / 0.05 /
   0.1 / 0.2 (computed from counts only). The issue is the stimulus, not the key. The spec allows it:
   derangements are over the non-SIL symbols, and OY is non-SIL. It is an undetermined value that the
   build report did not surface for seed 1. Decide whether to keep it as registered and disclose it, or
   to amend now to exclude OY and ZH from pair draws.

3. reports/impl_rename_an0_2026-09-25.md:14. The launch command uses the relative path
   `sis_env/bin/python`, which does not exist in the setup dir, so it fails at once and costs no compute.
   Use /e/project1/spell/wu24/env/sis_env/bin/python, as manager 3019650 does.

4. rename_emstep_jobs.py:483-487 vs :508-511 (report-only; it enters no reading). S1_before scores the
   phi table smoothed twice (0.81 ML + 0.19/500). S1_after scores 0.9 m1 + 0.1/500 once. dS1 therefore
   mixes a change of smoothing with the step. Do not read dS1 as the step's effect.

## Checks asked in the dispatch

1. prior_weight reaches the lattice, and tau and smoothing are as A12's:
   - The path is model.prior_weight -> prior_weight_at(1) (l.742-751) -> gm._setup dp["prior_weight"]
     (l.291) -> lattice_forward_backward -> _forward_ctx -> _prior_term = (1/tau) * beta * log P. The
     matmul path builds its matrices from the scaled term (lattice.py l.829-853).
   - _Bed asserts no schedule and base weight 1.0, and resets to 1.0 in a finally block. score() asserts
     1.0.
   - Re-run: the attribute route gives a log Z bitwise equal to a model built with prior_weight = 10, and
     log Z differs from lambda 1. The probe above shows the effect is large.
   - TAU = 1.0 is passed to e_step_batch. SMOOTHING = 0.1 goes to build_scoring_model, and set_tables
     recomputes log_emis_eff.
2. Derangement:
   - The relabelled table and durations are asserted equal to the source rows permuted by g (l.476-477).
   - check_derangement asserts SIL fixed, 10 symbols moved, no fixed point, and an involution.
   - key_before uses log_m0 of the deranged phi (l.495).
3. Key and identity:
   - The gold key comes from GoldUnitKeyJob.sLnMRRd2qO0t; its id and f0jaGuiJVe6A are asserted.
   - The weights are unit_frame_counts over train.segments, which is A20's convention.
   - key = argmax m(u|s) pi(s), with pi the same E-step's frame share, and rho = after - before.
   - After the step, pi cancels: key_after(u) is approximately argmax_s N(s,u).
   - OY/ZH on the reference row: the effect is at most about 0.004 (0.9958 at lambda 10 in the test). It
     cannot move the read against 0.05. The seed-1 K<->OY stimulus is finding 2.
   - The 9 units missing from the 300 utterances are re-keyed to the symbol with the largest N(s), which
     is SIL at lambda 10. The bias is one-directional, toward DEAD, and at most -0.0038. It can move the
     read only if rho(10) lands in [0.050, 0.054).
4. M-step:
   - m1 = (N + 1e-3) / rowsum is the closed form. Durations use em.m_step 'durinit' and enter S_1 only.
   - No table is written; the json carries keys and pi only.
   - The gold key enters only agreement, rename_moves and moved_share.
5. Graph and resources:
   - Graph as re-run above.
   - Nothing unfinished is shared with the managers of pids 2467260 (keyinit) and 3019650 (keyarms). The
     finished shared inputs are f0jaGuiJVe6A and sLnMRRd2qO0t.
   - Resources are ample. The expected GPU peak is about 14 GiB (A12's batches). The probe's RSS was
     2.2 GB against a 24 GB request, and the run should take a few minutes against a 0.5 h limit.
6. Rules and verdict lines:
   - AN0_RULE is verbatim in the phase file (unit PASS).
   - The outcome is "AN-0 DEAD" if rho(10) < 0.05, else "AN-0 OPEN". It is printed at the head of the
     report and again before the RULE, APPLIED and CONVENTION blocks. rho is finite by construction,
     so no NaN can pass the comparison.
