# Audit: fidelity of the SAE_i6 reference distillation against the frozen JUPITER phase-4A logs

Date 2026-09-24. Auditor: fresh context, read-only (no document edited). Scope: the seven distilled documents
`SAE_i6_ref.md`, `_objective`, `_blankfree`, `_lexicon`, `_lexlat_v2`, `_emc1`, `_emc2` in
`recipe/i6_experiments/users/wu/exp_logs/SAE_i6/`, checked against the gold sources in `exp_logs/SAE/`
(`SAE_4A*.md`, `SAE_ref.md`, `SAE.md`, `reports/`). Method: every distilled number below was searched
verbatim (fixed-string grep with GNU grep) in the sources and read in context; where the sources give
a rounded value only, rounding agreement counts as MATCH and is marked so.

## Verdict

**CONFIRMED_WITH_CORRECTIONS.** 193 table rows (single numbers or same-statement number groups, several hundred values) checked across all seven documents:
**188 MATCH, 4 MISMATCH, 1 UNTRACEABLE**. No gate verdict is misstated; every overturned or withdrawn
reading that the sources mark is marked as such in the distillation. The four mismatches are minor
(one audit status, one inherited sign label, one pre-audit range, one cost range). There are no
remnants (job hashes, Slurm ids, node names, cluster paths). Five omissions matter for future work (see
"Omissions"), and there are three document defects that are not source-fidelity errors (see "Other defects").

Note: `SAE_i6_ref_objective.md` changed WHILE this audit ran (mtime 15:14:55; 330 -> 356 lines). A new
section 10 was added from the i6 port code review. It is not JUPITER content and was not audited for
fidelity, but it makes the file internally inconsistent (see "Other defects" 2).

## Mismatches (distilled file:line; distilled value; source value and location)

1. `SAE_i6_ref_blankfree.md:244`: "The item's audit was pending at the source's last entry" (attribution
   step 6 item 2, E4). Source `SAE_4A_attrib.md:770-771`, the E4 result entry itself: "Audited from a fresh
   context (`reports/sae_attrib_step6_e4_audit_2026-09-20.md`, DONE_WITH_CONCERNS): every KL reproduced".
   Only the stale State line (`SAE_4A_attrib.md:23-25`) says "audit pending". The status should read
   "audited DONE_WITH_CONCERNS; reads unresolved on both rules".
2. `SAE_i6_ref_lexicon.md:108`: "The seed band ctrl_20_s1 − ctrl_20 at ep20 is −0.001 [−0.004, +0.001]".
   The primary read `SAE_4A_prepro.md:232` is **ctrl_20 − ctrl_20_s1** = −0.001 [−0.004, +0.001]
   (ctrl_20 0.8746 < ctrl_20_s1 0.8751, `SAE_4A_lexlat.md:498`). The distilled line faithfully copies
   `SAE_4A_prior.md:582`, which itself has the direction reversed. The effect is negligible (CI spans 0),
   but the sign label is wrong.
3. `SAE_i6_ref_lexicon.md:331` (D4): "Gold − decoded: ... −800 to −1190 at ep20". `SAE_4A_lexlat.md:568`
   gives −800 to −1190 "in every arm and epoch" (ep4-ep20). The fresh-context audit of the write-up
   (`SAE_4A_lexlat.md:582`) corrects the ep20 range to **945-1187** nats, "no longer growing in the
   treatments". (`SAE_i6_ref_lexlat_v2.md:9` quotes "800-1190" from `SAE_4A_lexlat_v2.md:25`, which is a faithful copy.)
4. `SAE_i6_ref_emc1.md:51`: "Bigram: 1.48-1.67 s per step". `SAE_4A.md:632` gives 1.648 s as the measured
   base step and `SAE_4A.md:665` gives 1.48 s/step for the bigram. The upper bound 1.67 is not in the
   source (1.65 would be).

## Untraceable

1. `SAE_i6_ref.md:95` (and `SAE_i6_ref_emc2.md:122`): gold-phi supervised reverse init, held-out NLL per
   frame **3.2888** at epoch 8. This number is in no JUPITER log or report. It appears only in the package
   (`experiments/unsupervised_asr/README.md:176`, `config/supervised_init.py:7`). The JUPITER logs give
   S = 3.4735 / 3.4702 for the refit gold phi on the CV holdout (a different set and quantity) and 3.38357
   for the S2g CTC-era fit. It cannot be verified from the sources. Its provenance should be stated as
   "package-banked".
   Related: the unrounded banked PERs 0.874568 / 0.873490 / 0.818615 / 0.823991 (`SAE_i6_ref.md:89-93`) are
   also only in the package README. The logs give 0.8746 / 0.8735 / 0.8186 / 0.8240, which agree to
   rounding, so these are counted as MATCH.

## Gate verdicts (item 2 of the brief)

All named gates state the source verdict:
- G4a.1 PASS (`_emc1` section 3; `SAE_4A.md:254`).
- G4a.2: S2b/S2c refine rows, and "nothing usable" (`SAE_4A.md:493-513`). S2e C, S2f U_joint and S2g
  REFINE, and no comparable usability verdict exists (`SAE_4A.md:1571-1572`).
- G4a.3 FAIL in S3, the blank-free first model, S3c, the DP64 control, S3d and M512.
- G4a.3b-R/-C/-BT/-CT: CLOSED FAIL.
- G4a.S2d: S2d FAIL in all arms. C IMPROVE, D PER-IMPROVE only, A and B fail HOLD, U_joint IMPROVE,
  ablations HOLD, S2g IMPROVE.
- G4a.4 FAIL (N = 50, six arms; N = 100 killed and never read). G4a.5 FAIL (4 arms). G4a.6 never run
  (cdrev deferred). G4a.7 never read at ep20. G4a.8 FAIL.
- G4a.9: CANNOT_TELL as written (ep4 clause), with PASS unreachable (clause 2 positive in all four pairs).
  Audited, matches `SAE_4A_lexlat.md` VERDICT.
- E60: continuity pass, PLATEAU in all four arms, form not PASS.
- G4a.L2.1: probe NO WAVE SETTING, then the A10 re-derivation gives durinit, 12 sub-epochs.
- G4a.L2.2: count-table family CANNOT_TELL (NO ELIGIBLE NULL); the wave is unread.
- G4a.L2.3 CANNOT_TELL. G4a.L2.4 unread. rho*_lift 0.7.
- A14 (ii) PHONETIC BASIN LOWER (audit correction on MFA durations carried). A16 (a) OBJECTIVE LABEL-BLIND
  OR WRONG, with the audit correction. A16 (a2) NO LAMBDA <= 3, with "model error" withdrawn. A16 (b)
  stage 0 J SEES THE KEY, fragile.
- D-series classes (D12 ROOM; D13 PHONES LOWER; D14 HELPS / LEXICON-SPECIFIC / FROZEN BETTER; D15 NOT
  NEEDED / not LEXICON-SPECIFIC / warm TRIGRAM NEEDED; D16 TWO BASINS; D17 HELPS AT THE EDGE / NO EFFECT;
  D8 CANNOT_TELL) all match.

Overturned or withdrawn readings carried and marked:
- step-5b "within half a nat" (overturned);
- S3b "not a relabelled phone code" (retracted);
- the "fixed point = R's posterior, ceiling 0.16-0.19" reading (overturned);
- S2d "settles at an optimum" (downgraded);
- "the labeling is already prior-best" (withdrawn);
- the stationary-point account (superseded from sub-epoch 10);
- D2 "not converged" (overturned by E60);
- D10 "added weight protects" (overturned by D10e);
- L2-0 "r100 HOLD, fitted phi suffices" (withdrawn);
- A15 "manner-class level" and the r100 fit comparison (overturned by A15-F);
- A16 (a2) "model error" (withdrawn);
- the positive-control FAIL, kept on record as mis-specified;
- the code-review entropy values, replaced by the registered reads.

No standing conclusion is presented that the source overturned.

## Omissions of consequence (item 3)

1. **Two user standing constraints are missing from `SAE_i6_ref.md` section 2.**
   - The min-duration topology d_min >= 2 is a standing user ruling, "a given, not revisitable"
     (`SAE.md:40-41`; `SAE_4A.md:246`). The distillation carries it only as a bed constant.
   - "No EMC stage launches with a bigram prior" (user, `SAE_4A.md:248-250`) appears only in `_emc1`
     section 1. A cost-motivated port could otherwise drop to a bigram.
2. **"No full joint cold restarts"** (`SAE_4A_lexlat_v2.md:31`; cold_ctl and controls exempt) is not in
   `SAE_i6_ref.md` section 2. That list does carry the alpha = 0 and neural split-merge items from the
   same source constraint, and it labels them "do not reopen without the user". The source records them
   as phase design constraints; only SylCipher, K4 and cdrev were user rulings.
3. **Runs that were live at the move are not flagged as launched.** `SAE_4A_lexlat_v2.md` State lists them
   as LIVE or submitted on JUPITER on 2026-09-24:
   - the L2-1 wave (6 packs);
   - the A14 (i) pack;
   - the A17 (i) pack and the A17 (ii) runs;
   - the keyinit pack (gold-key control plus A17 (iii)) and the A18 (b) lift pack in its graph;
   - the stage-1 key search.

   A19 was built and awaiting the user's shim. `SAE_i6_ref_lexlat_v2.md` "Open at the move" says only
   "no result exists". Outputs may exist on JUPITER storage after the freeze, so they should be checked
   before any i6 rerun is funded.
4. **An untested budget lever is dropped.** More updates per epoch through a smaller batch (max_seqs
   128 -> 32), or a GAN-scale update count, is listed in `SAE_4A_budget.md:116-121` as a lever not tested.
   D6's audit-corrected update deficit is 130x / 166x (`SAE_4A_lexlat.md:582`). E60 refuted "too short"
   only up to 3,420 updates, against the GAN's 148,000. The `_blankfree` section 10 "Untested" list and
   `_lexicon` D6 omit this. Minor to moderate.
5. **The user's idea of an explicit end-of-word symbol in the phone set** (`SAE.md:120-122`, 2026-09-21,
   a follow-up candidate, never run) is not carried. Minor.

Checked and carried (no omission):
- the prior-window defect and the standing n-gram-sample decision;
- the masked-feature bed ruling;
- the supervised-10 h open user decision;
- the lexlat open user forks and D12 pending arm;
- the L2 handoff items;
- the k2 pitfalls (int32 overflow, #0 loops, tropical determinization);
- CV-holdout overlap;
- repeat handling;
- the G4a.2 usability caveat;
- the "8 h / 24 h" caps (as JUPITER history);
- float64 DP on the blank-free bed (`reports/survey_sampled_prior_term_2026-09-20.md:10`: float64=True).

## Other defects (not source-fidelity errors)

1. `SAE_i6_ref_objective.md:328` cites `SAE_i6_ref_emc.md`, which does not exist (the files are `_emc1` / `_emc2`).
2. `SAE_i6_ref_objective.md:3-4` still claims the note is "carried over unchanged in substance". The newly
   added section 10 (lines 332-356) states that the tau = 2 minimizer is NOT exactly the Bayes posterior,
   because tempering acts on joint (path, segmentation) assignments. The following statements remain
   unannotated:
   - section 4.1, table row "tau 2 ... exactly q = post";
   - section 4.1, bullet "tau = 2 is the unique temperature at which the lattice term and the bound share
     their minimizer";
   - section 6, item 1.

   The note now contradicts itself. A pointer to section 10 is needed at those places.
3. The scope and cross-references of the cold-PER summaries are imprecise.
   - `SAE_i6_ref.md:104` says "PER 0.81-0.94 over every cold arm of both beds". This holds for gate and
     final reads only: the ep1 reads of odmprior_50 and odmbt_50 are 0.793 (`SAE_4A_budget.md:173-178`),
     and early CTC-era reads reach 0.97-1.0.
   - `SAE_i6_ref_blankfree.md:141-142` presents the separate supervised 10 h fit as "neither fit ran,
     open user decision". It does not point to the later blank-free supervised recognizer p0 (0.1894) and
     the blank-free gold-phi refit, which the lexlat phase trained (`_lexicon` B10, D10e).

## Remnants (item 4)

None. There are no job hashes, Slurm ids, node names, work/output paths or cluster file-system paths in
the seven documents (regex scan for `Job.<12 chars>`, bare 12-char hashes, slurm, jpbo/juwels, /p/, work/,
output/, pids, managers, watchers). The JUPITER operating conditions that remain are labelled as
reference hardware or history:
- GH200 96 GB, one GPU per arm;
- 11.5 h node allocation, exclusive 4-GPU node;
- 8 h and 24 h caps.

They are not i6 constraints (i6: L40S 46 GB). The flat-theta k2 memory peaks of 80-84 GB
(`_lexlat_v2` A13) will not fit an L40S.

## Table of checked numbers

Abbreviations for distilled files: R = `SAE_i6_ref.md`, O = `_objective`, B = `_blankfree`, L = `_lexicon`,
V = `_lexlat_v2`, E1 = `_emc1`, E2 = `_emc2`. Source abbreviations: A = `SAE_4A.md`, BF = `SAE_4A_blankfree.md`,
AT = `SAE_4A_attrib.md`, BU = `SAE_4A_budget.md`, IM = `SAE_4A_infomax.md`, PP = `SAE_4A_prepro.md`,
CD = `SAE_4A_cdrev.md`, PR = `SAE_4A_prior.md`, LX = `SAE_4A_lexlat.md`, L2 = `SAE_4A_lexlat_v2.md`,
OB = `SAE_4A_objective.md`, REF = `SAE_ref.md`.

| # | distilled | value | source | verdict |
|---|---|---|---|---|
| 1 | R:28 | GAN 0.214, seeds 0.168-0.215 | AT:35-36 | MATCH |
| 2 | R:48 | N = 20 for new arms | REF:14; BU:219-228 | MATCH |
| 3 | R:49 | 19 h projection, > 11.5 h without finishing sub-epoch 1 | A:2913-2914 | MATCH |
| 4 | R:60 | 28,539 utts; 28,254 train | BF:326; AT:579 | MATCH |
| 5 | R:61 | retained 15,427,853 of 18,088,388 | PP:201-202 | MATCH |
| 6 | R:62 | PCA-96 + k-means K = 500 | PP:75 | MATCH |
| 7 | R:64 | held-out trigram ppl 9.56 | REF:167 | MATCH |
| 8 | R:65 | rho 9.6619 | BF:112 | MATCH |
| 9 | R:68 | d in [2, 25], SIL 50 | BF:98 | MATCH |
| 10 | R:70 | band abs(s-3t) <= 25, float64 | BF:97; LX:53 | MATCH |
| 11 | R:70 | 3 x rate + 0.1 x agg | BF:112-113 | MATCH |
| 12 | R:71 | tau 8 -> 2 over 20 %, then 2 | BU:70 | MATCH |
| 13 | R:71 | Adam (0.5, 0.98), clip 5, 1e-4 / 3e-3 | BF:113-114 | MATCH |
| 14 | R:72 | 88,000 frames / 128 seqs; 57 updates | BU:48 | MATCH |
| 15 | R:73 | warmup to 2, hold 12, decay 20; kept 1/4/10/20 | BU:220-223 | MATCH |
| 16 | R:74 | dev-other 2864 / 33 / 177,275 | AT:689 | MATCH |
| 17 | R:78 | content-free band 0.83-0.91 | BU:35 | MATCH |
| 18 | R:80 | JSD4 gold ~0.25, GAN 0.28, cold 0.69-0.79 | LX:564; AT:303; PR:719-731 | MATCH |
| 19 | R:81-82 | noise 0.01-0.03 (0.013-0.020); s1 0.001 | LX:884; PP:232 | MATCH |
| 20 | R:89 | ctrl_20 0.855 / 0.875 / 0.869 / 0.874568 | PP:226; LX:498 (0.8746) | MATCH (rounding) |
| 21 | R:90 | step 1: -0.350 / -5.657 / 63.821 | LX:490 | MATCH |
| 22 | R:91 | ctrl_20_x60 0.873490 | LX:721 (0.8735) | MATCH (rounding) |
| 23 | R:92 | k2lat_20_ma3000 0.818615; -0.0560 [-0.0628, -0.0499] | LX:498, 507 | MATCH (rounding) |
| 24 | R:93 | k2lat_20_ma3000_x60 0.823991 | LX:723 (0.8240) | MATCH (rounding) |
| 25 | R:94 | off4_k2lat_20 -0.0399 [-0.0464, -0.0340] | LX:509 | MATCH |
| 26 | R:95 | gold phi held-out NLL 3.2888 | not in logs (package README:176) | UNTRACEABLE |
| 27 | R:96 | ctrl_20 601-610 s per sub-epoch | LX:43, 700 | MATCH |
| 28 | R:97 | 760.6 s; ~32 GiB rung 3000 | LX:700; LX:440 (31.77) | MATCH |
| 29 | R:110 | k2 lowers cold PER 0.03-0.06 | LX:582 | MATCH |
| 30 | R:113 | up to 70 % label noise lifts | L2:478-486 | MATCH |
| 31 | R:116 | EM degrades gold 0.193 -> 0.353 | L2:290 | MATCH |
| 32 | R:120 | WER 24.91 vs 26.50 | A:1871, 498 | MATCH |
| 33 | R:131 | 25 of 285 CV utterances | L2:123 | MATCH |
| 34 | R:132 | repeat floor 0.56 %; 533 words | L2:713-715 | MATCH |
| 35 | B:17-20 | dev-clean 2703; 831,372 of 968,057; dev-other 781,130 of 919,980 | BF:326; PP:201-202 | MATCH |
| 36 | B:19 | rVADfast 0.0.5 (declares 0.0.3), thr 0.4, 25 / 10 ms | BF:73-74 | MATCH |
| 37 | B:23 | LM corpus 39,630,169 lines | REF:158 | MATCH |
| 38 | B:24 | 1,010,000 lines, seed 0, every 101st held | REF:156-167 | MATCH |
| 39 | B:25 | rho 9.6619373279; 2.7 words/s | BF:112; A:~960 | MATCH |
| 40 | B:28-29 | BN scale 30; kernel 9, stride 3, pad 4, dil 1 | BF:70-71 | MATCH |
| 41 | B:37 | 240 x 500 table; 0.37 M MLP; 2 dur / 3 pos buckets | CD:11-12 | MATCH |
| 42 | B:45 | checkpoint stride 32 | BF:115 | MATCH |
| 43 | B:53-54 | lam_rate 3, FD eps 0.25; EMA 0.99, lam_agg 0.1 | BF:112-113; A:~3280 | MATCH |
| 44 | B:55 | tau (8, 5.04, 3.17, 2) | BF:112 | MATCH |
| 45 | B:57 | Adam eps 1e-6, wd 0 | BF:114 | MATCH |
| 46 | B:58 | grad norm ~0.1, clip inert | AT:656 | MATCH |
| 47 | B:59 | phi multiplier 30 (batch-8 reference) | BU:49; AT:668 | MATCH |
| 48 | B:61 | 57 updates / sub-epoch; 228 per pass | BU:6 | MATCH |
| 49 | B:63 | warmup 2 (5 % gives 1, rejected) | BU:221 | MATCH |
| 50 | B:64-67 | 1140 updates; 601 s; 3.3 h; 3 arms per 11.5 h | LX:559, 43; BU:223-224 | MATCH |
| 51 | B:69 | ctrl_20 vs s1 -0.001 [-0.004, +0.001] | PP:232 | MATCH |
| 52 | B:79 | positive controls +0.409, +0.696 | AT:622-623 | MATCH |
| 53 | B:94 | 73.5 / 20.7 / 5.0 % initials; P ~0.74; AH-first 58.1 % | AT:266-268; REF:161 | MATCH |
| 54 | B:96 | coverage KL 0.45 vs 0.83 | AT:587 | MATCH |
| 55 | B:112 | 18.24 s; 456 x 18.24 x 1.25 = 2.889 h; 8 h | BF:190 | MATCH |
| 56 | B:114 | 43m59s | BF:12, 202 | MATCH |
| 57 | B:118 | PER 0.834602 / 0.864877 | BF:220 | MATCH |
| 58 | B:120 | 120187 (6.53204) / 169036 (9.18694) | BF:214-215 | MATCH |
| 59 | B:121 | AH 14.08 %; N 12.94 %; types 39 / 38 | BF:216-217 | MATCH |
| 60 | B:123 | dev-clean 0.826548 / 0.848407 | BF:222 | MATCH |
| 61 | B:124 | gap +2.4808968 (own -4.4247514, donor -6.9056483); dev-clean +2.6842944 | BF:223-225 | MATCH |
| 62 | B:126-128 | P3 36 / P6 35 / 38; 4.3638 / 4.2363 / 4.389; AH 90.85, AA 77.65 | BF:250-252 | MATCH |
| 63 | B:135-141 | 2849 (2821 / 28); LR 5e-5, 24 passes; phi 3e-3, 8 passes, batch 8; 108 vs 131 | BF:270-276, 318-319 | MATCH |
| 64 | B:153-155 | ESS P3 395.00 / 384.14; P6 3.74 / 3.50; 106 / 165; 35 / 76 of 411 | BF:375-378 | MATCH |
| 65 | B:163-165 | 150k, 37552 s; S3b-OR 0.454 / 0.368 | AT:35-36, 49 | MATCH |
| 66 | B:176-177 | step 1: -6.2027 / 0.9359; -3.3424 / 0.7134; -3.0087 / 0.3021; -2.8898 / 0.2749; -0.334 [-0.347, -0.290] | AT:207-212 | MATCH |
| 67 | B:188-190 | norev 0.9357 +0.071; agg1 0.9202 +0.055; agg10 0.8986 +0.034 (CIs, gaps, first phone) | AT:235-237 | MATCH |
| 68 | B:194 | k64 0.9147, +0.050 [+0.039, +0.061], gap +0.75 | AT:252-253 | MATCH |
| 69 | B:201-202 | priorshuf ep1 0.8206, ep4 0.8829, +0.018 [+0.015, +0.022]; AH 0.003, HH 0.886 | AT:281-284 | MATCH |
| 70 | B:204-206 | JSD1 0.067 -> 0.026; JSD4 0.692 (gold 0.225); -3.611 vs -2.55 | AT:301-312 | MATCH |
| 71 | B:215-216 | lam_agg 0.009; 6e-3 | AT:397-400 | MATCH |
| 72 | B:219 | KL3 floor 8.242 | AT:448 | MATCH |
| 73 | B:223 | prior0 0.8814, 7.35, -0.0015 [-0.0091, +0.0052] | AT:415-417 | MATCH |
| 74 | B:224-225 | odm3 0.9008 / 7.76 / +0.0178 / 1.70; odm3_norev 0.8735 / 5.16 / -0.0095 / 1.72 | AT:438-439, 448-449 | MATCH |
| 75 | B:226-228 | lam0.1 0.8829 / 8.95 / -0.0001 / 0.64; lam1 0.9069 / 9.49 / +0.0240 / 0.45; prior 0.8756 / 9.26 / -0.0073 / 1.43 | AT:487-489 | MATCH |
| 76 | B:232 | 2x2 spans 0.876-0.901 | AT:502 | MATCH |
| 77 | B:236 | 2.016 ± 0.341; 1.042; 0.852; 0.830; EMA 0.450 | AT:587 | MATCH |
| 78 | B:243 | perm 1.270 vs 0.830 (EMA 0.644 vs 0.450); rules 0.93 / 2.2 | AT:554-555, 785-786 | MATCH |
| 79 | B:244 | "audit pending at last entry" | AT:770-771 audited DONE_WITH_CONCERNS | **MISMATCH** |
| 80 | B:245-246 | +0.18, +0.27; residual 0.26 | AT:795-797 | MATCH |
| 81 | B:249-251 | 20 of 24; max +0.025; < 4 %; null 0.836 -> 0.889 | AT:626-631 | MATCH |
| 82 | B:253 | Spearman 0.62 | AT:644 | MATCH |
| 83 | B:262-264 | beta 0, tau 1 (from 2); b = 16; 0.439 vs 0.258 s | AT:180-186 | MATCH |
| 84 | B:270-275 | step 4 table (picks, PERs, deltas, 7 rows) | AT:698-706 | MATCH |
| 85 | B:278 | collapse after 18k-34k updates | AT:723 | MATCH |
| 86 | B:282-283 | own -3.354 vs deranged -4.909; 22 % top-1 | AT:827-829 | MATCH |
| 87 | B:301 | rates 10.7-10.9; gaps 4.24-4.63 | BU:134-135 | MATCH |
| 88 | B:305-310 | budget table (6 arms x 5 epochs + paired) | BU:141-146, 171-181 | MATCH |
| 89 | B:312-313 | killed at 65-67; ep20 0.875-0.910 | BU:160-161, 172-182 | MATCH |
| 90 | B:314 | objective drift 0.4-2.2 % | BU:241 | MATCH |
| 91 | B:318-319 | lattice term 1.85 -> 1.73; rate 8.7-8.9; coverage KL 0.89 -> 1.33 | BU:194, 218 | MATCH |
| 92 | B:322 | rate stalls 8-11 / 5-7 | BU:216-217 | MATCH |
| 93 | B:338 | lam_cons 1.0 = 6.4x grad norm | IM:96 | MATCH |
| 94 | B:339-340 | ent 0.1 over 1-10, to 0.001 by 20; lam_cons 0.03 / 0.1 | IM:100-105 | MATCH |
| 95 | B:346-348 | ctrl_50 3.04 / 2.41 / 0.38 / 0.23; 2.82 / 4.56 / 4.99 / 4.99; +0.005 / +0.002 / -0.010 / -0.012 | IM:258-260 | MATCH |
| 96 | B:350 | superseded 3.25 / 2.71 / 0.30 / 0.19 | IM:262 | MATCH |
| 97 | B:352-353 | ep50: 0.916 (+0.019*), 0.891 (-0.006*), 0.898 (+0.002), 0.891 (-0.006*); max 0.021 | IM:269, 277-290 | MATCH |
| 98 | B:354 | ctrl_50 rate 9.16 vs 10.79 | IM:277; BU:141 | MATCH |
| 99 | B:355 | ent_50 1.21 vs 3.04 nats at ep1 | IM:294 | MATCH |
| 100 | B:363-369 | private-code table (7 rows x 3 epochs) | IM:334-342 | MATCH |
| 101 | B:371, 375 | random many-to-one 0.868; 1.3 of 5 bits | IM:328, 356 | MATCH |
| 102 | B:390-391 | trimmed 0.56-0.7 % fewer; agreement 0.7055-0.7377; 0.22-0.27 / 0.72-0.75 | PP:202-206 | MATCH |
| 103 | B:398-400 | prepro table (PERs, primary and seed-band pairs) | PP:226-233 | MATCH |
| 104 | B:402 | rates 9.03-9.26; ep1 3.3-3.6; gap -0.007 | PP:236 | MATCH |
| 105 | B:414 | Yeh 44.7 -> 44.9 | CD:65 | MATCH |
| 106 | B:420, 426-428 | ~355k params; 41^2; 41^3 | CD:88, 15, 78 | MATCH |
| 107 | O:62, 72 | d_min 2, D 25 / 50; ppl 9.56 | OB (identical text) | MATCH |
| 108 | O:146, 201 | lambda_rate 3, lambda_agg 0.1; rho 9.66, 2.7 words/s | OB | MATCH |
| 109 | O:299 | registered ctrl_50 entropies 3.04 / 2.41 / 0.38 / 0.23 | IM:258 | MATCH |
| 110 | L:15 | -3.98 vs -3.20 | PR:31 | MATCH |
| 111 | L:19 | 9.561 (old 9.469) | PR:49-50, 174 | MATCH |
| 112 | L:25-27 | SIL 20.3 % / 7.9 %; 5.7 % of 186,083; median 100 ms, 28 %; 4.4 % / 13.8 % | PR:70-76 | MATCH |
| 113 | L:29 | rVAD removes 60 %; 2.4x; w2vu 1.0 chose 0.25 | PR:78-89 | MATCH |
| 114 | L:41-47 | Step 0 table (7 priors x 6 columns) | PR:311-317 | MATCH |
| 115 | L:49 | 0.876 / 0.237 / 0.024 / 0.011; MKN-3 1.19; 0.20; +0.48; -0.044; 0.16 / 2.17 | PR:319-328 | MATCH |
| 116 | L:51 | word-bigram 2.287 vs 2.321 | PR:484-485 | MATCH |
| 117 | L:61-64 | Step 0b table (4 instances) | PR:399-402 | MATCH |
| 118 | L:66, 68 | 9.557 / 7.271 / 5.424; 4-gram gap 1.666; 1.849 -> 1.809 | PR:368-371, 409 | MATCH |
| 119 | L:78 | +0.95 / -0.67 / -2.2; +1.31 / -0.79 / -1.0; 494-510 distinct | PR:242-244 | MATCH |
| 120 | L:88-90 | neighbourhood table | PR:456-461 | MATCH |
| 121 | L:92 | 1.5 % edits; 5.2 % / 23.7 % | PR:463-464 | MATCH |
| 122 | L:98-102 | FFBS table; 45-75 nats | PR:505-507, 516 | MATCH |
| 123 | L:106 | epoch 10, ppl 5.0906, 3.31 M | PR:626 | MATCH |
| 124 | L:108 | seed band "ctrl_20_s1 − ctrl_20" −0.001 | PP:232 gives ctrl_20 − ctrl_20_s1 (PR:582 has the same flip) | **MISMATCH** |
| 125 | L:114 | soft probe 0.298 / 0.306 / 0.335; 0.306148; 0.326639; 0.0138; 0.979918 | PR:657-661 | MATCH |
| 126 | L:115 | sf probe 2.368 / 2.186 / 2.203; 2.20294; 0.0453938; 23.6 GiB; 14-19 s; 9.68 | PR:678-688 | MATCH |
| 127 | L:117 | 224.61 GiB; 1.14x | PR:667-669 | MATCH |
| 128 | L:119 | 717 / 849 / 852 / 849 s; 1.19-1.42x | PR:698-699 | MATCH |
| 129 | L:125-128 | interim ep10 table | PR:711-714 | MATCH |
| 130 | L:130 | JSD4 0.739 / 0.772 / 0.789 / 0.746 / 0.230; -3.81 / -3.71 / -3.52; NMI 0.066-0.093; sf 0.899, 74 % S | PR:719-731 | MATCH |
| 131 | L:148 | trie 151,731 / 132,049 / 33 / 291,476; 3.579 (6.486); [2.505, 5.010] | LX:62, 241-242 | MATCH |
| 132 | L:150 | 151,734 / 3,393,577 / 10,419,405 / 3,545,312 / 274,266,991 B | LX:243 | MATCH |
| 133 | L:158-161 | word-LM benchmark (4 LMs) | LX:347-350 | MATCH |
| 134 | L:171 | C_esc 64 | LX:21, 70 | MATCH |
| 135 | L:177-182 | E-1 2.3e-07, 2.2667, 2.2706, 133,942; E0 0.0512; kill -0.0509 (0.9095 -> 0.8604), -0.0330; 9.66 % (54.06 -> 48.63); E1 802-912 (856), 41 GiB, 48,800 s, 41x | LX:256-327 | MATCH |
| 136 | L:184 | E0 lam 1: -2.6507 / 0.69; -2.6095 / 0.77; -2.5594 / 2.07; 0.649; 2.630 | LX:279-281, 308 | MATCH |
| 137 | L:186 | 873 s vs 12.3 s (71-76x); 97 %; GEMM < 1.5 % | LX:362-373 | MATCH |
| 138 | L:188 | 4.4e9 arc-updates/s; 2e8; 150-227 s per step | LX:334, 115 | MATCH |
| 139 | L:196 | 3,120 unk arcs; -14.97; -28.4..-18.1; -8.562..-3.125 | LX:494, 390 | MATCH |
| 140 | L:198 | beam 20 / 8 / min_active 30; max_active 1000 / 3000; 16 seqs per intersect | LX:144; reports/impl_k2_probe_chunk_2026-09-21.md:10 | MATCH |
| 141 | L:202 | empty > 10 % abort; > 2 % UNINFORMATIVE | LX:147, 163 | MATCH |
| 142 | L:210-212 | -2.739 vs -2.803 (0.064); 772 of 2,864, 22 | LX:389, 396 | MATCH |
| 143 | L:216-222 | over-count table + audit split (+0.013 / +0.019; +0.033 / +0.017; +0.240 / +0.110; 0.274 / 0.117 / 0.122) | LX:400-438 | MATCH |
| 144 | L:234-242 | settling-probe table (7 graphs x 3 rungs); 0.09-0.13; theta 5 (145.3 M -> 7.6 M); 391 / 374 / 192 GiB; build costs | LX:390, 420, 428-442 | MATCH |
| 145 | L:244 | 31.22 / 3.302 / 789.2; 31.77 / 3.955 / 826.5; 17.4-21.9 GiB; 1.87-6.51 s | LX:440, 492 | MATCH |
| 146 | L:246 | 760.6 / 948.6 / 734.8 / 610.0 s; 4:16-4:58 h | LX:498, 700 | MATCH |
| 147 | L:254-265 | positive control table; 0.976 / 0.859; 0.529; 1.219; 1.9e-12; 99 / 100; +1.5; 2.811 vs 2.470 | LX:462-494 | MATCH |
| 148 | L:279, 281 | s1 identity -0.347 / -5.671 / 57.498; 0.8746 / 0.8751; ranges 0.819-0.843 / 0.814-0.843 | LX:490, 498 | MATCH |
| 149 | L:285-303 | 19 paired rows incl. clause-2 pairs and ep10 values | LX:504-535 | MATCH |
| 150 | L:305, 307 | B_k2 0.0092; B_null 0.0136; B_off4 0.0062; ep4 +0.0021; +0.0111 / -0.0033 / -0.0146; F 0.013-0.015; M 0.015; 0.0257 | LX:525-541 | MATCH |
| 151 | L:309 | stability <= 0.044; empty <= 0.0010; escape <= 0.24 (0.0001 / 0.05 / 0.20); ratio 1.2; ppw 3.4-3.6 vs 5.4-6.9; gap 3.04-3.61; rate 7.3-8.1 | LX:533 | MATCH |
| 152 | L:323 | D0 decile 9: 0.741 vs 0.784; 0.692 vs 0.757; 0.784 vs 0.714 | LX:547 | MATCH |
| 153 | L:325 | D1 JSD3 / JSD4; log P3 -2.96 / -2.76 / -2.54 / -3.37 | LX:564, 578 | MATCH |
| 154 | L:327 | D2 ~0.04 drop; 0.285-0.306 vs 0.315-0.359; 27-32 vs 16-17 words; agg 1.52 -> 1.03-1.47; 8.8 -> 8.2-8.5 Hz | LX:558 | MATCH |
| 155 | L:329 | D3 coverage / recall rows; precision 0.087 vs 0.565; +0.015..+0.047; 0.61-0.63 vs 0.49 | LX:570, 580 | MATCH |
| 156 | L:331 | D4 "−800 to −1190 at ep20" | LX:582 audit: ep20 945-1187 (LX:568: all epochs) | **MISMATCH** |
| 157 | L:331 | D4 -53.5; -5.6; 15-74; +17.7 / +47.2 / +46.2 | LX:568, 572 | MATCH |
| 158 | L:333 | D5 per-token log P rows; 0.43-0.52 vs 0.04-0.12 | LX:576 | MATCH |
| 159 | L:335, 337 | D6 1.42 M vs 1.46 M, ~830 passes; D7 0.009-0.023, 0 / 73, 270-430 | LX:559, 566, 582 | MATCH |
| 160 | L:339, 341 | D8 7.05e-4 / 3.48e-4, 1.948 [1.651, 2.249], 2.03, 2.6e-18, 1.83x; D9 0.61-0.70, 0.645 / 0.620, 0.424, 0.611 | LX:560, 574 | MATCH |
| 161 | L:343 | D11 140.12 / 857.8 / 2478 / 18508 / 10811 vs 21195 / 12487 vs 17445 | LX:588 | MATCH |
| 162 | L:349-353 | p0 0.1894 (24 steps), 0.2067; D10 pack PERs; -0.078; -0.0050 [-0.0081, -0.0015]; 0.846; 4.86 -> 2.81; 0.358 / 0.068; 767.8-1077.8 | LX:452-456, 486, 592, 606-607 | MATCH |
| 163 | L:355-366 | +1000.4; D10e table; paired rows; 450 / 208; -0.0097 misses by 0.0003; NMI 0.86 / 0.07 | LX:612-671 | MATCH |
| 164 | L:376-383 | E60 table; paired rows; PER 0.824; +0.0071; 7.81 Hz; gap 3.84; 10 h 31 min; 2,280 updates; 0.310 -> 0.320 | LX:715-752 | MATCH |
| 165 | L:391 | D12 bound 0.3632 (0.197 / 0.363); 1.786-2.229; -0.105 / +0.178 / -0.359 | LX:767 | MATCH |
| 166 | L:401-403, 407-419 | D13 table; D14 0.1277, B_warm 0.0019 / 0.0010, PER table, +0.0521 / +0.0326, -0.0593 / -0.0492, -0.1796, -0.0166 / -0.0193, -0.0107, 0.003 | LX:773, 815-850 | MATCH |
| 167 | L:421-429 | D15 M_c 0.0204, PERs, -0.0012, -0.0072, +0.0372; D16 totals, I -2.8119, +0.1818, -1.797 / -0.889, -0.255; D17 4.4138, -0.0363 / -0.0406; D18 +0.107 / +0.120, -0.025 / -0.160, 4.24-4.52 | LX:884-1066 | MATCH |
| 168 | V:8-10 | 0.180; 57 steps; 800-1190; 18-79; D13 terms | L2:25 | MATCH |
| 169 | V:41-50 | lr 3e-3 clip 5; tau [4, 1, 1, 1]; 285; durinit 4.41; 3.7 Hz / 0.98-1.41 Hz; 16 + 4 + 2 + 2 | L2:61-66, 426 | MATCH |
| 170 | V:64-66 | LIFT < 0.50, PARTIAL < 0.8164; HOLD <= 0.2894, COLLAPSE >= 0.70 | L2 A4; LX:616 | MATCH |
| 171 | V:100-113 | 0.246 h, 3.44 s; probe table; E[d]; 2.41e9 arcs; 16.1 s, 76.7 GiB, 2.23 h; 83.7 / 81.7 / 80.3 GB; 37,200 -> 500-1,400 | L2:128-137, 411-436 | MATCH |
| 172 | V:121-131 | K* 12 / 10 / 10; 3.4217 / 3.4571; 3.4e-5; 6.8-7.6 Hz; S@48 table; gold 3.4735 (3.4702); r100 4.6901; decode ranges | L2:442-457 | MATCH |
| 173 | V:142-150 | A11: 16 restarts, 33.6 ms, 300 utts, 7.1k; 6.88-7.38 vs 1.63-4.09 Hz; 3.536-3.608; -1.696; Spearman 0.28 / 0.10 / 0.59 | L2:108-114, 460-467 | MATCH |
| 174 | V:158-172 | L2-0 table (8 arms); node P values, +0.6635; statistics (a) / (b) | L2:478-492 | MATCH |
| 175 | V:191-218 | A15 table (10 rows); A15-F bounds and class shares; sharp nulls; -10 vs -6.0..-6.7 | L2:513-641 | MATCH |
| 176 | V:226-256 | A16 (a) +0.123..+0.308, -0.598; A16 (a2) ladder; 0.58; A14 (ii) table; -0.083 [-0.094, -0.072], 222 / 260; 0.193 -> 0.300 -> 0.328 -> 0.353; 4.71-4.92; 3.474 -> 4.814 | L2:151, 290-291, 562-663 | MATCH |
| 177 | V:268-309 | 0.487 vs 0.058; J ladder -4.818 / -5.610 / -6.334 / -6.560; -4.971; 0.153 / 0.144; fragility numbers; repeat 989 / 0.265 % / 0.0027; 3.289 (3.28903); PREACT_ON 2.0; 2.6-3.3x | L2:251, 280-281, 324, 676-715 | MATCH |
| 178 | E1:20-57 | 1.43 M, 41 outputs; 945 params; 26.8; ~123 s; 7.9x; 4.33 s / 16.84 GiB; 10.31 s / 9.90 GiB (6.9x); 1734 vs 277 s; ppl 14.23 / 9.47 / 7.03 / 10.67 / 9.94; 7.94 | A:152, 288, 602, 648-688, 726, 965, 1066 | MATCH |
| 179 | E1:51 | bigram "1.48-1.67 s per step" | A:632 (1.648), A:665 (1.48) | **MISMATCH** |
| 180 | E1:88-95 | S1a table (7 rows with CIs); +1.42 exact subset | A:258-274 | MATCH |
| 181 | E1:101-107 | inits 0.058 / 0.116, 0.068 / 0.130, 0.166 / 0.218, 0.13 nats; S2 0.353 / 0.384, 0.345 / 0.374 | A:312-327, 420, 449 | MATCH |
| 182 | E1:120-140 | 1.88 -> 1.75; S2c D / E ranges; T sweep; G4a.2 WER rows (both seeds); paired PER rows | A:437, 480-525 | MATCH |
| 183 | E1:156-186 | deletion rates; X X merges 25.6 -> 45.9 -> 55.6; E[d] 13.44, 0.103, 1.03 nats, 3.25 vs 6.22; fixed-point table; b2 remedies; b3 0.0981 / 0.1015 / 0.1116 | A:577-621, 770-830 | MATCH |
| 184 | E1:203-216 | S3 1.74 -> 1.21, 0.8955, 0.8387, rates, -0.3218; F1 21.6 -> 8.9; c1 0.431 / 0.279 / 0.238; c4 0.1380 / 0.1218 | A:902-938, 989 | MATCH |
| 185 | E1:223-270 | lam3 6.95 / 0.29 / 0.847 / +1.910 / +1.730 [1.606, 1.861] / 488 of 500 / +3.289; grid; chance 0.840-0.922, 0.003-0.019; P-BT; BT table; 0.836; -0.93 [-1.65, -0.13]; -0.55; +3.16; CT rows; 0.345 -> 0.528 | A:986-993, 1128-1145, 1379-1390 | MATCH |
| 186 | E1:289-324 | S2d table; +0.139 [+0.129, +0.148]; S3b-OR table; +0.272, -0.040; OR diagnostic; row sums [3.3e-9, 3891.9] | A:820, 1262-1270, 1344-1360, 1437-1438 | MATCH |
| 187 | E2:12-36 | 0.058003 / 0.115741; 50948; 193644; S2d S/D/I; +14.913 / +13.870; 12770 / 11393; deletion rates; 72.46 / 67.62; 25.9029 vs 26.5035 | A:391, 1481-1510, 1630 | MATCH |
| 188 | E2:48-94 | S2e table and contrasts; +0.1433; 13.95 vs 14.65; 4h43m19s; S2f table and contrasts; -0.3356, -0.8569, +0.3526, -0.0716, +0.196; 1h37m56s | A:1578, 1597-1740 | MATCH |
| 189 | E2:102-119 | 8m25s; NLL 4.46945 -> 3.29034, 3.51912 -> 3.38357, 3.36134; 1h38m17s; S2g tables; -0.5742 | A:1846-1911 | MATCH |
| 190 | E2:160-174 | S3c table; +2.187896; S/D/I 24,456 / 45,528 etc.; 0.845393; 8,530; 8,350; 5.2726 %; 2,014 | A:2188-2264 | MATCH |
| 191 | E2:180-216 | 3.713572, 0.024390245, 7,201 / 32,185, 0.987968-1.025847, 1.3878e-13, 0.067930; one-step check; DP64 table; gaps; 16,079 s, 33.71 s; S3d 0.8301678184 / 0.001991 / +1.342968 | A:2385-2551, 2691-2692, 3629 | MATCH |
| 192 | E2:224-260 | recognizer-factor table and -0.031790408; LM 91,100,286 phones; NLL / ppl; pilot table; 3.49797e-8 | A:3462-3522, 3611-3616 | MATCH |
| 193 | E2:270-327 | cost table; 19.1 h; 420 / 655 / 789; 897.085 / 1001.968; K16 / K4 ranges; M512 22.232-74.310, 45.787, 12.308 h; M1200; 144.629; ESS 1.0000016; 4h23m19s; M512 table; 85.2529; +2.213992; output table; 77.65 / 45.71 / 93.51 | A:14, 2733-3420 | MATCH |

Row count: 193 rows: 188 MATCH, 4 MISMATCH (rows 79, 124, 156, 179), 1 UNTRACEABLE (row 26). Grouped rows
were checked value by value; any single differing value would have been listed as a mismatch.

## What would be needed to close the untraceable item

The JUPITER output of the package-side supervised reverse fit that produced 3.2888: the `learning_rates`
file with its `dev_loss_nll_per_frame` column at epoch 8 and the job's config. Alternatively, a note in the
distilled documents that the number is banked by the port, not by the JUPITER logs.
