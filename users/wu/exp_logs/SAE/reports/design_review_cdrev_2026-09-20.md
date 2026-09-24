# Design review: SAE_4A_cdrev.md (context-dependent reverse model), 2026-09-20

Verdict: APPROVE_WITH_AMENDMENTS. Read: SAE_4A_cdrev.md (all), SAE_4A_objective.md s2/5/6,
SAE_4A_infomax.md Results + Private-code analysis, SAE_4A_budget.md Bed/Design/Results,
reports/survey_cdrev_2026-09-20.md, reports/lit_cdrev_2026-09-20.md, SAE.md:70-90; code:
sae/emc/reverse.py:175-350,587-660, sae/emc/lattice.py:330-390,540-580,725-750,860-930,1155-1262,
1295-1310, sae/emc/rate_term.py:365-400,595-640, blankfree_train_jobs.py:175,
blankfree_eval_jobs.py:150-170, train_steps/sae_blankfree.py:139-146; one grep of
SAE_4A_attrib.md for the no-reverse arm (line 235).

## Already known (checked)
- The reverse term carries content on this bed: attrib norev (emission 0) 0.9357 vs 0.8649 with
  the cold reverse model at ep4 (SAE_4A_attrib.md:235, paired +0.071 [+0.056,+0.088]). So the
  "slack prices the code" mechanism is live and the phase is not made unnecessary by attribution.
- ctrl_50's code at ep10 is a confident frame-level acoustic code (frame NMI 0.256, token NMI
  0.056, NMI(symbol,unit) 0.377; SAE_4A_infomax.md:286-298). Note the frame NMI jumps 0.057 ->
  0.256 between ep4 and ep10, i.e. it tracks the end of the anneal, so it is a fragile baseline.

## Q1. Does the Design test the mechanism? Fair instance? Sharper variant?
Partly. The Objective's quantity is the RELATIVE price of two codes under phi: the context term
helps iff the previous symbol predicts the boundary acoustics MORE for phones than for the
private code. The full run tests this only indirectly (via where training settles); but the
quantity itself can be measured offline before any node is funded (see amendment 1), and the
Design does not do so. The boundary-context factorisation (m = 2 unit frames, 40 ms, about 36 %
of frames at 9 tokens/s and d_mean ~5.5) is a fair, exact, normalised instance of carryover
coarticulation, but it omits anticipatory coarticulation (the previous segment's tail predicted
by the next phone), which in English is the larger effect. A bidirectional boundary window
(previous segment's last 2 frames scored by p(u | h, k_next) plus the current's first 2) costs
the same [B,41,40,S+1] tensors, is exact by the same argument (subtract S_last2 from G[h,s',d']
except where s'+d' = S_b, where no successor exists; add to C[h,k,s]), and doubles the context
coverage. It is the sharper variant within the survey's memory bound. Whether it is worth the
extra exactness surface is decided by the offline fit in amendment 1.

## Q2. Are the four arms the right four?
cdrev_50 / cdrevci_50 is the right pair (the architecture twin isolates the context input;
implement cdrevci as an index override so the two differ in the h index tensor only).
cdrevodm_50 is the user's coverage-bed item. The literature-supported levers do not fit this
lattice and bed without a new pipeline: a lexicon or word-level prior needs another lattice (a
41^3 state space does not fit the DP); SHMM-style subspace constraint needs cross-lingual phone
labels; REBORN segmentation is a new pipeline; K = 64 reverse already ran in attribution (no
branch fired). So no literature lever takes a slot this round. The weakest arm is cdrev_s2_50: a
second seed of a pre-registered null is the least informative of the four, and Chorowski Fig 6
says the effect of generative context is non-monotone, so a single dose point cannot settle it.
Recommendation: if the offline fit shows the bidirectional window's gain under gold phones is
materially above the carryover-only gain, the bidirectional arm takes the s2 slot (a PASS would
still need a second seed, in the next round); otherwise keep s2.

## Q3. Gate, early read, monitor, UNINFORMATIVE clause
G4a.6 (absolute PER < 0.50 at ep50, rate window, gap > 0, paired vs ctrl_50) measures the
campaign question. Two defects in the mechanism monitor:
(a) The doc carries two different definitions. Design (SAE_4A_cdrev.md:85-88): ctx_gain =
E_post[C[h,k,s] - C[BOS,k,s]], threshold 0.05 nats/token. Gate (lines 116-118): "share of the
reverse score, mean over arcs of C / G". The share is the log-prob of 2 frames over the log-prob
of ~5.5 frames, about 0.35 in any state of the head; it can never be near zero, so the
UNINFORMATIVE clause as written in the Gate can never fire. Delete the C/G form.
(b) The BOS row is not a context-free reference. It is trained on one segment per utterance
(the sentence-initial one, mostly a SIL onset after BOS, and under the trigram start history
only, lattice.py:1002), so it is undertrained and fit to a different distribution. ctx_gain > 0.05
then fires whether or not context carries anything: UNINFORMATIVE becomes unreachable and a dead
term reads as "engaged, coarticulation is real" (a null read as a positive at the mechanism
level). Fix: a monitor-only context-free twin head (same MLP without emb_prev, trained on the
same detached ctx_post summed over h, never entering the lattice; one extra [B,40,500] logits
and gather), ctx_gain = E_post[C_cd - C_ci]. The 0.05 threshold traces to nothing; take it from
the offline fit of amendment 1 or record it as a choice.
(c) Private-code discriminator: right table, wrong comparator and one wrong clause. ctrl_50
differs from cdrev_50 by head architecture AND context; the pre-registered NMI reads must be
cdrev_50 vs cdrevci_50 first, ctrl_50 second. The frame-NMI baseline 0.256 is one checkpoint at
the exact end of the anneal; a small shift in when confidence sharpens moves it by 0.1 with no
mechanism, so read ep10 AND ep25 (both kept; bank ctrl_50's ep25 row first). The clause "frame
NMI down with the reverse score per frame up = decoder took over" is confounded: cdrev's reverse
score contains C and more capacity, so it rises mechanically. The Chorowski reading is a CODE
change: frame NMI(symbol, phone) and NMI(symbol, unit) both down vs cdrevci, H(unit | symbol) up.

## Q4. Prediction
Both outcomes are informative for PER/NMI. Reword "engages (coarticulation is real)": in the
private code, engagement is acoustic continuity across the boundary under either code, not
evidence of coarticulation. State what a null licenses: not funding fuller reverse context
(full G[h,k,s,d]); the follow-ups already named (prior strength, K = 64) stand.

## Q5. Correctness risks
1. Production DP path is matmul (blankfree_train_jobs.py:175 `lattice_reduction: "matmul"`; the
   trigram history runs D3 only). On that path there is NO `mass` tensor with a group axis:
   lattice.py:1205 `lg = _logmm(src, bd_k^T)` contracts g inside the GEMM. Survey item 4's
   ":1205" pointer is the elementwise idea. The context posterior on the mm path is
   exp(src[o,g,k] + suffix[o,g,k] - log_z) with the `suffix` [B,O,G,K] already computed at
   lattice.py:~1198; no new GEMM. Check: sum_g ctx_post[g,k,s] == sum_d seg_post[k,d,s] to
   fp64 tolerance on every step of the 100-step probe, plus brute_force_log_z with C.
2. reverse_per_frame (train_steps/sae_blankfree.py:139, from out.expected_reverse, lattice.py:1211
   `(mass_k * gwin_k)`) will omit C once G' replaces G: the monitor drops by the first-2-frame
   score and every reverse-score read (plateau -3.26 -> -3.08 in SAE_4A_budget.md:161) is wrong.
   expected_reverse must add (ctx_post * C).
3. Rate FD passes (rate_term.py:598-638) run lattice_forward_backward with `**kw` and stack the
   batch in `_stacked_fd_call`; C must be in dp_kwargs and be stacked, else the tilted passes
   score a different model. `rate_fd_check` (sae_blankfree.py:146) must stay under its existing
   tolerance with context ON in the 100-step probe; make that a code-review pass condition.
4. Derangement read: DerangementGapJob rebuilds phi from `gap_reverse_config(self.reverse_kwargs)`
   (blankfree_eval_jobs.py:160-165) and loads strict; the context flag must be in reverse_kwargs
   (strict load fails loud; strict=False would silently read the health clause under a headless
   phi). forward_logsum: `seg[rows, k_i]` -> `+ C[y[:, i-1], k_i, :]` with BOS at i = 0, and the
   batch-wise BOS row must be the same row the lattice uses (group 40).
5. Exactness of G' + C: fine. position_bounds (reverse.py:115-123) cuts at relative positions, so
   S_first2 must be built by the same cumsum loop with lo' = max(lo, 2) rather than by a separate
   bucket lookup; then each frame is emitted by exactly one normalised categorical along any path.
   C at s >= S_b - 1 reads pad units: it must be finite (those (k,s,d) are NEG_INF in G').
6. Bit-identity: the off path must short-circuit on `C is None`, not add a zeros tensor (x + 0.0
   flips -0.0 to +0.0 and the 20 banked digests would catch it late).
7. Memory: +3 GiB omits autograd saves for the head's logits/log_softmax ([B,41,40,500] fp32 x 2,
   ~0.85 GiB); expect ~4 GiB. Not a blocker on 96 GiB; measure at a long-S batch as well as the
   first 100 steps (laplace ordering). Gather index must be an expanded view, as segment_scores
   already does (reverse.py:284), not materialised ([B,41,40,S] int64 = 1.2 GiB).
8. BOS/SIL: h axis 41 with BOS = 40 matches `group = last(h)` at succ = arange(41)
   (lattice.py:376-383) and the start history 40*41+40. SIL is an ordinary h and k; SIL-after-SIL
   forbidden stays in the prior mask. No issue found.

## Q6. Stop the launch?
No, provided amendment 1 runs before the node is funded; it is minutes of GPU and reuses the
implementer's forward_logsum change.

## Amendments, priority order
1. Pre-launch falsifier (the mechanism's own quantity, label-using diagnostic, disclosed):
   `reverse.fit` (reverse.py:621) a context-free and a context-dependent phi on dev-other even
   utterances, held on odd, with y = (a) gold phone strings, (b) ctrl_50 ep10 collapsed decode
   (the private code); report held delta log-lik per frame (CD - CI) for each y, and the same
   for the bidirectional window if implemented. Rule, pre-registered: the arms are funded only if
   gain(gold) - gain(private) > 0 by more than the two halves' spread; if gain(private) >=
   gain(gold), the context term prices the content-free code LOWER and the node is not funded
   (record as the mechanism's refutation). Also calibrates the 0.05 threshold.
2. Monitor: delete the C/G "share" in Gate; ctx_gain against a monitor-only context-free twin
   head, not the BOS row.
3. Private-code reads: cdrev_50 vs cdrevci_50 primary; ep10 AND ep25; replace the "reverse score
   per frame up" clause by the code-change signature (NMI(symbol,unit) and frame NMI down,
   H(unit|symbol) up vs cdrevci).
4. Implementer brief: mm-path context posterior via src + suffix; expected_reverse includes C;
   C in dp_kwargs and stacked in the FD passes; reverse_kwargs carries the flag for the gap job;
   consistency assert sum_g ctx_post == sum_d seg_post; None short-circuit.
5. Arm slot: bidirectional window replaces cdrev_s2_50 iff amendment 1 shows its gold-phone gain
   materially above carryover-only; else keep s2.
6. Prediction wording: engagement is not coarticulation evidence; add what a null licenses.

## Cheapest check
Amendment 1 (four phi fits, minutes). It falsifies the design before any 11.5 h node runs.
