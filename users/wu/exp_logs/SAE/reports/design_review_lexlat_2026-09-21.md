# Design review: SAE_4A_lexlat.md (lexicon inside the marginalised lattice), 2026-09-21

Read-only review before any code or compute. Files: `SAE_4A_lexlat.md` (spec), `SAE.md:60-112`,
`SAE_ref.md:31-80, 342-354`, `SAE_4A_prior.md:1-45, 135-345, 448-544`, `SAE_4A_prepro.md:1-32, 96-165`,
`SAE_4A_budget.md:108-131, 148-158, 183-212`, the three cited reports, and the code the spec cites
(`sae/emc/lattice.py` docstring, 640-670, 855-905, 1035-1165; `prior_gap.py:225-245, 420-475, 1545-1565`;
`blankfree_pack_jobs.py`; `blankfree_train_jobs.py` seed lines; `eval_jobs.py:1382-1399`; `rate_term.py`).

**Verdict: APPROVE_WITH_AMENDMENTS.** The phase asks the campaign question with the real quantity
(plain paired PER at ep20, a proper destroyed-structure null, no label in training or selection), and
the marginalised lexicon is the correct next arm after Step 0b and falsifier (ii). But the gate as
written cannot return FAIL in its most probable outcome, the ep4 identity check is unachievable by
construction, and the pruning rule has a failure mode the spec's own escape convention was meant to
prevent. All are fixable before the first job; one cheap label-using read inside E0 can kill the pack
before it is funded.

## A. Does it test the campaign question?

Yes. Objective (`SAE_4A_lexlat.md:9`) = "leave the content-free band at the same budget"; the read is
plain PER paired per utterance at sub-epoch 20 (`:116`), against ctrl_20 and a null that keeps trie
topology and the word LM and destroys only the pronunciation-to-word correspondence (`:77`). Given
(i) no phone LM reaches 2.01 (`SAE_4A_prior.md:11-16`) and (ii) the sampler never reaches the lexical
region (`SAE_4A_prior.md:514-516`), marginalising the constraint is the remaining route and the spec
says correctly what falsifier (ii) does and does not license (`:11`).

Confound that passes the gate for the wrong reason: none for PASS, because PASS needs PER < 0.50 from
a control at ~0.87-0.90 (`SAE_4A_budget.md:155`: ctrl_50 0.855/0.869/0.897/0.888), a 0.38 drop that
no rate artefact produces. The confound sits on the FAIL side: ADD mixing roughly doubles the prior's
per-phone price on every path (gold -3.20 + -2.19 nats per token, `SAE_4A_prior.md:318,322`), ctrl_50
already runs 10 % under rho (8.7-8.9/s vs 9.66, `SAE_4A_budget.md:202-203`), and the abort fires only
at 0.6 rho (`:129`). A 15-25 % deletion shift in both arms is inside every clause and would produce
a FAIL (or an unreadable pair) that says nothing about lexical routing. Amendment 6(b) prices this
before the pack.

Cheaper experiment that answers or kills first: the marginalised term's whole mechanism is stated at
`:40` ("the lexicon changes which paths carry mass"), i.e. each step moves theta toward the
lexicon-shaped posterior. Whether that posterior is closer to gold than the plain one at the on-set
code is measurable on E0's own 100 utterances at ctrl_20 ep10 with no arm: the max-plus decode under
the augmented lattice vs the plain lattice's max-plus decode, plain PER, paired. This is falsifier
(ii)'s analogue for the marginalised estimator, and it is what the literature predicts against (F):
every precedent put the lexicon on top of a phone stream that already carried content (Klejch's
phone-mapping baseline WER 35-67, wav2vec-U PER 0.2-0.3 before the WFST); here it has to create
content from PER 0.88. Amendment 6(a).

## B. Is the gate decidable before results exist?

Thresholds: M = max(|B|, 0.010) fixed (`:116`); engagement ceilings 0.05 / 0.90 / [0.7, 1.4] fixed
(`:124-126`), with the pronunciation-length constant taken from the build job before launch (`:59`);
abort rule copied from G4a.4 (`:129` vs `SAE_4A_budget.md:128-130`). PASS requires both comparisons
(`:118`). No read enters training or selection; the label-using reads are disclosed (`:131-134`).

**Defect 1, material: the gate cannot return FAIL in its most probable outcome.** `:114` puts
"greedy PER < 0.50 AND rate in [5.80, 14.49]" into the HEALTH clause; `:120` makes CANNOT_TELL fire
when "the health clause fails in BOTH lexlat_20 and ctrl_20"; `:119` makes FAIL require "the health
clause holds". In G4a.4 (`SAE_4A_budget.md:113-116`) PER < 0.50 AND the rate window ARE the gate, and
health is the derangement gap alone. Every cold arm on this bed sits at PER 0.82-0.91
(`SAE_4A_budget.md:155-158`), so ctrl_20 will fail the copied "health" clause, and a treatment at,
say, 0.80 with a delta of -0.08 reads CANNOT_TELL, not FAIL. The gate would then license neither
"not funded further" nor anything else, and the phase could not close on its own rule.

**Defect 2, material: the ep4 identity check (`:73`) demands 0.000 across packs, which nothing
guarantees.** The training sets seeds per step (`blankfree_train_jobs.py:340-341, 541`) but no
deterministic-algorithms flag anywhere in the bed (grep of `blankfree*.py`, `lattice.py`, the pack
config: none); `lattice.py` itself has no atomics (no `index_add_`/`scatter_add_`), but the
recognizer/reverse backward run under default cuDNN, and a forward-only rebuild of a banked
checkpoint already flipped 8/300 argmax decodes (`SAE_4A_prior.md:494-495`,
`reports/audit_sf_probe_2026-09-21.md:35-43`). Four sub-epochs of training in a content-free regime
amplify that. As written, a nonzero ep4 delta stops every arm for a cause that may be no fault; and
the check cannot distinguish "nondeterminism" from "wrong lam_lex = 0 path" because it has one
comparison. The pack itself carries the distinguishing read: lexlat_20, lexshuf_20 and lexlat_20_e5
are lexicon-free through sub-epoch 4 on the same node and code (`:66-68`), so their mutual ep4 identity
tests the code and their common offset from ctrl_20 measures the cross-pack floor.

**Defect 3, spec gap tied to 2: which DP runs before the on-set is not stated.** If the new trie DP
runs from sub-epoch 1 at lam_lex = 0 with C-pruning on, mass is split over score-equivalent
(n, sigma) contexts and pruned, so sub-epochs 1-7 already differ from ctrl_20 (a pruned lower
bound, `:46`) and the single delta (`:20`) is broken before the lexicon exists. Test (a) (`:95`)
covers lam_lex = 0 with pruning OFF only.

Seed band from another pack: valid for the decisive comparisons. Cross-pack noise enters
lexlat_20 - ctrl_20 but not lexlat_20 - lexshuf_20 (same pack), and PASS needs a 0.38 drop, so the
band decides only a within-band movement read. Weakness to disclose: B is one pair (n = 1 for the
band itself), as in G4a.8 (`SAE_4A_prepro.md:102-104, 143-147`); the 0.010 floor is the right guard.

Shuffled-pronunciation null: a feature, not a flaw. It isolates the 2.17 of 2.32 nats the audit
attributes to lexical routing (`SAE_4A_prior.md:333-334`) while keeping trie topology, segmentable set
and word n-gram marginals fixed; the segmentability constraint is common to both arms, so a
treatment-minus-null gain is attributable to word identity. One clause is inverted, though: `:127`
demotes the null if `lexlat_term_mean` differs by more than 2x over the run, but divergence of the
two terms is the SUCCESS signature (the treatment's term rises as the code becomes lexical; the
null's cannot). Magnitude match must be read at the on-set window, before divergence.

## C. Technical soundness of Design 1-2

State space: `(trie node, word-LM state, last two phones)` x band offset x repeat flag is consistent
with the bed's transitions (`lattice.py` docstring: only the emit arc moves h; blank and repeat leave
the context; d_min and the band are context-free), so the context axis is orthogonal to (o, f) and
pruning by logsumexp over (o, f) per context (`:46`) is well defined. Two source contexts can reach
one destination (different sigma with the same successor suffix), so the per-frame step needs a
merge (unique + logsumexp), not the injective `index_copy_` placement of `lattice.py:882-883`; this
is a full new forward and backward, not a table swap. The spec's module plan says so (`:93`).

Gradient claim (`:40`): correct. The lexical increment has no theta or phi dependence; the manual
backward's posteriors over the (pruned) lattice reweight `log q` and the segment table exactly as
`beta log P3` does now (`lattice.py:78-92` surrogate). The pruned objective is piecewise smooth and
optimising it exactly is beam-EM practice (Nuhn and Ney; Shinozaki's beam table); acceptable.

Pruning rule, defect: top-C by FORWARD mass has no lookahead. In-word prefixes are cheaper per phone
than escape-open contexts (`prior_gap.py:430-433`: order-1 + log 0.5 per phone on top of the trigram
under ADD), so escape contexts are pruned first; when SIL arrives (forces a word close, `:35`,
`prior_gap.py:461-467`; a SIL inside a word is unsegmentable, `test_prior_gap.py:101`) or the
utterance ends, the surviving mid-word prefixes die. Result: mass cliffs or Z = 0 utterances, which is
the zero-probability trap the spec chose ESCAPE to avoid (`:36`). The E0 rule reads the MEDIAN
pruned mass (`:81`), which hides exactly this tail. End-of-utterance finality (word end or escape
open only, prior_gap's convention) is also unstated.

E0 criterion: median pruned mass < 0.05 per frame and |log Z(1024) - log Z(4096)| < 0.05 nats per
retained frame is a reasonable budget (about 4 % of the lexical increment per retained frame) but is
measured at one lam_lex. At the ramp (`:54`: 1/3, 2/3, 1 over sub-epochs 8-10) the word-LM prices
are scaled down, mass spreads over more contexts and pruned mass rises; the clause at `:124` counts
"after the on-set", so it can fire during the ramp for a reason that is not C.

Escape convention as a length substrate: no. Inside log Z with per-retained-frame normalisation
(`:58`) there is no per-token mean to pay for length; a longer escape is strictly dearer
(`survey s1`). The length pressure that does exist is the ADD weight itself (section A), which the
rate term must carry.

## D. Efficiency gate E1

Bar 1202 s / 80 GiB fixed before measurement, fallbacks ordered (`:87`); sound. The 100-step,
four-point sample under laplace ordering is the standing protocol; two additions: the memory number
must include the batch with the largest T_max (laplace places it at one end and four evenly spaced
points can miss it), and the checkpoint E1 runs on must be pinned (ctrl_20 ep10, the code at the
on-set; reachable-context count depends on the code). Cost table: `:149-150` still say "3 arms"
and "3 arms x 4 kept epochs" while the rulings fund four (`:160`); charged GPU-h unchanged on an
exclusive node, read jobs 4 x 4. Actual pack wall time is 7 x 601 + 13 x 1202 s = 5.5 h at the bar,
inside 11.5 h. The 27-41 GPU-h total is credible.

## E. The five rulings

1. G2P trie verbatim: keep. The bed's prior text already goes through the same chain
   (`:23`); comparability with the banked 2.32 row is worth more than a purer trie, and E-1(a)
   prices the difference.
2. ADD: keep, conditional on the null being read as a weight control at the on-set (amendment 5)
   and on the E0 rate read (amendment 6b). ADD doubles the per-phone prior weight; the user's ruling
   that weight is not the lever (`SAE_4A_prior.md:38-40`) is why the null must carry the comparison.
3. Trigram with bigram fallback: keep; the banked 0.03 nats price (`SAE_4A_prior.md:486-488`) is
   negligible and the fallback is declared before measurement.
4. Second seed in slot 4: keep. The alternative, an in-pack lam_lex = 0 control, would make the ep4
   check exact by construction, but the three lexicon-free-through-ep4 arms already give that
   (amendment 2), PASS needs the replicate, and cross-pack noise does not touch PASS.
5. Both on-sets: keep. Note the schedule (`:19`): with on-set 8 and LR decay from 12, the lexicon
   acts at full strength and full LR for 4 sub-epochs only; e5 is the arm that tests the term at
   full LR. Disclose as the reason e5 exists, beyond "the precedent's curriculum".

## F. Literature

Klejch 2022 and Nuhn and Ney 2014 are read correctly (curriculum, pruning, silence anchor, no
re-smoothing because nothing lexical is learned). Two points the spec does not draw out:
(i) every precedent applied the lexicon to a phone stream with content (Klejch's phone-mapping
baseline 35-67 WER; wav2vec-U's WFST on PER 0.2-0.3 models; Shinozaki on gold phones), so the
literature predicts that lexicon+LM re-weighting of a PER-0.88 code does little. That is the
strongest reason to run amendment 6(a) before the pack, not a reason not to run. (ii) Klejch's
label-free selection among 50 restarts by training likelihood has no analogue here; n = 1 with a
replicate is the spec's choice and is disclosed (`:104`). wav2vec-U 2.0's audio-side anchor and Ni
2025's rare-word collapse are correctly carried as the frequency-stratified read (`:107, 133`).
Yang et al. 2026 is correctly not treated as evidence.

## Amendments (numbered; each: change, why, evidence)

1. **Restore G4a.4's gate structure.** PASS: PER < 0.50 AND rate in [5.80, 14.49] AND
   lexlat_20 - ctrl_20 <= -M AND lexlat_20 - lexshuf_20 <= -M; health = derangement gap > 0 only.
   FAIL: engagement clauses clear, no abort, and (PER >= 0.50 OR either paired delta > -M).
   CANNOT_TELL: an engagement clause fires, an abort, or E0/E1 re-declared unrecorded. Why: as
   written the default outcome (both arms in the band) is unreadable. `SAE_4A_lexlat.md:114,118-120`
   vs `SAE_4A_budget.md:113-116, 155`.
2. **Replace the 0.000 ep4 rule.** (a) Before the on-set every arm runs the banked `lattice.py`
   path unchanged (the trie DP is not called at lam_lex = 0). (b) Pre-launch: the arm's step-1 loss
   equals ctrl_20's logged step-1 loss (same batch, same seeds). (c) At ep4: lexlat_20, lexshuf_20,
   lexlat_20_e5 must agree with each other (same node, same code) -- disagreement is a fault, stop;
   their common delta vs ctrl_20 is the cross-pack floor, recorded, and M := max(|B|, 0.010,
   |delta_ep4|), fixed now. `:73, :46, :95(a)`; `blankfree_train_jobs.py:340-341`;
   `SAE_4A_prior.md:494-495`.
3. **Protect the escape from pruning and report the tail.** Reserve a declared sub-budget of C for
   escape-open contexts (or guarantee one per surviving (sigma, h)); state that only word-end or
   escape-open contexts are final; report in E0 and per sub-epoch the count of Z = 0 utterances
   and the per-frame pruned-mass MAX and 95th percentile beside the median. `:36, :46, :81`;
   `prior_gap.py:430-433, 461-467`.
4. **Measure E0 at the ramp's lam_lex values** {1/3, 2/3, 1} and count the pruned-mass clause from
   sub-epoch 11 (full ramp), as the other two clauses already do. `:54, :81, :124-126`.
5. **Read the null's magnitude match at the on-set window** (sub-epochs 8-11); after that,
   divergence of `lexlat_term_mean` is reported as a result, never as grounds to demote the null.
   `:127`.
6. **Add two label-using reads to E0 as pre-funding falsifiers, rules fixed now.** On the 100
   utterances at ctrl_20 ep10 (and ep4 for the e5 on-set): (a) plain PER of the max-plus decode
   under the augmented lattice (C = 1024, lam_lex = 1) vs the plain lattice's max-plus decode,
   paired; if the augmented decode is not closer to gold (paired mean delta >= 0) the pack is not
   funded and the phase reports "the marginalised lexicon does not re-route mass toward gold at
   the on-set code". (b) E[N] (expected non-SIL tokens, `lattice.py:409`) under the augmented vs
   the plain posterior; a drop above 10 % is recorded as confirmed deletion pressure and the ep20
   rate and S/D/I are the first read on any FAIL. Disclosed, enters nothing, same precedent as
   falsifiers (i)/(ii). `:40`; `SAE_4A_prior.md:318-322, 514-516`; `SAE_4A_budget.md:202`.
7. **Report S/D/I per arm at ep10 and ep20** beside the paired delta (per.json carries corpus S/D/I/N,
   `eval_jobs.py:1391-1393`). `:116`.
8. **E1 details**: pin the checkpoint (ctrl_20 ep10) and include the largest-T_max batch in the
   memory read. Fix the cost table to four arms. `:85, :149-150, :160`.

## Three biggest risks to the reading of G4a.9

1. Both arms stay in the band (the prior of every cold arm to date) and the gate, unamended, returns
   CANNOT_TELL instead of FAIL; with amendment 1 it returns FAIL, which licenses "not funded further"
   and nothing about a different C, on-set or bed -- and the ADD weight's deletion pressure (A)
   would be the first suspect, which is why 6(b) and 7 are pre-registered.
2. Pruning cliffs at SIL and utterance ends (C, defect) read as a pruned-mass or Z = 0 failure that
   looks like "C too small" and sends the next arm to a larger C when the cause is the missing
   escape reserve.
3. A nonzero cross-pack ep4 delta stops the arms under the unamended `:73` rule, or, if waved
   through ad hoc, leaves the band and the identity check without a pre-registered meaning.

## Cheapest check

Run E0 with amendment 6 before anything else (about 1 GPU-h, no new arm, no edit to a banked
module): if the lexicon-augmented max-plus decode at ctrl_20 ep10 is not closer to gold than the
plain one, the pack is not funded; if it is, the same job returns C, the pruned-mass tail, the Z = 0
count and the E[N] shift the amended gate needs.
