# SAE §4a — Context-dependent reverse model on the cold blank-free bed

## State

DEFERRED without limit by the user on 2026-09-20 ("deferred unlimited"), after the design review
and before any code was written; the node goes to the prior-strength lever (`SAE_4A_prior.md`).
Nothing built or launched; the pre-registered design, amendments and falsifier below stand as
written should the phase be reopened. The remainder of this State is as it stood at deferral.
Follow-up candidates recorded 2026-09-21 (user: revisit only after a success wants an ablation or
after every other route has failed): (i) the reverse model's width is not a lever, since with
inputs (type, 2 duration buckets, 3 position buckets, eta) it is a 240 x 500 table that the
0.37 M-parameter MLP already over-parameterises (`reverse.py:58-75`); "larger" means a richer
conditioning set, and any such variant runs through the pre-launch falsifier first; (ii) a
transducer-style reverse model is exact only with finite-state phone context (bigram = this
design, trigram = 41^2 states, no more) and arbitrary causal context over the observed units (a
prediction network, up to the ~11 M ceiling of `reverse.py:67-72`); a full-context phone encoder
forces sampled y, the route the prior phase's falsifier (ii) closed. Chorowski risk rises with
unit-side depth; a context limit or unit-history dropout goes into any such arm.

Phase opened 2026-09-20 on the user's approval ("Context-dependent Reverse Model"), after the
private-code analysis (`SAE_4A_infomax.md` Results) showed the budget control settling into a
confident frame-level acoustic code whose token sequence is not phone-like, and the training terms
on a plateau from sub-epoch 12 (`SAE_4A_budget.md` Results). Nothing built or launched. Survey
(`reports/survey_cdrev_2026-09-20.md`) and literature (`reports/lit_cdrev_2026-09-20.md`, against
the hypothesis; see Literature) are in; Design is pre-registered; design review returned
APPROVE_WITH_AMENDMENTS (`reports/design_review_cdrev_2026-09-20.md`, amendments applied below,
see "Design review"). Implementer dispatched for the context head, lattice term, monitor twin,
falsifier job and configs.
NEXT: implementer report; code review with the bit-identity tests, the ctx_post consistency
assert, rate_fd_check with context on, and a 100-step memory / step-rate measurement; launch the
falsifier config first and read its pre-registered rule (Design, "Pre-launch falsifier"); only
then launch the pack as one node; ep1 / ep4 / ep10 reads against ctrl_50 and cdrevci_50 through
the paired job; private-code table on cdrev_50 and cdrevci_50 at ep10 and ep25 (bank ctrl_50's
ep25 row first).

## Objective

Does making the reverse model's segment emission depend on the previous phone,
`p_phi(x_seg | k, h)` instead of `p_phi(x_seg | k)`, move a cold blank-free arm out of the
content-free band (dev-other greedy PER 0.83-0.91) at the budget round's operating point? Mechanism
under test (`SAE_4A_objective.md` section 5 item 2, section 6): the bound's slack
`KL(q_theta(x | y) || p_phi(x | y))` is where the recognizer's choice of code is priced, and a
segment-independent `p_phi` prices any confident frame-level partition about equally, so the
partition the joint optimum selects is not tied to phones. A coarticulation-aware emission changes
the price of a code by how much its previous symbol predicts the current segment's acoustics, which
is a property phones have and a unit-clustering code need not.

Stated risk, both directions, to be read from the same table: a more expressive `p_phi` can also
explain the acoustics with less help from the code (the code carries less, not more). The
private-code table separates the two outcomes (token-level NMI with phones up or down).

## Bed and constants (inherited from `SAE_4A_budget.md` ctrl_50, unchanged unless listed)

priorshuf bed, N = 50 sub-epochs, the budget round's schedule (tau 8 -> 2 over the first 10
sub-epochs then held; LR warmup 3 / hold to 30 / decay to 50), batch 88000 padded frames / max_seqs
128, kept checkpoints 1, 4, 10, 25, 50, registered evaluation at each, paired reads against ctrl_50
at the same sub-epoch. Reverse-model constants (table shapes, learning rate multiplier 30,
sentence-start handling) are those of the survey; any change beyond the context axis is listed in
Design.

## Literature (2026-09-20, `reports/lit_cdrev_2026-09-20.md`, read before the design was fixed)

No published support for the hypothesis and one published mechanism against it. The one direct
test is null in our regime: Yeh et al. (ICLR 2019) went from monophone to triphone emissions in
unsupervised phone recognition and got 44.7 -> 44.9 PER under a non-matching text corpus (our
condition; 41.5 -> 39.4 with a matching one). Chorowski et al. (TASLP 2019): what the decoder is
conditioned on, the code stops encoding; frame phone accuracy peaks at about 125 ms of decoder
context and falls beyond, and the fix was limiting the generative path. Ondel and Burget (2019):
a generative model that "maximizes likelihood while modeling non-phonetic information" is fixed by
constraining it, not by expanding it. wav2vec-U 2.0: richer auxiliary targets hurt (64 clusters
beat 128 and VQ), our K = 500 is on the wrong side. The reproduced gains in this literature are in
prior strength and lexicalisation (Klejch et al. 2022: bigram -> 5-gram -> word trigram with a
lexicon, emission context-free by design), segmentation and duration (REBORN 2024), and
constraining the generative model. Consequences taken into the design: the arm is paired with a
context-free twin of identical architecture; the reads are PER and the private-code NMIs, never
likelihood or prior satisfaction (both rose in every failure above); the pre-registered prediction
is a null; the next lever if this is null is prior strength (a lexicon-constrained or word-level
prior needs a different lattice; a 4-gram state space of 41^3 does not fit this dynamic program)
and a smaller reverse unit inventory (K = 64), recorded in `SAE.md` as the follow-up candidates.

## Design (pre-registered 2026-09-20, before any launch; survey `reports/survey_cdrev_2026-09-20.md`)

Model, "boundary-context emission". The reverse model keeps its duration table and its emission
`p(u | k, dur bucket, pos bucket, eta)` for all frames of a segment except the first m = 2 unit
frames (40 ms; m = d_min so the context score never depends on the duration), which are emitted by
a context head `p_ctx(u | h, k, eta)`, h the previous phone in the lattice state (41 values, BOS
= 40 for the first segment). Same MLP shape as the existing head (`emb_prev [41,192]` added to
`emb_type[k] + eta_proj(eta)`, own `lin1 [512,192]`, `lin2 [500,512]`; about 355k new parameters,
cold-initialised like the rest; phi LR multiplier 30 applies). Exact factorisation
(survey note g): `G'[k,s,d] = G[k,s,d] - S_first2[k,s,d]` stays in the duration-indexed table and
`C[h,k,s] = log p_ctx(u_s | h,k) + log p_ctx(u_{s+1} | h,k)` is added to the lattice's `src`
before the band contraction, where the group axis already is the previous phone at the trigram
history (survey item 5). Forward and backward mirror; the context posterior `[B,41,40,S+1]` (arc
posterior summed over d) is kept from the `mass` tensor before its group sum and enters the
surrogate as `(ctx_post * C).sum()`, so phi's context head trains exactly as the rest of phi does.
`forward_logsum` (the fixed-string HSMM used by the registered derangement read) gets the same
term; `sample()` (BT) is out of scope and asserts context off. The rate term's finite-difference
passes tilt `G'` and pass `C` through unchanged. All of it is default-off: with the context head
absent the code path is byte-identical and the 20 banked bigram digests and the checkpoint
bit-identity test (survey note e) must still pass. Memory at the operating point: about +3 GiB
(`C_pad` and its posterior in fp64, the head's `[B,41,40,500]` logits, the gathered `C`), to be
measured on the first 100 steps before the node is funded (efficiency rule); the lattice batch
budget estimator does not know the new tensors and is re-checked, not extrapolated.

Mechanism monitor, logged per sub-epoch (amended by the design review; the original BOS-row form
is struck because the BOS row is trained on one sentence-initial segment per utterance and is not
a context-free reference): a monitor-only context-free twin head, same MLP without `emb_prev`,
trained on the detached context posterior summed over `h` and never entering the lattice (one
extra `[B,40,500]` logits and gather); `ctx_gain = E_post[ C_cd[h,k,s] - C_ci[k,s] ]` in nats per
emitted token: how much the previous symbol improves the prediction of the boundary units under
the arm's own code. Engaged = gain > 0.05 by sub-epoch 10 (0.05 is a choice; the falsifier's
gain(gold) per token is recorded beside it); a gain near zero throughout reads the arm
UNINFORMATIVE. Engagement is acoustic continuity across the arm's own boundaries, not evidence of
coarticulation. In cdrevci_50 the same monitor must read near zero (both heads context-free).

Pre-launch falsifier (design review amendment 1; a disclosed label-using diagnostic, its fitted
models discarded, nothing from it enters any arm, checkpoint choice or selection). The
Objective's quantity is the relative price of two codes under phi, and it can be measured before
any node is funded: fit a context-free and a boundary-context reverse model (`reverse.fit`, the
survey's fit path, same table shapes and eta as the arms) on the dev-other even utterances and
score the odd ones, with y = (a) the gold phone strings and (b) ctrl_50's ep10 collapsed decode
(the private code); repeat with the folds swapped. Report per y the held delta log-likelihood per
frame and per token, CD minus CI = gain(y), for both fold directions. Pre-registered rule, in the
job's docstring: the pack is funded only if gain(gold) - gain(private) > 0 in both fold
directions by more than the two directions' spread |gain_fold1 - gain_fold2| of that difference.
If gain(private) >= gain(gold), the context term prices the content-free code no higher than
phones, the mechanism is refuted before spend, and the pack is not launched: recorded here and
reported to the user, whose approval of the arm stands above this rule. The falsifier runs in its
own config (`config/sae_4a_cdrev_falsifier.py`) before the pack config is started.

Arms (one node, N = 50, one seed each unless stated; everything else ctrl_50):

| arm | delta from ctrl_50 | question |
|---|---|---|
| cdrev_50 | boundary-context emission | the approved test |
| cdrevci_50 | same head, `h` index replaced by BOS for every segment (an index override, so the two arms differ in the `h` index tensor only) | architecture control: the extra head without the context; the cdrev_50 vs cdrevci_50 paired delta is the context effect |
| cdrevodm_50 | cdrev_50 + the coverage term of odmprior_50 (lam_agg 0.009, order 3) | context on the coverage bed (user item 2 bed) |
| cdrev_s2_50 | cdrev_50, second seed | the seed spread a PASS needs; also the n = 2 read of the null |

Paired reads (`PairedPerDeltaJob`, registered per kept epoch): each arm vs ctrl_50; cdrev_50 vs
cdrevci_50; cdrevodm_50 vs odmprior_50; cdrev_50 vs cdrev_s2_50. Private-code table on cdrev_50
and cdrevci_50 at ep10 and ep25 (dev-other), registered in the private-code config once the
decodes exist; ctrl_50's ep25 row is banked first (the ep10 frame NMI 0.057 -> 0.256 between ep4
and ep10 tracks the end of the anneal and is a fragile single-checkpoint baseline).

## Design review (2026-09-20, `reports/design_review_cdrev_2026-09-20.md`)

APPROVE_WITH_AMENDMENTS. Not redundant with attribution: the reverse term carries content on this
bed (attrib norev 0.936 vs 0.865 at ep4, `SAE_4A_attrib.md`), so the slack mechanism is live.
Applied: (1) the pre-launch falsifier (Design); (2) the monitor twin head replacing the BOS row,
and the Gate's "C / G share" form struck (about 2 / d_mean in any state, never near zero, so its
UNINFORMATIVE clause could not fire); (3) private-code reads cdrev_50 vs cdrevci_50 primary,
ctrl_50 second, at ep10 and ep25, the "reverse score up" clause replaced by a code-change
signature; (4) the correctness list carried into the implementer brief (matmul-path context
posterior from `src + suffix`, `expected_reverse` includes the context term, `C` in `dp_kwargs`
and stacked in the finite-difference passes, the flag in `reverse_kwargs` for the derangement job,
consistency assert, `None` short-circuit, `S_first2` by the same cumsum loop, expected memory
about +4 GiB with autograd saves); (6) prediction wording. Declined: (5) a bidirectional boundary
window (anticipatory coarticulation) in the cdrev_s2_50 slot; it is a second exactness surface in
the lattice for a phase whose prediction is a null, and it is recorded as the follow-up if the
falsifier passes and the carryover arm is null.

## Gate

**G4a.6** (per arm, dev-other, read at sub-epoch 50, or at the label-free-selected checkpoint if a
selector exists; never best-PER over the kept set): the budget round's G4a.4 thresholds, greedy
PER < 0.50 AND greedy emitted rate in [5.80, 14.49]/s, health clause derangement gap > 0. Read
against ctrl_50 at the same sub-epoch through the paired job. A PASS is audited from a fresh
context and needs a second seed before it is claimed; a negative reads "no take-off in this seed".

Pre-launch condition (design review amendment 1): the falsifier rule in Design passes; otherwise
the pack is not launched and the refutation is recorded in Results.

Early read at sub-epoch 10 (descriptive, no gate consequence): PER below the arm's own
length-and-unigram chance null by more than 0.05 = band exit; the private-code table
(`SAE_4A_infomax.md` "Private-code analysis", audited conventions) on the arm's ep10 and ep25
decodes, read against cdrevci_50's row at the same sub-epoch first (same head architecture, the
context is the only difference) and ctrl_50's second: token-level NMI(symbol, phone) up and
frame-level NMI up = the context term moved the code toward phones; frame-level NMI(symbol, phone)
and NMI(symbol, unit) both down with H(unit | symbol) up vs cdrevci_50 = the code changed and the
expressive decoder took over (the stated risk, the Chorowski signature). Original clause struck
by the design review: "frame-level NMI down with the reverse score per frame up" (the reverse
score contains the context term and rises mechanically). Mechanism monitor: `ctx_gain` against
the monitor-only twin head (Design); the original "share of the reverse score, C / G" form is
struck (never near zero in any state); a gain near zero throughout means the term never engaged
and the arm reads UNINFORMATIVE, not FAIL.

Abort rule per arm: the budget round's (NaN or |surrogate| > 100; expected phone rate < 0.6 rho
for 5 consecutive sub-epochs after the anneal).

Pre-registered prediction (after the literature read, wording amended by the design review before
any launch): the context head engages (ctx_gain > 0.05 nats per token by sub-epoch 10; this is
acoustic continuity across the arm's own boundaries, not coarticulation evidence), but PER stays
in the band at every kept checkpoint and the token-level NMI(symbol, phone) at ep10 and ep25 is
within 0.02 of cdrevci_50's in cdrev_50; the cdrev_50 vs cdrevci_50 paired PER delta is within the
seed spread (cdrev_50 vs cdrev_s2_50). If instead the frame-level NMI and NMI(symbol, unit) fall
below cdrevci_50's, the Chorowski mechanism (the decoder absorbs what the code used to carry) is
the reading. What a null licenses: not funding fuller reverse context (the full `G[h,k,s,d]`
table); the follow-ups in `SAE.md` (prior strength, K = 64 reverse units) stand.

## Results

(none yet)
