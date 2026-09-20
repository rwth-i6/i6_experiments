# SAE §4a — Context-dependent reverse model on the cold blank-free bed

## State

Phase opened 2026-09-20 on the user's approval ("Context-dependent Reverse Model"), after the
private-code analysis (`SAE_4A_infomax.md` Results) showed the budget control settling into a
confident frame-level acoustic code whose token sequence is not phone-like, and the training terms
on a plateau from sub-epoch 12 (`SAE_4A_budget.md` Results). Nothing built or launched. Survey
(`reports/survey_cdrev_2026-09-20.md`) and literature (`reports/lit_cdrev_2026-09-20.md`, against
the hypothesis; see Literature) are in; Design is pre-registered; design review dispatched.
NEXT: design review verdict; implementer (reverse head + lattice term + monitor + config, default-
off); code review with the bit-identity tests and a 100-step memory / step-rate measurement;
launch as one packed node; ep1 / ep4 / ep10 reads against ctrl_50 through the paired job;
private-code table on cdrev_50 and cdrevci_50 at ep10.

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

Mechanism monitor, logged per sub-epoch: `ctx_gain = E_post[ C[h,k,s] - C[BOS,k,s] ]` in nats per
emitted token (the BOS row of the same head is the context-free score, so this costs one slice):
how much the previous phone improves the prediction of the boundary units. Engaged = gain
> 0.05 by sub-epoch 10; a gain near zero throughout reads the arm UNINFORMATIVE.

Arms (one node, N = 50, one seed each unless stated; everything else ctrl_50):

| arm | delta from ctrl_50 | question |
|---|---|---|
| cdrev_50 | boundary-context emission | the approved test |
| cdrevci_50 | same head, `h` replaced by BOS for every segment | architecture control: the extra head without the context; the cdrev_50 vs cdrevci_50 paired delta is the context effect |
| cdrevodm_50 | cdrev_50 + the coverage term of odmprior_50 (lam_agg 0.009, order 3) | context on the coverage bed (user item 2 bed) |
| cdrev_s2_50 | cdrev_50, second seed | the seed spread a PASS needs; also the n = 2 read of the null |

Paired reads (`PairedPerDeltaJob`, registered per kept epoch): each arm vs ctrl_50; cdrev_50 vs
cdrevci_50; cdrevodm_50 vs odmprior_50; cdrev_50 vs cdrev_s2_50. Private-code table on cdrev_50
and cdrevci_50 at ep10 (dev-other), registered in the private-code config once the decodes exist.

## Gate

**G4a.6** (per arm, dev-other, read at sub-epoch 50, or at the label-free-selected checkpoint if a
selector exists; never best-PER over the kept set): the budget round's G4a.4 thresholds, greedy
PER < 0.50 AND greedy emitted rate in [5.80, 14.49]/s, health clause derangement gap > 0. Read
against ctrl_50 at the same sub-epoch through the paired job. A PASS is audited from a fresh
context and needs a second seed before it is claimed; a negative reads "no take-off in this seed".

Early read at sub-epoch 10 (descriptive, no gate consequence): PER below the arm's own
length-and-unigram chance null by more than 0.05 = band exit; the private-code table
(`SAE_4A_infomax.md` "Private-code analysis", audited conventions) on the arm's ep10 decode, read
against ctrl_50's ep10 row: token-level NMI(symbol, phone) up and frame-level NMI up = the context
term moved the code toward phones; frame-level NMI down with the reverse score per frame up = the
expressive decoder took over (the stated risk). Mechanism monitor: the context term's share of the
reverse score, `mean over arcs of C / G` under the arc posterior, logged per sub-epoch; a share near
zero throughout means the term never engaged and the arm reads UNINFORMATIVE, not FAIL.

Abort rule per arm: the budget round's (NaN or |surrogate| > 100; expected phone rate < 0.6 rho
for 5 consecutive sub-epochs after the anneal).

Pre-registered prediction (after the literature read, before the design review): the context head
engages (ctx_gain > 0.05 nats per token by sub-epoch 10, coarticulation is real), but PER stays in
the band at every kept checkpoint and the token-level NMI(symbol, phone) at ep10 is within 0.02 of
ctrl_50's 0.056 in both cdrev arms; the cdrev_50 vs cdrevci_50 paired delta is within the seed
spread (cdrev_50 vs cdrev_s2_50). If instead the frame-level NMI falls below ctrl_50's 0.256 while
the reverse score per frame improves, the Chorowski mechanism (the decoder absorbs what the code
used to carry) is the reading.

## Results

(none yet)
