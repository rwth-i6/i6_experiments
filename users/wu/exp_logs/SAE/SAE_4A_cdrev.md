# SAE §4a — Context-dependent reverse model on the cold blank-free bed

## State

Phase opened 2026-09-20 on the user's approval ("Context-dependent Reverse Model"), after the
private-code analysis (`SAE_4A_infomax.md` Results) showed the budget control settling into a
confident frame-level acoustic code whose token sequence is not phone-like, and the training terms
on a plateau from sub-epoch 12 (`SAE_4A_budget.md` Results). Nothing built or launched. In flight:
code survey of the reverse model and lattice (`reports/survey_cdrev_2026-09-20.md`) and the
literature read (`reports/lit_cdrev_2026-09-20.md`). Design review is required before the first job.
NEXT: fill the arm table and the memory constants from the survey; design review; implementer;
code review; launch as one packed node; ep1 / ep4 / ep10 reads against ctrl_50 through the paired
job; private-code table on the strongest arm at ep10.

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

## Design (to be pre-registered before any launch; arm table filled after the survey)

Context axis. The lattice state already carries the previous phone for the trigram prior, so an
emission indexed by `(h, k)` fits the same dynamic program; the segment table grows from
`G[k, s, d]` to `G[h, k, s, d]` (40x). Candidate variants, to be priced by the survey:
- full: every frame of the segment conditioned on `(h, k)`;
- first-frame: only the segment's first frame (or first m frames) conditioned on `(h, k)`, the rest
  on `k`; the table is `G[k, s, d] + C[h, k, s]`, which is additive per arc and cheap;
- parameter tying: `log p(u | k, h) = log p(u | k) + b[h, k, u]` with `b` initialised at zero, so
  every arm starts exactly at the ctrl_50 reverse model and the context term can only add
  evidence; sentence start uses a dedicated `h = <s>` row.

Arms: four per node, one seed each (disclosed); the table is filled after the survey.

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

Pre-registered prediction (before the design review): the context term engages (share > 0.05 by
sub-epoch 10) and the token-level NMI rises above ctrl_50's 0.056, while PER stays in the band at
sub-epoch 10; whether it leaves the band by sub-epoch 50 is the open question this phase funds.

## Results

(none yet)
