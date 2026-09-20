# SAE 4A: the training objective as a reverse-KL bound

Reference note (2026-09-20, user question on the mathematical core). It records how the live
blank-free EMC objective relates to the distribution-matching objective it approximates, which term
of that objective each implemented term stands for, and where the approximation is loose. It is
interpretation, not a result; nothing here changes a gate. ASCII throughout: `tau` is the lattice
temperature, `theta` the recognizer, `phi` the reverse model, `psi` the text prior.

Code this note describes: `sae/emc/lattice.py` (module docstring and `lattice_loss`),
`sae/emc/agg.py`, `prefix_lm/model/definitions/sae_blankfree.py` (`BlankfreeAggLoss`),
`sae/emc/bt_blankfree.py`, `sae/emc/blankfree_budget_jobs.py` (`budget_temperature_schedule`), all
under `recipe/2025-10-speech-llm/src/speech_llm/`.

## 1. Target objective

Speech `x ~ p_X`, recognizer `q_theta(y | x)`, text prior `P_Y(y)`. The induced output marginal is
`q^Y(y) = sum_x p_X(x) q_theta(y | x)` (integral over continuous x). The target is

    min_theta  KL(Q^Y || P_Y)  =  E_{y ~ Q^Y}[ -log P_Y(y) ]  -  H(Q^Y).

Both pieces need the marginal `q^Y`, which is not computable: the first as an expectation over it,
the second as its entropy.

## 2. Rewrite without the marginal

Chain rule on the marginal entropy, then the variational (Barber-Agakov) bound on the mutual
information, valid for ANY conditional model `p_phi(x | y)`:

    H(Q^Y) = H(Y | X) + I(X; Y)
    I(X; Y) = H(X) - H(X | Y) >= H(X) + E_{x ~ p_X, y ~ q_theta(.|x)}[ log p_phi(x | y) ]

with equality iff `p_phi(x | y)` equals the true posterior `q_theta(x | y)` of the joint
`p_X(x) q_theta(y | x)`. The expectation over `Q^Y` in the first piece is by definition an
expectation over `x ~ p_X` and then `y ~ q_theta(. | x)`. Substituting,

    KL(Q^Y || P_Y)  <=  E_{x ~ p_X}[ J(x) ]  -  H(X),
    J(x) = E_{y ~ q_theta(.|x)}[ -log P_Y(y) - log p_phi(x | y) ]  -  H( q_theta(. | x) ).        (B)

`H(X)` is a constant of the data. The slack of (B) is
`E_{y ~ Q^Y} KL( q_theta(x | y) || p_phi(x | y) )`, i.e. how far the reverse model is from the true
"which real utterances does the recognizer map to this transcript" posterior. That slack is the
only place the marginalization over real speech lives.

Equivalent form. With `w(y; x) = P_Y(y) p_phi(x | y)`, `p_gen(x) = sum_y w(y; x)` and
`post(y | x) = w(y; x) / p_gen(x)`,

    J(x) = KL( q_theta(. | x) || post(. | x) )  -  log p_gen(x).                                  (E)

So the reverse-KL bound is the negative ELBO of the generative model "text prior, then reverse
synthesis", with the recognizer as the inference network. Its minimizer in theta, at fixed phi and
psi, is `q_theta(. | x) = post(. | x)`, the Bayes posterior of that generative model. Its minimizer
in phi is maximum likelihood of the reverse model on `(x, y)` drawn from the joint
`p_X(x) q_theta(y | x)`.

Two readings of (B) that matter for the implementation:

- The entropy term of the target is not missing; it is split. `H(Y | X)` is the per-utterance
  conditional entropy of the recognizer, a closed-form quantity. `I(X; Y)` is what the reverse
  model estimates. The reverse model is therefore part of the core objective, not an independent
  addition.
- `H(Y | X)` is not driven down (as a confidence prior would) or up per se. (E) says the
  recognizer should be the posterior of the generative model, which is nearly deterministic for a
  long utterance under a sharp reverse model. The term matters in the transient, when phi and
  theta are both poor and the posterior is diffuse; a mode-seeking form kills exploration exactly
  then.

## 3. What the lattice term optimizes

The lattice term (`lattice.py` module docstring) is, at temperature tau, with `a` the recognizer
path, `B(a)` its run-collapse, and the segmentation summed inside the reverse score,

    L_tau(x) = -log sum_a exp( (1/tau) [ log q_theta(a | x) + log P_psi(B(a)) + log p_phi(x | B(a)) ] ).

Write `w(a) = P_psi(B(a)) p_phi(x | B(a))`. Then `L_tau = F_tau / tau` with
`F_tau(q) = -tau log sum_a ( q(a) w(a) )^(1/tau)`, and the gradient in theta is
`-(1/tau) sum_a post_tau(a) grad log q(a)` with `post_tau proportional to (q w)^(1/tau)`
(`lattice_loss`, the arc-posterior surrogate). Minimizing `F_tau` over `q` on the simplex:

| tau | `F_tau` is | minimizer in q |
|---|---|---|
| 1 | `-log E_q[w]`, linear in q | a point mass at the generative MAP: mode seeking, zero conditional entropy |
| 2 | `-2 log sum_a sqrt(q(a) w(a))` = Bhattacharyya distance between q and post, minus `log p_gen(x)` | exactly `q = post` |
| tau > 1 general | not a divergence | `q proportional to post^(1/(tau - 1))`; flatter than post for tau > 2 |

Consequences:

- tau = 2 is the unique temperature at which the lattice term and the bound (B)/(E) share their
  minimizer in theta. They differ only in the divergence (Bhattacharyya versus KL), which changes
  the gradient weighting away from the optimum, not the fixed point.
- At tau = 1 the lattice term drops the conditional entropy and, at its own optimum, sits above
  `-log p_gen(x)` by `-log post(y_MAP | x) >= 0` per utterance. That quantity is the "missing
  entropy" in numbers. It vanishes when the posterior is sharp and is large in the cold transient.
- The budget schedule (`budget_temperature_schedule`, `SAE_4A_budget.md` Design) anneals 8 -> 2
  and holds 2; the no-schedule control holds 2 throughout. Every live arm ends at the principled
  value. The early tau = 8 stretch targets `post^(1/7)`, a flatter-than-posterior exploration
  phase, not a bound. "tau ending at 1" (listed as a lever not in the round) is the mode-seeking
  form; this note is the objection to it.
- Summing over paths gives each string `y` an implicit prior proportional to its number of
  run-collapse alignments, which peaks near T/2 tokens. That is one reason a rate term exists.

## 4. Term-by-term map

| piece of (B) | implemented as | status |
|---|---|---|
| `E_q[-log P_Y(y)]` | prior arc weight `log P_psi(k | h)` in the lattice, full-history n-gram | exact, per utterance |
| `I(X; Y)` lower bound, `E_q[log p_phi(x | y)]` | reverse segment score `G[k, s, d]` (`build_segment_table`); phi trained through `seg_post` on the joint `p_X q_theta` | the marginal-entropy piece; bound slack = reverse-model capacity (section 5) |
| `-H(q_theta(. | x))` | no explicit term | supplied implicitly by tau = 2 (section 3) |
| `-H(Q^Y)` at the corpus level | `L_agg = KL(c_text || c_hat)` on expected 1/2/3-gram counts, EMA over steps (`agg.py`) | the one place the marginal `Q^Y` appears, projected to low order; forward KL on marginals is mass-covering, so it does the anti-collapse job of the entropy term, through low-order statistics only (`SAE_1f.md`: low-order matching cannot separate truth from a matched-length decoy) |
| other direction: `KL( P_Y p_phi || p_X q_theta )`, cross term | BT: `E_{y ~ P_Y} E_{x_hat ~ p_phi(.|y)}[ -log q_theta(y | x_hat) ]` (`bt_blankfree.py`) | forward-KL coverage complementing the reverse-KL lattice; bias = synthetic-real gap in `x_hat` |
| length marginal | rate term | moment matching; corrects the path-count prior above |

## 5. Where the approximation is loose, ranked

1. **Temperature.** Keep tau at 2; below 2 the term is mode seeking and discards the entropy. The
   no-schedule-at-2 arm of the budget round is the principled form, not merely a control. The
   empirical question the round answers is whether the early 8 -> 2 anneal helps or hurts.
2. **Reverse-model slack** `KL(q_theta(x | y) || p_phi(x | y))`. `p_phi` treats segments as
   conditionally independent given `y`; the true posterior has cross-segment dependence (speaking
   rate, coarticulation). The recognizer will prefer transcripts that `p_phi` reconstructs easily.
   The lattice state already carries the last phone `h`, so an emission conditioned on the previous
   phone fits the same DP at a K-fold larger segment table; conditioning only the segment's first
   frame is the cheap variant.
3. **Explicit conditional-entropy term.** The frame-factorized path entropy `sum_t H(q_t)` is
   closed form, but `L_1 - H(q)` is a LOWER bound on `J` (Jensen), so adding it does not recover
   (B); tau = 2 already reaches the optimum of (B). Not recommended.
4. **ODM in the reverse-KL direction.** The reverse-KL projection onto n-gram marginals is
   `KL(c_hat || c_text) = -H(c_hat) + CE(c_hat, c_text)`. The cross-entropy part is what the
   lattice prior already applies per utterance with the full history, so the only new content is
   `-H(c_hat)`: a bonus for the entropy of the pooled output n-gram distribution (the marginal
   entropy maximization of information-maximization clustering). Liu et al. 2017 chose the forward
   direction (ours) precisely because the reverse cross-entropy alone has the trivial solution
   "emit the most frequent n-gram" and only the entropy bonus holds it up. Low expected value:
   marginal level only, and the forward KL already forces `c_hat` to cover `c_text`, which pins its
   entropy near `H(c_text)`.

## 6. The content-free point is stationary (added 2026-09-20, design review of `SAE_4A_infomax.md`)

At `q_theta(y | x) = P_psi(y)` for every x and `p_phi(x | y) = p_phi(x)`: the generative posterior
is `P_psi(y)`, so q is at its own fixed point; phi is trained on pairs whose y carries nothing about
x, so its optimum stays the marginal; the agg term is exactly satisfied and the rate term nearly so.
Every term of the exact bound (B) and of the implemented loss is stationary there, while the true
solution has strictly lower loss through `E[log p_phi(x | y)]`. The frame-factorized recognizer
cannot represent "sample a whole sentence from the prior", so the actual point is the nearest
x-independent frame-marginal solution, which is the band. Escape needs a term whose gradient at an
x-independent recognizer is nonzero and x-dependent: the InfoMax penalty `+ lambda H(Y | X)` of
`SAE_4A_infomax.md` is such a term only through the network's residual input dependence (at an
EXACTLY x-independent recognizer its gradient is the same for every frame and points at the
constant output, which the marginal terms resist). It is a deliberate mode-seeking departure from
(B), the opposite sign of item 3 in section 5, and a cold-start device to be withdrawn.

Measured after writing this (code review of the InfoMax arms, `SAE_4A_infomax.md` Design
amendment): the budget control's eval-mode entropy falls from 3.25 nats per output frame at
sub-epoch 1 to 0.30 at sub-epoch 10 and 0.19 at 21 while its PER stays in the band. The
stationary point above describes only the first sub-epochs; from about sub-epoch 10 the recognizer
sits in a confident, input-dependent, content-free partition, a theta-phi private code that the
tau = 2 posterior matching reaches on its own. The binding problem is therefore identifiability of
the code against the text prior (which partition the joint optimum selects), not the entropy of
the recognizer. Section 5 item 2 (reverse-model slack: the recognizer prefers whatever phi
reconstructs easily) is the mechanism that produces such a code.

## 7. Open

- The Bhattacharyya form (tau = 2) and the KL form (B) share a fixed point but not a gradient
  field; whether the KL form's weighting (`log q - log w` as the per-path advantage) trains better
  from cold is untested and not cheap: `E_q[log p_phi(x | B(a))]` is not per-arc additive once the
  segmentation is summed inside `p_phi`.
