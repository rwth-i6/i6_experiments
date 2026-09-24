# SAE_i6 reference: the training objective as a reverse-KL bound

Carried over from the JUPITER campaign's objective note (phase 4A, written 2026-09-20/21), with the
code pointers (section 9) remapped to the i6 port and one addition: section 10 records where the
implemented loss differs from sections 3-5 (found in the i6 port review); where they conflict, section
10 governs (in particular the tau = 2 statements of sections 4.1 and 6 hold only up to its
segmentation-entropy factor).
It defines the setting and every object the objective uses, records how the live blank-free
training objective relates to the distribution-matching objective it approximates, which term of
that objective each implemented term stands for, and where the approximation is loose. It is
interpretation, not a result; nothing here changes a gate. Notation is ASCII throughout (`tau`,
`theta`, `phi`, `psi`, `eta`, `rho`, `lambda` are written as words). Implementation detail
(network shapes, batching, dynamic-programming layout) is deliberately left out; the code that
realises each piece is named once in section 9 for the reader who wants it.

## 0. Setting, in one paragraph

Unsupervised phone recognition: a recognizer must map speech to phone strings using unpaired
speech and unpaired text only. No transcript, alignment or label enters training or model
selection (absolute rule of the campaign); labels are used read-side only, to report a phone error
rate. The approach is a cycle: the recognizer proposes a phone string for an utterance, a text
prior says how English that string is, and a reverse model says how well the string explains the
utterance's acoustics. The three are trained jointly so that the recognizer's output distribution
over strings matches the text prior while remaining predictive of the speech.

## 1. Objects and notation

**Data.**
- Speech: LibriSpeech train-clean-100 (28,539 utterances, about 100 h), unlabelled. Audio silence
  is removed by an unsupervised voice-activity detector before anything else; "frames" below are
  the retained frames.
- Text: a uniformly sampled 1,000,000-line window of the LibriSpeech language-model corpus, no
  overlap with the audio transcripts, converted to phone strings by a pronunciation lexicon with a
  silence symbol inserted at word boundaries with probability 0.5 and at sentence edges. This is
  the only text the prior, the corpus-statistics term and the rate target are built from.
- Evaluation: dev-other (and dev-clean), greedy decoding, phone error rate (PER) against the
  reference phone strings with SIL stripped from both sides. Reported, never optimised.

**Symbols.**
- `V`: the symbol inventory, 39 ARPAbet phones plus `SIL`, 40 symbols; no blank symbol.
- `x`: one utterance's acoustic input, a sequence of `S` frozen self-supervised speech features
  at 50 Hz (a pretrained wav2vec 2.0 encoder's intermediate layer; frozen, never trained).
- `z = z_1..z_S`: the same utterance as a sequence of `S` discrete acoustic units at 50 Hz,
  obtained by clustering the same frozen features into `K = 500` classes (k-means, frozen). `z`
  is the observation the reverse model explains; `x` is what the recognizer reads.
- `eta`: a 16-dimensional speaker vector per utterance, a fixed projection of the utterance's mean
  features; frozen; it lets the reverse model condition on the speaker without learning it.
- `y = y_1..y_U`: a phone string over `V`, `U` tokens.
- `T = ceil(S / 3)`: the recognizer's number of output frames (it runs at stride 3 on the 50 Hz
  input, so about 16.7 output frames per second).

**Models (the three trained objects).**
- Recognizer `q_theta`: a small convolutional network over the frozen features producing, at every
  output frame `t = 1..T`, a categorical distribution `q_t(k | x)` over the 40 symbols. A *path*
  `a = a_1..a_T` is one symbol per output frame; under the recognizer the path probability
  factorises, `q_theta(a | x) = prod_t q_t(a_t | x)`. Its *run collapse* `B(a)` merges adjacent
  repeats (`AA B B B SIL SIL` -> `A B SIL`); the phone string the recognizer proposes is `y = B(a)`.
  There is no blank, so every frame carries a symbol and a string of `U` tokens corresponds to
  every path with exactly `U` runs. The recognizer's distribution over strings is
  `q_theta(y | x) = sum_{a : B(a) = y} q_theta(a | x)`. Greedy decoding takes the per-frame argmax
  path and collapses it.
- Reverse model `p_phi(z | y, eta)`: a segmental (hidden semi-Markov) generative model of the unit
  sequence given the phone string. Each token `y_u = k` occupies one contiguous segment of `d_u`
  frames with an explicit duration distribution `p(d | k)`, minimum duration `d_min = 2` frames
  (structural: shorter segments have probability zero), maximum `D_k` (25 frames for phones, 50
  for SIL), and emits its frames from a categorical over the 500 units that depends on the token,
  its duration, the position inside the segment, and `eta`. No phonetic context across tokens.
  `p_phi(z | y, eta) = sum over segmentations (d_1..d_U, sum d_u = S) of prod_u p(d_u | k_u)
  prod_{frames r in segment u} p(z_r | k_u, d_u, r, eta)`. Writing `G[k, s, d]` for the log score
  of "token `k` occupies the `d` frames ending at frame `s`" (emission plus duration), the sum over
  segmentations is an exact dynamic programme over `G`.
- Text prior `P_psi(y)`: a frozen interpolated Witten-Bell trigram over the 40 symbols, fitted on
  the phonemised text; `P_psi(y) = prod_u P_psi(y_u | y_{u-2}, y_{u-1})` with a beginning-of-sentence
  context and no end-of-sentence term (a fixed convention of the campaign). Held-out perplexity
  9.56 per symbol. `psi` is never trained.

**Temperature.** `tau >= 1` scales every log-score inside the lattice sum by `1/tau` (section 3).
The live schedule is geometric from 8 to 2 over the first 20 % of the training sub-epochs, then
held at 2; every arm ends at `tau = 2`.

**Marginals.** `p_X` is the speech distribution (the empirical corpus); `Q^Y(y) = E_{x ~ p_X}
q_theta(y | x)` is the recognizer's induced output marginal over strings; `P_Y` is the prior as a
distribution over strings. `H(.)` is entropy, `I(.;.)` mutual information, `KL(.||.)` the
Kullback-Leibler divergence.

## 2. Target objective

The target is to make the recognizer's output marginal match the text prior:

    min_theta  KL(Q^Y || P_Y)  =  E_{y ~ Q^Y}[ -log P_Y(y) ]  -  H(Q^Y).

Both pieces need the marginal `Q^Y`, which is not computable: the first as an expectation over it,
the second as its entropy. (The direction is the "reverse" KL, expectation under the model, which
is mode-seeking; the other direction appears in section 5 as back-translation.)

## 3. Rewrite without the marginal

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
only place the marginalization over real speech lives. (In the implementation the reverse model
explains the unit sequence `z` rather than the features `x`; both are deterministic functions of
the audio, and `z` is the coarser one, so the bound holds for the information `y` carries about `z`.)

Equivalent form. With `w(y; x) = P_Y(y) p_phi(x | y)`, `p_gen(x) = sum_y w(y; x)` and
`post(y | x) = w(y; x) / p_gen(x)`,

    J(x) = KL( q_theta(. | x) || post(. | x) )  -  log p_gen(x).                                  (E)

So the reverse-KL bound is the negative evidence lower bound (ELBO) of the generative model "draw a
sentence from the text prior, then synthesise its acoustics with the reverse model", with the
recognizer as the inference network. Its minimizer in theta, at fixed phi and psi, is
`q_theta(. | x) = post(. | x)`, the Bayes posterior of that generative model. Its minimizer in phi
is maximum likelihood of the reverse model on `(x, y)` drawn from the joint `p_X(x) q_theta(y | x)`.

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

## 4. The implemented loss

Per utterance, the training loss is the sum of a lattice term, a rate term and a corpus-statistics
term, averaged over the batch; the lattice term is normalised by the utterance's retained frame
count `S`:

    L(x) = l_tau(x) + lambda_rate * L_rate(x) + lambda_agg * L_agg(batch),      l_tau(x) = L_tau(x) / S,

with `lambda_rate = 3` and `lambda_agg = 0.1` in the control bed. Optional terms tested in the
current phases (back-translation, the strong-scorer terms, the lexicon term) are added to this sum
with their own weights and are defined in section 5.

### 4.1 The lattice term

The lattice term is, at temperature `tau`, with `a` a recognizer path, `B(a)` its run collapse and
the segmentation of the units summed inside the reverse score,

    L_tau(x) = -log sum_a exp( (1/tau) [ log q_theta(a | x) + log P_psi(B(a)) + log p_phi(z | B(a), eta) ] ).

Everything inside the sum factorises token by token: a path contributes `log q_t(a_t | x)` per
frame, one prior term `log P_psi(k | h)` per new token (with `h` the two previous tokens), and one
reverse segment score `G[k, s, d]` per new token, where the token's segment of `d` frames ends at
unit frame `s`. The sum over paths and segmentations is therefore one exact dynamic programme over
states (recognizer frame `t`, unit frame `s`, prior history `h`), with the constraint that the
segment end stays within 25 unit frames of the emitting recognizer frame (`|s - 3t| <= 25`) and
that segment durations lie in `[2, D_k]`. The DP and its gradient are exact; nothing is sampled.
`L_tau / S` is the quantity called `l_tau` in the phase files; `-L_tau` is called `log Z`.

Write `w(a) = P_psi(B(a)) p_phi(z | B(a), eta)`. Then `L_tau = F_tau / tau` with
`F_tau(q) = -tau log sum_a ( q(a) w(a) )^(1/tau)`, and the gradient in theta is
`-(1/tau) sum_a post_tau(a) grad log q(a)` with `post_tau proportional to (q w)^(1/tau)`, the
tempered posterior over paths (the arc-posterior surrogate). Minimizing `F_tau` over `q` on the
simplex:

| tau | `F_tau` is | minimizer in q |
|---|---|---|
| 1 | `-log E_q[w]`, linear in q | a point mass at the generative MAP: mode seeking, zero conditional entropy |
| 2 | `-2 log sum_a sqrt(q(a) w(a))` = Bhattacharyya distance between q and post, minus `log p_gen(x)` | exactly `q = post` (for the string-level form; the implemented joint form: section 10) |
| tau > 1 general | not a divergence | `q proportional to post^(1/(tau - 1))`; flatter than post for tau > 2 |

Consequences:

- `tau = 2` is the unique temperature at which the lattice term and the bound (B)/(E) share their
  minimizer in theta. They differ only in the divergence (Bhattacharyya versus KL), which changes
  the gradient weighting away from the optimum, not the fixed point.
- At `tau = 1` the lattice term drops the conditional entropy and, at its own optimum, sits above
  `-log p_gen(x)` by `-log post(y_MAP | x) >= 0` per utterance. That quantity is the "missing
  entropy" in numbers. It vanishes when the posterior is sharp and is large in the cold transient.
- The schedule anneals 8 -> 2 and holds 2. Every live arm ends at the principled value. The early
  `tau = 8` stretch targets `post^(1/7)`, a flatter-than-posterior exploration phase, not a bound.
  "tau ending at 1" is the mode-seeking form; this note is the objection to it.
- Summing over paths gives each string `y` an implicit prior proportional to its number of
  run-collapse paths, which peaks near `T/2` tokens. That is one reason a rate term exists.

### 4.2 The rate term

Nothing in `L_tau` prices an empty or near-empty output (durations price segments that exist). The
rate term pins the expected output rate to a label-free target:

    L_rate(x) = ( (E[N] / T_audio - rho) / rho )^2,

where `E[N]` is the expected number of emitted non-SIL tokens under the tempered posterior
`post_tau` (a by-product of the same DP), `T_audio` the utterance's original duration, and
`rho = 9.66` phones per second, obtained as (phones per word in the phonemised text) x (a disclosed
constant of 2.7 words per second for read English). No transcript or alignment enters `rho`.

### 4.3 The corpus-statistics term

    L_agg = KL( c_text || c_hat )   over symbol unigrams and adjacent-pair bigrams,

where `c_text` are the unigram and bigram frequencies of the phonemised text and `c_hat` the
expected unigram and bigram counts of the collapsed output strings under the recognizer's own
per-frame distributions, accumulated over the batch with an exponential moving average across
steps. Because the recognizer factorises per frame, these expected counts have a closed form (a
token of `k` starts at frame `t` iff `a_t = k` and `a_{t-1} != k`). It is a collapse guard priced
small, not evidence: low-order statistics cannot separate the truth from a length-matched decoy.

## 5. Term-by-term map

| piece of (B) | implemented as | status |
|---|---|---|
| `E_q[-log P_Y(y)]` | prior term `log P_psi(k \| h)` per new token inside the lattice, full trigram history | exact, per utterance |
| `I(X; Y)` lower bound, `E_q[log p_phi(x \| y)]` | reverse segment score `G[k, s, d]` inside the lattice; phi trained through the same tempered posterior on the joint `p_X q_theta` | the marginal-entropy piece; bound slack = reverse-model capacity (section 6) |
| `-H(q_theta(. \| x))` | no explicit term | supplied implicitly by `tau = 2` (section 4.1) |
| `-H(Q^Y)` at the corpus level | `L_agg` on expected unigram/bigram counts | the one place the marginal `Q^Y` appears, projected to low order; forward KL on marginals is mass-covering, so it does the anti-collapse job of the entropy term, through low-order statistics only |
| length marginal | rate term `L_rate` | moment matching; corrects the path-count prior above |
| other direction, `KL( P_Y p_phi \|\| p_X q_theta )` | back-translation (below) | forward-KL coverage complementing the reverse-KL lattice; bias = synthetic-real gap |

Optional terms under test, each defined against the objects of section 1:

- **Back-translation (BT), weight `lambda_bt`.** Draw a sentence `y` from the text side, let the
  current reverse model synthesise a unit sequence for it (durations and speaker sampled, the
  synthetic units rendered by collage from real retained frames), and train the recognizer to
  recover `y` from that synthetic input: `E_{y ~ P_Y} E_{x_hat ~ p_phi(.|y)} [ -log q_theta(y | x_hat) ]`,
  with `q_theta(y | x_hat)` the blank-free run-collapse sum over paths. The reverse model receives
  no gradient from it. It is the cross term of the forward KL and an auxiliary inside the cycle,
  never a standalone recognizer trainer.
- **Strong-scorer terms (prior phase).** A strong phone-string scorer `p_strong` (a 3.3 M-parameter
  transformer phone language model trained on the same phonemised text; frozen) cannot be placed
  inside the lattice, so it enters as a correction outside it, with reward per string
  `r(y) = log p_strong(y) - log p_uni(y)`, `p_uni` the unigram of the same text (the subtraction
  removes the length cost both models share; SIL dropped). Two estimators of `E_{y ~ q_theta(.|x)} r(y)`:
  the *soft* form feeds the scorer the per-segment posterior over symbols along the recognizer's
  best segmentation as soft symbol vectors, with a straight-through gradient (`lambda_soft`); the
  *sampled* form draws `G = 8` strings from the tempered lattice posterior and uses the
  score-function estimator with centred advantages `r(y_g) - mean_g r` (`lambda_sf`). Both are
  normalised per retained frame. A null arm permutes the scorer's symbol inputs.
- **Lexicon term (lexlat phase).** The prior inside the lattice is extended so that a path is also
  priced by a word trigram over dictionary words, the phone string being parsed into
  pronunciations from a fixed lexicon with an escape path for out-of-lexicon spans (`lambda_lex`,
  turned on after a warm-up). It stays inside the exact sum, so no separate normalisation is
  introduced. The form that ran is a second k2 graph (HLG) intersected with the recognizer's
  emissions; its design, cost and results are in `SAE_i6_ref_lexicon.md`.

## 6. Where the approximation is loose, ranked

1. **Temperature.** Keep `tau` at 2; below 2 the term is mode seeking and discards the entropy.
   The no-schedule-at-2 arm of the budget round is the principled form, not merely a control. The
   empirical question the round answers is whether the early 8 -> 2 anneal helps or hurts.
2. **Reverse-model slack** `KL(q_theta(x | y) || p_phi(x | y))`. `p_phi` treats segments as
   conditionally independent given `y`; the true posterior has cross-segment dependence (speaking
   rate, coarticulation). The recognizer will prefer transcripts that `p_phi` reconstructs easily.
   The lattice state already carries the last phone `h`, so an emission conditioned on the previous
   phone fits the same DP at a larger segment table; conditioning only the segment's first frame is
   the cheap variant.
3. **Explicit conditional-entropy term.** The frame-factorised path entropy `sum_t H(q_t)` is
   closed form, but `L_1 - H(q)` is a LOWER bound on `J` (Jensen), so adding it does not recover
   (B); `tau = 2` already reaches the optimum of (B). Not recommended.
4. **Corpus statistics in the reverse-KL direction.** The reverse-KL projection onto n-gram
   marginals is `KL(c_hat || c_text) = -H(c_hat) + CE(c_hat, c_text)`. The cross-entropy part is
   what the lattice prior already applies per utterance with the full history, so the only new
   content is `-H(c_hat)`: a bonus for the entropy of the pooled output n-gram distribution (the
   marginal entropy maximisation of information-maximisation clustering). Liu et al. 2017 chose the
   forward direction (ours) precisely because the reverse cross-entropy alone has the trivial
   solution "emit the most frequent n-gram" and only the entropy bonus holds it up. Low expected
   value: marginal level only, and the forward KL already forces `c_hat` to cover `c_text`, which
   pins its entropy near `H(c_text)`.

## 7. The content-free point is stationary (added 2026-09-20, design review of the InfoMax round, `SAE_i6_ref_blankfree.md`)

"Content-free" means a recognizer whose output carries no information about the utterance: a
cold-started arm that settles at a PER near the chance level of the prior (the "band" in the phase
files) while satisfying the prior and the corpus statistics well.

At `q_theta(y | x) = P_psi(y)` for every `x` and `p_phi(x | y) = p_phi(x)`: the generative posterior
is `P_psi(y)`, so `q` is at its own fixed point; phi is trained on pairs whose `y` carries nothing
about `x`, so its optimum stays the marginal; the corpus term is exactly satisfied and the rate
term nearly so. Every term of the exact bound (B) and of the implemented loss is stationary there,
while the true solution has strictly lower loss through `E[log p_phi(x | y)]`. The frame-factorised
recognizer cannot represent "sample a whole sentence from the prior", so the actual point is the
nearest `x`-independent frame-marginal solution, which is the band. Escape needs a term whose
gradient at an `x`-independent recognizer is nonzero and `x`-dependent: the InfoMax penalty
`+ lambda H(Y | X)` of the InfoMax round (`SAE_i6_ref_blankfree.md`) is such a term only through the network's residual input
dependence (at an EXACTLY `x`-independent recognizer its gradient is the same for every frame and
points at the constant output, which the marginal terms resist). It is a deliberate mode-seeking
departure from (B), the opposite sign of item 3 in section 6, and a cold-start device to be
withdrawn.

Measured after writing this (code review of the InfoMax arms, InfoMax design
amendment): the budget control's eval-mode entropy falls from 3.25 nats per output frame at
sub-epoch 1 to 0.30 at sub-epoch 10 and 0.19 at 21 while its PER stays in the band (the later
registered dev-other reads of ctrl_50 replace these: 3.04 / 2.41 / 0.38 / 0.23 nats at sub-epochs
1 / 4 / 10 / 25, `SAE_i6_ref_blankfree.md` section 7; the conclusion is unchanged). The
stationary point above describes only the first sub-epochs; from about sub-epoch 10 the recognizer
sits in a confident, input-dependent, content-free partition: a private code shared by theta and
phi that the `tau = 2` posterior matching reaches on its own. The binding problem is therefore
identifiability of the code against the text prior (which partition the joint optimum selects),
not the entropy of the recognizer. Section 6 item 2 (reverse-model slack: the recognizer prefers
whatever phi reconstructs easily) is the mechanism that produces such a code. The prior and lexlat
phases attack exactly this: a prior whose accepted family of codes is smaller (section 5, strong
scorer and lexicon).

## 8. Open

- The Bhattacharyya form (`tau = 2`) and the KL form (B) share a fixed point but not a gradient
  field; whether the KL form's weighting (`log q - log w` as the per-path advantage) trains better
  from cold is untested and not cheap: `E_q[log p_phi(x | B(a))]` is not per-arc additive once the
  segmentation is summed inside `p_phi`.

## 9. Where each piece lives

In the i6 port, all under `recipe/i6_experiments/users/wu/experiments/unsupervised_asr/`: lattice term
and its DP, `model/lattice.py` (module docstring); blank-free topology and expected run counts,
`model/blankfree.py`; the blank-free model and loss assembly, `model/blankfree_model.py` (base class
`model/emc_model.py`) and the train step `model/train_step.py`; reverse model, `model/reverse.py`; text
prior, `model/prior.py` (fitted by `lm/phone_prior.py`); rate term, `model/rate_term.py` (the train
step uses its finite-difference surrogate); corpus-statistics term, `model/agg.py`; temperature and
learning-rate schedules, `training/schedules.py`; lexicon term (k2 second graph), `model/lexlat_k2.py`,
`model/lexlat_k2_train.py`, curriculum `model/lexlat_train.py`, resources `model/lexlat.py` and
`lm/hlg.py`. NOT ported: back-translation (`bt_blankfree`) and the strong-scorer terms (`soft_scorer`,
`sf_scorer`); their history is in `SAE_i6_ref_emc1.md`, `SAE_i6_ref_emc2.md` and `SAE_i6_ref_lexicon.md`. Bed constants and
provenance: `SAE_i6_ref.md` and `SAE_i6_ref_blankfree.md`; the prior's text and SIL convention:
`SAE_i6_ref_lexicon.md` (prior phase, bed and constants).

## 10. Where the implemented loss differs from sections 3-5 (i6 port code review, 2026-09-24)

Found by reading the ported code (`SAE_i6/reports/test_plan_2026-09-24.md`, items S1, S2, S7) and
pinned numerically by the P0 tests (T1.3, T1.11). Decision (orchestrator): the CODE defines the bed,
because every banked number came from it; this note is corrected to it, not the reverse.

- **Temperature acts on joint (path, segmentation) assignments (S1).** The lattice sums
  `sum_a sum_sigma exp((1/tau)[log q(a) + log P(B(a)) + log p_phi(z, sigma | B(a), eta)])` over the
  segmentations sigma allowed by the band, not `exp((1/tau)[... + log sum_sigma p_phi(z, sigma | ...)])`.
  The two agree only at tau = 1 with a non-binding band (this was known on JUPITER: "the joint tempering
  equals the string-level formula only at tau = 1"). Consequence for section 4.1: writing
  `W_tau(a) = sum_sigma (P p_phi(z, sigma))^(1/tau)`, `F_tau(q) = -tau log sum_a (q(a) w~(a))^(1/tau)`
  with `w~(a) = W_tau(a)^tau`, so the table holds with w replaced by w~. At tau = 2,
  `w~(a) = P(B(a)) p_phi(z | B(a)) exp(H_1/2(pi_a))`, where `pi_a` is the posterior over the path's
  admissible segmentations and `H_1/2` its Renyi entropy of order 1/2. The tau = 2 minimizer is
  therefore NOT exactly the Bayes posterior `post`: it up-weights paths whose segmentation is ambiguous,
  by up to the number of admissible segmentations. "tau = 2 is the principled value" holds only up to this
  segmentation-entropy factor.
- **The band holds every lattice state (S2):** `|s - 3t| <= 25` is enforced at every recognizer frame,
  including repeat frames, so a segment end is compared with `3(t_emit + 1)` and with every frame up to
  the next emission, not only with the emitting frame.
- **The lexicon term is a separate added loss (S7):** `lam_lex * (log Z_H - log Z_HLG) / S` computed by k2
  on the recognizer's emissions alone (no phone trigram, no reverse model inside it), added to the
  unchanged lattice term; `SAE_i6_ref_lexicon.md` B4. Its own open numerical questions (pruned-G
  normalisation, tropical epsilon removal, escape multiplicity, segment order; S3-S6) are P0 test items.
