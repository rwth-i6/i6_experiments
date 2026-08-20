# Historical snapshot — PLAN_1G before the 2026-08-19 rewrite

**This file is non-normative.** The current plan is `../PLAN_1G.md`. This snapshot preserves the
original registered gates, completed verdicts, and unexercised specifications for provenance.
Experimental evidence remains in `../SAE_1g.md`.

---

# PLAN_1G — training-free unsupervised initialization: channel estimation decoded through the text language model

Sub-plan of PLAN.md section 1g, registered 2026-08-18 on the USER's phase-1g planning request (five
proposed methods, listed and adjudicated in "Proposal adjudication" below). Holds the 1g design
detail so the PLAN.md section stays a page; collapses to a verdict when 1g's question closes.

**What separates 1g from 1f in one sentence.** Every 1f entry learned a *fixed statistic map* from
units to text symbols and decoded it by per-segment argmax with no language model anywhere at decode
time; 1g learns a *channel* — the probability of each unit given each text symbol — and decodes it as
`argmax over symbol strings of [ text language model score + channel score ]`, so the text corpus
constrains the output at decode time and not only at fitting time.

**Text side, per the USER's 2026-08-18 ruling and binding on everything below: LEXICON-FREE IS
PRIMARY.** The hidden symbols are BPE tokens (or characters) learned on the raw unpaired corpus, not
phones; the phone side survives as one reference arm whose gap prices the pronunciation lexicon. Where
a section below still says "phone", read "text symbol" and see the text-side ruling for what the
symbol is.

---

## Standing measured anchors (all first-hand from the logs and job outputs; labels eval-only)

These are the numbers every 1g decision is read against. Audio side throughout: the 8,416-utterance
LibriSpeech seed stream, 20.48 h, frozen wav2vec2-Large-LV60 layer 15 at 50 Hz, k-means K=500;
`seg12.5` = adjacency-constrained Ward segment pooling to 9.77 segments/s, 720,315 segment tokens.
Text side: LibriSpeech language-model corpus phonemized to 39 stress-free ARPAbet phones, 9.86
phones/s, KenLM 4-gram perplexity about 8.5 (2.14 nats/phone).

| anchor | value | source |
|---|---|---|
| best per-unit map, `seg12.5`, dev-other | PER 0.414 = sub 0.230 + ins 0.067 + del 0.117 | `SAE_1f.md` approach 3 |
| best per-unit map, raw 50 Hz, dev-other | PER 0.832 = sub 0.132 + ins 0.692 + del 0.008 | `SAE_1f.md` approach 3 |
| supervised probe, kernel-4 context, same path | PER 0.3565 | `SAE_1f.md` conclusion 30 |
| phone information, `seg12.5` / raw | PNMI 0.581 / 0.682; H(phone) 3.292 nats | `SAE_1f.md` approach 3 |
| content-free nulls, `seg12.5` dev-other | random map 0.8946, pseudo-pair 0.9239 | `LexFreeMatchJob.rk48Zk5U6jzW` |
| best 1f candidate to date | statistics matching 0.8580; fingerprint solve 0.8809 | `SAE_1f.md` approach 8 / 4 |
| GAN incumbent this replaces | 0.168 dev-other PER (`SAE_1c.md`); its self-trained student 0.172 phone PER, whose lexicon + word-4-gram decode is 17.96/21.87 WER (`SAE_1d.md`) — the WER is off the student, not the GAN | `SAE_1c.md`, `SAE_1d.md` |
| coarticulation | 26.7 % of phone transitions invisible in the unit stream (5.9x inflation) | `SAE_1f.md` conclusion 5 |
| simulation calibration of the real stream | worse than a fertility-1 channel with 35 % random emissions | `SAE_1f.md` conclusion 11 |

**Information budget, stated so 1g's expectations are registered rather than hoped.** The text
language model leaves 2.14 nats/phone of uncertainty against H(phone) = 3.292, so 1.15 nats/phone of
English redundancy is available to a decoder that uses it and is thrown away by a per-segment argmax.
The `seg12.5` stream supplies at most 0.581 x 3.292 x 1.18 = 2.26 nats of phone information per gold
phone, against the 2.14 nats the language model still needs. The margin is +0.12 nats/phone and the
2.26 is an upper bound (adjacent segments are not conditionally independent), so the honest
expectation for a perfect channel plus a perfect language-model decode on this stream is a real but
not near-zero error rate. 1g.1 measures it instead of arguing it.

---

## Theory battery (registered 2026-08-18): what is, and is not, the obstacle

Findings of the dedicated identifiability/sample-complexity study run for this plan. **Provenance rule
applied throughout, because it decides how each line may be used.** Lines marked LITERATURE are
first-hand reads of published papers and are citable as literature. Lines marked CODE are re-reads of
this project's own source and I verified them myself. Lines marked SIMULATION are the study's own
synthetic runs on an uncommitted scratchpad: they are **motivation and pre-registered predictions, never
results**, and no gate below may be discharged by one — each is paired with the committed job that would
retire it.

**1. Identifiability is not the obstacle (LITERATURE).** Allman-Matias-Rhodes 2009 Theorem 6 requires
`C(k+K-1, K-1) >= V`; at V=39 hidden symbols and K=500 units, k=1 already gives 500 >= 39, so **three
consecutive segments generically identify the emission matrix up to symbol relabelling**. The study
reports the margin as 117 against a requirement of 80. Nothing in 1g fails for want of identifiability.

**2. Sample size is not the obstacle (SIMULATION).** In a well-specified pinned-language-model
simulation, 720,000 segments puts the induced decode within 0.0012 absolute of the model optimum, and
about 100,000 segments (2.8 h) within 0.01; re-dumping 960 h buys 0.0009. **Consequence registered:
960 h is not funded for emission-matrix accuracy.** The one exception the study found is a full diphone
emission table, where 758,979 parameters go from 0.9 to 44 segments per parameter — that is a real
reason and the only one.

**3. Pair statistics do not identify the channel even with the transition matrix pinned (SIMULATION,
exact Jacobian).** Rank of the moment map at (V phones, K units) = (6,15), (8,20), (10,20), (12,20):
unigram+pair leaves nullity 2, 3, 4, 5 — cleanly **V/2 - 1** free directions, with a machine-zero
spectral gap; adding the triple marginal drives nullity to exactly 0. Extrapolated to V=39 that is
about 18-19 free directions. This is an **algebraic** under-determination: no volume of audio and no
optimiser removes it. It covers, in one statement, 1f entries (1) statistics matching, (3) unary
fingerprints, (5) unit-BPE word matching and the unit-co-occurrence-graph work, and on the text side it
covers the USER's proposals 2 and 4. Full-sequence likelihood is the only estimator on the ladder that
consumes the triple statistic and every higher order, i.e. **the only one attacking an identified
problem**. Retires when 1g.2's E1 reads the same nullity on the real stream's own moment map.

**4. The moment/spectral route is dead on data volume by six orders of magnitude (SIMULATION).**
Resolving the 39th singular direction of the 500-by-500 unit pair table needs about 1e11 segments
(~3 million hours); at 720k segments only **10 of 39** directions clear the sampling-noise floor, and
28 of 39 at 34 M segments. **Registered consequence:** spectral and three-view tensor estimators are
not funded on the 500-unit alphabet. The one form that is not dead is on a **reduced observation
alphabet of about 30-60 units**, which is why the downward-K ceiling read in 1g.6 is worth its hour.

**5. The likelihood/accuracy anti-correlation is reproduced and attributed to emission
misspecification, not to decipherment (SIMULATION).** Starting EM *at the true channel* with
transitions pinned: a well-specified control stays put (likelihood -0.0038, error +0.0060, rank
correlation -1.000 = aligned). Under 30 % coarticulation, a two-speaker mixture, and variable fertility
the run **buys +0.067 to +0.080 nats/segment of likelihood and pays +0.019 to +0.151 absolute error**,
rank correlation up to +1.000. From random restarts, every misspecified generator finds solutions with
**strictly higher likelihood than the true channel** while transcribing far worse; the well-specified
control ranks the truth highest. **Scope amendment to the entry-(2) closure, same form as the 1a
amendment below:** the logged anti-alignment licenses "the fertility-HMM channel used there was
misspecified enough for the Kullback-Leibler projection to sit far from the truth", not "likelihood
decipherment cannot work on this stream". Entry (2)'s channel addressed fertility but not the two
mismatches this project has since measured directly — 26.7 % invisible phone transitions and 8,416
utterances of unmodelled speaker variation. **What does NOT change:** model likelihood still may not
select anything, anywhere in 1g. The study strengthens that rule rather than weakening it, and gives
the reason the section-1.0 decode-side metric behaves differently — a decode-side score is a functional
of the posterior, a data-side score a functional of the marginal, and only the latter is what the
Kullback-Leibler projection moves.

**6. The binding constraint is the optimisation basin — which is exactly what 1g is trying to build
(SIMULATION).** From a random start, EM reaches 0.80-0.98 error **even when the model is perfectly
specified, the language model is pinned, and the truth has strictly higher likelihood** (model optimum
0.309). Seeding a fraction p of the 500 units correctly and running 30 pinned-LM EM iterations:

| p (units seeded correctly) | 0.00 | 0.10 | **0.20** | 0.30 | 0.50 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|---|
| seed's own decode error | 0.972 | 0.907 | **0.847** | 0.807 | 0.640 | 0.492 | 0.364 |
| after pinned-LM EM | 0.979 | 0.485 | **0.399** | 0.373 | 0.338 | 0.331 | 0.327 |

There is a sharp cliff between p=0 and p=0.10, and **a seed only needs about 20 % of its units right**.
The registered 1f candidates sit at 0.8580 and 0.8809 against a 0.8946 null, and have been read as
failures *because they barely clear the null* — this table says that read may be measuring the wrong
quantity, since the question is not whether the seed is good but whether it is inside the basin.
**Nobody has run Baum-Welch with a pinned text language model from those two seeds.** That is the
cheapest untried experiment in the program and it is registered as 1g.1 E5 below. Three caveats carried
into the gate: the simulation is well-specified, so under real coarticulation and fertility the recipe
must be **EM with early stopping**, not EM to convergence; the stopping point cannot be chosen by model
likelihood (finding 5), so it must be a decode-side criterion; and the study's per-segment "map error"
is not this project's post-dedup edit-distance PER, so the correspondence must be calibrated by one CPU
pass **before** any number from the table is acted on.

**7. What a language model at decode time is actually worth (SIMULATION, and it sets 1g.1's
expectation).** An ideal noisy-channel decoder — exact emission matrix, exact language model, exact
inference — reaches about **0.20-0.25** against 0.365 memoryless on the same simulated channel: the
language model buys **21-34 % relative**, about a third of it from bigram to trigram. Applied to the
logged 0.4148 anchor that predicts **0.28-0.33** dev-other. The sequence-level Fano floor is 0.6-2.2 %,
and the gap between it and 0.20-0.25 is the quantitative statement that **English phone redundancy is
real but is not error-correcting distance** — two sequences differing in one phone are usually both
plausible English. Registered corollaries: the 0.4148 memoryless ceiling is beatable by a worthwhile
margin but not by an order of magnitude; and **closing to the 0.168 GAN incumbent requires better
acoustic information, not a better decoder** — the logged 0.3565 supervised kernel-4 probe is the
existence proof that the remaining information is in context.

**8. The anchor/separability condition is badly violated (SIMULATION, with a logged number forcing the
direction).** On a channel calibrated to `seg12.5`'s measured information content, units at posterior
purity >= 0.99 anchor 0 of 39 symbols; >= 0.95 anchors 1; >= 0.90 anchors 3; >= 0.80 anchors 6; >= 0.70
anchors 12. The logged mean unit purity of 0.652 forces the direction independently. Anchor-word
non-negative matrix factorization (Arora et al., Stratos et al.) is therefore not applicable without a
robustness extension. **Cheap committed read that would retire this:** the histogram of per-unit maximum
posterior from the existing audit job, one CPU pass — if some subset of symbols does have near-pure
units, a *partial* anchor solve (pin those, solve the rest) is training-free and becomes a real option.

**9. Precision on the entry-(4) closure (CODE, verified first-hand).** `match_screen.py:147` reads
`smin = float(s[cols - 1]) if s.size >= cols else 0.0`, so on every rung where the number of positions
is below the number of unit types the reported `sigma_min` is a literal `0.0` returned by the guard,
not a measured singular value. The closure is still correct — a matrix with fewer rows than columns
cannot have full column rank — but the load-bearing rows are `brown100` (2e-17, 100 columns and enough
rows) and `raw` (5e-33), not the exact zeros. The study also supplies the root cause on our own text:
the phone bigram chain mixes at |lambda_2| = 0.336, so the positional unigram is within 0.001 of its
asymptote after 6.3 phones, and the real 200-by-39 text-side matrix carries **8 usable dimensions at
400k sentences** (37 at the full corpus) with sigma_39/sigma_1 = 4.2e-4 — independently consistent with
the 3e-4 that arXiv:2603.02285 reports. **Entry (4) is closed permanently and for a stronger reason
than logged: no re-dump, no pooling change and no amount of audio can fix a limit set by English
phonotactics mixing in five phones.** The same read prices that paper's Theorem 1: its constant
`N^2 ||P_C^+||_1` is of order 1e7 here, so it is a possibility statement, not an error guarantee.

---

## Text-side ruling: lexicon-free is PRIMARY (USER 2026-08-18)

**The instruction.** 1g's goal is an approach as simple as possible and ideally lexicon-free — text
side at BPE or word level rather than phones. This replaces the phone-first default the sections below
were first drafted against, and it is a change of primary arm, not an added arm.

**The distinction that makes it cheap, stated once because everything below depends on it.** Two
different things get called a lexicon. A PRONUNCIATION lexicon (grapheme-to-phoneme) is real
linguistic knowledge and is what 1g is now trying to do without. A WORD LIST WITH SPELLINGS is not —
it is the vocabulary of the unpaired text corpus, readable off the corpus itself, and using it costs
nothing beyond the text we already consume. So a word-level language model over raw text, composed
with a spelling lexicon, is available to the lexicon-free arm at zero pronunciation-lexicon cost. This
matters more than it sounds: the verified precedent for noisy-channel speech decipherment deciphers
into GRAPHEMES, ladders a character n-gram language model from bigram to 5-gram, and takes its final
and strongest stage with a word trigram language model plus a grapheme lexicon. **The lexicon-free
form is the precedent's own form; the phone-level form was our deviation from it.**

**Granularity, chosen by measurement rather than by preference.** The channel is
`hidden text symbol -> unit`, so the hidden alphabet size sets the parameter count: 39 phones give
19,461 free parameters, characters about 28 give 14,000, BPE-512 gives 256,000 — which is 2.8
observations per parameter on the 720,315-segment stream and is not estimable here. The rate has to
match the audio rung as well: banked text rates are 14.55 characters/s (the length hinge's own
constant) and 5.39 BPE-512 tokens/s, so a log-interpolation puts a BPE vocabulary near 128 at about
the 9.77 segments/s of `seg12.5` — the rung every banked null and ceiling already sits on. RULING:
the primary text side is BPE at the vocabulary whose MEASURED token rate is closest to the chosen
audio rung, that vocabulary reported not assumed, with characters as the alternative if 1g.0 prefers
a finer rung. One reference arm only — phones from T_phi — kept for the reason ruling 3 of `PLAN_1F.md`
established: the gap between the arms is the measured price of the pronunciation lexicon, reported
rather than argued. No further arms; the simplicity instruction binds the arm count too.

**What this changes downstream.** (a) The lexicon-free arm outputs TEXT directly, so its natural read
is plain word error rate — the currency the deliverables ladder uses — while the phone arm needs the
lexicon a second time to become text at all, which is the extra touchpoint ruling 3 already required
disclosing. Both arms additionally report PER through eval-only lexicon expansion so the banked
`seg12.5` nulls price them, exactly the ruling-3 protocol. (b) Nulls and ceilings are re-banked per
text side and per rung by the existing `LexFreeMatchJob` construction; the lexicon-free oracle ceiling
needs the one eval-only ingredient ruling 3 named, gold word alignments plus tokenization of the gold
transcript. (c) 1g.4's articulatory-feature-table variant is DEMOTED — a phone-to-feature table is
phone-level lexical knowledge and pulls against the instruction, so it survives only as an option on
the phone reference arm, and the open supervision-cost decision that variant carried is withdrawn
rather than put to the user.

**Honest risk, recorded before the arm runs.** English orthography is irregular and non-local, so a
memoryless per-symbol channel is a WORSE fit to characters than to phones — `PLAN_1F.md` ruling 3
recorded this and it has not gone away. The repair is the precedent's own: the word-level language
model and spelling lexicon at decode, which is exactly the component available here for free. This is
why the word-level constraint moves from a conditional late rung into the 1g.1 decode battery.

---

## Scope amendment to the section-1a closure (planner ruling 2026-08-18, pre-run)

PLAN.md section 1a says decipherment is "CLOSED permanently, on a bound", and PLAN_1F's reference
verdict rejected arXiv:2603.02285's training loss as "the 1a fertility-HMM decipherment likelihood in
gradient form". The USER's proposal 1 is that lane, so the closure has to be re-read against what it
actually measured before 1g may spend anything. Read first-hand in `SAE_1a.md`, the closure rests on
two legs and **neither covers the discrete channel decoded through a language model on the pooled
stream**:

1. *Likelihood anti-aligned with accuracy.* Measured in `SAE_1a.md` approach 5 on a **continuous**
   Gaussian-emission hidden semi-Markov model over PCA-48 features: log-likelihood per frame rises
   monotonically (-150.22 to -148.80) while phone error rate degrades 0.275 to 0.392 from an oracle
   start. Conclusion 5 gives the mechanism in the log's own words — "the per-segment prior
   contributes O(few) nats against O(d*frames) of density term" — which is an argument about a
   48-dimensional continuous density outweighing the language model, and it does not transfer to a
   categorical channel: at 9.77 segments/s the emission term spans 0 to 6.2 nats per segment (at most
   61 nats/s) against the language model's 21 nats/s, a ratio near 3 to 1 rather than the one-to-tens
   of the continuous case. On **discrete** units the same log measured the opposite sign: approach 4,
   row "HMM channel, init from truth", reads recovery 0.97 and "highest LL — objective is
   well-aligned, EM is init-limited".
2. *Capped by the oracle-map ceiling.* That ceiling was 0.53-0.63 when 1a closed and is 0.4148 on
   `seg12.5` today, and — the load-bearing point — it is a **memoryless-decode** ceiling: it bounds
   methods that emit a fixed unit-to-phone lookup, which is exactly what every 1f entry did, and it
   does not bound `argmax over phone strings of [language model + channel]`.

3. *Third leg, added 2026-08-18 from the fan-out.* The 1a run that produced the anti-alignment used
   **Viterbi hard EM** (`SAE_1a.md` approach 5, stated in its own configuration line). Classification
   maximum likelihood is an inconsistent estimator outside special symmetric cases such as equal
   mixing proportions (Ahfock and McLachlan, arXiv:2004.06237), so the closed run used the biased
   variant of the estimator 1g proposes to use in its soft form. That does not explain the whole
   0.275-to-0.392 degradation, and it is not offered as one; it is a third respect in which the closed
   configuration and the proposed one differ.

What the closure DOES still license, unamended: not funding **continuous** generative maximum
likelihood over features (the 1a approach-5 lane), which 1g does not propose. What the real-data
evidence against the discrete lane actually is, stated exactly: one row, `SAE_1a.md` approach 5,
"GW-OT K=500 (the validated 1a(iii), truly unsupervised) -> PER 0.86, map frame-acc 0.146" — the
Gromov-Wasserstein *initializer* collapsing on the **raw, un-pooled** stream, whose oracle ceiling is
0.832 and whose observable graph correlation is 0.373, and whose frame-accuracy number the 2026-07-14
verifier flagged as uncommitted scratchpad. On the pooled stream (ceiling 0.4148, graph correlation
0.595, boundary-crossing pair share 0.700) the discrete lane has never been run. **Ruling: 1a's
"do not revisit" is amended in scope to continuous generative maximum likelihood over features; the
discrete channel with a pinned language model on pooled units is reopened as 1g, and every 1g
decipherment step is gated on 1g.2, which re-tests 1a's own anti-alignment finding on the new
representation before any real-stream fit is funded.** This amendment is a planner call, dated,
replacing nothing in `SAE_1a.md`, whose conclusions stand against their own configurations.

---

## Phases

### 1g.0 Structure-constraint screen: is a per-phone memoryless channel admissible on this stream at all?

**Purpose.** Every method in 1f and every method proposed for 1g — the channel likelihood, the
statistics matching, the graph matching, the moment estimator — assumes the SAME thing: that the units
are conditionally independent given the phone sequence, each unit generated by its own phone. That
assumption is a data-processing constraint, it is testable label-free from data already on disk, and
two independent measurements in the planning fan-out say it is violated on this stream by a large,
quantified factor. It has to be tested first, because it is upstream of every candidate 1g would fund.

**Approach.** Under the assumption, the chain unit(t) - phone(t) - phone(t+k) - unit(t+k) forces
`I(unit_t ; unit_t+k) <= I(phone_t ; phone_t+k)`. Measure both sides directly: the left on the
`seg12.5` stream, the right on the phonemized corpus T_phi, at lags 1, 2, 4, 8, with a small-sample
(Miller-Madow) correction. Then walk a ladder of model classes, computing the ceiling each one allows,
and report the first class — if any — whose ceiling contains the measured value: (a) one segment per
phone; (b) fertility with independent emissions, whose ceiling is the latent pair law
`0.30 x diag(phone prior) + 0.70 x phone bigram` at the measured within-phone adjacent-pair rate;
(c) two sub-states per phone with distinct emissions, the standing minimum-duration topology this
program already adopted for scorers; (d) emissions conditioned on the neighbouring phone. Decompose
the excess into its sources with the reads that already exist: cross-boundary-only pairs (gold
boundaries, eval-only), the long-lag and cross-utterance floors that price speaker and session, and
the silence-unit deletion control.

**Experiments.** One committed CPU job. Provenance, stated because this program has been bitten by it
before (`SAE_1a.md`'s uncommitted 0.146): the motivating numbers below come from two INDEPENDENT
scratchpad computations in the planning fan-out and are NOT citable until this job reproduces them —
they agree with each other to about one percent on both sides, which is why they are worth a job.
Motivating read: `I(unit_t ; unit_t+1)` = 2.436 and 2.467 nats on 919,248 segments, against a phone-
side lag-1 value of 0.558-0.565 nats; lag 2 1.204 against 0.170; lag 4 0.352 against 0.020; lag 8
0.107 against 0.0010. Class (b)'s ceiling rises only to about 0.800 nats, so roughly 1.6 nats per
segment pair — about two thirds of the observed dependency — is generated by something no per-phone
channel can produce. It is local rather than speaker-borne (lag 20 reads 0.211, within-utterance
shuffled 0.181, cross-utterance 0.148) and it is not silence (deleting the 60 most edge-enriched
units, 17.7 % of frame mass, moves lag 1 only to 2.345).

**RETRACTED IN FULL 2026-08-19 — implementer-found, planner-verified first-hand against
`channel_audit.py` source and four finished jobs. The paragraph that stood here claimed independent
confirmation from a committed job and was wrong three separate ways; nothing of it survives.** It read
`diag_frac` 0.385 as "38.5 percent of adjacent segment pairs carry the SAME unit identifier", inferred
0.385 x log(0.385 x 500) = 2.02 nats of adjacent-pair mutual information, concluded the measured excess
was segment REPETITION, and predicted class (c) would pass because it generates repetition natively.

1. **WRONG RUNG.** 0.385 is the RAW 50 Hz rung, in a section whose every other anchor is `seg12.5`.
Read from the finished jobs, `real/all.diag_frac` on dev-other by deduped-units-per-gold-phone:
raw 0.3847 at 2.824 units/phone (`ChannelStructureJob.G98AobA396ha`), `seg16` 0.2037 at 1.482
(`.BxyUz8Fha84d`), **`seg12.5` 0.1484 at 1.178** (`.Xf4J9E9gNiz4`), `seg9` 0.0962 at 0.862
(`.hisUE5DAz6EF`). The rung identification is not by hash-to-label but by each job's own
units-per-phone line, and 1.178 is the registered 1.18 fertility. `seg12.5` is lower by a factor of 2.6.

2. **WRONG QUANTITY, and this is the load-bearing error.** `channel_audit.projected_bigram` RUN-LENGTH
DEDUPLICATES each unit stream (`keep[1:] = u[1:] != u[:-1]`, line 125) and then projects the survivors
through the oracle map into a PHONE-by-PHONE table (`d = A[u[keep]]`, line 126). `diag_frac` is
trace/sum of that 39-by-39 table. So it is the share of adjacent DISTINCT-unit pairs whose two units
project to the same PHONE — within-phone flicker — and the function's own docstring says so. After
dedup, literal repetition is zero by construction, so the number cannot be measuring segment repetition
at all.

3. **THE ARITHMETIC IS WRONG TWICE.** It took the raw rung's value for a `seg12.5` claim, and
multiplied by an alphabet of 500 units for a quantity that lives on the 39-phone projected matrix.
There is no reading on which both factors are right.

**What this costs, stated plainly.** This was the only committed-job evidence in 1g.0, and it carried
the prediction that class (c) passes. That prediction now has no support from any artifact and is
WITHDRAWN — not refuted, since nothing has been measured about class (c), but unsupported. The
sentence "the screen's job is to confirm that prediction rather than to discover it" is deleted and
should never have been written: a screen's job is never to confirm. The screen is retained as an
instrument that measures both sides and reports the ratio whichever way it falls.

**What survives, and it is not nothing.** The data-processing inequality the screen rests on
(`I(z_t; z_t+k) <= I(y_t; y_t+k)` under a memoryless per-symbol channel) is a theorem and is untouched.
Two committed anchors for the ALLOWED side survive and are now cited correctly: the true phone bigram's
own pair mutual information, **0.5580 nats over 174,412 gold pairs**, banked in every audit job; and
`diag_frac` per rung above, which upper-bounds the post-dedup within-symbol pair rate. Note the
direction that follows and is worth registering before the job runs: the correct `seg12.5` rate is
**lower** than the 0.30 the class-(b) mixture assumed, so the class-(b) ceiling moves DOWN and the
screen gets TIGHTER, not looser. The retraction does not rescue class (b). It also widens the class-(c)
tension — a minimum duration of two forces a within-symbol pair rate of at least 0.5 against a measured
rate now nearer 0.15 than 0.30, a factor of about 3.4 rather than 1.7.

**Text-side and rung coverage under the 2026-08-18 ruling.** The allowed side is computed per TEXT
SIDE (rate-matched BPE primary, characters, phones as reference), because each has its own adjacent-
symbol dependency, and the measured side per AUDIO RUNG, because repetition rises with rate — `seg9`
at 7.04/s, `seg12.5` at 9.77/s where `diag_frac` is 0.385, `seg16` at 12.30/s, and any finer rung a
character text side would need at about 14.55/s. Registered prediction, so the screen is a test and
not a fishing trip: the violation gets WORSE at the finer rungs the lexicon-free rates want, which is
the concrete tension between rate-matching a character text side and satisfying the structure
constraint — and the resolution, if there is one, is the duration-modelling class rather than a rung
choice.

**Gate (pre-registered).** Report the ratio of measured to allowed dependency at lag 1 for each model
class in the ladder, for each (audio rung, text side) cell. A model class is ADMISSIBLE on this stream if its ratio is at or below 2. If
class (a) fails and class (c) or (d) passes, 1g.5 starts at that class instead of at rung 1, and the
simple form is dropped rather than run — this is the phase that decides which rung is rung 1. If NO
class in the ladder passes, the entire conditionally-independent-emission family is refuted on this
stream by measurement rather than by argument, and that closes 1g's channel branch, the parked ladder
entries 1 and 4, and the user's proposals 1 through 4 together, at the cost of one CPU job. A refuted
family licenses not funding it here — never "it could not work", and specifically not on any other
stream: the measurement is a property of THIS representation, so a re-segmented or context-dependent
stream (1g.6) would have to be re-screened rather than inheriting the verdict.

**Ruling on the MEASURED side, 2026-08-19: it is the DEDUPLICATED token stream, and there is no
re-dump.** The implementer established, and I verified at `repr_pool.py:302-303`, that
`SegmentPoolUnitsJob` banks only `units_r{rate}.pkl` plus a stats file — no segment boundaries — so two
adjacent segments carrying the same unit id are indistinguishable in the artifact from one longer
segment. The literal per-segment adjacent-pair statistic is therefore not reconstructible from anything
banked, and recovering it would move that job's hash and every hash below it, which is the entire 1f
bed. That cost alone would settle it, but there is a principled reason that would settle it anyway:
**a dependency that is invisible in the artifact is invisible to every estimator built on the artifact.**
1g.0 asks whether a per-symbol memoryless channel is admissible for the model class 1g would actually
fit, and that model consumes the same token stream every 1f matcher consumed. A repetition structure no
candidate could exploit and none could be blamed for missing does not belong on either side of the
ratio. So the measured side is defined on the deduplicated stream, the class-(b) within-symbol rate is
the post-dedup rate rather than a frame-level one, and the scratchpad figure of 2.436 nats on 919,248
segments is neither cited nor reproduced — it was never on a banked rung's pair counts and this plan
already barred it from citation.

**Ruling on whether 1g.0 keeps its closure clause, 2026-08-19: YES, with the prediction deleted, the
scope narrowed, and a robustness requirement added.** The retracted paragraph was evidence for a
prediction about WHICH class passes; it was never the evidence for the clause itself, which rests on
the data-processing inequality. A screen that measures both sides and reports a ratio is a valid
instrument whether or not anyone predicted the answer — what had to go was the framing, and it has.
Three amendments so a clause of this reach does not rest on one estimator: (1) the clause closes the
conditionally-independent-emission family ON THE DEDUPLICATED TOKEN STREAM, which is the representation
measured, and the standing not-on-any-other-stream restriction already in the gate covers the rest;
(2) the verdict must hold under BOTH the Miller-Madow-corrected and the uncorrected mutual-information
estimates — plug-in mutual information is upward-biased, which inflates the measured side and biases
toward closure, so an uncorrected estimate alone must never close four proposals; (3) the shuffled and
cross-utterance floors are reported in the same table as the ratio. If the two estimators disagree
about admissibility for any class, the screen returns INDETERMINATE for that class and closes nothing.

**Ruling on the ceilings for classes (c) and (d), 2026-08-19 — the implementer declined to invent
these and was right to. Classes (a) and (b) are unchanged and already pinned by the spec.**

**(c) sub-states per symbol: the duration constant is a SWEPT AXIS, never a pinned number.** The
standing minimum-duration-2 rule is a USER ruling about SCORER topology (2026-08-15); it is not a fact
about this stream and it may not be imported here as a given. Importing it would bake in exactly the
kind of untraceable design constant this phase exists to avoid, and worse, the stream contradicts it:
a minimum duration of 2 forces at least two segments per symbol, hence a within-symbol adjacent-pair
rate of at least 0.5, against the roughly 0.30 measured. A class that over-predicts repetition would
then pass the screen BECAUSE it over-predicts, and the pass would say nothing about whether sub-states
are the right structure. So (c) is reported as a CURVE: the allowed dependency as a function of the
within-symbol adjacent-pair rate, swept across a range spanning the measured value and 0.5. Two numbers
come off it. The gate reads the ratio AT THE MEASURED RATE — that is the honest per-class ratio and it
keeps the pre-registered gate readable exactly as written. Reported alongside, and more informative
because it needs no constant at all: **the rate at which the ratio first falls below 2**, i.e. how much
duration structure a model would have to posit to explain the dependency this stream actually shows.
The misfit is itself a reportable finding — the standing scorer topology does not fit this stream's
duration statistics, and that is worth knowing independently of 1g.

**(d) neighbour-conditioned emissions: no ratio, because it has no binding ceiling.** Conditioning
emissions on the neighbouring symbol drops the conditional-independence assumption outright, so the
class can account for arbitrary adjacent-pair dependency and a ratio against it is vacuous. Do not
build one. (d) gets a PARAMETER-BUDGET read instead, against the enlargement ladder registered in
1g.5: the factored left-context form at +19,500 parameters (36.9 segments per parameter at 20.5 h) is
affordable, the full diphone table at 758,979 (0.9 segments per parameter) is not. **This does not
weaken the pre-registered gate and is a reading of it, not an edit:** that gate refutes the
conditionally-independent-emission family, and (d) sits outside that family by construction, so it was
never among the classes whose failure would close the branch. If (a), (b) and (c)-at-the-measured-rate
all fail while (d) is affordable, the correct verdict is that the family is refuted AND the fundable
successor is the factored context form — not that nothing passed.

**Status.** REGISTERED 2026-08-18, pre-run. This is now the first fundable step of 1g, ahead of 1g.1;
classes (c) and (d) ruled 2026-08-19 as above.

**Status 2026-08-19 — RUN. THE CLOSURE CLAUSE DOES NOT FIRE, AND THE GATE DECIDES RUNG 1.** Dev-other
2,864 utterances, dev-clean 2,703, measured on the deduplicated token stream, both estimators, labels
eval-only.

| cell | class (a) | class (b) | class (c) at measured rate | rate clearing 2 |
|---|---|---|---|---|
| raw / phones | 5.76 REFUTED | **1.69 ADMISSIBLE** | 1.30 ADMISSIBLE | 0.40 |
| raw / chars | 5.62 REFUTED | 1.96 ADMISSIBLE (dev-clean 2.14 REFUTED) | 1.47 ADMISSIBLE | 0.50 |
| `seg16` / phones | 4.58 REFUTED | 2.42 REFUTED | 1.50 ADMISSIBLE | 0.30 |
| `seg12.5` / phones | 4.15 REFUTED | 2.81 REFUTED | 1.64 ADMISSIBLE | 0.25 |
| `seg9` / phones | 3.43 REFUTED | 3.01 REFUTED | 1.72 ADMISSIBLE | 0.15 |

Measured side, per rung: lag-1 dependency 3.2964 / 2.6171 / 2.3730 / 1.9596 nats and within-symbol pair
rate 0.7093 / 0.4226 / 0.3164 / 0.2037 for raw / `seg16` / `seg12.5` / `seg9`.

**VERDICT: the conditionally-independent-emission family is NOT refuted.** Class (c) is admissible on
all eight dev-other cells and seven of eight dev-clean cells, the eighth INDETERMINATE rather than
refuted. Nothing closes — not 1g's channel branch, not the parked ladder entries 1 and 4, not the
user's proposals 1 through 4. Class (a) is refuted on every cell of both splits with both estimators
agreeing everywhere. **Per the pre-registered gate, which said in advance that this is the phase that
decides which rung is rung 1: 1g.5 STARTS AT THE TWO-SUB-STATE CLASS, and the strict
one-segment-per-symbol form is DROPPED RATHER THAN RUN.**

**Independent cross-check I ran before registering any of this.** The measured within-symbol pair rate
is the complement of the gold cross-boundary share, and that share was banked weeks ago by a different
job with different code. Audit `real` cross-boundary share against this screen's implied value:
`seg12.5` 0.6812 against 0.6836 (0.0024 apart), `seg16` 0.5907 against 0.5774, `seg9` 0.7765 against
0.7963, raw 0.3409 against 0.2907. Closest agreement is on `seg12.5`, the rung every anchor in this
plan is stated on; the residual is largest on raw, which is what the dedup-across-boundary attribution
rule predicts, since raw has by far the most runs. The class-(a) ratios also reproduce from the banked
phone-bigram pair information of 0.5580 nats to within about 2.5 percent uniformly, in the direction a
downward bias correction on the measured side produces. Two sides of this screen were computed by two
independent jobs and they agree.

**The minimum-duration tension resolves the OPPOSITE way from the worry that motivated the ruling, and
the swept axis is what made the pass readable.** The ratio clears 2 at a within-symbol pair rate of
0.15-0.40 across cells, against the 0.5 that a minimum duration of two forces. So the duration
structure a model must posit to explain this stream is LESS than the standing rule implies: minimum
duration two is SUFFICIENT everywhere and NECESSARY nowhere. Had the constant been pinned at 0.5, class
(c) would have passed automatically and the pass would have carried no information; because the axis
was swept, the pass is at rates well below the one that would have made it free. That is the ruling
earning its keep, and it is the reason to keep refusing pinned constants in screens.

**On the withdrawn prediction, and this is not a reinstatement.** The retracted paragraph predicted
class (c) would pass, and class (c) passed. The three errors in that paragraph are still errors, the
withdrawal stands, and the reason for it — a screen's job is never to confirm — is untouched. What has
happened is that the ANSWER is now carried by a measurement independent of the retracted arithmetic;
the retraction removed the reason, not the answer. Registered in those terms deliberately: a prediction
that lands by coincidence after its stated grounds are refuted is not evidence that the grounds were
sound, and the temptation to read it as vindication is exactly what the logging rules exist to block.

**CORRECTION to a direction I registered before this run (implementer-found, my error).** I registered
that `diag_frac` upper-bounds the post-dedup within-symbol pair rate, so the class-(b) ceiling would
move DOWN and the screen would get TIGHTER. That is wrong. `seg12.5` measures a rate of 0.3164 against
its `diag_frac` of 0.1484, so the ceiling moves UP and the screen is LOOSER there. The two count
different things and neither bounds the other: `diag_frac` counts pairs PROJECTING to the same symbol
under the oracle map, the rate counts pairs lying INSIDE one gold symbol run, and the gap is the flicker
where the oracle map disagrees with itself across one symbol's own frames — only about 47 percent of
within-symbol pairs stay on the diagonal. This changes no verdict: class (b) is still refuted on every
pooled rung, and at the true rate the class-(b) allowed value comes out near 0.845 nats against the
0.800 this plan had assumed, which is the same number to within the difference in rate.

**Class (b) survives on raw, and that is coherent with the promotion of 1g.5 rung 2.** Fertility with
independent emissions is admissible on the raw stream on phones on both splits (1.69) and marginal on
characters (1.96 dev-other, 2.14 dev-clean, i.e. split-dependent at the bar). The rung 1g.5 expects to
win is the rung on which the weaker model class is already admissible.

**The two-estimator control fired once in sixteen cells, exactly where it was predicted to matter.**
`seg9`/characters on dev-clean returns INDETERMINATE at a corrected ratio of 1.99 where the uncorrected
estimate alone would have refuted it. A control that fires once, at the bar, on a cell of the class that
decides the phase, is working rather than wasteful.

**Ruling on the floors: add a floor-corrected SENSITIVITY column, do not touch the gate.** The floors
rise as the measured dependency falls — cross-utterance floor 0.2423 / 0.4005 / 0.4995 / 0.6444 against
lag-1 of 3.2964 / 2.6171 / 2.3730 / 1.9596 — so the non-local share runs 7.4 percent on raw to 32.9
percent on `seg9`. A third of `seg9`'s measured dependency sits at a floor that cannot be symbol
structure. The implementer was right not to subtract it: the gate reads the raw ratio as registered and
that stands. But the sensitivity read is worth one column because it tests the ROBUSTNESS of the class
(b) refutation, which is the only verdict close enough to the bar to move: scaling each measured value
by one minus its non-local share gives class (b) ratios near 2.05 / 2.22 / 2.02 for `seg16` /
`seg12.5` / `seg9`, all still above 2. If that reproduces, the class-(b) refutation on pooled rungs is
robust to the most aggressive floor correction available, and should be reported as such. Note the
column is a SENSITIVITY read and not a corrected measurement — mutual information does not decompose
additively, so it bounds the order of magnitude of the concern and nothing finer. Independently of the
gate, the floor profile is a fact about the rungs worth carrying into 1g.1's re-opened rung selection:
coarse pooling buys rate at the cost of a dependency that is increasingly not about symbols, and raw
has both the cleanest floor ratio and the best insertion-forgiven error.

### 1g.1 Frame: what an initialization must reach, and what a language model at decode time is worth

**Purpose.** Fix the destination before spending on candidates. Nothing in 1f ever measured (a) what
the phone language model buys at decode time over the per-segment argmax every 1f entry used, (b) how
good a channel must be for the downstream pipeline to care, or (c) whether any already-banked 1f map
crosses its own gate once decoded properly — so every 1f verdict to date prices a decoder, not a map.

**Approach.** One CPU battery on banked artifacts. Build a ladder of channels on `seg12.5`: the
gold-fitted oracle channel; that channel mixed toward the unit marginal at mixing weights giving a
spread of per-segment error rates (the unbiased degradation); a structured degradation that merges
confusable phones within manner classes (the realistic degradation, since a learned channel's errors
are systematic, not symmetric); the entry-5 statistics-matching candidate's own segment posteriors;
the entry-3 fingerprint map; and both registered nulls. Decode every one of them twice — per-segment
argmax as the screens do, and beam search under the banked KenLM 4-gram phone language model
(`CreateBinaryLMJob.hvZoC014xnIe`) — and score both by the screens' protocol on the same scored fifth.

The decoder has two free knobs — language-model scale and an insertion/deletion penalty — and this
project forbids tuning them on labels, so every rung is reported TWICE: at the label-tuned optimum
(the upper bound of the whole proposal) and at the operating point the LABEL-FREE selector picks (the
realistic one). The gap between them is a first-class result: a large gap means the operating point
cannot be chosen without labels, which is disqualifying here on its own. The label-free selector is
the section-1.0 unsupervised metric — the phone-language-model perplexity of the decoded hypotheses
weighted by vocabulary coverage, calibrated at rank correlation 0.89 against gold PER in `SAE_1a.md`
approach 2 — and deliberately NOT the model likelihood, which is the quantity section 1a measured as
anti-aligned and which 1g is trying to test rather than to rely on. Two recorded limits on that
selector, both from the logs: it was calibrated on dev-clean over a coarse model spread, and entry 5's
verdict registered it for picking checkpoints and seeds within an arm and explicitly not for ranking
methods — 1g uses it only for knobs and restarts inside an arm.

**Experiments.** (E1) the ladder above, both decoders, both knob-selection modes, dev-clean and
dev-other. (E2) the WORD-LEVEL decode, promoted out of a late rung by the 2026-08-18 text-side ruling and run on
the two best rungs from the start: a word n-gram language model over the raw unpaired corpus composed
with a SPELLING lexicon read off that same corpus — no pronunciation dictionary anywhere — reported as
plain WER, the deliverables ladder's currency and, per the verified precedent, the component that
actually carried the published noisy-channel speech result. The phone reference arm's equivalent uses
the section-1d pronunciation lexicon and is reported beside it, the gap being the measured price of
that lexicon. (E3) the
ceiling decomposition idea 5 needs: re-pool the same frames within **gold phone boundaries**
(eval-only), relabel by the same codebook, read the oracle-map PER — the split of the 0.414 ceiling
into boundary-placement error and unit confusability. (E4) the silence question, because the published
precedent anchors on silence and our conventions delete it: repeat the best rung with a silence symbol
restored on both sides (the proxy-silence segments currently cut, and a silence/word-boundary symbol
in the text) while scoring in the unchanged silence-free currency, so the model may use the anchor
without the gate changing.

**E5 leg-B seed ruling, 2026-08-19, registered when the build found the spec assumed an artifact that
does not exist.** Neither banked seed's map was ever persisted — both were computed in memory, scored
and discarded. The two seeds are not the same kind of object and are handled differently.

*Fingerprint solve (0.8809).* RECOMPUTE-AND-ASSERT approved: the solve is deterministic at fixed
regularization, so identical arguments give an identical map, and the job asserts the recomputed map
reproduces its banked reads before the estimator may touch it. The assertion covers the FULL
decomposition rather than the scalar — phone error rate, its substitution/insertion/deletion split, the
symbol count and the silence-unit count — because five simultaneous constraints pin a map far tighter
than one. Registered limit: this is a reproduction argument and not an identity proof, so the claim it
supports is "a map of this method at this quality is inside the basin", not "the banked map is". If the
assertion fails, amending the producing job to emit its map becomes live rather than the arm being
dropped.

*Statistics-matching arm (0.8580).* NOT recomputed and NOT asserted against 0.8580. Two facts settle
it, both read from disk. Its trained checkpoint EXISTS — the picked arm is `entry5_seg12.5_full_s1` per
`EspumPickJob.W9HzeOEviPO4`, checkpoint at `EspumMatchTrainJob.lALR9ldNG8f1/output/model.pt` — so this
is an artifact read, strictly better than a reproduction, and retraining a network for 40,000 updates
would be a new run rather than a reproduction in any case. But the arm is a KERNEL-4 CONVOLUTION over a
window of one-hot units, not a per-unit map, so its 0.8580 is the error of a context-dependent decode
and no symbol-by-unit table exists to recover. Any table derived from it is a projection that discards
the context the model was using, and an assertion against 0.8580 would fail by construction because the
projected object is not the scored object. Registered projection rule: the average posterior per unit
over all its occurrences, the marginal projection, with the projected seed's quality MEASURED fresh
rather than inherited. If it lands far from 0.8580 that is a finding about how much of that arm's
performance was context, not a defect.

*Positioning, which dissolves the problem for both.* Each leg-B seed is placed on the leg-A curve by
its OWN measured error at iteration 0 under the estimator's decoder — a quantity leg A already
reports — and not by its banked headline number. The banked figures then serve as provenance labels
rather than load-bearing quantities, both axes are measured through one decoder, and any gap between a
seed's headline and its measured iteration-0 error is itself reportable: it prices the decoder change
for the fingerprint seed and the projection for the statistics-matching one.

**E5 leg-A endpoint sanity check, pre-registered 2026-08-19 before the curve exists, and SHARPENED the
same day still before any result exists.** Read the curve's ENDS before interpreting its middle. At
corruption 0 the seed is the gold-fitted map; at corruption 1.0 it is content-free and the
absorbing-fixed-point argument puts it near the banked nulls of 0.8946 and 0.9239.

The sharpening applies to this check the same separation the planner's failed descriptor point estimate
taught, because the first form of it conflated the two. **The HARD STOP is the premise-free bound: the
corruption-0 read must beat 0.4148.** An oracle channel decoded through a language model and a duration
model cannot be worse than the same channel decoded by per-segment argmax unless the decoder is
subtracting, so failing that is a defect and the middle of the curve may not be read until it is
explained. **The 0.28-0.33 band is a MODELLED expectation, not a bound** — it is extrapolated from the
theory battery's simulation, which used a fertility-1 well-specified model and an ideal decoder, where
this runs the two-sub-state class under a beam. So a corruption-0 read that beats 0.4148 but lands
above 0.33 indicts the EXPECTATION rather than the estimator, is reported as such, and does NOT stop
leg B. Three candidate causes are on the record in advance for any miss — a decoder or estimator
defect, an optimistic extrapolation, or loss in projecting the gold-fitted map into the two-sub-state
topology — and the direction of the miss discriminates: worse than 0.4148 is near-certainly the first.

**Free byproduct worth reporting explicitly rather than burying in a curve.** Leg A's corruption-0
cell, read at iteration 0, IS the quantity 1g.1's main gate is written on — the oracle channel decoded
through the language model against the 0.4148 bar, the one clause registered as able to close 1g on its
own. Report it named and against that bar. It DISCHARGES that gate only if the decode knobs were fixed
the registered way, by the label-free section-1.0 selector; otherwise it is a first read of the gate
quantity at leg A's own knobs and the gate stays open pending the knob sweep.

**E5 spec additions 2026-08-19, registered before the code lands rather than after.** (1) Leg A's
deliverable is a CURVE and a curve without spread cannot locate a threshold, so each corruption level
runs several independent draws and the fixed point is reported with its spread across them, not as a
single number. (2) The decode-side stopping metric is used to pick a STOPPING ITERATION within each
run, which is within-arm use and is licensed; comparison ACROSS corruption levels is by PER with labels
as a measurement instrument, which E5 is. The registered limit that this metric may not rank METHODS is
untouched and E5 does not rank methods. (3) The registered restart budget belongs to 1g.5's search and
does NOT transfer here — E5 starts from given seeds and is not searching — so restarts are draws for
spread, not a budget to exhaust.

**(E5) SEED-TO-FIXED-POINT PRICING — added 2026-08-18 on theory-battery finding 6, and this is the
experiment that sets the bar every other 1g candidate must clear.** Two independent screens converged
on it: the identifiability study found the basin, not identifiability or sample size, to be the binding
constraint, and the candidate-generation screen independently named the amplifier loop "the cheapest
experiment that prices the entire list". Build seeds by corrupting the gold-fitted map to a controlled
per-unit correctness fraction (measurement instrument, same quarantine as the rest of the ladder — a
corrupted oracle may never initialize anything reported as ours), then run the pinned-language-model
estimator from each seed and plot the fixed point against the seed's own quality. Read on the same
axes as E1, in this project's own post-dedup PER, which is the calibration the simulation could not
supply. Then the read nobody has taken: **run the same estimator from the two BANKED 1f seeds** (the
0.8580 statistics-matching map and the 0.8809 fingerprint map) with early stopping selected by the
label-free section-1.0 decode-side metric — never by model likelihood (finding 5) — and report where
they land. Both nulls get the identical treatment, so a null that also repairs is the finding.

**Second gate clause, pre-registered with E5 (it re-prices the whole phase and can raise or lower every
later bar).** The deliverable is the seed-quality-to-fixed-point curve with the repair threshold named:
the seed quality at which the estimator's fixed point first comes in below the 0.4148 memoryless
ceiling. If the curve shows a repair threshold, **every downstream candidate is measured against that
threshold and not against the null margin**, and the 1f reading that a 0.858 seed is a failure is
replaced by a measured statement about whether it is inside the basin. If the curve shows no repair at
any seed quality short of the oracle — that is, the estimator only preserves what it is handed — then
the amplifier contributes nothing on this stream, 1g's channel branch is an initialization problem
alone, and the standard against which 1g.4, 1g.7 and 1g.8 are judged reverts to the registered 1f arm
gate. REGISTERED PREDICTION, written before the battery runs and traceable to finding 6: a repair
threshold exists near 20 % of units correct, and the two banked seeds are inside the basin. If that
prediction is right it is the largest single change to this program's operating assumptions, and if it
is wrong the phase costs one CPU battery.

**Gate (pre-registered, one clause, and it can close 1g on its own).** With the LM removed, maximum-a-
posteriori decoding under a memoryless channel collapses exactly to the per-unit rule already measured
at 0.4148 dev-other, so 0.4148 is what the channel alone buys and everything 1g bets on is the
language model plus the alignment model. Therefore: the ORACLE channel, decoded through the language
model at the knob setting the LABEL-FREE section-1.0 selector picks, must come in below 0.4148
dev-other. If a
perfect channel plus the language model plus the alignment model cannot beat the plain lookup under
label-free knob selection, the probabilistic apparatus is subtracting rather than adding on this
stream — misspecification has already eaten the language model's contribution — and 1g.5 is not
funded, no matter what the simulation says. Beyond that clause this phase registers numbers rather
than passing or failing. Its deliverable is the tolerance curve
(channel quality on the x-axis, always as the per-segment-argmax dev-other PER of that channel, which
is the currency every 1f verdict is already written in; output quality on the y-axis, as
language-model-decoded PER and, on the two best rungs, word error rate) with two reference lines drawn
on it: REPLACEMENT QUALITY = the GAN init this program is trying to replace (0.168 PER, 17.96/21.87
WER) and ABOVE-NOTHING = the banked content-free nulls. Named as NOT measured, and deliberately not
funded here: the SAE loop's own tolerance to a bad init — no experiment in this program has ever
started the loop from a controlled-quality init, so "what PER does the loop need" cannot be answered
from existing artifacts; the priced option is an SFT plus two sub-epochs from two rungs of the
degradation ladder (GPU). That option is NOT quarantine-clean — the degradation ladder is built from
the gold-fitted oracle channel, so anything it trains is a diagnostic arm of the section-2S kind whose
artifacts may never enter the unsupervised ladder, and running it at all is a USER decision, not a
planner one. Discipline that
binds every later read: candidate and both nulls are decoded by the identical decoder, and the arm
gate's margins are read after the decode change, never across it.

**Registered expectation for that gate, added 2026-08-18 (a prediction beside the gate, not an edit to
it).** Theory-battery finding 7 puts an ideal noisy-channel decoder at 21-34 % relative below the
memoryless error on a channel calibrated to this stream, which lands the oracle-plus-language-model
read at **0.28-0.33** dev-other against the 0.4148 bar. So the gate is expected to pass with room, and
the informative outcome is the size of the margin, not the sign. The same finding registers what the
margin will NOT be: an order of magnitude. Anything that looks like one is a bug, and the standing
reason is that English phone redundancy is real (1.15 nats/phone) but is not error-correcting distance.

**RUNG SELECTION IS RE-OPENED, and the expectation is registered before the battery runs
(2026-08-18).** 1f chose `seg12.5` on memoryless phone error rate, a selector that prices insertions
and deletions symmetrically. 1g decodes through a language model and a duration model, which repair
insertions far more cheaply than deletions, so the right selector is the insertion-forgiven error
(substitutions plus deletions). That statistic is pure arithmetic off `SAE_1f.md` approach 3 and it
REVERSES the ranking exactly:

| rung | memoryless PER | insertions | deletions | substitutions | sub + del |
|---|---|---|---|---|---|
| `raw` (deduped) | 0.832 | 0.692 | 0.008 | 0.132 | **0.140** |
| `seg16` | 0.452 | 0.178 | 0.054 | 0.220 | 0.274 |
| `seg12.5` | **0.414** | 0.067 | 0.117 | 0.230 | 0.347 |
| `seg9` | 0.481 | 0.014 | 0.290 | 0.177 | 0.467 |

REGISTERED LIMIT ON THIS STATISTIC, added 2026-08-18 after it was misapplied once: insertion-forgiven
error ranks REPRESENTATIONS at a matched output policy, where the length ratio is the property being
priced. It does NOT rank ARMS that over-generate by different factors — there it is exactly the
statistic an over-generating arm games, and the entry-7 stage-A read is the worked example (hypotheses
at 2.07x and 1.55x reference length, where recall favours one arm and precision the other and neither
ordering is a fact about the treatment). The headline anywhere in 1g stays the plain PER as scored.

The stream 1f selected as best is third of four under 1g's own decoder, and the raw stream — dropped
in 1f because its insertions are catastrophic for a lookup decode — is first by a wide margin. The
supervised probe's own 9.7-to-1 deletion-to-insertion asymmetry (0.2056 against 0.0213) points the
same way. REGISTERED PREDICTION, testable on banked artifacts with no new stream: under the
language-model decode the coarser rungs beat `seg12.5`, and `raw` with a duration model beats them
all. E1 therefore runs the rung axis, nulls are re-banked per rung, and if the rung ordering flips
between the two decoders that flip is itself the reportable result.

**Gate-hygiene clause, registered before the battery runs.** E1 re-decodes the banked entry-3 and
entry-5 artifacts, which already have closed gate verdicts read under the screens' per-segment argmax.
Those verdicts are NOT reopened and NOT edited: they stand against the protocol they were registered
under. If a banked artifact clears the margins under the 1g decoder with both nulls re-decoded
identically, that is a NEW 1g reading of an old artifact — reported as a 1g result with its decoder
named — and it is a fact about the decoder, not a reversal of the 1f verdict. Stating this before the
numbers exist is the point: the first read of 1g must not become a retroactive rescue of 1f.

**Status.** REGISTERED 2026-08-18, pre-run.

### 1g.2 Objective-alignment audit at the known-best map

**Purpose.** Ask of every candidate objective the question 1f never asked of any of them: does the
objective rank the truth first? Entry 5 optimized its objective to convergence and landed 0.44 above
the memoryless ceiling with 80 % of its error as substitutions, which is the signature of an
objective whose optimum is not at the truth — but that was inferred, not measured, and the same
measurement decides the user's proposals 1, 2 and 3 for CPU-hours instead of GPU-days.

**Approach.** Evaluate each objective at each point of the 1g.1 channel ladder (oracle, both
degradation ladders, entry-5 solution, entry-3 solution, many random and marginal-matched maps),
holding the stream, the text sample and the scored rows fixed. Objectives audited: (a) the pinned-
language-model channel log-likelihood at bigram and 4-gram order (the user's proposal 1); (b) the
entry-5 statistics-matching loss — positional unigram plus bi- and tri-skipgram L1, evaluated with
`espum_model.py` at a fixed map (the user's proposals 2 and 3); (c) the off-diagonal total-variation
transition objective already measured inverted by coarticulation (`SAE_1f.md` conclusion 9), included
so the audit's own calibration is visible. Two objectives ADDED 2026-08-18 after the candidate screens,
both of them changes of objective rather than of optimizer, and both evaluated at a fixed map so they
cost a score and not a training run:

(d) **TOLERANT WORD-PARSE COST of the decoded string.** Decode to text symbols, then take the cheapest
segmentation of that string into words, charging one unit per word plus one per substitution used, at
most one substitution per word — a single left-to-right dynamic program over a word trie. On the
lexicon-free primary arm the trie is the **spelling lexicon read off the raw unpaired corpus**, which
costs nothing in the ledger; the phone reference arm uses the section-1d pronunciation lexicon and the
gap prices it. The screen's own first-hand sweep on this project's text is the reason it is worth a
column: as a map is corrupted from correct to fully corrupted the cost moves 41.9 % on a stream already
carrying 30 % substitutions, against 12.3 % for a phone 4-gram control on the identical sweep and the
9-11 % relative separation entry 5's own objective managed; it is monotone at every step and the truth
was a strict local minimum in 30 of 30 single transpositions. The mechanism is that a word list is a
hard 0/1 phonotactic filter, not a reweighting — exact parseability was 100 % of sentences under the
true map against 1.83 % under a random symbol permutation. Two honest limits carried into the gate:
the best-neighbour margin is thin (+0.20 %), so this ranks coarsely-different hypotheses and not
near-ties; and the naive counting argument for the constraint is VACUOUS on this project's own lexicon
(concatenation growth rate 5.535 bits/phone against log2(39) = 5.285), so its strength is empirical and
phonotactic and must be argued from measured columns only.

(e) **EXACT FULL-BATCH COVERAGE-SEEKING OUTPUT-DISTRIBUTION MATCHING** — the Liu-Chen-Deng NeurIPS 2017
cost with the language model OUTSIDE the logarithm and the model's expected n-gram frequency inside.
It is included because it dissolves that family's own discard reason rather than arguing with it: with
a per-unit table the objective is a closed-form contraction of a precomputed sparse unit n-gram tensor
over the whole corpus, so the batch IS the corpus, the biased-minibatch problem disappears, and each
exact gradient is sub-second at 19,500 parameters. The attribution that makes it worth a column is
checkable rather than rhetorical: entry 5's measured signature — all 39 symbol types used, 0.963-0.974
of the reference symbol count, about 80 % substitutions, 0.8580 — matches the published small-batch
bias rows (83.09 / 78.05 / 67.14 / 56.48 % error at batch 1e1/1e2/1e3/1e4 against 9.21-9.59 % unbiased)
rather than a collapse, and entry 5 ran 40,000 minibatch updates on batch count statistics. Note the
direction of the fix: entry 5 used a symmetric bounded L1 between batch statistics, which imposes no
penalty for putting near-zero mass on an n-gram the text calls common; this cross-entropy diverges
exactly there. Registered as an AUDIT column only — it is the family that failed the audit's own gate
in its L1 form, so it earns a real-stream fit by passing (i), (ii) and (iii) and in no other way.

**Experiments.** (E1) STATIC RANKING: one job per objective, three reads each — the oracle's rank
against every null and random map, the rank correlation between objective value and PER along both
degradation ladders, and the sign of (objective at the learned solution minus objective at the
oracle), each term reported SEPARATELY with its across-batch standard error and against its sampling-
noise floor, because the fan-out's accounting says entry 5's tri-skipgram term may have been running
with more sampling noise than the bigram term's largest attainable signal. Three add-ons on the same
job, each answering a question no existing artifact answers: the per-term loss as a function of batch
size (640 / 2,560 / all 8,416 utterances), which says whether this objective family is well-posed
below 960 h at all; the smallest singular value of BOTH design matrices — the audio-side positional
matrix over 500 units and the text-side one over 39 phones — so the identifiability direction is a
measured column rather than an argument; and the supervised ceiling of the matrix class itself, fitted
by gold through the identical pooling path, currently unknown and bracketed only between 0.3565 and
0.4148. (E2) ORACLE-START DRIFT, the decisive one and the exact protocol the 2026-07-14 verifier ran
on the continuous model in `SAE_1a.md`: start each objective's optimizer AT the oracle channel on the
real `seg12.5` stream and run it to convergence, logging objective value and PER every iteration. For
objective (a) this is Baum-Welch from an oracle emission matrix (seconds per iteration at 39 states
over 720k segments); for objective (b) it is the entry-5 optimizer from an oracle-initialized
generator for a few hundred updates. Rising objective with degrading PER is a misspecified objective
measured rather than inferred, and it is the finding that closed the continuous lane.

**(E3) SELECTION-STATISTIC VALIDATION ACROSS BANKED ARMS — added 2026-08-18, and it addresses this
program's own measured failure point rather than a new one.** Selection is where 1f broke twice: entry
5's label-free selector picked a 0.8580 arm against a 0.8446 bar, and section 1a closed because
decipherment likelihood was anti-aligned with error. The incumbent decode-side metric is registered as
valid for choosing checkpoints and seeds WITHIN an arm and explicitly not for ranking methods, and that
is exactly the gap. Score the arms whose plain dev-other PER is already banked — 0.3565 supervised
probe, 0.4148 gold-fitted map, 0.8580, 0.8809, 0.8946 and 0.9239 nulls — with objective (d), with a
text n-gram control on the identical arms, and with the incumbent metric, and require two things
registered in advance: a strongly negative rank correlation against PER across those arms, and STRICT
separation of both content-free nulls from every arm that is not content-free. One artifact gap must be
budgeted rather than discovered: no banked arm dumps hypothesis strings — the eval jobs carry only
PER/substitution/insertion/deletion summaries — so the per-unit-map arms regenerate in minutes from
map plus unit stream, and the one trained arm needs a re-decode from its existing checkpoint. If the
n-gram control orders the arms as well, the control is adopted instead, because it is simpler. If
neither orders them, **1g has no method-ranking selector at all**, and that is a finding that binds
every later phase: no arm may then be chosen label-free across methods, and the phase reports that
constraint rather than working around it.

**Gate (pre-registered, before any audit number exists).** An objective is FUNDABLE for a real-stream
fit only if, on dev-other, all three hold: (i) the oracle channel strictly beats every null and every
random map on the objective; (ii) the rank correlation between objective and PER along the structured
degradation ladder is at least 0.7 with the correct sign; (iii) under E2 the objective value rises (so
the optimizer is working) while PER degrades by at most 0.05 absolute from the oracle's own PER. That
0.05 is traceable, not chosen for convenience: it is the arm gate's own margin unit, and it sits
between the 0.01-0.03 degradation the published literature reports as NORMAL for maximum-likelihood
training started from a good model (Merialdo 1994 part-of-speech tagging, 97.0 to 95.2 over ten
iterations; Berg-Kirkpatrick and Klein 2013 cipher accuracy, 92 to 89 at higher likelihood) and the
0.117 collapse section 1a measured on the continuous model (0.275 to 0.392). A gate at 0.05 therefore
passes the published-normal drift and catches the 1a-scale failure. Failing only (i)/(ii)-consistent search reads — that is, all three pass
but the entry-5 or entry-3 learned solution scores WORSE on the objective than the oracle does — means
the family is search-limited and the fundable work is initialization (1g.3, 1g.4). Failing (i), (ii)
or (iii) means the objective is misspecified on this stream, and no optimizer, capacity change or
regularizer is funded for it. A failed audit licenses not funding that objective — never "it could not
have worked". E2 additionally yields, for free, the ceiling every later phase is bounded by: the
converged oracle-start solution is the best basin any initializer could reach.

**Status.** REGISTERED 2026-08-18, pre-run.

### 1g.3 Simulation at the measured operating point

**Purpose.** The 1a simulated ladder is the best case for the whole channel-estimation family — the
model is exactly correct by construction — and it was only ever run at 5-15 % emission noise, while
`SAE_1f.md` conclusion 11 places the real pooled stream *worse than* 35 % random emissions at matched
fertility. If the family already fails in simulation at the real stream's calibrated operating point,
it fails on real data a fortiori and 1g closes cheaply.

**Approach.** Re-run the committed `scripts/unsupervised_asr/decipher.py` ladder at the real stream's
measured constants — 39 phones, 500 units, fertility matched to 1.18 units per phone, corpus size
720k segments — sweeping emission noise across and past the calibrated point, and sweeping the
initializer: random, frequency-CDF, entropic Gromov-Wasserstein (the user's proposal 4), the
acoustic-phonetic anchor of 1g.4, and truth. Report recovery and decoded PER per cell, under both the
per-segment argmax and the language-model decode of 1g.1.

**Experiments.** The noise x initializer grid, three seeds per cell; CPU-minutes to CPU-hours. The
code is committed and was validated in 1a; the work is the operating-point re-parameterization and a
sisyphus CPU job wrapper.

**Gate (pre-registered).** Read in the simulation's OWN currency, recovery — the share of unit mass
mapped to the true phone — because a simulated PER and a real dev-other PER are different arms and
only the within-arm number is comparable: the family proceeds to a real-stream fit only if, at or
beyond the real stream's calibrated noise level, at least one LABEL-FREE initializer reaches recovery
0.5, against the 0.97 the same ladder reaches from a truth start and the roughly 0 it reaches from
random. Decoded PER is reported per cell as context, never as the gate. Gromov-Wasserstein enters 1g
only as an initializer inside this grid and inside 1g.5; it is not funded as a standalone method at
any point. A failure here closes the channel-estimation family on an a-fortiori argument — the model
is correct by construction in simulation — and licenses not funding it, never "it could not have
worked".

**Status.** REGISTERED 2026-08-18, pre-run.

### 1g.4 Acoustic-phonetic class anchoring (new candidate; the initializer 1a lacked)

**Purpose.** `SAE_1a.md` says the discrete objective is init-limited and `SAE_1f.md` conclusion 16
says why every purely distributional initializer failed — under the fingerprint cost the true phone
is in the top five for only 0.23-0.32 of unit mass against a chance 0.128, about twice chance, and
manner separation reads 0.477 against a 0.403 majority baseline. Acoustic-phonetic measurements of
the units are a different information source entirely, and voicing and manner are close to trivially
measurable from the waveform, which is exactly the symmetry-breaking the family is missing.

**Approach.** Per unit, aggregate deterministic signal statistics over the frames it occupies —
periodicity/harmonicity, short-time energy, spectral centroid and tilt, zero-crossing rate, mean
segment duration, silence adjacency (numpy/scipy only; the shared environment's librosa numba cache
is corrupt). Map units to a small set of broad phonetic classes by pre-registered textbook rules, and
match those classes to the ARPAbet inventory's own class membership, yielding a block constraint on
the channel matrix rather than a full map. The output is an initializer and a hard constraint for
1g.5, never a standalone init.

**Experiments.** The descriptor dump plus the class assignment; then the kill-test read below. Two
text-side variants are priced separately: classes from a phone-to-articulatory-feature table (the
disclosed extra prior knowledge) and classes induced from T_phi statistics alone (no new resource).

**Gate (pre-registered kill-test).** Eval-only manner-class accuracy of the induced unit classes must
reach 0.75, against entry 3's measured 0.477 and the 0.403 majority baseline; voicing accuracy is
reported alongside. Below 0.75 the anchor is not funded as an initializer, and 1g.5 keeps only the
Gromov-Wasserstein and multi-restart initializers.

**Amended 2026-08-18 by the text-side ruling; the open supervision-cost decision is WITHDRAWN rather
than put to the USER.** A phone-to-articulatory-feature table is phone-level lexical knowledge and
pulls against the lexicon-free instruction, so that variant is demoted to an option on the phone
REFERENCE arm only and is not part of the primary path. What survives on the lexicon-free arm is the
half that needs no linguistic table: the acoustic descriptors themselves, which are deterministic
transforms of the audio, plus the coarse anchor English orthography supplies for free — vowel letters
against consonant letters, matched to periodic and energetic units against the rest. That anchor is
much weaker than the phone-level one and is registered as such: it separates two classes rather than
five, so its kill-test bar is the two-class version of the same read, and it is expected to constrain
rather than to determine.

**Sharpened 2026-08-18 after the candidate screens; five changes, none of them a new information
source, and the first is a change of PRIMARY form.**

1. **The zero-cost distributional anchor becomes the primary form of the coarse partition.** Take the
eigenvector of the LARGEST eigenvalue of the normalized Laplacian of the symmetrized text-symbol
bigram matrix — see the spectrum-end correction below, this said "second eigenvector" and was
wrong — and the
same on the 500-by-500 unit bigram matrix; each gives a two-class partition with unknown polarity, and
the two polarity bits are fixed with no search — audio side by which class has the higher periodicity
and energy, text side by which class functions as syllable nuclei (readable off the corpus as the class
whose per-word count is bounded below by one). The mask is where the two sides agree. This costs
**nothing in the supervision ledger** — no feature table, no formant table, no synthesizer — which is
precisely why it now leads: it establishes whether a coarse anchor exists at all before anyone argues
the price of a table. The published warrant is on written symbol streams (Goldsmith-Xanthos 2009;
Thaine-Penn 2017 report near-perfect vowel/consonant recovery from character co-occurrence alone), and
the honest gap is that the AUDIO-side eigenvector has no precedent at 1.18 units per symbol with 26.7 %
of transitions invisible. Two free label-free pre-checks run first and can kill it in seconds: is the
TOP eigenvalue separated from the one below it on each side, and does the class called syllabic have the
lower mean edge-enrichment (if the eigenvector is tracking silence or session structure rather than
syllabicity, that is where it shows). Note this is exactly the "vowel-letter against consonant-letter"
anchor the text-side ruling already registered, now with a method attached.

**SPECTRUM-END CORRECTION, 2026-08-19 — implementer-found, planner-verified independently, and this
was a planner error in the method as first written.** The sharpening above originally specified the
SECOND-SMALLEST eigenvector (the Fiedler vector). That is the wrong end of the spectrum, and not
marginally. Syllabic and non-syllabic symbols ALTERNATE, so the two classes form a near-BIPARTITE
structure — many edges between the classes, few inside — whereas the Fiedler vector minimises the
normalized cut and is therefore built to find groups that stick TOGETHER. The bipartite signal sits at
the TOP of the Laplacian spectrum. I re-derived this myself on the project's own phonemized corpus
(60,000 sentences, stride 97, 39 symbols) rather than accepting the report:

| eigenvector | mass accuracy vs gold vowels | classes returned |
|---|---|---|
| second-smallest of the Laplacian (Fiedler) | **0.5642 — BELOW the 0.6115 majority** | noise: mixes F G K L P R TH W Y ZH with vowels |
| **largest of the Laplacian** | **1.0000** | all fifteen ARPAbet vowels including ER, zero errors |

The eigengap pre-check moves with it: the top gap is 0.2899 against 0.0136-0.0342 for every other gap
at that end, while the non-trivial gaps at the bottom are 0.0104-0.0482. **Had this shipped as written,
both pre-checks would have failed on a flat bottom spectrum and 1g.4 would have been killed by a false
negative before the audio side was ever read** — the most expensive kind of spec error, since it would
have closed the only zero-ledger-cost anchor in the phase. The convention question is settled too, and
it does not rescue the original text: eigenvectors of the normalized co-occurrence matrix and of the
Laplacian coincide with reversed order, so the co-occurrence matrix's SECOND-LARGEST eigenvector is the
Fiedler vector and scores the same 0.5642; the object that works is its MOST NEGATIVE eigenvector. The
likely reconciliation with the published "second eigenvector" phrasing is that those results decompose
a rectangular left-context-by-right-context matrix by singular value, which is a different object —
noted as the probable source of the error and not asserted, because it does not change what to compute.
Ratified as a METHOD CORRECTION and not a tuned knob: the end of the spectrum was chosen on the text
side, whose answer is known a priori and published, before any audio-side number existed.

2. **Quantile calibration replaces absolute thresholds.** No physical constant is fitted or assumed:
each cut is placed where the unit mass above it matches that class's own token proportion read off the
unpaired text. Only the SIGN of each discriminant is pre-registered from textbook acoustics — more
periodicity means voiced, never the reverse — which is what makes this a stipulated correspondence
rather than a discovered one, and therefore the one candidate the measured coarticulation inversion
cannot reach.

3. **Two named training-free detectors replace generic descriptors** where they exist: `arctan(A(1))`,
the arc-tangent of the summed linear-prediction coefficients, published at 99.07 % sonorant-versus-
fricative frame accuracy over the whole TIMIT database with no training stage (arXiv:1411.1267); and
the bandpass extremum rule published at about 83.5 % broad-class onset accuracy (arXiv:1411.0370).
Sonority contributes extra columns only — excitation strength from the linear-prediction residual
envelope plus periodicity and relative energy — and its expensive published components (zero-time-
windowed group delay, ten-pitch-period correlation) are cut as outside the one-day bar.

4. **A block-CONTAINMENT read is added to the kill-test, and it is the one that actually decides.**
Report the fraction of unit mass whose true majority symbol lies INSIDE its assigned block; register
0.85. A sharp but wrong mask is worse than no mask, and the accuracy gate alone cannot see that.

5. **The output form is a soft table, not a hard mask.** Estimate six independent binary memberships,
one per articulatory question, each from one descriptor by a two-component one-dimensional mixture
whose mixing weight is PINNED to that feature's text-side token proportion — four free parameters
against 500 points, so it cannot drift — and take the product across features. Registered arithmetic
that binds this: a conjunction cannot survive weak factors, and all six at 0.70 accuracy gives 0.12
joint agreement, so **at least four of the six must clear 0.80** or the product form is dropped for the
hard mask. The soft form is what 1g.5 rung 1 needs, since a 0/1 mask cannot carry a posterior-
regularization projection. On the lexicon-free primary arm the six questions are the ones the
descriptors answer directly (syllabic, voiced, continuant, strident) plus the two the eigenvector split
supplies; the full six-question articulatory version belongs to the phone reference arm, per the
text-side ruling.

**Refused here, with the reason, so the record shows what was considered.** A within-vowel formant
Procrustes onto a published American-English F1/F2 chart is folded in as a within-block sub-step in its
UNIVERSAL three-corner form only, never as a standalone arm and never with the English chart on the
primary path: its own source measured vowels POORLY separated by static formant samples and reaching
useful accuracy only once duration and spectral change were added (Hillenbrand 1995), and regional
variation dominates exactly the low and high back vowels on an accent-mixed corpus (Clopper 2005). An
articulatory-axis Procrustes on the frozen codebook geometry is REFUSED outright: it is the discarded
embedding-isometry family and, in objective form, the already-adjudicated proposal 4, whose synthetic
ladder at this stream's own coarticulation rate found the true coupling was not the minimiser in a
single one of sixteen configurations, and a solver started AT the truth walked downhill to a worse map.

**Status.** REGISTERED 2026-08-18, pre-run; amended the same day by the text-side ruling, no USER
decision outstanding; sharpened 2026-08-18 by the candidate screens (five changes above), with the
zero-ledger-cost distributional anchor promoted to the primary form.

**Status 2026-08-19 — THE ZERO-LEDGER-COST DISTRIBUTIONAL ANCHOR FAILS ITS PRE-REGISTERED GATE. The
method is correct and does not transfer to audio.** Dev-other, 572 scored utterances, labels eval-only;
dev-clean agrees within 0.02 on every row.

| side | top eigengap | silence check | accuracy | measured majority | over majority | verdict |
|---|---|---|---|---|---|---|
| text, phones | pass 0.2899 | — | **1.0000** | 0.6095 | +0.3905 | control passes |
| text, characters | pass 0.2254 | pass | **0.9764** | 0.6130 | +0.3634 | control passes |
| audio, raw | FAIL 0.0038 | pass | 0.5488 | 0.5711 | -0.0223 | killed at pre-check |
| audio, `seg16` | FAIL 0.0325 | pass | 0.5548 | 0.5449 | +0.0099 | killed at pre-check |
| audio, `seg12.5` | pass 0.1962 | pass | 0.5032 | 0.5312 | **-0.0280** | FAIL |
| audio, `seg9` | FAIL 0.1507 | **FAIL** | 0.7867 | 0.5154 | +0.2713 | killed at pre-check |

**Why this is a strong negative rather than an inconclusive one.** The positive control passes at
essentially ceiling on both text sides, so the implementation is not in question. The one rung that
clears both pre-checks lands BELOW its own majority baseline. It is not a polarity accident: flipped-
polarity accuracy is 0.39-0.43 on the failing rungs, so neither orientation is informative. And mask
density is 0.49-0.53 throughout — the two sides agree at the rate two INDEPENDENT balanced partitions
would, which is the stronger statement that the audio partition carries no information about the text
partition, not merely too little. The mask closes essentially nothing.

**~~The `seg9` row is the pre-check earning its keep, and it is worth registering as a method fact.~~
WITHDRAWN 2026-08-19 — the check fired on noise and earned nothing.** The reading registered here was
that `seg9` is simultaneously the best-looking rung (+0.2713) and the only rung whose silence check
fails, edge enrichment 0.021 against 0.017, and that the coincidence of "best number" with "only
rejected rung" argued for keeping cheap label-free pre-checks ahead of eval reads. A mass-weighted
permutation test, 20,000 reshuffles per rung holding each class's total mass fixed, says NO rung's
enrichment gap is distinguishable from chance: raw -0.0025 (p 0.4265, null sd 0.0031), `seg16` +0.0055
(p 0.1490, sd 0.0038), `seg12.5` +0.0029 (p 0.5442, sd 0.0044), `seg9` +0.0034 (p 0.5769, sd 0.0050).
I re-derived the p-values against a two-sided normal approximation before accepting them and they
reproduce to within the permutation null's own non-normality. The silence check decides a strict
inequality between two noisy weighted means, and on this table those means are indistinguishable. What
survives of the original reading is only the half that was never load-bearing: `seg9` fails the 0.85
accuracy bar on its own, which is what actually decides it. Registering
the complementary limit too, because `seg12.5` shows it: the pre-checks are NECESSARY, not sufficient —
a rung can pass both and still carry no signal.

**CORRECTION 2026-08-19, and it is mine to make rather than the implementer's to soften.** That
"necessary but not sufficient" line was an EMPIRICAL registration drawn from this table, and it rested
entirely on `seg12.5` passing both pre-checks while carrying no signal. `seg12.5` passed them only
under the mean-segment-duration polarity SUBSTITUTE. Under the registered periodicity-and-energy rule,
which has now been exercised, `seg12.5` fails the silence pre-check — and since raw and `seg16` already
failed the eigengap check and `seg9` failed both, **no rung passes both pre-checks under the registered
rule.** So the claim has no supporting example in this data at all; it survives as a logical truism,
which is not what it was registered as, and it is withdrawn in its empirical form. ~~What replaces it
is a stronger statement in the opposite direction: the pre-checks would have killed every rung the eval
read killed, so they were jointly sufficient here and were better than I credited them.~~ **THAT
REPLACEMENT IS ALSO WITHDRAWN, 2026-08-19, same day it was written.** The permutation test above shows
the silence half of the battery decides on differences within noise, so its kills are not evidence of
anything; a battery cannot be called jointly sufficient when half of it was firing at random. The
pre-checks HAPPENED to fire in the same direction as the eval read, which is not the same as working.

**What is actually established about the pre-checks, stated per check rather than per battery, because
both the implementer and I have now overshot in opposite directions.** The SILENCE half is refuted on
this table by the permutation test — measured, not merely doubted. ONE eigengap kill is thin and was
flagged as such when it was made: `seg9` fails by 0.0032 against its own third gap. The remaining two
eigengap kills were left as an open question here, to be settled by reading each rung's top gap against
its own runner-up rather than by argument.

**ANSWERED 2026-08-19 by measurement, and this closes the pre-check question.** A bootstrap over
utterances, 25 resamples, rebuilding the adjacency and the spectrum and re-running the check each time
— which is the right instrument, since reading a margin off a point estimate is what the replacement
lesson below forbids:

| rung | top gap | its runner-up | verdict | bootstrap pass rate | top gap mean +- sd |
|---|---|---|---|---|---|
| raw | 0.0038 | 0.0086 | FAIL | **0.00** | 0.0041 +- 0.0026 |
| `seg16` | 0.0325 | 0.1166 | FAIL | **0.04** | 0.0299 +- 0.0217 |
| `seg12.5` | 0.1962 | 0.0974 | PASS | **1.00** | 0.1986 +- 0.0296 |
| `seg9` | 0.1507 | 0.1539 | FAIL | **0.36** | 0.1452 +- 0.0309 |

**Final per-check characterization, which is what the whole exchange was for.** The SILENCE check is
refuted. The EIGENGAP check is sound on three rungs of four and unstable on the fourth — and its stable
PASS on `seg12.5` matters as much as its kills, because a check that only ever rejects is merely
conservative. `seg9` remains the coin flip its 0.0032 margin predicted. So the battery is neither
"happened to fire", which was the implementer's summary and too harsh, nor "jointly sufficient", which
was mine and too generous. Reading per CHECK rather than per battery is the whole of the difference.
One precision note, since 25 resamples has granularity 0.04: the pass rates support the
sound/unstable classification and nothing finer.

**An observation worth keeping, because it is counter-intuitive and it vindicates the rule's form.**
The eigengap check is MOST reliable exactly where the spectrum is flattest: raw's top gap is 0.0038
with a spread of 0.0026 — a tiny number with a tiny spread — and its kill is unanimous, because on a
flat spectrum every gap is small and the top one has no path to winning its own comparison. Absolute
gap size is not what makes the check trustworthy; separation between the top gap and its own runner-up
is, and that is what the relative rule already compares. The rule was right; the error was confidence
in reading it off one sample.

**The withdrawn caveat has an example after all, at reduced scope — registered as a NEW claim, not as a
reinstatement.** The claim withdrawn above was battery-level ("a rung can pass BOTH pre-checks and
still carry no signal") and stays dead: no rung passes both. With the silence check refuted, the
battery reduces by measurement to one check, and the necessary-but-not-sufficient statement holds of
that one check with `seg12.5` as its instance — a stable 25-of-25 eigengap PASS at 0.5032 accuracy
against a 0.5312 majority, i.e. clearing the check that works while carrying less than nothing. An
existence claim needs one instance and now has one, at a scope the original claim did not have.

**The pre-check question is CLOSED.** The battery is characterized, the sub-phase it belongs to is
closed NOT FUNDED on the accuracy bar alone, and no further reads against it are funded.

**REPLACEMENT LESSON, and this one is the implementer's rather than mine.** A label-free pre-check that
gates on a strict inequality between two noisy statistics needs a spread requirement, or it is a coin
flip dressed as a screen. That applies to BOTH halves here — the silence check compares two weighted
means, and the eigengap check compares two gaps in a sampled spectrum — and it is cheaper and more
durable than either of the two readings discarded above. It is registered against every future screen
in this program that wants pre-checks, and it is the only thing from this exchange that should
propagate.

**Ruling: the DESCRIPTOR route remains funded, and the phase's next 1g.4 step is the descriptor dump.**
Three reasons. The failure of the distributional form is not evidence against the descriptor form; it
is evidence that the unit graph's leading structure is not sonority, which was the original argument
for a STIPULATED rather than a discovered audio-to-text correspondence. Waveform measurement is one of
the exactly four information sources in this phase that are not statistics of the unit sequence, and
the sorting rule registered in the candidate screen says those are the ones to fund. And it costs 1-3
CPU-hours of numpy/scipy on artifacts already on disk.

**Ledger correction, because the price of the fallback is smaller than it looks and the phase must not
over-price it into oblivion.** What failed is the audio side of a distributional method; the descriptor
route replaces that audio side with deterministic transforms of audio we already hold, which is not a
supervision cost at all. On the LEXICON-FREE primary arm the text side stays free as well — the
character control at 0.9764 is exactly the vowel-letter/consonant-letter anchor English orthography
supplies for nothing. So the incremental ledger cost of the descriptor route on the primary arm is a
handful of bits of stipulated language-universal SIGN (more periodicity means voiced, never the
reverse), not the 39-row articulatory feature table, which stays demoted to the phone reference arm
under the 2026-08-18 text-side ruling. The coarse anchor has acquired a price; the price is small and
it is not the demoted table.

**Bar for the descriptor route, pre-registered now while no descriptor number exists.** It inherits the
SAME two-class gate — 0.85 accuracy AND at least 0.20 absolute over the measured majority, plus the
0.85 containment read — because the bar states what the anchor must deliver downstream and is not a
property of how it is computed. It is read FIRST on the same two-class vowel/consonant question rather
than on the five-way manner question, because that is now directly comparable against a banked failure
(`seg12.5` 0.5032 against a 0.5312 majority) instead of against nothing. The manner gate at 0.75 stands
unchanged for the phone reference arm.

**STANDING PROTOCOL CLAUSE for 1g, registered 2026-08-19 out of this error and strengthened the same
day by its own first execution.** Any read that will be compared against a banked number must reproduce
that number's scoring protocol and ASSERT it in the job, dying rather than reporting if it does not
match. The strengthened wording, which is the implementer's and is the version to keep: **a protocol is
a SET of decisions — the utterance subset, the unit subset, and the population every derived quantity
is fitted on — and repairing the ones you thought of does not establish that you thought of all of
them.** That is not a rhetorical flourish. On its first execution the assertion caught an INCOMPLETE
fix: two of the three protocol differences had been repaired and the axes were believed to match, but
the silence proxy was still fitted on all 8,416 stream utterances rather than the 2,864 labelled ones,
leaving an offset of 0.007 to 0.015 — small enough to read as rounding, large enough to move a margin
sitting near a bar, and invisible to the eye-check that caught the first discrepancy. The job died
instead of reporting a comparable-LOOKING number, which is the correct failure mode.

**Two downstream consequences, registered so they are not discovered later.** 1g.5 rung 1's block-
constraint regularizer is OFF — it was conditional on this kill-test and the kill-test failed — leaving
codebook-geometry row tying and the phone-marginal posterior-regularization projection as rung 1's
tuning-free regularizers. And 1g.3's initializer grid has an empty cell: the acoustic-phonetic anchor
was one of its five initializers, so that column is not run unless the descriptor route clears the bar
above. Neither is a gate change; both are the arithmetic of a failed pre-condition.

**Status 2026-08-19 (second read, CORRECTED 2026-08-19 — read only this table). THE DESCRIPTOR ROUTE
ALSO FAILS ITS GATE. 1g.4 CLOSES AS NOT FUNDED, both forms.** Dev-other, held-out fifth of 572
utterances, silence-proxy units excluded with the proxy fitted on the same population the spectral
read used; labels eval-only. The protocol-matching assertion passed, so the majority column below is
bit-identical to the spectral form's and the two are on one axis.

| rung | best descriptor | accuracy | measured majority | over majority | factors at 0.80 | verdict |
|---|---|---|---|---|---|---|
| raw | energy | 0.7929 | 0.5711 | +0.2218 | 0 | FAIL |
| `seg16` | energy | **0.8130** | 0.5449 | **+0.2681** | 1 | FAIL |
| `seg12.5` | energy | 0.7859 | 0.5312 | +0.2548 | 0 | FAIL |
| `seg9` | energy | 0.7824 | 0.5154 | +0.2670 | 0 | FAIL |

SUPERSEDED and not to be quoted: the first read of this table (accuracy 0.8001 / 0.8065 / 0.7821 /
0.7820 against majorities 0.5903 / 0.5767 / 0.5627 / 0.5420) was scored on a different protocol from
the spectral form it was being compared against — all dev-other utterances rather than the held-out
fifth, and every unit carrying mass rather than only the units the partition placed. Each form's own
verdict was internally consistent and unaffected; the CROSS-FORM comparison drawn from it was void.

**It fails on the accuracy bar alone, and the correction moved it AWAY from the spectral form rather
than toward it.** The margin half is cleared on all four rungs (+0.2218 to +0.2681) while accuracy tops
out at 0.8130 against 0.85, and the gate is a conjunction. The spectral form missed both halves and
landed BELOW its own majority. Two predictions were registered before this read: the implementer's,
that the corrected numbers would fail BOTH bars — wrong, and recorded as wrong; and the planner's, that
accuracy would fall to 0.78-0.80 with the margin at +0.24 to +0.26 — right on the verdict, wrong on the
magnitude and wrong on the mechanism, see below.

**THE PREMISE BOTH PREDICTIONS SHARED IS REFUTED, and this is worth more than either prediction.** Both
decompositions assumed the excluded units are EASY — near-perfect accuracy on them, hence removing them
would lower the measured accuracy. The corrected job reports the held-out mass fraction directly, so
the held-out accuracy follows from the identity with no assumption left in it:

| rung | held-out mass fraction | accuracy on held-out units | accuracy on placed units |
|---|---|---|---|
| raw | 0.131 | 0.848 | 0.7929 |
| `seg16` | 0.128 | 0.762 | 0.8130 |
| `seg12.5` | 0.132 | 0.757 | 0.7859 |
| `seg9` | 0.141 | 0.780 | 0.7824 |

The held-out units are scored at 0.757-0.848 — comparable to or WORSE than the units that remain, and
below them on three rungs of four. Edge-enriched transitional units are the HARDER population, not the
easier one, which is why removing them RAISED accuracy on three rungs instead of lowering it. Each row
recombines to the first read's all-units accuracy to four decimals, so the decomposition is exact.

**What survived and what did not, in the planner's own prediction, separated because the distinction is
the methodological point.** The load-bearing half was premise-free and held: the matched-protocol
majority is not a prediction but a measurement — it is the spectral form's own column — so only the
accuracy was unknown, and the margin could fail only if accuracy fell below 0.7449, a drop of 0.062.
That bound needed no assumption about the held-out population and it was correct. The POINT ESTIMATE
layered on top of it (0.78-0.80, falling) used the shared premise and was wrong in direction. Register
the general form: state the premise-free bound, and treat any point estimate that requires modelling an
unmeasured population as the weaker claim it is.

**RULING: the gate is NOT relaxed and the near-miss licenses nothing.** Two reasons, and the second is
now the load-bearing one. (1) The bar was registered before the numbers existed, and no rung comes
within 0.037 of it. (2) SUBSTANTIVE, and untouched by any of the corrections: at 0.81 a HARD mask
zeroes the correct entry for about a fifth of unit mass, and a wrong hard constraint is unrecoverable
downstream — which is exactly why the containment read was registered beside it.

**CORRECTION to the planner's third reason, withdrawn 2026-08-19.** I had written that the two halves
of the gate order the rungs oppositely, so relaxing the accuracy half would select `seg16` while
relaxing the margin half would select `seg9`, making a relaxed bar's choice an artifact. On the matched
protocol that specific claim is FALSE: `seg16` is top of BOTH halves (0.8130 and +0.2681), so a relaxed
bar would select it unambiguously. The clean inversion was an artifact of the unmatched axis. What
remains true is weaker and does not carry the ruling: the halves are still non-redundant below the top
(raw is second by accuracy and last by margin; `seg9` the reverse), so they are a genuine conjunction
rather than one redundant test — which was the correction I made to the implementer's diagnosis, and
that correction still stands. The ruling rests on (1) and (2) above and never needed the third reason.

**No fishing, registered explicitly because the dump turned out to cost twelve minutes rather than the
1-3 hours estimated.** Cheapness is not a licence to add descriptors, rungs or contrasts until
something clears 0.85. That would fit the gate rather than pass it. Any further descriptor work on this
question needs a new pre-registered bar and a stated reason that is not "the last set nearly made it".

**The soft product form is dropped by its own pre-registered arithmetic**, correctly and without
renegotiation: at most one of seven factors clears 0.80 against the four of six required, and none
clears it anywhere on dev-clean.

**Two corrections to what this plan registered about the descriptors themselves.** (1) Sharpening item
3 above said two named published training-free detectors REPLACE generic descriptors where they exist.
The measurement reverses that: `arctan(A(1))` reads 0.6182-0.6575 per unit against plain energy's
0.7821-0.8065, making the named detector the second-worst item in the set, ahead only of zero-crossing
rate. The generic descriptor wins and the sharpening was wrong. (2) The published 99.07 percent figure
does not transfer to this bed — the same detector reaches 0.8524 area under the curve on sonorant
against fricative over 12,657 dev-other frames, and the two numbers are different metrics so this is a
failure to transfer rather than a contradiction. TIMIT is clean hand-segmented studio speech; this is
the noisy half of LibriSpeech with forced-alignment boundaries that place coarticulated transition
frames inside their neighbours. The shortfall is recorded in the module rather than left to be
rediscovered.

**One registered sign is contradicted by measurement and was correctly LEFT UNFLIPPED.** `excitation_db`
was registered +1 and reads 0.4265 on the sonorant/fricative contrast, the wrong side of chance.
Flipping a sign after seeing the data converts a stipulated correspondence — which costs a handful of
bits in the ledger — into a fitted one, which does not, and that conversion is the entire ledger
argument for this route. It stays as registered. Recorded as a refuted stipulation rather than a silent
non-event: it is a small hit to the route's premise that the physics signs are known in advance, and it
is the sort of thing that should be visible if this family is ever revisited.

**Two implementation defects were caught by calibrating against gold BEFORE the job was built**, and
both would have been invisible in the aggregate: missing pre-emphasis in the predictor fit (suppressing
the published detector from 0.8524 to 0.7714, where sweeping the model order — the obvious first
guess — does nothing), and an autocorrelation normalized by each frame's zero lag, which made the
statistic decay with lag and therefore scored low-pitched voices below high-pitched ones AT IDENTICAL
PERIODICITY. That is a speaker confound sitting inside the descriptor whose job is finding syllable
nuclei, on a mixed-sex corpus. A pure 150 Hz tone read 0.63 and reads 1.00 under a proper normalized
cross-correlation.

**Downstream, now final rather than pending.** 1g.5 rung 1 — the two-sub-state class 1g.0 selected —
gets NO block constraint and NO soft table from 1g.4, leaving codebook-geometry row tying and the
symbol-marginal posterior-regularization projection as its tuning-free regularizers. 1g.3's initializer
grid loses the acoustic-phonetic column permanently rather than provisionally. **1g.4 closes NOT
FUNDED**, and per the standing rule that licenses not funding it here — never "a coarse acoustic anchor
could not work", and specifically not on another representation or another descriptor family, which
would have to be screened rather than inheriting this verdict.

**Outstanding, carried forward.** The audio polarity bit in the failed runs was fixed by mean segment
duration, the substitute ratified 2026-08-19 when the descriptors it should have used were queued
behind the job that needed them. A failed partition cannot be rescued by a better polarity bit and the
flipped accuracies prove that, so this does not qualify the verdict — but when the dump lands, the
`descriptors_json` switch must be exercised and whether the bit flips reported. One reporting duty
added for reproducibility: the eigengap pre-check's threshold must be stated numerically in the log,
since `seg9` fails at 0.1507 while `seg12.5` passes at 0.1962 and the rule separating them is not
recoverable from the table alone.

### 1g.5 Channel estimation on the real pooled stream (the USER's proposal 1, simplest form first)

**Purpose.** The candidate itself: estimate the unit-given-phone channel from unpaired audio under a
pinned text phone language model, and decode through that language model.

**Approach.** The estimated object is `P(unit | phone)` and NOT `P(phone | unit)`; the direction is a
registered design constant, not a convention. The identifying design matrix for `P(unit | phone)` is
the text-side positional matrix over 39 phones, measured full column rank with smallest singular value
about 3.6e-4, while `P(phone | unit)` is identified through the audio-side positional matrix over 500
units, whose smallest singular value `SAE_1f.md` conclusion 8 measured as exactly 0 on every pooled
rung, structurally — an utterance supplies about 86 positions where full column rank needs 500. That
one difference is why 1f entry 2 died and why proposal 1 is not killed by the same measurement.
RUNG 1, the simple form and the one that is directly comparable to every banked 1f
number: hidden states are the 39 phones, transitions pinned from the phone language model and never
estimated, emissions a 39-by-500 categorical channel in the direction above, one segment per state, Baum-Welch on the
`seg12.5` stream, multiple restarts, selection by the section-1.0 unsupervised metric only. Three
regularizers, all free of a weight that would need tuning, and all aimed at the measured
37-observations-per-free-parameter regime: smoothness of the channel rows over the codebook's own
centroid geometry (units close in encoder space share phone distributions — label-free, the centroids
already exist); the 1g.4 block constraint if it clears its kill-test; and a phone-marginal POSTERIOR
REGULARIZATION constraint — 39 dual variables, closed-form projection, no weight to select — in place
of the entropy and coverage terms proposal 2 suggested, which have no admissible label-free selector
on this stream. RUNG 1 IS NOT NECESSARILY THE STRICT ONE-SEGMENT-PER-PHONE FORM: 1g.0 decides that,
and if the strict form fails its admissibility ratio while the two-sub-state or context-conditioned
class passes, rung 1 IS that class and the strict form is never run. RUNG 2, the USER's proposal 1 in full and PROMOTED 2026-08-18 to the form 1g
expects to win rather than a conditional follow-on: explicit fertility and alignment states —
substitution, repetition, deletion — on the run-length-DEDUPED raw stream rather than the pooled one.
Three measurements now point here and they compose into one argument. The insertion-forgiven error of
`raw` is 0.140 against `seg12.5`'s 0.347. The pooled rungs exist only to fix the token RATE, which the
reference's own successor established when it deleted its k-means stage and showed by ablation that
the clustering was doing rate work rather than linguistic work. And the boundary question, which is
what 1g.6 was built to chase, DISSOLVES here rather than being solved: a fertility channel keeps 2.82
observations per phone at the raw stream's 0.132 substitution level instead of pooling them into one
segment at 0.230, so it never has to place a boundary at all. Under the lexicon-free target this
matters more, not less — BPE token durations vary far more than phone durations, so a
one-segment-per-symbol assumption is a worse fit there than it was for phones. RUNG 3,
conditional on rung 2: the language-model order ladder 2 to 3 to 4 to 5 with beam pruning — the
precedent's own schedule — and the vocabulary constraint, expectation counts taken through the word
n-gram and the corpus's own SPELLING lexicon so only real word sequences are decodable. That is the
strongest hypothesis-space reduction available here, it is the mechanism the published noisy-channel
speech decipherment work actually credits, and under the 2026-08-18 text-side ruling it costs no
pronunciation lexicon on the primary arm. ALTERNATIVE
ESTIMATOR, same model, admitted at rung 1 if 1g.2 licenses it and CPU-minutes to run: the
fixed-core moment estimator parked as ladder entry 1 in `PLAN_1F.md` — fit the channel so that the
unit co-occurrence matrix is reproduced by the text-pinned phone-bigram core, which is the same
equation as the USER's proposal-4 graph-matching form with the core pinned instead of free, and is
therefore that proposal in its identifiable form rather than a separate method.

**MODEL-ENLARGEMENT LADDER, added 2026-08-18 from the theory battery, and it replaces "add capacity if
it underfits" with an order and a budget.** Theory-battery finding 5 measured which mismatch costs what
when EM is started at the true channel, and finding 2 measured what the corpus can afford. Both are
simulation, so this ladder is an ORDER OF WORK rather than a result, and each rung's own 1g.2 audit is
what discharges it:

| enlargement | parameters | segments per parameter at 20.5 h | measured cost of omitting it |
|---|---|---|---|
| context-independent emission | 19,461 | 37.0 | (the base model) |
| **+ per-symbol fertility over {0,1,2,3}** | **117** | 6,157 | **+0.151 absolute — the largest** |
| + per-symbol duration over 1..8 | 273 | 2,639 | (same mismatch, richer form) |
| **+ FACTORED left context** `P(u | y_t, y_{t-1})` ~ `B[y_t,u] C[y_{t-1},u]` | **19,500** | 36.9 | **+0.080 absolute** |
| + FULL diphone context (39x39 states) | 758,979 | **0.9** | not affordable at 20.5 h; 44.4 at 960 h |
| + full triphone | 29,600,181 | **0.0** | out of reach at both corpus sizes |
| **+ 8-cluster per-utterance channel gain only** | **4,000** | 180 | +0.019 absolute — first to drop |

The defensible bundle is fertility, then factored left context, then the per-utterance gain: 43,078
parameters at 16.7 segments per parameter, thin enough that the FACTORED form is mandatory and the full
diphone table is not an option here. Two registered consequences. **960 h is not funded for
emission-matrix accuracy** (finding 2 prices that at 0.0009 absolute); the single reason that would
justify a re-dump is the full diphone table, and it must be argued as that and not as "more data".
And **the enlargement does not itself promise the likelihood/error correlation flips** — no theorem
says so — which is why every rung re-enters 1g.2's oracle-start drift test rather than being adopted on
the strength of its parameter count.

**Refused at rung 1, with the reason, because it was proposed and the refusal is load-bearing.** A
variant that DELETES fertility — one symbol per segment on the pooled stream, licensed by the 0.9 %
average rate match — is refused on two independent measurements. The rate match is an average, not an
alignment: the insertion-forgiven error is 0.140 on the raw stream against 0.347 on `seg12.5`, which is
why fertility was promoted rather than deleted. And omitting fertility is the LARGEST single
misspecification the theory battery priced (+0.151 absolute), larger than coarticulation and eight
times the speaker term. The licensing condition that variant cited — the memoryless-per-symbol
structure condition of arXiv:2603.02285 — is the condition this stream is already measured to violate
at 26.7 % of transitions, and that paper's own bound carries a constant of order 1e7 here, so it is an
existence result and not a certificate.

**Two structural details imported from the verified precedent, and one thing NOT imported.** Import:
the alignment model is bounded at **at most one insertion or one deletion in a row** (a free channel
of insertions is the mechanism by which a decoder emits a high-language-model-probability string almost
independently of the observation); and **silence is retained as a hard word-boundary anchor**, already
registered as 1g.1 E4, because the precedent credits it with pruning the word-segmentation space and
this project currently detects silence with a label-free proxy and then discards it. NOT imported: the
weighted-finite-state-transducer implementation itself. The environment has no OpenFST, no OpenGrm and
neither `pynini` nor `pywrapfst`; an aarch64 build plus a 500-symbol flower transducer composed with a
lexicon and word language model is a multi-week toolchain project before any number appears. The
fundable route to the same object is a hand-rolled trie decoder over the objective already registered
as 1g.2 (d), which reuses that dynamic program plus per-segment emission scores and a backtrace.

**Experiments.** Rung 1 on `seg12.5`, with `seg9` (the label-free-defensible rung) repeated only if
rung choice becomes decision-relevant; rungs 2 and 3 conditional on their predecessors. Decode by the
1g.1 language-model decoder throughout; the word-level lexicon decode is run on the winner only and
reported as WER. The sequencing simplest-first is a planner call under the standing simplicity ruling,
not a judgement that the alignment states are unnecessary — they are rung 2 precisely because the
measured fertility argument for them is strong. TWO ESTIMATOR CONSTANTS, both registered rather than
left open, both from verified results. (1) SOFT Baum-Welch, never Viterbi or hard EM: classification
maximum likelihood gives an inconsistent estimate outside special symmetric cases such as equal
mixing proportions (Ahfock and McLachlan, arXiv:2004.06237, quoted with its exception intact), and our
symbol distribution is strongly skewed in every text side under consideration, so the biased regime is
exactly the one we would be in. This is also a third, independent reason section 1a's closure does not
transfer: its continuous run used Viterbi hard EM. (2) The initialization matters more than "EM finds
a local optimum" suggests. Wu 1983 (Annals of Statistics 11(1):95-103) proves convergence to
STATIONARY points, upgrades that to local maxima only under a condition he calls typically hard to
verify, and gives a worked case where a symmetry present at the start is preserved by every subsequent
iteration so the sequence converges to a saddle point. Combined with the measured fact that the
uninformative channel — emitting the unit marginal regardless of symbol — is an exact fixed point of
this objective, that means a content-free start is not merely a bad start but an ABSORBING one. Hence:
RESTART BUDGET registered as a constant rather than left to the implementer: at least 1,000 restarts
at bigram order, which forces a vectorized (batched,
GPU) forward-backward rather than the committed per-utterance loop. The number is not arbitrary — the
verified reference for this exact algorithm (emissions-only EM under a pinned n-gram backbone) reports
18 percent accuracy at one restart and 90 percent at one hundred on a cipher with 2.08 cipher symbols
per plaintext symbol, and continued gains to 100,000 on a harder one; our stream is at 500/39 = 12.8
symbols per phone and noisy on top, so a few dozen restarts is not a defensible budget here. Two free
reads come with the restart sweep and are required in the report: the rank correlation between
held-out likelihood and PER across restarts (the direct re-measurement, on the current inventory, of
the claim that closed section 1a — and the test of whether label-free selection is possible at all),
and the histogram of converged likelihoods (multi-peaked and sparse is the published signature of real
recoverable structure; a smooth unimodal blob is the signature of nothing to decipher).

**Gate.** The registered 1f arm gate UNCHANGED (USER ruling 2 of 2026-08-16 and the 2026-08-16
margins): on dev-other, plain PER as scored, labels eval-only — (M1) candidate PER at most
min(random-map null, pseudo-pair null) minus 0.05, and (M2) a PER rise of at least 0.05 under the
audio-swap control. Both nulls are re-banked under the 1g.1 language-model decoder before the read, so
candidate and nulls never differ in decode privilege. Reported alongside, never as a gate: where the
candidate sits on the 1g.1 tolerance curve — the number that says whether a passing candidate is
merely better than nothing or actually usable.

**Status.** REGISTERED 2026-08-18, pre-run; funded only after 1g.2 and 1g.3 pass their gates.

**Status 2026-08-19 — RUNG 1 IS NOW FIXED BY MEASUREMENT, not by the simplest-first ordering.** 1g.0
refuted class (a) on every cell of both splits and found class (c) admissible on all of them, so under
that phase's pre-registered gate **rung 1 is the two-sub-state class and the strict
one-segment-per-symbol form is dropped rather than run.** The simplest-first sequencing registered
above was always conditional on 1g.0 and is now discharged: the simple form is not run, and no
judgement about it is owed beyond the screen's own. Two consequences. The registered
restart-budget constant, the soft-Baum-Welch constant and the direction constant `P(unit | symbol)` all
carry over unchanged — none depended on the topology. And **1g.1 E5 runs the gate-selected class**,
because the basin of a model class this phase has just ruled inadmissible is the wrong thing to
measure; where it is nearly free to add as a second row of the same sweep, the simple form is reported
alongside purely as the calibration row against the theory battery's simulation, which used it. The
minimum-duration-two topology is adopted here as SUFFICIENT and explicitly not as necessary: 1g.0 put
the rate that clears the bar at 0.15-0.40 against the 0.5 that topology forces.

### 1g.6 Iterative unit refinement against the channel (the USER's proposal 5), decomposed

**Purpose.** The proposal is to stop treating the k-means layer as fixed and alternate acoustic
clustering with decipherment: over-cluster, decipher, merge units that decipher to the same phone,
split units that decipher to several, decipher again. It contains three operations with different
mathematics and different measured support, so 1g funds them separately rather than as one loop.

**Approach, per operation — REWRITTEN 2026-08-18 after the dedicated verification returned. Two of the
three operations are refuted, so this phase is now mostly a not-funded record plus one cheap
replacement pointed the opposite way.**

(a) MERGING units whose deciphered posteriors agree is REFUTED three independent ways and is dropped —
including the estimation-variance rationale THIS PLAN offered for it earlier the same day, which was
wrong by a factor of 36 and is retracted. Merging is a deterministic map, so the data-processing loss
is an identity rather than a bound: the mass-weighted divergence between the merged units' posteriors,
zero only if those posteriors are identical, which units sharing an argmax are not — and that
difference is exactly the allophonic information the proposal wants to keep. The plug-in estimation
bias merging could remove is about 0.0132 nats, 0.68 percent of the 1.93-nat mutual information, while
the one merge this project ran cost 0.472 nats. Data volume is not the constraint in the first place:
the map needs roughly 1,200 tokens to be determined against the 720,315 we have, about 600x
over-determined. And it is a ONE-STEP FIXED POINT — for any decipherment whose per-unit decision
depends on the unit through statistics LINEAR in its counts, which is every statistic in play here,
the merged block's statistic is the mass-weighted average of its members' and retains their shared
argmax, so the second decipherment reproduces the same partition and the merge injects nothing the
first one did not already have. The defensible version is soft shrinkage, already registered inside
1g.5 as the codebook-geometry row-tying prior; hard merging is its infinite-shrinkage limit and is
strictly dominated.

(b) SPLITTING is NOT FUNDED: its headroom is bounded at about 0.100 and has already been collected by
something cheaper. Substitutions 0.230 minus this codebook's irreducible substitution floor of about
0.130 — three convergent estimates, raw deduped substitutions 0.132, the supervised probe's 0.1296,
and the coarticulation diagonal — leaves 0.100 for any finer inventory, and the E1 probe already
reaches 0.1296 using the SAME 500 units with a four-frame window. Splitting and context modelling
target the same term and it is collected once; 1g.5's context-conditioned emissions are the cheaper
collector. Two further bounds: our units sit within 0.022 phone-normalized mutual information of the
best published frozen-feature clustering at this inventory size, and at K=2000 the unit bigram would
carry 0.18 observations per cell.

(c) RE-SEGMENTING is the only operation left, and this plan's own earlier premise-correction —
"suspect the units resolves into suspect the boundaries" — is now itself QUALIFIED and must not be
repeated as established. The boundary-versus-confusability split was NEVER MEASURED on our stream; it
was predicted by converting three published boundary swaps into relative error reductions, which puts
a gold-boundary ceiling at 0.30 to 0.40 against the current 0.414 — boundaries as the SMALL term
(central share 0.059 of the ceiling), confusability as 72 to 97 percent of it. Those scalings are all
phone-target measurements and transfer to the lexicon-free target only as order-of-magnitude priors.
So the operation is gated on MEASURING the split, not on assuming either direction — and 1g.5's
raw-stream fertility form dissolves the boundary question rather than solving it, which is the cheaper
route if it works.

**REPLACEMENT, cheap and pointed the other way.** The proposal's instinct is to over-cluster; the
evidence points DOWNWARD. Phone discriminability on self-supervised units is U-shaped in inventory
size with its published optimum near K=200 while speaker information rises monotonically with it — and
phone-normalized mutual information, the metric this program has been reading throughout, rises
monotonically with K by construction and therefore structurally cannot see that trade. Our K=500 is
past the published optimum. Two CPU-cheap probes replace the merge and split operations: a k-means
sweep DOWNWARD in FEATURE GEOMETRY (K = 100, 200, 1000), which is not what `brown100` tested — that
merged by bigram context and collapsed the graph correlation 0.370 to 0.103 — and PER-SPEAKER
mean-variance normalization before clustering, speaker identifiers being permitted supervision under
the 2026-08-14/16 amendment and disclosed as a cost. The second is the only operation anywhere in this
phase that injects information from OUTSIDE the decipherment loop, which is precisely the property the
proposal's own three operations lack. STANDING ORDER from a published negative: any re-clustering
happens BEFORE decipherment, never after it — re-clustering after grounding is measured to hurt.

**The sweep is extended downward to K = 39, 60 and 80, added 2026-08-18, and the reason is not the one
that motivated the sweep.** Theory-battery finding 4 says any moment or spectral estimator is dead by
six orders of magnitude on a 500-unit alphabet — 1e11 segments to resolve the 39th singular direction,
with only 10 of 39 directions above the sampling-noise floor at 20.5 h — and that **the only form of
that route which is not dead is on a reduced alphabet of about 30-60 units**, where the same tables
carry hundreds of counts per cell. So the coarse rungs are worth their hour for a reason independent of
the U-shaped discriminability argument. What is read at each rung is the CEILING only: refit k-means in
the same feature space, fit the best gold map (eval-only), and report the dev-other PER that map
reaches — the honest upper bound of anything built on that alphabet, at K=39 additionally under a
bijection, which separates "coarsening is fatal" from "bijectivity is fatal". The artifact this needs
already exists: the 1024-dimensional 50 Hz states are banked for all 8,416 utterances, so re-clustering
inside the same principal-component space needs no new encoder forward.

**What is REFUSED at K=39, stated because the coarse rung invites it.** Solving the map as a BIJECTION
— every symbol used exactly once — is not funded, on two independent grounds. Structurally, k-means at
K=39 produces roughly variance-balanced clusters while the English symbol unigram spans about two
orders of magnitude, so no bijection can satisfy the pinned marginal every other arm in 1g enforces.
Empirically, identity was already unconstrained-and-correct in the failed entry-5 run — all 39 types
used at 0.963-0.974 of the reference count with about 80 % substitutions — so "every symbol used
exactly once" is demonstrably not the constraint that was missing. The unicity-distance argument
offered for it assumes a noiseless deterministic cipher and this channel is 41.5 % noisy at its own
ceiling. The rung is a ceiling measurement, not a search.

**Experiments.** Nothing new is built until two free reads land. FREE READ 1 is 1g.1's E3 gold-boundary
ceiling, which splits the 0.4148 into boundary error and unit confusability. FREE READ 2 is already
registered and pending compute elsewhere: entry 7's full arm runs the reference pipeline's own
relabeling iterations on this bed, and `PLAN_1F.md` entry-7 stage-A RULING 3 already registers the
iteration-3-minus-iteration-1 delta as the read on what relabeling buys here. **FREE READ 2 IS
SUSPENDED 2026-08-18 and is no longer free.** Entry 7's iteration-1 checkpoint failed both arm-gate
margins with a flat audio-swap control, and the relabeling iterations now in flight were seeded from
it, so their delta would answer "what does relabeling buy starting from a degenerate checkpoint" rather
than the registered question. Until entry 7 has an iteration-1 checkpoint that clears M1 with a
non-flat swap, 1g.6 has ONE free read, not two, and its gate — which already required both — cannot be
discharged. That is registered as a suspension rather than a gate edit: the gate stands as written. Only then, one bounded
build: re-derive segment boundaries from the 1g.5 channel's own posteriors, re-quantize, refit, decode
— at most two passes, each taking the arm-gate read, with the pass abandoned if PER does not improve.

**Gate (pre-registered).** The re-segmentation loop is funded only if BOTH free reads support it:
E3 attributes at least 0.10 absolute of the 0.4148 ceiling to boundary placement, AND entry 7's
relabeling delta on this bed is an improvement rather than a degradation. Registered expectation
against which that gate will be read: the literature-anchored prediction is a gold-boundary ceiling of
0.30 to 0.40, i.e. a boundary share of 0.012 to 0.118 — mostly BELOW the 0.10 bar, so this gate is
expected to fail, and it was written before the prediction existed. Splitting is NOT funded whatever
E3 says, per the bound in (b). The eval-only lexicon question this raises is ruled here rather than
sent to the USER: a pronunciation dictionary remains permitted for MEASUREMENT, since the project
already reads MFA gold alignments (themselves lexicon-derived) under the prior-knowledge table's
evaluation-only license and ruling 3 already licensed phonemizing an output for scoring. So under the
lexicon-free target, where no gold BPE boundary exists because BPE tokens have no acoustic correlate,
E3 runs against gold PHONE boundaries as a proxy reference, and every threshold above is re-derived on
the new target before it is read rather than inherited. Each
pass of the loop must improve dev-other PER or the loop stops at that pass: this is a self-training
loop over an unsupervised objective, the family that produced this program's two logged collapses
(the section-3e.1 scorer co-training run and the section-3g from-scratch arm), so it gets a stop rule
before it gets compute. One further pre-condition, from Wu 1983 and registered here because it is
specific to THIS loop: a merge step seeded from a content-free decipherment cannot repair itself, since
the uninformative channel is an exact fixed point and a symmetry present at the start is preserved by
every later iteration — an absorbing state, not a slow one. So no pass of 1g.6 may run off a channel
that has not first cleared the 1g.5 arm gate; the loop refines a working map or it does not run.

**Status.** REGISTERED 2026-08-18, pre-run; both free reads pending.

### 1g.7 Repeated-token consensus: constraints from the same word said twice

**Purpose.** Every method 1f tried, and every distributional method 1g screens, reads AGGREGATE
statistics of the unit stream, in which a segment's identity and its context are confounded — which is
the measured cause of failure, not a guess: the transition objective is inverted by coarticulation
(`SAE_1f.md` conclusion 9, the real stream beating a by-construction factorized control), and
theory-battery finding 3 says second-order statistics leave about 18-19 free directions as a matter of
algebra. Two tokens of the same spoken word share identity and DIFFER in context, so a consensus over
repeats decorrelates the two with no labels. This is the only candidate the screens surfaced that
attacks that cause rather than routing around it.

**Approach.** Audio side only, no text at all in the first stage: align the unit stream against itself
with local alignment, seeded by exact repeated substrings so it stays tractable on 720k segment tokens;
cluster matched fragments into pseudo-word types by connected components; take a per-column plurality
consensus over each type's tokens. The unit of constraint changes, and that is the point — a type of
5-8 symbols imposes 5-8 simultaneous constraints on the channel where entry 3's unary fingerprints
imposed one weak constraint per unit. Distinguish this from 1f entry 5's unit-BPE word matching, which
is closed: BPE merges are FREQUENCY-derived and carry no requirement that two instances be the same
acoustic event, whereas every member of a discovered cluster is an actual repeated acoustic event.
The published configuration this borrows runs on exactly our stream type (a self-supervised layer plus
k-means) and reports state of the art on the spoken-term-discovery track. A downstream free diagnostic,
if and only if the fragments exist: same-length fragment pairs differing in exactly one aligned column
give a substitutability graph whose text-side counterpart is one pass over the corpus vocabulary — the
one read anywhere in 1g in which phonetic CONTEXT IS HELD CONSTANT BY CONSTRUCTION. Its own yield
measurement already argues it cannot stand alone (1,633 such pairs among 3,049 repeated types at 5.7 h,
309 among the 821 types seen five or more times), so it is a scorer inside this phase, never an arm.

**Experiments.** Yield and purity ONLY, before any assignment code is written, and the assignment half
is explicitly not funded on this gate. Mine repeats, form clusters, and read two numbers: how many
pseudo-word types of 4-10 segments have at least three tokens, and whether a cluster's tokens decode
(under the gold-fitted map, eval-only) to strings that agree with each other better than two random
corpus tokens do. One CPU-day on banked artifacts.

**Gate (pre-registered).** Both reads must pass: at least about 1,000 usable pseudo-word types at
20.48 h — the text-side estimate says roughly three times the 1,705 that exist at 5.7 h scale, so a
large shortfall means the discovery step failed rather than the idea — and within-cluster agreement
strictly above the random-pair baseline. If clusters mix word identities, the "same word twice"
equality this rests on is simply false on this stream and the phase closes for one CPU-day. Registered
caution carried into any later stage: fully unsupervised large-vocabulary segmental systems are
published at 70-95 % word error, so term discovery is a source of CONSTRAINTS and never a recognizer.

**Status.** REGISTERED 2026-08-18, pre-run. Ledger cost on the primary arm: none in the discovery
stage; the assignment stage would read the corpus's own spelling vocabulary, which is free under the
text-side ruling.

### 1g.8 Synthetic acoustic atlas: manufacture the pairing instead of discovering it

**Purpose.** Every other candidate must break a near-symmetry from unpaired statistics, and the
measurements say that symmetry is barely broken on this stream. A rule-based FORMANT synthesizer is a
parametric signal generator — hand-written formant and duration rules plus a source-filter oscillator,
containing no recorded speech and fitted to no corpus — so every frame it produces carries its symbol
label BY CONSTRUCTION. Push synthetic audio through the identical frozen encoder and the identical
frozen codebook, and the channel is read off by counting. There is no search, no expectation
maximization, no gradient and nothing to select, which makes it the simplest item anywhere in 1g.

**Approach.** Synthesize from the unpaired text (and from balanced nonsense frames covering each symbol
in varied contexts), across the synthesizer's voice variants, speeds and pitch settings for variability.
Encode with the frozen wav2vec2-Large-LV60 layer 15, correct the synthetic-to-real domain shift in
closed form (per-utterance mean-variance normalization, then whiten by the synthetic covariance and
re-colour by the real one — one symmetric square root, no training), assign to the FROZEN 500 centroids
with no refit, count. HARD LINE, stated because this is the one place the no-paired-data constraint
could be violated silently: neural, unit-based, concatenative and diphone voices are DISQUALIFIED —
they are labelled recordings laundered through a model. Only the parametric formant path qualifies.

**The lexicon-free complication, which the screens did not price and which decides where this arm
lives.** A formant synthesizer labels its output in PHONES. Bridging those labels to the lexicon-free
primary arm's BPE or character symbols needs a phone-to-spelling correspondence, which is a
pronunciation lexicon by another name — so **on the primary arm this atlas is not free, and it is
registered as a PHONE REFERENCE ARM candidate** until a bridge is demonstrated that does not smuggle
one in. The one candidate bridge is the synthesizer's own rule-based letter-to-sound module, which is
not a dictionary; the shipped exception dictionary is not admissible, and the gap between the two must
be measured or the dictionary disabled outright before the arm is reported as lexicon-free.

**Experiments.** The kill test IS the method at one-twentieth scale, which is what makes this cheap:
synthesize 30 minutes, encode, correct, quantize, and read the coverage number FIRST. Then, and only if
coverage passes, the direct decode read against the registered anchors.

**Gate (pre-registered, coverage first and it is label-free).** The fraction of REAL-stream unit MASS
whose unit ever wins a synthetic frame must reach 0.60. Below that, synthetic speech lands in a
different region of the encoder's space, the table is undefined exactly where the real stream lives,
and the arm dies in one afternoon with no eval read taken. Running coverage first is not caution for
its own sake: the closest published measurement of this synthesizer's audio found it the farthest from
natural of four systems as far as a recognizer was concerned (32 % false alarms), so the domain gap is
real and already measured by someone else. If coverage passes, the decode read is the ordinary 1f arm
gate against the two banked nulls, with the top-1 agreement against the gold-fitted map reported
alongside so it is visible WHICH classes transfer rather than only how many.

**Status.** REGISTERED 2026-08-18, pre-run; not started. Ledger: this is a NEW disclosed prior — the
synthesizer's English formant and duration rule tables — and it must appear in the supervision-cost
ledger as such, priced by the gap between an English voice and a language-neutral one. Tooling gap
budgeted rather than discovered: no `espeak`/`espeak-ng` binary is on the environment's PATH, so an
aarch64 install precedes the one small GPU forward job this needs.

---

## Proposal adjudication (the USER's five proposals, 2026-08-18)

**Proposal 1, probabilistic noisy-channel decipherment — ADOPTED as 1g's mainline (1g.5), funded
behind the 1g.2 and 1g.3 gates.** Three corrections to it as stated, all first-hand. (i) It already
exists here: `scripts/unsupervised_asr/decipher.py::hmm_decipher` is exactly the bigram form — hidden
phones, transitions pinned from the language model and never estimated, a learned unit-given-phone
emission, scaled forward-backward — validated on simulated units in 1a; 1g's build is a real-stream
input path, the regularizers, and a sisyphus job, not a new algorithm. (ii) The proposal says the
difference from the frequency experiment is that it does not ask frequency-of-unit to equal
frequency-of-phone; that undersells what already failed — entry 5 matched positional unigrams AND
bi/tri-skipgrams, not marginals, and still landed at 0.8580. The difference that actually matters is
the DECODER: every 1f entry decoded per-segment argmax, this decodes through the language model, and
that difference is measurable on banked artifacts before any fit, which is why 1g.1 exists and runs
first. (iii) The alignment states are right and are 1g.5 rung 2, placed on the run-length-deduped raw
stream (2.82 units per phone, substitutions 0.132) rather than the pooled one (1.18, substitutions
0.230), because that is where an explicit fertility model earns its keep. The restart point matches
this project's own measurement — 1a recovers 0.97 from a truth start and 0.23/0.31/about 0/0 from
frequency-CDF, bigram, random and annealed starts — so initialization is the binding constraint and
1g.4 is the response to it. Adopted verbatim from the proposal: preserve N-best/lattices rather than
committing to 1-best.

*Literature verification for proposal 1 (planner fan-out 2026-08-18, every source fetched and quoted
first-hand; full agent report in the session transcript).* Four findings change how it is registered.
(i) HONEST ANCHOR, and it is not what the proposal implies: in Klejch/Wallington/Bell,
arXiv:2111.06799, Interspeech 2022, the decipherment stage ALONE reads 31.0 / 49.0 / 70.8 / 53.8 /
105.3 / 93.5 / 34.5 word error rate across its seven languages — above 49 on four of seven, outright
failure on two — and the usable systems (10.3-30.3) come from flat-start lattice-free MMI
semi-supervised training ON THE DECIPHERMENT LATTICES. The precedent supports "decipherment can emit
a lattice worth self-training on"; it does not support "decipherment alone yields a good
transcription", and importing its success means importing its neural second stage. Their front end
also consumed 110 h of paired speech in six languages plus six phonemic lexicons, so none of those
numbers transfer as an anchor for us; the substitution they name in their own conclusions — replace
the universal phone recogniser with automatic unit discovery — is exactly what 1g attempts, and it is
future work there, not a result. (ii) Klejch deciphers phones into GRAPHEMES and its final and
strongest stage composes a WORD trigram language model with a grapheme lexicon; a phone 2-to-5-gram
ladder alone discards the component the paper credits. This project already permits the lexicon, so
the word-level constraint enters 1g early — it is in the 1g.1 decode battery (E2) rather than waiting
for 1g.5 rung 3. (iii) The algorithm is published and works elsewhere: Berg-Kirkpatrick and Klein,
EMNLP 2013, run exactly emissions-only EM under a pinned character-trigram backbone, and report 18 %
accuracy at ONE restart against 90 % at one hundred, with gains continuing to 100,000 restarts on
their harder cipher — hence the restart budget registered as a constant in 1g.5. Their Zodiac-408 run
also measured the anti-alignment in this exact model class and measured it SMALL at the top (best of a
million restarts reaches higher likelihood than a gold-key start, -1466.5 against -1467.4, at 89 %
against 92 % accuracy), and Merialdo 1994 measured the same shape for part-of-speech tagging (97.0 to
95.2 over ten Baum-Welch iterations from a good start). That is the scale 1g.2's drift gate is set
against: one to three points is the published normal, and section 1a's continuous run lost 11.7. (iv)
Identifiability is favourable and misspecification is the whole risk: Allman-Matias-Rhodes 2009
Theorem 6 needs only three consecutive observations at 39 hidden and 500 observed states, and pinning
the text transitions removes the label-swap ambiguity; but Yang et al. arXiv:2603.02285's Condition 1
requires the TRUE channel to be memoryless per label, which our own coarticulation measurement
violates by a quantified amount, and their error bound's constant is about 1.2e15 at our utterance
length — an existence result, not a certificate. Under misspecification the only theorem available
(Douc and Moulines 2012) says the estimator converges to a relative-entropy projection with no
statement about recovering the hidden states. One error the fan-out flagged explicitly and this plan
must not make: 1f entry 2 died on sigma_min of the positional-unigram matrix over the 500 UNITS, which
is a different matrix from Yang et al.'s full-column-rank condition over the 39 PHONES (measured
non-zero on LibriSpeech text), so entry 2's structural death does not kill proposal 1. Registered
expectation, from the fan-out and not softened here: probability that a label-free-selected run
reaches dev-other PER at most 0.70 is put at 0.25 (0.40 with the word-level language model), at most
0.79 — clearing the arm gate's bar by its own margin — at 0.45, and at most 0.55, the range where the
result would be a genuinely seedable initialization, at 0.12.

**Proposal 2, matrix PUSM — NOT FUNDED as a standalone arm; admitted as a SINGLE conditional run
behind 1g.2, with two registered modifications, because it costs one configuration flag.** Verified at
source: entry 5's generator is one `Conv1d(500 -> 39, kernel 4, no bias)` over one-hot FRAME units
with segment pooling afterwards (`espum_model.py` Generator), and over a one-hot input that
convolution is exactly a sum of four lookup tables, so the proposed per-unit matrix is its kernel-1
restriction — 19,500 parameters against 78,000, a strict subclass of a class that already ran to
convergence and failed at a stable three-seed attractor. Two things the fan-out established that
change the verdict from "pointless" to "cheap and conditional":

- THE ONE REAL ARGUMENT FOR IT, and it is a mechanism rather than a hope: with `padding="same"` the
  kernel-4 window makes each pooled segment's posterior depend on frames outside its own segment
  (about 60 ms of reach), so the CNN can manufacture correct-looking n-gram count statistics from
  neighbouring units WITHOUT ever identifying a unit — which is precisely entry 5's measured failure
  signature (right emission rate, all 39 types used, identity near chance). The matrix forbids that
  leak by construction, and it is the only edit on the table that enforces the per-position
  factorization both published theories in this family assume.
- THE DECISIVE ARGUMENT AGAINST, and it is the same one that closed 1f entry 2: DIRECTION. Proposal 2
  estimates `P(phone | unit)`, whose identifying design matrix is the AUDIO-side positional-unigram
  matrix over 500 units — the matrix whose smallest singular value `SAE_1f.md` conclusion 8 measured
  as exactly 0 on every pooled rung, structurally, because an utterance supplies about 86 positions
  where full column rank needs 500. Proposal 1 estimates the OTHER direction, `P(unit | phone)`, whose
  design matrix is the TEXT-side positional matrix over 39 phones, measured full-rank with smallest
  singular value about 3.6e-4. So entry 2's structural death transfers to proposal 2 and does NOT
  transfer to proposal 1 — the two proposals are not two flavours of one idea, they sit on opposite
  sides of an identifiability boundary this stream has already been measured against.

Refuted outright, so they do not enter any conditional run: the coverage regularizer has nothing to
fix here (the arms already emit all 39 types at 0.963-0.974 of the reference count, so the term is
zero-gradient at the solution); the soft-to-sharp entropy anneal has no support in this family (the
reference uses none, and the only published matrix generator, arXiv:1804.00316, found straight-through
discretization beat plain sharpening); and the "is the learned acoustic CNN necessary" framing does not
apply, because our port's generator consumes one-hot unit ids and learns no acoustics at all — the
in-house generator that DOES work on this bed is the GAN's kernel-9 stride-3 convolution over
continuous features, i.e. more capacity and more context, not less. Registered modifications if the run
happens: drop the tri-skipgram term and raise the batch, because the fan-out's noise accounting puts
the tri-skipgram term's sampling noise above the bigram term's largest possible signal at entry 5's
operating point — 1g.2 measures those per-term standard errors empirically before anything is launched.

**Proposal 3, hybrid likelihood plus statistics matching — NOT FUNDED as a weighted hybrid; its
useful half is folded into 1g.5 as a tuning-free constraint.** The fan-out settled this algebraically
rather than by preference, and three results decide it. (i) The positional-unigram term is VACUOUS on
a shared matrix: when the same channel supplies both the decode posterior and the statistics, the term
is zero at EVERY marginal-matched channel — including the content-free random null that scores 0.8946
— so it cannot separate a 0.41 map from a 0.89 one, which is exactly what entry 5's outcome looked
like (marginals matched, identity wrong). (ii) The two terms share their WORST critical point: the
uninformative channel that emits the unit marginal regardless of phone is an exact fixed point of the
likelihood and an exact zero of the unigram term simultaneously; only the skipgram part has any
gradient there. (iii) The statistics term's entire measured worth on this stream is the 0.0366 that
separates the random null 0.8946 from entry 5's 0.8580 — about eight percent of the 0.44 gap to the
memoryless ceiling — so it cannot bend a misalignment of that size whatever weight it carries. Add
that the alternating schedule has no convergence guarantee and that the weight has no admissible
label-free selector here, and the honest form of the proposal's good instinct is a CONSTRAINT rather
than a weighted term: phone-marginal posterior regularization, 39 dual variables and a closed-form
projection, now registered inside 1g.5.

**Proposal 4, Gromov-Wasserstein / graph matching — FUNDED ONLY as an initializer, inside 1g.3 and
1g.5, which is what the proposal itself recommends.** Two corrections. (i) In its identifiable form —
pinning the phone-side core from the text corpus instead of matching two free graphs — the equation
`A = M B M-transpose` is the fixed-core moment estimator already parked as `PLAN_1F.md` ladder entry
1, so proposal 4 and that entry are one method and 1g merges them rather than running both. (ii) The
obstacle is not only the many-to-one quotient the proposal names: on every pooled rung the real stream
BEATS a control that factorizes by construction on the matching objective itself (`SAE_1f.md`
conclusion 9, span 3.8-11.4 percent with the position term above 1) — coarticulation pushes correlated
errors onto the diagonal — so the graph objective is measurably INVERTED on this stream, and 1g.2
carries it as an audited objective precisely to size that. The proposal's claim that the
Gromov-Wasserstein objective value tracks matching accuracy without labels is exactly what that
inversion contradicts here, and it is the reason the objective is audited before it is trusted as a
label-free selector.

*Literature and measurement verification for proposal 4 (planner fan-out 2026-08-18).* Four further
corrections, the first three of which lower its ranking and the fourth of which is worth keeping
whatever happens to the method. (a) The published support for "the objective tracks accuracy without
labels" is a within-run convergence curve on a near-isometric equal-size problem, not a ranking across
misspecified regimes; in a synthetic ladder built on the real English phone bigram with our measured
coarticulation rate, the TRUE coupling was not the minimiser in a single one of sixteen
configurations, and a solver started AT the truth always walked downhill to a lower objective and a
worse map. That is the same shape as section 1a's likelihood finding, in a different objective. (b)
Entropic Gromov-Wasserstein and the factorization the proposal writes are DIFFERENT estimators: with
the squared loss the former reduces to maximizing a trace form over the transport polytope and never
forms the factorization residual at all; the residual form is ladder entry 1. The proposal inherits
neither theory cleanly, which is the second reason 1g merges it into entry 1 rather than running it as
written. (c) Multi-lag is refuted twice — on our own data the ratio of observed acoustic dependency to
what the phone side can explain GROWS with lag (about 4x at lag 1 rising to about 100x at lag 8,
because the text side's long-range dependency collapses to 0.001 nats while the acoustic side still
carries 0.107), so the extra lag graphs are nearly pure nuisance; and in the synthetic ladder adding
lags 2, 4 and 8 lowered accuracy from 0.880 to 0.677. (d) THE FINDING WORTH KEEPING REGARDLESS: the
representation of the graph dominates everything else. At matched ambiguity, cosine similarity of unit
CONTEXT PROFILES scored 0.880 against 0.536 for raw transition matrices and 0.037 for pointwise mutual
information — and pointwise mutual information is the form this program's own screens have been
reading throughout 1f. It is the only scale-free, quotient-invariant form of the graph, and any
second-order matcher built on this stream, entry 1 included, should use it.

**Proposal 5 — REVISED VERDICT 2026-08-18 after its dedicated verification returned: NOT FUNDED as
proposed (probability 0.05, range 0.03-0.08), retained only as the not-funded record in 1g.6 plus two
cheap replacement probes.** The single strongest argument against it is measured on our own banked
data and does not depend on any literature: across four pooled representations spanning ceilings 0.405
to 0.475, the unsupervised candidate does NOT track the ceiling — `seg12.5` has the best ceiling
(0.405) and gives 0.882, while `seg9`, whose ceiling is 0.070 WORSE, gives 0.828; `ubpe12.5` carries
the most phone information of any pooled rung (0.671 against 0.581) and gives the worst candidate
(0.944). Making the units carry more phone information has not once made an unsupervised matcher
better here. Honest caveat, because it matters: all four candidates sit at or near the content-free
null (0.828 to 1.001 against 0.8946), so this is an anti-correlation among near-chance solutions — a
strong caution, not a law. The framing number is the same one: everything ever tried on this stream
has recovered 7.6 percent of the span from the content-free null to the ceiling
((0.8946-0.8580)/(0.8946-0.4148)), and every operation proposal 5 offers moves the CEILING, not the
0.44 gap where the failure actually lives. The three operations' individual verdicts, the refutations,
and the two replacement probes are in 1g.6.

**Proposal 5, as originally adjudicated the same day and kept as dated history:** The premise is that k-means gives the wrong equivalence classes; on this stream
the measurements say the opposite about identity and agree about geometry-in-time. Unit identity
resolution is the best this program has measured on any inventory — raw substitutions 0.132, phone-
normalized mutual information 0.682 — while what pooling to the phone rate costs is boundary
placement: substitutions rise 0.132 to 0.230 and deletions 0 to 0.117. So "suspect the units" resolves
here into "suspect the boundaries", which is why 1g.6 funds the re-segmentation operation against a
measurement (the gold-boundary ceiling read, 1g.1 E3) and takes the merge operation only in its soft
row-tying form, merging being bounded above by the data-processing inequality. Part of the proposal is
already in flight and costs nothing extra: entry 7's full arm runs the reference pipeline's own
relabeling iterations on this bed, and `PLAN_1F.md` entry-7 stage-A ruling 3 already registers the
iteration-3-minus-iteration-1 delta as the read on what relabeling buys here.

**Sharpening of these five verdicts by the theory battery (2026-08-18, appended — the verdicts above
are not edited).** Proposals 2 and 4 both consume second-order statistics: matrix statistics matching
is a positional-unigram-plus-skipgram method and graph matching is a pair statistic by construction.
Finding 3 refutes that whole class algebraically rather than empirically — with the transition matrix
PINNED and infinite data, the pair moment map still leaves about V/2 - 1 free directions at V symbols,
driven to exactly zero only by adding the triple marginal. This is a stronger and more general statement
than the arm-level measurements those verdicts rested on, and it upgrades the reason without changing
the verdict. It also identifies why proposal 1 is the odd one out: full-sequence likelihood consumes
the triple statistic and every higher order implicitly, so it is the only proposal on the list attacking
an identified problem. Proposal 3, the hybrid, inherits both readings — its statistics-matching half is
covered by finding 3 and its likelihood half is not. Proposal 1 additionally gains the entry-(2) scope
amendment recorded in the theory battery: the logged likelihood/error anti-alignment is now attributed
to emission misspecification, reproduced from first principles, which licenses "that channel family was
too small", not "likelihood decipherment cannot work here" — while leaving the ban on selecting
anything by model likelihood not merely intact but strengthened.

## Candidate screen (2026-08-18): what else was generated, and what was refused

Nineteen candidates were generated across three independent lenses (acoustic-phonetic priors,
structural/identifiability constraints, and a sweep of the modern and classical decipherment
literature) and put through two independent screens — one for novelty against this project's own
discard record and mathematical validity, one for buildability against the artifacts and toolchain
actually on disk. Recorded here so the refusals are retrievable and are not re-proposed. **The
candidates were generated phone-centrically, before the text-side ruling was written into their
prompts, so each surviving entry above was re-read against the lexicon-free target rather than adopted
as returned** — 1g.8's phone-labelling problem is the one place that re-read changed a verdict.

| candidate | disposition |
|---|---|
| physics-signed manner/voicing sieve with quantile-calibrated cuts | ADOPTED into 1g.4 (five sharpenings) |
| two-sided spectral vowel/consonant split, polarity fixed by one acoustic bit | ADOPTED as 1g.4's PRIMARY form — the only zero-ledger-cost anchor |
| feature-factorized product decode | ADOPTED as 1g.4's output form (soft table, not hard mask) |
| vowel-space Procrustes onto the vowel triangle | MERGED into 1g.4 as a within-block sub-step, universal three-corner variant only |
| sonority-profile matching | MERGED into 1g.4 as extra descriptor columns; its expensive published components cut |
| word-parse cost as an objective and as a selection statistic | ADOPTED as 1g.2 objective (d) and experiment E3 |
| exact full-batch coverage-seeking output-distribution matching | ADOPTED as 1g.2 objective (e), audit column only |
| lexicon-constrained word-lattice decoding with recounting (the "amplifier") | ADOPTED as 1g.1 E5, the pricing experiment |
| repeated-token consensus / spoken-term discovery | ADOPTED as 1g.7 |
| minimal-pair frame matching | MERGED into 1g.7 as a free downstream diagnostic |
| synthetic phone atlas through the frozen encoder (two proposals, identical mechanism) | ADOPTED as 1g.8, phone reference arm |
| coarse-alphabet ceiling read at K = 39/60/80 | ADOPTED into 1g.6's downward sweep, as a CEILING measurement |
| articulatory-axis Procrustes on the codebook geometry | REFUSED — discarded embedding-isometry family; is proposal 4 in objective form, already measured refuted |
| one-to-one bijection SEARCH on a coarse alphabet | REFUSED — marginal mismatch (see 1g.6); identity was already unconstrained-and-correct in the failed run |
| rate-matched one-to-one likelihood with fertility DELETED | REFUSED — see 1g.5; contradicted by the insertion-forgiven ordering and by the largest measured misspecification term |
| homophonic beam search scored by a phone n-gram alone | REFUSED — discarded Nuhn-beam family, length-preserving key with no edit model, against 26.7 % invisible transitions |
| port of the published weighted-finite-state-transducer decipherment recipe | REFUSED as an implementation (no OpenFST/OpenGrm/pynini/pywrapfst in the environment, multi-week build); two structural details imported into 1g.5 |
| homophonic beam search over deterministic maps, lexicon-scored | PARTLY ADOPTED — its landscape probe is 1g.2 objective (d); the depth-500 beam itself is not funded |

**One screen-level finding worth keeping, because it explains the shape of this table.** Of the
nineteen, exactly four introduce an information source that is not a statistic of the unit sequence:
per-unit waveform measurement (1g.4), an externally-labelled synthetic acoustic reference (1g.8),
repetition equality (1g.7), and a hard word-list filter (1g.2 (d)). Every other candidate is a
different optimizer or a different loss over the same second-order statistics that theory-battery
finding 3 shows to be algebraically under-determined. That is the sorting criterion this phase should
keep using when new candidates arrive.

## Supervision-cost ledger

| resource | who uses it | status |
|---|---|---|
| frozen wav2vec2-Large-LV60 (60 kh LibriLight audio, zero transcripts) | every phase | already carried by the program |
| BPE (or character) tokenization learned on the raw unpaired corpus | PRIMARY arm, every phase | no lexical knowledge — the corpus tokenizes itself |
| n-gram language model over those tokens | PRIMARY arm, 1g.1/1g.2/1g.3/1g.5 | derived from the line above, no new cost |
| word n-gram + SPELLING lexicon (the corpus's own vocabulary) at DECODE | PRIMARY arm, 1g.1 E2 | not a pronunciation lexicon; readable off the text we already consume, so no new line item |
| pronunciation lexicon + grapheme-to-phoneme, to build the phone text T_phi | REFERENCE arm only | permitted, PLAN.md prior-knowledge table; the arm exists so its gap to the primary arm PRICES this resource |
| pronunciation lexicon a second time, to turn phone output into text | REFERENCE arm only | the extra touchpoint ruling 3 requires disclosing; the primary arm outputs text directly and does not pay it |
| phone-to-articulatory-feature table (39 rows of textbook phonetic class) | REFERENCE arm option only | DEMOTED 2026-08-18 by the text-side ruling; not on the primary path, no USER decision outstanding |
| speaker identifiers, for per-speaker normalization before clustering | 1g.6 replacement probe | permitted under the 2026-08-14/16 amendment; disclosed as a supervision cost when used |
| rule-based formant synthesizer rule tables (English formant and duration values) | 1g.8 only | NEW disclosed prior, not carried by any other phase; priced by the English-versus-language-neutral gap |
| MFA gold alignments and reference transcripts | 1g.1 oracle channel and degradation ladders, 1g.1 E3 gold-boundary read, 1g.1 E5 seed corruption, 1g.2 audits, 1g.4 kill-test, 1g.6 coarse-rung ceilings, 1g.7 purity read, every PER | EVALUATION ONLY |

**Hard rule for this phase, because the whole diagnostic design leans on a gold-fitted channel.** The
oracle channel, its degradation ladders and the gold-boundary stream are measurement instruments that
live inside 1g.1 and 1g.2. They never initialize a candidate, never select a checkpoint, seed or
hyperparameter, and never train anything reported as ours; the structural quarantine that entry 5
adopted — the training job receives identifiers only and no gold path, and a separate job is the only
one that opens gold — is carried forward verbatim into every 1g job.
