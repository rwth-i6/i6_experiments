# PLAN_1G — A simple weak starting point for the SAE loop

Sub-plan of `PLAN.md` Phase 1g. Rewritten 2026-08-19 after the user clarified the role of
Phase 1: it should provide a simple, weak initialization for the speech autoencoder (SAE) loop,
not solve ASR by itself.

This rewrite replaces the old live specification from 2026-08-19 onward. It does not rewrite any completed
gate or result. The pre-rewrite plan is preserved in
`archive/PLAN_1G_pre_rewrite_2026-08-19.md`. Detailed measurements and artifact records belong in
`SAE_1g.md` and the earlier phase logs; the durable scientific conclusions that determine the live
direction are summarized here.

## 1. Method definition

For one speech utterance, let `U=(u_1,...,u_T)` be the observed sequence of K-means audio units. Let
`Y=(y_1,...,y_N)` be an unknown text-symbol sequence: phones in the reference experiment and
characters in the main lexicon-free experiment. A fixed fitting language model `G_fit(Y)`, learned
from separate unpaired text, says which symbol sequences are plausible. No speech utterance is
paired with a text line during fitting. `Y` contains only emitting symbols; character word spaces
are non-emitting boundaries, not skippable characters.

The tested model is a small duration-aware hidden-state (semi-Markov) speech–text channel:

```text
unpaired text ──> fitting language model G_fit(Y)
                                  |
                                  v
text symbols Y ──> duration/state path A ──> audio units U
                         D_p(A|Y)          emissions B_s(u)
                              ^                    |
                              |                    v
                    frozen duration p       soft repair on
                                            unpaired speech

fitted channel A ──> decode U ──> pseudo-text ──> initial policy
fitted channel B ──> score S_B(U,Y) ────────────> initial scorer
```

For a state path `A`, let `s_t(A,Y)` be the text state active at audio position `t`. The model is

`P(U,Y,A) = G_fit(Y) D_p(A|Y) ∏_{t=1}^T B_{s_t(A,Y)}(u_t)`.

Here `B_s(u)=P(audio unit u | text state s)` is the speech–text channel to be learned, and
`D_p(A|Y)` is the duration model. More explicitly,
`P_B(U|Y) = Σ_A D_p(A|Y) ∏_t B_{s_t(A,Y)}(u_t)`. Repair maximizes
`Σ_i log Σ_Y G_fit(Y)P_B(U_i|Y)`, so it sums over both unknown text and alignment and never
requires a speech–text pair.

Phase 1g compares only two live channel shapes. Both require every text symbol to emit at least one
audio unit and use the same geometric duration `P(L=l)=(1-p)p^(l-1)`:

```text
one-state:       [ B_y ] --p--> [ B_y ] --p--> ...
                     |1-p
                     +-------------------------> next symbol

two-state:       [ B_y,1 ] --p--> [ B_y,2 ] --p--> [ B_y,2 ] ...
                     |1-p             |1-p
                     +----------------+----------> next symbol
```

The **one-state variable-duration model** has one emission row per symbol. Historical 1g.0 artifacts
call this “independent duration.” The **two-state model** has a first-position row and a
later-position row; it adds within-symbol acoustic order, not a minimum length of two. The old
one-unit-per-symbol model (`L=1`) is retained only as an already rejected diagnostic.

The fitting sequence is fixed. First tie every emission row to the observed audio-unit marginal and
fit the single duration parameter `p` from unpaired audio lengths under `G_fit` and its end-of-sentence
probabilities. Concretely, if `H_G(N)` is the probability that `G_fit` emits `N` symbols before EOS,
then an observed length `T` has probability
`sum_(N=1)^T H_G(N) choose(T-1,N-1) (1-p)^N p^(T-N)`. Choose the global maximum of the summed
per-utterance log likelihood on update audio; numerical code must verify that maximum on the full
`0 <= p < 1` interval, and an exact tie chooses the smaller `p`. Freeze that value rather than
refitting it after selection audio is added. At that fitted value, choose the one-state model if its 1g.0 dependence check passes;
use two states only if one state is decisively rejected and two states pass. Freeze the shape and
duration. Then initialize `B`: phone starts provide `Q(y|u)` and use
`B_y(u) proportional to m(u)Q(y|u)`, while the character start uses
`B_y(u) proportional to m(u)exp(r_yu)`, where `m` is the audio-unit marginal. Finally run exactly
0, 1, 2, or 4 soft re-estimation updates of `B` on unpaired speech. These updates use posterior
expected state–unit counts under the model above; they do not use transcripts.

The repair forward–backward recursion must preserve the complete `G_fit` sentence probability,
including BOS initialization and EOS termination. For phones, a path starts with
`G_fit(y_1|BOS)`, exits a symbol toward `y'` with `(1-p)G_fit(y'|y)`, and terminates after the final
audio unit with `(1-p)G_fit(EOS|y)`; in the two-state topology either sub-state may take that exit.
The emitting-symbol transition block is therefore generally sub-stochastic and must not be
renormalized to erase EOS mass. Posterior counts and the reported likelihood are conditioned on a
properly terminated path. Before a gate run, a tiny enumerated example must reproduce both the
sequence likelihood and state posteriors for the one- and two-state forms.

The sequence decoder returns
`Ŷ = argmax_Y [log P_B(U|Y) + λ log G_dec(Y) + β|Y|]`, with the frozen decoder language model,
language-model scale, and symbol penalty specified below. The baseline phone arm uses the add-one
phone bigram from complete `T_phi` as `G_fit` and the banked phone 4-gram as `G_dec`. They enter separate
computations and are never multiplied in one objective. The conditional H4-LM arm in 1g.2a changes
only the fitting-LM identity during its diagnostic and retains this decoder; within every arm `G_fit`
is fixed. For the scorer handoff, `S_B(U,Y)=log P_B(U|Y)/T` measures how well a fixed hypothesis
explains the audio without adding its language-model score. Channel A produces
pseudo-text for policy training. Channel B is initialized and fitted independently and supplies the
reconstruction score; it never trains on channel A's one-best pseudo-text. The policy-only and
scorer-only handoffs are tested before the fixed combined SAE pass. A mediocre standalone error rate
does not disqualify a channel if its audio dependence is real and the fixed loop improves on equally
treated controls.

The route order is: phones first to validate the mechanics; characters second as the first primary,
pronunciation-lexicon-free candidate; and the loop's exact BPE vocabulary only if the scorer
interface demonstrably requires it. The phone and character routes use the same model above, but
different symbol inventories and audio streams as listed in Section 4.

### Plain-language terminology

| Term | Meaning in this plan |
|---|---|
| audio unit | One K-means code; the phone route uses pooled segment tokens and the character route starts from adjacent-deduplicated raw codes |
| audio-unit marginal | Overall observed frequency of each audio unit, ignoring all text |
| text symbol | A phone in the reference route; a character or the loop's BPE token in the lexicon-free route |
| BPE token | A frequently occurring text fragment used as one token by the SAE loop |
| channel | The table `P(audio unit | text symbol)` plus a rule for aligning speech and text sequences |
| policy channel / channel A | The independently fitted channel whose decoded pseudo-text trains the policy |
| scorer channel / channel B | A second, independently initialized channel that scores policy outputs and never trains on channel A's one-best text |
| policy start | The audio-to-text model after ordinary cross-entropy training on Phase-1g pseudo-transcripts |
| score start | The initial audio-conditioned reconstruction score used by the SAE loop |
| content-free control | A map built and processed like the candidate but not tied to the utterance's speech content |
| random-map control | A content-free channel whose symbol assignments are random |
| pseudo-pair control | A channel fitted after deliberately pairing each audio utterance with unrelated, length-matched text |
| ESPUM seed | The banked Phase-1f neural statistics-matching seed |
| fingerprint seed | The banked Phase-1f deterministic position-based unit-to-phone seed |
| E5 | The phone experiment that measures outputs before and after fixed channel-repair steps |
| `theta_0` | The existing Phase-2 starting audio-to-text policy, fixed here at epoch 50 |
| p10 scorer | The existing Phase-2 audio-unit reconstruction model, fixed here at epoch 50 |
| decode-swap check | Replace an utterance's audio with matched donor audio, rerun decoding, and keep the original reference for evaluation |
| score-swap check | Keep a hypothesis fixed and compare its label-free channel score on its own versus matched donor audio |
| H4-LM | Conditional H4 fitting-language-model study; it is not the H6 character route |
| oracle | A label-fitted diagnostic that shows what the same representation and decoder can do; never a candidate |
| repair step | One fixed soft re-estimation update of the channel; no transcript is used |
| rank correlation | Whether a score orders better and worse transcripts correctly; reported with Spearman's statistic |
| held-out | Fit without the fixed evaluation fifth and score only on that untouched fifth |

## 2. North star and hard constraints

**North star.** Produce the simplest label-free, audio-dependent starting point whose fixed combined
SAE handoff beats equally treated content-free controls. Separate policy-only and scorer-only tests
first identify which component carries the benefit.

The following constraints are binding:

- **Unpaired inputs only.** Reported candidates may use unpaired speech and raw text. Phones are a
  disclosed reference route because their construction pays for a pronunciation lexicon.
- **Labels evaluate; they never build or select.** Transcripts, forced alignments, gold boundaries,
  gold-fitted channels, and gold-degraded channels may diagnose a method after it is fixed. They may
  not initialize a candidate or select its restart, decoder setting, checkpoint, or repair count.
- **Same treatment for candidate and control.** They receive the same representation, silence
  convention, data split, estimator updates, decoder grid, language model, and selection rule.
- **The decoder must be checked before the seed is judged.** A failed oracle check indicts the
  decoder or test, not every possible channel initializer.
- **The training score must reward audio content.** Model likelihood alone cannot select a run;
  this project has already observed likelihood/error anti-alignment in a misspecified model.
- **Do not teach both loop halves the same mistake.** In the primary combined test, channel A
  produces the policy's pseudo-text while a separately initialized and fitted channel B supplies the
  scorer; B never trains on A's hard one-best text. Different folds alone are insufficient because
  two copies can learn the same filler or cipher. A direct channel-A scorer is allowed only if it
  passes the reward-rank probe on the actual Phase-1 policy's rollouts against audio-free and
  same-speaker score-swap controls.
- **Keep uncertainty.** Preserve sequence alternatives, posterior probabilities, and confidence
  alongside the one-best pseudo-transcript.
- **Pin downstream compute before labels are opened.** Candidate and control use identical
  fine-tuning and short-loop budgets. Gold performance never selects how long either one runs.
- **Test on the 20.48-hour seed bed first.** More speech is funded only when a measured
  model-capacity or coverage limit, rather than weak initialization, is the bottleneck.
- **Use the GAN only as a positive control.** It is already strong enough to validate the handoff
  machinery, but it is not the desired simple Phase-1 method.
- **Completed results keep their actual scope.** The 1f (0.05/0.05) verdict, 1g.0's label-free
  rejection of the one-segment channel, 1g.0's gold-duration diagnostics, and 1g.4's spectral and
  hard-descriptor failures remain historical facts. The gold-duration cells in 1g.0 do not set a
  prospective candidate's duration; that parameter is fitted label-free. A 2026-08-19 verifier read
  corrected the supposed six-factor soft-product result to **not answerable**: the implementation
  tested seven alternative descriptors for one binary target, not the six independent memberships
  that were registered. That unrun route remains unfunded. New gates below apply only to future runs.
- **Lock a final evaluation.** Development results guide the campaign only after each run is
  frozen. Any final claim is repeated once on untouched test data after all method choices stop.

### Scientific conclusions carried into the live design

These are the compact, decision-bearing conclusions from completed project evidence. Their detailed
measurements and artifacts remain in `SAE_1a.md`, `SAE_1f.md`, and `SAE_1g.md`.

| Completed evidence | Conclusion and consequence for Phase 1g |
|---|---|
| On dev-other, the raw-unit memoryless oracle has PER `0.832 = 0.132 sub + 0.692 ins + 0.008 del`; `seg12.5` pooling changes this to `0.414 = 0.230 sub + 0.067 ins + 0.117 del`. A supervised context-kernel probe on the pooled path reaches 0.3565. | Raw units retain substantial phone identity but a lookup decoder fails mainly through over-segmentation; pooling repairs rate while trading away some identity. This motivates explicit duration and local context before changing the codebook, and retaining the raw stream for the first character experiment; it does not claim that gold boundary error has been isolated. |
| The one-unit-per-symbol channel failed the historical screen. On the prospective construction-only read, the one-state duration model is rejected and the two-state model is admissible at the label-free fitted `p=0.23560298`. | Duration alone is insufficient on the live stream; within-symbol acoustic order is required. The phone H3/H4 path therefore freezes the two-state topology rather than reopening a larger topology search. |
| Coarticulation hides 26.7% of real phone transitions, and the best banked aggregate starts remain close to the content-free control: ESPUM 0.8580 and fingerprint 0.8809 versus random-map 0.8946. | Matching marginals, rates, skipgrams, or positional fingerprints is not reliable evidence of recovered content on this stream. These methods remain weak initializers only; H4 must judge them after common repair with audio swaps and content-sensitive decoding, and H5 must judge the actual SAE handoff. |
| The positional-unigram inverse design is severely rank-limited: `sigma_min` is `5e-33` on raw units, exactly zero when usable position rows are fewer than the 500 unit columns, and `2e-17` even for the full-column-rank `brown100` case. The registered spectral and hard-descriptor routes also failed their gates. | The exact row-fewer-than-column failures cannot be repaired by more samples at unchanged shape, and the remaining tested rows fail through severe conditioning. Standalone positional inversion, spectral partitioning, and hard descriptors remain closed; the live route instead estimates `P(unit | symbol)` with a full-sequence channel. This closure does not extend to every possible sequence model. |
| Coarsening the inventory to `brown100` worsened the dev-other oracle PER to 1.152 and reduced graph correlation from 0.370 to 0.103, while the original 500-unit inventory supports the 0.3565 contextual probe. Across tested pooled representations, better oracle ceilings also did not consistently produce better unsupervised matches. | Unit identity is load-bearing, and representation surgery is not a substitute for a working channel or optimizer. Hard merging stays dropped; splitting, resegmentation, or smaller `K` is funded only after a content-bearing channel diagnoses the specific unit failure. |
| A prior misspecified channel exhibited likelihood/error anti-alignment. | Model likelihood may certify numerical health but may not select a seed, checkpoint, repair count, or method. The cause of the anti-alignment is not established by this observation; the live plan tests content with the frozen label-free own-minus-donor selector and reports accuracy only after choices freeze. |

The archived theory battery has a narrower status. [Allman, Matias, and Rhodes (2009), Theorem
6](https://doi.org/10.1214/09-AOS689) establishes generic identifiability for a standard finite HMM
under its assumptions. It rules out treating generic HMM non-identifiability as a blanket
impossibility argument, but it does **not** establish identifiability of this tied, duration-bearing,
potentially misspecified channel and fires no Phase-1g gate. The archived sample-sufficiency forecast,
pair-versus-triple nullity extrapolation, moment sample estimate, 10--20% repair-basin threshold,
language-model gain forecast, and anchor-count prediction came from uncommitted synthetic scratch
runs. They motivated bounded experiments but are not scientific conclusions and cannot discharge a
gate unless reproduced by a catalogued job.

### Why the old (0.05/0.05) gate is not reused

That gate asked whether a Phase-1 system was already substantially better than content-free
decoding. The clarified question is weaker: whether a seed contains reproducible audio information
and helps the SAE loop. A fixed 0.05 cliff mixes those two questions and can reject a real but weak
seed.

Future runs therefore use two decisions:

1. **Content decision:** is the candidate measurably more tied to the correct audio than matched
   content-free controls?
2. **Usefulness decision:** after fixed repair or SAE compute, does it improve while the controls do
   not?

The old margins remain reported for comparison. They no longer decide admission for future runs.

## 3. Status and priority queue

**Planner/verifier read: 2026-08-20.**

1. **H1--H3 and the baseline H2 engine are accepted; do not relaunch them.** H2 consumes one explicit
   deleted-silence boundary law and passes 23/23 channel tests including exact enumeration. H3's
   calibration and construction-population fingerprint, random-map, pseudo-pair, selected ESPUM
   seed-0/update-30,000 refit, and strict projections are complete. Distinct H1-LM/H2-LM artifacts
   below are conditional extensions, not permission to rerun these accepted baseline graphs.
2. **Repair the H4 calibration interfaces now.** Keep the completed non-soft repair work only after
   exact input/hash checks, replace the donor table with the shared support-compatible law in Section
   4, restore count-0 `Q`, and rerun the ten redefined soft trajectories. Finish the count adapter,
   direct-`Q` decoder, no-swap-aware gate, selector-freeze artifact, final-refit mode, and production
   decoder resource contract before selection starts.
3. **Complete the baseline bigram H4 assay before evaluation opens.** Calibration consumes H3
   `calibration` starts, repairs on exactly 6,414 update utterances, and reads the 890 selection
   utterances without fitting. Only after the selector freezes may final repair consume H3
   `final_refit` starts and all 7,304 construction utterances. Open the 1,112 evaluation utterances
   once after every baseline or triggered H4-LM choice is fixed.
4. **Run H4-LM only under its pre-evaluation trigger.** It is a conditional H4 fitting-context
   follow-up, not H6. Its **assay prerequisites** are mechanics, the positive control, donor-score
   calibration/correlation, and selector validity; they explicitly exclude the method-specific
   nonzero-count/update-health outcome and every evaluation-label gate. A failed or unresolved assay
   prerequisite is fixed first and does not trigger H4-LM. Once the prerequisites pass, call baseline
   H4 **pre-evaluation-ready** only when the frozen label-free selector assigns a safe nonzero repair
   count to at least one real start. Such an arm may skip H4-LM and open evaluation; this is not a
   held-out content verdict. Otherwise trigger H4-LM and resolve it or document measured infeasibility
   before interpreting that operating point as evidence against repair.
5. **Test the two SAE handoff paths separately after H4 passes.** First ask whether Phase 1g helps the
   policy, then whether it helps the scorer, before coupling both.
6. **Start the character route after one phone handoff is valid.** The combined phone loop may run
   in parallel or only if its assay still needs validation; it must not delay the primary route.
7. **Run the fixed combined test on the lexicon-free candidate.**
8. **Use exact loop BPE only for a demonstrated scorer-interface need.**
9. **Keep resegmentation, repeated-speech mining, synthetic speech, and adaptive restart or
   hyperparameter searches parked** until a failed direct experiment identifies which missing
   information they would supply. The fixed 29-draw character yield test below is a bounded
   preregistered experiment, not a search for the best labelled-development result.

### What is reused and what is rerun

| Item | Decision |
|---|---|
| 1g.0 channel-shape screen | Reuse the one-segment rejection and keep the gold-duration cells as diagnostics. The accepted H1 artifact completed the route-specific label-free update-audio read: one state is rejected and two states are admissible for both live routes; no further topology run is required. |
| 1g.4 spectral and hard two-class descriptor tests | Reuse. Both failed their registered accuracy gates and remain not funded. Do not reconstruct the missing uncertainty runs; they are not load-bearing. |
| 1g.4 six-factor soft product | **Not answerable and unrun.** The prior prerequisite claim used the wrong measurements; keep it parked unless separately revived. |
| 1f ESPUM and fingerprint artifacts | Reuse their fixed recipes, settings, and original outputs as provenance/transductive rows. They saw the evaluation audio and cannot enter the held-out gate unchanged. |
| Held-out ESPUM and fingerprint inputs | Build calibration rows on exactly 6,414 update utterances, use the disjoint 890 utterances only for label-free selection, then rebuild on all 7,304 construction utterances after choices freeze. ESPUM calibrates fixed seeds 0/1/2 plus the seed-0 collapse control and reruns only the selected seed/update; do not reopen a sweep. |
| 1f content-free controls | Reuse their construction rules, but rebuild them on the same construction-only population before the held-out comparison. |
| First E5 job | Completed as exploratory evidence; do not use it for a gate. |
| Corrected phone gate experiment | Implement the held-out assay below. Do not repeat or extend the flawed full-population E5 configuration. |
| Completed H4 calibration repairs | Reuse up to 71 non-soft controlled trajectories and four H3 trajectories only after their canonical count-0 `B` hashes and start provenance pass. Supersede and rerun the ten soft trajectories. A mismatching controlled trajectory is rerun alone from its verified canonical start; a mismatch in one of the four already-persisted H3 `Q`/`B` pairs is a provenance blocker and does not authorize an H3 relaunch. The original donor table is provenance only and is replaced for every production swap. |
| Higher-order phone fitting LM | Conditional new H4-LM work under 1g.2a. It cannot delay a pre-evaluation-ready bigram H4 arm and is not H6. |
| SAE handoff | New work. The old plan measured a channel in isolation and never tested the clarified Phase-1 purpose directly. |
| Character/BPE channel | New work, staged only after the phone mechanics are sound. |

**Rewrite verdict.** A from-scratch rewrite of the document is warranted. A from-scratch redo of
Phase 1g is not. Completed results remain; only invalid or unanswered comparisons and the
construction-only versions required by the new held-out gate are run.

## 4. Resources, notation, and anchors

### Data and fixed audio representation

- Unpaired speech seed bed: 8,416 LibriSpeech utterances, 20.48 hours—2,849 dedicated train
  utterances plus all 2,703 dev-clean and 2,864 dev-other utterances.
- Audio representation: frozen wav2vec2-Large-LV60 layer 15, 50 Hz, K-means with 500 units. The
  normalization, PCA, and K-means were fitted only on the 2,849 dedicated train utterances; dev
  audio was only assigned through that frozen transform. Segment pooling is per utterance and fits
  no corpus statistic. Thus the encoder/codebook/pooling do not have the evaluation leakage found in
  fingerprint and ESPUM. The old `UnitWordStreamJob` silence-proxy mask did use all 8,416 utterances,
  however, so it is fixture-only and is rebuilt from update audio below.
- Phone-reference stream: `seg12.5`, actually 9.77 segment tokens/s and 720,315 tokens.
- First character stream: raw K-means codes after collapsing adjacent repeats. It preserves the most
  temporal support and avoids committing the first trial to destructive pooling. Path feasibility
  and the shared duration are determined from unpaired sequence-length distributions before any
  character identity is fitted. The earlier 14.55-character/s comparison came from gold transcripts
  on a different population and is not a design input here.
- The full labelled diagnostic population is 2,703 dev-clean plus 2,864 dev-other utterances. The
  fixed evaluation fifth contains 540 clean and 572 other; these 1,112 utterances define the
  uncertainty of held-out reads. Removing them leaves 2,163 clean plus 2,292 other dev utterances.
- Preserve the original `holdout_split(..., seed=0)` evaluation fifth. Apply the same fixed 80/20
  helper with seed 1 separately inside each remaining split: update/selection counts are
  1,731/432 clean and 1,834/458 other. Only audio from the 2,849 dedicated train utterances is used;
  their transcripts remain quarantined. These utterances always belong to update, never selection
  or evaluation. Thus calibration uses 6,414 update and 890 label-free selection
  utterances. After label-free settings are frozen, refit candidates on all 7,304 permissible
  construction utterances—the 2,849 dedicated train plus 4,455 non-evaluation dev—and open the
  original 1,112-utterance evaluation fifth once. Do not rotate evaluation folds.
- Phone reference: 39 stress-free ARPAbet phones.
- Character route: inventory learned from the raw unpaired text; word space is a non-emitting
  boundary symbol.
- BPE route, if reached: exactly the vocabulary and tokenization consumed by the SAE scorer.

### Measured anchors

All values below are at their recorded operating point. PER is phone error rate; lower is better.

| Method or control | Dev-other PER | Interpretation |
|---|---:|---|
| Memoryless oracle map on `seg12.5` | 0.4148 | Acoustic-information anchor, not an unsupervised candidate |
| Supervised contextual probe on the same path | 0.3565 | Shows remaining local-context headroom |
| Random-map control | 0.8946 | Content-free control |
| Pseudo-pair control | 0.9239 | Content-free control |
| Banked ESPUM seed | 0.8580 | Historical transductive row; it saw evaluation audio and is not a held-out input |
| Banked fingerprint seed | 0.8809 | Historical transductive row; it saw evaluation audio and is not a held-out input |
| GAN reference | 0.214 selected / 0.168 oracle-best | Positive handoff control, not the desired method; only 0.214 is selection-honest |

At its historical transductive operating point, selected ESPUM improved over the stronger
content-free control by 0.0365 on dev-other and its audio-swap loss was 0.0466, versus 0.0091 for that
control. The near-matching clean/other effects suggest a weak signal, but the old run retained
neither per-utterance bootstrap data nor multiple swaps, and its training included evaluation audio.
That is why a construction-only rebuild and corrected test are required rather than either declaring
success or discarding the recipe.

### How future seeds are measured

There are two different swap tests and they must not be conflated.

1. **Evaluation-only decode swap.** For utterance `i`, replace its audio-unit sequence with a
   same-speaker, duration- and unit-rate-matched donor, rerun the decoder, and compare the result with
   utterance `i`'s original reference. `A_eval` is the resulting error increase over decoding its own
   audio. Report different-speaker donors secondarily, never as the primary content test.
2. **Label-free score swap.** Keep hypothesis `z_i` fixed and compute its length-normalized channel
   score on its own audio minus its score on the same matched donor audio. This is the selector used
   without transcripts; 1g.2 must validate it before deployment.

Construct the primary donors without text, decoded hypotheses, or labels. Build exactly two tables per
route/audio stream. The **selection table** has the 890 H1 selection IDs as both sources and candidate
pool. The **evaluation table** has the 1,112 H1 evaluation IDs as both sources and candidate pool.
Exclude the source itself. Update/construction IDs are not donor candidates: this keeps both sides of
each comparison in the same read-only role and keeps evaluation audio out of selection. Bind both
source and candidate-pool ID hashes. The earlier exploratory
792/890 coverage read used all 8,416 construction/evaluation-bed utterances as candidates and is not a
production-pool result; recompute and report coverage under these exact pools before decoding.

Let `T_s` and `C_s` be a source utterance's retained-unit count and number of silence-delimited retained
chunks. Require a same-speaker donor to satisfy `T_d >= T_s` and `C_d <= C_s`. Any hypothesis
with finite source score obeys `C_s <= |z| <= T_s`, so this audio-only law guarantees
`C_d <= |z| <= T_d` for every arm and decoder without conditioning donor choice on output length.
Among support-compatible peers, first retain those within plus or minus 5% of source encoder-frame
duration and sort them by absolute log audio-unit-rate ratio, then absolute frame-count difference,
then utterance ID. If that band is empty but a compatible peer exists, use the nearest compatible peer
under the existing sum of absolute log duration and unit-rate ratios and flag the fallback. Assignment
`s=0,...,9` uses candidate `s mod k` among the first `k=min(10, number of candidates)` rows. If no
compatible same-speaker peer exists, emit an explicit `no_swap` row with no assignments. Donor reuse
across sources is allowed.

Freeze each immutable replacement donor table and use it unchanged for every
reference, candidate, control, repair count, and decoder. `D_plain` and ordinary own-output metrics
retain every source. Fixed-text donor scores, `A_eval`, and `D_audio` use the identical precomputed
swap-eligible source population for every arm; bootstrap only those sources for donor quantities while
retaining the role-specific clean/other weights below. Report eligible, `no_swap`, fallback, match-distance,
and split counts and require every retained own/donor score to be finite. Report variation across the
ten assignments; resample donor assignment first and utterances second while recomputing aggregate
edit counts. Never clip an exact `-inf`, truncate or re-decode the fixed hypothesis, or add zero-
duration phones to rescue a donor comparison. The superseded table remains provenance only.

Let `E(c)` be held-out error under own audio: PER for phones, character error for character output,
and WER for the common downstream word interface. Compare the candidate jointly with the two
identically treated controls:

- `D_plain` = the smaller control advantage over candidate error.
- `D_audio` = the candidate's `A_eval` minus the larger control `A_eval`.

Use simultaneous paired 95% intervals over both controls, with 10,000 resamples and bootstrap RNG
seed 0. Bootstrap within clean and other. Every selection aggregation, including label-free selector
means and controlled validation, uses fixed weights 432/890 and 458/890; every evaluation aggregation
uses 540/1,112 and 572/1,112. Do not renormalize these target weights after `no_swap` removal: compute
each split's donor statistic on its eligible rows, report its coverage, and combine using the fixed
role weights. A split with no eligible source leaves the donor component unresolved. Do not require
two noisy split point estimates to share a sign. Instead, no split may show a statistically supported
effect in the opposite direction.

- Both lower bounds above zero: **content-bearing**.
- Either upper bound at or below zero: **negative for this candidate and protocol**.
- Any other result: **unresolved**, not negative.

A treated content-free start that itself becomes content-bearing after repair is a simpler success of
the estimator, not a failed control. Reclassify it as the candidate and compare its repaired output
with its own step-0 result and unchanged controls.

These tests say whether a seed contains audio information. Usefulness is measured separately in the
SAE handoff.

### Supervision cost

| Resource | Main lexicon-free route | Phone reference |
|---|---|---|
| Unpaired speech | Allowed | Allowed |
| Raw unpaired text | Allowed | Allowed |
| Character or BPE tokenizer learned from that text | Allowed | Not needed |
| Spelling dictionary made from text words | Allowed | Not needed |
| Pronunciation lexicon / G2P | Not allowed | Required and disclosed |
| Transcripts, alignments, gold boundaries | Evaluation or quarantined diagnosis only | Same |
| GAN outputs | Positive-control diagnosis only | Same |

### Evidence and artifacts

- Results log: `SAE_1g.md`
- Historical detailed specification:
  `archive/PLAN_1G_pre_rewrite_2026-08-19.md`
- Completed exploratory E5 job:
  `work/speech_llm/sae/seed_basin/SeedBasinJob.Zm3EuTveSGBL`
- Current E5 implementation:
  `recipe/2025-10-speech-llm/src/speech_llm/sae/seed_basin.py`
- External method and fitting-order motivation:
  `Lecture_UnsupervisedTraining_Seminar_UPV_26Sep24.pdf`; its synthetic results motivate H4-LM but
  are not project measurements or gates.

### Launch and evidence contract

Before a corrected gate job starts, persist the sorted utterance IDs and content hashes for update
(6,414), selection (890), final construction (7,304), and evaluation (1,112), plus the exact
`T_phi`, unit-stream, LM, and code revision identities. Assert the four roles are disjoint where they
must be and that their counts match Section 4. The frozen representation is anchored by
`AvStatesJob.c4Ak1rACchRC`, `QuantizeStatesJob.FWpGhC941JMi`, and
`SegmentPoolUnitsJob.IHRNqQfnxrQ3`.

Every H3 consumer must read those role lists and hashes from the accepted H1 artifact itself. Its
interface names an explicit `calibration` or `final_refit` mode: observed fitting IDs must equal the
H1 update set in calibration mode and the H1 construction set in final-refit mode. The selection set
is read-only and may supply label-free selection reads only; the evaluation set supplies neither a
fit nor a selection read. A caller may not redefine one role's expected IDs by passing that list
under a generic `construction_ids` argument.

Resource requests are part of this preflight. Accepted H1 completed, but its worker reached 141.52 GB
maximum RSS despite declaring 24 GB. Do not copy that request into H3/H4 or rely on the partition's
whole-node allocation to mask it. Measure a representative full-coverage path, request explicit
headroom, and retain the worker usage record with the job evidence.

Manager cleanup is also a launch prerequisite. An earlier 2026-08-20 verifier read found the
effective workspace `settings.py` at `JOB_AUTO_CLEANUP = False`, contrary to the standing project
rule; this was corrected to `True` at 15:31 CEST. Every future manager launch must still verify the
effective imported value from that manager's workspace.

The construction-only rebuilds inherit their algorithms from these canonical sources; the split and
complete-text corrections in this plan override only the old full-bed or subsampled inputs:

| item | canonical recipe |
|---|---|
| fingerprint and its controls | `config/sae_1f_entry3.py`; `speech_llm/sae/fingerprint_match.py` |
| ESPUM | `config/sae_1f_entry5.py`; `speech_llm/sae/espum_jobs.py`, `espum_match.py`, `espum_model.py` |
| phone 4-gram | `KenLMplzJob.0aJeN88X6EdW`; binary `CreateBinaryLMJob.hvZoC014xnIe` |
| matched H4-LM 2/3/4 recipe | `i6_experiments/users/wu/experiments/unsupervised_asr/phoneme_lm.py`; compile to the proper fitting automaton specified in 1g.2a |
| frozen local phone fixture | `speech_llm/sae/espum_jobs.py` and the historical `seg12.5` prefix-0 stream |

Every future gate job must save enough evidence to recompute its verdict: fitted duration and
topology read; initial `Q` and `B`; every state table at repair counts 0/1/2/4; seed/checkpoint and
selection records; one-best, alternatives, posteriors, and confidence; per-utterance edit and score
sufficient statistics; donor tables; all RNG seeds; bootstrap configuration; and final intervals.
Catalog the concrete job directories in `SAE_1g.md`. An aggregate report alone is not a gate
artifact. This requirement directly addresses the unpersisted uncertainty and missing-map problems
found in the audit.

Every prospective phone start must persist its original `Q(phone | unit)` before conversion, its
canonical `B(unit | phone)`, the audio marginal, probability floor, inventory order, generator inputs,
RNG seed, and hashes of both arrays. Count-0 local decoding reads that persisted `Q`; no consumer may
infer it from `B`. A completed repair trajectory is reusable only when its recorded count-0 `B` hash
exactly matches the canonical `B` regenerated from its bound start. A lossless count adapter must bind
one chosen table from a multi-count repair bundle to the decoder snapshot schema without changing any
number. The selector-freeze artifact must bind the selected initializer/restart, `G_fit` identity,
repair count, decoder family, `lambda`, `beta`, beam, donor table, selection sufficient statistics,
and every input hash; it is the only artifact allowed to authorize a construction-population final
refit.

## 5. Phases

### 1g.H — GPT audit handover: corrective implementation package

**Purpose.** Give the implementer one bounded entry point for every implementation change or rerun
created by the 2026-08-19 plan/evidence audit. This is not a new model family or an extra scientific
gate; the detailed normative specifications remain in the referenced subphases below.

**Approach.** Execute one ordered corrective phone path, followed only by the handoff and fallback
branches that its gates fund. Baseline H1--H3 are accepted. H4 calibration repairs use H3
`calibration` starts on the update role; only a selector-frozen final rerun uses H3 `final_refit`
starts on the construction role. H4-LM is a conditional fitting-context extension before H5, not a
renumbering of H6. H5 and H6 remain conditional follow-ups.

**Experiments.** Implementation TODOs, checked only after the corresponding artifacts and assertions
have been recorded:

- [x] **H1 — Freeze data and provenance.** Persist and hash the 6,414 update, 890 selection,
  7,304 construction, and 1,112 evaluation IDs; rebuild route-specific proxy-silence masks using
  update audio only; recompute the label-free duration/topology read without evaluation audio.
  Detailed specification: Sections 4 and 1g.0.
- [x] **H2 — Build the common channel engine.** Implement the one-state and two-state duration
  models, update-only duration fit, common probability floor, BOS/EOS-aware marginal forward–backward
  repair, marginal-over-path sequence score, occupancy-weighted two-state local postprocessing, and
  the registered decoder grid. Verify the repair likelihood and posteriors against exact enumeration
  before an integration run. Detailed specification: Sections 1 and 1g.1.
- [x] **H3 — Rebuild clean phone initializations.** Reproduce the historical fingerprint fixture
  for provenance, then build hash-pinned calibration/final-refit fingerprint, ESPUM, random-map, and
  pseudo-pair rows with no fitting or selection on evaluation audio. Add Sisyphus jobs that enforce
  update-only ESPUM training, read-only selection, and a no-selection fixed-update final rerun.
  Detailed specification: 1g.1.
- [ ] **H4 — Complete the corrected phone assay.** Replace the donor table by the common support-
  compatible audio-only table; restore count-0 `Q`; retain up to 75 hash-compatible trajectories,
  rerun any mismatching controlled trajectory, and rerun the ten redefined soft starts; add the
  direct-`Q` decoder, count adapter, no-swap-aware gate,
  decoder resource contract, selector-freeze artifact, and 7,304-ID final-refit mode. Then run the
  decoder check, label-free selection, and held-out content gate. Persist every per-utterance, donor,
  resampling, and uncertainty artifact. Detailed specification: Sections 4, 1g.1, and 1g.2.
- [ ] **H4-LM — Conditionally test fitting context before rejecting repair.** Under the trigger in
  1g.2a, build the exact H2-LM engine and run the bounded fixed-duration matched-2/3/4 diagnostic.
  Then give the resource-feasible matched 4-gram its own coherent H1-LM duration/topology refit and
  complete H4 gate; only a label-free selection between the baseline and that full arm may advance.
  This is not required when baseline H4 is pre-evaluation-ready. Evaluation labels may never choose
  an order. Detailed specification: 1g.2a.
- [ ] **H5 — Build the SAE handoff, only after H4 passes.** Freeze independently initialized
  channel A and channel B; run policy-only and scorer-only checks before any combined loop. Detailed
  specification: 1g.3.
- [ ] **H6 — Run the character fallback, only after one separate H5 phone handoff is valid.** Run
  the fixed character LM/spelling decoder, positive control, score check, 29 random starts,
  pseudo-pair auxiliary, and separate/combined handoffs. An optional combined phone loop may run in
  parallel but cannot delay this branch. Detailed specification: 1g.5.

Do **not** create work for the following audit corrections: the flawed full-population E5 replay;
missing spectral permutation/bootstrap reconstruction; another Fiedler, silence-precheck, or hard
descriptor run; the unexercised six-factor soft product; another gold-duration sweep; a full-bed
ESPUM retrain; or automatic BPE, resegmentation, repeated-speech, synthetic-speech, or soft-energy
arms. Their funding status is already settled below.

**Gate.** H1--H4 are accepted only when the frozen fixture reproduces, all split/provenance and
construction-only assertions pass, the positive controls make the phone assay valid, and the
catalogued artifacts suffice to recompute every gate. A failed assertion makes the affected result
unresolved; it does not license an additional initializer or hyperparameter search. A triggered
H4-LM must resolve before evaluation opens; otherwise it cannot alter the H5 input. H5 and H6 retain
their own scientific gates.

**Status.** **2026-08-20 — H1--H3 accepted; H4 calibration repair complete but its production
interfaces require the frozen corrections above.** `Phase1gH1Job.HbxKiuBTJ8aN` fixes the baseline
two-state phone topology at `p=0.23560298`; baseline H1, the H2 engine/timing grid, and H3 calibration
and final graphs must not be relaunched. H4 preparation and all 85 bigram repair trajectories at
counts 0/1/2/4 finished, but no selection decode, selector freeze, final repair, or evaluation ran.

The old donor table lacks the hard support conditions of the accepted duration topology and is
superseded for production. The controlled start bundle also omitted original count-0 `Q`, and the ten
soft starts used a B-space flooring rule whose `q=1` endpoint is not the canonical reference. Build
immutable replacement donor/Q artifacts, rerun the ten soft trajectories, and reuse up to 75 others
only after exact start and `B`-hash verification. A mismatching controlled row is rerun alone;
an H3 `Q`/`B` mismatch blocks for provenance. Section 4 fixes the replacement donor roles and requires
a new production-pool preflight.

The next run is H4 calibration decoding/scoring on the 6,414/890 roles, followed by an immutable
selector choice and construction-only final refit. H4-LM is planned but unrun and fires only under
1g.2a before evaluation labels open. H5--H6 remain gated on the resulting valid H4 output. Before a
manager starts, satisfy the cleanup and measured-resource prerequisites in the launch contract.

### 1g.0 — Choose the smallest channel shape that the data do not reject

**Purpose.** Rule out a channel that is too simple, then define a label-free check for the first
duration model.

**Approach.** Compare adjacent-unit dependence in the observed audio stream with what increasingly
flexible channel shapes can represent. The measured audio dependence and the one-segment ceiling are
label-free. The completed one-state variable-duration and two-state cells were evaluated at within-symbol
rates obtained from gold boundaries; those cells are useful diagnostics but cannot set a prospective
candidate's duration. The prospective choice recomputes the observed dependence on the 6,414 update
utterances only; the old full-dev read included evaluation audio and cannot choose a held-out model.

**Experiments.** The completed screen compared one segment per symbol, independent variable duration,
and two ordered sub-states per text symbol. For each live route—`seg12.5`/phones and raw
adjacent-deduplicated units/characters—fit its shared duration on update audio as defined in Section
1, recompute plug-in and Miller–Madow lag-1 dependence on those same update sequences, and evaluate
the one- and two-state analytic bounds exactly at that fitted `p`. Do not interpolate the historical
grid and do not repeat the gold-boundary duration screen. Selection and evaluation audio do not
choose the shape.

**Gate.** The historical gate calls a shape admissible when measured lag-1 dependence divided by its
allowed dependence is at most 2 under both the plug-in and Miller–Madow estimates. Both ratios above
2 decisively reject the shape; one on each side makes it indeterminate. Shuffled and cross-utterance floors are reported but not
subtracted. For prospective work, the one-segment verdict can stand directly; a duration-bearing
shape is admissible only at its independently fitted label-free duration. A larger gold-boundary
diagnostic value cannot rescue it. Choose the one-state variable-duration model if it passes.
Consider two states only after the one-state model is decisively rejected; an indeterminate smaller form leaves the
shape unresolved rather than licensing a jump to the larger model.

**Status.** **Historical screen and prospective route-specific read closed.** The one-segment channel
is rejected in every tested stream/text cell; that conclusion is label-free. At the historical
gold-derived duration operating points, the one-state variable-duration model failed on every pooled
stream and was split-dependent for raw characters, while the two-state form passed all dev-other
cells; on dev-clean five cells passed, two were indeterminate, and one was rejected. Those historical
reads do not choose duration. The accepted construction-only H1 read fits each route on the 6,414
update utterances: one state is decisively rejected and two states are admissible for both live routes.
Freeze the baseline phone value `p=0.23560298`; see `SAE_1g.md` Approach 6. No further baseline
topology run is required; a triggered deployable higher-order `G_fit` uses the distinct H1-LM read in
1g.2a.

### 1g.1 — Phone reference: can the existing weak seeds be repaired?

**Purpose.** Test the complete estimator and decoder cheaply, and determine whether construction-only
versions of the two 1f seed recipes are better starting points than content-free controls after a few
fixed updates.

**Approach.** After the label-free one-segment rejection, fit the shared phone duration without
labels and read both the one-state variable-duration and two-state dependency curves at that value.
Use the one-state form if it is admissible. Only if it is decisively rejected may a passing two-state form
be used. An indeterminate smaller form, or failure of both forms, leaves the phone mechanics test
unresolved and does not judge the seeds. First reproduce the frozen 1f fixture with the original
construction/evaluation IDs, preprocessing, map, and tie rule; only this diagnostic row must match
the banked edit counts exactly. Separately construct a prospective label-fitted reference
`Q(phone | unit)`. During calibration it is fitted on update counts and scored on selection data;
after all label-free settings are frozen, it is refitted on all non-evaluation dev counts and scored
once on the evaluation fifth. The reference uses only the 3,565 update dev utterances during
calibration and all 4,455 construction dev utterances for the final fit; the dedicated 2,849 train
utterances contribute no labels. It is a positive control, not a candidate. If its preprocessing or
map differs from the frozen fixture, report a fresh local baseline rather than demanding 0.4148.
Never fit and score a channel on the same utterances.

Every prospective start first supplies and persists `Q(phone | unit)`. Multiply it by the measured
audio-unit marginal, apply the common floor, and normalize each phone row to obtain and persist
`B=P(audio unit | phone)`. Store both array hashes, the marginal, floor, inventory order, inputs, and
seed; count-0 `Q` is never reconstructed from `B`. The complete unpaired phone text `T_phi` supplies
the baseline add-one fitting bigram; the fixed Phase-1a/1f phone 4-gram supplies the sequence decoder
prior. Construction transcripts supply neither to
a real seed or control. Every prospective marginal, per-unit projection, and pseudo-pair count is
estimated without the fixed evaluation fifth: first on the 6,414 update utterances, then on all 7,304
construction utterances only after label-free settings are frozen. Before channel repair, tie all phone emission rows
to the audio-unit marginal, fit the single shared duration parameter from the update audio lengths
under the unpaired phone language model and its end-of-sentence probabilities, then freeze it for the
reference, both real seeds, and both controls. The gold-derived 0.3164 repetition rate and mean
duration 1.463 used by the first E5 rehearsal are diagnostics only. After the shape check, only the
selected model's one or two `P(audio unit | phone-state)` tables move during soft re-estimation. Do
not refit baseline duration on the 890 selection or final 7,304-construction populations. A distinct
coherent H1-LM arm under 1g.2a may refit duration on the same 6,414 update IDs only; that is not a
baseline rerun.

Rebuild the proxy-silence convention before these fits. On the 6,414 update utterances only, apply
the canonical 0.2-second edge-enrichment statistic and weighted two-means high-cluster rule separately
to `seg12.5` and raw units. Freeze each resulting unit mask, use it to construct every calibration,
selection, final-construction, and evaluation stream for that route, and persist the masks. The
historical full-bed mask and `UnitWordStreamJob.eIxgmMh99RSE` remain only in the frozen-fixture row.
Candidate, reference, and controls must share the new route mask.

Convert every starting method to that common channel representation:

- **Frozen fingerprint fixture.** Recompute the original full-bed map and assert its recorded PER,
  substitution, insertion, deletion, symbol-count, and silence-unit values. This verifies provenance
  only; it saw the evaluation audio and cannot enter the held-out gate.
- **Held-out fingerprint.** During calibration, run the fixed-reg-0.1 deterministic recipe on the
  6,414 update utterances. After all choices are frozen, rerun it on all 7,304 construction
  utterances, then apply the common marginal-and-row-normalization conversion. Preserve the
  canonical fingerprint, silence, hard-argmax, and tie rules; discard its old stride-80 text sample
  in favor of the complete `T_phi` required here. Its operating point is new.
- **Frozen ESPUM artifact.** The saved checkpoint
  `EspumMatchTrainJob.lALR9ldNG8f1/output/model.pt` is reported as a transductive provenance row only;
  training used all 8,416 utterances, including the fixed evaluation audio.
- **Held-out ESPUM.** Preserve the original full-loss model, optimizer, 40,000-update ceiling,
  2,000-update reads, seeds 0/1/2, and bigram-only seed-0 collapse control. The only input corrections
  are the split and complete text: train on the 6,414 update utterances, never the 890 selection
  utterances, and read every `T_phi` line rather than stride 400/cap 100,000. Within each full-loss
  seed choose the first checkpoint attaining its minimum weighted phone-LM perplexity on the 890
  selection utterances; choose the seed by the same strict metric. Then rerun that seed from scratch
  on all 7,304 construction utterances for exactly the selected number of updates, with no further
  checkpoint choice. Average that final checkpoint's posterior over construction occurrences of
  each audio unit and apply the common conversion. Retain all three calibration curves and the
  bigram-only control. No new hyperparameter or seed search is allowed. The job interface must take
  disjoint H1 update and selection roles: update batches draw only from update IDs, while selection
  reads draw only from selection IDs. The final rerun has no selector and stops at exactly the frozen
  update count. The historical `EspumMatchTrainJob` does not enforce these roles and must not be wired
  unchanged into H3.
- **Controls.** Rebuild the canonical marginal-random map and proportional pseudo-pair rules on the
  same calibration/final populations. Random-map seeds are 1000 onward. Pseudo-pairs use the
  canonical nearest-length window 16, allow text-line reuse, and use pairing seed 0; their unit/text
  positions are aligned proportionally. Use the complete `T_phi` and the same silence convention.
- **Probability floor.** Add `1e-8` to every cell and renormalize every emission row for the
  reference, both real seeds, and both controls after initialization and after every M-step. It is
  not a control-only smoothing choice. Report corpus log likelihood as total log probability divided
  by total observed audio units; the M-step itself uses the unnormalized sum over utterances.
- **Two states, if selected.** For repair count 0 and the decoder check, copy the initial phone distribution to
  both states without perturbing it. Immediately before repair step 1, copy the same rows again and,
  for every phone `y` and unit
  `u`, draw one fixed `+1` or `-1` value with equal probability, using RNG seed 0. Call it `r[y,u]`.
  Multiply state 1 by
  `exp(log(1.1) r[y,u])` and state 2 by `exp(-log(1.1) r[y,u])`, then normalize each row. This is the
  positivity-preserving, symmetric “10% perturbation”; it is never swept.

For the completed H4 recovery, regenerate and provenance-bind `Q` for the 71 non-soft controlled
starts (reference, retain/redraw maps, and marginal-random maps) from their pinned inputs and seeds;
import and hash-bind the already persisted `Q` and `B` from the four H3 calibration starts. Forward
conversion must reproduce each retained count-0 `B` hash exactly before its repair trajectory is
reused. If a regenerated controlled start misses its old hash, supersede and rerun only that
trajectory after the pinned construction checks pass. If one of the four accepted H3 pairs fails its
own stored `Q`-to-`B` check, stop for planner/verifier provenance review; do not relaunch H3. The ten
old soft trajectories do not satisfy this interface and are replaced under 1g.2.

Measure every method's iteration-0 output under the common decoder. Do not inherit the ESPUM 0.8580
or fingerprint 0.8809 headline after projection and decoder changes.

At step 0, the **local decoder** reads each start's `Q(phone | unit)` directly with its fixed tie
rule. Only the frozen fixture must reproduce the complete banked 1f edit counts. The prospective
reference must reproduce its own construction-fitted majority map and gets a fresh held-out local
baseline; every real seed and control is also reported at its newly measured operating point. After
repair, the local decoder uses the one-state form's single row or averages the two-state rows using
their expected occupancy under the frozen duration transitions: weights `1-p` for the always-visited
first-position row and `p` for the later-position row. It then chooses the phone maximizing
`P(audio unit | phone)` times that phone's frequency in unpaired text. For every local phone read,
delete the fixed proxy-silence segments, use first/lowest sorted phone ID on an exact argmax tie,
collapse adjacent equal phones only within each silence-delimited chunk, then concatenate chunks.
This is the frozen Phase-1f convention; do not collapse across a removed silence boundary.
The production implementation therefore requires a direct-`Q` decoder path; the repaired
`B`-times-text-prior helper is not a substitute at count 0.

The **sequence decoder** uses
the same unperturbed channel rows
at repair count 0, so its comparison does not also perturb the channel. Counts 1/2/4 use the fixed
symmetry break and subsequently learned state rows. Sequence decoding uses the banked KenLM phone
4-gram at
`work/i6_core/lm/kenlm/CreateBinaryLMJob.hvZoC014xnIe/output/lm.bin` and beam search. Beam prefix
scores sum probabilities over duration/state paths that yield the same text prefix; a Viterbi-max
alignment is not a second decoder candidate. During baseline sequence decoding, this 4-gram
**replaces** the across-phone bigram used during repair; in H4-LM it likewise replaces whichever
fitting order that arm used. Only the within-phone duration transitions remain. The fitting and
decoder language models are never multiplied together. Its language-model scale
and per-phone insertion penalty use the new fixed grid
`lambda={0.5,1,2,4}` by `beta={-2,-1,0}`. This grid contains the natural unit scale and neutral
penalty plus the only nearby project anchor, Phase 1d's cross-decoder `(2,-1)` operating point; no
banked phone grid exists. The label-free selector in 1g.2 chooses a pair on the 890 selection
utterances. Test beams 64, 128, 256, and
512 on update audio
without transcripts. Freeze the smallest beam for which the one-best output is unchanged on at least
99.9% of utterances and the best score changes by less than `1e-4` per audio unit at the next larger
beam. If no pair stabilizes, the sequence-decoder result is unresolved. Oracle edit counts are opened
only after the beam is frozen.

A deleted proxy-silence run is a duration boundary in the common channel law. The first retained
observation after such a gap must begin a new phone duration in repair forward--backward, fixed-text
marginal scoring, and sequence decoding alike. The phone-language-model history is retained, so an
identical phone may occur again across the gap; only a single duration path is forbidden from
bridging it. This extends the frozen local-decoder rule that never collapses equal phones across a
deleted-silence boundary and prevents repair and evaluation from using different path laws.

Retain the top eight alternatives alongside the one-best. This is an output-size choice, not a search
or decoder operating point: one-best and confidence are computed against the complete surviving beam,
so truncating the persisted list to eight does not change decoding, selection, or the reported
confidence. Keep the complete surviving-prefix log mass so the retained posterior evidence remains
interpretable.

Read every line of the canonical `T_phi` artifact exactly once when estimating phone statistics; do
not use a stride, cap, or prefix. Run repair counts 0, 1, 2, and 4—a short logarithmic schedule—and
forbid adding counts after labels are read. Generate and retain every fixed output now; 1g.2 freezes
the selector before any count or decoder setting is deployed.

**Experiments.**

1. **Decoder check.** Frozen-fixture reproduction, prospective held-out reference channel,
   controlled reference degradations, and treated controls under local and sequence decoding. The
   frozen fixture must reproduce the complete recorded edit-count baseline, not merely a rounded
   PER; a changed prospective reference gets its own freshly reported local baseline.
2. **Repair curve.** Let `q` mean the fraction of unit identities whose reference phone is retained:
   `q=1` is the reference and `q=0` redraws every phone label. At each level retain exactly
   `round(q K_live)` unit identities, sampled uniformly without replacement. Draw every other label
   independently from the complete-`T_phi` unigram prior; a redraw may equal the reference label, so
   report realized map correctness as well as configured retention. For level index `j` and draw
   `s`, initialize NumPy with `SeedSequence([0,j,s])`; use draws 0--4 initially and 5--19 only for
   an extended level. Then apply the same
   marginal-and-row-normalization conversion used by the real starts; the observed audio-unit
   marginal remains fixed. The second damage family is the canonical `Q`-space mixture defined in
   1g.2; it is not a direct mixture of stored emission rows.
   Use
   `q={0,.05,.10,.15,.20,.30,.40,.60,.80,1}` with five independent maps initially. For each repair
   count, a transition bracket is the adjacent `q` pair whose observed gate-pass fractions lie on
   opposite sides of 0.5; if no pair does, use the two levels with fractions closest to 0.5. Extend
   the union of those levels to 20 maps before reading the real seeds. For each fixed repair
   count, report the fraction of maps ending below that prospective row's freshly measured local-
   reference PER, with a binomial interval. The threshold is 0.4148 only if the prospective row is
   fixture-identical. This is a descriptive phone repair curve, not a character-restart probability.
3. **Real inputs.** Construction-only ESPUM, fingerprint, random-map, and pseudo-pair starts at
   repair counts 0, 1, 2, and 4. Compare the full trajectories, not only their endpoints. Report the
   original full-bed ESPUM and fingerprint artifacts separately as transductive provenance rows;
   exclude them from the held-out gate.
4. **Boundary diagnosis.** Gold phone boundaries and restored silence only as quarantined
   diagnostics if the oracle check leaves unexplained error.

**Gate.**

- The test is valid only if the frozen fixture reproduces the local edit counts exactly and the
  prospective held-out reference's `D_plain` lower bound against the treated controls exceeds zero.
  Otherwise repair conclusions are **unresolved** and preprocessing or decoding is fixed first.
- The sequence decoder becomes **eligible** only if the upper paired confidence bound on
  `PER(sequence)-PER(local)` at the reference is at most `+0.01`, and the lower bound on its equally
  weighted mean PER improvement over local decoding across the predeclared non-endpoint `q` levels
  exceeds zero. For the first condition, a lower bound above `+0.01` is negative; overlap is
  unresolved. For the second, an upper bound at or below zero is negative; overlap is unresolved.
  An unresolved decoder is not deployed and does not close channel estimation. A negative result
  rejects only this decoder/grid. Even after eligibility, the frozen label-free selector—not PER—
  chooses local or sequence decoding for a real seed. The 0.01 safety margin is one fifth of the
  historical 0.05 relevance unit and tiny against the roughly 0.48 reference-to-null span.
- After 1g.2 freezes the selector, a real seed advances when its selected repaired output is
  content-bearing by Section 4 and beats treated starts. If a treated content-free start itself
  becomes content-bearing, adopt that simpler estimator-from-scratch result instead of calling the
  control “failed.”
- The controlled repair curve explains the result; it does not substitute for running the actual
  seeds and controls.
- This verdict applies only to the phone reference.

**Status.** **2026-08-20 — H4 calibration repair is partially reusable; decoding and the gate remain
pending.** All 85 baseline bigram trajectories finished, but the ten soft trajectories are
superseded by the canonical `Q` definition above. Reuse up to 75 after exact `Q`/`B` and manifest
checks, with the mismatch handling above. No selection decode, selector freeze, construction refit,
or evaluation has run.

The first E5 job remains exploratory, non-decisive evidence. It fit and scored the same 2,864 dev-
other utterances under different preprocessing, gold-derived mean duration 1.463, a hard 0.9-smoothed
map, fixed-stride text, and bigram posterior-argmax decoding; it ran neither real seed nor treated
control. Its configured retention 0 moved PER 1.0109 to 0.8409, while retention 1 moved 0.4865 to an
LM-selected 0.4589 and then 0.6699 at step 30. These engineering observations fire no gate, and the
historical 1f verdict is unchanged.

### 1g.2 — Check that the training score rewards speech content

**Purpose.** Ensure that the score used to fit and select a channel points toward better
speech-dependent transcripts rather than merely toward fluent or frequent text.

**Approach.** Preserve the original fixed evaluation fifth: 540 dev-clean and 572 dev-other
utterances from `holdout_split(..., seed=0)`. Apply that helper again with seed 1 inside the remaining
four fifths. Its fixed 80/20 split gives 1,731 clean plus 1,834 other **update** utterances and 432
clean plus 458 other **label-free selection** utterances. Repair updates maximize full-sequence log
likelihood per audio unit only on the 6,414 update utterances: the 3,565 dev-update utterances plus
the 2,849 dedicated train utterances whose transcripts remain quarantined. Repair count, decoder
setting, and restart are chosen only by label-free scores on the 890 selection utterances. After
those choices are frozen, refit candidate and control channels on all 7,304 permissible construction
utterances and open the fixed evaluation fifth once. The label-fitted reference is the exception: it
uses labelled counts from the 3,565 dev-update utterances during calibration and all 4,455
non-evaluation dev utterances for its final fit. There is no rotating evaluation.

Final-fold likelihood is only a prospective health read after every output is fixed; it never
supplies a gradient, update, or checkpoint choice. The selection score is the average per-unit
channel-score advantage of a decoded hypothesis on its own audio over the same-speaker matched donor
audio; higher means more audio dependence.

Calibrate and freeze both scores only on controlled reference degradations and independent null maps.
The real ESPUM, fingerprint, and later character candidates are prospective tests, not calibration
points. For a deployed seed, choose repair count, decoder settings, and restart without labels and
open final error only on the final evaluation fifth. Controlled calibration may read its quarantined
reference labels only after every output is fixed; those labels may never choose a real seed's
restart, repair count, or decoder setting.

The production interfaces are explicit. Calibration repair accepts only an H3 `calibration` start,
fits exactly the 6,414 update IDs, and never updates on selection audio. A lossless count adapter
exports one of counts 0/1/2/4 into the decoder schema while binding its source array, inventory,
duration, boundary law, role, count, and hashes.

The sequence-decoder resource contract is deterministic and role-specific. Retain the accepted H2
32-way sharding: for each H1 role, chunk `j` receives canonical role IDs `j::32`. Among every adapted
channel/count table that the role may decode, deduplicate identical array hashes and sort by
`(mean emission-row entropy, canonical arm name, repair count)`. Select indices `0`, `(n-1)//2`, and
`n-1`, deduplicating those
indices when `n<3`. For each chosen table, run all
48 `(lambda,beta,beam)` cells on the role's longest retained-unit
utterance, chosen by descending `(retained-unit count, utterance ID)` exactly as in H2. Rerun the cell
with maximum elapsed time and the cell with maximum RSS—once if they coincide—on the actual 32-way
chunk with the largest total retained-unit count, tie by lower chunk index. The selection contract may
read only the 890 selection IDs; construction and evaluation contracts are created only after the
selector freezes and use only their own role IDs. Persist all probe IDs, table identities, settings,
chunk contents, duration/topology, `G_fit`/`G_dec` identities, code hashes, elapsed time, maximum RSS,
and queue limits. Production requests must be at least
1.5 times the maximum measured elapsed time and 1.5 times the maximum measured RSS, rounded up to
whole hours and GiB; failure to fit the
declared queue limits is resource-unresolved. Every production chunk binds this contract and retains
its actual usage. The accepted H2 timing grid supplies probe anchors but cannot substitute for these
channel- and role-bound measurements or select a scientific setting.

After all calibration outputs exist, one immutable selector artifact freezes the
initializer/restart, fitting-LM identity, repair count, local/sequence decoder choice, `lambda`,
`beta`, beam, donor table, and score evidence. Final-refit repair starts from the corresponding H3
`final_refit` artifact, applies exactly that frozen count to all 7,304 construction IDs, and has no
selector, selection-label, or evaluation reader.

**Experiments.**

1. Use the 1g.1 `q` grid under two damage mechanisms. The map family is the exact-size
   retain/redraw construction and draw schedule defined there. For the deterministic soft family let
   `pi_ref(y)=sum_u m(u)Q_ref(y|u)` and
   `Q_q(y|u)=q Q_ref(y|u)+(1-q)pi_ref(y)`. Convert every `Q_q` through the same canonical `Q`-to-`B`
   operation and floor as every other start. Thus `q=1` is exactly the reference and `q=0` is class-
   independent. Persist both tables and rerun the ten soft trajectories; the old B-space trajectories
   are superseded. It has one row per `q`, not five fake replicates. Retain outputs at repair counts
   0, 1, 2, and 4 for every trajectory; score calibration must cover the exact checkpoint choice it
   will later make.
2. Add the pseudo-pair control at pairing seed 0 and 20 marginal-random maps with seeds 1000--1019.
   Seed 1000 is the named random-map control; the remaining maps estimate the null spread. Twenty
   probes roughly the upper 5% null tail; inference nevertheless uses the simultaneous maximum-null
   statistic rather than treating the observed maximum as a new threshold.
3. Measure rank agreement between each score and `-error` globally and in the predeclared local
   operating band with starting PER 0.80–0.93, which contains the actual phone seeds and controls.
4. Within each controlled trajectory, test whether the frozen selector orders repair counts sensibly.
   Report its rank agreement, its error relative to the best of counts 0/1/2/4, and its change from
   count 0. Starting from the reference channel, also check repair steps 1, 2, and 4 separately for
   drift.
5. Treat independently generated maps or training trajectories as the independent units. Repeated
   checkpoints from one trajectory stay in one cluster during resampling and never count as extra
   independent examples.
6. The own-minus-donor contrast is a **selection** score, not an update objective. Repair updates
   continue to maximize construction-fold likelihood. If construction likelihood passes as an
   update score but fails as a checkpoint selector, the validated own-minus-donor contrast is the
   sole backup selector. If likelihood fails the update-health checks, repair is unresolved; no
   contrastive update is invented after labels are read.
7. After the score definitions are frozen, report ESPUM, fingerprint, and control outcomes without
   refitting the selector.

**Gate.**

- The reference channel must beat the strongest predefined null under a simultaneous 95% interval
  over all null maps and both damage families.
- Globally and within the 0.80–0.93 local band, the lower 95% bound of Spearman rank correlation
  between score and `-error` must be above zero. Resample map/trajectory first and utterance
  second.
- For each reference-start repair count, an upper paired 95% bound on PER increase no greater than
  0.05 is safe. A lower bound above 0.05 is negative and removes that count. An interval overlapping
  0.05 is unresolved and makes that count non-deployable without closing the score. This is the
  archived objective-health margin, not the discarded candidate-admission cliff. At least one
  nonzero count must be safe for the fitting score to support repair; if none is safe but at least one
  is unresolved, repair remains unresolved rather than negative.
- Across the controlled trajectories, the lower clustered 95% bound on within-trajectory rank
  correlation must exceed zero. The upper bound on both selection regret—selected error minus the
  best available error—and selected error minus count-0 error must be at most 0.05. Crossing a bound
  is unresolved; a lower bound above 0.05 is negative for that selector. These checks prevent a score
  that looks good globally from consistently choosing an over-repaired checkpoint.
- The label-free own-minus-donor selector is usable only if its validation correlation has a lower
  95% bound above zero and its held-out selected output remains content-bearing.
- A lower bound above zero passes. An upper bound at or below zero is negative for this score. An
  interval crossing zero is unresolved and cannot close the score or initializer.
- A failure closes only the tested score, channel shape, decoder, and representation combination.
  It cannot close all Phase-1 initializers.

**Status.** **2026-08-20 — Normatively unblocked; execution pending.** The original donor table can
assign exact zero-support comparisons and is superseded. The replacement audio-only law has a
exploratory all-8,416-candidate preflight of 792/890 swap-eligible and 98 `no_swap` utterances; it is
not the production-pool coverage. Production must rebuild the two role-local tables in Section 4,
hash them, and report split-specific coverage. A common feasible subset of
the old table is rejected because only 3,398/8,900 assignments survive across all 340 existing local
outputs and 267 utterances retain none. The controlled library still supplies the independent score-
validation set; no score, selector, or gate result exists yet. The earlier likelihood/error anti-
alignment remains a warning, not a verdict on the selected duration model. The character route must
repeat this compact calibration because phone score scaling and duration do not transfer.

### 1g.2a — Test higher-order fitting context before rejecting phone repair (H4-LM)

**Purpose.** Determine whether the baseline fitting bigram is a load-bearing operating-point choice
without attributing decoder-language-model gains to repair or rejecting the estimator at one context
order.

**Approach.** This is a conditional H4 follow-up before H5, not H6. Trigger it only before evaluation
labels open and only after the baseline assay prerequisites pass: mechanics, the positive control,
donor-score calibration/correlation, and selector validity. These prerequisites explicitly exclude
the method-specific nonzero-count/update-health result. A failed or unresolved prerequisite is fixed
first and does not trigger an LM arm. After they pass, H4-LM triggers unless the frozen label-free
selector assigns a safe nonzero bigram repair count to at least one admissible real H3 start. This
includes the cases where no nonzero count is safe or both ESPUM and fingerprint select count 0. A
pre-evaluation-ready baseline bigram arm
leaves H4-LM parked and may open evaluation. This does not assert that the held-out content gate will
pass. Ney's synthetic order sweep motivates the question but changes
recognition order for the learned- and correct-channel rows together; it does not isolate fitting
order or predict a LibriSpeech gain.

Keep smoothing and order separate. The current H4 add-one bigram is `legacy-2g`. Build a matched
unpruned modified-Kneser--Ney `G_fit` family at orders 2, 3, and 4 from the same complete `T_phi`,
39-phone inventory, BOS/EOS convention, `interpolate_unigrams=True`, and fallback discounts
`[0.5,1.0,1.5]` used by the canonical `phoneme_ngram_lm` recipe. Compile each ARPA into a proper
generative automaton: BOS is history only, `<unk>` is not an emitting phone, the legal 39-phone-plus-
EOS continuations are explicitly normalized at every history, and hashes bind source text, ARPA,
compiler, and symbol order. The clean order contrast is matched 2/3/4; comparison with `legacy-2g`
separately reports the smoothing bridge. The banked phone 4-gram remains `G_dec` for every arm and
replaces rather than multiplies `G_fit` during decoding.

The bounded full-method comparison is nevertheless an operating-point contrast between add-one
`legacy-2g` and matched-MKN order 4, so it changes smoothing and order together. The D bridge makes
that confound visible but does not remove it. Report any deployable difference as a `G_fit`-identity
effect, not as a causal order-only gain; a full matched-order-2 control is not funded here.

Separate two questions. **H4-LM-D** freezes the accepted `p=0.23560298`, two-state topology, starts,
floor, boundary law, repair counts, donor table, decoder, and selector and changes only `G_fit` during
repair. It is a bounded read on the reference and four accepted H3 starts: matched order 2 measures
the smoothing bridge, order 3 is directional, and order 4 is the proposed context probe. It cannot
feed H5. **H4-LM-F** is the coherent matched-4 method arm: recompute its EOS-conditioned sentence-
length law, refit `p` on exactly the 6,414 update IDs, repeat the one-/two-state H1 gate, and rerun the
complete H4 calibration under that operating point. Only a label-free selection between the coherent
`legacy-2g` baseline and this full F arm may feed H5. D neither selects that comparison nor supplies a
deployable channel. Matched orders 2 and 3 do not receive full controlled-library arms and cannot feed
H5; no result may silently expand that scope.

**Experiments.**

1. **Matched LM artifacts.** Persist `legacy-2g` and matched 2/3/4 identities, explicit BOS/EOS/
   backoff normalization checks, history/arc counts, and complete-`T_phi` hashes. No 5-gram is funded.
2. **Exact H2-LM engine.** Implement a context-state automaton whose state is the up-to-`order-1`
   phone history and duration sub-state; emission rows remain tied by current phone/sub-state and the
   M-step aggregates fractional counts over histories. Preserve exact BOS initialization, EOS exit,
   backoff probability, and the deleted-silence rule that forces a phone exit while retaining LM
   history. On tiny examples, orders 2/3/4 must reproduce exhaustive likelihoods, posteriors, and
   counts. Instantiated with `legacy-2g`, the generalized sparse engine must reproduce the accepted
   dense H2 likelihood, posteriors, M-step `B`, and boundary behavior. The matched MKN order-2
   automaton is checked separately against exhaustive enumeration.
3. **Measured resource gate.** Use the accepted H1 update-ID order and the same deterministic 32-way
   `ids[j::32]` sharding as the decoder. Deduplicate identical array hashes and sort all corrected
   baseline start/count tables by `(mean emission-row entropy, canonical arm name, repair count)`.
   Select indices `0`, `(n-1)//2`, and `n-1`. For each selected table,
   preflight one exact E-step for trigram and then 4-gram repair on the longest update utterance chosen
   by descending `(retained-unit count, utterance ID)`. Rerun the maximum-time and maximum-RSS cells
   on the update chunk having the largest total retained-unit count, tie by lower chunk index. Exact
   production EM aggregates shard likelihoods and expected counts before each single common M-step;
   no shard performs an independent M-step. Persist probe IDs, table identities, timing, maximum RSS,
   reached histories/arcs, shard contents, queue limits, and headroom. Production requests must be at
   least 1.5 times the maximum measured elapsed time and 1.5 times the maximum measured RSS, rounded
   up to whole hours and GiB; a cell that
   cannot fit the declared queue limits is resource-infeasible. A 4-gram has at most `39^3=59,319`
   full phone histories; including the empty and shorter BOS histories gives at most
   `1+39+39^2+39^3=60,880` context identities, each with two duration sub-states. Implementation
   decisions use measured reached states/arcs, not that bound. The accepted 48 H2 cells are beam-
   decoder timings and provide
   probe anchors only. No context/beam pruning may silently change the EM objective.
4. **Fixed-duration diagnostic.** Run `legacy-2g` and the matched 2/3/4 family only on the prospective
   reference and four accepted H3 calibration starts, with identical update/selection roles, counts
   0/1/2/4, masks, donors, and the already frozen baseline decoder setting for each start. Report
   label-free likelihood and own-minus-donor trajectories plus later descriptive error; do not run the
   81-row controlled library, refit a selector, choose an order, authorize final refit, or close an
   unrun coherent arm from D.
5. **Coherent matched-4 arm.** If exact order 4 passes the resource gate, create its distinct H1-LM
   duration/topology and H2-LM engine artifacts and repeat the complete calibration for the 81-row
   controlled library and four accepted H3 starts. Before that fan-out, repeat the order-4 longest-ID
   and worst-32-way-shard resource reads with the F arm's actual `p` and topology and bind the new
   contract; the fixed-`p` D contract is not a substitute. Combine this coherent arm with the already
   valid `legacy-2g` tuples, extend the frozen selection tuple to `(G_fit identity, repair count,
   decoder setting, restart)`, and validate it over every deployable full-method tuple. Real-start
   selection uses only the validated own-minus-donor score on the 890 selection utterances; exact ties
   retain `legacy-2g`, then choose the lower repair count and existing decoder/restart order. Only then
   repair the selected H3 `final_refit` starts on 7,304 construction utterances and open the 1,112
   evaluation utterances once. Initial H3 `Q`/`B` tables are reused; the LM arm does not rebuild
   initializers.
6. **Descriptive perplexity.** After the order/selector artifact is frozen, report conventional per-
   phone perplexity including EOS for every fitting LM and fixed `G_dec` on phonemized text disjoint
   from `T_phi`: the 890 selection transcripts when their labels are opened for controlled
   validation, and the 1,112 evaluation transcripts only when evaluation itself opens. Persist ID,
   text, sentence/phone-count, OOV/drop, and scorer hashes, and assert sentence-level disjointness from
   `T_phi`. Any overlap makes that descriptive PPL unavailable rather than a gate failure. Perplexity
   never selects or gates order; the ESPUM checkpoint statistic 32.5352 is not an LM perplexity.

**Gate.**

- H4-LM execution is valid only after exact enumeration, `legacy-2g` dense compatibility, and a bounded
  production resource contract pass. An exact 4-gram that exceeds available scheduler resources is
  **resource-infeasible/unresolved**, not a scientific failure; an approximate variant requires a new
  preregistered approximation and stability gate. The exact matched trigram remains directional and
  cannot advance or stand in for an unrun 4-gram.
- Calibration labels validate the expanded full-method selector but never choose `G_fit` for a real
  start. Perplexity and PER cannot select order. In F, candidate and both controls receive the
  identical `legacy-2g`/matched-4, count, and decoder grid; do not thin controls after timing or
  results are seen. D retains only its five specified starts and cannot be expanded after its reads.
- H4-LM-D is diagnostic only. A negative fixed-duration result cannot close the coherent higher-order
  method, and a positive D result cannot enter H5.
- The matched-4 H4-LM-F arm may enter H5 only if its refitted duration/topology is admissible, its label-free
  selector passes the existing H4 validation, and its frozen held-out output passes the same content
  gate against both controls. Failure closes only the tested matched-4 operating point; the diagnostic
  matched-2/3 reads do not close their coherent methods, and a resource-infeasible 4-gram remains
  unresolved.

**Status.** **2026-08-20 — Planned, conditional, and unrun.** The current exact repair engine is
bigram-specific; neither the higher-order engine, matched fitting LMs, resource evidence, diagnostic,
nor coherent arm exists. H4-LM does not delay a pre-evaluation-ready baseline bigram arm and H6
remains the separately gated character route.

### 1g.3 — Pass the weak seed into the SAE loop

**Purpose.** Test the clarified Phase-1 question directly: does Phase 1g improve the starting point
of the policy, the scorer, or both?

**Approach.** Separate the two handoff paths before coupling them.

| Check | Initial policy | Initial scorer | Question |
|---|---|---|---|
| Policy only | Trained on channel-A pseudo-text | Fixed scorer already validated by Phase 2.5 | Did the pseudo-text teach an audio-dependent policy? |
| Scorer only | Fixed `theta_0` ep50 and its frozen rollouts | Independently fitted channel B | Does the channel rank better hypotheses higher? |
| Combined | Channel-A policy | Channel B, never trained on A's hard one-best text | Does the complete weak start retain an advantage under one loop pass? |
| Matched controls | Trained separately on random-map and pseudo-pair text | Their identically treated control channels | Would the same compute work without speech content? |

Freeze the two channel roles before evaluation labels are opened. Rank every predeclared repaired
candidate by the validated label-free selector on the 890 selection utterances. Channel A is the
first row. Channel B is the first remaining row whose initial emission table came from a different
initializer or RNG seed; rerun its fitting independently and never expose it to A's pseudo-text. For
phones this may pair fingerprint with an ESPUM seed or two distinct ESPUM seeds; for characters it
uses two distinct members of the fixed 29-seed set. If no second eligible row exists, the separate
policy/scorer tests may still run but the combined path is unresolved. A later gold result may not
substitute another row. For the two content-free arms, use independent A/B constructions:
marginal-random seeds 1000/1001 and pseudo-pair seeds 0/1, with identical repair and selection
budgets. The direct-channel-A scorer remains only the exception allowed by the hard constraint in
Section 2.

The phone reference uses one fixed interface. Convert phone lattices to words with the exact Phase-1d
pronunciation lexicon, word 4-gram, normalization, homophone handling, and OOV rule; retokenize those
words with the existing policy tokenizer. Convert policy rollout words back to phones with the same
fixed Phase-1d normalization and G2P/lexicon before channel scoring. This lexicon cost is why the
phone result remains reference-only. Character pseudo-text detokenizes directly to words, and rollout
text converts directly to normalized characters.

Each handoff has its own quarantined positive control, pinned at one operating point:

- **Policy mechanics:** subset the banked Phase-1d word hypotheses
  `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/word_decode/`
  `Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks/output/word_hyps.json` to the exact Phase-1g construction
  utterances and replay the fixed policy-training budget below. The existing full-train-clean-100 AV
  student, `work/i6_core/returnn/training/ReturnnTrainingJob.2fb02hGUdHNj` epoch 10, is a banked
  sanity anchor; by itself it validates only generic AV learnability, not the same-bed assay.
- **Scorer mechanics:** the held-out reference channel versus audio-free and score-swap controls.
- **Combined loop mechanics:** the known Phase-2S pair—`theta_0` ep50
  (`work/i6_core/returnn/training/ReturnnTrainingJob.OLzy9Q2oC3mU`) with the frozen p10
  reconstruction model at ep50
  (`work/i6_core/returnn/training/ReturnnTrainingJob.ExCoQDKtXAGH`). Replay this pair on the same
  20.48-hour construction-utterance list, rollout seed, and one-pass budget used by the candidate.
  A replay on a different corpus or budget validates only generic loop mechanics and cannot clear
  this assay's positive-control requirement.

If a reference policy or scorer uses unavailable supervision, it diagnoses only the other half and
cannot enter the reported initialization.

**Experiments.**

1. Decode the construction portion of the 20.48-hour bed once, saving one-best text, alternatives,
   channel posteriors, and confidence. Exclude the 540 clean and 572 other evaluation utterances from
   pseudo-text training and checkpoint selection. A final all-data refit is evaluated only on
   untouched test data.
2. Train the candidate policy, both matched-control policies, and the same-bed policy positive
   control with the `theta_0` AV cross-entropy model, batch/accumulation, optimizer, and label
   smoothing. Give each exactly the number of optimizer updates used by
   `ReturnnTrainingJob.OLzy9Q2oC3mU` through epoch 50, stretch the same cosine schedule over that fixed
   count, and read only the final update. Use training seeds 0 and 1. Use every construction
   pseudo-label; do not select an epoch or confidence cutoff with gold data.
3. First run the canonical Phase-2.5 probe on the frozen `theta_0` ep50 dump: 512 utterances,
   12 rollouts per utterance, temperature 0.7. Candidate and control scorers score the identical
   rollout groups.
4. Repeat that probe on the actual Phase-1 policy and both matched-control policies using the same
   512 utterances, group size, temperature, sampling seed, and decoder. Combined-loop admission
   requires the actual-policy result; passing only on `theta_0` is a scorer-mechanics result.
5. If the separate paths pass, run the candidate, both treated controls, and the combined positive
   control for one loop pass over the identical construction-utterance list, one presentation per
   utterance and 12 rollouts at temperature 0.7. Copy learning rate, optimizer, reward normalization,
   KL settings, and frozen-scorer rule from the known Phase-2S one-pass recipe; only the initial policy
   and reconstruction score change. Report checkpoints at 0%, 25%, 50%, and 100%; the primary endpoint
   is fixed at 100%, not selected by WER.
6. Run training seeds 0 and 1 per condition for the end-to-end claim. For each utterance, first form
   the paired candidate-control effect within each seed, then average those two effects. Bootstrap
   utterances—and donor assignments where applicable—over these fixed-seed averages; never pool the
   two seeds as independent utterances or resample two seeds. Also report each seed separately and
   require the direction to reproduce. The claim is only about this fixed-seed average. Keep ordinary
   self-training from the same policy start, on the same utterance/update budget, as the non-RL
   comparison.

**Gate.**

- **Policy path:** after fixed pseudo-text training, greedy-decode own audio and each matched donor
  audio. With the original utterance reference kept fixed, the policy must be content-bearing under
  the Section-4 `D_plain` and decode-swap comparisons.
- **Scorer path:** the existing Phase-2.5 gate applies: within-group Spearman confidence interval
  above zero, true-hypothesis score gap above zero, reward-selected WER no worse than group mean, and
  a same-speaker score-swap margin over the content-free scorer. It must pass on the actual Phase-1
  rollout distribution before the combined loop.
- **Combined path, control advantage:** for each control `c`, define
  `Delta_control[c] = WER(final control[c]) - WER(final candidate)`. The simultaneous paired 95%
  lower bound must exceed zero for **both** controls; equivalently, the candidate must beat the
  stronger final control as well as the weaker one. An upper bound at or below zero is negative; an
  interval crossing zero is unresolved.
- **Combined path, bounded drift:** define
  `Delta_drift = WER(final candidate) - WER(candidate start)`. An upper paired 95% bound at or below
  `+0.5` absolute WER passes; a lower bound above `+0.5` is negative; overlap with `+0.5` is unresolved.
  The 0.5 margin is the project's existing loop relevance unit. An already-better seed need not show
  a significant within-arm improvement during this deliberately short assay.
- Both fixed training seeds must have positive point estimates for both `Delta_control` values. The
  simultaneous interval gate is computed by averaging each utterance's paired effect across the two
  seeds and then resampling utterances, as specified above. This is not a population-level
  random-seed claim. A sign disagreement is unresolved/non-reproduced, not evidence of a negative
  effect.
- Failure of the positive control for the affected path makes that path's candidate result
  **unresolved**, not negative.
- Passing on phones validates mechanics and the existence of a useful weak seed; it does not establish
  the lexicon-free claim.

**Status.** **Pending 1g.1, 1g.2, and any triggered 1g.2a arm.** A passing policy-only or scorer-only
row is a promising component, not yet a usable end-to-end Phase-1 initialization. Existing project
evidence makes the separation essential: an audio-to-text model can bind to audio even above 100 WER,
while the G-track
showed that a policy and scorer trained around the same filler errors can reward their shared mistake.

### 1g.4 — Preserve the closed anchor result; conditionally test one new soft-energy hint

**Purpose.** Preserve what the completed anchor tests ruled out, while defining one narrow new
downstream question: can energy give a near-threshold seed a harmless soft nudge?

**Approach.** Historical 1g.4 is closed **not funded** for the spectral partition/constraint route
and the deterministic hard two-class descriptor route. The registered six-factor soft product was
not actually tested: the completed screen compared seven alternative descriptors on the same
syllabic/non-syllabic target, not six independent articulatory memberships. Its old “failed
prerequisite” statement is corrected to **not answerable**. It remains unfunded, and no rerun is
needed unless that abandoned route is explicitly revived. The 2026-08-19 soft-energy proposal is a
new direction after the user clarified that Phase 1 only needs a weak downstream seed; it did not
“survive” the old gate and does not revise either completed failure.

The new test, if triggered, uses one fixed full-support energy bias. It may gently change initial
probabilities but never forbid a unit–symbol pairing. Energy was identified after label-based
inspection, so its effect must be confirmed on a locked population not used to choose the descriptor.
No new descriptor, sign, coefficient, or representation search is allowed.

**Experiments.**

1. Completed spectral partition and constraint tests.
2. Completed deterministic hard two-class descriptor test.
3. Unexercised six-factor soft-product specification, preserved only in the historical snapshot; no
   result is claimed.
4. Conditional new test: one frozen soft-energy bias, plus shuffled-energy and reversed-sign
   controls, on a locked split. Its numerical coefficient and locked population must be added to this
   plan before launch; until then the experiment is not funded.

**Gate.**

- Historical exercised gates remain unchanged. The descriptor hard route required at least 0.85 mass
  accuracy, at least 0.20 over its measured majority, and the registered containment check. The soft
  product's four-of-six prerequisite remains its historical specification but has no verdict.
- “Near” means the real seed's iteration-0 error interval overlaps the controlled 1g.1 repair
  transition band but its selected repaired result does not pass. Otherwise the new test is not run.
- The new soft bias advances only if simultaneous paired 95% lower bounds show improvement over the
  no-bias seed in both the label-free selector and repaired held-out error, while shuffled and
  reversed controls do not improve.
- Failure closes the energy hint, not channel estimation.

**Status.** **Historical spectral and hard-descriptor routes closed not funded; six-factor product
not answerable and parked; one new test conditional and not yet funded.** The spectral route failed.
The best hard descriptor, energy, reached 0.8130 mass accuracy versus a 0.5449 majority baseline and
therefore missed the registered 0.85 accuracy bar; containment was not the deciding failure. The
six-factor product has no result and does not enter 1g.5. A single fixed soft-energy bias has never
been tested downstream and may be registered only under the trigger above.

### 1g.5 — Build the first lexicon-free channel

**Purpose.** Produce the Phase-1g candidate that can support the project's main unpaired,
pronunciation-lexicon-free claim.

**Approach.** **Dated direction change, 2026-08-19:** characters now precede BPE after the user chose
phones before BPE and clarified that simplicity matters more than a standalone Phase-1 score. This
supersedes the archived nearest-rate BPE-first specification for future runs; it changes no completed
result. Characters are the smallest direct lexicon-free output, and their words can be retokenized
for policy training without estimating a BPE channel.

Use the raw adjacent-deduplicated unit stream and the complete normalized text artifact catalogued in
`SAE_1g.md`. `G_fit` is an unpruned character bigram over exactly the emitting characters observed
in that text, with BOS, EOS, and word-boundary transitions. The boundary itself emits no audio. The
sequence decoder instead uses an unpruned word trigram from the same text composed with a
deterministic spelling lexicon containing exactly its observed word types; spelling characters emit
and word boundaries are epsilon transitions. The fit bigram and decoder word LM replace one another
in their respective roles and are never multiplied. Preserve the artifact's case, apostrophe, and
normalization rules. Report reference OOV coverage, but do not add words after seeing it. Use the
same `lambda={0.5,1,2,4}`, `beta={-2,-1,0}` grid as the phone sequence decoder, with `beta` applied
per output word, and the same label-free beam-stability rule. A per-unit local character decode is
diagnostic only and cannot select or deploy the route because repeat collapse would destroy genuine
double letters.

Every emitting character consumes at least one audio unit. Remove the fixed proxy-silence units using the
existing Phase-1f/1g.0 raw-stream rule before fitting; do not discover or tune a new silence list. A
character emits `L>=1` units with probability `P(L)=(1-p)p^(L-1)` and mean `1/(1-p)`. Estimate `p`
before learning character identities:
tie every character emission row to the measured audio-unit marginal, fit only the audio-length
likelihood under `G_fit` and its end-of-sentence probabilities on the 6,414 update utterances, then freeze the
duration. Read both 1g.0 duration-bearing curves at that value. If the one-state variable-duration model is
admissible, use one emission row per character. Only if it is decisively rejected may a passing
two-state form be used; then the first unit uses state 1 and later units use state 2 with its
self-loop. An indeterminate smaller form, or failure of both forms, leaves the trial unresolved. This
is a label-free construction-only choice. Report finite-path coverage and,
for two states, state occupancy; do not add character skips after looking at transcripts.

The primary random initializer has one fixed law. For character `y` and unit `u`, draw
`r[y,u] ~ Normal(0,1)` and set `P(u|y)` proportional to `m[u] exp(r[y,u])`, where `m` is the measured
audio-unit marginal; normalize every row and use RNG seeds 0 through 28. Apply the fixed symmetric
two-state perturbation from 1g.1 immediately before repair step 1 only if two states were selected.
Run all 29 before any
transcript-based restart read. For each
trajectory, the frozen label-free selector chooses one of repair counts 0/1/2/4. It also ranks the 29
trajectories and freezes channel A plus the distinct channel B required by 1g.3 before error is
opened. Channel A is the single deployment candidate for the restart-law
diagnostic, “success” means that a trajectory's already selected repair count passes the
content-bearing gate on the fixed final evaluation fifth after all 29 outputs and the single
deployment candidate are frozen. The
benchmark probability 0.10 is a practical compute target—about ten restarts per content-bearing
output—not a claim about the true chance of character decipherment. Under that target, zero gate
passes in 29 trajectories has probability `0.9^29 = 0.047`. This statement concerns the probability
of clearing the registered gate at its present precision, not the probability that latent content
exists. A successful non-selected trajectory diagnoses a selector miss; it may not replace the
preregistered deployment candidate.

The length-matched pseudo-pair table is one separately named auxiliary initializer. Stable-sort text
lines by their number of emitting characters; for an audio sequence of length `L`, draw uniformly
with seed 0 from the up-to-32 lines in the canonical insertion-point window `[j-16,j+16)`. Text-line
reuse is allowed. Align audio position `a` to character position
`floor(a * N_text / N_audio)`, form the soft count table, apply the common `1e-8` floor, and
normalize. It is never pooled with the 29 random trajectories for the
binomial statement. A pass advances it; a miss is only one failed auxiliary start. Candidate and
control conditions receive the same repair counts and frozen label-free selection rule.

BPE is considered only when the scorer handoff demonstrably requires the loop's exact BPE tokens.
A character duration or spelling failure may motivate a separately registered future model, but it
does not automatically fund BPE: larger text pieces do not by themselves solve a duration model.
Merely matching the audio token rate is not enough reason to create a BPE-128 side task.

**Experiments.**

1. **Path feasibility.** On construction data, report the unpaired audio-length and text-length
   distributions used by the shared-duration fit. For the held-out supervised character control,
   require 100% finite decoder paths; if two states are selected, report first-state, second-state,
   and self-loop occupancy on construction and held-out folds.
2. **Character positive control.** Fit a supervised character channel with latent alignment on the
   construction folds and evaluate held out. There are no natural character boundaries, so do not
   call this an exact oracle.
3. **Compact character score check.** Repeat 1g.2 with controlled degradations of that positive
   control, the equal-row audio-marginal null, marginal-random seeds 1000--1019, and pseudo-pair seed
   0 before selecting a real run.
4. **Real starts and controls.** Complete all 29 fixed-law random starts and the one separately
   reported length-matched pseudo-pair start before a transcript-based restart read. They use the
   same character inventory, text corpus, state model, decoder, repair counts, and frozen selector.
   Freeze the A/B roles by the label-free selection score as specified in 1g.3. Report all 29 errors
   only as a quarantined basin-yield diagnostic; never substitute a better-looking row for A after
   labels open. Do not combine the pseudo-pair start with the 29-run empirical rate.
5. **SAE handoff.** Separate policy and scorer checks from 1g.3, then the fixed combined loop.
6. **Conditional exact-loop-BPE repeat.** Before fitting, pin the exact tokenizer artifact, vocabulary
   rate, rare-token backoff, zero-probability handling, path-feasibility read, and a held-out
   supervised BPE positive control. Use the same null and handoff logic.

**Gate.**

- The character positive control must have 100% finite paths, be content-bearing, validate decoder
  ordering, and pass the character sequence-decoder 1g.2 score check. Failure is a character-model mechanics problem, not
  evidence that unpaired decipherment is impossible.
- A real character seed must be content-bearing by Section 4.
- A separate 1g.3 path beating its matched control identifies a promising component. A useful
  end-to-end Phase-1 initialization requires the combined 1g.3 path to pass.
- A restart-based negative applies only to the fixed random-initializer law and the declared
  0.10 practical **gate-pass yield** target. It requires zero passes among all 29 already selected
  trajectory outputs on the fixed final-evaluation diagnostic, valid positive controls, and a
  frozen selector. It does not say that characters, all character initializers, or every unresolved
  trajectory lacks content. If any trajectory's content interval is unresolved, the content verdict
  remains unresolved even though zero of 29 can still reject the declared practical gate-pass yield.
  The independently selected deployment candidate must pass once on the final evaluation fold; a
  non-selected success cannot rescue it. If another trajectory succeeds but the selected one fails,
  the selector is invalid for deployment and the character result is unresolved rather than
  cherry-picked.
- If phones work but characters fail, report the phone-versus-character route gap. It bundles
  alphabet, language model, duration/skip model, decoder, error currency, and pronunciation lexicon;
  do not attribute the whole difference causally to the lexicon.
- BPE receives no automatic rescue run after an unexplained character failure.

**Status.** **Pending the corrected phone mechanics and one valid separate handoff.** Characters are
the first primary candidate; phones remain a bounded reference and BPE remains conditional. Phone
yield may set an engineering floor for compute but never powers a negative character verdict.

### 1g.6 — Change the acoustic units only after a working channel exposes a unit problem

**Purpose.** Avoid changing the representation to compensate for an unvalidated decoder, score, or
initializer.

**Approach.** Diagnose boundary loss and unit confusion separately. Prefer a more expressive channel
before destructive merging or splitting. Simulation may estimate restart cost near a measured real
operating point, but may not close the whole method family.

**Experiments.**

1. Quarantined gold-boundary and silence-restored diagnostics from 1g.1.
2. If a content-bearing real channel exposes boundary loss, at most one- and two-pass fixed
   resegmentation from that channel.
3. A downward-K ceiling check only if the diagnosis specifically points to excess unit fragmentation.
4. Simulation only around the actual 1g.1 repair region and with enough draws to report restart
   success uncertainty.

**Gate.**

- No representation change starts before a real channel is content-bearing.
- Pass count and stopping are fixed or label-free; development PER never selects a retained pass.
- Simulation failure rejects that initializer at that operating point, not the channel family.
- The GraphUnsupASR entry-7 reproduction is closed **not answerable** because neither Stage-A decode
  was content-bearing. Its relabeling delta is not a gate input.

**Status.** **Partly closed and otherwise deferred.** Hard merging is dropped. Splitting is not
funded; context-dependent channel states are the cheaper way to collect that headroom. Resegmentation
is undecided. Published GraphUnsupASR Stage A is closed **not answerable** because neither decode
carried content. The old broad simulation phase is superseded by this targeted role.

### 1g.7 — Use repeated speech only if direct fitting needs a new source of evidence

**Purpose.** Obtain equality constraints from repeated spoken content rather than from aggregate
stream statistics.

**Approach.** Discover repeated speech segments and measure within-cluster agreement against a
control that undergoes the same discovery and filtering process.

**Experiments.** Predeclare minimum covered hours, cluster-size rules, speaker and duration matching,
and the content-free discovery control before measuring agreement. Assignment back into the full
channel is a later step, not part of the discovery screen.

**Gate.** Fund a seed only if repeated clusters cover the predefined minimum speech mass and their
agreement advantage over the discovery-aware control has a positive confidence interval after
matching length, frequency, and speaker. “Above random” without coverage and uncertainty is
insufficient.

**Status.** **Deferred and not launch-ready.** Run only if 1g.1 shows that the existing seeds are
genuinely outside the repairable region or 1g.5 identifies missing cross-utterance information. The
numerical coverage minimum, cluster rules, and simultaneous confidence procedure must be added to
this plan before launch.

### 1g.8 — Use rule-based synthetic speech only as a disclosed prior

**Purpose.** Test whether a simple parametric speech generator can anchor enough real audio units to
help initialization.

**Approach.** Encode rule-based synthetic speech through the frozen encoder and codebook. No recorded
or neural voice is allowed. Treat phone labels as a phone-reference prior unless a lexicon-free bridge
is explicitly demonstrated.

**Experiments.** Small coverage study across fixed voices and settings, repeated-support analysis on
real speech mass, then the same candidate/control repair test only if coverage is stable.

**Gate.** The atlas must repeatedly support enough real speech mass, with enough correct covered mass,
to plausibly reach the measured 1g.1 repair region. A unit winning one frame once is not coverage.
Failure closes this prior, not channel estimation.

**Status.** **Deferred and not launch-ready.** It is not needed before the direct phone and character
routes are tested. Numerical support, covered-mass, stability, and repair-region thresholds must be
added to this plan before launch.

## 6. Deliverables ladder

| Step | Deliverable | Decision it enables |
|---|---|---|
| 0 | Label-free one-segment rejection, gold-duration diagnostics, and hard-anchor verdicts | Reuse valid Phase-1g work at its actual scope |
| 1 | Correct phone decoder and repair curve with construction-only seeds and controls | Decide whether weak phone seeds can be repaired |
| 2 | Validated content-sensitive training and selection score | Select without transcripts |
| 2a | Conditional matched-2/3/4 diagnostic and coherent matched-4 H4-LM arm before evaluation | Resolve a valid but inert/non-deployable bigram repair without conflating decoder order |
| 3 | Separate phone policy-start and score-start results | Validate at least one concrete handoff and start characters |
| 4 | Optional combined phone-reference loop | Validate the coupled assay without delaying characters |
| 5 | Character channel, separate handoffs, and fixed combined test | Establish or refute the first lexicon-free end-to-end initialization |
| 6 | Exact-loop-BPE channel, only if required by the scorer interface | Remove a demonstrated token-interface mismatch |
| 7 | At most one evidence-matched alternative | Address a diagnosed missing signal without reopening a method zoo |

An isolated policy or scorer pass is a promising component. Phase 1g is successful only when a
label-free lexicon-free candidate passes the combined SAE path over its matched content-free control.
A phone-only success is a useful reference and engineering milestone, not the final claim.

Phase 1g may close as a well-localized negative only after the relevant decoder, policy, scorer, and
combined-loop positive controls pass, the corrected real phone starts are exercised, and the
character route completes its registered evidence budget. A confidence interval crossing zero or a
failed positive control is unresolved; repair the measurement path before judging another seed family.
