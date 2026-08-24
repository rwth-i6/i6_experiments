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
explains the audio, where `T` is that input's own retained-unit count after the frozen silence mask.
`P_B` sums the frozen duration/state paths and includes their exit probabilities; it excludes
`G_fit`, `G_dec`, insertion penalty, beam score, posterior confidence, and every other decoder term.
Channel A produces
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

**Planner/verifier read: 2026-08-22 latest (replaces the morning read because the user greenlit
1g.9).** Item 3 is discharged with a NEGATIVE selector verdict (1g.2 Status 2026-08-22): H4 is
unresolved, the evaluation stays closed, and items 4-6 are blocked by their own conditions
(trigger not fired; no H4 pass; no valid phone handoff). The user's direction word arrived as
item 0 below; items 1-9 are kept as the registered frame they would resume into.

0. **1g.9 anti-collapse constrained-repair probe — CLOSED by its own clause-0 off-ramp
   (replaces the launch item, 2026-08-22, because the diagnostic completed and the planner
   ruled).** All five starts' training posteriors already satisfy both proposed constraints; the
   babble is decode-resident and start-specific; no constrained arm runs. Ruling and the
   user-facing direction fork in 1g.9 Status; evidence SAE_1g.md approach 15, verdicts 26-29.
   Phase 1g again holds for the user's direction word.
1. **H1--H3 and the baseline H2 engine are accepted; do not relaunch them.** H2 consumes one explicit
   deleted-silence boundary law and passes 23/23 channel tests including exact enumeration. H3's
   calibration and construction-population fingerprint, random-map, pseudo-pair, selected ESPUM
   seed-0/update-30,000 refit, and strict projections are complete. Distinct H1-LM/H2-LM artifacts
   below are conditional extensions, not permission to rerun these accepted baseline graphs.
2. **Preserve the verified H4 prerequisite graph; do not rerun it.** Its 821 jobs recovered and bound
   all count-0 `Q`, reused exactly 75 trajectories, reran the ten canonical soft starts, exported all
   85-by-4 count tables, built the role-local selection donors, and passed the update/selection decoder
   resource contracts. It contains no full-role decode, normalized score, selector, final refit, or
   evaluation result.
3. **Complete the baseline bigram H4 assay before evaluation opens.** Add the bounded global-beam
   stability extension and a new aggregation/selector boundary around the preserved raw-score
   interface, using only `Sel(c)` from Section 4; do not edit or reinterpret the hash-bound prerequisite
   jobs. Before controlled labels open, decode and score every controlled tuple plus the prospective
   reference/four-H3 selection surfaces, persist their provisional maxima and winner audits, then
   validate the selector and freeze the unchanged passing choices.
   Construction likelihood is update-health evidence only and has no fallback or tie-breaking role.
   Only after the selector freezes may final repair consume H3 `final_refit` starts and all 7,304
   construction utterances. Open the 1,112 evaluation utterances once after every baseline or
   triggered H4-LM choice is fixed.
4. **Run H4-LM only under its pre-evaluation trigger.** It is a conditional H4 fitting-context
   follow-up, not H6. Its **assay prerequisites** are mechanics, the positive control, donor-score
   calibration/correlation, and selector validity; they explicitly exclude the method-specific
   nonzero-count/update-health outcome and every evaluation-label gate. A failed or unresolved assay
   prerequisite is fixed first and does not trigger H4-LM. Once the prerequisites pass, call baseline
   H4 **pre-evaluation-ready** only when the controlled method-level safety read finds at least one
   safe nonzero count **and**, independently, the frozen label-free selector assigns a nonzero count to
   at least one real start. Controlled labels never mark, remove, or rank the real start's selected
   count. Such an arm may skip H4-LM and open evaluation; this is not a held-out content verdict.
   Otherwise trigger H4-LM and resolve it or document measured infeasibility before interpreting that
   operating point as evidence against repair.
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
2. **Label-free score swap.** For deployable tuple `c`, decode source `i` once to fixed hypothesis
   `z_ic`. Let `ell_c(U,z)=log P_Bc(U|z)` be the exact marginal channel score from Section 1. For
   frozen donor assignment `s`, define

       Delta_ics = ell_c(U_i,z_ic)/T_i - ell_c(U_d(i,s),z_ic)/T_d(i,s).

   Each denominator is that audio input's own positive retained-unit count after proxy-silence
   deletion. Do not use `(ell_own-ell_donor)/T_i`, `(ell_own-ell_donor)/|z_ic|`, original frame
   count, or a common denominator: raw log probability is extensive in audio length, and the accepted
   donor law permits `T_d >= T_i`. `P_Bc` includes the frozen duration/state law but none of the
   language-model or decoder terms listed in Section 1. `Delta` is the sole transcript-free selector;
   1g.2 must validate this exact statistic before deployment.

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
seed 0. Keep every resampling index paired across compared tuples and controls. Bootstrap within clean
and other. For each selection assignment `s`, compute the equal-
utterance mean of `Delta_ics` on the frozen eligible clean rows and separately on the frozen eligible
other rows, then set

    A_cs = (432/890) mean_clean(Delta_ics) + (458/890) mean_other(Delta_ics),
    Sel(c) = (1/10) sum_(s=0)^9 A_cs.

Higher is better. Preserve repeated donors created by the frozen `s mod k` assignment law. Compute
the point statistic in sorted utterance-ID and assignment order with `math.fsum` over float64 inputs;
confidence intervals validate the selector and gates but never replace its point ranking. Every
evaluation aggregation uses fixed weights 540/1,112 and 572/1,112. Do not renormalize target weights
after `no_swap` removal: compute each split's donor statistic on its common eligible rows, report its
coverage, and combine using the fixed role weights. A split with no eligible source, a nonpositive
denominator, any non-finite retained score, or a donor record that does not hash-bind the exact same
`z_ic` as its own record leaves selector construction unresolved; never clip, impute, or drop tuple-
specific rows. Do not require two noisy
split point estimates to share a sign. Instead, no split may show a statistically supported effect in
the opposite direction.

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
number. The selector-freeze artifact must bind all five selection-row identities—the prospective
reference and four H3 rows—and, for each, its initializer/restart, `G_fit` identity, repair count,
decoder family, `lambda`, `beta`, derived beam, donor table, score name and revision, per-input
normalization, split/assignment aggregation, total tie order, complete mechanically eligible tuple
table, provisional pre-label maximum/audit, selection sufficient statistics, and every input hash. It
must prove that every frozen row is its unchanged registered maximum; only this artifact may authorize
the corresponding construction-population final refits. A separate immutable evaluation-release
artifact must bind that selector artifact, the exact
final-refit channel/manifests and health evidence, every required construction/evaluation decoder
resource contract, the frozen role-local evaluation donor table and coverage, and either the passing
construction-role beam audit or the selected-local exemption. Only this second artifact may authorize
the one evaluation read.

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
- [ ] **H4 — Complete the corrected phone assay.** The 821-job prerequisite graph has restored and
  hash-bound count-0 `Q`, retained exactly 75 compatible trajectories, rerun the ten canonical soft
  starts, exported all count tables, frozen the support-compatible role-local donors, and passed both
  decoder resource contracts; preserve it. Add the bounded global-beam stability extension and a new
  deterministic aggregation/selector boundary plus the final-refit evaluation-release boundary, then
  run all controlled and five selection-row decodes/raw scores, persist provisional maxima and winner
  audits before labels, validate `Sel(c)`, and freeze the unchanged passing choices. Final-refit the
  H3 settings on 7,304 construction IDs and the reference on 4,455 dev IDs, then open the held-out
  content gate once. Persist every per-utterance, donor, resampling, uncertainty, admissibility, and
  total-order artifact. Detailed specification: Sections 4, 1g.1, and 1g.2.
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

**Status.** **2026-08-21 — H1--H3 accepted; H4 prerequisites complete and selector semantics
frozen.** `Phase1gH1Job.HbxKiuBTJ8aN` fixes the baseline two-state phone topology at
`p=0.23560298`; baseline H1--H3 must not be relaunched. The verified 821-job H4 graph recovered all
count-0 `Q`, reused exactly 75 compatible trajectories, reran the ten canonical soft starts, exposes
all 85-by-4 channel tables, freezes 513/890 selection sources as donor-eligible with 377 `no_swap`,
and passes the update and selection resource contracts. It contains no full-role decode, fixed-text
score, selector, final repair, or evaluation result.

The next implementation boundary first derives the bounded global beam table, then computes the exact
Section-4 `Delta`/`Sel` statistic from raw sufficient statistics for all controlled and five selection
rows. Persist provisional maxima and winner audits before controlled labels open; validation may only
freeze those choices unchanged. Likelihood remains update-health evidence and cannot fall back or
break a tie. The selector artifact authorizes the H3 7,304-ID and reference 4,455-ID final refits; the
final-refit health/resource/beam release in Section 4 authorizes the one evaluation read. H4-LM
remains conditional under 1g.2a; H5--H6 remain gated on a valid H4 output.

**2026-08-22 (planner): pre-label boundary COMPLETE and verifier-confirmed; controlled reference
labels MAY OPEN.** The global beam table classified all 12 grid points ineligible (baseline surface
local-only by construction), the 340-decode / 3,400-score selection surface and the provisional
maxima are persisted and hash-bound (`SAE_1g.md` approach 11, verdict 17; artifacts in its
Catalog), and all 85 provisional winners are local — so the frozen-versus-next-beam winner audit
is discharged by the registered local-winner exemption and the at-most-320 budgeted audit cells
are not needed at this boundary. The implementer's proposal 1 is ratified as exactly that
registered exemption. The next step is the registered one and no other: open the CONTROLLED
REFERENCE labels for selector validation and the method-level safety read; the four H3 rows' own
errors remain quarantined, no maximum may be recomputed or reranked, and likelihood remains
update-health evidence only. Two verifier observations bind the validation read (detail in
`SAE_1g.md` Verifier feedback 2026-08-22): only 79 of the 85 start channels are distinct (the five
`map_q09` draws share one channel; `soft_q09` is bit-identical to the reference), so clustered
intervals or null spreads over the controls must collapse duplicate channels — 76 effective of 81;
and the pre-label cross-start `Sel` ordering places the random-map null 9th of 85 against the
reference at 73rd, which the selector-validity verdict must be read against, not around.

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
utterances contribute no labels. Before controlled labels open, `Sel` chooses the reference's repair
count and local/sequence/grid setting exactly as for the four H3 rows; final reference fitting and
evaluation reuse that frozen setting. It is a positive control, not a candidate, and it never enters
the H3 cross-start ranking. If its preprocessing or map differs from the frozen fixture, report a
fresh local baseline rather than demanding 0.4148. Never fit and score a channel on the same
utterances for selection or content evidence; the post-refit construction beam audit below is a
label-free numerical-convergence check and fires no scientific gate.

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
utterances. Beam is not a selection-score coordinate. Reuse the three deduplicated min/lower-median/
max-entropy channel tables and the largest-total-unit update shard already frozen by the resource
contract in 1g.2. For each `(lambda,beta)`, run beams 64, 128, 256, and 512 on that shard for every
representative table. For each adjacent pair and representative, compute the exact-one-best unchanged
fraction and `sum_i |score_next(i)-score_beam(i)| / sum_i T_i`. Freeze one global beam per
`(G_fit identity,lambda,beta)`: the smaller beam of the first adjacent pair, in ascending order, for
which **every** representative has at least 99.9% unchanged one-bests and aggregate score change below
`1e-4` per retained unit. This adds at most 144 representative table/grid/beam shard cells per
fitting-LM identity, reusing any byte-identical completed cell. If no pair passes, that grid point is
label-free ineligible everywhere; local decoding is unaffected.

After `Sel` names the provisional winner for each of five selection rows—the prospective label-fitted
reference plus ESPUM, fingerprint, random-map seed 1000, and pseudo-pair—but before the selector
artifact freezes, every sequence winner
must repeat its frozen-beam versus next-beam comparison over all 6,414 update IDs for that exact
channel/count/grid tuple. The same two thresholds must pass. Failure makes H4 unresolved; it may not
substitute a runner-up or refit the selector. With five rows and 32-way sharding, this is at most 320
additional decode-shard cells for one completed baseline or coherent full-LM choice. A local winner
needs no beam audit. Oracle edit counts are opened only after the global beam table and every required
calibration winner audit are hash-bound.

Final refit changes `B`, so calibration stability does not license the deployed table. After each
H3 row is refitted on all 7,304 construction IDs, and after the reference is refitted on its 4,455
non-evaluation dev IDs, but before any evaluation decode, a selected sequence tuple must repeat the
same frozen-versus-next-beam audit on its own construction population under its exact final-refit
channel and resource contract. This adds at most 320 more 32-way shard cells across the five rows.
Failure is unresolved with no reranking, beam escalation, refit, or evaluation read; a selected local
tuple again needs no audit.

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
- Validate the **fixed sequence-family selection rule**, not individual grid points with labels.
  Before opening controlled labels, use `Sel` and its total tie law to choose one globally beam-stable
  sequence grid point separately for every controlled `(start,count)` table; do not include local in
  this within-family choice. No runner-up may be substituted after labels open. On the four reference-
  start counts, every simultaneous upper paired confidence bound on
  `PER(selected sequence)-PER(local)` must be at most `+0.01`. For the improvement read, within each
  count, each of the eight non-endpoint `q` levels, and each of the two damage families, average map
  draws inside their `(count,family,q)` cell, define
  `I = PER(local)-PER(selected sequence)`, then weight the resulting 64 cell-level `I` values equally;
  the lower confidence bound on `I` must exceed zero. Any reference
  lower bound above `+0.01`, or an improvement upper bound at or below zero, is negative; overlap is
  unresolved. This gives one verdict for the predeclared sequence family and its label-free grid
  rule. A negative or unresolved verdict excludes the whole sequence method without choosing a
  different setting; the separately registered local method and channel estimation remain live. A
  pass admits every globally beam-stable sequence grid point to real-start `Sel`, which—not PER—then
  chooses local or sequence and its setting. Controlled labels never prune a count or grid row. The
  0.01 safety margin is one fifth of the historical 0.05 relevance unit and tiny against the roughly
  0.48 reference-to-null span.
- After 1g.2 freezes the selector, a real seed advances when its selected repaired output is
  content-bearing by Section 4 and beats treated starts. If a treated content-free start itself
  becomes content-bearing, adopt that simpler estimator-from-scratch result instead of calling the
  control “failed.”
- The controlled repair curve explains the result; it does not substitute for running the actual
  seeds and controls.
- This verdict applies only to the phone reference.

**Status.** **2026-08-21 — H4 repair prerequisites are complete; decoding and the gate remain
pending.** The verified graph retains exactly 75 canonical trajectories and replaces the ten old soft
rows with their Q-space versions. All 85 starts and counts 0/1/2/4 are hash-bound and decoder-ready;
the role-local donor and measured resource contracts also pass. No full-role selection decode, raw
fixed-text score, normalized selector, construction refit, or evaluation has run. Execute only the
remaining 1g.2 boundary; do not rerun repair prerequisites.

The first E5 job remains exploratory, non-decisive evidence. It fit and scored the same 2,864 dev-
other utterances under different preprocessing, gold-derived mean duration 1.463, a hard 0.9-smoothed
map, fixed-stride text, and bigram posterior-argmax decoding; it ran neither real seed nor treated
control. Its configured retention 0 moved PER 1.0109 to 0.8409, while retention 1 moved 0.4865 to an
LM-selected 0.4589 and then 0.6699 at step 30. These engineering observations fire no gate, and the
historical 1f verdict is unchanged.

### 1g.2 — Check that the training score rewards speech content

**Purpose.** Keep the likelihood used to fit a channel numerically healthy, and independently ensure
that the sole deployment selector points toward speech-dependent transcripts rather than merely
toward fluent or frequent text.

**Approach.** Preserve the original fixed evaluation fifth: 540 dev-clean and 572 dev-other
utterances from `holdout_split(..., seed=0)`. Apply that helper again with seed 1 inside the remaining
four fifths. Its fixed 80/20 split gives 1,731 clean plus 1,834 other **update** utterances and 432
clean plus 458 other **label-free selection** utterances. On a fitting role `R`, report

    L_R(B) = [sum_(i in R) log sum_Y G_fit(Y) P_B(U_i|Y)] / [sum_(i in R) T_i].

During calibration `R` is exactly the 6,414 update utterances: the 3,565 dev-update utterances plus
the 2,849 dedicated train utterances whose transcripts remain quarantined. `L_R` is the EM objective.
Its complete numerical-health rule is: the recorded value at every registered count must be finite,
and the existing manifest checks for finite positive normalized rows, finite input paths, exact role,
and provenance must pass. Because update 1 first introduces the registered two-state symmetry
perturbation, no monotonic-likelihood tolerance or ordering gate is inferred across stored counts;
report every change descriptively. `L_R` may update `B` but may never rank, stop, select, or break a
tie among `G_fit` identities, starts/restarts, repair counts, decoder families, `lambda`, `beta`, or
beams. After choices freeze, final refit applies the selected fixed count to all 7,304 permissible
construction utterances under the same objective and must record a finite selected-count likelihood
and pass the same row/path/role/provenance checks; no construction-, selection-, or evaluation-role
likelihood reopens the choice. The label-fitted reference is the
exception: it uses labelled counts from the 3,565 dev-update utterances during calibration and all
4,455 non-evaluation dev utterances for its final fit. There is no rotating evaluation.

The sole deployment score is `Sel(c)` from Section 4, computed on the 890 selection utterances;
higher means more audio dependence. Report likelihood/error association as a diagnostic, but do not
calibrate a likelihood selector or define a likelihood fallback. The exact `Delta`/`Sel` formula,
normalization, aggregation, and tie law are frozen prospectively by this revision; controlled
reference degradations and independent null maps may only validate or reject them, never tune a
variant. The real ESPUM, fingerprint, and later character candidates are prospective tests, not
calibration points.
For a deployed seed, choose admissible repair and decoder settings without labels and open final error
only on the fixed evaluation fifth. Controlled calibration may read its quarantined reference labels
only after every output is fixed; those labels may never choose a real seed's start, repair count, or
decoder setting.

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

Freeze choices before reading controlled labels. First generate every controlled output and the five
selection-row decode/raw-score surfaces: the prospective reference plus the four H3 rows. A tuple is
**mechanically eligible** when its count is in
the fixed `{0,1,2,4}` grid, all likelihood/manifest and role/hash checks pass, and its decode and score
rows are complete and finite; a sequence tuple additionally needs a global beam for its grid point.
For each controlled `(start,count)` table, record the within-sequence choice needed by the family gate.
For the prospective reference and each of the four H3 rows, maximize `Sel(c)` over the mechanically
eligible fitting-LM identity when applicable, repair count, local versus sequence decoder, `lambda`,
and `beta`, then complete any required 6,414-ID winner beam audit. Persist these as immutable
**provisional pre-label maxima**.

Only then may the controlled reference labels open; the four H3 rows' own errors remain quarantined.
Use the controlled outputs to validate the already-frozen selector, issue the count method-level read,
and issue one sequence-family verdict. A baseline provisional maximum may freeze unchanged only if
the selector passes and either the sequence family passes or that maximum is local. A failed or
unresolved family with any provisional sequence winner makes H4 unresolved; it cannot fall back to a
local runner-up. The count read may trigger H4-LM but never changes a provisional maximum. If H4-LM
triggers, the baseline maxima cannot authorize final refit. Baseline labels may fire only this already
preregistered arm: every H4-LM decode, score, and choice job remains label-reader-free under the fixed
grid, and its combined provisional maxima plus required beam audits must freeze before the newly
generated matched-4 controlled errors or expanded gate verdict are read. Within either the baseline
or H4-LM boundary, no score is recomputed and no maximum is reranked after that boundary's verdict.

H4 does not reopen H3's frozen ESPUM seed/update or create a phone restart search. Exact point-score
ties use this total order: `legacy-2g` before matched-4 when H4-LM applies, lower repair count, local
before sequence, then sequence settings in `lambda=(0.5,1,2,4)` outer order and
`beta=(-2,-1,0)` inner order, then canonical initializer name and ascending integer seed/update. The
derived beam is bound to its grid point and never participates in ranking. Likelihood is never a
tiebreaker. One immutable selector artifact freezes the unchanged reference winner, every unchanged
H3 winner and their cross-start ranking for 1g.3, the donor table, complete score evidence, pre-label
choice artifact, and the binding fields required in Section 4. H3 final-refit repair starts from the
corresponding `final_refit` artifact and applies exactly that frozen count to all 7,304 construction
IDs; the label-fitted reference applies its frozen count on the distinct 4,455-ID dev construction
population. Neither has a selector, selection-label, or evaluation reader. The evaluation consumer
accepts only the Section-4 release artifact after final-refit health and any required construction-
role beam audit pass.

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
3. Measure rank agreement between the exact `Delta`/`Sel` selector and `-error` globally and in the
   predeclared local operating band with starting PER 0.80–0.93, which contains the actual phone seeds
   and controls. Report likelihood/error association separately as a non-gating diagnostic.
4. Within each controlled trajectory, test whether the frozen selector orders repair counts sensibly.
   Report its rank agreement, its error relative to the best of counts 0/1/2/4, and its change from
   count 0. Starting from the reference channel, also check repair steps 1, 2, and 4 separately for
   drift.
5. Treat independently generated maps or training trajectories as the independent units. Repeated
   checkpoints from one trajectory stay in one cluster during resampling and never count as extra
   independent examples.
6. The own-minus-donor `Sel(c)` contrast is the **sole selection score**, not an update objective.
   Repair updates continue to maximize the role-appropriate likelihood at their fixed counts. A
   failure of the finite likelihood or manifest health rule above makes the affected trajectory
   unresolved and ineligible; a pass never ranks or breaks ties. If `Sel` fails or is unresolved, H4
   has no selector: likelihood cannot rescue it, and no contrastive update is invented after labels
   are read.
7. After the score definitions are frozen, report ESPUM, fingerprint, and control outcomes without
   refitting the selector.

**Gate.**

- Under `Sel`, the reference channel must beat the strongest predefined null under a simultaneous 95%
  interval over all null maps and both damage families.
- Globally and within the 0.80–0.93 local band, the lower 95% bound of Spearman rank correlation
  between `Sel` and `-error` must be above zero. Resample map/trajectory first, donor assignment
  second, and utterance within split third.
- Count 0 is the no-repair baseline. For each nonzero count `r in {1,2,4}`, compute the paired
  reference-start local-decoder difference `PER(r)-PER(0)`. An upper 95% bound no greater than 0.05
  calls that controlled operating point safe; a lower bound above 0.05 is negative and overlap is
  unresolved. This is a method-level repair-safety read and the archived objective-health margin, not
  a label-based admissibility filter: it never removes or ranks a count for a real start. Every count
  with valid numerical/provenance evidence remains in the `Sel` maximum. At least one nonzero
  controlled count must be safe for baseline repair to be pre-evaluation-ready. If none is safe but at
  least one is unresolved, repair remains unresolved; if all are negative, the baseline repair method
  is negative at this controlled operating point. Either outcome fires the prospective H4-LM trigger
  rather than choosing a count with labels.
- Across the controlled trajectories, the lower clustered 95% bound on within-trajectory rank
  correlation for `Sel` must exceed zero. The upper bound on both selection regret—selected error minus the
  best available error—and selected error minus count-0 error must be at most 0.05. Crossing a bound
  is unresolved; a lower bound above 0.05 is negative for that selector. These checks prevent a score
  that looks good globally from consistently choosing an over-repaired checkpoint.
- The label-free own-minus-donor selector is usable only if its validation correlation has a lower
  95% bound above zero. After the selector and tuple freeze, the 1,112-ID evaluation output must still
  be content-bearing to advance, but that post-freeze result cannot trigger reselection.
- A lower bound above zero passes. An upper bound at or below zero is negative for this score. An
  interval crossing zero is unresolved and cannot close the score or initializer.
- A failure closes only the tested score, channel shape, decoder, and representation combination.
  It cannot close all Phase-1 initializers.

**Status.** **2026-08-21 — Selector semantics resolved prospectively; decode/score execution
pending.** The verified 821-job prerequisite graph supplies all 85-by-4 channel tables, direct-`Q`
starts, the frozen role-local donor table, and passing update/selection resource contracts; do not
rerun it. The production table has 513/890 eligible sources and 377 explicit `no_swap` rows. No full-
role decode, fixed-text score, normalized aggregate, selector, final refit, or evaluation job has run.
The user-raised likelihood/selector conflict is resolved in favor of `Sel(c)` alone, and Section 4 now
freezes separate own/donor retained-unit normalization plus the exact split/assignment aggregate.
Construction likelihood remains update-health evidence only. Implement the bounded global-beam
extension and a new aggregation/selector boundary around the preserved raw-score interface, validate
it on the complete controlled tuple set,
then apply it to the real starts; if validation fails, H4 remains unresolved with no likelihood
fallback. Add the consumer in a new module/config: do not modify the source-hash-bound
`recipe/2025-10-speech-llm/src/speech_llm/sae/h4_production.py` or
`recipe/2025-10-speech-llm/src/speech_llm/sae/h4_decode_jobs.py` and thereby invalidate the
prerequisite identities. The character route must repeat this compact calibration because phone
score scaling and duration do not transfer.

**2026-08-22 — GATE VERDICT (planner, verified round: SAE_1g.md verdicts 18-20 and the
2026-08-22 verifier entry).** The selector clauses read NEGATIVE, not unresolved: the reference
loses to the strongest control by 5.02 with a one-sided 95% interval of [-5.0719, -4.9782], and
all three rank-correlation clauses have upper bounds below zero (global -0.6226, band -0.2522,
within-trajectory -0.7900). The count method-level safety read PASSES (all three nonzero counts
safe; count 1 slightly improves), so the H4-LM trigger does NOT fire. The sequence family is
UNRESOLVED-untested (no eligible tuple; all winners local). Consequences, exactly as
pre-registered: `Sel` is failed and H4 has no selector; no likelihood fallback; no contrastive
update may be invented now that labels are read; the 85 provisional maxima stay frozen and
unreranked; no `H4SelectorFreezeJob` may be built; the 7,304-ID and 4,455-ID final refits and the
1,112-ID evaluation stay CLOSED. The failure closes the tested own-minus-donor score on this
channel-shape/decoder/representation combination only — it is a decision not to fund this
selector, not a finding that repair cannot work (the safety read shows repair itself is safe at
this operating point; the score that chooses among repairs is what failed, inverted with
substantial magnitude, consistent with the pre-label ordering and with the earlier
likelihood/error anti-alignment on this bed). Queue items 5 (handoffs, need an H4 pass) and 6
(character route, needs a valid phone handoff) are therefore blocked by their own registered
conditions: Phase 1g has NO live registered next step. Whether to close the phone-repair route
here, fund a new pre-registered selector science (the current controlled labels are burned as a
validation instrument — a successor selector would need fresh controls), or amend the plan to
open the lexicon-free character route without a valid phone handoff is a direction fork put to
the USER with the verifier round of 2026-08-22; no successor work starts before the user's word.

**2026-08-22 — USER-FUNDED DESCRIPTIVE PER READ (user: "I again am not against just using real
dev other data to compute PER").** Authorized as a measurement over the closed gate, not a gate
revision: one new CPU job computes plain per-split PER (dev-clean 432 / dev-other 458
selection-role utterances, same bed as the controlled read) for the FOUR REAL H3 rows at all four
repair counts 0/1/2/4, from the frozen surface's existing decode artifacts against the
`GoldPhonesJob.ZGSp0hxyd2YP` gold — no new decode, no modification of
`H4ControlledValidationJob` or its firewall. The job's docstring must carry the reporting rule:
descriptive evaluation-only; these numbers select nothing and fund nothing; any future decision
that picks a seed or count using them must be re-registered with the label circularity disclosed
as supervision cost. The 1,112-ID held-out evaluation stays SEALED. The 1g.2 gate verdict and
all its consequences stand unchanged.

**2026-08-22 — OUTPUT AUDIT OF THE BEST-PER ROW; RECOMMENDATION AGAINST SAE INIT FROM IT.** The
user proposed initializing SAE work from the best row of the descriptive PER table
(`real/pseudo_pair_seed0`, pooled 0.8096 at count 4) and asked whether its outputs look related
to the references. Planner audit of the frozen decode artifacts (`H4LocalDecodeJob.fmuKOR9QfZT9`
count 0, `H4LocalDecodeJob.5PcPAqGhDLYT` count 4, vs `GoldPhonesJob.ZGSp0hxyd2YP`; recomputed
pooled PER reproduces the banked `H4RealSeedPerJob.vu6Dp6HkJ2pH` values to full float precision
at both counts): the count-4 hypotheses average 0.389 of gold length (deletion rate 0.625,
insertion 0.0005, substitution 0.184, aligned-correct 0.191) and collapse onto a tiny inventory
— AH is overproduced by +0.417 relative frequency, only AH/N/T/DH are net overproduced at all.
No utterance of 890 is below 0.50 PER; 1 of 890 is below 0.65; dev-clean and dev-other are
equally affected (0.8088/0.8105). The count-0-to-count-4 gain (0.9136 to 0.8096) tracks the
hyp-vs-gold unigram total-variation distance (0.837 to 0.689): unigram calibration, not
sequence-level correctness. Reading: the row carries no sequence-level signal worth
transplanting; its margin over the random-map control is unigram-statistics deep. Planner
recommendation: do NOT initialize SAE from any real H4 start; the project's banked label-free
anchor for usable phone quality is the Rung 0 CTC student at 0.172 dev-other phone PER
(PLAN.md phase 1d), which no fitting-order change moves an 0.81-PER collapsed channel toward.
1g.2a (H4-LM) remains available as user-fundable science on the order question — a D-style
diagnostic on this start is cheap (dense-tensor exact engine; the registered resource preflight
binds the contract) — but it is registered as an estimator-family question, not a performance
route. Descriptive read only; the 1g.2 gate verdict and all its consequences stand unchanged.

### 1g.2a — Test higher-order fitting context before rejecting phone repair (H4-LM)

**Purpose.** Determine whether the baseline fitting bigram is a load-bearing operating-point choice
without attributing decoder-language-model gains to repair or rejecting the estimator at one context
order.

**Approach.** This is a conditional H4 follow-up before H5, not H6. Trigger it only before evaluation
labels open and only after the baseline assay prerequisites pass: mechanics, the positive control,
donor-score calibration/correlation, and selector validity. These prerequisites explicitly exclude
the method-specific nonzero-count/update-health result. A failed or unresolved prerequisite is fixed
first and does not trigger an LM arm. After they pass, H4-LM triggers unless the controlled method-
level read finds at least one safe nonzero bigram count and, independently, the frozen label-free
selector assigns a nonzero count to at least one real H3 start. Controlled labels do not attach
safety to or prune the selected real count. Thus either no safe nonzero controlled count or both
ESPUM and fingerprint selecting count 0 triggers the arm. A
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
   contract; the fixed-`p` D contract is not a substitute. After the coherent repaired tables exist,
   deduplicate and select that arm's own min/lower-median/max-entropy representatives exactly as in
   1g.2, then derive its matched-4 global beam table on the same largest update shard under the 1g.1
   rule. This is at most 144 new cells; neither the baseline beam table nor D's fixed-`p` tables
   transfer. Combine this coherent arm with the already
   valid `legacy-2g` tuples, extend the frozen selection tuple to `(G_fit identity, repair count,
   decoder setting, restart)`, and generate every matched-4 controlled and five-row selection score
   with label-reader-free jobs. Before reading the matched-4 controlled errors, compute the combined
   provisional maxima using only the frozen own-minus-donor score and total tie order from 1g.2, then
   repeat the five-row calibration winner audit, reusing an exact legacy audit only when every bound
   hash and tuple is unchanged; this adds at most 320 shard cells. Rerun the expanded selector gates
   and the method-level sequence-family gate on F's own 81-row controlled library; the baseline family
   verdict transfers only to legacy tuples. A failed/unresolved F sequence verdict with a provisional
   matched-4 sequence winner makes the combined arm unresolved and cannot choose a runner-up. Only a
   passing expanded selector may freeze the unchanged combined maxima. Only then repair the selected
   H3 `final_refit` starts on 7,304 construction utterances, refit the reference on its 4,455 dev IDs,
   and complete the final-table audits/evaluation-release boundary from 1g.2 before opening the 1,112
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
- After F, combined H4 is pre-evaluation-ready only if the matched-4 controlled method-level read has
  at least one safe nonzero count and, independently, at least one of the unchanged combined ESPUM or
  fingerprint provisional maxima has a nonzero count. Controlled safety never attaches to the real
  selected count. If every matched-4 nonzero count is negative, H4-LM is negative at this operating
  point; any unresolved count with none safe leaves it unresolved. If both real combined maxima remain
  at count 0, H4-LM has not rescued repair. In every case evaluation stays closed, with no count
  substitution, fallback selector, second higher-order arm, or post-verdict reranking.
- The matched-4 H4-LM-F arm may enter H5 only if its refitted duration/topology is admissible, its label-free
  selector passes the existing H4 validation, and its frozen held-out output passes the same content
  gate against both controls. Failure closes only the tested matched-4 operating point; the diagnostic
  matched-2/3 reads do not close their coherent methods, and a resource-infeasible 4-gram remains
  unresolved.

**Status.** **2026-08-20 — Planned, conditional, and unrun.** The current exact repair engine is
bigram-specific; neither the higher-order engine, matched fitting LMs, resource evidence, diagnostic,
nor coherent arm exists. H4-LM does not delay a pre-evaluation-ready baseline bigram arm and H6
remains the separately gated character route.

**2026-08-22 — USER FUNDS EXECUTION (out-of-trigger; user: "run trigram/4-gram is MANDATORY
now").** The registered trigger record stands (it did not fire); this is a user direction, not a
trigger revision. Funded scope = Experiments items 1-4: matched LM artifacts, the exact
context-state engine with its tiny-example and legacy-2g compatibility harness, the measured
resource gate, and the D fixed-duration diagnostic on the reference plus the four accepted H3
starts. The F arm and every selector-shaped consequence stay closed per the 1g.2 verdict. A
descriptive PER read of the new D decodes runs under the 2026-08-22 reporting rule
(labels-as-evaluation-only, circularity disclosed, 1,112-ID set sealed).

**2026-08-22 — PLANNER IMPLEMENTATION RULING (efficiency).** The engine must be the batched
history-tensor contraction form: alpha/beta indexed (utterance, duration sub-state,
39^(order-1) histories). Per frame, duration moves are elementwise-diagonal; the only
contraction is over phone-exit arcs against the explicitly-normalized per-history continuation
table — exact by construction under this section's normalization requirement, no pruning, no
approximation gate. Emissions stay in the tied (phone, sub-state) form and broadcast, never
materialized per history; expected counts aggregate over histories on the fly for the M-step;
float64 with the existing per-frame scaling law. Default compute bed: CPU numpy under the
existing node-sharing cpu-8 request — GPU jobs on this cluster take an exclusive whole 4-GH200
node and would idle it at D scale; a torch device path is optional and not funded for D. Cost
projection (the item-3 preflight remains binding; production request = 1.5x measured): an
order-4 E-step is ~5.6e12 floating-point operations over the 6,414-utterance update fold, i.e.
tens of minutes per (start, order) cell on cpu-8; order 3 is ~39x cheaper. The current dense
engine cannot represent order 4 at all (its dense transition matrix would be ~118 GB), so this
restructure is the minimum runnable implementation, not an optimization add-on.

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

### 1g.9 — Anti-collapse constraints on the channel-repair objective (USER-proposed and greenlit 2026-08-22; highest priority)

**Purpose.** Decide whether two explicit anti-collapse constraints added to the unsupervised
channel-repair objective recover phonetic content in the decoded output, or whether the collapse the
1g.2/1g.2a audits measured (deletion-dominated output at 0.37 of the reference length, AH
overproduced by +0.42, unigram total-variation distance 0.68 from the text distribution) enters
elsewhere: at initialization, at the decoder operating point, or below the objective in emission
discriminability. The probe must LOCATE the collapse before spending the training arm, because the
audited babble is already present at repair count 0, before the repair objective has touched
anything.

**Approach.** The repair criterion `sum_i log sum_Y G_fit(Y) P_B(U_i|Y)` becomes
`maximize L_HMM - lambda_uni * L_uni - lambda_rate * L_rate` with two terms:

1. *Corpus-level symbol-distribution matching.* `q_bar(v)` is the corpus-normalized posterior
   expected number of ENTRIES into symbol `v` (symbol entries, not frame occupancy, so duration
   cannot distort the estimated distribution), accumulated by forward-backward under the full
   model; `p_text(v)` is the unigram distribution of the same unpaired text corpus `T_phi` the
   fitting LM is estimated from (corpus and preprocessing pinned in the producing job's
   docstring). The divergence runs in the COVERAGE direction, `L_uni = KL(p_text || q_bar)`,
   which diverges when any text-supported symbol's usage goes to zero. The originally proposed
   mode-seeking direction `KL(q_bar || p_text)` is rejected: it prices a collapse onto the
   highest-frequency phones at only `-log p_text` of those phones, roughly 2.5-3 nats — a bounded
   cost the likelihood can pay — and gives zero gradient to symbols with `q_bar(v) = 0`
   (correction accepted by the user 2026-08-22). Jensen-Shannon is the admissible alternative;
   the choice is made before launch and stated in the docstring.
2. *Rate regularization.* `L_rate = mean_i (N_i/T_i - r_target)^2` on the posterior expected
   emitted-symbol count `N_i` per retained-unit count `T_i`. The topology and the geometric
   duration parameter stay FROZEN as H1 fixed them; this subphase does not reopen H1. `r_target`
   is NOT a new free constant: it is the symbols-per-retained-unit rate implied by the frozen H1
   length-law fit on update audio (that fit already maximized exactly this length law), derived
   and pinned in the producing job's docstring. Only if that derivation is shown degenerate may
   an external label-free segmentation estimate be proposed, and that is a new planner decision.

A hard maximum-duration cap is the registered escalation if the soft rate term proves insufficient
AND experiment 1 locates the collapse in the alignment posterior; it would unfreeze the H1
duration law and is not funded here. Because both constraints act on posterior EXPECTATIONS, which
soft posteriors can satisfy by hedging while the decoded best path stays collapsed, every gate
statistic below reads the DECODED output, never the expectations.

**Experiments.** All model-forward computation runs as sisyphus GPU jobs. The constraints break
the closed-form emission update; the constrained update rule is the implementer's choice, pinned
in the docstring, and the tiny enumerated example is extended to verify the constrained update
improves the constrained criterion.

1. *Locate-the-collapse diagnostic (runs first, alone).* One registered job computes, for the
   five 1g.2a starts at repair counts 0 and 4: the posterior expected symbol-entry distribution
   `q_bar`, the posterior expected rate `N/T`, the same two statistics from the banked decoded
   one-bests (the audited 1g.2a cells; no new decode), and both constraint terms' gradient norms
   at the current parameters (a term the optimizer cannot feel does nothing). This banks the
   "before" row, verifies the two proposed statistics flag the pathology, and answers WHERE the
   collapse enters: posterior versus decode.
2. *Constrained refits, smallest decisive set.* Two starts — the selected ESPUM start (H3's
   label-free selected real seed) and the pseudo-pair reference — at count 4 only, from the
   shared banked count-0 start: arms uni-only, rate-only, and both-terms at one lambda magnitude,
   plus both-terms at a second magnitude (four constrained cells per start; the banked
   unconstrained count-4 cells are the controls). Decode with the frozen 1g.2a decoder
   configuration; evaluate on the frozen 890 selection utterances (432 dev-clean + 458
   dev-other) with gold as EVALUATION ONLY; the 1,112-ID evaluation stays sealed.
3. *Unigram-matched babble null.* A registered job draws, per utterance, 100 random symbol
   sequences of the decoded length i.i.d. from `p_text` and reports the null distribution (mean,
   p99) of pooled correct-phone fraction and PER, printing its alignment convention. This is the
   audio-free floor: matching length and the unigram marginal is free, and the gate charges
   for it.

**Gate (pre-registered 2026-08-22, before any run).**

- *Clause 0, off-ramp.* If experiment 1 shows the POSTERIOR `q_bar` and rate already near their
  targets (total variation to `p_text` <= 0.15 and rate within 20 % relative of `r_target`)
  while only the DECODED statistics collapse, the constrained-training arm does not run as
  specced: the constraints would be aimed at the wrong stage, and the finding returns to the
  planner with the diagnostic as the deliverable.
- *Clause 1, effectiveness.* A constrained cell is READABLE when its decoded length ratio to
  gold is within [0.80, 1.25] and its decoded unigram total-variation distance to `p_text` is
  <= 0.30. If no cell is readable, the verdict is "constraints do not bind through decode at
  these operating points" — a not-funded outcome, not proof the idea cannot work.
- *Clause 2, content against the null.* A readable cell shows phonetic content when its pooled
  correct-phone fraction on the 890 exceeds the babble null's p99 by >= 0.05 absolute. The edit
  decomposition (deletions/insertions/substitutions) is reported next to any PER: with deletions
  at 0.63 of the current babble, fixing the rate mechanically converts deletions to
  substitutions and produces a large headline PER drop with zero content gain, so headline PER
  alone is never a pass.
- *Clause 3, paired reading.* Every constrained cell is read as the paired delta against its own
  start's banked unconstrained count-4 and count-0 cells, per start, never pooled across starts.
- Any selection among constrained cells is by label-free statistics only (decoded total
  variation to `p_text`, rate residual); gold reads are reporting-only. Passing clause 2 funds a
  scaled arm as a new planner decision and REOPENS the selection-rule question (the 1g.2
  selector inversion; SAE_1g.md verdict 25's two disagreeing halves) as its own item; it does
  not revive own-minus-donor.

**Status.** Registered 2026-08-22 from the user's proposal with the planner's corrections
(coverage-direction divergence; gate statistics read decoded output because expectations can be
hedged; duration control stays within the frozen H1 law; `r_target` traced to the H1 fit),
user-greenlit the same day at HIGHEST priority. Nothing launched; evidence goes to `SAE_1g.md`.
2026-08-22 later: experiment 1 is launched (`H4CollapseLocateJob.gZ9d6e3E7ZGu`, verifier-confirmed
on disk and running; experiment 2 correctly unbuilt pending the clause-0 ruling). Pre-stated
before any artifact exists: the clause-0 DECISION read is the count-4 row, where posterior and
decode read the same repaired emission table; the count-0 row is context only, because there the
decode reads the start's direct `Q` while the posterior reads its `B` — not two views of one
table. The implementer's three pins (r_target derived from the frozen H1 length-law fit with the
forced deleted-silence-boundary count as the one legitimate excess; gradient reporting as
`lambda_equal` in the softmax parameterization; gradients on the update role with the clause-0
read on the matched 890) are ratified — `SAE_1g.md` Verifier feedback.
2026-08-22 latest (planner ruling: CLAUSE 0 FIRED; 1g.9 CLOSES as the registered off-ramp
outcome). Verifier-confirmed cell by cell from `H4CollapseLocateJob.gZ9d6e3E7ZGu`: at the
pre-stated count-4 decision read, all five starts' posteriors satisfy BOTH proposed targets
(total variation to `p_text` 0.0120-0.0736 against the 0.15 criterion; rate residual -5.5 % to
0.0 % against the 20 % band), and `lambda_equal` runs 8.1e+05 to 1.5e+08 — a penalty on already-
satisfied quantities that the optimizer cannot feel at any sane weight. Per the registered
off-ramp, experiments 2 and 3 DO NOT RUN, their graphs stay unbuilt, no constrained arm and no
lambda is funded, and experiment 1 is the subphase's deliverable (SAE_1g.md approach 15,
verdicts 26-29). The gate licenses "the constraints are not funded at this operating point
because their target quantities are already met" — never "anti-collapse constraints cannot
work". Findings carried forward, verifier-confirmed: the audited babble is DECODE-RESIDENT and
specific to the pseudo-pair start under the frozen local decoder, which consults neither the
fitting LM nor the duration law (verdict 27); posterior unigram closeness is content-blind —
it is satisfied most easily by the least informative channel, so the likelihood column is
mandatory beside any such statistic (verdict 28); decoded unigram closeness is content-blind
too — the random-map control passes it best, so the registered clause 1 admits rather than
evidences and only the babble-null comparison would have discriminated (verdict 29); recorded
as design constraints for any future gate. The DIRECTION FORK exceeds 1g.9's greenlit scope and
goes to the USER: (i) close the phone-repair route — the planner's recommendation, now
strengthened by verdicts 28-29 — or (ii) fund a bounded descriptive decode-stage follow-up
(full-model sequence decode of the count-4 channels on the 890), noting the prior global-beam
stability extension ruled every sequence setting ineligible on stability grounds, which such a
follow-up must first explain. No further 1g work is authorized meanwhile.
2026-08-23 (USER resolves the direction fork): option (ii), strengthened -- the user is
surprised the language model was never used in production decoding and directs that it be
("we did not use LM for decoding at all?? ... this should ofc be used"). The follow-up is
registered as 1g.10 below; the "must first explain" duty attaches to 1g.10's score-margin
read. The "no further 1g work" hold is lifted for 1g.10 only.

### 1g.10 — Full-model (LM-aware) descriptive decode of the audited channels (USER-directed 2026-08-23)

**Purpose.** Answer, by measurement, whether the phone-repair collapse is an artifact of the
LM-blind readout rather than of the channel. The registered sequence decoder (prefix-mass beam
search under the frozen H1 two-state law with the KenLM phone 4-gram `G_dec`) has existed
since 1g.1 but never entered production: the pre-registered global-beam stability gate found
no stable beam at any of the 12 grid points (best adjacent-pair one-best agreement 0.7313
against the required 0.999; best score drift 0.005448 against the required 1e-4 nats per
retained unit), so every banked decode fell back to the frozen local decoder -- a per-unit
argmax over `Q * prior` with run collapse that consults neither the fitting LM nor the
duration law. 1g.9 then located the babble exactly there: pseudo-pair's count-4 emission
table yields healthy posteriors under the full model (TV to `p_text` 0.0120, rate residual
-0.1 %) while its local decode emits 9 of 39 phones at TV 0.687 and rate -50.6 % (verdict
27), and uninformative channels ride the LM marginal (verdict 28) -- which also predicts the
observed beam instability: near-tied hypothesis scores wherever the channel is weak.

**Approach.** Descriptive only, labels as evaluation. Decode the audited count-4 repaired
channels with the registered sequence decoder exactly as frozen: `prefix_beam_decode` under
the frozen H1 law (`p = 0.23560298`; both sub-states may exit -- the H1 law imposes NO
minimum duration, and `d_min>=2` is the psi-track topology, which does not govern this
channel), the deleted-silence boundary policy `force_duration_reset_keep_lm_context_v1`, the
banked KenLM 4-gram (`CreateBinaryLMJob.hvZoC014xnIe`) REPLACING the fitting bigram (the two
are never multiplied), prefix scores summed over duration/state paths (no Viterbi-max), and
the registered grid `lambda = {0.5, 1, 2, 4}` by `beta = {-2, -1, 0}`. Beam is NOT an
eligibility bar here: decode at beam 512 and report the adjacent 256-vs-512 one-best
agreement and score drift as descriptive columns per cell, plus the per-utterance
one-best-vs-runner-up score-margin distribution -- the registered explanation duty: tiny
margins where instability was measured confirm verdict 28's flat-score mechanism; large
margins with persisting instability would instead indicate a decoder defect and block any
reading. The label-free selection surface stays local-only and CLOSED: nothing here promotes
a sequence setting into selection; that would need a new stability law and the user's word.
All decodes are sisyphus GPU jobs on the existing
`H4SequenceDecodeChunkJob`/`H4SequenceDecodeMergeJob` contract, channels bound through the
existing count adapters; no new modeling code.

**Experiments.** (1) pseudo-pair (`H4RepairJob.aeetC3NfgPxB`, count 4), the controlled
reference (`H4RepairJob.x1TyHJMfEVpb`, count 4), and the espum start
(`H4RepairJob.ViPSmq4Am8vX`, count 4) on the 890 selection-role utterances, all 12 grid
points, beams 256 and 512 (espum added 2026-08-23, replacing the same-day two-start form,
because the USER asked for the old PUSM approach decoded with an LM — the espum channel is
its projection into this route; the fairseq-side companion is `PLAN_1F.md` entry 8). A true
count-0 direct-Q sequence decode is NOT mechanically supported (the Q-start artifact schema
`phase1g-h4-q-start-v1` fails the sequence decoder's `phase1g-phone-channel-v1` gate) and is
NOT funded — no new modeling code stands; the count-0 B-table cell is decodable but is a
different object from the banked count-0 direct-Q read and enters only if the planner asks
after (1). (2) The Gate read; extension to the remaining two starts (fingerprint,
random-map) at the same cells only on the planner's read of (1). (3)
Readers are registered jobs printing their conventions: decoded TV to `p_text`, decoded rate
residual to `r_target`, distinct-phone count, correct-phone fraction against the
unigram-matched babble null p99 (the one statistic of the family verdict 29 showed
discriminates), descriptive per-split PER on the same labels-as-evaluation surface as the
funded 1g.2 descriptive read, the two beam-agreement columns, and the score-margin
distribution.

**Gate.** Pre-declared reading; per the standing 2026-08-23 rule this is a measurement, not a
kill gate. The question "did the full model repair the decoded surface" is answered per cell
by (i) readability -- pseudo-pair's decoded TV and rate entering the 1g.9 clause-1 band (TV
<= 0.30, rate within the 0.80-1.25 length-ratio equivalent) that its local decode fails
outright; and (ii) content -- correct-phone fraction clearing the babble-null p99; with the
reference channel as the positive control (it must stay readable under the sequence decoder,
else the decode path itself is broken and no cell is readable). The beam-agreement and
score-margin columns qualify every cell. Whatever the outcome, the result feeds the USER's
next direction decision on the phone-repair route; nothing closes automatically, and no
selection surface opens. The reporting rule goes verbatim into the producing job's docstring
before any result exists.

**Status.** REGISTERED 2026-08-23 on the user's direction word. Implementation not started;
the implementer proposes shard/resource sizing in `SAE_1g.md` State before launch.
2026-08-23 later (USER: "maybe even old PUSM approach should be decoded with LM?"):
experiment (1) is amended by replacement to include the espum start's count-4 cells (see the
replacement note in Experiments); the extension clause now covers fingerprint and random-map
only. The fairseq-side LM-decoded PER of the PUSM/ESPUM arms is registered separately as
`PLAN_1F.md` entry 8 and awaits the user's launch word there.
2026-08-23 latest (planner ruling on the launched build; verifier round in `SAE_1g.md`). The
launch is VERIFIED AND ACCEPTED: constants trace (banked KenLM `CreateBinaryLMJob.hvZoC014xnIe`,
frozen H1 `Phase1gH1Job.HbxKiuBTJ8aN`, count adapters, selection resource contract
`H4ResourceContractJob.kyMk7fwm027C`, registered grid and beams, 2 GiB / 2 h at the contract's
1.5x), the docstring conventions are ratified (1g.9's `p_text`/`r_target` imported not restated;
the 1g.2 `edit_distance` as the one alignment convention; the per-cell length-and-histogram-
matched babble null with 1,000 draws at seed 42 answering verdict 29 by construction; the FLAT
SCORES vs DECODER DEFECT SUSPECTED explanation duty). On the flagged budget (each single-cpu
chunk occupying its own booster node; order 900-2,100 node-hours), the planner TAKES THE OFFERED
CUT: beam 256 is restricted to ONE fixed shard per cell -- the canonical heaviest selection-role
shard by retained units, the same index for every cell, mirroring the historical beam-table
convention -- dropping 1,152 of the 2,304 chunks; the adjacent-pair agreement and drift columns
are then computed on that shard alone and every quote of them names its utterance count. The
reader's key set is re-registered accordingly; already-created surplus beam-256 jobs are deleted
per the standing delete rule. Beam 512 runs at full registered scope -- the decoded surfaces,
margins, babble nulls and PER all read from it and are uncut.

2026-08-23 result (1g.10 COMPLETE; the table is BLOCKED by its own explanation duty on the
DECODER-DEFECT branch; verified; 1g.10a defect diagnostic REGISTERED). All 1,332 chunks and 36
merges finished with zero error markers and `H4FullModelDecodeReadJob.MXhi20TtG1I0` landed on
the pre-registered duty's second branch: ZERO of 36 cells reaches the 0.999 adjacent-beam
agreement (min 0.2222, median 0.6111, max 0.8889 on the probe's 27 utterances) while ZERO of 36
has a median score margin at or below the 1e-3 nats-per-retained-unit flat threshold (min
1.210e-03, median 4.345e-03) -- wide margins with persisting instability, so the report prints
"DECODER DEFECT SUSPECTED -- no cell of this table may be read until that is explained" and no
cell is read (verdict 30). The positive control is healthy as a decoder-health observation only
(verdict 31, fenced): sensible content flows while adjacent beams disagree on roughly one
utterance in three. The implementer's 27-vs-28 correction is accepted from the artifact. The
duty fired exactly as designed and blocks reading, not spend.

1g.10a -- REGISTERED (pre-results): the cross-beam defect diagnostic, from BANKED data only (no
new decoding). The chunk artifacts bank per utterance `one_best_symbol_ids`, `one_best_text`,
`decoder_log_score`, `retained_prefix_log_mass`, `n_audio_units`; the diagnostic joins, per cell
and probe utterance, the beam-256 probe chunk with the beam-512 chunk of the same shard.
STEP ZERO, before anything runs: state from the decoder's code which rule selects the one-best
-- the tests below are registered ONLY for a beam-independent per-sequence score (argmax of
`decoder_log_score` over completed hypotheses); if the selection rule is beam-dependent (e.g.
mass-renormalized), STOP and report, and the planner re-rules the invariant. TEST A (scoring
determinism): wherever the two beams' winners are the SAME symbol sequence, their banked scores
must agree to 1e-9 nats per retained unit; any violation = DEFECT CONFIRMED (nondeterministic
scoring). TEST B (pruning monotonicity): wherever the winners differ, score(beam-512 winner)
minus score(beam-256 winner) must be >= -1e-9 nats per retained unit; any violation = DEFECT
CONFIRMED (widening the beam lost a better-scoring hypothesis). CONTEXT, never gating: the
distribution of the 512-winner's score gain over the 256-winner; the disagreement-vs-flat
cross-tab per utterance at the registered 1e-3 flat bar; agreement stratified by lm_scale
(mechanism check: pruning error should grow with the LM weight). CONSEQUENCES, pre-registered
before any diagnostic number exists: (i) any DEFECT CONFIRMED -> the table stays blocked, the
code investigation becomes its own registered item, no cell is quoted anywhere; (ii) zero
violations -> the suspicion is DISCHARGED AS SEARCH ERROR (beam 256 under-beams; the instability
lives between the beams, not in a broken scorer), the duty's "until that is explained" clause is
satisfied, and the beam-512 table becomes readable as DESCRIPTIVE with each quoted cell carrying
its own 256-vs-512 agreement beside it; (iii) under (ii) only, 1g.10b is AUTHORIZED -- a
beam-1024 probe on the same single shard (36 chunks, the 27 probe utterances, same contract) --
and CROSS-CHANNEL comparisons may be quoted only from cells whose 512-vs-1024 agreement is at
least 26 of 27 (at most one flip; the 0.999 bar is not reachable at n=27 and is deliberately not
reused for the probe). The reporting rule goes verbatim into the diagnostic module's docstring
before any result exists. The route question ("does LM-aware decoding reopen phone repair?")
stays open and with the USER; nothing here closes it.

2026-08-23 re-rule (1g.10a step zero came back STOP-AND-REPORT; TESTS RE-REGISTERED). The
implementer's stop was the registered behavior and is endorsed. The planner verified both code
claims directly (`channel_h.py`): the one-best IS the registered argmax form
(`max(final, key=final.get)`, no renormalization), but the score it selects on is a PRUNED
PATH-SUM -- `prune` ranks prefixes by the logsumexp of their surviving states and discards whole
prefixes, so a sequence's banked score sums only the alignment paths that survived pruning and
is legitimately beam-dependent. TEST A as registered is VOID BY CONSTRUCTION (it would convict a
decoder behaving as designed: only 117 of 588 same-winner scores agree to 1e-9); TEST B's
premise falls with it. The scouting counts (588 same-winner of 972; 13 same-sequence scores
LOWER at beam 512, largest -0.299 nats; 8 utterances with lower TOTAL retained mass at 512) are
received as description, not verdicts -- and non-nested kept sets ARE consistent with this
pruning rule (later-timestep rankings depend on earlier pruning), so none of these is by itself
a defect. REPLACEMENT INVARIANTS, pre-registered before any verdict (replaces TESTS A/B,
2026-08-23, because the registered score is a pruned path-sum, not a beam-independent
per-sequence score):
- TEST D (determinism): inside the diagnostic job, decode each probe utterance TWICE at beam 256
  through the identical entry point and inputs, for THREE cells -- the minimum, median and
  maximum observed-agreement cells (a disclosed bug-hunt selection, never a quoted comparison).
  Sequences must be identical and scores equal to within 1e-12 nats. Any difference = DEFECT
  CONFIRMED (nondeterminism).
- TEST U (upper bound): for every one of the 972 probe utterance-cells and BOTH beams' banked
  winners, the banked `decoder_log_score` must be <= the EXACT forced score of that sequence --
  the all-alignments forward sum over (sequence position, duration state) x time under the same
  frozen H1 law, KenLM 4-gram, silence policy, count adapter and the cell's lm_scale/beta, with
  the LM and insertion terms applied once per emitted symbol exactly as the decoder applies them
  -- plus 1e-6 nats absolute. A pruned sum exceeding its own unpruned total is impossible; any
  violation = DEFECT CONFIRMED. This is also the invariant that polices the 13/8 scouting
  anomalies; they get no separate branch.
- EXACT-CURRENCY COMPARISON (context, and the discharge reading's companion): where the two
  beams' winners differ, print the distribution and sign counts of exact(w512) minus
  exact(w256), raw and per retained unit -- the beam-independent currency in which "which beam
  found the better sequence" is well defined.
CONSEQUENCES: any DEFECT CONFIRMED -> branch (i) as registered (table stays blocked, code
investigation becomes its own registered item). Both tests pass -> the suspicion is DISCHARGED
AS A DESIGNED-IN APPROXIMATION -- the score is a pruned path-sum and adjacent-beam disagreement
is pruning reshuffle, not a broken scorer -- satisfying the duty's "until that is explained"
clause; branches (ii) and (iii) then stand exactly as registered, with the exact-currency sign
distribution printed beside the discharge. The reporting rule goes verbatim into the diagnostic
module's docstring before any result exists.

2026-08-23 discharge (1g.10a COMPLETE: DISCHARGED; consequences (ii) and (iii) in effect;
1g.10b registered). `H4CrossBeamDefectJob.2pV5rHuWJW3d` (verified: docstring carries the
re-ruled rule verbatim; the exact score is the pre-existing `marginal_path_log_score` the chunk
jobs already bank, so the identity is not self-referential; inputs trace to the registered
KenLM, frozen H1 and banked chunks; tests 9/9): TEST D re-decoded 81 utterances (three
disclosed cells, twice each) with ZERO violations at 1e-12; TEST U checked all 1,944 banked
winners against their exact unpruned totals with ZERO violations at 1e-6. Exact-currency
context: of 384 differing-winner cases, exact(w512) beats exact(w256) in 352, loses in 32
(median gain +0.0109 nats per retained unit) -- the wider beam mostly finds genuinely
better-scoring sequences, the search-error signature. THE DUTY'S BLOCK IS DISCHARGED: the
1g.10 beam-512 table is READABLE AS DESCRIPTIVE, every quoted cell carrying its own 256-vs-512
agreement beside it; verdict 30 stands as written for its date (the block was correct until
explained). CROSS-CHANNEL comparisons stay unquotable pending 1g.10b. 1g.10b -- REGISTERED:
beam-1024 probe on the same contract shard (index 28, 27 utterances), all 36 cells, the
existing chunk class at the same contract, agreement and drift computed against the beam-512
chunk of the same shard, reader extended with the 512-vs-1024 columns; the pre-registered
quoting bar stands: cross-channel comparisons only from cells whose 512-vs-1024 agreement is
at least 26 of 27, every such quote naming its 27-utterance support. Nothing else changes; no
selection surface opens; the route question stays with the USER.

2026-08-23 ruling (1g.10b beam-registration conflict: OPTION (b) -- a dedicated probe class;
option (a) rejected). The implementer's blocker is verified in the code by the planner:
`DECODER_BEAMS = (64, 128, 256, 512)` is load-bearing outside 1g.10, `H4GlobalBeamTableJob`
takes its `cells` as a hashed constructor argument, and the 1g.10 read consumes that table's
`out_global_beams` -- so extending the beam tuple (option a) would orphan the banked global-beam
table cited in closed verdicts AND the just-discharged `H4FullModelDecodeReadJob.MXhi20TtG1I0`,
while demanding 48 new jobs first. REJECTED. The registration line "the existing chunk class at
the same contract" is amended by replacement (2026-08-23, because the beam tuple is load-bearing
outside this subphase and extending it orphans verdict-cited artifacts): 1g.10b uses A DEDICATED
PROBE JOB CLASS under these identity guards, so "same decode" is checkable rather than asserted:
(1) the class IMPORTS `prefix_beam_decode`, `_load_h1_units`, `_retained_runs`, `_load_channel`
and `_validate_contract` from their existing modules -- no copied decode or input code; (2) its
beam is hard-pinned to 1024 plus a beam-512 PARITY mode, and its docstring states it exists
solely for 1g.10b; (3) a PARITY CELL runs first: the probe class at beam 512 on the contract
shard for the median observed-agreement cell (the disclosed, key-tie-broken cell 1g.10a already
named) must reproduce the banked beam-512 chunk's one-best sequences and scores exactly, and the
1g.10b reader REFUSES to read any beam-1024 column unless the parity artifact matches. Option
(c) stays rejected for the reason given (re-hashes all 1,332 chunks). Everything else stands as
registered: 36 cells x contract shard 28 (27 utterances), agreement and drift against the banked
beam-512 chunk of the same shard, the 26-of-27 cross-channel quoting bar, every quote naming its
27-utterance support.

2026-08-23 result (1g.10b COMPLETE: parity PASS, quoting bar NOT CLEARED -- 0 of 36 cells;
cross-channel quoting STAYS CLOSED; beam escalation NOT FUNDED). Verified from
`H4Beam1024ReadJob.tKbQ0MHLdX03`'s own report: the parity cell reproduces the banked production
chunk exactly (the dedicated-class mechanism is validated), and 512-vs-1024 one-best agreement
on the 27 probe utterances reaches at best 24 of 27 (0.889), median 0.704 across the 36 cells,
against the pre-registered bar of at least 26 of 27. The search is converging but slowly:
median agreement rose from 0.611 (256-vs-512) to 0.704 (512-vs-1024) -- roughly +0.09 per beam
doubling -- and per-unit drift fell in 25 of 36 cells and ROSE in 11 (CORRECTED 2026-08-23,
replacing this entry's original "fell everywhere", which the implementer's verdict 35 caught
and the planner re-verified from both artifacts; the 11 rising cells strengthen, not weaken,
the no-escalation reading below), so on even a generous linear read the bar sits
about three more doublings away (beam 8192) at doubling cost per step. RULED: no further beam
escalation is funded; the bar fired as designed and gates QUOTING, not the route. The 1g.10
grid's standing currency is therefore: WITHIN-CHANNEL comparisons, per-utterance paired with
agreement disclosed (1g.10c's instrument), remain readable; CHANNEL-VS-CHANNEL rankings from
this grid are not quotable and will not become so at feasible beams under the exact-match bar
-- if a cross-channel claim is ever needed, it gets its own registered paired per-utterance
instrument with the beam uncertainty priced in, on the USER's word. Nothing here touches the
route question, which stays with the user.

2026-08-23 extension (USER: "insertion bonus makes sense, please try that" -- 1g.10c
REGISTERED). PURPOSE: the 1g.10 grid is truncated at its best edge -- beta 0 is the best
insertion-penalty column at every lm_scale, and the high-lm_scale failure is deletion (each
emitted phone pays lm_scale times its LM log-probability plus beta, both negative, with nothing
paying it back; output length falls to 50-65 pct of reference at lm_scale 4). 1g.10c measures
whether a POSITIVE insertion bonus recovers those deletion losses. CELLS: lm_scale in {1, 2} x
beta in {+1, +2} on the two content-bearing channels (`controlled/reference`,
`real/espum_seed0_update30000`) -- 8 cells; `real/pseudo_pair_seed0` is EXCLUDED because its
failure is a flat likelihood, not deletion; a bonus there buys only more fluent hallucination.
Beam 512 at the full 890-utterance selection role, 32 shards per cell; one beam-256 probe per
cell on contract shard 28 for the agreement and drift columns (27 utterances, named in every
quote). MECHANISM PRE-RULING, pre-empting the 1g.10b conflict: `DECODER_GRID` is load-bearing
exactly as `DECODER_BEAMS` was (`decoder_grid_rows()` with the len==48 assertion; the
global-beam table hashes its cells), so EXTENDING `DECODER_GRID` IS PRE-REJECTED (same blast
radius -- it orphans the banked global-beam table and the discharged 1g.10 read), and the
option-(b) pattern is PRE-APPROVED: a dedicated extension class importing the same decode and
input functions (imports-not-copies enforced by a source-greping test), its grid points
hard-pinned to the four new (lm_scale, beta) pairs plus a parity mode at a registered point,
and a PARITY CELL that gates everything -- the class at (lm_scale 1, beta 0, beam 512) on the
contract shard for `controlled/reference` must byte-reproduce the banked production chunk's
one-best sequences and scores, and the reader emits no extension column otherwise. READING,
pre-registered (the standing paired-data rule applies): the headline question per new cell is
WITHIN-CHANNEL against the same channel's (same lm_scale, beta 0) production cell --
per-utterance paired correct-phone deltas over the shared 890 utterances, 10,000-resample
bootstrap CI at seed 42, never two pooled numbers -- with the pooled description (TV, length
ratio, phone inventory, PER, per-cell length-and-histogram-matched babble null, 1,000 draws
seed 42) banked beside it. Every quote carries the cell's 256-vs-512 agreement. NO CELL SELECTS
ANYTHING: the label-free selection surface stays closed, and an operating-point choice, if the
route continues, uses a label-free selector with the label-oracle best disclosed beside it.
QUOTING: same regime as 1g.10 -- descriptive with agreement disclosed; cross-channel
comparisons stay gated by 1g.10b's 26-of-27 bar, and extension cells enter cross-channel quotes
only after their own beam-1024 check if that is ever wanted (not built now). COST: 256 beam-512
chunks + 8 probes + 8 merges + 1 parity cell, about a quarter of the 1g.10 bill; the
implementer proposes resources in `SAE_1g.md` State per the sizing convention. The reporting
rule goes verbatim into the producing module's docstring before any result exists.

2026-08-23 launch ruling (1g.10c build ACCEPTED; the flagged resample convention is RULED:
STRATIFIED primary, unstratified beside as sensitivity -- decided before any statistic exists).
Ratified: the sizing (4 h / 4 GiB per chunk against the contract's 2 h / 2 GiB, justified by a
real mechanism -- a positive bonus lengthens hypotheses and KenLM prefix scoring is linear in
hypothesis length, so the beta-0 cells' measured time is not a safe bound); the production
merge reused unchanged with its own coverage and validation applying to the extension cells;
the three guards as pre-approved (imports-not-copies by source-grep test, the grid hard-pinned
with out-of-set pairs refused, the parity cell suppressing every extension column); the
excluded pseudo-pair row refused in code carrying the registration's reason; and the paired
read being genuinely paired with non-finite deltas refused. THE OPEN CONVENTION: the
registration was silent on stratification. The implementer's literal unstratified read is
honest and conservative, but the bed's dev-clean/dev-other composition (432/458) is fixed by
construction, so replications of this bed cannot vary the split proportion -- the resample
should hold it fixed -- and the family convention (`h4_harness._bootstrap_content_values`,
"utterances within fixed evaluation splits") already stratifies. RULED: the PRIMARY interval
resamples utterances WITHIN each split at the fixed 432/458 counts (n_boot 10000, seed 42
unchanged); the unstratified interval prints beside it as a named sensitivity column, and the
payload states both conventions. Only the reader re-hashes; the 273 decode jobs are
convention-independent and untouched.

2026-08-23 1g.10c result (COMPLETE: parity PASS, the two rows SPLIT BY SIGN -- verdicts 36-37,
`H4InsertionBonusReadJob.da3bGeQIkS0R`, verifier-checked cell for cell against the artifact).
On the positive control the bonus recovers phones at every extension point (paired delta +0.0222
to +0.0555, every stratified interval excluding zero, best at lm_scale 2 / beta +2 improving
74.0 percent of utterances); on the real ESPUM arm it REMOVES them at three of four points
(-0.0090 to -0.0294, intervals excluding zero) and straddles zero at the fourth. Decoded length
rises with beta in every cell on both rows, so the bonus does what it is priced to do -- the
recovered length is content only where the emissions carry content. CONSEQUENCES. (i) The 1g.10
deletion mechanism (each emitted phone pays lm_scale*LM + beta) is confirmed causal, and its
converse is now measured: on the real arm the LM-pressure deletions were not suppressed content,
because paying to reinsert brings back wrong phones. (ii) The truncated-grid concern from the
1g.10 read is DISCHARGED -- the grid's best-at-the-beta-0-edge pattern does not hide a better
real-arm operating point beyond the edge. (iii) No further decode-parameter probes are funded on
this harness: beam escalation closed by 1g.10b, the insertion axis closed here, and the
stratification ruling closed as immaterial by verdict 37 (every bound within 1e-4 of the
unstratified read -- reported, not load-bearing). 1g.10c CLOSES. The decode route's remaining
live measurement is the fairseq-side companion (`PLAN_1F.md` entry 8, cells 1-2 running), which
decodes the real generator with the real flashlight decoder rather than this channel harness.

### 1g.11 — Continuous-emission twin of the table channel (USER-proposed and greenlit 2026-08-23)

**Purpose.** Attribute the phone-route failure between the categorical table's lack of geometric
inductive bias and the LM-driven generative training paradigm itself, by swapping ONLY the
emission model: same HMM shape, duration law, fitting LM, repair procedure, and decode readout,
with the categorical `B(unit|phone)` replaced by one low-capacity diagonal Gaussian per emission
row over continuous segment features. The claim under test is INDUCTIVE BIAS, not information
loss — information loss is already contradicted by banked anchors (the memoryless oracle on the
same `seg12.5` stream reaches 0.4148 while unsupervised repair lands 0.8580–0.8946, barely above
the random-map control; the supervised continuous context-kernel probe reaches 0.3565, a modest
ceiling gain). Mechanism: a 500-way table can represent ANY unit-phone assignment, so the EM
landscape is dense with content-free optima that satisfy marginal and LM statistics (verdict
28's mechanism); a tied diagonal Gaussian can only represent geometrically coherent feature
regions, pruning EM's search space to acoustically plausible solutions. Disclosure: the same
single-variable swap ran once on the 3a scorer bed (`PLAN_3A` §5b.1 cell M2) and was
INDISTINGUISHABLE at the decision temperature (delta eta +0.0027 [-0.027, +0.035]; worse at
T=1.0) — but M2 asked whether continuous observations improve a TRAINED scorer's ranking, not
whether they rescue unsupervised EM from content-free optima; that question is untested and is
this subphase's.

**Approach.** Observations are the continuous twin of the `seg12.5` stream at the TOKEN
segmentation (replaces the bare "segments" wording, 2026-08-23, because the frozen pooling
has a Ward layer the registration did not name: raw codebook runs are merged by adjacent-pair
Ward cost to a per-utterance target, each surviving segment is assigned a centroid and its
code written over the segment's frames, and the discrete stream is the run-length view of
those codes -- 2,184 of 921,432 Ward segments, 0.237 pct, merge in that view): one vector per
token of the frozen discrete stream, the mean over exactly that token's frames of the same
normalized PCA-space frame vectors the frozen unit pipeline pooled and assigned
(normalization, PCA basis, and centroids fitted on the 2,849 dedicated train utterances only,
read verbatim from the frozen artifacts). Per-utterance count identity with the discrete
token sequence holds by construction at this reading, which is what the paired per-utterance
read requires. Fidelity rests on two checks that do not substitute for each other: the
PIPELINE CHECK, asserted exact -- at the Ward segmentation the re-assigned segment means
reproduce the frozen stream bit-for-bit through the pipeline's own pooling code, which is
what binds features, basis, and segmentation to the frozen pipeline rather than a lookalike
-- and the TWIN CHECK, reported never asserted -- the share of tokens whose mean re-assigns
to the frozen code; given a passing pipeline check a non-absorbed token cannot mismatch, so
any mismatch outside the absorbed-token set is a STOP, while absorbed-token mismatches are
expected (their means span two Ward segments). The basis is the frozen one in FULL: it
measured 96-dimensional (read off the artifact by implementer and verifier independently), so
the registered leading-128 pin (replaces the truncation clause, 2026-08-23, because 128
exceeds the frozen dimension) is discharged as a capacity ceiling plus a no-refit instruction
that 96-of-96 satisfies, and the full-dimension sensitivity cell is dropped as vacuous -- it
coincides with the primary. Components are standardized by a scale fitted over SEGMENT
vectors (the objects this model observes; the fit/assign split governs the fit population,
not the object type) on the same 2,849 dedicated train utterances and shipped for reuse. Emissions: one diagonal Gaussian
per emission row of the frozen topology; PRIMARY covariance is one shared diagonal across all
rows, with per-row diagonals as the single disclosed relaxation; both carry the M2 variance
floor (`logvar = 2*min_log_std + softplus(raw)`, `min_log_std = log 0.1`, or the twin pinned in
the producing docstring) — without it a state wins by variance collapse, which would fake a
collapse verdict. Duration and shape stay frozen as H1 fixed them (rate matched by construction
on `seg12.5`); `G_fit` and `G_dec` unchanged. Starts: each discrete start maps through the
codebook geometry — each emission row's mean initializes to that row's `B`-weighted mix of
centroid vectors, `mu = sum_u B(u) c_u` — so all five 1g.2a starts and their controls get
paired table/Gaussian initializations. A log-density and a log-mass are different currencies
(the M2 rule): absolute likelihoods are void across arms and every comparison reads decoded
output.

**Experiments.** All model-forward computation runs as sisyphus GPU jobs; derived statistics
come from registered readers that print their conventions.

1. *Feature build and fixture.* The layer-15 dump, the frozen-transform read, and the
   segment-mean job, with the asserted segment-count identity and (if a refit was needed) the
   exact-reproduction acceptance check.
2. *Gaussian repair cells.* The five 1g.2a starts at repair counts 0 and 4 under the
   constrained update rule (implementer's choice, pinned in the docstring; the tiny enumerated
   example extended to verify the update improves the criterion); the per-row-covariance
   relaxation on the selected real start only. Decode every cell with the frozen local decoder
   twin (per-segment argmax over Gaussian log-density times prior, with run collapse) — the
   same readout family as the banked table cells, which is what makes the paired read valid;
   the table arm is NOT re-decoded (the audited 1g.2a one-bests are the comparator).
   Sequence-decoded (LM-aware) cells are descriptive and readable only after passing the same
   stability duty 1g.10 imposed.
3. *Nulls and controls.* The 1g.9 experiment-3 babble null, never built there, is registered
   again verbatim (per utterance 100 random symbol sequences of the decoded length i.i.d. from
   `p_text`; mean and p99 of pooled correct-phone fraction and PER, alignment convention
   printed); plus the continuous observation null — each segment's vector redrawn i.i.d. from
   the corpus segment-vector marginal (length-preserving, structure-destroying), fitted and
   decoded exactly like the real arm.
4. Evaluation on the frozen 890 selection utterances with gold as EVALUATION ONLY; the
   1,112-utterance evaluation fifth stays sealed; any selection among cells is by label-free
   statistics only.

**Gate (pre-registered 2026-08-23, before any run).**

- *Clause 1, admission.* A cell is READABLE when its decoded length ratio to gold is within
  [0.80, 1.25] and its decoded unigram total variation to `p_text` is <= 0.30 — admission
  only, never evidence (verdict 29: a content-free control passes this best).
- *Clause 2, content.* A readable cell shows content when its pooled correct-phone fraction on
  the 890 exceeds the babble null's p99 by >= 0.05 absolute, with the edit decomposition
  reported beside every PER (converting deletions to substitutions fakes headline PER drops
  with zero content gain).
- *Clause 3, attribution — the decision read.* The paired per-utterance delta of correct-phone
  fraction, Gaussian cell minus its own start's banked table cell, at count 4, per start,
  bootstrap interval over utterances. The continuous route is funded (a NEW planner decision)
  only if the selected real start's interval excludes zero in the Gaussian arm's favor AND
  neither content-free arm (random-map-initialized Gaussian, observation null) shows a
  comparable gain — a gain that appears on controls is a rate or length artifact, not content.
- *Clause 4, honesty.* The fraction of variance components at the floor is reported per cell; a
  fit living on the floor is surfaced, never silently read. Failure of clause 3 licenses
  "continuous emissions are not funded at this operating point; evidence toward the training
  paradigm as the binding constraint, jointly with the banked oracle gap (0.4148 achievable on
  this stream, 0.85+ found)" — never "the paradigm cannot work"; the attribution is
  conditional on the shared `seg12.5` segmentation both arms inherit.
- The wav2vec-U-faithful variant (change-point segmentation on raw codes, adjacent-segment
  pooling, duration refit at that rate via the per-rung H1 procedure) is NOT funded here; it is
  the registered follow-up if clause 3 passes, or a separate planner decision if the shared
  segmentation itself comes under suspicion.

**Status.** Registered 2026-08-23 from the user's proposal (test whether the failure is caused
by discrete k-means IDs as acoustic observations rather than by the HMM/LM training paradigm
itself), with the planner's corrections accepted in the same exchange: the claim is inductive
bias, not information loss; segmentation shared with the comparator rung rather than
re-derived; the frozen PCA basis truncated to 128, never refit; M2 disclosed. USER-greenlit
2026-08-23 ("I greenlight 1g 11") — the 1g work hold, previously lifted for 1g.10 only, is
lifted for 1g.11 by that word. Nothing launched; evidence goes to `SAE_1g.md`. Entry 8 cells
3-4 (`PLAN_1F.md`) remain a separate open user decision.
2026-08-23 result (experiment 1 COMPLETE and VERIFIED; both implementer flags RULED as the
Approach amendments above). `G11ContinuousSegmentsJob.hImWJG0X4eZh` (speech-llm 16b1063):
8,416 utterances, 919,248 token vectors, 921,432 Ward segments, frozen PCA dimension 96 kept
in full, scale fitted on the 2,849 dedicated train utterances over 448,204 segment vectors.
PIPELINE CHECK passed on all 8,416 utterances (bit-for-bit at the Ward segmentation, through
the pipeline's own pooling code); TWIN CHECK 919,248 of 919,248 tokens (100.0000 pct)
re-assign to their frozen code -- above the guaranteed floor (absorbed tokens need not match;
all did), so no STOP question arises. The verifier independently re-read the quantizer
artifact (pca components (96, 1024), centroids (500, 96)) and re-ran
`g11_continuous_test.py`: 19/19. One standing pin for the unbuilt experiments 2-4: the
Gaussian repair cells consume the same RETAINED token stream the table arm consumes (after
the frozen silence mask), selected from the twin stream by per-token keying -- the twin's
full-stream build is correct because masking is a downstream selection, and the repair
registration must state the retained counts it actually trains on.
2026-08-23 (constrained update rule RATIFIED before experiment 2 is built; speech-llm 537f968,
module `sae/g11_gaussian.py`). Three design points accepted as pinned: (i) the variance-floor
clamp is the exact constrained maximizer, not an approximation -- the M-step objective is
unimodal per variance component with its unconstrained maximum at the weighted second moment,
so on `{var >= v_min}` the maximizer is `max(var_hat, v_min)` and EM monotonicity survives;
checked on a fixture where the floor binds. (ii) ONE RECURSION SERVES BOTH ARMS:
`channel_h.marginal_forward_backward` gained an optional `emissions_by_time` argument so the
Gaussian arm runs the table arm's own forward-backward -- the attribution claim "only the
emission model changed" requires shared DP code, and this is the registered way it holds.
(iii) `gaussian_local_decode` replaces exactly the per-unit-ID lookup in
`channel_h.frozen_local_decode`; the silence rule still keys on the token's FROZEN unit ID, so
silence handling is identical across arms by construction. Verifier checks on the seam: the
`source_identity` mechanism is a runtime guard (constructor attribute plus run-time
assertion), not part of the sisyphus hash, so no banked hash moves; the only live manager is
D9.1's, whose graph does not consume `channel_h`; the only channel_h-family dirs without
finished markers are five known debris dirs (three H1-era orphans, the inert
`vJSHAkECj8hH`, one `.cleared` rename), so no constructed-but-unrun job can trip the
assertion. Suites re-run by the verifier: `g11_gaussian_test` 21/21,
`h4_context_engine_test` 36/36, `h4_collapse_locate_test` 6/6 -- the pre-existing consumers
of the shared recursion pass unchanged. Experiment 2's job and config may be built.
2026-08-23 sizing clarification (planner, pre-empting a false deviation read): the
registration's "all model-forward computation runs as sisyphus GPU jobs" covers model-forward
computation only -- the layer-15 feature dump, which ran as one. The Gaussian EM repair is
numpy on precomputed features, measured at 5.9 s / 1.8 GB per 1,024 update utterances through
four updates, and correctly runs as a sisyphus CPU job; that is compliance, not deviation.
2026-08-24 GATE VERDICT: CLAUSE 3 FAILS -- continuous emissions are NOT funded at this
operating point; evidence toward the training paradigm as the binding constraint, jointly with
the banked oracle gap (0.4148 achievable on this stream, 0.85+ found); never "the paradigm
cannot work"; the attribution is conditional on the shared `seg12.5` segmentation both arms
inherit. Experiments 2-4 complete and verified (`G11GaussianRepairJob.NogH62uMEI7T`,
`G11ObservationNullJob.orOc9h6K3cuR`, `G11EvaluateJob.sWoS1bP4Nd12`; tables in `SAE_1g.md`
approach 19, verdicts 45-51). Clause 1 excluded 7 of 24 cells and is confirmed admission-only
in BOTH directions (it excluded the two highest-content cells and admitted both
observation-null cells). Clause 2: only the table reference cell shows content; the Gaussian
reference cell carries content (margin +0.4952/+0.3990 over the babble bar) but is
inadmissible on decoded length -- verdict 47's reading is part of this record. Clause 3: the
selected real start's paired gain over its own banked table cell is +0.0098 [+0.0058,
+0.0137] (first condition passes); the content-free random-map Gaussian control gains +0.0251
[+0.0202, +0.0302] against the same comparator. Two planner rulings, both required because
the evaluate job correctly declined to rule: (i) "comparable" carries no registered number,
and needs none here -- a control gain that EXCEEDS the arm's with a non-overlapping interval
is beyond any admissible reading of "comparable", so no threshold choice can rescue the arm;
the second condition fails. (ii) The clause-3 control is itself clause-1 inadmissible
(decoded length 0.7832 of gold); the registration scopes clause 1 to clause-2 readability and
names clause 3's controls with no admission precondition, so the control counts as
registered. Filtering it post hoc would be an unregistered gate edit made after seeing that
it flips the verdict, and would delete the very evidence of the length pathology the control
exists to expose (verdict 46: the positive-control start LOSES under the swap, -0.0208
[-0.0260, -0.0154], identifying the small real-start gains as length/rate effects, not
content; verdict 50: the arm does read the acoustics -- the observation null costs it 0.0761
of pooled correct-phone fraction -- so the finding is that the categorical table already
extracts as much, not that the swap was inert). Clause 4 clean everywhere (floor share
0.0000; the babble bar is stable, 100-draw vs 1,000-draw within 6.1e-04 in the worst cell
against a smallest clause-2 gap of 0.0090). The registered wav2vec-U-faithful follow-up is
NOT funded (it required a clause-3 pass); whether the shared segmentation itself comes under
suspicion is a separate planner decision, currently not raised. 1g.11's question is ANSWERED;
what the answer means for the phone route's direction is the USER's call and joins entry 8
cells 3-4 (`PLAN_1F.md`) on their desk.
2026-08-24 (verifier's ad-hoc unigram read of the banked hypotheses, on the USER's question
"do the Gaussian outputs look less collapsed" -- descriptive, decides nothing, and any
load-bearing use needs a registered reader; convention: unigram over all decoded symbols of
the 890 selection fold, per cell, from `G11GaussianRepairJob.NogH62uMEI7T/output/
hypotheses.json` and the banked `H4LocalDecodeJob` one-bests; parse validated against the
banked distinct counts and the evaluate table's length identities). Only pseudo_pair ever
collapses to few phones, in BOTH arms (3 distinct at count 0). At count 4 the Gaussian arm
ESCAPES it (39 of 39 phones, top-1 share 0.074, unigram perplexity 32.4 vs gold-shape 28.4)
while the table stays stuck (9 phones, top-1 0.52, perplexity 3.6). All other cells in both
arms use 35-39 phones at roughly gold-shaped unigrams; the Gaussian cells run systematically
MORE uniform than gold (perplexity 31-36, top-1 0.05-0.09). The trap, and why this changes
no verdict: unigram diversity is content-blind here -- the content-free random-map Gaussian
control has the MOST uniform output of the whole table (perplexity 35.8), and with a tied
covariance the local decode is a nearest-mean classifier whose argmax regions tile the
feature space, so many-phone output is close to structural. The Gaussian family fixes the
degenerate-output pathology and still finds no content (espum's near-gold unigram is
substitution-dominated at 0.85 PER) -- consistent with, and mildly sharpening, the clause-3
attribution toward the objective rather than the readout pathology.

## 6. Deliverables ladder

| Step | Deliverable | Decision it enables |
|---|---|---|
| 0 | Label-free one-segment rejection, gold-duration diagnostics, and hard-anchor verdicts | Reuse valid Phase-1g work at its actual scope |
| 1 | Correct phone decoder and repair curve with construction-only seeds and controls | Decide whether weak phone seeds can be repaired |
| 2 | Validated content-sensitive training and selection score | Select without transcripts |
| 2a | Conditional matched-2/3/4 diagnostic and coherent matched-4 H4-LM arm before evaluation | Resolve a valid but inert/non-deployable bigram repair without conflating decoder order |
| 2b | Locate-the-collapse diagnostic and anti-collapse constrained-repair probe (1g.9) | Decide whether the phone-repair collapse is fixable at the objective, lives elsewhere, or closes the route |
| 2c | Full-model (LM-aware) descriptive decode of the audited channels (1g.10) | Decide whether the collapse is a readout artifact and whether the phone-repair route stays open |
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
