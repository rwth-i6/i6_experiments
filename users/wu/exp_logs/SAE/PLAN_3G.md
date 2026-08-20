# PLAN_3G — Z-track: from-scratch fully-unsupervised joint loop

Sub-plan of PLAN §3g (moved here 2026-08-14 because the track outgrew a page: two closed arms
and one live registration). The Z-track asks the user's question "real unsupervised ASR
without GAN": start the joint GRPO loop from zero paired data — text-only LBS-SFT Qwen donor,
min-duration psi co-trained from scratch on the policy's own rollouts, LM prior at the bed's
units-normalized settings — and classify what happens. No gold pairs, no GAN init, no seed
SFT anywhere in training; labels evaluate only (standing quarantine). Log: `SAE_3G.md`.

## Anchors and notation

- Bed: train-clean-100 (28 539 utts, partition 4), G=12, T=0.7, frozen wav2vec2-lv60
  encoder, pinned vanilla-w2v2 k-means units (500).
- Derangement gap: nats/frame saved by the true audio-text pairing over a length-matched
  shuffled one; ~0 = no coupling. Read on the frozen 1 500-pair set
  (`FrozenHeldPairsJob.E8UaEwRF65HW` — 725 dev-clean + 775 dev-other, never trained on).
- Allegiance: gold nll/frame minus own-decode nll/frame under the same scorer; positive =
  scorer prefers the policy's text to gold.
- Scale anchors: unit-unigram floor 6.095 nats; a psi fitted on real pairings shows held
  gap 4.86 and matched level ~2.6 nats/frame.
- Standing rulings (2026-08-14): encoder stays FROZEN through the grounding phase (reward
  reads only pinned units; frozen features are the guarantee phone content survives — §0a
  linear probe PER 0.127–0.145; capacity-over-frozen-features is the identifiability device;
  encoder FT belongs in a later distill/self-training stage). In-loop psi trains on the WHOLE
  rollout group — selection by psi's own confidence is inadmissible (the D4' admissibility
  read measured psi's score filler-positive at matched WER; self-selection is the allegiance
  ratchet accelerated); legal cheap variants if compute binds: policy-greedy one-per-utt,
  uniform-sample-one, duplicate-downweighting.

## Gate — outcome taxonomy (registered 2026-08-12, carried verbatim; never edited)

(A) AUDIO-FREE non-convergence: text_explained ~ 0 and derangement contrast ~ 0 throughout
while LM fluency stays high — the loop never couples to the audio. (B) PRIVATE-CODE
convergence: text_explained and derangement contrast grow (real audio-text mutual
information) while PER/WER stays at chance and a unit-emission purity read fails — the pair
agrees on a self-consistent non-human mapping. (C) GENUINE partial convergence: PER
materially below the audio-free floor with (B)'s purity read passing. The deliverable is the
classification plus the trajectories; WER/PER confirm, never select.
Gate-design gap (noted 2026-08-14, gate text unedited): the taxonomy does not by itself
distinguish (A)-with-a-length-channel from early-(B); the duration-matched derangement read
is the discriminator.

## 3g.1 — Z1 base arm

**Purpose.** The pure loop's failure mode: nothing added, full D5 forensics from step 0.
**Approach.** Cold audio pathway (adapter + LoRA r=128 untrained), min-duration psi from
random init co-trained on rollouts (plain per-frame NLL), LM prior lam_lm 1.0 units-norm.
**Experiments.** 6 sub-epochs; table and catalog in `SAE_3G.md` approach 1.
**Gate.** The taxonomy above.
**Status: CLOSED 2026-08-13 — outcome (A) CONFIRMED.** Policy mode collapse to one constant
fluent sentence for every utterance by step 346 (under 5 % of one sub-epoch); derangement
gap never leaves zero; psi becomes an unconditional segmental unit model (5.12 nats, below
the 6.095 floor via duration structure alone) whose allegiance prices every deviation
negative; with all 12 rollouts identical GRPO's gradient is ~0 — the point is absorbing.
Mechanism finding: every reward term was per-utterance, so the joint objective's global
optimum sits at zero coupling. Job tree deleted on the user's instruction (log table is the
record).

## 3g.2 — Z2 (diversity price + pseudo-pair init + derangement hinge)

**Purpose.** Remove the zero-coupling optimum: can the loop couple at all when collapse is
priced (cross-utterance bigram-overlap price lam_div 0.3), audio-dependence exists at step 0
(1e.1 length-paired SFT re-run on the §0d donor, encoder frozen, read at SFT ep 5 by the
label-free plateau rule), and the scorer must beat a length-matched shuffled pairing by a
margin (hinge 0.5 nats/frame, weight 1.0)?
**Approach/Experiments.** `SAE_3G.md` approach 2 (constants derived not swept; per-step
anti-collapse instrumentation). Gate addition (registered 2026-08-13 before launch): minimum
non-(A) reading = derangement gap leaves zero and grows two consecutive sub-epochs on both
gold and own text; alarms per ~50 steps.
**Status: STOPPED BY USER 2026-08-14 mid sub-epoch 5 ("the result is telling already").**
Verdict, from sub-eps 1–4 (all rows verified against job outputs): the loop escaped the
zero-coupling fixed point by climbing a NUISANCE-CHANNEL LADDER — duration first (the code
is repeat-count-of-a-stem-phrase; corr(audio frames, decode tokens) 0.856), speech density
second (residual-residual corr 0.252: at fixed length, wordier gold gets more repeats), no
phone content; held own-text gap +0.085 = ~2 % of a real scorer's; gold gap ~0/negative
(psi cannot fit English at all — 0.7 nats/frame worse than the code); lexicon churns
(14/2703 stable across sub-epochs); the diversity price is outbid (0.086 paid vs 0.062 LM
earned — the overlap is funded by psi recon gains, the number any lam_div raise must beat).
CLOSE-OUT BATTERY on the last completed checkpoint, pinning Z3's baseline (registered
2026-08-14, pre-results): (i) duration-matched derangement read — partner matched on audio
n_units AND text length; prediction: small positive survival, well below 0.085; (ii) the
same within-speaker (speaker IDs evaluate only) — the speaker meter; (iii) residual-density
probe vs gold phone count at matched duration (evaluate only); (iv) code-persistence count.
Artifacts KEPT — the run is the registered diagnosis and the reads need its checkpoints.

## 3g.3 — Z3 (perturbation-consistency package; USER-directed 2026-08-14)

**Purpose.** Force the coupling ladder past the nuisance channels (duration, then
speaker/channel — the next-cheapest shortcuts) toward tempo- and voice-invariant content:
phones. Each nuisance channel gets a perturbation whose consistency requirement makes it
worthless, while the hinge keeps demanding cross-utterance discriminability — the only
channel satisfying both pressures is content.
**Approach.** Z2's loop and init (reuse the ep-5 pseudo-pair SFT — right lineage) plus:
(1) PERTURBATION-CONSISTENCY reward (the user's named core): the policy must emit the same
text for perturbed copies of the utterance — tempo 0.9x/1.1x (speed preserves phone identity
but changes the frame count, so duration coding is punished by construction) and additive
noise/reverb (kills recording-channel coding); pitch/VTLP warp (kills speaker-timbre coding,
label-free) included as the third perturbation on the user's speaker concern — one-time
perturbed-unit dumps through the frozen encoder; the agreement term varies within a rollout
group, so GRPO sees it. (2) HINGE NEGATIVES upgraded to duration-matched AND
acoustically-nearest: the shuffled partner is chosen by unit-histogram similarity (label-free
pseudo-speaker/channel matching), so the margin cannot be earned through duration or global
voice/channel similarity — only content separates nearest neighbors. (3) lam_div RAISED —
value derived before launch from Z2's measured recon-funding of overlap (must exceed the
recon gain per unit overlap, not the LM gain; planner constant, user may override).
(4) OPTIONAL slow-teacher persistence (EMA or last-sub-epoch greedy decode) at low weight,
entering only after the content signal is confirmed — persistence applied early freezes the
duration code. The diversity price stays throughout (plain consistency alone rewards
collapse — a constant sentence is perfectly consistent). No speaker labels train anything,
ever; speaker IDs remain the evaluation meter only.
**Experiments.** Same bed, schedule, per-step alarms, and forensics battery as Z2, plus the
duration-matched gap and the speaker meter as STANDING per-sub-epoch columns.
**Gate (pre-registered 2026-08-14, before launch).** Primary: the held duration-matched
derangement gap exceeds Z2's close-out baseline and grows over two consecutive sub-epochs.
Secondary: code persistence strictly above Z2's 14/2703 at matched read points; and the
within-band speaker enrichment (eval-only) does NOT grow while the primary grows — coupling
gains must not be speaker-carried. Final letter at close by the standing taxonomy plus the
purity read.
**Status: REGISTERED AND FUNDED 2026-08-14** (user: stop Z2, go to Z3). Build order:
close-out battery on Z2's checkpoint first (it is Z3's baseline), then the perturbed-unit
dumps, then launch. Planner constants the user may override: the lam_div value (derivation
above) and the pitch/VTLP perturbation strength.
Status 2026-08-15 (planner mid-run read; launched 2026-08-14, live in sub-epoch 4 of 6;
sub-eps 1-3 verified against job outputs): the primary clause is failing so far — held
dur-matched own gap -0.0137/-0.0160/-0.0142, flat and negative, vs the required "exceeds
+0.0272 and grows two consecutive sub-epochs". Collapse is prevented (in-loop distinct frac
0.90, within-group reward std rising 0.047->0.136, cross-utt overlap held at 0.043 vs Z2's
0.335) but the coupling channel is the SAME duration code rebuilt more purely: after a
transient sub-ep-1 break (canned-sentence mode, corr(n_units, tokens) 0.255), sub-eps 2-3 are
98.7% exact stem-times-k with 2-token stems, corr 0.811, no content-word signal above chance,
and the stem vocabulary turns over every sub-epoch (persistence 0/2703). Two prices are paid
rather than binding: lam_div is cross-utterance only (blind to within-sequence repetition,
the code's carrier — 0.807 bigram repetition), and only the tempo half of the perturbation
mixture punishes duration coding. Restructure proposals surfaced to the user (within-seq
repetition price, activating the dormant lam_len speaking-rate hinge, discrete psi refresh
instead of continuous co-training, optional corpus-level n-gram distribution match ADDED
beside the LM prior); the LM-prior demotion proposal is WITHDRAWN (replaces the 2026-08-15
proposal line, 2026-08-15, because the user pushed back and the mechanics review confirmed
it wrong: under units-norm the prior is a charge with a sign guarantee, recon + prior at
lam_lm 1.0 is exactly the group-standardized posterior over transcripts, and the Z2/Z3
repeat code is recon-funded, not LM-funded). No Z4 registered pending the user's word.
Full forensics in `SAE_3G.md` verifier feedback 2026-08-15.

**Speaker-information fallback ladder (USER amendment 2026-08-14 to the 2026-07-16 label
ruling; transcripts/alignments stay absolute — they ARE the unpaired claim).** Escalate one
tier at a time; each tier is disclosed as a supervision cost (usability framing):
- Tier 0 (default, fully signal-derived — the Z3 package above): voice-warp consistency,
  acoustically-nearest negatives, per-utterance feature normalization. No metadata anywhere.
- Tier 1 (recording provenance): hinge negatives drawn from the SAME chapter/session —
  grouping by how the audio arrives; discloses metadata use but no explicit identity model.
- Tier 2 (explicit speaker IDs in training): within-speaker duration-matched hinge negatives
  (the supervised version of tier 0's nearest-neighbor trick, strictly cleaner); a
  speaker-adversarial head (gradient reversal) on the adapter/code; per-speaker feature
  normalization for a re-derived unit inventory. Precedent: the ZeroSpeech benchmarks provide
  training-audio speaker IDs and permit exactly this use; crosses the wav2vec-U "corpus prep
  yes, annotations no" line but not the paired-data line — headline becomes "no
  transcriptions; speaker metadata used", still unpaired.
- TRIGGER: replaced 2026-08-16 (was: tiers 1-2 unlock only on Z3's speaker clause failing —
  primary coupling growing while the eval-only within-band speaker enrichment grows with it —
  despite tier 0 running) because (a) the USER confirmed speaker-label use is OK as a first
  try, and (b) the Z4 diagnosis showed the registered condition could never fire in the actual
  failure shape: the primary coupling read collapsed to ~0 rather than growing
  speaker-carried, yet the speaker channel was real (the init-fed scorer's whole +0.0303).
  Now: tiers 1-2 may be registered first-line in any successor arm; each tier is still
  disclosed as a supervision cost, and transcripts/alignments stay absolute.

## 3g.4 — Z4 (discrete psi refresh + carrier-channel closure; USER-funded 2026-08-15)

**Purpose.** Z3 beat the collapse machinery (distinct rollouts 0.90, reward spread rising) but
rebuilt the duration code more purely than Z2 under its live co-trained scorer, and the
diagnosis is structural: the code is recon-funded — the co-trained psi converges to whatever
text family it can fit and entrenches it (Z3's stem vocabulary turned over 100% per sub-epoch
precisely because psi tracked it live) — and its carrier, within-sequence repetition on a free
length channel, is invisible to every current price. Z4 removes the funder's agility and
closes the carrier channels, keeping the posterior-form reward intact (lam_lm 1.0 units-norm
stays — ruled 2026-08-15: recon + prior at 1.0 is the group-standardized posterior over
transcripts, and the prior is a charge with a sign guarantee, not the funder).

**Approach.** Z3's bed, init (pseudo-pair SFT ep 5), perturbed-unit dumps, nearest-negative
store, lam_lm / lam_div / lam_cons and the perturbation mixture all unchanged (stores are
reused, not recomputed), plus three changes run as one package:
(1) DISCRETE PSI REFRESH replaces continuous co-training (train_psi=False in the loop). The
reward psi is FROZEN for each whole sub-epoch — a stationary critic the policy cannot
negotiate with — and at every boundary a fresh psi is refit FROM SCRATCH (random init, same
min-duration psi_align idiom) offline on the policy's greedy one-decode-per-utterance over the
full 28 539-utt bed (the registered legal variant), with Z3's hinge (margin 0.5, weight 1.0,
duration-banded acoustically-nearest negatives) in the offline fit; the refit serves the next
sub-epoch. Sub-epoch 1 is served by a psi fit on the init checkpoint's decodes. Code churn now
earns nothing: a from-scratch refit retains only structure that is stable in the policy's
output.
(2) WITHIN-SEQ REPETITION PRICE, new reward term lam_rep: per rollout, 1 - distinct bigrams /
total bigrams (the statistic measured at 0.807 on Z3 ep 3). Taxes the repeat carrier that
lam_div cannot see (cross-utterance only) and the AR prior undercharges (repeat tokens are its
cheapest tokens).
(3) LAM_LEN ACTIVATED (the dormant speaking-rate hinge): prices |log(chars/(rate x duration))|
outside a dead band, so decode length tracks audio duration by construction — length then
carries no discriminative information and the derangement margin can only be earned by
content. Deferred, not in the package: the corpus-level n-gram distribution match (optional
extension registered 2026-08-15; lam_div covers its main role for now).
Constants (derived, not swept; planner values, user may override): lam_rep sized so one added
stem-repeat costs more than it earns — it must exceed the recon gain per repeat measured on
Z3's sub-ep-3 checkpoint (implementer measures pre-launch; the price is a fraction in [0, 1],
so lam_rep is the term's full swing, same logic as lam_div). lam_len 0.5 with rate nu_c = 14
chars/sec (read-English literature value, label-free) and dead band eps = 0.35 in log units
(prices only mismatches beyond about +/-40%, never fine rate). Refresh refit budget follows
the offline psi plateau practice.

**Experiments.** Same schedule (6 sub-epochs), per-step alarms, forensics battery, and
standing per-sub-epoch reads as Z3, plus two new standing columns: within-seq bigram
repetition (must fall below Z2's 0.68) and the length-hinge violation rate. Z3 runs untouched
to its registered 6-sub-epoch end (~hours; it self-terminates long before Z4 builds) and its
close-out is Z4's like-for-like comparison.

**Gate (pre-registered 2026-08-15, before launch).** Primary: held duration-matched own-text
gap (dur_text, ce_loo) exceeds Z2's close-out baseline +0.0272 and grows two consecutive
sub-epochs. Secondary: (a) code persistence strictly above Z2's 14/2703; (b) the speaker meter
read as dur_text_spk MINUS its like-for-like dur_text control on the same rows (the raw column
is contaminated by the nearest-negative training — verifier 2026-08-15) does not grow while
the primary grows; (c) within-seq repetition falls below Z2's 0.68 — the price must bind, not
be paid. Registered failure reading: if the frozen-critic loop instead dies to zero
within-group spread (no earnable variance once the cheap channels are closed), that is
audio-free non-convergence by exhaustion at this init and points to §1f (statistics-matching
init) — a not-funding verdict, never "it would not have worked".

**Status: REGISTERED AND FUNDED 2026-08-15** (user: "time to write plan for Z4"). Build
order: (i) lam_rep derivation measurement on Z3's sub-ep-3 checkpoint; (ii) the refresh
machinery — offline psi refit job chain plus boundary swap, reusing the PsiAlignInitJob idiom
and the existing perturbed/negative stores; (iii) launch. Planner constants the user may
override: lam_rep (derivation above), lam_len / nu_c / eps, and the refresh decode rule
(greedy one-per-utt).
2026-08-16 planner read (rounds 1-3 verified, rounds 4-6 in flight): the package's own targets
are met — within-seq repetition 0 in dev decodes, no collapse, length violations falling, the
duration/repeat code has NOT rebuilt — but the primary read collapsed instead of growing:
+0.0303 at round 1, then +0.0018 / +0.0036. (Corrected same day, replaces the init-map
attribution: the r1 read is psi_1 — the only scorer fit on the init's decode pool — reading
the post-round-1 policy, so the above-bar coupling survived round 1 under an init-fed reader;
the r2 collapse changes scorer fit corpus and policy at once and the two losses are
confounded. Log has the trace.) The
route is a closed library of canned fluent donor-style sentences (round 1: 298 distinct
strings over 2703 dev-clean utts, top string 742x) at the length band's ~29 tokens; the
higher dev WER vs Z3 is decode-length arithmetic (1.44x gold vs Z3's 0.84x — Z4 has MORE
words correct at matched rounds). If this holds to the registered end it is the exhaustion
route in spirit: with the cheap channels closed, the remaining within-group variance (falling,
0.074 -> 0.058) is funded by fluency/canned-sentence identity, which outpays the
<=0.03-nats/frame coupling differential, and the from-scratch refresh is a one-way ratchet (a
refit can only price structure still present in the decodes), so coupling lost in round 1 is
unrecoverable later. Candidate follow-up to register at gate close (not now): protect the
init's coupling inside the reward, or fall to §1f. Arm runs untouched to its 6-round end.
2026-08-16 (later), USER RULING on the follow-up space: refit corpora stay current-policy-only
— no pooling of previous rounds' decodes, no init-pool anchor (the planner's pooled-refit Z5
sketch is WITHDRAWN, not registered; ruling recorded in assistant memory).
2026-08-16 (later, replaces the diagnosis-in-flight lines that stood here): diagnosis CLOSED
from the standing reads, no new jobs needed. The r1 +0.0303 is a SPEAKER/voice-channel code:
on rows with a same-speaker partner available, a different-speaker duration-matched audio swap
costs +0.0475 nats/frame vs +0.0015 for a same-speaker swap — psi_1 (the init-fed scorer)
priced speaker-conditioned sentence choice inherited from the init, not content. The user's
fine-length hypothesis is refuted as the driver (no sub-5%-band length information exists in
any text set, gold included; gold reads dur_text ~0 under every scorer), but its spirit is
vindicated: the init-fed reader's advantage was a nuisance channel all along. Consequences:
(a) the withdrawn pooling idea also loses its empirical basis — the init pool never taught
content; (b) no Z4 round shows any coupling once duration, length, density, and speaker are
matched (dur_text_spk +0.0015 / -0.0032 / +0.0031), and the refreshed scorers' growing
gold-side lenmatched gap (+0.013 -> +0.028) is density pricing, another nuisance; (c) on the
reward side there is nothing legitimate left to add on the length axis (lam_len already covers
the coarse channel; finer does not exist even in gold) and the speaker channel must not be
rewarded — the missing ingredient is a content channel, which no corpus the policy has
generated contains; weight shifts to the registered exhaustion reading and §1f at gate close.
The 2x2 cross-read is no longer needed.
2026-08-16 addendum (frame check on the refit design): the registered element "Z3's hinge moved
into the offline fit" is not operative in the served scorers — the hinge terms only switch on
once the alignment-prior anneal ends (epoch 5), but the pre-registered held-nll pin lands at
epoch 2-4 in every round because held nll cliffs to the unigram floor the moment the guide
comes off (the canned-library decodes carry almost nothing conditional to learn), so every
scorer the loop has used is a guided-warmup checkpoint with zero hinge updates. Not a code bug
(the wait-for-anneal is documented and the pin is pre-registered); an emergent interaction to
weigh at gate close and in any refresh-based follow-up. Raised with the implementer/user
2026-08-16; the logged coupling reads are unaffected (they read the served scorer as-is).
2026-08-16 GATE VERDICT (all six rounds; read as registered, planner-verified against the
sclite outputs, the training jobs' learning_rates, and the decode dumps — persistence and
distinct-string counts independently recomputed, they match): **FAILS**. Primary: the
duration-matched own-text gap clears +0.0272 only at round 1 (+0.0303, then +0.0018 /
+0.0036 / +0.0083 / +0.0084 / +0.0126) — it grows over the last three rounds but never
regains the level, so the two clauses are never satisfied together. Secondary (a) met only
at rounds 5-6 (25 and 26 of 2703 vs Z2's 14); (b) FAILS as written (the speaker meter grows
over exactly the rounds the primary grows: +0.0039 / +0.0140 / +0.0136); (c) met decisively
(within-seq repetition 0.0023-0.0068 vs the 0.68 bar — the price binds). The registered
exhaustion failure reading does NOT fire: within-group reward spread falls only to round 3
and recovers past its start (0.0737 / 0.0601 / 0.0579 / 0.0741 / 0.0698 / 0.0785, verified
in learning_rates). This REPLACES the exhaustion framing in the two in-flight reads above
(the "remaining within-group variance (falling, 0.074 -> 0.058)" clause and the "weight
shifts to the registered exhaustion reading" clause, both 2026-08-16) — both described
rounds 1-3 only and are WRONG for the full run: this is a gate failure with earnable
variance still present, not exhaustion. Two scope notes, neither reaching the gate: the
round-1 canned-library description dissolves by round 6 (298 -> 1504 distinct dev-clean
strings, top string 742x -> 34x, dev WER 143.8 -> 105.6 by insertion removal 27584 -> 5460
counting from round 1); and the own-text within-speaker column is only readable as own
MINUS same-round gold (gold reads +0.0165 under the identical rule at round 6 — a meter
floor no policy-side code explains; the residual +0.0093 / -0.0012 / +0.0093 over rounds
4-6 is non-monotone and within meter noise, so no coupling claim — reading rule ratified).
Instrument defect recorded, nothing invalidated: the reward-rank probe read 0.00000 /
nan uniformly in all six rounds because its dump holds ONE sampled rollout per utterance
(each utterance's three rows are one T=0.7 sample + gold + greedy; the probe rightly keeps
only samples, leaving group size 1) — the striking part is that a broken instrument
emitted exactly the registered exhaustion signature while the in-loop spread was 0.058-
0.079; the gate never references the probe, and its fix (a multi-sample dump, G>1) is
funded only if a refresh-family follow-up is ever funded. CONSEQUENCE: Z4 closes at its
registered end; the verdict licenses not funding a continuation, never "a refresh loop
could never work". The follow-up space is the USER's decision: any Z5 must respect the
current-policy-only refit ruling and address the emergent hinge-never-trains interaction
(addendum above); the alternative is standing on §1f (independently funded, running) as
the sole initialization effort. Nothing new launched.
