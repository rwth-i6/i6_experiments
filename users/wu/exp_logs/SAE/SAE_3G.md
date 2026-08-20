# SAE §3g — Z-track: from-scratch fully-unsupervised joint loop

## Approach

**1. Z-track base arm (launched 2026-08-12).** The joint loop with no paired data on either side, run
as a diagnosis arm against PLAN §3g's (A)/(B)/(C) taxonomy. Policy = the §0d LBS-adapted text-only
Qwen donor with the audio pathway (concat-downsample adapter + decoder LoRA r=128) at init, over the
frozen pretrained wav2vec2-lv60 encoder; scorer = a D6 rung-3 min-duration psi_align at random init,
co-trained in-loop on the policy's own rollouts (`train_psi=True`, plain per-frame unit NLL, ce scale
1.0); reward = that scorer plus the base-LM prior at the bed's registered `lam_lm` 1.0 under
`lm_prior_norm="units"`. Bed = train-clean-100 (28 539 utts, `partition_epoch` 4 = 7135 utts per
sub-epoch), G=12, T=0.7, max_lr 2e-5, 6 sub-epochs, batch 1e6 with accum 2; measured 2h15m per
sub-epoch at 55.8-57.7 GB of 95. Every sub-epoch is read on the frozen 1500-pair gold dev set, with
sub-epoch 0 the untrained pair (random scorer, cold policy, same frames and same beam) as the arm's own
audio-free floor.

Anchors for the table: unit-marginal floor of the bed's own unit unigram = 6.095 nats (500 units,
perplexity 444); a derangement gap is nats saved by the true pairing over a length-matched deranged
one, so gap ~ 0 = no audio-text coupling; allegiance = gold nll/frame minus own-decode nll/frame under
the same scorer, positive = the scorer prefers the policy's text to the gold text.

| sub-ep | dev-clean | dev-other | psi_ce end | within-group reward std | gen. tokens | dev-clean decode (distinct / mean tok) | derangement gap | allegiance |
|---|---|---|---|---|---|---|---|---|
| 0 (untrained pair) | 316.55 | 336.05 | 6.59 init | 0.28 | cap-length | 1852 / 59.3 | 0.0005 | -0.140 |
| 1 | 108.88 | 111.81 | 5.492 | 0.0041 | 15.9 | 1 / 15.0 | -0.0106 | 0.255 |
| 2 | 108.87 | 111.79 | 5.420 | 0.0049 | 16.0 | 2 / 15.0 | -0.0087 | 0.350 |
| 3 | 130.54 | 139.66 | 5.236 | 0.0059 | 24.0 | 4 / 22.9 | 0.0007 | 0.161 |
| 4 | 130.78 | 140.11 | 5.191 | 0.0056 | 24.0 | 3 / 23.0 | 0.0019 | 0.264 |
| 5 | 132.91 | 139.19 | 5.119 | 0.0068 | 33.9 | 9 / 24.8 | 0.0045 | 0.244 |
| 6 | 103.74 | 102.86 | 5.110 | 0.0058 | 32.3 | 10 / 18.4 | 0.0054 | 0.593 |

**2. Z2 arm (launched 2026-08-13).** Approach 1's bed, loop, schedule, scorer init and five
forensics reads verbatim, plus three additions run as one package because approach 1's finding is
that the joint objective's optimum sits at zero coupling rather than that the loop is too weak:
the policy starts from §1e.1's length-paired pseudo-pair SFT re-run on the §0d donor with the
encoder frozen, the reward gains a cross-utterance bigram-overlap price at lam_div 0.3, and psi's
in-loop loss gains a hinge requiring the true pairing to beat a length-matched other utterance's by
0.5 nats/frame at weight 1.0. The anti-collapse instrumentation reads every step rather than every
sub-epoch -- within-group reward spread, the distinct-hypothesis fraction of the step's rollouts,
and the realized in-loop derangement gap.

Constants registered before launch, derived rather than swept. lam_div 0.3: the overlap is a
fraction, so the weight IS the term's whole swing, and it is set below the 0.39 nats of language-model
prior that separates approach 1's cold rollouts from its collapsed ones (-0.43 to -0.036) so that
unreadable-but-varied text can never outbid fluent-but-varied text, while still exceeding recon's
0.05 within-group spread that the term exists to restore. Derangement margin 0.5 nats/frame: two
orders of magnitude above approach 1's collapsed +0.0045 and roughly a seventh of a scorer fitted on
real pairings (+3.69 and +5.30 on the offline twin of this read, `SAE_3E1.md` approaches 4 and 9), so
it is a floor the loop must clear rather than a target it must hit. The §1e.1 SFT is re-run rather
than reused because the shipped one sits on the stock donor; frozen-encoder plus the §0d donor makes
its `av_model_args` identical to the loop's, verified equal key-by-key before launch.

The init is read at SFT epoch 5, not the shipped schedule's epoch 50 (replaces epoch 50, 2026-08-13):
2849 utterances over 50 passes with no audio-text information to fit makes the assignment memorizable,
and epoch 50's pseudo-pair training frame error of 0.135 is that memorization. No transcript may pick
the checkpoint, so the rule reads the pseudo-pair training loss alone -- the last epoch before the
curve's slow-grind phase, the first improving by under 1% over its predecessor (ep3->4 1.27%,
ep4->5 0.69%). As an after-the-fact diagnostic and not the selector, real-pair dev cross-entropy
bottoms at ep6 (5.079) and climbs to 6.816 by ep50.

Two derangement reads per sub-epoch: the gold gap is approach 1's read on the frozen gold pairs, the
own-text gap the same read with the policy's own decode substituted for the transcript.

| sub-ep | dev-clean | dev-other | decode distinct / mean tok | in-loop distinct frac | within-group reward std | cross-utt overlap | gold gap | own-text gap | allegiance |
|---|---|---|---|---|---|---|---|---|---|
| 0 (init, SFT ep 5) | 290.70 | 315.52 | 1484 / 59.2 | - | - | - | - | - | - |
| 1 | 126.96 | 130.47 | 259 / 23.0 | 0.919 | 0.048 | 0.122 | +0.0005 | +0.0021 | -0.089 |
| 2 | 113.31 | 116.38 | 294 / 19.2 | 0.472 | 0.036 | 0.225 | -0.0028 | +0.0530 | +0.184 |
| 3 | 114.84 | 116.51 | 316 / 20.7 | 0.301 | 0.022 | 0.288 | -0.0071 | +0.0688 | +0.485 |
| 4 | 108.53 | 107.33 | 398 / 19.5 | 0.281 | 0.021 | 0.335 | -0.0081 | +0.0852 | +0.696 |
| 5 | 104.70 | 103.85 | 284 / 18.2 | 0.306 | 0.023 | 0.331 | -0.0045 | +0.1021 | +0.734 |
| 6 | 101.94 | 101.55 | 361 / 16.8 | 0.303 | 0.021 | 0.362 | -0.0053 | +0.1167 | +0.796 |

**3. Controlled derangement nulls (Z2's close-out battery, run 2026-08-14; standing per-sub-epoch
reads for approach 4).** The partner rule behind the gap is tightened in two steps from the logged
length-matched rotation: `dur_text` draws it from utterances within 5% of this one's frame count and
then takes the nearest in TEXT length, so a code that maps a frame count to a text length earns
exactly zero, and `dur_text_spk` additionally requires the same speaker (speaker ids describe the
partner set here and select nothing). Each rule is scored on the rows where IT is defined and alignable, with
its n reported: a same-speaker partner exists for only 965 of the 1500 pairs, and that subset is the
utterances whose speaker happens to have similar-length neighbours, so intersecting the rules would
measure the primary read on a biased and needlessly smaller sample; the speaker meter is therefore
additionally reported against `dur_text` on its own rows. Two decode-side reads accompany them: the
share of utterances whose decoded string survives a sub-epoch, and the correlation of decode length
with the gold phone count once the frame count is partialled out.

| read on Z2 sub-ep 4 | n | gold text | own text |
|---|---|---|---|
| gap, length-matched (the logged rule) | 1493 | -0.0033 | +0.0882 |
| gap, duration + text-length matched | 1498 | -0.0122 | **+0.0272** |
| gap, the same within speaker | 965 | -0.0035 | +0.0392 |
| the row above's like-for-like control (`dur_text` on those 965) | 965 | -0.0156 | +0.0216 |
| code persistence, sub-ep 3 -> 4 | 5567 | 27 (0.49%) | - |
| corr(decode words, gold phones) given frames | 1500 | +0.2326 (se 0.026) | - |

The partner's text length is 0.4% (own) / 1.1% (gold) off under `dur_text` against 13.9% / 23.8%
under the logged rotation, which is the sense in which the logged rule never controlled length at all.
These gaps are differences of `ce_loo`, the leave-one-out predictive the held-set gate clause is
written on, where approaches 1 and 2 report differences of `nll`; the two differ by a few thousandths
and the `lenmatched` row is the bridge (+0.0882 here against +0.0852 there on the same checkpoint).

**4. Z3 arm (launched 2026-08-14).** Approach 2's bed, policy init (the same pseudo-pair SFT
checkpoint at epoch 5), random scorer init, schedule and forensics verbatim, plus the
perturbation-consistency package: every utterance carries the units of ONE perturbed copy of itself --
tempo 0.9x/1.1x, a vocal-tract warp or added noise, the kind assigned from its seq tag and encoded by
the same frozen encoder and pinned codebook -- and the reward charges each rollout `lam_cons` times
the per-frame gap between how well its text explains the recording and how well it explains that copy.
The in-loop hinge keeps approach 2's margin and weight but its negative is now chosen once over the
whole bed as the utterance with the nearest unit histogram inside a 5% frame-count band, so the
partner shares the clip length, and in 16.0% of cases the speaker as well against 0.32% at chance.
`lam_div` rises from 0.3 to 1.5.

Constants registered before launch, derived rather than swept. lam_cons 0.5 is the symmetric point:
the reward's reconstruction becomes half the recording's and half the perturbed copy's, so neither
reading of the utterance is privileged; above it the perturbed stream would outweigh the one psi is
trained on, below it the term is a tie-breaker, and approach 2 showed a tie-breaker loses to a channel
that pays. lam_div 1.5 comes from approach 2's own trajectory: it bought cross-utterance overlap at
3.02, 1.47 and 1.21 nats of recon per unit of overlap over its three transitions, so 1.21 is the rate
at the operating point where it stopped and 1.5 clears it -- while staying under the ~2.1 at which the
term's whole swing would exceed the 0.7 nats/frame that abandoning the code costs a rollout (psi at
sub-ep 4 fits the code at 5.19 and gold English at 5.96), and raising the price's within-group spread
to 0.067 against recon's 0.016. The mixture is 50% duration, 30% vocal tract, 20% channel, the
ladder's own order. Measured against the unperturbed stream on all 34 106 utterances, the perturbed
units come out at 1.1110 and 0.9087 times the frame count for the two tempo kinds and at exactly
1.0000 for the other three, with 46% of unit positions unchanged under noise and 82% under the warp.

The two rightmost columns are approach 3's `dur_text` and `dur_text_spk` on the arm's own decodes --
the pre-registered primary read and speaker meter -- and are `ce_loo` differences where the two gap
columns beside them are `nll` differences, as approach 3 notes. Sub-epoch 0 is approach 2's init
decode read again, identical by construction: the two arms share the checkpoint, the decoder and the
beam.

| sub-ep | dev-clean | dev-other | decode distinct / mean tok | in-loop distinct frac | within-group reward std | cross-utt overlap | cons penalty | gold gap | own-text gap | dur-matched | within-speaker |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (init, SFT ep 5) | 290.70 | 315.52 | 1484 / 59.2 | - | - | - | - | - | - | - | - |
| 1 | 108.79 | 107.31 | 388 / 16.0 | 0.999 | 0.047 | 0.019 | +0.011 | -0.0016 | -0.0030 | -0.0137 | +0.0012 |
| 2 | 106.19 | 106.40 | 153 / 16.9 | 0.983 | 0.085 | 0.025 | +0.035 | +0.0017 | -0.0021 | -0.0160 | -0.0005 |
| 3 | 97.18 | 98.19 | 336 / 15.6 | 0.904 | 0.136 | 0.043 | +0.052 | +0.0056 | +0.0093 | -0.0142 | +0.0094 |
| 4 | 96.24 | 97.45 | 427 / 15.9 | 0.787 | 0.142 | 0.066 | +0.070 | +0.0049 | +0.0178 | -0.0122 | +0.0090 |
| 5 | 95.49 | 96.61 | 605 / 14.7 | 0.686 | 0.129 | 0.075 | +0.063 | +0.0038 | +0.0250 | -0.0125 | +0.0093 |
| 6 | 94.87 | 96.14 | 785 / 13.8 | 0.623 | 0.119 | 0.083 | +0.039 | +0.0040 | +0.0265 | -0.0093 | +0.0113 |

**5. Z4 arm (launched 2026-08-15).** Approach 4's bed, policy init, unit stores, `lam_lm` / `lam_div`
/ `lam_cons` and perturbation mixture verbatim, with the scorer's update rule and the two carrier
channels changed as one package: the scorer is FROZEN for a whole sub-epoch and refit from random
init at every boundary on the policy's own greedy one-decode-per-utterance over the whole 28 539-utt
bed (no anchor -- the track has no gold text to anchor with -- and with approach 4's hinge, margin 0.5
weight 1.0 against the same duration-banded nearest partner, moved into that offline fit), a new
`lam_rep` charges each rollout for the share of its own bigram slots that repeat an earlier one, and
the dormant `lam_len` speaking-rate hinge is switched on. The six sub-epochs are six jobs rather than
one, so each round's corpus is a graph object (a quarter of the bed, 7135/7135/7135/7134 utterances)
rather than a RETURNN epoch counter that would reset to the same quarter every round.

Constants registered before launch, derived rather than swept. lam_rep 0.7 comes from approach 4's own
trajectory read at the transition where the code was ACQUIRED: bigram repetition in its dev decodes
goes 0.117 -> 0.775 -> 0.792 -> 0.774 while recon goes -5.861 -> -5.490 -> -5.372 -> -5.274, so the
loop bought repetition at 0.564 nats per unit across sub-epoch 1->2 and then held it flat, and the
later ratios (+7.1, -5.6) are a near-zero denominator rather than a rate. The price is a fraction in
[0, 1], so lam_rep is its full swing, 0.7 clears 0.564 by the margin lam_div's 1.5 clears its 1.21,
and its swing at the observed level (0.7 x 0.775 = 0.54 nats) is the size of the entire recon
improvement approach 4 made over four sub-epochs. lam_len 0.5 at the reward module's measured rate
14.55 chars/s and dead band 0.4 -- tc100's own corpus-level rate, and the band at which the hinge
charges ~1.6% of gold transcripts rather than the ~18% the older 15.0/0.2 pair did.

Banked sub-epoch means. `x_std` is the within-group spread of term x, the only channel a price can
steer through; the yardstick beside it is `recon_std`, and a price's reach is its weight times its own
spread. `distinct` is the share of rollouts in a step that are distinct strings (1.0 = no collapse).
Approach 4's last banked sub-epoch is carried as context, not as a control -- different arm, different
point in training. `dur-matched` is approach 4's own primary read, the derangement gap under a partner
matched on both frame count and text length, taken on the round's own refit; it is scored on the rows
where the rule is alignable, and that row set grows with the round (1243 / 1298 / 1498), so the column
is not a fixed-row comparison across rounds and the read carries no interval. Two registered clauses
are not columns: dev-clean string persistence between consecutive rounds runs 1 / 4 / 5 / 25 / 26 of
2703, and the same decodes hold 298 / 824 / 1085 / 1263 / 1561 / 1504 distinct strings with the most
frequent one used 742 / 303 / 51 / 112 / 43 / 34 times. The same battery is run on GOLD text at every
round, which is the floor each own-text column has to be read against: gold `dur_text` is flat at zero
throughout (+0.0019 / -0.0002 / -0.0039 / +0.0051 / +0.0079 / +0.0010) but gold `within-speaker` is
not (+0.0019 / +0.0109 / +0.0096 / +0.0098 / +0.0208 / +0.0165), so that rule carries a floor no
policy-side code explains.

| point | dev-clean | dev-other | rep | rep_std | recon_std | len_viol | distinct | text_len | reward_std | dur-matched | within-speaker |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Z4 round 1 | 146.44 | 162.34 | 0.0068 | 0.0134 | 0.0473 | 0.170 | 0.9998 | 29.8 | 0.0737 | +0.0303 | +0.0015 |
| Z4 round 2 | 156.26 | 166.80 | 0.0025 | 0.0066 | 0.0373 | 0.112 | 0.9999 | 30.3 | 0.0601 | +0.0018 | -0.0032 |
| Z4 round 3 | 143.82 | 151.07 | 0.0026 | 0.0066 | 0.0416 | 0.041 | 1.0000 | 35.5 | 0.0579 | +0.0036 | +0.0031 |
| Z4 round 4 | 130.25 | 133.79 | 0.0033 | 0.0077 | 0.0571 | 0.033 | 0.9997 | 38.0 | 0.0741 | +0.0083 | +0.0192 |
| Z4 round 5 | 118.61 | 121.70 | 0.0028 | 0.0069 | 0.0529 | 0.013 | 0.9993 | 36.7 | 0.0698 | +0.0084 | +0.0196 |
| Z4 round 6 | 105.57 | 107.08 | 0.0023 | 0.0058 | 0.0620 | 0.007 | 0.9983 | 35.9 | 0.0785 | +0.0126 | +0.0258 |
| approach 4, last | 94.87 | 96.14 | -- | -- | 0.0222 | -- | 0.6860 | 21.7 | 0.1291 | -0.0093 | +0.0113 |

## Conclusion

1. Outcome (A), audio-free non-convergence, over all six sub-epochs: the derangement gap never leaves
zero on either the gold text or the policy's own text, so no audio-text information enters the pair at
any point, while the language-model prior reward improves an order of magnitude (-0.43 to -0.036 by
sub-epoch 5, -0.058 at the zero-learning-rate sub-epoch 6).

1a. The mechanism is a policy mode collapse to one constant fluent English sentence emitted for every
utterance — within-group reward spread falls from 0.28 to below 0.01 by step 346 and the language-model
prior's within-group spread is exactly zero from then on, so GRPO has no gradient at all (1e-5 against
3e-3 at step 0) and the scorer's remaining progress (psi_ce 6.59 to 5.12, below the 6.095 unigram
floor) is an unconditional segmental model of the units, not shared information; the sub-epoch 1-2
string is "then you can tell me what you want to know and i will tell you", which by sub-epoch 5 has
lengthened by clause repetition to "then tell me what you want to know what you want to know what you
want to know what you want to know and i will tell you".

1b. WER tracks the collapsed string's length rather than any recognition (16 tokens = 108.9, 24 = 130.5,
34 = 132.9 dev-clean), and the arm therefore has no reachable lever on the reward side: with
within-group spread at 0.005, re-weighting any reward term scales a number that is already zero.
WRONG in its length pairing — correction: those token counts are the training rollouts, and on the
dev-clean decode itself (1-10 distinct strings over 2703 utterances at every sub-epoch, mean
15.0/15.0/22.9/23.0/24.8/18.4 tokens against WER 108.9/108.9/130.5/130.8/132.9/103.7) WER never falls
below 100% and its ordering follows decode length only loosely, so the arm recognizes nothing at any
point; the no-reachable-lever half stands.

2. Outcome (A) again over the first three sub-epochs, by a route the base arm could not show: the
derangement hinge is satisfied on the policy's own text (own-text gap +0.0021 -> +0.0530 -> +0.0688)
while the gold-pair gap stays at zero and then turns negative (+0.0005 -> -0.0028 -> -0.0071), so the
constraint added to force audio-text coupling is cleared by a route that carries no coupling to real
transcripts.
WRONG in its taxonomy letter (planner 2026-08-14) — correction: the frozen read set is dev-clean/
dev-other audio the loop never trains on, so a growing own-text gap there is real, generalizing
audio-text information and the registered (A) definition (contrast ~ 0 throughout) no longer holds;
the arm is on the (B) private-code route, final letter pending the purity read at close; the
no-coupling-to-real-transcripts half stands.

2a. The diversity price slows the collapse without changing its destination -- distinct rollouts per
group 0.919 -> 0.472 -> 0.301 and 259 -> 294 -> 316 distinct dev-clean hypotheses over 2703 utterances
against the base arm's 1, while cross-utterance overlap doubles (0.122 -> 0.288) past the point where
the price it pays (0.086 nats) exceeds the language-model prior it earns (0.062), the scorer defects
from the gold text to the policy's own within two sub-epochs (allegiance -0.089 -> +0.184 -> +0.485),
and dev-clean words correct fall 2715 -> 831 -> 648 of 54 402.

2b. The arm ran to its registered end without changing direction: over sub-epochs 5 and 6 the
own-text gap kept climbing (+0.0852 -> +0.1021 -> +0.1167) and the scorer kept defecting from gold
(allegiance +0.696 -> +0.734 -> +0.796) while dev-clean WER fell only from 108.5 to 101.9 and never
crossed 100%, so the close-out reads below describe a converged state rather than an interrupted one.

3. Two thirds of the coupling approach 2 built is the duration channel and what remains is not
speaker-carried: on its sub-epoch-4 scorer the own-text gap falls from +0.0882 under the logged
length-matched rotation to +0.0272 once the partner matches text length as well as frame count, does
not fall when the partner is additionally the same speaker (+0.0216 -> +0.0392 on the 965 rows where
both are defined), and stays absent on gold text under every rule (-0.003 / -0.012 / -0.004).

3a. The map from utterance to string is still being redrawn rather than refined -- 27 of 5567 held
utterances keep their decoded string from sub-epoch 3 to sub-epoch 4 (0.49%) -- while decode length
carries a content-adjacent signal beyond duration, correlating +0.2326 (se 0.026) with the gold phone
count at matched frame count.

4. The perturbation-consistency package ran to its registered end (all six sub-epochs) without ever
turning its pre-registered primary read positive -- the duration-matched gap stayed negative at every
sub-epoch (-0.0137, -0.0160, -0.0142, -0.0122, -0.0125, -0.0093) while the own-text gap under the
length-matched null climbed to +0.0265 -- so what this arm accumulates is again the channel that null
fails to control rather than content, and dev WER never crossed 100% (94.87 / 96.14 at the end).
WRONG in the final WER clause: 94.87/96.14 is below 100, and the arm first crossed below 100 at
sub-epoch 3. Correction: WER remained unusably near 100 and supplied no content evidence; it did not
stay above 100 throughout. The primary-gap conclusion is unchanged.

5. (5) **Z4 ran to its registered six-round end and FAILS its pre-registered primary**: the
duration-matched own-text gap clears the +0.0272 bar only at round 1 (+0.0303, +0.0018, +0.0036,
+0.0083, +0.0084, +0.0126), and although it does grow over the last three rounds it never regains the
level, so the two clauses are never satisfied together. Of the secondaries, (c) binds decisively --
within-sequence repetition is 0.0068 falling to 0.0023, two orders under the 0.68 bar -- (a) is met
only in the last two transitions (25 and 26 of 2703 against Z2's 14), and (b) fails as written, the
speaker meter growing (-0.0460, -0.0028, -0.0070, +0.0039, +0.0140, +0.0136) over exactly the rounds
the primary grows. The registered exhaustion reading does NOT fire: within-group reward spread falls
only to round 3 and then recovers past its starting value (0.0737 / 0.0601 / 0.0579 / 0.0741 / 0.0698
/ 0.0785), so this is a gate failure, not a loop that ran out of earnable variance.

6. (5) The canned-library route diagnosed at round 1 largely dissolves by round 6 and the arm's
decodes improve steadily, without either fact reaching the gate: dev-clean distinct strings go 298 ->
1504 of 2703 and the most-used string 742x -> 34x, the refit pool goes 8.50% -> 61.41% distinct, and
plain dev WER falls monotonically from round 3 (143.82 / 151.07 -> 105.57 / 107.08) as decode length
falls toward gold (dev-clean insertions 27 238 -> 5 460 over the same rounds 3 to 6). The coupling this buys does not survive its
own controls -- own-text `within-speaker` reaches +0.0258 at round 6, but gold text scores +0.0165
under the identical rule, leaving a residual (+0.0093 at round 6, +0.0094 at round 4, -0.0012 at
round 5) that is neither monotone nor larger than the round-to-round movement of the meter itself.

## Catalog

`T/` = `work/i6_core/returnn/training/`, `S/` = `work/speech_llm/sae/`.

| artifact | path |
|---|---|
| code | `sae/psi_align_jobs.py` (`PsiAlignInitJob`), `sae/grpo/extract_av_checkpoint.py` (`ColdAvCheckpointJob`) (+ their test modules, 383 tests over `sae/`) |
| entry point | `config/sae_3g_ztrack.py` |
| approach 1 job tree — DELETED 2026-08-13 on the user's instruction, once the arm closed (84 dirs, ~30 GB); the table above is the surviving record | was `T/ReturnnTrainingJob.rG68dv1zOv9O`, `S/grpo/extract_av_checkpoint/ColdAvCheckpointJob.lgJCUfBedSwe`, `S/psi_forensics/PsiForensicsTableJob.z1dyl5dRG4TX`, closing reads `S/psi_align_jobs/PsiHeldNllJob.DJDq3bvQGsVk` (gold) / `.ulZeaWuNYxLX` (own), `work/i6_core/returnn/forward/ReturnnForwardJobV2.pjOuxrSnKFrt`, `work/i6_core/recognition/scoring/ScliteJob.smdhPQbCOZhI` |
| its scorer init (random, min_dur=2), kept — Z2 reuses it | `S/psi_align_jobs/PsiAlignInitJob.ICyJ7Jhw6lyv` |
| the text donor it starts from (§0d) | `S/lm_prior_jobs/ExportHfLmDirJob.460dedSQ4kAG` |
| frozen gold dev pair set (shared, finished) | `S/scorer_diag/FrozenHeldPairsJob.E8UaEwRF65HW` |
| Z2 code (approach 2) | `sae/grpo/diversity.py`, the `derangement_*` branch of `sae/grpo/psi_scorer.py` `ce_loss`, `lam_div` in `sae/grpo/reward.py` (+ their tests, 406 tests over `sae/`) |
| Z2 entry point | `config/sae_3g_z2.py` |
| Z2 policy init (§1e.1 pseudo-pair SFT on the §0d donor, encoder frozen), read at epoch 5 | `T/ReturnnTrainingJob.wB50bVrFNZRR` |
| Z2 loop (approach 2) | `T/ReturnnTrainingJob.hjxeQdZbG9TY` |
| Z2 per-sub-epoch forensics table | `S/psi_forensics/PsiForensicsTableJob.LMD3vfrMymMy` |
| controlled-null + code-channel code (approach 3) | `sae/coupling_reads.py` |
| Z2 close-out battery (approach 3) | `S/coupling_reads/CouplingNullsJob.Zy5ooKdFZNMF` (gold) / `.S6ajC8aIN3zT` (own text), `S/coupling_reads/CodeChannelJob.EK4hYY18GuzX` |
| Z3 code (approach 4) | `sae/perturb.py`, `sae/negatives.py`, `lam_cons` in `sae/grpo/reward.py`, the perturbed and negative streams in `sae/grpo/trainer.py` / `psi_scorer.ce_loss` / `train_steps/sae_grpo.py`, `perturb` in `sae/av_states.py` + `sae/build_av_states.py`, `AttachUnitStoresV1` in `prefix_lm/model/util/units_attach.py` |
| Z3 entry point | `config/sae_3g_z3.py` |
| Z3 perturbed unit stream | `S/av_states/AvStatesJob.loD3pR4fhvLn` -> `S/quantize_states/AssignUnitsJob.iRguKxcUyGuU` -> `S/quantize_states/PackUnitsJob.WcpJCKlpbfWG` |
| Z3 hinge negatives | `S/negatives/NearestNegativesJob.UO6nWJJvjK6q` -> `S/quantize_states/PackUnitsJob.7SSYAdcE94AR` |
| Z3 loop (approach 4) | `T/ReturnnTrainingJob.q8QJb05Jm0oS` |
| Z3 standing per-sub-epoch reads | `S/coupling_reads/CouplingNullsJob.*` under alias `sae/3g/z3/ep{k}/coupling_{gold,own}` |
| Z2 closing forensics table (all 6 sub-epochs) | `S/psi_forensics/PsiForensicsTableJob.LMD3vfrMymMy` |
| Z4 code (approach 5) | `sae/psi_align_fastbw.py` and `PsiAlign.run_dp`, `within_sequence_repetition` in `sae/grpo/diversity.py`, `lam_rep` in `sae/grpo/reward.py`, `GreedyPoolJob` in `sae/curate.py`, `_transform_train_shard` in `sae/data.py`, the `derangement_*` and `fast_bw` branches of `PsiAlignTrainJob` (+ their tests, 435 tests over `sae/`) |
| Z4 entry point | `config/sae_3g_z4.py` |
| Z4 bed quarters (7135/7135/7135/7134) | `work/i6_core/datasets/huggingface/TransformAndMapHuggingFaceDatasetJob.` `88z7WUCX2OuT` (0), `n0wPtdRUbqn4` (1), `t7NIUMiHrBPI` (2), `Tt1ACmJCKWft` (3) |
| Z4 refresh decodes, rounds 1-6 | `work/i6_core/returnn/forward/ReturnnForwardJobV2.` `3vNKwcAm6KoC`, `ZZXm3j5tuuIj`, `akawreVAboJV`, `6J5qwpr0P9D9`, `zDDvFhap5Rks`, `clJJ8cL8EFqz` |
| Z4 pools, rounds 1-6 | `S/curate/GreedyPoolJob.` `CdvWjGWjgSr6`, `zSfAxQFTWOIp`, `qQAG3WDp2HnG`, `Y0BtLEajG3Bi`, `bLLQJtZ0Mfnf`, `JjobiUAusicZ` |
| Z4 refits, rounds 1-6 | `S/psi_align_jobs/PsiAlignTrainJob.` `5VD9GAQIry6s`, `TkEZ16cgHm4b`, `L4yacX9tM4ZM`, `V4pfWIkUcetQ`, `U9sMpWgZAIaC`, `LtfZ4wHEfPYJ` |
| Z4 loop, rounds 1-6, one sub-epoch each (approach 5) | `T/ReturnnTrainingJob.` `ARNecjhwoSkE`, `nQMglzSGyFc8`, `4cDSB3JZIbpp`, `lwFIe24y1o0S`, `irZ9jE04tBy5`, `BPzZgyT0EmH6` |
| Z4 standing per-round reads, own text, rounds 1-6 | `S/coupling_reads/CouplingNullsJob.` `ruLfo0iBloA0`, `NzSue79lKqPk`, `elJRSC03nd5J`, `v5nGILXJXRdI`, `tKIOgfsO9EVe`, `i88fQDqkaXOQ` |
| Z4 standing per-round reads, gold text, rounds 1-6 | `S/coupling_reads/CouplingNullsJob.` `7Op06QUxMtMk`, `kFxLYAm5derc`, `82pdlYLtWZPk`, `qkerp2AWS76t`, `gsGPHyAdVc3t`, `gPt1rv8PBVi1` |
| Z4 dev WER, rounds 1-6 | `work/i6_core/recognition/scoring/ScliteJob.` `SYX5xMWzUsG2` / `CUD6CCzHc4Wv` (r1), `OTYulBzf9LD8` / `a5jXshMasjMy` (r2), `8DX7DqUnzdnk` / `t3gQ6dXDg3SQ` (r3), `LRnnEoUTJkZA` / `qlZtLpjqB19s` (r4), `EKfufHcJ9kPx` / `LUl0lItfON55` (r5), `zUMMfUshW1SQ` / `3SQQsch2WvKk` (r6), each pair dev-clean / dev-other |
| Z4 dev-clean decodes, rounds 1-6 (the persistence and distinct-string reads) | `work/i6_core/returnn/forward/ReturnnForwardJobV2.` `E6l55PIltKpC`, `spP2gP7nkeTJ`, `JjbK1T1lHhH7`, `boicjaBRxWUV`, `OXv2E5lToWCX`, `bw2goum8WGXb` |

## Verifier feedback

- 2026-08-20: Plan/artifact reconciliation closes two stale execution states. Z2
  `ReturnnTrainingJob.hjxeQdZbG9TY` has finished markers, checkpoints 1--6, and six epochs in
  `learning_rates`; it completed the registered run rather than stopping mid-sub-epoch 5. Z3 also
  completed all six and its primary duration-matched gap remained negative throughout, as Approach 4
  and Conclusion 4 record. `PLAN_3G.md` now reflects both endpoints. The mechanism evidence strongly
  favors nuisance/private coding, but the pre-registered B/C taxonomy is formally incomplete: no
  unit-emission purity/PER result exists, so the controlled nulls must not be presented as that missing
  deliverable.

- 2026-08-13: Base-arm spot-verify against ground truth passes — all 2 703 dev-clean hyps are
  one sentence family (top variants 1 361 + 1 117, `ReturnnForwardJobV2.12kyAcqkZCoY`); wer
  file 132.91; within-group reward std 0.0097 and reward_recon −5.141 in the live
  `learning_rates`; sub-ep-5 derangement gap 0.0045 = matched 5.7359 vs shuffled 5.7404
  (`PsiHeldNllJob.3sbcdJh2IBUj`), matching the Approach table.
- 2026-08-13: Catalog: `PsiForensicsTableJob.z1dyl5dRG4TX` resolves nowhere under work/ —
  fix the hash or cite the job that actually assembled the per-sub-epoch table; until then
  the derangement/allegiance columns for sub-eps 0–4 have no resolvable provenance (the
  sub-ep-5 row is independently confirmed via the held-NLL job above).
- 2026-08-13: Planner verdict recorded in PLAN §3g: outcome (A) confirmed, base arm closed
  after the sub-ep-6 read — do not extend it; Z2 fix package registered there (cross-utterance
  diversity price + 1e fallback audio-pathway init + derangement margin in psi's in-loop
  loss), awaiting the user's funding word.
- 2026-08-14: Z2 sub-ep 1–4 rows verify against the per-sub-epoch held-NLL jobs (own-text
  gaps +0.0021/+0.0530/+0.0688/+0.0852 = `PsiHeldNllJob.Ahn8hbEoAjlR/.1RabgGO7Wn5c/`
  `.KZTKSEUjOLSn/.tNNpbbP8qgl5`; gold −0.0028/−0.0071/−0.0081 = `.INHf00Lwf035/.Ujf4Jv0k7QfZ/`
  `.QaChfknRfj2B`); loop live mid sub-ep 5, in-loop TRAIN-side derangement gap 0.2396 with
  policy gradient ~5e-4 (nonzero, unlike the base arm's 1e-5).
- 2026-08-14: Decisive frame fact behind the conclusion-2 correction:
  `FrozenHeldPairsJob.E8UaEwRF65HW` draws 725 dev-clean + 775 dev-other utterances — the bed
  never trains on them, so the own-text gap cannot be utterance memorization; and the gold
  gap turning negative rules out residual length as the carrier (gold text has the true
  length for its audio, so a length route would lift both gaps). The train-vs-held gap ratio
  (0.240 vs 0.085) says most of psi's hinge progress IS train-side memorization, but the
  held residue is real coupling.
- 2026-08-14: Catalog: `PsiForensicsTableJob.LMD3vfrMymMy` does not resolve yet (presumably
  created at loop close) — until it exists the held-NLL jobs above are the table's provenance.
- 2026-08-14 (later), decode forensics on the sub-ep-3/4 dev-clean decodes
  (`ReturnnForwardJobV2.Pniw84ZqTPWv` / `.ncJEYu3eFtwq`): the code's 316/398 "distinct"
  strings are repeat-count variants of ONE stem phrase per sub-epoch ("come away with me" x k
  at sub-ep 3, "come with us" x k at sub-ep 4); corr(audio n_units, decode tokens) = 0.856 on
  the 725 held dev-clean utts; the utterance-to-string map is unstable across sub-epochs
  (14 of 2703 unchanged); same-speaker rate within a shared codeword 4.5 % vs 2.7 % chance
  (speaker IDs used to evaluate only). Control: GOLD corr(n_units, transcript tokens) = 0.956
  on the same held set. So the code is primarily a DURATION code — repetition count encodes
  clip length under a fluent English carrier phrase.
- 2026-08-14 (later), self-correction of the 08-14 bullet above: "the negative gold gap rules
  out residual length as the carrier" was too strong. Gold's length-duration coupling (0.956)
  is tighter than the code's (0.856), yet gold shows no gap — because psi has no model of gold
  text at all (gold matched nll 5.958 vs own-text 5.262: psi fits the code 0.7 nats/frame
  better than English), so it cannot price ANY channel through text it cannot fit. The held
  own-text gap therefore proves a generalizing audio-code link, but its information content is
  most plausibly duration (plus a weak speaker/rate residue), not phonetic content. Decisive
  read registered in PLAN §3g: duration-matched derangement (partner matched on audio n_units
  as well as text length) — what survives it is content beyond duration.
- 2026-08-14 (later), live sub-ep-5 trend from the training job: psi_ce 6.24 -> 5.75 -> 5.59
  -> 5.52 -> ~5.48 and train-side in-loop gap 0.0046 -> 0.1418 -> 0.2396 -> 0.2399 -> ~0.256
  (both still moving, both decelerating); distinct-frac still falling (~0.28).
- 2026-08-14 (later), content-beyond-duration probe on the sub-ep-4 decode
  (`ReturnnForwardJobV2.ncJEYu3eFtwq` x held set): the 398 strings decompose into 3 large
  repetition families ("come away/with/down" x k) plus a one-off tail (246 stem families);
  stem choice within a duration band is near-independent of speaker (joint 0.0072 vs 0.0059
  independence). But the repeat count carries a second channel: corr(decode-length residual,
  GOLD-length residual | audio n_units) = 0.252 on 725 held dev-clean utts (~7 SE from zero) —
  at fixed clip length, wordier gold gets more repeats, i.e. the code partially encodes SPEECH
  DENSITY, the first content-adjacent signal. Sharpens registered read (iii)'s expectation and
  predicts a small positive survival on read (i) (duration-matched derangement), well below
  0.085.
- 2026-08-15: Z3 approach-4 table (all columns, sub-eps 1-3) and the approach-3 close-out rows verify
  exactly against job outputs (WERs `ScliteJob.Y4ARxAocxxtB/.RLh81oOQZw0F/.168dFcA2rMKQ/.ZhmU1eYucf3q/`
  `.LQoi8pTrti0q/.Q7oY3sIwuDm3`; gold/own gap columns = the nll reads `PsiHeldNllJob.OH8Bf6Zh00XO/`
  `.YxkUQXUnWJhM/.HJuW4R24EYX1` and `.wnmzkugAyvy0/.tZFBvktAAmne/.rT5I2AqUN5cI`; dur-matched /
  within-speaker = the ce_loo reads `CouplingNullsJob.eKxARV7Porme/.c08zHLEusbBE/.1sK3slmPiV3S`;
  close-out `.S6ajC8aIN3zT/.Zy5ooKdFZNMF` + `CodeChannelJob.EK4hYY18GuzX`). The footnote that the two
  gap columns are nll where dur-matched/within-speaker are ce_loo is load-bearing — checking the gap
  columns against CouplingNullsJob's lenmatched rule false-mismatches.
- 2026-08-15, decode forensics ep0-3 (`ReturnnForwardJobV2.3GailHU9q5vn/.CgTP5IdPYuOm/.JgkzORZvpD5S/`
  `.8N6uteP2BAsK` x the 725 held): ep1 briefly breaks the repeat code (exact stem-x-k 0.15%,
  corr(n_units, tokens) 0.255) but is its own degeneracy — a closed set of ~380 canned fluent
  sentences reused across 2703 utts; ep2-3 rebuild the duration code STRONGER than Z2: exact stem-x-k
  98.7% (Z2 42-48%), median stem 2 tokens (Z2 4), mean repeat factor 7.6, corr 0.811, and the whole
  stem vocabulary turns over each sub-epoch (string persistence 0/2703 at every consecutive pair;
  stems canned -> "how long" -> "the life/the shot/the ship").
- 2026-08-15, content: none — content-word overlap with own gold sits at or below its deranged-gold
  chance at every sub-epoch (ep3 0.0072 vs 0.0108; the all-word 0.265 vs 0.257 is a "the" artifact of
  the stem family); the density channel is 0.317 at ep3 (Z2 sub-ep4 0.252), i.e. speaking rate through
  the repeat count. Attribution fix for the earlier Z2 reference numbers: 0.856 / 0.252 / 14-of-2703
  all belong to Z2 sub-ep 4 (sub-ep 3 is 0.809 / 0.092).
- 2026-08-15, gate: the pre-registered primary read is failing so far — held dur-matched own gap
  -0.0137/-0.0160/-0.0142 vs "exceeds +0.0272 and grows two consecutive sub-epochs" — while surface
  metrics improve (the first sub-100 WER 97.18 is a decode-length effect at mean 15.6 tokens; in-loop
  distinct frac 0.904 and reward std 0.136, ~0.25 live in ep4, are real but content-free). Two prices
  are paid, not binding: within-seq bigram repetition is 0.807 (above Z2's 0.68-0.71) because lam_div
  is cross-utterance only and cannot see it (`sae/grpo/reward.py:10`), and only the tempo half of the
  perturbation mixture punishes duration coding (VTLP/noise copies keep the frame count, ratio
  1.0000), so cons is paid 0.011 -> 0.052 while the code rebuilds.
- 2026-08-15: the within-speaker column is not a clean speaker meter under Z3's negative rule — the
  training negatives are acoustically-nearest (16.0% same-speaker vs 0.32% chance), so psi trains
  against same-voice mispairings; read it only beside its like-for-like control row (ep3 +0.0094 vs
  control -0.0166).
- 2026-08-15, live ep4 (train-log only, ~74% through): grpo_text_len 6.8 -> 27.7 within the epoch
  (the repeat factor growing again), lm_prior -0.205 -> -0.042, reward std ~0.25, psi_ce ~5.81.
- 2026-08-16: Z4 rounds 1-3 verify against ground truth — the dur-matched column is exactly the
  dur_text gaps +0.0303/+0.0018/+0.0036 (`CouplingNullsJob.ruLfo0iBloA0/.NzSue79lKqPk/`
  `.elJRSC03nd5J`), WERs match the cataloged ScliteJobs. The WER level is decode-length
  arithmetic, not behavior: r3 hyp 78 217 words vs 54 402 gold (1.44x, 27 238 ins) against Z3
  ep2's 45 638 (0.84x, 3 865 ins); Z4 r3 has 6.2% words correct vs Z3 ep2's 0.9%.
- 2026-08-16, decode forensics r1/r3 (from the sclite alignments): adjacent-word repetition is
  exactly 0 in the dev decodes — lam_rep killed the repeat carrier outright — and the arm's route
  is a closed library of canned fluent sentences reused across utterances: r1 298 distinct strings
  over 2703 (top string 742x), r3 1085 distinct at 8-gram reuse 0.92, mean 28-29 tokens pinned by
  the length band. This is Z3 ep1's canned-sentence degeneracy, now stabilized.
- 2026-08-16, load-bearing: round 1's +0.0303 (above the +0.0272 bar) is the refit on the INIT's
  own decodes (`ReturnnForwardJobV2.3vNKwcAm6KoC` decodes `ReturnnTrainingJob.wB50bVrFNZRR`
  epoch.005) — the SFT init map itself carries that coupling; one sub-epoch of GRPO under the Z4
  reward erased it (+0.0018) and round 3 stays at chance (+0.0036). The primary gate's "grows two
  consecutive sub-epochs" is failing so far; rounds 4-6 not finished on disk at this read.
- 2026-08-16, refit convergence shape (monitors.json of `PsiAlignTrainJob.5VD9GAQIry6s/`
  `.TkEZ16cgHm4b/.L4yacX9tM4ZM/.V4pfWIkUcetQ`): held nll is best during the 4-epoch guided
  warmup (5.12-5.29, alignment prior annealing 1.0->0.25), then jumps a full nat to the unigram
  floor (~6.11 vs floor 6.095) the epoch the prior reaches zero and the contrastive/margin/
  derangement terms switch on, and 26 further epochs never recover it (5.98-6.06). The held-nll
  pin (pre-registered) therefore lands at epoch 2-4 in every round — before the first hinge
  update — so every SERVED Z4 scorer has had zero hinge training; the hinge phase converges in
  its own terms (contrastive 3.38->0.48 / 2.31->0.21 / 2.10->0.11 / 1.77->0.11) but at
  unigram-floor likelihood. Refit held ce across rounds is flat (5.44 / 5.55 / 5.22).
- 2026-08-16, in-loop reward is a sawtooth with no cross-round accumulation (train logs of
  `ReturnnTrainingJob.ARNecjhwoSkE/.nQMglzSGyFc8/.4cDSB3JZIbpp`, first/last-300-step means):
  within every round recon climbs ~0.2 nats (-5.35->-5.18, -5.46->-5.28, -5.35->-5.17) and
  lm_prior halves (-0.18->-0.11), while within-group reward std collapses to ~0.05 by round end
  (r1 starts at 1.01 step-0, later rounds ~0.10-0.15) — each fresh scorer is exhausted within
  one sub-epoch, and each swap resets recon ~0.2-0.3 nats down; end-of-round levels are flat.
  Rollout text length ratchets across rounds (28.7->32.8 | 26.3->33.9 | 31.9->38.3 | r4 live
  31.0->41.4) with train length violations ->0.008 — the policy rides the length band's ceiling.
- 2026-08-16 (later), correction of the "load-bearing" bullet above: the r1 coupling read's own
  text traces to the POST-ROUND-1 checkpoint, not the init (`CouplingNullsJob.ruLfo0iBloA0` <-
  `ReplaceHeldPairsTextJob.4DVjJcd5ZFq9` <- `SearchOutHypsJob.CdW50uedoPg9` <-
  `ReturnnForwardJobV2.E6l55PIltKpC` <- `ExtractAvSubmodelJob.dJoikC82LxJv` <-
  `ReturnnTrainingJob.ARNecjhwoSkE` epoch.001). So +0.0303 = psi_1 (the only scorer FIT on the
  init's decodes) reading the post-round-1 policy — the above-bar coupling SURVIVED round 1
  under an init-fed reader; "one sub-epoch of GRPO erased it" is WRONG as stated. The r2
  collapse (+0.0018) changes the scorer's fit corpus (init pool -> round-1 canned library) and
  the policy being read simultaneously, so scorer-side and policy-side loss are confounded in
  the round reads. What stands: the only above-bar reading on the arm came from the only scorer
  fit on the init pool; every scorer fit on post-GRPO canned decodes reads ~0 and cliffs to the
  unigram floor once unguided (its fit corpus has ~300 distinct texts vs the init pool's full
  variety).
- 2026-08-16 (later), mechanism of the r1 +0.0303 settled from the standing reads (no new jobs).
  First, the null's direction: the own TEXT is held fixed and the AUDIO is swapped to the partner's
  (coupling_reads.py:189-192). The speaker split then isolates the channel: on the 818 r1 rows
  where a same-speaker partner exists, a different-speaker duration-matched audio swap costs
  +0.0475 nats/frame while the same-speaker swap costs +0.0015 (`CouplingNullsJob.ruLfo0iBloA0`,
  dur_text_on_these_rows vs dur_text_spk) — the r1 read is a SPEAKER/voice-channel code
  (speaker-conditioned sentence choice inherited from the init, priced by the init-fed psi_1),
  not content. The spk rule's worse state matching (rel_state_diff 0.099 vs 0.002) works against
  this attribution and it survives anyway. Speaker IDs entered on the evaluation side only.
- 2026-08-16 (later), the fine-length hypothesis (sub-tolerance length match driving +0.0303) is
  refuted by two controls: within the null's own 5% duration band the residual corr(text chars,
  n_units) is ~0 in every text set including GOLD (0.019 gold / 0.020 r1 / -0.026 r2 / 0.059 r3;
  sampling scale 0.026) — there is no sub-tolerance length information to leak — and gold text,
  whose coarse length tracking is the tightest of all (corr 0.963), reads dur_text ~0 under every
  round's scorer (+0.0019 / -0.0002 / -0.0039).
- 2026-08-16 (later), the refreshed scorers are not generically blind: on the bit-identical gold
  pair set the lenmatched gap grows monotonically in the round (+0.0127 -> +0.0236 -> +0.0282;
  z3-scorer scale ~0.007) — later scorers price speech DENSITY (equal-duration, different
  text-length audio) ever more strongly, i.e. the length-coded canned corpora teach
  frames-per-state structure: another nuisance channel, not content. Caveats: r2 reads at
  pinned_epoch 2 (r1/r3 at 4), so cross-round scorer trends mix checkpoint stage;
  ReplaceHeldPairsTextJob headers carry the gold job's n_tokens/n_frames (stale metadata, off
  48-60% vs actual row sums — read row sums, not headers).
- 2026-08-16 (later), contrast read against Z2 for the successor design: Z2's close-out coupling
  was NOT speaker-carried — its approach-3 table already holds the split: same-speaker swap on
  the 965 spk-feasible rows costs +0.0392 vs +0.0216 for the different-speaker control on the
  same rows, i.e. the gap SURVIVES removing the voice channel — where Z4 r1's +0.0303 dies under
  the same rule (+0.0015). The one scorer on this track to show non-speaker held coupling (Z2's)
  is also the only one trained with the derangement hinge operative (co-trained in-loop); every
  served Z4 refit pins pre-hinge (epochs 2-4, hinge needs prior_w==0). Whether Z2's +0.0392 is
  content or density is not settled by that table (dur_text controls density only as far as
  decode length tracks it, corr +0.2326 given frames); the gold-set density read now exists as
  the disambiguator for any successor battery.
- 2026-08-16 (gate close, rounds 4-6): verified against ground truth — round-3/6 sclite
  totals and insertions reproduce (143.8 with 27238 ins; 105.6/107.1 with 5460 ins; splits
  pinned by reference word count 54402/50948), the reward-spread recovery reproduces from
  the training jobs' own learning_rates (train_loss_reward_std_within_group 0.0737/0.0579/
  0.0741/0.0785 at r1/r3/r4/r6), and the persistence and distinct-string reads recompute
  independently from the decode dumps (25 and 26 of 2703; 1561/1504 distinct, top 43x/34x).
  Gate verdict FAILS recorded in PLAN_3G.md; the exhaustion framing in the planner's own
  in-flight reads is replaced there (it described rounds 1-3 only). Conclusion 6's
  insertion span (27584 -> 5460) is round 1 -> 6 while its WER sentence spans round 3 -> 6
  (round 3 is 27238) — wording only, no number wrong. Probe defect mechanism sharpened
  beyond the implementer's report: the dump is NOT one-decode-per-utterance — each
  utterance holds three rows (one T=0.7 sample, gold, greedy; groups of 3 in the file) and
  the probe correctly filters to sampled rollouts, leaving G=1 — so the proposed fix
  (multi-sample dump, G>1) is right and the wiring claim ("greedy refit corpus") is not;
  nothing logged is invalidated (the gate never references the probe). Catalog rule: the
  Z4 standing per-round coupling reads are cited by alias only (sae/3g/z4/r{k}/...) — pin
  the concrete CouplingNullsJob hashes.
