# Audit: G4a.3b-R read on S3b-R arm lam3 (2026-09-15)

Fresh-context, read-only re-derivation. No edits, no reruns.

## Verdict

- **PER clause: OVERTURNED (fails).** dev-other greedy PER at sub-epoch 4 = **0.846651**, the gate
  needs < 0.50.
- **Gap clause: CONFIRMED (passes).** gap_per_frame = **+1.910386**, macro mean **+1.729825**
  [+1.605862, +1.860729], CI excludes 0.
- **Rate clause (in the gate text, omitted from the dispatch): CONFIRMED (passes).** 6.9530/s inside
  [0.6, 1.5] x rho = [5.797, 14.493]/s at rho = 9.6619373279 Hz.
- **"The gap is positive and not explained by a length or emission-count artifact": CONFIRMED.**
- **Gate G4a.3b-R is conjunctive, so the gate is NOT met by lam3.**

## 1. PER, phone rate, decode statistics (re-derived)

`output/.../sae_4a_s3b_rate/lam3/ep{4,8}/dev-other/per.json`, convention stated in the job itself:
`(S+D+I)/N` over the split, greedy per-frame argmax, collapse repeats, drop blank, SIL dropped
(`drop_sil=True`, `blank_id=0`), scored against `GoldPhonesJob.ZGSp0hxyd2YP` (dev-clean/dev-other MFA
gold; the only label use, evaluation only).

| | ep4 | ep8 |
|---|---|---|
| S / D / I / N | 92940 / 53246 / 3904 / 177275 | 120077 / 28962 / 8031 / 177275 |
| (S+D+I)/N recomputed | **0.8466507** | **0.8860245** |
| banked `per` | 0.8466507 | 0.8860245 |
| macro PER | 0.876479 | 0.931434 |
| emitted ph/s (no SIL) | 6.95303 | 8.49714 |
| emitted ph/s (with SIL) | 7.66196 | 9.07204 |
| ref ph/s (MFA) | 9.63472 | 9.63472 |
| distinct strings | 2864 of 2864 | 2864 of 2864 |
| symbol types | 37 | 39 |
| empty hyps | 0 | 0 |

Arithmetic reproduces exactly; 2864 utts and N = 177275 identical at both epochs, so the two reads
share one denominator and one gold. Not a collapsed decode (distinct = 1.0000, most common string
count 1). `phone_rate` files agree with `per.json`.

Comparability note on the band: `DecodeStatsJob`'s own printed band is [0.6, 1.5] x **9.8** (gold
text rate, hash-bound) = [5.88, 14.70]/s and prints `IN BAND = True`. The campaign's label-free band
is [0.6, 1.5] x rho = [5.797, 14.493]/s. 6.9530 is inside **both**, so the clause holds either way;
the code-review rule (read `phone_rate`, never `rate_in_band`) was followed here.

Provenance: posteriors `ReturnnForwardJobV2.EmiPxkQx13JF` <- `ExtractSubmoduleCheckpointJob.JftcuHnQEzEU`
(prefix `recognizer.`) <- `ReturnnTrainingJob.DF6blPpto23t/output/models/epoch.004.pt`. Resolved
`returnn.config` lines 43-46: `lam_rate=3.0`, `rate_rho_hz=9.6619373279`, `rate_fd_eps=0.25`,
`rate_fd_mode="central"`. All 8 sub-epochs present in `learning_rates`.

## 2. Derangement gap at ep4

`work/speech_llm/sae/emc/s3_jobs/S3DerangementGapJob.dcvnxBubnFD1/output/derangement_gap.json`
(+ the job's unsymlinked `per_utterance.json`, 500 rows).

- scored 500 of 500 selected, eligible 2863, dropped all-zero (`empty_decode`, `no_units`, `no_eta`,
  `infeasible_own_decode`, `no_donor` = 0), speakers 33.
- own log p/frame -5.471288, deranged -7.381674, **gap_per_frame +1.910386**.
- Recomputed from the 500 rows: sum(own-der)/sum(frames) = **+1.910386**; macro mean of per-frame
  deltas = **+1.729825**. Both match the banked values to 6 dp.
- 488 of 500 per-utterance deltas positive; range -1.9573 to +4.2704.
- CI [+1.605862, +1.860729], speaker-clustered bootstrap, 10000 resamples, seed 42; excludes 0.
- identical own/donor strings 0 of 500; distinct decoded strings 500 of 500 -> not the collapse mode
  the convention warns about.

Checks demanded by the brief:

- **Draw is the pre-registered one.** `s1a_job.select_utterances(eligible, 500, seed=SELECT_SEED=0)`
  = `sorted(tags)` permuted by `RandomState(0)`, first 500; eligible = split tags with non-empty MFA
  gold (gold opened for the tag list only). This is the same helper S1a and
  `emc_train_jobs.DerangementGapJob` call, so the sample does not move with the decode. 33 speakers
  matches the `_SELECTION_NOTE` prediction for dev-other.
- **Pairing is index-keyed, not positional.** `_rows()` builds `{tags[r["index"]]: r for r in rows}`
  with `assert sorted(r["index"] for r in rows) == list(range(len(items)))`, and the source comment
  states `evaluate` emits rows in length-bucket order. The d69f938 fix is present in the code that
  produced this json.
- **Decodes are this checkpoint's.** `hyps` = `GreedyPerJob.02WAVwvzkVTi/output/greedy_phones.json`,
  the same job whose `per.json` is the ep4 PER above; `checkpoint` =
  `ExtractSubmoduleCheckpointJob.bUO5EqlcKweW` <- `DF6blPpto23t/output/models/epoch.004.pt`
  (prefix `reverse.`), alias `sae/4a/s3b_rate/lam3/ep4/phi`. Same sub-epoch on both sides, no refit.

## 3. Can this gap arise without phone content?

What the reader already controls:

- **Frames are identical on the two sides by construction.** Both scores are
  `log p_phi(z_u | y, eta_u)` on the *same* z_u with the *same* eta_u; only the conditioning string
  y changes. Utterance length therefore cannot produce the gap at all.
- **Speaker is held fixed** (donor is another utterance of the same speaker, `build_derangement`),
  and eta is the own utterance's, so speaker-specific unit statistics are matched.
- **Emission count is matched by construction only approximately**: the donor is the same-speaker
  utterance with the *nearest* decoded phone count (tie-break by tag, feasibility-filtered).

Controls I computed from the per-utterance rows (own tokens, donor tag -> donor tokens):

| subset | n | macro | corpus | frac positive |
|---|---|---|---|---|
| all | 500 | +1.7298 | +1.9104 | 0.976 |
| exact token match (|dtok| = 0) | 123 | **+1.4821** | +1.5171 | 0.959 |
| |dtok| <= 1 | 243 | +1.5818 | +1.6189 | 0.979 |
| donor LONGER than own (donor advantaged) | 170 | **+1.5868** | +1.6759 | 0.959 |
| own longer than donor | 207 | +1.9945 | +2.2016 | 1.000 |

mean(own_tok - donor_tok) = +2.01, median 0; r(gap, own_tok - donor_tok) = 0.38, r(gap, |dtok|) = 0.37.
There is a token-count component, but the gap survives exact emission-count matching (+1.48) and
survives the adversarial subset where the donor has *more* tokens (+1.59). So: **not a length or
emission-count artifact.**

What this does NOT establish. The gap is a theta/phi self-consistency measure: both strings are the
recognizer's own output and phi was trained jointly against this theta. A positive gap says only
that phi prefers the *utterance-specific* string theta produced for these units over a same-speaker
donor string. It does not require the string to be phonetic, and in this run it demonstrably is not:
the same 500 decodes come from the checkpoint whose PER against MFA gold is 0.847, and the decodes
are utterance-specific (500 distinct strings for 500 utterances) exactly as the "non-phonetic but
utterance-specific" alternative predicts. Per the standing rule that magnitudes are per-arm, the
+1.91 must not be compared in magnitude to the S3 baseline's -0.3218 (own log p/frame -5.47 here vs
-14.72 there; different phi, different units).

Baseline comparability caveat, if the sign flip is quoted: the S3 cold-start read
(`S3DerangementGapJob.Ez8bGMB7LnPX`, ep4 dev-other) scored **314 of 500** (164 `infeasible_own_decode`,
22 `no_donor`, 12 identical donor strings, 256 distinct strings) at PER 0.8955 and 1.5796 ph/s. lam3
scored 500 of 500. The two gaps are on different utterance subsets and are not a paired contrast.

## 4. Checkpoint selection

`UnsupervisedCheckpointSelectionJob.zMRJTSwSmjRe`: `selected_epoch = 4`, label-free
(weighted_lm_ppl = 10^(-lm_score_sum/(num_pred_chars+nsentences)) / vocab_seen_pct^2.0, argmin),
5567 utterances (dev-clean + dev-other pooled), kenlm `CreateBinaryLMJob.hvZoC014xnIe`.

| ep | weighted_lm_ppl (banked) | recomputed | lm_ppl | vocab_seen | held_out_l_tau |
|---|---|---|---|---|---|
| 4 | 112.63495 | **112.634947** | 96.373 | 37 | 1.96868 |
| 5 | 140.27420 | 140.274203 | 133.348 | 39 | 1.75801 |
| 6 | 183.16741 | 183.167407 | 174.124 | 39 | 1.79126 |
| 7 | 195.21899 | 195.218995 | 185.580 | 39 | 1.75968 |
| 8 | 139.58336 | 139.583364 | 132.691 | 39 | 1.72214 |

The formula reproduces every entry; argmin is ep4. `held_out_l_tau` matches `dev_loss_l_tau` in
`learning_rates` (ep4 = 1.9687) and is tie-break only, so it did not drive the pick. The pool is
sub-epochs 4..8 only, which traces to the arm config's own docstring (line 49: selection over the
tau = 2 checkpoints, the anneal 8 -> 2 runs over sub-epochs 1-4) -- a pre-registered restriction, not
a post-hoc one, and not load-bearing for the gate, which is pre-registered at sub-epoch 4 anyway.
Note ep4 is the *earliest* member of the pool and also the chosen one.

## Frame

Everything constant in this run traces: lam_rate 3 and rho 9.6619 Hz from the resolved config, the
8-sub-epoch / tau 8->2 schedule from the S3 reference setup, the 500-utterance derangement draw from
the S1a helper, the gold used for eligibility and PER only. One frame gap worth stating: the gate
reads "in **any** arm at sub-epoch 4", but `ACTIVE_ARMS = ("lam3",)` -- lam1 and lam10 were never
built. lam3 alone can therefore falsify a PASS (it does), but a FAIL verdict for the whole gate needs
either those two arms or an explicit decision not to fund them.

Note also that the pre-registered gate carries three clauses (PER, gap with CI excluding 0, rate in
band); the dispatch quoted only two. Restating the gate as PER + gap would still fail on PER, so the
conclusion is unchanged, but the gate should be read as written (SAE_4A.md:~924).

The outcome matches the gate's own written expectation: "S3 sub-epochs 5-8 already emitted 4-8
phones/s at PER 0.84-0.90 with a negative gap, so a non-zero rate alone may still be content-free."
Here the rate is in band and the gap is now positive, and the PER is still 0.85.

---

## Addendum (same day): sub-epoch 8 derangement gap

`S3DerangementGapJob.0scGuFzaKnMr`, alias `sae/4a/s3b_rate/lam3/ep8/dev-other/derangement`.
Provenance: phi = `ExtractSubmoduleCheckpointJob.WGZtvhXfqYZR` <- `DF6blPpto23t/output/models/epoch.008.pt`;
hyps = `GreedyPerJob.MNJiZAcCcNhE`, the same job whose `per.json` gives the ep8 PER 0.886025. Same
sub-epoch on both sides, no refit. Same draw (eligible 2863, selected 500), same bootstrap
(10000, seed 42), same index-keyed pairing code.

Re-derived from the job's `per_utterance.json` (500 rows):

- gap_per_frame **+3.289171** (banked +3.289171); macro **+3.183771** (banked +3.183771);
  CI [+3.044177, +3.330466], excludes 0.
- own log p/frame -4.377021, deranged -7.666193.
- **500 of 500 per-utterance deltas positive** (ep4: 488/500). 500/500 scored, 0 dropped, 33
  speakers, 0 identical donor strings, 500 distinct decoded strings.

Matched controls at ep8 (frames identical on both sides by construction, as at ep4):

| subset | n | macro | corpus | frac positive |
|---|---|---|---|---|
| all | 500 | +3.1838 | +3.2892 | 1.000 |
| exact token match | 79 | +3.2406 | +3.2471 | 1.000 |
| |dtok| <= 1 | 211 | +3.1232 | +3.1719 | 1.000 |
| donor LONGER than own | 192 | +3.0449 | +3.0640 | 1.000 |
| own longer than donor | 229 | +3.2806 | +3.4468 | 1.000 |

mean(own_tok - donor_tok) = +2.50, median 0; r(gap, own_tok - donor_tok) = 0.29. So at ep8 too the
gap is **not** a length or emission-count artifact -- it is if anything cleaner than at ep4, since
every matched subset is positive for every utterance.

**This is the decisive within-arm control for "is the gap phone content?" and it says no.** The
gap-vs-PER pairing inside one arm, one metric, one utterance set, one reader:

| sub-epoch | dev-other greedy PER | emitted ph/s | gap_per_frame | macro [CI] | frac utts positive |
|---|---|---|---|---|---|
| 4 | 0.846651 | 6.9530 | +1.910386 | +1.7298 [+1.6059, +1.8607] | 0.976 |
| 8 | 0.886025 | 8.4971 | +3.289171 | +3.1838 [+3.0442, +3.3305] | 1.000 |

PER gets **worse** from ep4 to ep8 (+0.039 absolute, both squarely in the content-free 0.84-0.90 band
the gate's own text names for S3 sub-epochs 5-8), while the gap **grows by 72 %** and its
speaker-clustered CIs do not overlap. A quantity that rises monotonically as phonetic accuracy falls
is not measuring phonetic content; it is measuring theta/phi self-consistency, which the joint
training increases with every sub-epoch. The magnitudes are per-arm and per-checkpoint (own log
p/frame moves -5.47 -> -4.38 between the two phis), so the +1.91 -> +3.29 change is not itself a
calibrated effect size either -- but the *sign of the trend against PER* is the point, and it is
unambiguous.

Consequence for the gate read: the gap clause at sub-epoch 4 still passes as written
(gap > 0, CI excludes 0) and the PER clause still fails at 0.846651, so the verdict above is
unchanged -- G4a.3b-R is not met by lam3. What the ep8 read adds is that a future PASS on the gap
clause alone should not be read as evidence of content: this arm shows the gap clause passing, and
passing harder, at PER 0.886.
