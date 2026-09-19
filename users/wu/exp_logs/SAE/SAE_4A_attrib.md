# SAE 4A attribution: why the GAN takes off and the cycle does not

Registered 2026-09-19 on the user's authorization ("implement and execute 1,2,3,4"; parallel where
possible). Diagnostic phase inside §4a. Nothing here is cold-start unsupervised progress; the GAN arms
are diagnostics of the objective, not mainline components (`SAE_ref.md` label quarantine and no-GAN
constraint stand). Gold enters evaluation only.

## State

Phase registered; design review and three implementers dispatched in parallel (steps 1, 2+3, 4).
No job launched yet. Watcher: none. NEXT: code review of each delta, then launch steps 1-3 at once;
step 4 launches after its weight-0 profile (100 updates) projects a run under the allocation limit.

## Question

The blankfree recognizer (`SAE_4A_blankfree.md`) now equals the wav2vec-U 2.0 generator (input 1024,
BN scale 30, residual linear, kernel 9 / stride 3 / pad 4, no blank). The local GAN reproduction
`FairseqW2vu2TrainJob.HOb2GgtYT7Bc` (seed 0, 150k updates, 37552 s) reaches dev-other greedy PER 0.214
(`W2vu2PerEvalJob.ptwMk3TuPPYb`; seeds 0-3: 0.214/0.205/0.215/0.168, `SAE_1c.md`). Every cold cycle
run stays at 0.86-0.90. Two hypotheses (user, 2026-09-19):

- H1: the fixed trigram prior does not match the text distribution the way a discriminator does.
- H2: the jointly trained reverse model degrades the recognizer.

Orchestrator hypothesis (pre-registered): H1 in its directional form is primary. The cycle objective is
a likelihood under a fixed prior (mode-seeking: rewards frequent n-grams, never penalizes a missing
mode); the GAN is mode-covering and has no second trainable module, so the generator alone must carry
the content. The cold reverse model is the cycle's only input-dependence and supplies none from random
initialization (P-BT probe, `SAE_4A.md`); jointly it absorbs the mismatch (S3b-OR joint vs frozen
+0.19/+0.26). Meta's auxiliary K=64 term is not the difference: added to our objective it gave 0.888
(S3b-CT). Prior evidence that the trigram plus exact marginalization learns from flat theta when the
reverse side is informative and frozen: S3b-OR frozen phi 0.454 (ep4) / 0.368 (ep8).

## Design: a 2x2 with one banked corner, plus two side arms

| | no reverse term | reverse term (jointly trained) |
|---|---|---|
| GAN distribution term | banked: 0.214 (s0), spread 0.168-0.215 | step 4 (exp 1) |
| trigram prior, exact marginalization | step 2 (new cheap arm) | banked: blankfree ep4 0.865 |

Step 3 (exp 2) changes the reverse observation stream (enc50 K=500 to MFCC k-means K=64) inside the
bottom-right cell. Step 1 is training-free.

### Step 1: training-free mode-seeking check (CPU sisyphus job, `analysis/` reader)

Rows: blankfree ep1 and ep4 dev-other greedy output (`greedy_phones.json`, SIL-removed), GAN seed-0
dev-other greedy output (from `W2vu2PerEvalJob.ptwMk3TuPPYb` or a registered decode of
`HOb2GgtYT7Bc`), gold dev-other phones (`GoldPhonesJob.ZGSp0hxyd2YP`). Text side: the phonemized
unpaired text corpus behind the training trigram, SIL stripped. Primary convention: SIL-removed
strings everywhere; a trigram estimated on the SIL-stripped text by the job itself (same smoothing
for every row); n-gram JSD for n=1..4 between the row's n-gram distribution and the text corpus's
(Lin et al. 2022 axis; threshold 0.27 at n=4). Secondary: mean log P3 per token under the training
SIL-inclusive prior for the two model rows on their SIL-inclusive strings (gold n/a). Uncertainty:
utterance-block bootstrap, 1000 resamples, seed 0, CI95 on each JSD and each mean log-prob.

Predictions (fixed before the numbers): cold ep4 mean SIL-free trigram log-prob per phone >= GAN's
minus 0.10 nats; cold ep4 4-gram JSD > 0.27 and exceeds the GAN's by >= 0.05 with non-overlapping
CIs; GAN 4-gram JSD within 0.10 of gold's. All three true = mode-seeking signature CONFIRMED.
Any false = the mechanism claim is not supported by this read; report which.

### Step 2: trigram marginal, no reverse term (cold blankfree bed, 4 subepochs)

Registered blankfree model, data, schedule, seed and evaluations unchanged; single delta: the reverse
segment score is identically zero for every segment (emission and duration both removed), phi receives
no update. Band, duration support, repeat rule, trigram, tau anneal 8 to 2, rate and aggregate terms
unchanged. Evaluate ep1 and ep4 exactly as the blankfree graph (dev-clean/other PER, decode stats,
literal examples) plus paired per-item dev-other PER versus blankfree ep4 with speaker bootstrap.

Predictions: dev-other ep4 PER within +-0.03 of 0.865 (paired CI covering zero or |delta| < 0.03) and
the same profile (phones/s, modal first-phone share within 15 points of 58.1%). Reading rules: within
band = the cold reverse model contributes nothing; no-reverse worse by > 0.05 = the cold reverse model
carries some content; no-reverse better by > 0.05 = the reverse model actively harms. Ceiling 1 GPU x
1 h.

### Step 3: trigram marginal, reverse observation MFCC K=64 (cold blankfree bed, 4 subepochs)

Single delta: the reverse observation stream z becomes the MFCC k-means K=64 codes already used by
the S3b-CT content term (`mfcc_codes.py`), on the identical retained 50 Hz clock, same VAD mask, same
utterance mapping, verified per utterance against `BlankfreeVadHdfJob.SAjz8y1cT06g`; reverse emission
vocabulary 64. Everything else unchanged, including eta, durations, trigram, schedule, evaluations.
Named difference from the paper: theirs is a forward per-frame CE onto fixed K=64 targets; this is a
generative segmental reverse model over K=64 observations.

Predictions: fails like the other cold runs (dev-other ep4 PER > 0.80). Reading: PER < 0.50 at ep4 with
positive own-phi gap = an unsupervised route (original G4a.3 PER clause), then extend to 8 subepochs;
otherwise report the paired delta vs blankfree ep4 and stop. Ceiling 1 GPU x 1 h.

### Step 4: GAN plus the reverse term (exp 1), fairseq stack

Host: the reproduction's own `FairseqW2vu2TrainJob` code path (fairseq w2vu2, env `w2vu`), so the
control is the banked run. Delta: add to the generator loss lam_rev x L_rev, where L_rev is the
blankfree exact alignment-sum reverse marginal at beta=0 (no trigram; the lattice keeps only the
last-phone repeat rule), tau=2 fixed, over the generator's stride-3 logits (T=ceil(S/3), band 25,
d_min=2, phone D=25, SIL D=50), z = enc50 K=500 units on the artifact's VAD-masked 50 Hz clock
(per-utterance length equality with the fairseq `.lengths` verified), phi = the registered blankfree
reverse model with frozen eta, trained jointly with Adam(0.5, 0.98) lr 3e-3 in the generator update.
Normalization per retained frame S as in the blankfree loss. No rate or aggregate term. The GAN
losses, aux MFCC CE, schedule, 150k updates and seed handling stay as in the reproduction. At
lam_rev=0 the code path must be byte-identical to the reproduction (no phi, no unit field); existing
job hashes unchanged.

Arms: lam_rev in {0.01, 0.1, 1.0} x seeds {0, 1} = 6 runs, plus a weight-0 seed-0 rerun under the
modified code. Before the full launch: 100-update profile at lam_rev=1.0 and 0 at the run shape;
projected wall time x 1.25 must fit the allocation, else the job is made resumable from
`checkpoint_last.pt` across allocations before launch. Evaluation: the reproduction's greedy PER
job on dev-clean/other (`W2vu2PerEvalJob`), same items, plus paired per-item dev-other PER vs the
seed-0 control with speaker bootstrap; dev reverse term per retained frame reported per arm.

Predictions: weight-0 rerun within +-0.03 of 0.214 (else the port is not a no-op: STOP, fix). At
lam_rev=0.01 both seeds within +0.03 of the control = the reverse term is harmless under a
mode-covering objective, H1-directional confirmed. All six arms >= control + 0.10 = the jointly
trained reverse model absorbs even the GAN's signal, H2 primary. Anything else: weight-dependent,
report the curve; no single-arm claim. Ceiling: 7 runs x 2 allocations x 11.5 h.

## Decision rule for the phase

Steps 1, 2, 4 as predicted: the cycle needs a mode-covering distribution term and the reverse model can
stay; next design is that term inside the cycle. Step 4 collapsing at every weight: freeze or
bottleneck phi before touching the prior. Step 3 taking off: an unsupervised route, supersedes both.

## Results

(none yet)
