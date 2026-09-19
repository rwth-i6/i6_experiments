# SAE 4A attribution: why the GAN takes off and the cycle does not

Registered 2026-09-19 on the user's authorization ("implement and execute 1,2,3,4"; parallel where
possible). Diagnostic phase inside §4a. Nothing here is cold-start unsupervised progress; the GAN arms
are diagnostics of the objective, not mainline components (`SAE_ref.md` label quarantine and no-GAN
constraint stand). Gold enters evaluation only.

## State

Design review done and folded in (amendments below). Steps 2 and 3 (four arms: norev
`BoundedBlankfreeTrainingJob.QolqasLCAL94`, agg1 `.81Kc6iySxEBH`, agg10 `.bDYARBE8qXp6`, k64
`.3AwI6Poud7xt`) DONE; their managers and the priorshuf and ngram_priorshuf managers have exited.
The ONLY live manager is step 4's, pid 885934; re-arm its watcher first on resume (command below).
Step 1: DONE and audited (Results). Prior-window defect found and recorded (Results, `SAE_ref.md`);
priorshuf arm implemented (speech-llm commit 34ada2b, `config/sae_4a_attrib_priorshuf.py`,
`BoundedBlankfreeTrainingJob.gBec5S4Wa2F5`, prior `PhoneNgramPriorJob.RtzbESkOedsT` on
`SampleLinesJob.orN768ARKwlt`) DONE (Results: not the cause). Steps 2 and 3 DONE and audited
(Results). Only step 4 is live. Step 4: review clean
(`reports/sae_attrib_step4_review_2026-09-19.md`), profile read (amendment above), seven arms RUNNING
since ~13:40 at b=16 under manager pid 885934 (`config/sae_4a_attrib_ganrev.py`,
`log/sae_4a_attrib_ganrev.manager.log`): w0_s0 `FairseqW2vu2TrainJob.9HnmO6ULORKl` 1891056;
lam0.01_s0 `.reCovgFXvSDj` 1891050; lam0.01_s1 `.ZTklLDlrH8Vv` 1891058; lam0.1_s0 `.YM9FkZ2qzoVW`
1891057; lam0.1_s1 `.h7YWOSQjA7AH` 1891054; lam1.0_s0 `.oF22UaRGYy2k` 1891052; frozen_lam0.1_s0
`.mumOHh9l2vkK` 1891055. rqmt time 11.5 h (partition cap), so each arm spans two allocations via
checkpoint_last resume. Watcher: `bash ~/.claude/skills/sis/sis_watch.sh 885934 config/sae_4a_attrib_ganrev.py 600`.
priorshuf n-gram JSD read DONE (Results, `NgramModeSeekingJob.vlhnotaFKKj5`, manager exited).
NEXT (user 2026-09-19: progress without waiting for step 4; NO GAN component in the route, user
2026-09-19): design step 5 = the corpus-level coverage term as the MAIN distribution term (Results,
"Step 5 design"): lattice prior_weight 0 (the cycle keeps only the reconstruction tie), L_agg at
order 3 with the ratio gradient and lam_agg O(1), cold blankfree bed, 4 subepochs, control = the
reference `5lBwcDjv2ItL`; pre-register the take-off gate (ep4 dev-other greedy PER < 0.50, gap > 0,
rate in band) before the first job; implementer spec + tests, code-reviewer, design-reviewer, then
launch. Stage 2 (only after a take-off): phi fit on the step-5 posteriors, frozen, cycle + bt_a
(pack6 recipe). Then: on each step-4 watcher verdict dispatch executor; when all seven arms finish,
extractor for the step-4 PER table, gate read and synthesis.
Launch reports: `reports/sae_attrib_steps23_launch_2026-09-19.md`, `reports/sae_attrib_step4_launch_2026-09-19.md`.

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

## Design-review amendments (2026-09-19, `reports/sae_attrib_design_review_2026-09-19.md`; accepted before any launch)

Original step texts above stand as provenance; the following supersede them where they conflict.

- **Step 1.** The 0.27 threshold is Lin's gold-vs-text, corpus-size-dependent value; keep it as a
  descriptive reference only. Decisive comparisons become paired-difference bootstraps: (i) cold ep4
  mean SIL-free trigram log-prob >= GAN's minus 0.10; (ii) cold ep4 4-gram JSD minus gold's >= 0.05
  and minus GAN's >= 0.05, both difference CIs excluding zero; (iii) GAN minus gold within 0.10.
  All true reads "the cold output is prior-shaped despite high prior likelihood", a mode-seeking
  signature; it is consistent with the directional H1 and does not by itself establish it. The GAN
  row is decoded from the weighted_lm_ppl-selected checkpoint (seed 0, update 148000, `SAE_1c.md`),
  since `per.json` stores no strings.
- **Step 1 implementation notes (before any read).** Reader and job:
  `recipe/i6_experiments/users/wu/experiments/unsupervised_asr/ngram_mode_seeking.py`
  (`NgramModeSeekingJob`, Witten-Bell trigram fit by the job; config `config/sae_4a_attrib_ngram.py`).
  The reproduction's PER job stored no strings, so the GAN row is decoded from `checkpoint_best.pt`
  (= update 148000) by the existing `GanPseudoLabelJob` on the valid split, restricted to the
  2864 dev-other ids; its labels are SIL-stripped, so the SIL-inclusive secondary is n/a for the GAN
  row. Bootstrap convention fixed now: decisive comparisons are per-resample differences with
  percentile CIs; single-row JSDs render the plug-in estimate with a reverse-percentile CI, because
  resampling inflates the plug-in JSD. Known defect: `PhoneNgramPrior.per_token_log_probs`
  double-counts length-1 sequences (worked around in the reader; training-path use to be checked
  in code review).
- **Step 2.** Zeroing emission and duration together leaves an unnormalized segmentation count that
  favors maximal phone rate, a non-content confound. Amended delta: the emission score is zero, the
  duration model is frozen at its cold initialization (a proper distribution over legal d), phi gets
  no update. Phones per second is a required readout. Two additional single-delta arms on the
  unchanged blankfree control (reverse term present): lambda_agg = 1 and 10 (control 0.1). The
  aggregate term already matches expected run-unigram/bigram counts to text, i.e. it is the bed's
  existing mode-covering term; these arms are the cheapest falsifier of "add a mode-covering term".
  Prediction: neither beats 0.865 by more than 0.05 paired. Ceiling for step 2 becomes 3 GPU x 1 h.
- **Step 4.** tau is set to 1 (plain marginal likelihood) instead of 2: the tau=2 alignment sum is
  concave in q and pays for high-entropy posteriors, which the discriminator can exploit; tau=2 was
  the cold anneal endpoint, not a GAN constant. To separate "term present" from "phi updated", the
  lam_rev=1.0 seed-1 run is replaced by a frozen-phi arm at lam_rev=0.1 seed 0 (phi = the blankfree
  ep4 reverse model, `BoundedBlankfreeTrainingJob.5lBwcDjv2ItL`, no update). Controls are
  seed-matched (s0 0.214, s1 0.205) and every arm is read at its own weighted_lm_ppl-selected
  checkpoint exactly as the reproduction. "Harmless" requires an activity criterion: the arm's dev
  reverse log-likelihood per retained frame must exceed the cold phi's, or its own-minus-donor gap
  must be positive; a term that stays inactive at 0.01 says nothing. Cost: the trigram lattice
  measured 18.2 s per 128-utterance update; even at beta=0 the term may dominate the 0.24 s GAN
  update. The profile runs at b = 160 (all) and b = 16 utterances per generator update; the launch
  uses the largest b whose projected wall time x 1.25 is at most 20 h, and b is recorded as part of
  the delta. A 2 to 50x overrun is not cured by resumability.
  *Profile read and ceiling amendment (2026-09-19, `reports/sae_attrib_step4_profile2_2026-09-19.md`,
  jobs FairseqW2vu2ProfileJob.{oeUcv631Ief8,yTNMDgEA6aNF,1EXLBLMTUblA}, warm mean over updates 20-100):*
  sec/update lam0 0.258, lam1.0 b=16 0.439, lam1.0 b=160 0.890; peak GPU 6.8 / 8.6 / 17.6 GB. The
  term's cost is mostly a fixed per-update part (b=16 adds 0.18 s, b=160 adds 0.63 s), so at 150,000
  updates b=160 projects 37 h and b=16 18.3 h; x1.25 = 22.9 h, over the 20 h ceiling by 15 %, and no
  smaller b fixes that. Amendment: ceiling raised to 24 h projected, launch at **b = 16** (recorded as
  part of the delta), arms resumable from checkpoint_last.pt; a resumed arm is not bit-reproducible
  because the sub-batch RNG is not checkpointed (code review). Reverse loss at update 100: 6.31 (b=160)
  / 6.43 (b=16).
- **Decision rule.** A discriminator inside the cycle would breach the no-GAN mainline rule; that
  branch is a user decision, not an orchestrator next step. The "freeze phi" branch is void unless
  step 2's no-reverse arm beats 0.865 by more than 0.05 paired. Any outcome pattern not listed is
  reported without a mechanism claim.

## Decision rule for the phase

Steps 1, 2, 4 as predicted: the cycle needs a mode-covering distribution term and the reverse model can
stay; next design is that term inside the cycle. Step 4 collapsing at every weight: freeze or
bottleneck phi before touching the prior. Step 3 taking off: an unsupervised route, supersedes both.

## Results

### Step 1 (2026-09-19): mode-seeking signature NOT supported, cell (i) fails
`NgramModeSeekingJob.o2V5IRjMDy3K` (`output/sae/4a/attrib/{ngram_mode_seeking.json,summary.md}`),
dev-other, count-matched primary (budget 120,187 phones set by ep1), text side = the training prior's
window (SIL stripped), Witten-Bell trigram, 1000 utterance-block resamples. Audited from a fresh context,
every number reproduced to 6 decimals: `reports/sae_attrib_step1_audit_2026-09-19.md`
(CONFIRMED_WITH_CAVEATS).

| row | mean SIL-free log P3 / phone | JSD4 vs text |
|---|---|---|
| blankfree ep1 (PER 0.835) | -6.2027 | 0.9359 |
| blankfree ep4 (PER 0.865) | -3.3424 | 0.7134 |
| GAN s0 checkpoint_best = update 148000 | -3.0087 | 0.3021 |
| gold (MFA) | -2.8898 | 0.2749 |

Cells: (i) ep4 minus GAN log P3 = -0.334 [-0.347, -0.290], needed >= -0.10: FAIL. (ii) ep4 minus gold
JSD4 +0.439, ep4 minus GAN +0.411, CIs exclude 0: PASS. (iii) GAN minus gold +0.027: PASS.
Reading: the cold epoch-4 output is far from text at every order AND has lower trigram likelihood
than the GAN and than gold. The cycle is not sitting on a high-likelihood, low-coverage mode; it has
not reached the prior's high-likelihood region at all. H1 in the "likelihood is mode-seeking" form is
not supported by this read. Repeats/SIL cannot drive (i) (audit section 6). Unverified: the
"PER 0.214, ppl-selected" label of the GAN checkpoint (its curve job is gone; the checkpoint identity
itself is byte-verified).

### Step 2 (2026-09-19): removing the reverse term hurts; more aggregate weight does not rescue
Cold blankfree bed, 4 subepochs, ep4 dev-other greedy PER, paired vs the reference
`5lBwcDjv2ItL` ep4 (0.8649, AH-first 0.581, 59.0 phones/utt). Reads: `reports/sae_attrib_norev_read_2026-09-19.md`,
`reports/sae_attrib_agg_read_2026-09-19.md`, first-phone shares computed identically for all four
decodes in `reports/sae_attrib_firstphone_2026-09-19.md`. Audited with step 3 from a fresh context
(`reports/sae_attrib_steps23_audit_2026-09-19.md`, CONFIRMED_WITH_CAVEATS): PERs, S/D/I and the paired
deltas with speaker-clustered CIs reproduce exactly; config diffs vs the reference show only the
intended delta per arm; norev's reverse tensors are bit-identical ep1 vs ep4. Caveats: norev zeroes
the reverse EMISSION term and keeps the cold duration model, so its cell is about that term; all
arms are failing decodes, so the cells license "not funding", never "would not have worked"; the
shared prior is common-mode within the bed.

| arm (training job) | ep1 PER | ep4 PER | paired delta [CI] | ep4 gap | first phone | phones/utt |
|---|---|---|---|---|---|---|
| norev `QolqasLCAL94` (emission 0, phi frozen) | 0.8505 | 0.9357 | +0.071 [+0.056, +0.088] | -0.026 | AH 0.950 | 68.5 |
| agg1 `81Kc6iySxEBH` (lambda_agg 1) | 0.8213 | 0.9202 | +0.055 [+0.051, +0.060] | 3.44 | AH 0.919 | 61.0 |
| agg10 `bDYARBE8qXp6` (lambda_agg 10) | 0.8582 | 0.8986 | +0.034 [+0.027, +0.040] | 3.39 | AE 0.778, AH 0.004 | 58.9 |

Against the pre-registered rules: norev is worse by > 0.05, so the cold reverse term carries content
the trigram marginal alone lacks (it is not a passive absorber; H2 in the "reverse degrades" form is
not supported on this bed). Neither agg arm beats the reference, both regress, so a stronger
mode-covering unigram/bigram term does not rescue the cycle here (H1 in the "needs a mode-covering
term" form is not supported either, with the caveat that the agg target is derived from the biased
prior below). The sentence-initial pattern tracks the prior window in every arm: AH dominates where
the trigram rules, and lambda_agg 10 flips it to AE, the window's second-most-frequent initial,
without improving PER. So the initial-phone collapse is a symptom of the prior, and the PER
collapse persists once it is removed.

### Step 3 (2026-09-19): MFCC K=64 reverse target does not take off
`BoundedBlankfreeTrainingJob.3AwI6Poud7xt` (reverse observation = MFCC K=64 codes, everything else
the reference bed), read `reports/sae_attrib_k64_read_2026-09-19.md`. ep1 dev-other PER 0.8370, ep4
0.9147 (dev-clean 0.8952); paired vs reference ep4 +0.050 [+0.039, +0.061], improved 545 / worse 2011;
ep4 gap +0.75 (own -3.67 vs deranged -4.43 per frame, 500 utts); first phone AH 0.870; 61.6 phones/utt.
Prediction "PER > 0.80" held; the route criterion (PER < 0.50 with a positive gap) is not met, so the
K=64 target is not extended. The positive gap says the K=64 reverse model does learn some
utterance-specific structure, yet the recognizer still collapses, which separates "reverse target
informativeness" from "recognizer takes off" on this bed.

**Cross-arm observation (descriptive, all five cold arms):** every arm gets WORSE from epoch 1 to
epoch 4 (reference 0.835 to 0.865, norev 0.850 to 0.936, agg1 0.821 to 0.920, agg10 0.858 to 0.899,
k64 0.837 to 0.915) while its training losses fall. Whatever is optimized is anti-correlated with
PER on this bed after the first subepoch. This is the pattern the priorshuf arm tests first.

### Prior-window defect (2026-09-19, surfaced by the step 1 audit, verified by count)
The training trigram's text window is the first 1.01 M lines of an alphabetically sorted corpus:
73.5 % of its sentences start with AH, 20.7 % with AE, 5.0 % with AA. Details and consequences in
`SAE_ref.md` ("Training-prior text window is alphabetically biased"). It offers a direct, prior-side
explanation of the AH-first collapse (58.1 % at ep4) that none of steps 1-4 tests, and it confounds
every cycle-vs-GAN comparison (the GAN's text data are the full corpus). Added arm, pre-registered:
**priorshuf** = the cold blankfree reference (`5lBwcDjv2ItL`, 4 subepochs) with the only delta a
trigram refit on a seeded uniform-random 1,010,000-line sample of the same corpus (same recipe,
same defaults). Prediction if the window is a main cause: ep4 AH-first share drops below 20 % and
dev-other PER improves on 0.865 by > 0.05 paired; if PER stays within +-0.03 and AH-first stays
above 40 %, the window is not the cause of the collapse (only of its AH flavour). Steps 2-4 keep
running: their reads are within-bed and stay valid, but the H1/H2 verdicts are provisional until
priorshuf is read, and step 4's GAN corners are compared to the cycle only through PER.

**priorshuf result (2026-09-19, `reports/sae_attrib_priorshuf_read_2026-09-19.md`):**
`BoundedBlankfreeTrainingJob.gBec5S4Wa2F5`, prior `PhoneNgramPriorJob.RtzbESkOedsT` on
`SampleLinesJob.orN768ARKwlt` (1,010,000 uniform lines, seed 0; initial phones DH 0.167, HH 0.130,
IH 0.083, AY 0.083, AE 0.062; held-out trigram ppl 9.56 vs the biased window's 9.47). ep1 dev-other
PER 0.8206 (best ep1 of any cold arm), ep4 0.8829 (dev-clean 0.8653); paired vs reference ep4
+0.018 [+0.015, +0.022] (improved 837 / worse 1610); gap 2.44; first phone AH 0.003, AE 0.001,
**HH 0.886**, W 0.066; 60.2 phones/utt, 39 distinct. Reading against the pre-registration: PER is
inside the +-0.03 band (slightly worse), so the biased window is NOT the cause of the PER collapse.
The AH-first share did drop below 20 %, but only because the collapse moved to HH, a frequent
initial of the new window: a single-phone sentence start is a property of the objective on this bed,
the prior window only chooses which phone. The defect stays recorded for absolute statistics and
GAN comparisons; it is closed as a cause. Read through the same registered chain the steps 2/3
audit reproduced; not separately audited.

**priorshuf n-gram read (2026-09-19, `NgramModeSeekingJob.vlhnotaFKKj5`,
`config/sae_4a_attrib_ngram_priorshuf.py`, `output/sae/4a/attrib/{ngram_mode_seeking_priorshuf.json,summary_priorshuf.md}`,
report `reports/sae_attrib_priorshuf_jsd_launch_2026-09-19.md`):** the step-1 reader with the text
side moved to the priorshuf window (`SampleLinesJob.orN768ARKwlt`, SIL stripped) and its trigram
(`RtzbESkOedsT`); same budget rule, 1000 utterance-block resamples, seed 0. Not audited.

| row | mean SIL-free log P3 / phone | JSD1 | JSD2 | JSD3 | JSD4 vs unbiased text |
|---|---|---|---|---|---|
| blankfree ep1 | -5.919 | 0.218 | 0.565 | 0.799 | 0.937 |
| blankfree ep4 | -3.289 | 0.067 | 0.280 | 0.495 | 0.711 |
| priorshuf ep1 | -6.069 | 0.226 | 0.593 | 0.819 | 0.946 |
| priorshuf ep4 | -3.611 | 0.026 | 0.214 | 0.443 | 0.692 |
| GAN s0 (update 148000) | -2.727 | 0.002 | 0.019 | 0.078 | 0.254 |
| gold (MFA) | -2.534 | 0.002 | 0.016 | 0.067 | 0.225 |

Cells (priorshuf ep4 as "ep4"): (i) ep4 minus GAN log P3 = -0.886 [-0.908, -0.858]: FAIL; (ii) ep4
minus gold JSD4 +0.444, CI excludes 0: PASS; (iii) GAN minus gold +0.029: PASS. Reading: the unbiased
prior closes the unigram gap (JSD1 0.067 to 0.026, the L_agg target is now unbiased) and leaves
orders 3-4 where the biased window left them (JSD4 0.711 to 0.692, gold 0.225); the decoded
trigram likelihood is LOWER under the unbiased prior (-3.61 vs -3.29). The training statistic
`prior_per_token` (posterior-expected, SIL-inclusive, lattice tokens) reaches -2.55 at ep4 in the
same run (`output/.../priorshuf/train/learning_rates`), a full nat above what the greedy decode
scores on the same trigram, and the priorshuf and reference loss curves are identical to two
decimals at every subepoch. The prior term is satisfied inside the lattice posterior, not on the
emitted string, and it moves the marginals, not the sequence structure JSD3/4 measures.

### Step 5 design (2026-09-19, pre-registration pending, no job yet)
Literature (`reports/lit_odm_direction_2026-09-19.md`): Liu, Chen, Deng (NeurIPS 2017, sec. 2.3)
prove that the label-space cost "LM score of the model's own output" is mode-seeking and converges
to the majority guess, while the forward cross-entropy sum_w p_LM(w) log p_out(w) between the LM
n-gram distribution and the model's CORPUS-LEVEL expected output n-gram frequencies is
coverage-seeking and trains (OCR 9.6 % vs 83 %); the estimator matters as much as the direction
(SGD at batch 10k 56 % vs their per-n-gram dual 9.6 %). Yeh et al. (ICLR 2019) train a frame-local
classifier (11-frame splice) with that forward cost at order 5 and reach TIMIT 42.6 PER fully
unsupervised (36.5 with HMM self-training; LM-decoded, not comparable to greedy dev-other). No
published work pairs such a seed with a cycle or back-translation refinement. Reading against this
phase: once phi absorbs (cold fixed point), the only pressure left on q is the lattice prior term,
which is Liu's mode-seeking label-space cost; the sentence-initial single phone is its majority-guess
signature. BT is the conditional coverage side and needs a content-carrying phi (pack6, P-BT);
L_agg is the marginal coverage side and needs none, but it is currently order <= 2, priced 0.1, and
its gradient is damped by (1 - count_ema_decay) = 0.01 through the EMA blend (`agg.py:196`), so
agg10 was an effective 0.1 at order 2 under a dominant mode-seeking term.
Design: lattice prior_weight 0 (L_tau = reconstruction tie only); L_agg order 3 by the exact
"last two non-blank phones" recursion (K^2 states per frame); loss = -sum_w p_text(w) c_batch(w) /
c_EMA(w).detach() (ratio gradient, EMA only in the denominator), value reported as the forward CE on
the EMA; lam_agg O(1), matched to the removed term's gradient norm at init; recognizer unchanged
(1-layer conv, kernel 9, stride 3: the frame-local class Yeh's result needs). Arms: odm3 (as above);
odm3_norev (reverse emission 0, the pure Yeh objective on this bed); control = reference ep4 0.865.
Gate (take-off, pre-registered before launch): ep4 dev-other greedy PER < 0.50 AND positive gap
AND rate in [0.6, 1.5] x rho, paired vs the reference; stage 2 funded only on a take-off.

### Interim synthesis after steps 1-3 and priorshuf (step 4 pending, ~18 h)
Within the cold blankfree bed nothing that changes the distribution term rescues it: the trigram
marginal alone (norev) is worst; unigram/bigram matching at 10x weight, a K=64 MFCC reverse target
and an unbiased prior each leave ep4 PER at 0.88-0.92; every arm degrades from ep1 to ep4 while its
losses fall. The cold reverse term is the only tested component whose removal makes things clearly
worse. Step 1 shows the collapsed output is not a high-likelihood mode of the prior either. So the
failure is not "the trigram is too weak a matcher" in the sense of missing modes, and not "the
reverse model degrades the recognizer"; the objective as a whole has descent directions that lower
PER-relevant content while raising likelihood (the sentence-initial single phone is one visible
instance). Step 4 tells whether the same reverse term inside the GAN objective is harmless, which
would locate the fault in the alignment-sum-times-prior term and make a sequence-level
mode-covering term inside the cycle the next design.
