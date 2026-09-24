# SAE — the reward's lam_1 (LM prior) and lam_2 (KL anchor)

**User amendment, 2026-09-18, after M512 output inspection:** main direction is a new cycle model
with unsupervised silence removal, wav2vec-U 2.0 CNN stride, no CTC blank and adjacent repeat collapse.
Evaluate, implement and execute autonomously; verify the cited paper's actual preprocessing rather
than assume it removes SIL. This authorizes a new bounded round, preserving label quarantine and
historical gates. Diversity within M512 sampled alignment groups is a parallel side task. Live
protocol, resource limits and new-model results: `SAE_4A_blankfree.md`.
**Follow-up instruction:** run both independent 10 h supervised initialization and cold unsupervised
trigram joint training with this new topology. The supervised branch is disclosed and separate;
it supplies no parameters or labels to the cold branch.
The user explicitly chooses separate initialization only for the supervised branch, with no
100 h joint adaptation after those fits.
**Training budget for new arms (user item 1, 2026-09-20): N = 20 sub-epochs** with the budget
round's proportional schedule (anneal 4 sub-epochs 8 -> 2, LR warmup 2, hold to 12, decay to 20,
kept checkpoints 1, 4, 10, 20); decided from label-free curves only, evidence and table in
`SAE_4A_budget.md` "Sub-epoch count for future arms". Each new pack carries its own ctrl_20.
**Reference preprocessing verified from the paper's full text (2026-09-20,
`SAE/reports/lit_w2vu2_preprocessing_2026-09-20.md`):** wav2vec-U 2.0 (arXiv 2204.02492v2 §4.1)
removes audio silence with rVAD before extracting layer-15 wav2vec 2.0 Large features and
inserts SIL at word boundaries with probability 0.5 plus sentence-edge SIL; its "no audio-side
pre-processing" covers segmentation, k-means, PCA and pooling only. The bed's rVAD + sil_prob 0.5
is therefore the reference combination. Differences from the reference that stand disclosed:
features masked after full-waveform SSL extraction (paper cuts the waveform first; unmeasured
anywhere); no batch-norm/residual generator, no auxiliary MFCC k-means head, no GAN. Reference
checkpoint selection is label-free: 4-gram phone-LM perplexity divided by the squared fraction of
vocabulary seen, SIL stripped before both. Gold silence on dev-other after the bed's rVAD: 7.9 %
of retained frames, 5.7 % of phone runs, against 13.8 % SIL tokens in the prior text
(`SAE_4A_prior.md`, Bed).

## Current research constraints (user priority 2026-09-16)

The objective is pure unsupervised ASR without GANs, with cold-start improvement as the active research
priority, **within phase §4a cycle consistency**. Unpaired speech and text and the existing speech-only SSL
features remain the mainline inputs. User clarification: improve the cycle itself; a standalone SylCipher
initializer or reproduction is outside the requested direction and its proposal is withdrawn. Unsupervised
machine translation is a source of ideas for the cycle objective and training, not a replacement pipeline.
No GAN-derived or paired-transcript-derived recognizer, reverse model, pseudo-labels or teacher may supply
the mainline initialization or training signal. Existing phonemized text is a text-side resource, not paired
speech supervision. True transcripts and forced alignments remain quarantined to evaluation and disclosed
diagnostics; they cannot select mainline checkpoints or tune the new initialization.

Supervised controls, including S2d and the gold-derived S3b-OR checkpoint swaps, explain failure modes but
do not count as unsupervised progress. Optimizing supervised-initialized performance is not the focus.
The proposed additional seeded-S2d frozen-reverse run is withdrawn. Earlier GAN-fallback and GAN-init
carve-outs are historical provenance, not authorization for new mainline work. The user explicitly invites
methodological thinking and literature research on making this cycle learn from cold start. The current experimental
bed, original gates and historical results remain in `SAE_4A.md`; this priority change does not rewrite them.

**User amendment, 2026-09-17:** the user reopens supervised-initialized refinement as an important
parallel analysis and authorizes autonomous error-pattern/mechanism analysis followed by a new bounded
training round aimed at improving the 10 h supervised initializer on the 100 h adaptation bed. This
supersedes the withdrawal above for this disclosed seeded track. The inherited setting is 10 h labeled
initialization followed by 100 h speech-only cycle adaptation, not an additional 100 h of paired labels.
Gold may explain errors and assess results; it cannot select checkpoints or enter adaptation targets.
Register the exact follow-up, controls and resource limit before launch; retain G4a.2 and G4a.S2d.
These results do not count as cold-start unsupervised progress. The six-gram cold run is separate;
§4b is left unchanged at the user's request. Live protocol: `SAE_4A.md`, "Reopened seeded refinement".

**User amendment, 2026-09-18:** priority 1 is making six-gram cycle training affordable; the user
explicitly permits reducing the phoneme-string or alignment search space. This supersedes the
previous prohibition on reducing the 256-draw space, without changing the cold-start ASR gate or
label quarantine. First screen a smaller candidate budget on representative actual batches and
both initial/trained model states; do not repeat the first-batch whole-run extrapolation. Priority 2
is improving the 10 h initializer with a trainable reverse model and an LM ablation. The registered
follow-up separates removing the sequence-LM factor from removing both it and the text-derived
aggregate regularizer. Shared warm initialization and the evaluation decoder retain their disclosed
LM history/use; this is not an LM-never-used claim. Live cost and seeded protocols are in `SAE_4A.md`.

**User clarification, 2026-09-18 (M512 output inspection):** no WER work is needed for the current
read. Inspect actual phone-output collapse patterns and diversity, using saved hypotheses and
evaluation-only references. The original cold gate is retained. Word decoding is disabled in the
M512 recipe; prior status wording that WER was pending was incorrect. Live diagnostic scope is
in `SAE_4A.md`, "Output-collapse inspection".

**User amendment, independent supervised initialization (2026-09-18):** add a baseline that fits
the reverse model independently on the same labeled 10 h used for the recognizer. Condition on
the gold phone sequences and observed speech units, marginalizing segment durations; do not use
recognizer predictions or additional labeled utterances/boundaries. Retain the original recognizer
checkpoint, then change only the reverse initializer in the matched 100 h joint-adaptation arm.
This explicitly authorizes gold phone sequences for reverse initialization; the 100 h adaptation
remains speech-only. Existing S2f LM ablations and the separate alignment-budget screen continue.

**User proposals, seeded refinement (later 2026-09-18):** reopen back-translation on the supervised
initializer and consider subepoch-level alternating ASR/reverse updates, including reciprocal
training with the other model frozen. Keep the LM-prior ablation independent of these proposals.
The user clarifies single-pass odd/even subepoch alternation, without replaying batches; the exact
schedule is specified in `SAE_4A.md`.
These additions concern the seeded track; prior cold and seeded BT results and the M512 protocol remain
unchanged. Define separate schedule and BT comparisons before combining them; freezing has not
been established as mandatory. Follow-up specifications and release conditions are in `SAE_4A.md`.

**User supersession, later 2026-09-18:** the user vetoes the K4/few-string direction because it
defeats the intended exploration. Sample complete alignments instead, starting at 512 and choosing
as many as measured throughput permits under **24 h wall clock for the complete new training run**.
Directly rescore and train over those paths, removing per-string conditional inference. This is
an explicitly authorized approximation change, not an exact full-space sum. The 24 h prospective
training cap supersedes the old 8 h release cap; earlier costs/gate failures remain historical facts.
Register the path estimator, numerical checks, cost-selection rule and bounded profiling allocation
before launch. Preserve random real batches, the cold ASR gate, and the separate S2f/§4b scopes.

**User selection, later 2026-09-18:** start the matched complete-path experiment at **512 paths**
per utterance. Leave the adaptive benchmark running unchanged; its search for larger counts no
longer determines this run's draw count or launch time. The numerical/timing release gate and
24 h complete-training ceiling still apply. A passing cost screen is not evidence of ASR benefit.

**Testing clarification, 2026-09-18:** the user requires randomly sampled training batches because
`laplace:1000` sequence sorting makes early-batch timing optimistic. Future cost screens sample real
batches across the registered training schedule with a fixed recorded seed; retain a long-batch stress
case separately. First-batch or length-quantile measurements alone cannot support a whole-run estimate.

Separately, the user authorizes the frozen higher-context diagnostic and then direct six-gram candidate
reweighting training in §4a, independently of S3d. On 2026-09-17 the user additionally authorizes starting
`SAE_4B.md`: reproduce a sparse autoencoder on frozen w2v2 and extract phoneme information with supervised
diagnostics following AudioSAE. This supersedes §4b's earlier planning-only restriction. The SAE itself
uses speech alone; labeled diagnostic fitting and held-out evaluation are authorized, with their artifacts
quarantined from unsupervised training and checkpoint/hyperparameter selection. A supervised phone probe
does not establish unsupervised phone discovery. §4b does not yet authorize a cycle-target substitution.
Its audit inputs, matched baselines, fixed operating point and evidence criteria are registered in
`SAE_4B.md`, "First run: fixed operating point and readout".
The user subsequently requests the regularization from *Scaling Monosemanticity* (2024), authorizing
a separate decoder-norm-weighted L1 SAE arm. Its normalization and objective must follow the cited
method, with wav2vec2/data/budget adaptations disclosed. Labels remain diagnostic only; the completed
BatchTopK arm is retained. The follow-up protocol and comparison scope are registered in `SAE_4B.md`.

For cold-start comparison, use the registered `lam3_tri` reference in
`PackedEmcTrainJob.byYMQmBNEpLZ` (the tc100/trigram run), with its fixed endpoint and label-free-selected
checkpoint reported separately. Its concrete inputs and the prior failed initializers are extracted in
`reports/codex_cold_prior_inventory_2026-09-16.md`; full-split scoring uses the same `GoldPhonesJob.ZGSp0hxyd2YP`
references, greedy SIL removal and speaker-clustered comparisons as the existing cold results. New tokenizations
must identify their native metric and cannot equate CER with this PER. Gold speech/text pairs are opened only
after the new model, checkpoint and decoding rule are fixed. Original cold take-off gates remain in `SAE_4A.md`.

**Hypothesis-diagnostic qualification (2026-09-17):** `analysis/emc_hyp_inspect.py` fits a Hungarian
phone map to the existing edit-alignment confusion matrix and evaluates it on the same gold. This is
not a globally minimum-PER permutation after realignment; small gains cannot exclude other mappings
or latent phonetic information. Its arm-specific length/unigram-matched nulls are descriptive, and
five-draw SDs are not paired speaker confidence intervals. Audit and the narrowed earlier interpretation:
`reports/codex_4a_s3c_hyp_inspect_audit_2026-09-17.md`, `SAE_4A.md` S3c result.

The frozen-checkpoint recognizer-factor diagnostic requires valid posteriors and matched DP precision
in both arms. Its numerical validity defect, correction and engineering checks are specified in
`SAE_4A.md`, "Recognizer-factor diagnostic"; the banked float32 profile is provenance, not a substitute
for the recomputed control.

**Cold-initial numerical validity (2026-09-17):** the canonical float32 lattice also violates its existing
posterior-conservation tolerance at the actual flat initialization, in the clean/eval epoch-1 read.
Promoting identical DP inputs to float64 passes that read. The measured conditions, gate failure and audit
are in `SAE_4A.md`, "Cold-initialization numerical result". This does not establish the cause of banked
PER results. The subsequent canonical first-batch loss/backward check passes; see the audited
"DP-only training-step result" in `SAE_4A.md` for its limited operating point. The matched full training
control is now complete; its operating point, endpoint comparison and limitations are under "Precision-only
control result". Any content-treatment comparison must use a control with the same numerical computation.

**Training-prior text window is alphabetically biased (2026-09-19, verified by count):** the blankfree
trigram prior `PhoneNgramPriorJob.TRPE0D5nF3bh` (defaults `n_count_lines` 1,000,000, every 101st line
held out) is fit on the FIRST 1,010,000 lines of `PhonemizeWithSilJob.DbFgvZOGZQ8F/output/text.phn.gz`
(39,630,169 lines of librispeech-lm-norm.txt, which is alphabetically sorted). Sentence-initial phone in
that window: AH 742,873 / AE 208,622 / AA 50,473 / EY 6,578 / EH 1,257 (all sentences start with
"a ..."); a slice at line 2,000,000 is 100 % AE-initial. The prior conditions on a BOS context
(`sae/emc/prior.py:69,271`), so P(AH | BOS SIL) is about 0.74 under the training prior. Every cycle arm
on this bed (the cold blankfree reference `BoundedBlankfreeTrainingJob.5lBwcDjv2ItL`, 58.1 % AH-first at
epoch 4, and the 4A attribution arms) shares this prior; within-bed comparisons stay valid, but the
prior-shaped absolute numbers (step 1 JSDs, sentence-initial statistics) and any comparison against the
GAN, whose fairseq text data binarize the full 39.6 M-line file, carry this confound. Corrected prior
available: `PhoneNgramPriorJob.RtzbESkOedsT` on a seeded uniform 1,010,000-line sample
(`SampleLinesJob.orN768ARKwlt`, held-out trigram ppl 9.56). The cold rerun with it (`SAE_4A_attrib.md`,
priorshuf) lands at ep4 PER 0.883 vs 0.865 with the sentence-initial collapse moved from AH to HH, so
the window is not the cause of the collapse; new cycle beds should still use the sampled prior.
Surfaced by the step 1 audit `reports/sae_attrib_step1_audit_2026-09-19.md`.
**Standing decision (user, 2026-09-19):** every new experiment that uses an n-gram (prior, coverage
target, text-side statistic, selection LM) uses the unbiased uniform-sample fit
(`SampleLinesJob.orN768ARKwlt` -> `PhoneNgramPriorJob.RtzbESkOedsT`, or the same seeded uniform
sampling at another size), never the alphabetical head window. The only arms still on the biased
window are the pre-defect steps 2-3 of `SAE_4A_attrib.md` (norev, k64, agg1, agg10); their reads
are within-bed only and are not compared in absolute terms against priorshuf-bed arms.

**Lattice semantics for sequence-prior extensions:** a complete latent path includes frame labels,
token-emission choices and reverse segmentation. Existing SIL transitions permit either a repeat or
a new adjacent SIL token, so the shorthand `B(pi)` in earlier formulas is not strict standard CTC
collapse for SIL. The timing band couples CTC and reverse states, and temperature applies to joint
assignments before summation. A fixed-string marginal must preserve those rules; multiplying an
ordinary CTC likelihood by the existing reverse-only marginal is not equivalent. Source trace and
available/missing inference components: `reports/codex_4a_context_rescoring_code_2026-09-17.md`.

## Approach

**1. Wire the two side-inputs the live train step never passed.** `compose_reward` has implemented
both terms since the reward was written, but each `_require`s a rollout-dependent side input —
`base_logprob_sum`/`n_text_tokens` for lam_1, `kl` from `ref_logprobs` for lam_2 — so setting either
raised by construction and every run to date had been recon-only. Neither can be precomputed (both are
functions of the rollout), so they enter the trainer as callables on `SampledText`, the same shape as
the existing `reward_fn` hook. lam_1's `p_base` is the AV's own decoder with LoRA-A switched off and
the audio prefix dropped, so "adapters off" *is* the frozen donor LM, teacher-forced identically to
`SpeechLmAvPolicy.logprobs`; lam_2's reference is a second fully frozen `SpeechLmV2` loaded from the
same checkpoint the live policy starts at, with every param frozen in *both* models shared, so the
snapshot costs one extra adapter + LoRA-A rather than a second 4 GB model. Non-trivial choices: the
snapshot is held in `eval()` mode whatever RETURNN does (LoRA dropout 0.1 — a reference that redraws
its own mask every step is not a reference); `ref_av` is built only when `lam_kl != 0` and each
side-input callable is invoked only when its lambda is nonzero, so at lambda=0 the module tree, the
state_dict and the resume path are untouched for the four live loops that re-import this source on
every resubmit; and each term gets a `reward/<term>_std_within_group` monitor, because advantages are
group-normalized and `lam * std(term) / std(recon)` is the mixing ratio a lambda actually sets.

**2. Smoke run on the validated 10 h bed, 1 GPU, 1 epoch, audio <= 8 s, two arms.** `recon` (`{}`)
against `anchors` (lam_1 = lam_2 = 0.02, placeholders chosen to be small and nonzero, not
recommendations). The control is not decoration — it makes "the anchors changed nothing they
shouldn't" checkable on the same bed.

| check | result |
|---|---|
| `reward_kl` ~ 0 at step 0 | 0.010, then 7.2e-4, -1.4e-4, 0.002, ... — sign-varying, 1e-3..1e-2 nats/token, exactly the LoRA-dropout residual predicted; a stub would give exactly 0.0 |
| `reward_lm_prior` in a plausible range | -6.3 .. -9.1 nats/token, well clear of -11.93 = log 1/151646; tracks length cleanly (text_len 8.5 -> -6.27, 6.58 -> -8.20) |
| mixing ratios, first 10 steps | std(recon) 0.071; lm_prior 0.462 -> **13 %** at 0.02; kl 0.017 -> **0.5 %** at 0.02 |
| full epoch (203 steps), bucket means | kl share 0.5 % -> 4 % as the policy moves; lam_1 share rises to ~22 % mid-epoch as std(recon) shrinks 0.071 -> 0.028 |
| control tracking + cost | `reward_recon` identical to 3 decimals at steps 0-1, separating at 1e-3 from step 2; GPU 10.7 vs 10.0 GB (the tying worked); +11.0 % s/step on cumulative `elapsed` |

**3. The full-length 4-epoch anchored pair on the validated 10 h bed** (user call: lam_1 = 0.01,
lam_2 = 0.02 before any §1e loop leans on the anchors). Frozen AR, per-epoch dev recogs. The control
is **not re-run**: `baseline()` gained `reward_kwargs`/`freeze_ar`/`eval_subset` whose defaults
reproduce the finished recon-only run byte-for-byte, proven against disk, so the anchored arm is the
same bed with exactly one variable changed.

| | anchored lam_1 .01 / lam_2 .02 | control (recon-only) |
|---|---|---|
| theta_0 | 16.9 / 20.7 | 16.9 / 20.7 |
| ep1 | 16.2 / 21.3 | 13.07 / 15.89 |
| ep2 | **12.91** / 16.49 | 13.87 / 16.51 |
| ep3 | 13.91 / 18.33 | **12.99** / 16.20 |
| ep4 | 13.99 / 17.90 | 14.47 / 17.09 |
| dev-clean mean (excl ep1) | **13.60** | 13.78 |
| dev-other mean (excl ep1) | 17.57 | **16.60** |
| ep3 -> ep4 delta | **+0.08 / -0.43** | +1.48 / +0.89 |

Epoch-1 in-run monitors on the real bed (steps >= 100, n = 797 rank-lines): recon -5.7635 with
within-group sd 0.0131; lm_prior -5.008 with sd 0.2236 = **17.1 %** at lam_1 = 0.01; kl 0.01417 with
sd 0.02181 = **3.3 %** at lam_2 = 0.02. Over the epoch in 200-step buckets, lam_1's share rises 11 %
-> 17 % and then sits at 16-17 % while lam_2's grows 1.2 % -> 13 % as KL rises 0.006 -> 0.09
nats/token and plateaus.

**4. Resume test (theta_0 across a restart), 2 epochs on the smoke bed.** Two failure modes, only one
of them loud: a hard key error if the checkpoint's 115 untied `ref_av.*` records cannot be mapped, and
a **silent** one where the restart re-snapshots theta_0 from the *current* policy, making lam_2 a
no-op from that point while every monitor keeps printing a plausible number. `reward_kl` discriminates
them, which is why this is worth a GPU rather than another audit.

## Conclusion

1. (2) All four smoke checks pass; the implementation is verified end to end, with the frozen snapshot
   demonstrably forwarding real weights rather than a stub.
2. (2) lam_1 carries a built-in **length coupling** — rollouts scored from BOS with no prompt context
   amortize the high-entropy first token over few tokens, so shorter rollouts get a worse per-token
   prior and `lam_lm * prior` rewards longer text. Within a group lengths differ, so it reaches the
   advantages. Plausibly a feature (it pushes against degenerate short codes) but it is a coupling, not
   a pure fluency term, and must be named as such wherever lam_1 is set for real. Measured at +0.26
   tokens (+1.4 %) mid-epoch against the control on identical batches.
3. (2 vs 3) **The smoke's +11 % step-time cost does not transfer**: on the full bed it is +0.6 %
   (2.320 vs 2.307 s/step over steps 50-200), because the encoder and policy forwards dominate. The
   +11 % was real but measured where the two extra forwards are a large slice of a small step. Only
   startup regresses, by +45 s, which is the `ref_av` load.
4. (3) **lam calibration numbers do not transfer across beds.** lam_1 = 0.01 was picked for a ~10 %
   mixing ratio and lands at 17 %, not because the prior changed (its spread only halved) but because
   the *recon* spread collapsed 5x — within a group of 12 the AR returns nearly the same recon to
   every member, so any side term of fixed size buys a larger share of the advantage than an easier
   bed suggests. Read the ratios on the bed that will be run.
5. (3) **dev-other is consistently worse with the anchors** — every epoch, mean 17.57 vs 16.60, never
   crossing, so unlike dev-clean this is not oscillation. dev-clean is a wash (best 12.91 vs 12.99).
6. (3) lam_2's late-regression claim is supported but not established: the control falls off at ep4
   (+1.48/+0.89) while the anchored arm does not (+0.08/-0.43) and ends better on dev-clean; the 1.4
   abs delta-of-deltas is the largest effect in the table, on n=1 transition per arm.
7. (3) **Verdict.** For §1e the trade is acceptable — lam_2 is mandatory there because nothing pins
   the text side, the anti-collusion guarantee is its purpose, and it buys late-training stability.
   For §2S, where the seed already pins the text, the anchors are not worth the dev-other cost: keep
   those loops recon-only.
8. (3) **The methodological result, and the more valuable one.** Four per-epoch readings were taken
   during this run and three were wrong or premature — ep1 read as a 5.4-abs catastrophe, ep2 as an
   overtake, ep3 as unresolvable anti-phase — and none survived the next data point. The epoch-level
   noise floor on this bed is ~1 abs, the same size as the effect being looked for. Consequence:
   **lam_1-only / lam_2-only arms at this budget would be unreadable.** Attribution needs multiple
   seeds per arm, a lambda deliberately large enough to clear 1 abs, or a bigger bed; do not spend GPU
   on a 1-seed 4-epoch single-anchor pair. Related discipline: per-step reward stats on 2 groups of 12
   are far too noisy to calibrate against — a single step-800 line read as a runaway 50 % lam_1 share
   where the bucket mean was 16 %.
9. (4) **Resume PASS, exactly.** Pre- and post-restart epoch-2 statistics are bit-identical
   (kl 0.0833, recon -5.5061, lm_prior -6.462 over n=46), so theta_0 was restored rather than
   re-snapshotted; the bit-identical reproduction additionally rules out optimizer-state and data-order
   drift across the restart. Operational note: the retry **overwrites `log.run.1`**, so the
   pre-restart baseline must be captured before the `scancel`.

## Catalog

`T/` = `work/i6_core/returnn/training/`.

| artifact | path |
|---|---|
| implementation | `sae/grpo/anchors.py` (new), `sae/grpo/trainer.py`, `definitions/sae_grpo.py`, `train_steps/sae_grpo.py` |
| tests | `sae/grpo/test_anchors.py` (10 new) + `test_trainer` (+4); 76 CPU tests pass |
| smoke arms | `T/ReturnnTrainingJob.QyeYIctvpywK` (anchors), `.Vj2m15i9zqxj` (recon control) |
| **full-length anchored arm** | `T/ReturnnTrainingJob.iNZtd4CRrSLR` |
| control (finished, byte-identical rebuild) | `T/ReturnnTrainingJob.qmkzvAX3gOVW` |
| resume test | `T/ReturnnTrainingJob.1hp40l6O9rga` |

Two tests carry most of the weight:
`test_base_lm_on_a_real_peft_wrapped_qwen3_is_the_pre_lora_donor` (a real `Qwen3DecoderV1` with a real
PEFT LoRA whose `lora_B` is randomized first, otherwise "adapters off" passes vacuously) and
`test_kl_vanishes_when_the_reference_is_the_sampling_policy`. `adapters_disabled` now **raises** if a
decoder carries LoRA params but has no `disable_adapter`, because otherwise `p_base` would quietly be
the *adapted* LM and lam_1 would degenerate into "reward the policy for being confident in its own
output" — an entropy-collapse driver no monitor would distinguish from a working prior.

Known bias, quantified and accepted rather than fixed: the k1 estimator differences two *different*
forward paths (`sampled.logprobs` from the KV-cached incremental sampler against `ref_logprobs` from
teacher-forced `decode_seq`), which do not agree exactly under bf16 autocast even at theta = theta_0,
so `reward_kl` carries a ~0.01 nats/token path-mismatch offset against real drift reaching 0.03-0.10
by step 150 — ~10x signal-to-offset, improving as drift grows. The estimator-consistent fix requires
reordering `grpo_step`, which four live loops import; not worth that risk for a 10 % bias on a term
that is itself 0.5-4 % of the reward spread.

## Verifier feedback

**2026-08-01 (planner).** Code read end to end (`anchors.py`, `reward.py`, `trainer.py::grpo_step`,
`definitions/sae_grpo.py`, `train_steps/sae_grpo.py`) — matches this log; all four gating decisions
are in the code as described. Tests independently rerun 55/55; smoke log steps 0-5 match verbatim.
Hash neutrality is observed rather than argued: at check time both 960 h arms were training *through*
the edited source tree.

Preflight audit of the anchored checkpoint, each item verified against the real file:
- The frozen-base tying survives `state_dict()` + `torch.save` — `epoch.001.pt` has 1862 keys over 947
  unique storages, `ref_av` contributing 115 untied records = **759 MB** (5.927 GB total vs the recon
  arm's 5.168 GB). This **corrects the log's earlier "~150 MB" snapshot estimate** (LoRA-A at r=128 is
  bigger than assumed) and is consistent with the observed +0.7 GB GPU.
- theta_0 cannot drift: AdamW skips params with `grad is None` and `epoch.001.opt.pt` is +11.8 KB on
  the anchored arm — names only, no moment tensors.
- `ExtractAvSubmodelJob` returns 719 keys, set-identical to the recon checkpoint's; no `ref_av.*` key
  matches, so the leak assertion does not fire.
- There is no DDP (`reduce_type="param"` makes `maybe_make_distributed_module` return None); the
  param-averaging path all-reduces frozen params including `ref_av`, but they are bit-identical across
  ranks, so it is a no-op to <= 1 ulp.
- `clear_autocast_cache()` still sits after *all* no-grad work including both new closures, so the
  RUN-1/2 zero-gradient bug cannot recur here.

## Live reward: no G2P anywhere (source-verified correction, 2026-08-17)

Moved here from the index 2026-09-14; the index keeps the reward formula, this is its source trace.
The formula was corrected 2026-08-17 and replaces the G2P form, because the live reward contains NO G2P
anywhere, verified at source: psi re-encodes the decoded string under its own graphemic BPE
(`psi_scorer.py:141-146`, `psi_align_jobs.py:87-104`; the "phones" branch exists but no live arm sets
it), the hinge is `len(decoded_string)` (`train_steps/sae_grpo.py:205-212`; `reward.py:14-15` documents
the deviation), and the old lam_4 OOV term is unwired dead code that raises if enabled. The G2P map —
first pronunciation, stress-free — survives in probes and analyses as phi = G2P(z), NOT in the reward.
Consequence, load-bearing for §3e.1 D6-PERIODIC/GAN+HOM and index queue 7: the orthographic channel is
LIVE — homophone spellings are NOT reward-invariant; the scorer carries a per-state price on
orthographic length (the minimal-state exploit's substrate) plus any spelling-specific emissions it
learned.
