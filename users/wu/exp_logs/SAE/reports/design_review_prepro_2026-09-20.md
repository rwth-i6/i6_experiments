# Design review: SAE_4A_prepro.md (paper-faithful waveform cut), 2026-09-20

Verdict: APPROVE_WITH_AMENDMENTS. Reviewed before any job. Read-only; nothing else touched.
Files read: SAE_4A_prepro.md (all), SAE.md:79-105, SAE_ref.md:1-60, SAE_4A_budget.md (all),
SAE_4A_blankfree.md:65-100, SAE_4A_attrib.md:105-130, 500, 700-735, reports/lit_w2vu2_preprocessing,
survey_audio_pipeline, extract_sil_rate, impl_gold_sil_retained; code: blankfree_data.py:44-166,
vad_port.py:30-43, blankfree_eval_jobs.py:46-175, rate_term.py:1-20,142-144,214-270,
train_steps/sae_blankfree.py:36-60, quantize_states.py:48-74, encoders/wav2vec2.py:105-125,
eval_jobs.py:1354-1435 (PairedPerDeltaJob), fairseq vads.py:31-92, remove_silence.py:34-51,
wav2vec_extract_features.py:59-62, bed summary.txt (BlankfreeVadHdfJob.SAjz8y1cT06g).

## Q1. Does the design isolate the placement of the cut?

Yes as a treatment, no as a single mechanism. Everything below moves with the cut and is part of
"the paper's chain"; none is a confound of the QUESTION as posed ("does the paper's cut change the
bed's outcome"), but a result cannot be attributed to one of them without the measurements in (M):
1. Retained-frame set: raw 10 ms labels (vads.py:80-92) vs the bed's ORed 50 Hz mask
   (vad_port.py:30-43, subframes 2, tail truncated, tail padded silence blankfree_data.py:95-98).
   The raw rule drops <= one 10 ms sliver per segment boundary; expected < 1 % of 15,427,853.
   Measure, don't fund: the data job holds the same raw labels, so it can re-derive the bed's OR
   mask and must reproduce 15,427,853 / 831,372 / 781,130 EXACTLY (rVADfast is deterministic on
   the same HF 16 kHz decode). The spec (prepro.md:64-66) only promises "close" for T'; add the
   exact OR-mask assert. Failure = the cut is not from the bed's decisions -> STOP.
2. SSL context and splice seams: the intended delta. Effect size is measurable label-free before
   training: agreement of the bed's units with the new units on frames matched by raw_index,
   overall and by distance to the nearest splice; wav2vec2-large's attention is utterance-wide, so
   some disagreement is expected everywhere, more near seams.
3. Front-end normalisation on the trimmed waveform (wav2vec2.py:109-116 per-utterance zero-mean
   unit-var == fairseq layer_norm, wav2vec_extract_features.py:59-62): silence removal raises the
   waveform variance, so every retained sample is rescaled. Part of the paper's chain; not
   separable from the cut in fairseq either. Disclose.
4. Frozen quantizer applied out of distribution (PCA96+K500 fit on full-waveform states incl.
   ~15 % silence frames; standardisation uses the fit dump's global mean/std,
   quantize_states.py:48-53). Prediction: silence-dominated centroids go unused, unit unigram
   entropy drops, k-means distortion rises. Report distortion and unit entropy new vs bed; if
   distortion jumps, the arm tests "quantizer mismatch" as much as "cut placement" and a refit
   variant becomes the follow-up, not a reason to stop.
5. T' vs T: training normalises by retained S (blankfree.md:101); the rate term is E[N]/T with T
   retained (rate_term.py:8-13) and rho = phones_per_word x 2.7 words/s (a disclosed constant,
   rate_term.py:142-144), so rho is not tied to either clock; a ~1 % change in retained seconds is
   immaterial against the [0.6,1.5] x rho band. The gate's rate clause reads
   phone_rate_original_hz (attrib.md:358) = 50 x phones / sum(orig_length): orig_length MUST be
   the untrimmed 50 Hz count (spec :59 says so); assert sum == 919,980 (dev-other), 968,057,
   18,088,388. Steps per sub-epoch may move by one (max_seqs 128 dominates); print it.
6. raw_index: PER (BlankfreeGreedyPerJob: gold strings, feature lengths, orig_length) and the
   derangement gap (units, eta, raw_hyps) do NOT use raw_index, so both gate clauses are exact.
   Only the gold-frame diagnostics (private_code) use it -> approximate, disclose. Frame centre
   under the HF conv stack is 320 t' + 200 (kernel 400, stride 320), not +160 (spec :59); assert
   monotone non-decreasing and < orig_length.
7. eta (frozen PCA-16 speaker vector, reverse.py:16, from the S1b chain on full-waveform states)
   is carried unchanged. It is a speaker code consumed by a cold-trained phi; disclose, no action.
Third variant "raw 10 ms mask, no cut": NOT worth a training slot; item 1's frame-level count
bounds its effect, and items 2-4 are not separable by that variant anyway.

## Q2. Comparison and gate

ctrl_20 vs prepro_20 in the same pack, same seed, N = 20: correct (budget.md "never against
ctrl_50 at sub-epoch 20"). G4a.8's PER/rate/gap clauses copy G4a.4 and read the claim. The PRIMARY
read is undefined as written: "within the seed band of the attribution step" -- no seed band exists
on the blank-free bed at any N (every cold arm is n = 1; budget.md:104 "needs a second seed";
attrib step 6 item 4 is a chance-string null, not a seed spread; the only two-seed pairs are the
fairseq GAN arms, attrib.md:700-712, a different bed). PairedPerDeltaJob's speaker-clustered
bootstrap covers item sampling only; previous cold-arm paired deltas of 0.007-0.02 had CIs
excluding zero while both arms sat in the 0.83-0.91 chance band (attrib.md:500). So a within-band,
CI-excluding-zero delta is the likely outcome and decides nothing.
Cheap fix: the pack occupies 2 of the 4 GPUs of a whole-node allocation (budget.md "packed four per
node (the partition allocates whole 4-GPU nodes)"; pack_jobs GPUS_PER_NODE = 4). Add ctrl_20_s1
(second seed, same everything) at zero node-hours; ctrl_20 vs ctrl_20_s1 IS the seed band the gate
names. Optionally prepro_20_s1 in the fourth slot. This is not a new arm of the user's question.
Do not reuse the prior pack's ctrl_20 (same seed) as a band: it reads GPU nondeterminism only.
Label-free fidelity read: the weighted-LM-perplexity selection statistic compares two content-free
decodes and cannot certify fidelity to the paper's chain. Fidelity to the reference is a
FEATURE-level property: the unit-agreement / distortion / retained-frame reads of Q1 (2, 4, 1).

## Q3. Prediction

The 1.0 ablation (29.3 vs 21.4, silence in vs out) does not bear on this pair: BOTH arms remove
~85 % of raw frames and both keep the same 7.9 % gold-SIL retained residue (the retained sets are
near-identical by construction). The only mechanism that differs is what the SSL transformer and
the front-end normaliser see. Given the attribution and prior phases located the content-free
outcome in the objective (private code satisfying the trigram; SAE_4A_prior.md Step 0), the
honest pre-registered expectation is: prepro_20 stays in the 0.83-0.91 band; unit agreement with
the bed well below 1 but far above chance (1/500). A null licenses "cut placement is not the
missing ingredient for cold take-off at N = 20 on this bed" -- NOT "the masking convention is
equivalent to the paper's" (prepro.md:79-80 overclaims; two chance-level PERs cannot certify
equivalence). A take-off needs the standing second seed before it is claimed.

## Q4. Data-job correctness risks (assert in the job, print in summary)
- Exact reproduction of the bed's OR-mask totals from the same raw labels (Q1.1).
- sum(orig_length) per split == bed raw totals; T' >= 2 for every utterance (the bed had 0 with
  S < 2; a lone 160-sample segment yields T' = 0 under the 400-sample kernel -> HF error).
- Units assigned from the fp16-ROUNDED states (the bed's units come from the fp16 pickles of
  AvStatesJob), via reconstruct_quantizer/standardize_frames/quantize_utt with the frozen pkl.
- Tag sets per split equal the bed's HDF tag sets (dev 2864 / 2703; train == CvHoldoutSplitJob
  ids); shard membership may differ, tag set may not.
- raw_index int32, monotone, < orig_length, len == T'.
- Per-utterance loop at batch 1 with the trimmed waveform passed to the same Wav2Vec2EncoderV1
  (its normaliser then acts on the trimmed signal, as the paper's extractor does).

## Q5. Launch decision and cheapest falsifier

Do not stop. Before the pack is funded, run the data job on dev-other alone (~15 min GPU, 2864
utts) and read: (a) recomputed OR-mask total == 781,130 and sum(orig_length) == 919,980 (else
STOP: wrong decisions or wrong clock); (b) total T' and the per-utterance T'/T distribution;
(c) unit agreement bed vs new on raw_index-matched frames, overall and by splice distance;
(d) k-means distortion and unit unigram entropy, new vs bed; (e) count of T' < 2. If (c) is ~1.00
the treatment is empty and the pack should not be funded; if (c) is low with a large (d) jump, the
arm is a quantizer-mismatch test as much as a cut test (record before launch).
Cannot be falsified as written: "equivalent to the paper's" -- no label-free or labelled read in
this design can establish equivalence of two content-free arms; rewrite the null clause.

## Amendments in priority order
1. prepro.md:77-80 and Gate: replace the seed-band clause with the ctrl_20 second seed in the idle
   slot (ctrl_20_s1; the band = |PER(ctrl_20) - PER(ctrl_20_s1)| paired), and rewrite what a null
   licenses (no equivalence claim).
2. Data-job spec: exact OR-mask reproduction assert, orig_length total assert, T' >= 2 assert,
   fp16-rounded unit assignment, tag-set equality; print retained-frame delta OR-vs-raw.
3. Register the label-free effect-size reads (unit agreement by splice distance, distortion,
   unit entropy) in the data job's summary; they are the fidelity read, not weighted LM ppl.
4. Minor: frame centre +200 not +160 for raw_index; disclose eta carried from the bed's chain.
