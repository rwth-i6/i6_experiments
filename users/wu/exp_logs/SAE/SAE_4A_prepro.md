# SAE 4A — paper-faithful silence handling (wav2vec-U 2.0 waveform cut before SSL extraction)

## State

Phase opened 2026-09-20 on the user's directive ("do what wav2vec-U 2.0 actually does"; the
feature-masking convention was never approved). Survey
(`reports/survey_audio_pipeline_2026-09-20.md`), pre-registered design and gate G4a.8 below;
design review and code review both APPROVE_WITH_AMENDMENTS, amendments applied
(`reports/review_prepro_2026-09-21.md`; code speech-llm d92109b, f184df4, 1b25144, dbfe6fb;
build/launch reports `reports/impl_prepro_2026-09-20.md`,
`reports/exec_prepro_devother_launch_2026-09-21.md`, `reports/exec_prepro_pack_launch_2026-09-21.md`).
Orchestrator constants: LR warmup 2 sub-epochs at N = 20 (`SAE_4A_budget.md` amendment);
ctrl_20_s1 moves flat_seed 1, random_seed 1, random_seed_offset 1000 (full replicate band;
sub-epoch-1 batch overlap with ctrl_20 0.26, chance 0.25).
Pre-funding inputs all FINISHED and read (Results): data jobs `TrimmedAudioBlankfreeDataJob`
train qb4o6dlW3urA / dev-clean hxIx0ItTvx15 / dev-other 0IOLr6hZnYWj (OR totals exact, unit
agreement 0.74 / 0.73 / 0.71, plateau 0.72–0.75 far from splices); extraction-path null
`UntrimmedEncodeAgreementJob.WnGSatwxUEYY` exact (1.0000), so the plateau is the cut's effect.
Their managers (4154973, 127865, 129170) exited cleanly.
CLOSED 2026-09-21, G4a.8 FAIL (Results, "G4a.8 read of the pack"): the pack
`PackedBlankfreeTrainJob.5EIGJJ1MkcO9` (arms ctrl_20 / ctrl_20_s1 / prepro_20, flat inits
`FlatRecognizerInitJob.0J9d6wjrkRYH` / `.DMSwTLXT9MWG`, Slurm 1921103, 3 h 27 wall,
`reports/exec_prepro_pack_check_2026-09-21.md`) finished all arms at sub-epoch 20 with greedy PER
0.875–0.889 on dev-other, far above the 0.50 clause; the paper's cut is +0.014 PER worse than the
masked control at ep20, outside the seed band (−0.001), and worse at every kept epoch from 4 on.
The silence-handling convention is not what stalls the bed (directive item 2, negative branch).
Manager 347372 exited cleanly; no manager or watcher of this phase is live.
Pending experimental decision (orchestrator ruling, user may overturn): future arms keep the
banked masked-feature bed, because its paired controls exist (ctrl_20 / ctrl_20_s1 of this pack)
and the cut measured no better; the trimmed streams (data jobs above) stay banked for a swap.
NEXT: none in this phase; successor work is `SAE_4A_lexlat.md`.

## Objective

The reference (`reports/lit_w2vu2_preprocessing_2026-09-20.md`, arXiv 2204.02492v2 §4.1) removes
silence with rVAD from the WAVEFORM and extracts layer-15 wav2vec 2.0 Large features from the
trimmed audio; the bed extracts features from the full waveform and masks the non-speech frames
afterwards (`SAE_4A_blankfree.md`, "Preprocessing": a disclosed difference inherited from the §1c
reproduction). The two differ in what the SSL transformer sees: silence context in ours,
concatenation seams in the paper's. No paper measures the difference. Question: does the paper's
cut change the cold blank-free bed's outcome at the same budget?

## Constraints

- One arm (user: "one arm is probably enough"): the paper's cut, everything else the bed's
  (sil_prob 0.5 text prior, trigram in the DP, reverse units from the same quantizer applied to
  the new features, schedule of the budget round), at the N decided under the user's item 1.
- Control: the budget round's ctrl arm at the same N (ctrl_50 if N = 50; otherwise the matched
  sub-epoch of ctrl_50 / ctrl_100, read at matched completion, or a fresh ctrl_N if the schedule
  differs).
- Labels enter nothing: rVAD is unsupervised; the trimmed-audio frame index must keep the
  original-frame mapping so PER and the gold-frame diagnostics stay computable.
- Cut rule: fairseq `vads.py` + `remove_silence.py` semantics, from the bed's own rVADfast
  decisions (same version and threshold), so the retained-speech set is the bed's; only the
  placement of the cut moves.

## Design (pre-registered 2026-09-20, from `reports/survey_audio_pipeline_2026-09-20.md`, before any job)

Today's chain: full-waveform wav2vec2-large-lv60 layer-15 states (`AvStatesJob`, 50 Hz, 1024-d)
-> rVADfast 0.4 on 10 ms frames, ORed to 50 Hz (`vad_port.py`, subframes 2) -> non-speech frames
dropped from features and units (`BlankfreeVadHdfJob.SAjz8y1cT06g`: feats / units / raw_index /
orig_length HDFs) -> blank-free training reads those streams. The §1c reproduction did the same
in feature space; no trimmed audio or fairseq `.vads` exists on disk.

The paper's chain, reproduced as ONE new data job (GPU, per-utterance loop, sharded like
`AvStatesJob`) that emits the SAME four streams so every downstream job is reused unchanged:
1. decode the utterance at 16 kHz (the HF ogg dataset the bed uses);
2. rVADfast, threshold 0.4, the bed's own installation, taking its RAW 10 ms labels;
3. fairseq cut rule (`vads.py:80-92`, `remove_silence.py:44-51`): segments start at the first
   speech frame x 160 samples and end at the first non-speech frame x 160, an open tail ends at
   the last sample, no margin, no minimum length, no merging; the speech segments are concatenated
   into one trimmed waveform; an utterance with no speech segment is kept whole;
4. wav2vec2-large-lv60 (the same HF weights, front-end waveform normalisation on the TRIMMED
   waveform, as the paper's fairseq extractor does), hidden_states[15] -> [T', 1024] fp16;
5. units: the bed's frozen quantizer (`QuantizeStatesJob.FWpGhC941JMi`: PCA 96 + k-means 500),
   applied per frame to the new states exactly as `AssignUnitsJob` does, no refit;
6. raw_index: each trimmed frame t' maps through the segment sample offsets to the original
   sample of its centre (320 t' + 160) and to original frame // 320; orig_length = the original
   50 Hz frame count. Both approximate after splices (the encoder's receptive field crosses
   them); used by PER / rate / gold diagnostics only, never by training.
Disclosed differences from the paper's scripts: rVADfast (pip) vs fairseq's `speechproc` rVAD
(same algorithm and constants, not bit-identical), HF weights vs the fairseq checkpoint of the
same model, in-memory concatenation vs a re-saved wav. Expected retained frame counts: close to
but not equal to 15,427,853 / 831,372 / 781,130 (the 50 Hz OR-aggregation is gone); the job
prints both totals and the per-split difference.

Arms (one pack, N = 20, budget-round schedule at N = 20, kept 1 / 4 / 10 / 20, registered PER /
rate / derangement-gap evaluations at every kept epoch, PairedPerDeltaJob prepro_20 vs ctrl_20):
- `ctrl_20`: the bed's streams (`SAE_4A_budget.md` ctrl arm at N = 20);
- `prepro_20`: the new streams, everything else identical (same segments split, same prior,
  same reverse quantizer, same seed).
Cost: data job about 2 h GPU train (4 shards in parallel: ~30 min) + 15 min dev; the pack 6.7 h.
Label-free monitors as in the budget round; the wav2vec-U 2.0 selection statistic (4-gram
phone-LM perplexity / vocabulary-seen fraction squared, SIL stripped) computed for both arms at
every kept epoch from the existing greedy decodes.
Pre-registered prediction (original): none directional. If the paper's cut helps, the earliest
sign is a lower lattice term at matched sub-epoch and a different phone rate; if PER differs by
less than the seed band, the placement of the cut is not a lever.

**Design review amendments (2026-09-20, `reports/design_review_prepro_2026-09-20.md`,
APPROVE_WITH_AMENDMENTS, applied before any job):**
- Seed band: no seed band exists on the blank-free bed at any N (every cold arm n = 1), so
  "within the seed band" was undefined. The pack gets a third arm `ctrl_20_s1` (second seed);
  ctrl_20 vs ctrl_20_s1 IS the band; paired reads prepro_20 vs each ctrl and ctrl vs ctrl_s1.
- Overclaim struck: a within-band null between two content-free arms licenses only "the cut's
  placement is not the missing ingredient at N = 20", never "the masking convention is
  equivalent to the paper's". Expectation, stated now: both arms remove about 85 % of frames
  with the same 7.9 % gold-silence residue, and the 1.0 ablation contrasts silence-in vs
  silence-out, so it predicts prepro_20 in band.
- Fidelity is a feature-level property and is read in the data job, not from two chance
  decodes: unit agreement bed-vs-new on raw_index-matched frames, overall and by distance to
  the nearest splice; k-means distortion and unit entropy new vs bed; OR-vs-raw retained delta.
  A unit agreement near 1.00 is an empty treatment and the pack is not funded.
- Data-job asserts: the OR-aggregated mask recomputed from the same raw labels must reproduce
  the banked 781,130 / 831,372 / 15,427,853 exactly; sum(orig_length) equals the banked raw
  totals (the gate's rate clause reads phone_rate_original_hz); T' >= 2; units from the
  fp16-rounded states; frame centre 320 t' + 200. PER and gap jobs do not use raw_index.
- Cheapest check, pre-funding: the dev-other data job alone (~15 min GPU): OR total == 781,130
  (else STOP), T'/T, unit agreement, distortion, T' < 2 count. The pack is funded only after it.

**Code review amendments (2026-09-21, `reports/review_prepro_2026-09-21.md`,
APPROVE_WITH_AMENDMENTS; applied before the pack is funded):**
- Seed band: RETURNN `random_seed` moves theta / phi init and the RNG but not the sequence order
  (laplace ordering is seeded by epoch + `random_seed_offset`, default 0), so ctrl_20_s1 as built
  was an INIT band only. Amended: ctrl_20_s1 also sets `random_seed_offset` (written into the
  seq-order-controlling dataset; verified in RETURNN basic.py that laplace ordering is seeded by
  full-epoch index + offset), so the band is the full replicate (init + order + sub-epoch
  composition). Offset 1 was measured to be a one-epoch SHIFT of the same shuffles (s1 sub-epoch k
  = ctrl sub-epoch k + 4), so the offset is 1000: no full-epoch shuffle is shared.
- Extraction-path null: the fidelity statistic (unit agreement bed vs new) mixes the cut with any
  encoder / numerics / index-mapping difference. A separate small job
  (`UntrimmedEncodeAgreementJob`, 50 dev-other utterances) encodes the UNTRIMMED waveform through
  the identical path and reports agreement with the banked units; expected near 1.00, and the
  cut's own effect is read as the data job's agreement against this null. Convention: the data
  job's denominator is all T' trimmed frames, including frames the bed's OR mask dropped.
- Sound per the review: ctrl_20 is the budget bed at N = 20 (sub-epoch = partition 4 of the
  corpus, independent of N; 601 s per sub-epoch carries over), lr / tau lists match the intended
  shape, prepro_20 differs in the four HDF paths only, every read uses the arm's own streams, the
  rate clause divides by the identical orig_length, no labels reach training or selection.

## Gate

G4a.8 (pre-registered here before any job; thresholds copy G4a.4, `SAE_4A_budget.md`): greedy PER
< 0.50 on dev-other at the final sub-epoch, emitted rate in [5.80, 14.49]/s, derangement gap > 0;
primary read: paired PER delta (PairedPerDeltaJob) prepro_N vs ctrl_N at the same sub-epoch, with
the within-band rule of the attribution step (a paired delta inside the seed band is not
content; amended: the band is ctrl_20 vs ctrl_20_s1 in the same pack, see Design amendments). Reference-fidelity read (label-free): the wav2vec-U 2.0 selection statistic (4-gram
phone-LM perplexity of the decode divided by the squared fraction of vocabulary seen, SIL
stripped) for both arms at every kept epoch. Abort rule as G4a.4.

## Results

### Pre-funding check: dev-other data job (read 2026-09-21)

`TrimmedAudioBlankfreeDataJob.0IOLr6hZnYWj` (speech-llm d92109b; Slurm 1920278, booster, 1 GPU;
`output/summary.txt`), 2864 utterances:

| statistic | value | bed / reference |
|---|---|---|
| OR mask re-derived from the raw rVAD labels | 781,130 | 781,130 banked (delta 0: no STOP) |
| original frames (sum orig_length) | 919,980 | 919,980 banked raw |
| trimmed frames T' | 775,542 | 781,130 OR-retained (−5,588, −0.7 %) |
| samples kept | 0.8436 of the raw waveform | 6,881 speech segments, 2.40 per utterance |
| utterances kept whole (no speech) / T' < 2 | 0 / 0 | |
| unit agreement bed vs new, raw_index-matched, all T' frames | 0.7055 | not near 1.00: the treatment is not empty |
| by distance to the nearest splice (frames) [0,1) / [1,2) / [2,4) / [4,8) / [8,16) / [16,inf) | 0.22 / 0.40 / 0.50 / 0.64 / 0.75 / 0.72 | n = 8,014 / 7,983 / 15,916 / 31,773 / 62,948 / 648,908 |
| k-means distortion (PCA-96 squared distance) | 211.59 | 209.69 bed (+0.9 %) |
| unit unigram entropy (nats; dead units) | 6.011 (17 dead) | 6.034 (11 dead); ln 500 = 6.215 |
| quantizer path check on a banked utterance | 532 / 532 | |

Reading: the funding rule passes (OR total exact, treatment non-empty). The splice-distance
profile shows a local seam effect (agreement 0.22 within one frame of a splice, rising over 8
frames) on top of a 0.72 plateau far from any splice, i.e. 28 % of frames change unit even where
no seam is near; whether that plateau is the removed silence context acting through the
transformer's whole-utterance attention or an extraction-path difference is exactly what the
untrimmed-encode null (`UntrimmedEncodeAgreementJob`, code review amendment) decides: a null near
1.00 attributes the plateau to the cut. The new states fit the bed's quantizer slightly worse
(+0.9 % distortion, 6 more dead units), as expected for a quantizer fitted on full-waveform
states; the same frozen quantizer is used, as pre-registered.

**Extraction-path null (read 2026-09-21): `UntrimmedEncodeAgreementJob.WnGSatwxUEYY`
(speech-llm 1b25144; Slurm 1920624), first 50 dev-other utterances in banked order, UNTRIMMED
waveform through the new job's encoder + quantizer path: unit agreement with the banked units
15,544 / 15,544 = 1.0000 on all original frames and 13,087 / 13,087 on the bed-retained frames;
fp16 states identical to the banked states (max-abs difference 0.000000 on every frame).** The
extraction path is bit-identical, so the data job's 29.5 % unit disagreement (denominator: all T'
trimmed frames) is entirely what trimming did: a seam effect within about 8 frames of a splice and
a 28 % change of unit far from any splice, i.e. removing the silence context changes layer-15
states throughout the utterance (whole-utterance self-attention), not only at the seams. This is
the treatment the pack measures; the pack is funded (data jobs running).

### Train and dev-clean data jobs (read 2026-09-21; funding inputs of the pack)

`TrimmedAudioBlankfreeDataJob` train qb4o6dlW3urA (Slurm 1920622, 4 shards) and dev-clean
hxIx0ItTvx15 (Slurm 1920623), `output/summary.txt`; same job class and constants as the dev-other
row above. Every assert held: OR mask re-derived exact (delta 0), no utterance kept whole, no
T' < 2, quantizer path 1.000 on one utterance per shard.

| statistic | train (28,539 utts) | dev-clean (2,703 utts) | dev-other (above) |
|---|---|---|---|
| original frames | 18,088,388 | 968,057 | 919,980 |
| trimmed frames T' vs OR-retained | 15,341,417 vs 15,427,853 (−0.56 %) | 825,888 vs 831,372 (−0.66 %) | 775,542 vs 781,130 (−0.7 %) |
| samples kept | 0.8485 | 0.8537 | 0.8436 |
| speech segments per utterance | 4.57 | 2.54 | 2.40 |
| unit agreement bed vs new (all T') | 0.7377 | 0.7268 | 0.7055 |
| agreement [0,1) / [8,16) / [16,inf) from a splice | 0.27 / 0.78 / 0.75 | 0.27 / 0.78 / 0.74 | 0.22 / 0.75 / 0.72 |
| k-means distortion new vs bed | 219.18 vs 219.93 (−0.3 %) | 203.44 vs 201.84 (+0.8 %) | 211.59 vs 209.69 (+0.9 %) |
| unit entropy new vs bed (dead) | 6.063 (0) vs 6.075 (0) | 6.021 (11) vs 6.041 (11) | 6.011 (17) vs 6.034 (11) |

Reading: the three splits show the same picture (seam effect within about 8 frames, a 0.72–0.75
plateau far from any splice, distortion within 1 % of the bed, no dead-unit collapse); the
training split fits the frozen quantizer no worse than the bed. With the extraction-path null
exact, the funding rule holds on every split and the pack is funded.

### G4a.8 read of the pack (dev-other, greedy, read 2026-09-21)

Source: `PackedBlankfreeTrainJob.5EIGJJ1MkcO9`, outputs
`output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/{arm}/ep{1,4,10,20}/dev-other/`
(`per.json`, `decode_stats.json`, `derangement_gap.json`) and `.../paired/{pair}/ep*/dev-other/summary.txt`
(PairedPerDeltaJob, 95 % speaker-clustered bootstrap, 2000 resamples);
extract `reports/extract_prepro_pack_2026-09-21.md`.

| arm | PER ep1 | ep4 | ep10 | ep20 | rate/s ep1 | ep4 | ep10 | ep20 | gap ep1 | ep4 | ep10 | ep20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctrl_20 | 0.855 | 0.875 | 0.869 | 0.875 | 3.30 | 9.17 | 9.16 | 9.16 | −0.007 | 2.84 | 4.15 | 4.27 |
| ctrl_20_s1 | 0.852 | 0.873 | 0.880 | 0.876 | 3.65 | 9.15 | 9.15 | 9.03 | −0.008 | 3.06 | 4.33 | 4.60 |
| prepro_20 | 0.851 | 0.881 | 0.891 | 0.889 | 3.45 | 9.14 | 9.26 | 9.18 | −0.007 | 2.72 | 3.85 | 4.23 |

| paired delta (PER, dev-other) | ep1 | ep4 | ep10 | ep20 |
|---|---|---|---|---|
| prepro_20 − ctrl_20 (primary) | −0.004 [−0.007, −0.001] | +0.006 [+0.003, +0.009] | +0.022 [+0.019, +0.024] | +0.014 [+0.011, +0.017] |
| ctrl_20 − ctrl_20_s1 (seed band) | +0.004 [+0.001, +0.006] | +0.002 [+0.000, +0.004] | −0.011 [−0.014, −0.008] | −0.001 [−0.004, +0.001] |
| prepro_20 − ctrl_20_s1 | (not read) | +0.008 [+0.005, +0.011] | +0.010 [+0.007, +0.013] | +0.013 [+0.010, +0.016] |

Verdict: **FAIL** on the absolute clause (no arm below PER 0.50 at the final sub-epoch; all three
0.875–0.889). The rate clause holds from sub-epoch 4 on (9.03–9.26/s; sub-epoch 1 is under LR
warmup at 3.3–3.6/s) and the derangement gap is positive from sub-epoch 4 on (2.7–4.6 nats per
frame; sub-epoch 1 is at −0.007 for every arm, i.e. before any learning). Primary read: the
paper's cut is worse than the masked control by 0.014 PER at sub-epoch 20, with a confidence
interval that excludes the seed band (−0.001 [−0.004, +0.001]), and worse at every kept epoch from
4 on against both controls; the only point in its favour is sub-epoch 1 (−0.004), before the gap
opens. Reading: the pack sits in the same 0.85–0.91 PER band as the six N = 50 budget arms
(`SAE_4A_budget.md`, 0.889–0.905) and the InfoMax arms (`SAE_4A_infomax.md`, 0.89–0.92); the
waveform-cut-versus-feature-mask convention is not what stalls the bed, and the ablation answers
directive item 2 in the negative branch. The three-way agreement of the controls with the earlier
packs also bounds the seed spread of this bed at about 0.01 PER at N = 20.

Not produced: the wav2vec-U 2.0 selection statistic (4-gram perplexity over squared vocabulary
fraction) named in Design and Gate. The pack config registers no reader for it (its docstring:
"computed from the registered greedy decodes and is not a job of this config"), and an ad-hoc
computation is not admissible as a project number. With the gate failed on the absolute clause no
decision turns on it, so it is dropped here with disclosure; any future use needs a registered job.
No audit (FAIL, routine read).
