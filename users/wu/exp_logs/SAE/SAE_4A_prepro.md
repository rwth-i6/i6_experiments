# SAE 4A — paper-faithful silence handling (wav2vec-U 2.0 waveform cut before SSL extraction)

## State

Phase opened 2026-09-20 on the user's directive ("do what wav2vec-U 2.0 actually does"; the
feature-masking convention was never approved). Survey done
(`reports/survey_audio_pipeline_2026-09-20.md`), design pre-registered and reviewed
(APPROVE_WITH_AMENDMENTS, amendments applied below), gate G4a.8 fixed. Implementer building
`sae/emc/trimmed_audio_data.py` (TrimmedAudioBlankfreeDataJob) and
`configs/config_sae_4a_prepro_pack_v1.py` (ctrl_20, ctrl_20_s1, prepro_20; N = 20 from
`SAE_4A_budget.md` "Sub-epoch count"); no job launched. Blank-free modules stay untouched until
the budget pack's 11.5 h resume has passed (they are re-imported then).
NEXT: on hand-back, executor (own manager, own watcher) runs the dev-other data job ALONE and
reports the pre-funding reads (OR total == 781,130 else STOP; T'/T; unit agreement overall and by
splice distance; distortion; T' < 2 count); if unit agreement is not near 1.00, fund the
three-arm pack and register the paired reads; record both here.

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

## Gate

G4a.8 (pre-registered here before any job; thresholds copy G4a.4, `SAE_4A_budget.md`): greedy PER
< 0.50 on dev-other at the final sub-epoch, emitted rate in [5.80, 14.49]/s, derangement gap > 0;
primary read: paired PER delta (PairedPerDeltaJob) prepro_N vs ctrl_N at the same sub-epoch, with
the within-band rule of the attribution step (a paired delta inside the seed band is not
content; amended: the band is ctrl_20 vs ctrl_20_s1 in the same pack, see Design amendments). Reference-fidelity read (label-free): the wav2vec-U 2.0 selection statistic (4-gram
phone-LM perplexity of the decode divided by the squared fraction of vocabulary seen, SIL
stripped) for both arms at every kept epoch. Abort rule as G4a.4.

## Results

(none yet)
