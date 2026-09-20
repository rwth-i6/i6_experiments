# SAE 4A — paper-faithful silence handling (wav2vec-U 2.0 waveform cut before SSL extraction)

## State

Phase opened 2026-09-20 on the user's directive ("do what wav2vec-U 2.0 actually does"; the
feature-masking convention was never approved). Nothing built yet. Code survey of the current
audio pipeline running (`reports/survey_audio_pipeline_2026-09-20.md`); the implementer spec, the
design review and the gate thresholds follow it. N = 20 sub-epochs (decided from the budget
round's label-free curves, `SAE_4A_budget.md` "Sub-epoch count"): pack of prepro_20 + ctrl_20,
about 6.7 h on one node.
NEXT: survey -> spec (trimmed audio job, feature extraction on trimmed audio, units, HDFs, control
at the same N) -> design review -> implementer -> launch.

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
Pre-registered prediction: none directional. If the paper's cut helps, the earliest sign is a
lower lattice term at matched sub-epoch and a different phone rate; if PER differs by less than
the seed band, the placement of the cut is not a lever and the masking convention stands as
equivalent to the paper's.

## Gate

G4a.8 (pre-registered here before any job; thresholds copy G4a.4, `SAE_4A_budget.md`): greedy PER
< 0.50 on dev-other at the final sub-epoch, emitted rate in [5.80, 14.49]/s, derangement gap > 0;
primary read: paired PER delta (PairedPerDeltaJob) prepro_N vs ctrl_N at the same sub-epoch, with
the within-band rule of the attribution step (a paired delta inside the seed band is not
content). Reference-fidelity read (label-free): the wav2vec-U 2.0 selection statistic (4-gram
phone-LM perplexity of the decode divided by the squared fraction of vocabulary seen, SIL
stripped) for both arms at every kept epoch. Abort rule as G4a.4.

## Results

(none yet)
