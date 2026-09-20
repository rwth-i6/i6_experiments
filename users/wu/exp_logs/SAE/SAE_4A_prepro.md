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
