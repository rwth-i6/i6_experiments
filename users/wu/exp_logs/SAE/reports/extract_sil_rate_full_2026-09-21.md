# SIL Rate Extraction - Full Report 2026-09-21

## Task 1: Gold SIL frame share on RETAINED frames (ctrl_50 ep10 dev-other)

### Frame Statistics
- **Total retained frames:** 261,295
- **Source:** `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/private_code/PrivateCodeAnalysisJob.cQbcIJtOamLm/output/private_code.json` (A_frame.frames)
- **Split:** ctrl_50/ep10/dev-other, 2,864 utterances

### Confusion Matrix Axis Convention
- **Rows:** decoded symbols (recognizer output, indices 0-39)
- **Columns:** gold phone labels (MFA labels, indices 0-39)
- **Built via:** `np.add.at(fconf, (frame_sym[tag], frame_gold_out[tag]), 1.0)` at line 833
- **Source:** `/e/project1/spell/wu24/2026-07-13_unsupervised/recipe/2025-10-speech-llm/src/speech_llm/sae/emc/private_code.py` line 833
- **SIL index:** 39 (last symbol)

### SIL Gold Frame Count and Fraction
**Status:** CANNOT_EXTRACT
- The frame-level confusion matrix [40 × 40] is computed during job run but not banked in JSON output
- Only summary statistics are stored: frame_error_identity, frame_error_one_to_one, frame_error_many_to_one, NMI, and mapping dictionaries
- To extract per-gold-label frame counts would require reconstructing the matrix from source data (not an extraction)

### Earlier Extractor Field Clarification (j5ybPkSFOSBq ep1)
- **Value:** "SIL token runs: 38,205 of 99,047 = 0.3862"
- **Stored in:** C_nuisance.run_count["SIL"], line 226 of JSON
- **What it is:** Count of contiguous run segments of DECODED SIL symbol (not gold)
- **Built from:** frame_sym (recognizer's output sequence), lines 890-898 of private_code.py
- **Span:** Entire dev-other split, not individual frames

---

## Task 2: SIL token share in prior's phonemised training text

### Job Identification
- **Hash:** DbFgvZOGZQ8F
- **Path:** `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/text/PhonemizeWithSilJob.DbFgvZOGZQ8F`
- **Config:** sae_1c_gan_pilot.py, line 14, sil_probs=(0.1, 0.5)
- **Alias:** sae/1c/text_sil0.5

### Phonemization Parameters
- **sil_prob:** 0.5
- **surround:** True
- **seed:** 0
- **Source:** `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/text/PhonemizeWithSilJob.DbFgvZOGZQ8F/info` lines 7-9

### Training Corpus Statistics
- **Input lines:** 40,418,261 (librispeech-lm-norm.txt)
- **Output lines:** 39,630,169 (after OOV drop)
- **Dropped OOV:** 788,092 lines
- **Total tokens:** 3,232,620,004
- **SIL tokens:** 448,460,735
- **SIL token rate:** 0.138730 (13.873%)

### Surround Effect
- **Definition:** surround=True adds SIL at start and end of each line
- **Fraction of lines starting with SIL:** 100% (sampled first 200,000 lines of output/text.phn.gz)
- **Token format:** `<SIL> token1 token2 ... <SIL>` (angle-bracket wrapped)

### Data Sources
- **Stats:** `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/text/PhonemizeWithSilJob.DbFgvZOGZQ8F/output/stats.txt`
- **Text output (gzipped):** `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/text/PhonemizeWithSilJob.DbFgvZOGZQ8F/output/text.phn.gz`

---

## Summary Table

| Question | Answer | Status |
|----------|--------|--------|
| Retained frames (dev-other) | 261,295 | Found |
| SIL gold frame count | Cannot extract (matrix not banked) | MISSING |
| SIL gold frame fraction | Cannot extract (matrix not banked) | MISSING |
| Confusion matrix axes | Rows=symbols, Cols=gold phones (line 833) | Found |
| Earlier "38205" field | Decoded SIL runs (C_nuisance.run_count) | Found + Clarified |
| Prior SIL token rate | 0.138730 (448.46M / 3.23B) | Found |
| Prior sil_prob param | 0.5 | Found |
| Prior surround param | True | Found |
| Lines starting with SIL | 100% | Found |
