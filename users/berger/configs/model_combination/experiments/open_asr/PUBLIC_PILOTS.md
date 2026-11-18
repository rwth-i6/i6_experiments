# Independent public follow-ups — 2026-11-17

Completed 2026-11-18: see [results and decision](PUBLIC_PILOT_REVIEW_20261118.md).

These two opt-in configs implement the approved next steps while the AppTek
forced-scoring diagnostics continue. Neither is imported into the large main
experiment graph. Both require the existing completed public frozen selection
and model/data artifacts; they do not reconstruct download/build jobs.

## 1. Transcription-convention audit

Run from the usual Sisyphus setup directory:

```bash
./sis m config/convention_audit.py
```

One CPU analysis reads 75 frozen records: five conditions on both dev corpora
and the 13 completed evaluation datasets/subsets, plus available native
Granite/Parakeet controls from the existing snapshots. Frozen conditions are the selected
Granite/Parakeet AM-only controls, Speech-Qwen 1.7B standalone, and each AM's
one-pass Speech-Qwen 1.7B fusion. Native controls are reported separately to
distinguish the model's surface convention from tree-lemma capitalization.
The report counts cased words, contractions
(including ambiguous 's/'d), hyphens, dotted abbreviations, titles, digit forms
and lexical number words. It then compares existing AM/Speech/fusion error counts
on overlapping slices defined by Speech-Qwen's output features.

This is descriptive output-conditioned analysis, **not a search-error test or a
transcript conversion recipe**. Number-word flags include ambiguous words such
as "oh"; they are not semantic annotations. No transcripts are rewritten, no
new WER normalization is introduced, and no hypotheses are forwarded through a
model. Existing compound-equivalent per-system denominators are retained.
Loquacious totals/subsets remain separate; their overlapping rows are not pooled.

Outputs:

- `output/reports/asr-lm/transcription-conventions.md`
- `output/convention-audit/f4212424889ee5d3/{report.md,summary.json}`

## 2. Controlled noise pilot

```bash
./sis m config/noise_pilot.py
```

The graph contains one preparation job, **nine GPU recognitions**, nine scoring
jobs and one report. It uses the 48 GB GPU queue with six-hour limits per
recognition; no sharding or full-suite evaluation is scheduled.

| Factor | Frozen choice |
| --- | --- |
| Corpus | Existing lossless HF LibriSpeech dev-other |
| Subset | 256 utterances; seed 20261117; hashed chapter order and hashed utterance order, round-robin across chapters |
| Selection information | IDs only; no text, recognition errors or model scores |
| Conditions | Clean, 20 dB and 10 dB additive stationary Gaussian white noise |
| Noise pairing | One PCG64 realization per utterance, shared across noise levels and all systems |
| SNR | Whole-segment RMS, including silence; no VAD |
| Gain | Common per-utterance gain for all three copies, based on their maximum peak; no clipping |
| Storage | Float32 WAV embedded in the HF dataset; no lossy encoding or temporary-file dependency |
| Systems | Parakeet selected AM-only search; Speech-Qwen 1.7B beam 8/lp 0; selected Parakeet + Speech-Qwen 1.7B one-pass fusion |
| Tuning | None: exact clean-dev-selected checkpoints, search config files and LM/word scales |
| Scoring | Existing updated HF normalizer, compound equivalence and kaldialign |
| Uncertainty | 10,000 paired chapter-bootstrap replicates, common chapters across all conditions; unadjusted/exploratory |

The common gain may change clean amplitude relative to the original corpus;
therefore the clean control is rerun on the same prepared subset rather than
substituting the old full-corpus result. The manifest records exact IDs, source
audio hashes, seeds, common gains, measured SNRs, source fingerprint, reference
checksum and data-library versions. Every recognizer gets the same prepared
waveforms. Fusion depends on the standalone Speech scoring for its own SNR.
No prior/ILM, pruning, casing, prompt or fusion-scale change is introduced.
Stepwise RASR statistics remain disabled in the frozen configurations.

The report gives WER, I/D/S, recorded RTF and paired fusion differences against
the **better constituent at each SNR**, recomputed within bootstrap replicates.
The seed fixes one noise realization per utterance; intervals do not measure
variation over alternative noise realizations. This pilot tests a specific
synthetic acoustic perturbation, not realistic environmental-noise robustness.

Outputs:

- `output/reports/asr-lm/noise-pilot.md`
- `output/noise-pilot/f4212424889ee5d3/n256-seed20261117/`
  (manifest, references, per-condition hypotheses/scoring/runtime, report/JSON).

## Decision checkpoint

First inspect whether either constituent degrades differently and whether
fusion improves on the better one at the same SNR. Expand the noise study only
if this pilot reveals an informative pattern. The convention audit can motivate
targeted public forced-scoring tests after the AppTek scorer is validated;
surface-form co-occurrence alone does not establish why fusion loses.
