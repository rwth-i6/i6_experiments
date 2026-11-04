# Granite + Qwen across Open ASR and Loquacious

`users/berger/configs/baselines/open_asr.py` now calls `open_asr.qwen.run()`
for the full suite plus Loquacious, including existing two-pass runs. Existing
runtime fingerprints are unchanged. This page covers Granite; Parakeet has a
separate driver (RNNT_QWEN.md). Cohere fusion remains out of scope.

## Current development round (2026-10-15)

`config/qwen_tuning.py` selects only LibriSpeech dev-other and Loquacious
dev.short (subset selected before decoding). Both Qwen3-0.6B-Base and
Qwen3-1.7B-Base get one-pass scales 0.1/0.3/0.5 at beams 32/128 and thresholds
6/6, with the existing token/word rewards and casing policy fixed. Each model
also rescores the same 50-best candidates with scales 0.1/0.3/0.5 and the fixed
initial-capital-period policy. Candidate generation is shared between sizes;
changing rescoring scale only runs selection/scoring, not another GPU forward.
Use LibriSpeech open-tree candidates for both dev sets and corpus-lexicon
candidates for Loquacious. No held-out 1.7B grid is enabled before selection.

## Coverage

All eight pinned `data.open_asr.DATASETS` entries, including LibriSpeech clean
and other, AMI, GigaSpeech, SPGISpeech, VoxPopuli, Monsoon and Earnings22. Reuse
the existing prepared HF datasets and native Granite jobs. Include native
scores in the new overview so neither a degraded first pass nor a rescoring
gain over a degraded first pass is mistaken for a gain over native decoding.

Loquacious: dev and test from `speechbrain/LoquaciousSet`, config `small`, revision
`0e84cdb9e4b826afaabca5d33ec9453b11aacef3`. The public feature schema was checked:
`ID`, `duration`, `wav` (Audio, decode=False), speaker metadata, `text`. Rename
ID/wav to id/audio, decode/resample at 16 kHz, save without re-encoding. No OggZip
conversion, preemphasis, peak normalization or training-data preparation.
HF cache lives under `/work/asr4/berger/huggingface`.

Decode Loquacious dev/test once per method; derive Common Voice, LibriSpeech,
VoxPopuli and YODAS subsets by source IDs and dev.short by the existing canonical
segment list. Empty-normalized reference IDs are explicitly recorded and may
be omitted from dev.short; other missing IDs fail validation. Subsets share the
parent decode, so no subset-specific runtime is claimed.

## Frozen methods — no new test-corpus tuning

One-pass: lexicon-free `llm-timesync-beam-search`, original pinned 9804f5aa build,
Qwen3-0.6B-Base float32, LM scale **0.3**, token reward **0.5**, word reward **1**,
initial-capital/lowercase formatting, no terminal period. Final/pre-LLM caps
**32 / 128**, thresholds **6 / 6**. Reuse the original raw-piece inventory and
passed native numerical gate. Stepwise statistics remain off.

Two-pass: fixed **50-best** open-tree no-LM first pass on the ff206f90 build,
with known-excluding fallback, beta **1**, alpha **2**, final/word beams **256**,
threshold **16** — the same candidate-generation condition used in LibriSpeech
rescoring. First-pass no-LM does not mean zero fallback biases. Exact top-text
and cumulative-cost parity with independent one-best decoding remain mandatory.
Compact candidate files omit per-candidate frame tracebacks to keep memory/disk
bounded, while the first-pass recognition log remains registered.

Score only the selected initial-capital-period formatting (not a corpus-wide
formatting grid), using float32 GPU full-sequence Qwen forwards. Select by:

```
AM + first-pass LM/bias component + 0.3 * Qwen NLL - 0.5 * Qwen text-token count
```

EOS is scored at full weight and excluded from the token reward. No word reward
is added in two-pass selection, no first-pass column is dropped, and no token
averaging is applied. Candidate selection sees no references. These frozen
settings attained 2.5220% dev-other in the full-dev rescoring sweep.

The default first-pass lexicon is LibriSpeech, with open fallback. Also register
corpus-lexicon alternatives for GigaSpeech, VoxPopuli and both Loquacious splits.
Both candidate generators have **no word LM**: the external LM is Qwen, not an
additional transferred or corpus 4gram. These variants isolate vocabulary/tree
effects rather than stacking a second LM. No vocabulary is extracted from
evaluation transcripts. One-pass needs no lexicon and has no resource variant.

## Grouped scoring and outputs

Use the existing updated HF normalizer, compound equivalence and ID-based
`ScoreBenchmarkJob`. Earnings22 chunks are decoded/rescored independently and
then concatenated in chunk order before normalization and scoring against the
parent reference, exactly as in the existing suite. No per-chunk oracle is
computed against a whole-session reference. Neither method carries Qwen history
across published audio chunks.

Outputs and aliases: `open-asr/qwen-granite/`. Conditions are `native`, `one-pass`,
`two-pass-librispeech-lexicon`, `two-pass-corpus-lexicon` where available, and
their corresponding `first-pass-*` controls. `overview` includes all full-corpus
and Loquacious subset scores, with lexicon/LM source labels and runtimes.

One-pass runtime is measured by the decoder job. Two-pass job-wall RTF includes
candidate generation **and its independent one-best parity re-decode**, plus
Qwen scoring; selection/scoring overhead is excluded. It is a conservative
measurement, not a production decoder-only speed claim.

There are 10 new one-pass datasets and 14 first-pass candidate-generation
conditions, followed by 14 fixed Qwen scoring jobs. Open ASR native/data downloads
are reused; two Loquacious native baselines are new. All corpus-subset scores are
CPU-only and reuse hypotheses. No hyperparameter Cartesian sweep is scheduled.
