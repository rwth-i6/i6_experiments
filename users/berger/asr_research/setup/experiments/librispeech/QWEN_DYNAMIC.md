# Dynamic lexicon-free Qwen search: 9804f5aa prototype

## Fixed-casing beam sweep

`dynamic_qwen.run_beam_tuning()` remains available as an opt-in factory, but is
no longer registered in the LibriSpeech entrypoint: tune thresholds first.
It retains the completed best transfer setting: LM scale 0.3, token reward 0.5,
word reward 1, initial-capital/lowercase formatting, no terminal-period suffix.
The reference scores 2.5374% dev-other at RTF 0.2634. Both zero-token-reward
transfer controls reproduced all original hypothesis texts exactly.

Cross final beams **16 / 32 / 64** with pre-LLM beams **64 / 128 / 256**, plus
one wider **128 / 256** point. This creates nine new GPU recognition jobs and
reuses the completed **16 / 64** reference without changing its hash. Outputs:
`qwen-dynamic-beam/dev-other/`. Final and pre-LLM score thresholds stay **14**;
checkpoint, precision, scoring normalizer, binary, rewards and formatting stay
fixed. Stepwise statistics remain off. Compare WER and RTF before changing any
thresholds. Larger caps alone need not help if score-threshold pruning dominates.

The period parser defect discovered in the transfer results (`yes` vs `true`)
is not changed by this sweep. All conditions explicitly disable the suffix;
the existing period-enabled results cannot establish a punctuation effect.

### Revised active order: thresholds, then beam caps

The threshold stage used `run_threshold_tuning()`, not `run_beam_tuning()`.
Fix final/pre-LLM beam caps at **128 / 256**, and cross final score thresholds
**4 / 6 / 8 / 10 / 12** with pre-LLM thresholds **4 / 6 / 8 / 10 / 12**:
25 conditions in total under
`qwen-dynamic-threshold/dev-other/`. All scoring parameters remain fixed at the
selected transfer setting. The completed beam-16/pre-64 reference is reused;
no higher-threshold reference is newly scheduled.

Select the lowest threshold operating point preserving the best dev WER,
checking the threshold-12 boundary for a quality plateau. Only then register a
beam-cap tightening sweep with those thresholds fixed. The fixed caps are a
practical upper budget, not a proof of exhaustive search; if quality has not
stabilized, revisit that budget rather than declare threshold convergence.
Removing the earlier grid from registration does not cancel any running jobs
or delete results. Its factory and existing artifacts remain available.

### 2026-10-07: selected thresholds and cap tightening

All 25 threshold conditions completed. Final/pre-LLM thresholds **6 / 6** are
the lowest pair attaining the best WER: **2.52963%**, 1302 errors, RTF **0.143**.
Thresholds 4/4 give 1305 errors (2.53546%, RTF 0.100); 12/12 give 1303 errors
(2.53157%, RTF 0.764). Raising either threshold alone from 6 to 12 leaves the
6/6 transcripts identical. These small error differences do not constitute a
statistically established accuracy gain, but support using 6/6 for cap tuning.

The active entrypoint now calls `run_tightened_beam_tuning()` instead of the
completed threshold grid. Keep thresholds **6 / 6** and every scoring parameter
fixed; register **17 new runs** plus the unchanged **128 / 256** reference:

- Final beams **1 / 2 / 4 / 8 / 16 / 32 / 64**, with pre-LLM cap 256.
- Pre-LLM caps **4 / 8 / 16 / 32 / 64 / 128**, with final cap 128.
- Joint smaller points **4/16, 8/32, 16/64, 32/128**.

Outputs: `qwen-dynamic-beam-thr6/dev-other/`. Pre-LLM caps below the final cap
are intentional independent-axis diagnostics: early pruning limits the number
of candidates that the LM can score. Select a fast operating point preserving
the wide reference's quality, then refine the joint caps if needed. No existing
results are deleted or jobs cancelled; the prior factories remain callable.

## Rescoring transfer (2026-10-06)

`dynamic_qwen.run_rescoring_transfer()` is registered in the LibriSpeech
entrypoint. Outputs are under `qwen-dynamic-transfer/dev-other/`. It reuses the
original pinned binary, HF audio/model downloads, raw-piece inventory and native
Qwen numerical gate. The original scorer/jobs/files are not modified; the new
implementation lives in `model_pipelines/common/dynamic_qwen_tuned.py` with a
separate job class and version hash. No RASR changes are needed for this grid.

18 bounded first-pass runs, all beam 16 / pre-LLM beam 64 / thresholds 14:

- Casing alternatives, scales 0.2/0.3/0.4, token rewards 0.25/0.5 and word
  penalties 0/-1: 12 runs.
- Fixed initial-capital/lowercase formatting, scale 0.3, token reward 0.5,
  word penalties 0/-1, with/without terminal period: 4 runs.
- Original casing, scale 0.3, zero token reward, word penalties 0/-1:
  2 behavioral controls against the completed original implementation.

The reward is outside the LM scale: Python returns NLL - reward / scale for
each eligible Qwen token; RASR then multiplies by the LM scale. EOS/BOS IDs and
the fixed period token IDs are excluded, including any occurrences of those
period IDs in ordinary word tokenization. No clipping or probability
renormalization is applied: this is an explicit length bias, not a changed LM
distribution. A native toy check confirmed negative costs and total accounting
(expected -1.31, observed -1.30999994). GPU full-history float32 inference,
stepwise-statistics-off, original HF scoring and runtime reporting are retained.

Initial-capital formatting detects the first word from the absence of RASR's
leading separator and lowercases subsequent words. The period condition appends
a fixed period token sequence before EOS, jointly scored with the final word.
It is an LM-only suffix: it does not change the acoustic transcription. It is
not an exact optional punctuation model: an AM-produced final period is not
conditionally removed, and internal punctuation/history alternatives are absent.

`wordwise-tokenization` audits the completed original scale-0.3 hypotheses.
Local audit on all 2,864 dev-other hypotheses found zero word-wise/full-string
tokenization mismatches without a suffix, and one with a period: `67%.` is
merged by the full-string tokenizer but split into `%` and `.` by the word-wise
interface (`1585-131718-0010`). Report:
`/var/tmp/qwen-wordwise-pv8nitzk/report.json`. The audit job is also registered
for reproducibility; this is evidence on these hypotheses, not a universal
tokenization guarantee.

RASR-side work would still be required for competing punctuation histories,
exact context-dependent retokenization at boundaries, or within-word LLM
look-ahead. These experiments do not fix delayed word scoring. Tune on dev only;
do not assume the rescoring optimum transfers unchanged to first-pass search.

The LibriSpeech entrypoint registers `dynamic_qwen.run()` alongside the two-pass
n-best baseline. No non-LibriSpeech or held-out runs are added. The previous
closed-tree Qwen sweep remains opt-in; earlier build/recognition identities are
unchanged.

## Build and data

RASR is built by an isolated CMake job at exact commit
`9804f5aa545d1fcf69df4843382a41242eb9f722`, with TensorFlow/TFRNN off and
x86-64-v3 enabled. This no-pronunciation-lexicon search does not need the old
separator XML patch. It uses the existing pinned Granite benchmark checkpoint
and `Qwen/Qwen3-0.6B-Base` revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.

All new runs use the same explicit HF **dev-other** dataset as n-best rescoring:
`openslr/librispeech_asr`, config `other`, split `validation`, revision
`71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`. Recognition and scoring use no
reference-dependent candidate vocabulary or formatting. HF caches remain under
`/work/asr4/berger/huggingface` via the existing global Sisyphus environment.

`GraniteRawPiecesJob` writes **raw** tokenizer pieces in acoustic output order,
with index zero replaced by the blank. This is a `vocab-text:` label lexicon,
not a word/pronunciation lexicon. Existing sanitized RASR phoneme inventories
cannot be reused: they obscure `▁`, punctuation and piece spelling. The job
checks a dense/injective inventory and rejects pieces with multiple/internal
word boundaries or whitespace.

The scorer converts contiguous `<0xHH>` byte-fallback runs to UTF-8 before
tokenization/casing. Invalid byte runs use replacement characters, not silent
deletion. Inventory validation checks this behavior against Granite's actual
SentencePiece detokenizer. The conversion does not change acoustic label ids.

## Qwen interface and numerical gates

The Python `LlmScorer` backend uses float32 GPU full-history forwards,
`use_cache=False`, bounded length-bucketed batches of eight, and no chat prompt.
It scores every continuation token using its preceding prefix, without token
averaging or context truncation. RASR owns the tokenization/history-trie cache;
the Python backend does not assemble KV inputs or keep an additional growing
score cache. BOS/EOS follow the pinned model config, falling back to tokenizer
EOS for BOS only if needed.

A scheduled native gate must pass before **every** new recognition condition:

- Forced multi-piece `hello world` and byte-fallback `¢37 and` paths.
- LM scales 0.3/1.0 and word penalties 0/-1, checking EOS and once-per-word
  accounting against independent whole-string HF teacher forcing.
- Greedy casing selection with the final word selected jointly with EOS,
  checked against an independent HF oracle for the same greedy policy.
- Unequal-prefix-length, padded continuation requests versus separate forwards.

Native total-cost tolerance is 0.01 nats; per-token batch tolerance is 0.002.
Each of the eight native validation cases runs in its own fresh subprocess:
FLF's lexicon is process-global and asserts when a second search instance tries
to create it, even after deleting the first instance. The parent aggregates
per-case JSON reports and logs and fails if any child fails. Full pretrained
validation is still required; process isolation does not relax numerical gates.
Local tests passed the native zero-scorer word-reward/CTC-collapse case on the
available binary and tiny-Qwen continuation indexing. These are **not** a claim
that the new pinned CMake job/full pretrained gate has completed.

## Initial dev grid

Ten conditions, with final and pre-LLM score thresholds both 14:

| LM scale | Word penalty | Final beam | Pre-LLM beam | Casings |
| --- | --- | --- | --- | --- |
| 0.1, 0.3, 0.5 | 0, -1 | 16 | 64 | original/lower/capitalized/upper |
| 0.3 | 0 | 16 | 64 | lower only |
| 0.3 | 0 | 32 | 128 | original/lower/capitalized/upper |
| 0 | 0 | 1, 16 | 1, 16 | zero-cost scorer |

Negative word penalty is a reward, outside the LM scale. Zero-scale controls
use a lightweight zero-cost scorer rather than executing Qwen unnecessarily;
they additionally save direct acoustic greedy text/parity diagnostics.
All conditions request 24 GB GPU memory / 24 GB host memory and 24 hours.
Report WER/I/D/S, normalized per-segment results, runtime and actual Python LLM
request/batch/max-context counts. No per-step RASR statistics are enabled.

Outputs are under `output/librispeech/qwen-dynamic-9804f5aa/dev-other/`.
The pinned build is under `output/librispeech/tools/rasr-llm-9804f5aa/` when
registered in the LibriSpeech corpus context.

## Limits of the prototype

- Formatting is a greedy choice at each word, not a joint search over all
  capitalization histories. This can discard a casing preferred by later words.
- Punctuation must come from the acoustic model. No unobserved punctuation is
  inserted. The n-best style diagnostics are therefore complementary, not
  numerically identical to this search.
- Wordwise Qwen tokenization is checked on the native gate examples, not
  guaranteed to equal whole-string tokenization for arbitrary punctuation or
  unusual spellings. Inspect differences rather than silently normalizing them.
- Word LM costs remain deferred until closure. Word merging and pruning bias
  remain possible; there is no look-ahead in this prototype. Wider pre-LLM/final
  beams are diagnostics, not a proof of search correctness.
- RASR's history trie grows within a segment and is reset at segment start.
  Long-context or request-count explosions remain a runtime risk. No silent
  history truncation is introduced to hide them.

Keep the n-best first-pass/oracle/Qwen results alongside these conditions so we
can distinguish Qwen scoring weakness from in-search candidate/pruning losses.
