# Upfront-beta pruning fix: 3ed6d27b

Historical setup: the active entrypoint has moved to
[6e07a2e5](OOV_6E07A2E5.md), which adds the first-piece pruning regression.
The description below records the preceding experiment stage.

The active staged experiment entrypoint is `oov_length.py`. It pins RASR to
`3ed6d27b7ca628d3a6b46ff2ac65033f16d16810` plus the previously approved XML
separator patch. Source, build and recognition hashes are separate from both
5083edde and 101ed636; neither historical build getter was changed.

## Why this revision

The 101ed636 probe returned the expected seven DOGs at cost 35 in the closed
tree and at threshold 200, but empty output in the open tree at threshold 16,
even with beam 4096. A pending fallback word deferred beta, set an artificially
cheap pruning reference and eliminated the ordinary path. Charging beta on
opening addresses that timing error without changing completed-word scores.

The completed Granite boundary diagnostic measured a mean marked/unmarked
margin of 13.920 nats (315 comparable starts in 32 utterances), versus 1.546
scaled LM nats per word. This does not support the proposed cheap-boundary
scoring explanation. These are forced-reference, same-frame comparisons, not
a proof covering every alternative tokenization or alignment.

Granite now uses **alpha = 0**, including in beta sweeps. Nonzero Granite
`alpha0_by_lm` values are rejected to catch stale experiment calls. The old
alpha0 = 0.326898 remains a diagnostic, not a Granite setting. BPE/transducer
alpha exploration remains available with an explicitly supplied calibration.

## What the default entrypoint schedules

- A new pinned CMake build with the same build options as 101ed636.
- The 34-case experiment-side native decoding/accounting smoke suite.
- The unchanged four-condition beta=100 pruning probe, plus a hard acceptance
  job requiring **all four** transcripts and costs to match the closed control.
- A fresh real-Granite state-tying dump and validation on the new binary.
  All four boundary-flag combinations for every model label must map to that
  label's model output index (16,384 labels / 65,536 checks). Tying is monophone,
  one state per phone, no phonetic context. Each recognition lexicon is also
  individually checked before recognition.
- A small Approach A comparison on dev-other: closed versus top-100k
  training-corpus candidates, no LM versus 4-gram scale 0.3, beams 256 and 1024,
  threshold 16, beta 0, plus a same-build lexicon-free control (nine recognitions).
  Candidates come only from LM training text, excluding the existing lexicon
  and ARPA vocabulary. No evaluation words are used for extension.

Approach A contains ordinary lexical entries and no fallback pieces; it does
not wait for the pruning probe. It still depends on the smoke suite and its
own state-tying/lexicon validation. The candidate/preparation reports expose
how many proposed entries survive tokenizer validation.

The probe has now passed all four conditions at cost 35, along with all 34
integration cases and 65,536 Granite boundary-index checks. The completed
Approach A comparison at beam 256 gave 2.850% -> 2.771% without LM and
2.722% -> 2.689% with the 4-gram; the same-build greedy control gave 2.817%.

The default entrypoint now additionally calls `run_limited_fallback`, registering
**17 new dev-other recognitions** on the standard (not reduced) lexicon:

| Model | Word LM | Fallback beta | Alpha | New runs |
|---|---|---|---|---|
| BPE CTC | 4-gram, scale 0.6 | 10, 12, 20, 100 | 0 | 4 + 1 closed control |
| Granite CTC | none | 0, 2, 4, 8, 16, 100 | 0 | 6 |
| Granite CTC | 4-gram, scale 0.3 | 0, 2, 4, 8, 16, 100 | 0 | 6 |

Every fallback run uses `known-excluding`, beam 256, score threshold 16,
word-end beam 256 / threshold 16, prior and blank penalty zero, no word reward,
Viterbi recombination and `sentence-end-fall-back=false`. The beta-only phase
fixes alpha=0 for BPE too, isolating the beta timing change from per-piece costs.
The BPE control uses the identical XML with fallback disabled; the existing
Granite closed and greedy controls remain registered by `run_approach_a()` and
are not duplicated. Granite's new condition is Approach B, without extension.

The beta=100 endpoints test whether real-utterance results approach the closed
controls. Check empty/merged hypotheses, IV/OOV errors and paired regressions,
not just aggregate WER. High-beta equality is a diagnostic expectation, not
a guaranteed property of finite-pruning search with delayed refunds.

All fallback lexica retain the acceptance-report dependency, their individual
state-tying validation and the runtime degeneracy guard. Broad alpha sweeps,
reduced lexica, adapted LMs, Approach C, transducers and test-set runs remain
opt-in; no subword LM is added.

The AM margin job retains its original 101ed636 smoke dependency, preserving
the completed `GraniteWordStartMarginJob.RiigwjieBGmV` result. The AM, data and
measurement are unchanged; a decoder revision is not a reason to redo that
GPU measurement. The historical scoring audit is also reused.

## Outputs to inspect

Under `output/librispeech/`:

- `tools/rasr-oov-3ed6d27b/decoder-smoke-report` and `decoder-smoke-logs`
- `tools/rasr-oov-3ed6d27b/delayed-penalty-probe` and `delayed-penalty-logs`
- `tools/rasr-oov-3ed6d27b/pruning-acceptance` (must say `passed: true`)
- `datasets/LibriSpeech/oov-3ed6d27b/granite-boundary-indices/validation`
- `datasets/LibriSpeech/oov-3ed6d27b/extension-candidate-report`
- `datasets/LibriSpeech/oov-3ed6d27b/A-100k/preparation`

Recognition descriptors begin `oov-3ed6d27b_`; no new WER should be interpreted
using an old-build closed baseline.

## Test-count clarification

The 34 checks are **Python-driven librasr integration cases in this experiment
repository**, not RASR's C++ unit-test count. They include CTC repeat collapse,
known refunds, ordinary UNK entries, separators, refunds after entering the
Granite fallback region, BPE legacy beta and four RNA cases. Their runner and
configs live in `data/oov_length_smoke.py`, extending `data/oov_decoder_smoke.py`.
No unshared C++/RASR changes were made, and nothing was pushed upstream.

LM look-ahead is not merged in this round. A passing toy probe establishes a
regression fix, not full-corpus search equivalence; keep the no-collapse guard,
IV/OOV analysis and same-build controls for subsequent fallback experiments.
