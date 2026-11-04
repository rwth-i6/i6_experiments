# ASR/Sisyphus test audit

Reviewed 2026-11-04 against `berger_dev` at `f2378df6e`. Scope: all 30
research test files plus `tests/test_rasr_templates.py`: 220 test definitions,
5,577 lines excluding the tests package initializer. The fresh combined run
expanded parameterizations to 257 cases: **254 passed, 3 skipped in 17.14 s**.
No tests, model runtime, experiment settings or running jobs were changed.

## Judgment

Tests are useful here when they protect a mathematical/model interface,
scientific comparison, data transformation, or Sisyphus dependency/cache
contract. They do not need a training run to be valuable: a ten-line error in
score indexing or reference grouping can invalidate days of inference.

The current suite also contains too many tests written around individual
experiment batches. Exact old grids, variant counts and tuned parameters are
usually duplicated configuration, rather than evidence that inference works.
I would substantially reduce that part.

My manual triage of the 220 definitions is:

| Decision | Definitions | Meaning |
| --- | ---: | --- |
| K: keep | 122 | A useful behavior or failure case is exercised; retain its stated scope |
| R: rewrite/consolidate | 70 | Preserve a useful intent, but remove snapshot noise, strengthen weak mocks or consolidate overlapping cases |
| A: retire/archive from routine tests | 28 | Historical tuning snapshots, one-time migration assertions or exact duplication |

These counts are review judgments, **not coverage percentages**. A mixed test
can contain both a valuable dependency assertion and twenty unhelpful settings
assertions. Parameterized cases also do not represent independent capabilities.
The per-definition decisions/reasons are in [test_audit.tsv](test_audit.tsv).

## What meaningfully works already

### Model/score interface checks

- `test_dynamic_qwen.test_continuation_positions_padding_and_order` executes a
  real, tiny randomly initialized Qwen model. Batched continuation costs are
  compared to a separate full forward/cross-entropy computation across unequal
  prefix lengths and an empty continuation. This can catch shifted target
  indices and incorrect padding/order handling.
- `test_open_asr_benchmark.test_adapter_matches_full_model` uses tiny real
  Cohere and Parakeet models and compares adapter scores to full-model scores.
  It covers the AED prompt/BOS policy and RNN-T encoder/predictor indexing.
- `test_qwen_speech.test_audio_embeddings_and_variable_length_teacher_forcing`
  uses a tiny real multimodal model, checks audio-prompt embedding reuse against
  full forwarding, and verifies that changing audio changes costs. This guards
  against accidentally scoring a text-only model or stale audio.
- `test_qwen_fusion.test_tiny_qwen_cached_onnx_matches_huggingface` really exports
  and runs ONNX models, compares against tiny HF models, and injects NaNs/bad
  alias masks. It includes nonstandard head dimensions. It is much stronger
  than inspecting an export Job's fields.
- The last-hidden projection test checks both numerical equality and the
  tensor shape *before* the vocabulary projection. This protects the memory
  optimization that avoids projecting every position. Keep one copy: it is
  currently duplicated in the template and wordwise suites.
- Prior and ILM fixtures independently check the sign, nonblank normalization,
  blank invariance and history dependence. Reward/UNK-bias fixtures check the
  scoring equation at different LM scales and exclusion of EOS where intended.

### Data, scoring and analysis correctness

- Hand-written fixtures check CTC repeated labels separated by blanks, UTF-8
  byte fallback, punctuation/word boundaries, label ordering, safe pronunciation
  transformations and boundary-independent emission-index validation.
- Actual scoring Jobs are run on tiny files with known I/D/S and denominators,
  including empty hypotheses, grouped chunks and independent utterances. They
  exercise the installed alignment library, not a mocked WER calculator.
- Tests reject missing/duplicate sequence coverage and mismatched audio/raw
  references/normalizer provenance. These guard invalid comparisons rather
  than recognition itself. Audio-origin checks compare preparation provenance;
  they do not compare waveform samples.
- Actual plain/gzip subset IO is checked, including selecting the dataset
  before recognition. This prevents both scoring mismatches and unnecessarily
  decoding the full corpus for a development subset.
- Controlled-selection/complementarity fixtures check dev/test separation,
  missing selected conditions, better-*standalone* comparisons rather than an
  oracle baseline, paired error attribution and incompatible reference grids.
- The periodic-report test calls Sisyphus's actual update callback while a
  dependency is absent, then present. That directly reproduces the earlier
  missing-report warnings and is useful infrastructure coverage.

### Native RASR and preflight checks

`test_open_asr_benchmark.test_native_rasr_smoke` writes a config, starts the
actual installed librasr in a subprocess and decodes forced CTC/AED paths,
including repeated labels, tree versus lexicon-free output and the AED partial
sequence at its cap. These four cases **ran** in this audit. The dynamic
zero-LM/word-penalty test and wordwise CTC/RNA gate also execute native code.

There are valuable Sisyphus validation Jobs outside pytest as well:
`DynamicQwenGateJob`, `RnntLlmGateJob`, the n-best gate, OOV decoder/pruning
probes and acceptance Jobs. Actual native forced-path/model-score checks should
remain prerequisites of expensive recognitions and use the same binary/model/
vocabulary. A small generic test that confirms that dependency is connected is
worth keeping.

However, `test_real_decoder_smoke_graph` only constructs the cases and calls
`compile()` on runner text. `test_cases_cover_multitoken_scores_history_and_finalization`
examines case definitions and expected costs. Neither executes RASR. Likewise,
`test_nbest_scaled_costs_deduplication_and_parity` validates Python packing of
synthetic tracebacks; the native n-best gate establishes the RASR-side parity.
Test names and reported coverage should make these distinctions explicit.

## What to remove or change

1. **Old tuning-grid snapshots.** Large parts of `test_oov_length`,
   `test_dynamic_qwen_transfer`, size/next-round tests and the large development
   matrix checks assert counts such as 25/40/116, exact beam/scale products or
   a particular winning point. Updating the experiment then requires updating
   an almost identical test. Move frozen protocols into experiment provenance;
   retain general invariants such as dev-only tuning, unique descriptors, shared
   data/candidates, compatible LM vocabulary and same-binary gates.
2. **Literal source hashes and job IDs.** They had value during a targeted
   migration or selective invalidation fix. A permanent test should usually
   check a *relationship*: a scoring-only change leaves inference identity
   unchanged; a result-affecting option changes identity; a time-limit-only
   change preserves it. It should not demand one historical digest forever.
3. **Source-manifest freeze.** `test_research_source_manifest_is_complete_and_unchanged`
   is a one-time relocation audit. Keeping it as an ordinary regression test
   makes intentional edits to most runtime files fail. Keep the manifest and
   1,197-job comparison as migration evidence, plus lasting import/worker
   compatibility tests; remove the blanket source freeze from the default suite.
4. **Weak or overclaimed mocks.** The template Qwen scorer test returns uniform
   fake logits; it cannot detect selecting the wrong output column or last
   position. The fake cached/stateless LM sessions return the same uniform
   probabilities and do not demonstrate real cached numerical equivalence.
   Use nonuniform known probabilities for guard/index tests and real tiny
   models/exports for numerical-equivalence claims.
5. **Missing distinguishing examples.** The test named
   `test_selection_is_cost_minimization_not_token_average` checks ranking, but
   its candidates do not provide different token lengths. It does not isolate
   the averaging error named in the test. Add a case where sum and mean costs
   deliberately choose different candidates, including EOS and retained LM cost.
6. **Formatting and implementation snapshots.** Exact Markdown substrings,
   ten-report counts, 300-second refresh values, AST scans for particular call
   names and assertions about a physical module filename are brittle. Preserve
   semantic report tests—especially matching the correct Speech-Qwen size and
   normalizer—but check structured result rows and a small rendering smoke test.
7. **Incomplete serialization coverage.** Pickling a class already loaded in
   the same process, or directly calling state methods with `object()` fields,
   does not test reconstruction of a real job/native object in a fresh worker.
   Keep import identity checks; add an actual worker/spawn round-trip.

Hardcoded expected values are not inherently bad. A known blank index, a hand-
computed cost, two errors in a fixture, or a required API field can encode a
real contract. A chosen beam size or an old Cartesian sweep usually encodes a
temporary experimental choice. Mixed tests need to be split along that line.

## Important remaining gaps

- No full real-checkpoint audio-to-ASR/RASR-to-scoring smoke for each current
  model/search/container combination. Tiny CPU models do not establish correct
  real tokenizer behavior, waveform preparation, precision or transcript style.
- No direct regression for the actual `NormalizeStmCtmJob` file conversion,
  despite the earlier speaker/time-column and STM-label failures. JSON benchmark
  scoring and lexicon-normalization tests do not cover that file pipeline.
- No direct tests here for `CountLexiconOovJob`,
  `ReduceLexiconToTargetOovJob`, `BoostArpaUnknownProbabilityJob` or
  `CollapseOovScoresInOnnxJob`. In particular, an actual tiny ONNX session should
  verify merged UNK probability mass, suppressed removed outputs, unchanged
  known-word scores and correct history/output vocabulary mapping.
- No focused regression for streaming feature dtype/shape conversion or
  chunk/frame-shift accounting. Inspecting the chosen data-config fields does
  not prove the actual waveform or emission timing is equivalent.
- CPU ONNX tests cannot expose missing CUDA libraries/provider fallback in the
  actual Apptainer/GPU environment. GPU memory growth, long-history stalls and
  RTF also require targeted pilot Jobs on the intended hardware/container.
- Most graph tests inspect in-memory RasrConfig objects. They cannot prove
  that the emitted INI syntax/key spelling is accepted by the selected binary.
  Native smoke/preflight should materialize and load the actual search configs.

The audit run skipped the real Granite-tokenizer traceback test (local asset
unavailable), native casing-variant route test (`RASR_LIB_DIR` unset), and
external scheduler-settings test. Make paths explicit fixtures/arguments.
Skipping an optional local integration check is fine for the fast suite;
requesting an expensive model run should require its applicable preflight to
finish successfully, rather than count a skip as validation.

## Recommended ongoing structure

1. **Fast behavior tests:** score math, small real CPU model parity, vocabulary/
   lexicon transforms, scoring IO, provenance and analysis. Allow focused mocks
   with explicit scope. Consolidate duplicate fixtures and tests.
2. **Small graph-contract suite:** active families only, dev/test separation,
   input/control/candidate sharing, valid scorer/pruning-vector dimensions,
   matching binaries/vocabularies, unique outputs and semantic cache behavior.
   Parameterize meaningful input changes instead of freezing historical grids.
3. **Sisyphus preflight Jobs:** materialized configs, selected binary ABI,
   actual tokenizer/checkpoint and target container/provider, tiny forced-path/
   reference-score parity and a few real development utterances. Cache against
   their actual dependencies and gate expensive jobs on their outputs.
4. **Separate performance pilots:** include long inputs/history and memory/RTF
   measurements before full-suite scheduling. Do not expect pytest to certify
   WER, RTF <1.5 or stability of a 100-hour dataset run.

A meaningful test need not reimplement the whole decoder. It needs a failure
case that could fool the real experiment and an expectation established by an
independent calculation, a trusted computational path, or an explicit lasting
interface/data contract. If changing an ordinary tuning choice is the only
thing that makes a test fail, it belongs in provenance rather than the routine
correctness suite.

## File-by-file triage

K/R/A counts are definitions, not parameterized cases. See the TSV for exact
test names and reasons; R means retain the useful purpose while rewriting it.

| File | Keep | Rewrite/consolidate | Retire/archive |
| --- | ---: | ---: | ---: |
| `test_rasr_templates.py` | 3 | 3 | 0 |
| `test_audio_provenance.py` | 4 | 0 | 0 |
| `test_baselines_entrypoint.py` | 1 | 5 | 1 |
| `test_controlled_comparison.py` | 11 | 1 | 0 |
| `test_dynamic_qwen.py` | 3 | 2 | 0 |
| `test_dynamic_qwen_transfer.py` | 2 | 0 | 4 |
| `test_error_complementarity.py` | 5 | 1 | 0 |
| `test_hf_lexicon_normalization_aliases.py` | 7 | 0 | 1 |
| `test_hf_lm_text_normalization.py` | 5 | 1 | 1 |
| `test_job_reports.py` | 2 | 0 | 0 |
| `test_migration.py` | 2 | 2 | 2 |
| `test_nbest_rescoring.py` | 2 | 2 | 0 |
| `test_oov_length.py` | 10 | 9 | 11 |
| `test_oov_multitoken.py` | 1 | 2 | 0 |
| `test_oov_neural_lm.py` | 2 | 2 | 0 |
| `test_oov_prototype.py` | 5 | 4 | 3 |
| `test_open_asr_benchmark.py` | 14 | 8 | 1 |
| `test_pretrained_correction.py` | 4 | 1 | 0 |
| `test_qwen_benchmark.py` | 3 | 1 | 0 |
| `test_qwen_full_dev.py` | 1 | 0 | 1 |
| `test_qwen_fusion.py` | 6 | 2 | 0 |
| `test_qwen_nbest_diagnostics.py` | 2 | 0 | 0 |
| `test_qwen_next_round.py` | 3 | 1 | 1 |
| `test_qwen_runtime_and_earnings.py` | 1 | 3 | 0 |
| `test_qwen_speech.py` | 2 | 1 | 0 |
| `test_research_diagnostics.py` | 8 | 0 | 0 |
| `test_result_collection.py` | 0 | 7 | 0 |
| `test_rnnt_qwen.py` | 4 | 4 | 1 |
| `test_speech_qwen_round.py` | 4 | 4 | 0 |
| `test_speech_qwen_sizes.py` | 0 | 3 | 0 |
| `test_wordwise_transformer.py` | 5 | 1 | 1 |
