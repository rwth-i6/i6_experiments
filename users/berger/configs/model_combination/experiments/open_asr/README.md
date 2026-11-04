# Open ASR: native vs. lexicon-free vs. closed-tree recognition

The active suite is registered by `users.berger.configs.baselines.main()` through
`users/berger/configs/baselines/open_asr.py`, alongside the unchanged existing
LibriSpeech/Loquacious runs. It keeps its own `open-asr/` output namespace.
Nothing has been submitted to Slurm by editing these files.

The baseline wrapper currently calls
`run_all(neural_lms=False, aed_tree=False, granite_open_tree=True,
granite_open_transformer=True, corpus_open_tree=True)`: Qwen and untuned Transformer-LM conditions are held back until their pruning/runtime is tuned on
development data. Native, lexicon-free, closed-tree, 4-gram and corpus-resource
conditions remain active for Granite. Cohere AED is evaluated only with native
and lexicon-free decoding while label-sync tree search is backlogged (see below).
The active grid contains 127 recognition jobs (67 existing plus 32 Granite
open-tree/matched-control runs, 16 tuned Transformer comparisons and 12
corpus-resource extended-tree/control runs). It includes
the validated GPU/stateless Transformer export; Qwen dependencies are omitted. Existing
outputs are retained; already submitted/running scheduler jobs are not cancelled.
After tuning and fixing AED tree search, re-enable the full matrix with
`neural_lms=True, aed_tree=True` in the wrapper.

Stepwise RASR statistics are disabled at job execution, including for reused
configuration files. This logging-only override preserves the completed
benchmark job identities. Shared offline/streaming forward steps apply the same
runtime override. Config-builder defaults retain their historical values to
avoid registering all old recognition jobs under new hashes.

For a standalone Sisyphus config, the suite can also be called directly:

```python
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import run_all

run_all(neural_lms=False, aed_tree=False, granite_open_tree=True,
        granite_open_transformer=True, corpus_open_tree=True)
```

For a staged first run in a standalone config (or in the baseline wrapper):

```python
run_all(model_names=("granite-ctc",),
        dataset_names=("librispeech-test-other",), corpus_resources=False,
        neural_lms=False, aed_tree=False, granite_open_tree=True,
        granite_open_transformer=True, corpus_open_tree=True)
```

The subset switches select jobs, not hyperparameters. The summary deliberately
does not compute an eight-dataset macro average from an incomplete subset.
Outputs are registered under `open-asr/`, including `open-asr/results.json`.
`open-asr/resource-overview.md` lists every registered condition/dataset with
explicit lexicon and LM sources, WER and RTF, including corpus-specific rows.
Like the existing aggregate JSON it runs once its score dependencies complete;
individual condition outputs remain available before that.
Independent reports at `open-asr/overviews/<model>/<dataset>.md` depend only on
that model/dataset's conditions, so failed AED runs do not block Granite results.

## Matrix and implementation status

| Model | Native HF | RASR lexicon-free | LS tree | + LS 4-gram | + LS Transformer | + Qwen Base |
|---|---|---|---|---|---|---|
| Granite Speech 5.0 470M TurboCTC NC | configured | configured | configured | configured | configured | configured |
| Parakeet RNN-T 1.1B | configured | configured | pending RASR | pending RASR | pending RASR | pending RASR |
| Cohere Transcribe 03-2026 | configured | configured | configured | configured | configured | configured |

Each configured column runs on all eight public English short-form datasets:
LibriSpeech test-clean/test-other, AMI cleaned, GigaSpeech cleaned, SPGISpeech,
VoxPopuli cleaned AA, Monsoon English India, and Earnings22 cleaned AA chunked.
This list follows the current benchmark submission script, not the older ESB
Common Voice/TED-LIUM list. Model and dataset revisions are recorded in the code.

The full matrix (`neural_lms=True, aed_tree=True`) contains 112 primary recognition jobs, plus
six corpus-resource comparisons (118 total), with scoring and native/RASR
comparison jobs in addition. Neural-LM columns in the table above are currently
disabled in the baseline entry point; they remain available for tuning.

Completed pretrained-model results and failures are recorded in
[the 2026-10-03 audit](results_audit_2026-10-03.md). CPU tests compare the new AED and RNN-T scorer adapters against full
forwards of small randomly initialized HF models. Native-library smoke tests
exercise the pinned CTC/AED lexicon-free and tree searches with synthetic scores.
Cohere's actual tokenizer/lexicon coverage remains a runtime validation because
its checkpoint is gated. RNNT tree search is not substituted with RNA or TDT.

## Backlog: AED label-sync tree search

Cohere tree and tree+LM conditions, including corpus-specific trees, are held
back. Investigate score accounting, pruning and EOS termination before enabling
them again. Counterexamples include Monsoon tree+4gram WER 33.01% versus 3.43%
native and Earnings22 tree+4gram WER 41.54%; recognized text contains repeated
generic phrases rather than just missing OOV words. Two tree+4gram runs have no
completed EOS path, and other tree runs hit the token limit with repetitions.
Use short reproductions with native/free/tree comparisons and per-scorer costs.

Native and lexicon-free AED also hit the 500-token cap on matching GigaSpeech
and Earnings22 samples. Their consistent benchmark token-limit handling is a
separate outstanding issue; disabling trees does not repair those failed runs.

## Granite extended-tree evaluation

`granite_open_tree=True` adds four conditions to each of the eight datasets:

| Condition | Fallback | LM scale | Beta | Alpha | Beam / word beam | Threshold / word threshold |
|---|---|---:|---:|---:|---:|---:|
| tree-open-control | disabled | 0 | — | — | 256 | 8 |
| tree-open | known-excluding | 0 | 1 | 2 | 256 | 8 |
| tree-open-control-4gram | disabled | 0.3 | — | — | 64 | 16 |
| tree-open-4gram | known-excluding | 0.3 | 6 | 2 | 64 | 16 |
| tree-open-control-trafo | disabled | 0.3 | — | — | 64 / 16 | 16 |
| tree-open-trafo | known-excluding | 0.3 | 4 | 2 | 64 / 16 | 16 |

The two Transformer conditions require `granite_open_transformer=True` as well
as `granite_open_tree=True` and are independent of the broad `neural_lms`
switch. They use the original LibriSpeech word Transformer on GPU with
full-history/stateless inference and LM batch size 1 to bypass the speculative
history queue. Beam 64 / word beam 16 was selected on dev-other: 1280 errors,
total RTF 0.393, versus 1281 errors and RTF 3.801 at 256/256. Both thresholds
remain 16. The closed Transformer control uses the identical prepared lexicon
and pruning. Qwen and AED trees remain disabled in the active entrypoint.

These settings come from LibriSpeech dev-other tuning. No benchmark test-set
tuning or reference-derived vocabulary is used. All four use the same prepared
LibriSpeech lexicon, with per-token word-start/internal fallback lemmas and
unchanged acoustic labels. The builder filters unsupported multi-boundary
pronunciation variants and reports every removal; matched closed controls use
the identical prepared file, isolating fallback from preparation and pruning.
Compare these controls with the earlier closed runs before attributing gains.

The pinned patched RASR 8c1ac793 build is shared with the existing closed-tree
suite. Recognition depends on tokenizer classification, decoder smoke and
pruning acceptance, finite UNK probability, and boundary-independent monophone
emission-index validation. Both LM conditions retain the original 4gram; there
is no LM adaptation or subword LM in this addition; the optional Transformer
pair adds only a word LM.

Marked piece runs are detokenized together with Granite's actual tokenizer,
including UTF-8 byte fallback. Raw marked traceback is retained in hypotheses;
`fallback_piece_count` and `max_fallback_word_pieces` are route diagnostics,
not counts of true OOVs or observed UNK events. Existing native/closed/AED/RNNT
job identities and results are reused. Scoring and RTF use the usual pipeline.

### Corpus-resource extended-tree comparisons

`corpus_open_tree=True` adds six Granite runs per resource-bearing dataset:
open/closed pairs without an LM, with a 4gram, and (when enabled) with the tuned
word Transformer. The LibriSpeech-resource runs above remain as the comparison.

| Dataset | Corpus lexicon | 4gram for corpus-lexicon runs | Transformer |
|---|---|---|---|
| GigaSpeech | GigaSpeech | GigaSpeech public ARPA | LibriSpeech (transferred) |
| VoxPopuli | VoxPopuli | LibriSpeech (transferred; public KenLM binary unavailable to this loader) | LibriSpeech (transferred) |

Only GigaSpeech has both its own usable lexicon and n-gram in this setup. We do
not claim a corpus-trained Transformer for either dataset. Condition names end
in `corpus4gram`, `ls4gram`, or `ls-trafo` to make that distinction visible.
These are transferred LibriSpeech tuning settings, not corpus-specific optima.
GigaSpeech fallback validation checks finite UNK probability against its actual
ARPA, whose unknown spelling is `<UNK>`. Existing corpus closed-tree outputs
are retained, and the newly matched closed controls share each fallback lexicon.

CTC/AED use the existing separately pinned multi-token RASR build at
`8c1ac793675fb7aba0198cc165dca7a2e624375d` with the approved XML separator patch.
RNNT lexicon-free gets an isolated CMake build at
`59767f21727326691de3812affac4c4900449c65` from the existing RNNT search branch.
After a tree-capable RNNT merge is provided, its binary and tree configuration
must be added explicitly and its lexicon-free control rerun on that build.

## Controlled conditions

- Identical saved HF audio for all conditions; processor preprocessing only.
  No extra preemphasis, peak normalization, or lossy transcoding.
- Float32, one audio segment at a time, TF32 matrix multiplication disabled.
  Native HF decoding and RASR use the same model/processor revisions.
- CTC native argmax/CTC collapse versus RASR beam 1. RNNT and AED also start
  with beam 1 for the lexicon-free comparison; repeated non-CTC symbols survive.
- Cohere's English/no-punctuation prompt, any HF-prepended decoder-start token,
  and generation-time token policy are reused in the RASR scorer. The native
  token cap is 500; a missing EOS at the cap is flagged, not silently scored as
  a successful complete decoding. Cohere's internal audio chunks are processed
  sequentially using its original features, masks and chunk indices, then joined
  using the processor's English reassembly policy. Native and RASR reset decoder
  history (and external LM history for RASR) for each chunk. Scoring still uses
  one hypothesis per original dataset segment. Multi-chunk outputs retain raw
  token paths and chunk-local tracebacks under `chunks`; their times must not
  be interpreted as absolute utterance times.
- Closed trees have beam 32, score threshold 12, the same word-end limits, and
  no OOV fallback. Acoustic priors, blank penalties and word rewards are zero.
  The complete model label inventory is retained in output-index order, with
  one-state monophone tying independent of word-boundary flags. AED EOS is an
  acoustic label; its prompt is not an epsilon tree exit.
- All external LM scales start at 0.3. These are **frozen initial values**, not
  claimed optima for Cohere or the other corpora. Do not tune on these test sets.
  Establish native/RASR agreement first, then choose future settings on dev data.
- Word Transformer uses the existing LibriSpeech checkpoint, stateless GPU
  inference and LM batch size 1 (the retained-history queue workaround).
  Qwen3-0.6B-Base reuses the correctness-gated ONNX export, capitalization/spacing
  variants and initial-position aliases, also with LM batch size 1. Neither LM
  is retrained; no second/subword LM is combined with it.
- AM scoring for RNNT/AED is initially a Python scorer with bounded full-history
  GPU batches. It is a correctness-oriented integration, not an optimized decoder.

The native Parakeet baseline uses **Transformers**, not NeMo. It is consequently
not advertised as bit-identical to historical NeMo leaderboard submissions.
In particular, native RNNT greedy emission limits and beam-search forced-blank
handling can differ. Likewise, full-history versus cached AED computation can
introduce numerical differences. The per-segment comparison reports preserve
raw token IDs, raw/normalized text differences and RASR traces so these can be
investigated instead of hidden behind rounded WER.

## Scoring and provenance

Every hypothesis ID must occur exactly once and cover the reference IDs exactly.
The existing updated HF English normalizer and compound-boundary equivalence
are shared across all conditions, with `kaldialign` error counts. Reference-only
empty-text filtering happens once during data preparation.

Only Earnings22 is reconstructed at call level, using its explicit parent IDs,
chunk indices, and separately pinned parent references. All other datasets are
scored utterance by utterance. The full macro average is the unweighted mean of
the eight corpus WERs, not a pooled word error rate.

Registered results include hypotheses, exact I/D/S and reference counts,
per-scoring-unit results, runtime/RTF, and native/RASR differences. Recognition
logs flush segment progress. Runtime reports record the Transformers/Torch
versions, dtype and GPU; runtime source content enters the new job hashes.

## Public corpus-specific resources

GigaSpeech gets its own word-list-only tree and its own tree + 4-gram for CTC
and AED. The [public GigaSpeech LM repository](https://huggingface.co/wgb14/gigaspeech_lm)
is pinned at `57147dfe03d40315c0ea62cb8b22e9f9d3f13329`; only `lexicon.txt` and
`4gram.arpa.gz` are downloaded. The [icefall recipe](https://github.com/k2-fsa/icefall/blob/master/egs/gigaspeech/ASR/prepare.sh)
uses this resource. Its ARPA unknown spelling was checked as `<UNK>`.

VoxPopuli gets an additional corpus-word-list tree. The [official resource page](https://github.com/facebookresearch/voxpopuli#asr)
provides the English lexicon and 3-/5-gram **KenLM binaries**, not ARPA files.
The lexicon download has a checked SHA256. These binaries are not passed to
RASR's ARPA loader: a compatible backend or an ARPA release is still needed
before adding VoxPopuli LM conditions.

Both public word lists retain their own capitalization/LM spellings; their
original phonetic pronunciations are replaced with each AM's tokenizer pieces.
No evaluation-reference words are added to any lexicon.

AMI has a [public Kaldi LM-training recipe](https://github.com/kaldi-asr/kaldi/blob/master/egs/ami/s5/local/ami_train_lms.sh),
but a standalone downloadable ARPA was not verified. No corpus-specific
pretrained ARPA was verified for SPGISpeech, Earnings22 or Monsoon. Those sets
currently get the six LibriSpeech-resource comparisons only (two for RNNT).

## Runtime requirements

Use the CUDA/ONNX Apptainer environment used by the existing RASR jobs, with
Python 3.12 matching `librasr`. This adapter is tested with and explicitly
requires **Transformers 5.16.1** because it uses generation-preparation APIs.
Also install `kaldialign==0.12.0`, `datasets`, `huggingface_hub`, `sentencepiece`,
and the compatible audio decoder dependencies (e.g. torchcodec/FFmpeg for the
new datasets AudioDecoder interface). The existing Qwen/Transformer export
dependencies are unchanged. No job installs packages into a shared environment.

[Cohere](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) and some
datasets require accepted HF access terms. Provide the existing authenticated
HF cache/token through the job environment; credentials are not put into job
arguments, hashes or logs. Check access before scheduling the large matrix.

Tests:

```bash
PYTHONPATH=..:../sisyphus pytest -q tests/test_open_asr_benchmark.py
```
