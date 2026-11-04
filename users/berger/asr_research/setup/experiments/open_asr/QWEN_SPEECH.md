# Qwen3-ASR baseline and Granite combination

Checkpoint: `Qwen/Qwen3-ASR-0.6B-hf`, revision
`7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`. This is the official
Transformers-native conversion of Qwen3-ASR-0.6B, not Qwen3-0.6B-Base.
Reference: https://huggingface.co/docs/transformers/model_doc/qwen3_asr

The baseline entrypoint registers ten standalone runs and thirteen fusion runs:
the eight Open-ASR corpora and Loquacious dev/test, plus three additional
development-only Loquacious fusion scales. Dataset/audio preparations are reused.
Scoring uses the same updated HF normalizer as the other benchmark conditions.

Standalone recognition uses the official English-prefilled transcription prompt,
FP32, greedy generation and a 500-generated-label limit. Partial hypotheses are
scored without forcing EOS. No reference text or vocabulary hotwords enter the
prompt. Outputs are under `open-asr/qwen-speech-0.6b/native`.

Fusion is lexicon-free dynamic-word Granite CTC search. It combines Granite
acoustic cost with `scale * [-log P(text | audio)]` from the Speech-LLM, at word
boundaries, using each model's own tokenizer. This is a combination of two
audio-conditioned recognizers, **not** a text-only external LM experiment.
Granite score scale is 1; there is no additional ARPA/Transformer LM.

Each fusion job depends on its standalone baseline's completed scoring summary.
The Speech-LLM encodes audio once per segment and retains its embeddings while
the decoder recomputes full histories in GPU batches of eight without KV caching,
matching text-Qwen's maximum batch size. Earlier batch-two fusion results remain
available; native baseline identities are preserved by a reviewed migration.
Before the first segment is decoded, continuation scores (including EOS and
unequal batch lengths) must agree with independent full multimodal forwarding.
Failures stop the job; `validation` is a registered output.

Search starts with final/pre-LLM beams 32/128 and thresholds 6/6. The transfer
scale is 0.3; Loquacious dev also sweeps 0.05, 0.1 and 0.5. Token reward and word
penalty are zero rather than transferring tuned pure-LM rewards blindly.
Future Granite fusion runs preserve the AM's hypothesized capitalization,
matching text-Qwen's `original` policy, with one variant. Parakeet RNN-T fusion
instead scores lowercase and capitalized variants of each word and carries the
cheaper variant into context for both scorers. These replace the initial-capital
policy used in the preceding runs. Assistant-turn `<|im_end|>` ends the fusion history.
Native generation accepts either EOS in the model config.
Punctuation must come from Granite: no punctuation insertion mechanism is added.

All Loquacious domain subsets and dev.short are scored from the same parent
hypotheses. Use development results, especially Yodas deletion counts, to select
scales before any further held-out tuning. The .3 held-out runs are explicitly
transfer references, not claims of an optimum. Native numerical/model API tests
use a tiny real Qwen3-ASR model locally; full-checkpoint GPU tests run in Sisyphus.

Corpus-specific word lexica are no longer paired with LibriSpeech word LMs.
Keep corpus lexicon/no-LM controls, matched GigaSpeech lexicon/4-gram, and the
matched LibriSpeech lexicon/LM conditions. Previously registered mismatched
output files are retained, but the conditions are not scheduled or included in
new overviews. Existing submitted jobs are not cancelled automatically.
