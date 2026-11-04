# Development pilots: our BPE models + dynamic word-scored LMs

Run `./sis m config/wordwise_lm.py`. The combined baselines entrypoint also
includes this group. No held-out recognition or new acoustic training setup
is added; the existing training jobs/checkpoints are referenced.

| AM training corpus | Acoustic models | Evaluation |
| --- | --- | --- |
| LibriSpeech | BPE CTC, FFNN monotonic transducer | dev-other |
| Loquacious medium | BPE CTC, FFNN monotonic transducer | dev.short |

Each of these four AMs retains six uncorrected pilot conditions (24 recognitions):

- Greedy and beam-32 lexicon-free no-LM controls.
- LibriSpeech 128-merge BPE Transformer (96 layers), LM scale 0.8.
- LibriSpeech word Transformer (96 layers), LM scale 0.8.
- Qwen3-0.6B-Base and Qwen3-1.7B-Base, LM scale 0.3 each.

These are starting values, not newly tuned optima. Fusion uses final/pre-LM
beams 32/128, thresholds 8/8, GPU batches of eight and full-history inference,
without retained KV tensors. Acoustic prior and ILM subtraction were zero in
these historical pilots. They are **not** calibrated comparisons against the
older tuned tree-search results.

The additional corrected grid uses CTC prior scale **0.2** or monotonic
transducer ILM scale **0.2**, reusing the existing corpus-specific prior jobs
and decoder exports. These values match the existing recognition recipes;
they are starting points, not proven optima for all LMs. Each AM adds:

- A corrected beam-32 no-LM control, to isolate the correction itself.
- Each LibriSpeech Transformer LM at scales **0.4, 0.8, 1.2**.
- Each Qwen size at scales **0.3, 0.5, 0.7**.

This adds 52 dev recognitions, for **76 total**, with distinct `prior0.2` /
`ilm0.2` descriptors; historical pilot job identities stay unchanged. Pruning,
blank penalties and word/token rewards stay fixed. Select LM scales anew on
dev data: correction can change their optimum, but a higher optimum is not
guaranteed. Correction-strength and pruning tuning can follow this scale grid.
An optional LibriSpeech FFNN transducer diagnostic uses word Transformer
scale **0.8**, ILM scale **0.2**, pre/final beams **1024/1024** and pre/final
thresholds **16/16**. Its descriptor ends in
`ls-word-trafo-lm0.8-wide-beam1024-thr16-ilm0.2`. It is now disabled by default
because the large budget was too expensive (reported RTF about 14); enable
explicitly with `run(include_wide_diagnostic=True)` only for diagnostics.
It matches the main score/pruning settings of the previous
3.8% dev-other job (`ReturnnForwardJobV2.nHZNUHXQUcrj`), but retains dynamic
wordwise search: lexical constraints, dedicated word-end pruning and the LM
backend are not identical. Compare it with the existing corrected scale-0.8
run to isolate the larger search budget before interpreting the remaining gap.

## Word-LM UNK-penalty tuning

For each of the four active AMs, add ten dev-only word Transformer conditions:
LM scales **0.4 / 0.8** crossed with UNK penalties **1 / 2 / 4 / 8 / 12 nats**.
The corresponding corrected zero-penalty controls already exist. All use the
original efficient pre/final beams **128/32**, thresholds **8/8**, GPU batch
size eight, prior/ILM correction **0.2**, and zero uniform word penalty.
There are **116 active recognitions** in total (76 existing plus 40 new).

The descriptor includes `unk-betaN`. The dedicated scorer adds `beta / lm_scale`
to the cost of the UNK token, so after RASR scales it the actual per-unknown-word
bias is exactly **beta**, outside the LM scale. The word tokenizer produces
one LM token per completed word; known words, blanks and BOS/EOS are unaffected.
UNK still advances LM history normally and the hypothesized surface spelling
remains in the transcription. This is a decoding bias, not renormalization of
the LM probabilities, and it is not applied to BPE LMs or Qwen.

The wrapper validates CUDA execution rather than silently accepting CPU
fallback. It is registered only for the penalty runs; the original scorer,
runtime version hash and zero-penalty job identities are unchanged by this
addition. Penalty-run identities include the wrapper's source digest.
The sweep is intended to balance suppression of invented spellings against
recognition of genuine OOV words; do not choose beta from overall WER alone.

No held-out suite sweep is enabled here. Pretrained Granite/Parakeet require
their own measured acoustic priors / validated ILM implementation before
applying the same comparison; the own-model correction is not transferred.

Qwen retains the previously tested token reward 0.5 and word reward 1.0 outside
the LM scale; these are not applied to the LibriSpeech LMs. Qwen considers
lowercase and capitalized spellings and keeps the best spelling in its history.

Qwen revisions are pinned:

- `Qwen/Qwen3-0.6B-Base`: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
- [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base):
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.

Both CTC and monotonic/RNA models use `llm-timesync-beam-search`, with
`continuation-marked`/`@@` assembly. CTC collapses repeated frame labels; RNA
does not. Standard non-monotonic RNN-T's inner label loop is not used here.
The pinned `d000386c` build already contains both the new LM Python interface
and BPE word assembler. A native CPU gate validates repeats, word completion,
scaled LM cost and EOS before fusion recognition can run.

The acoustic label inventory is unchanged. At a completed word, the BPE LM
retokenizes uppercase text with **its own LibriSpeech codes and indexed vocabulary**,
independently of the AM's codes. The word LM maps the uppercase spelling to
exactly one vocabulary id, or UNK. UNK changes the LM context, not the output
transcription: an unseen acoustic spelling remains recognizable. BOS and EOS
are id 0, matching these existing LM training configurations. ONNX returns
next-token negative log probabilities; multi-token words are teacher-forced,
including every token exactly once. Runtime checks cover dense vocabulary
indices, output dimension, probability normalization and CUDA availability.
The Python LM scorer uses the shared Apptainer image's Python package, which
must pin `onnxruntime-gpu==1.26.0` to match its CUDA-12 environment. This pip
package is separate from the native ONNX Runtime library used by RASR.
There is no job-local package installation; the runtime rejects CPU fallback.
These jobs use a last-hidden-only stateless export: the final valid hidden
state is selected before vocabulary projection, avoiding a `[batch,time,vocab]`
temporary for the large word vocabulary. Training and older exports are unchanged.

Waveform preprocessing matches the existing own-model training/recognition
configs, including pre-emphasis. All new conditions share the updated HF
English scoring normalizer. Stepwise RASR statistics are disabled.
Individual CTM/log/scoring/RTF outputs retain corpus-context prefixes; the
combined report `wordwise-lm/dev/summary` includes the training-corpus context
to distinguish LibriSpeech from Loquacious medium.
