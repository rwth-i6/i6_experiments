# RASR sequence-to-sequence templates

Examples for training and recognizing CTC, context-1 monotonic RNN-T,
full-context monotonic RNN-T and AED models, plus a pretrained Qwen-ASR 0.6B
SpeechLLM. Dataset preparation, training, model export, label scorer and search
configuration are kept separate.

| Model | Pipeline | Example |
| --- | --- | --- |
| CTC | `model_pipelines/ctc` | `experiments/librispeech/recognition/ctc_bpe.py` |
| Context-1 monotonic RNN-T | `model_pipelines/ffnn_transducer` | `experiments/librispeech/recognition/ffnn_transducer_bpe.py` |
| Full-context monotonic RNN-T | `model_pipelines/full_ctx_transducer` | `experiments/librispeech/recognition/full_ctx_transducer_bpe.py` |
| AED | `model_pipelines/aed` | `experiments/librispeech/recognition/aed_bpe.py` |
| Qwen-ASR 0.6B | `model_pipelines/qwen_asr` | `experiments/qwen_asr.py` |

The trainable examples include lexicon-free and closed-tree search where
supported, with a word 4-gram and subword LSTM/Transformer or word Transformer
LM. See `data/librispeech/lm.py` and `model_pipelines/{lstm_lm,transformer_lm}`.
Prior and internal-LM corrections remain explicit in recognition variants.
LMs must use the vocabulary/tokenization expected by their label scorer or
lexicon. These examples do not combine mismatched token vocabularies through
delayed fusion. Qwen-ASR is a standalone label-synchronous example, not another
word-LM combination.

From an appropriately configured Sisyphus setup with `config` pointing at
`users/berger/configs/baselines`, use `./sis m config/librispeech.py`,
`config/loquacious_small.py`, `config/loquacious_medium.py` or
`config/qwen_asr.py`. The package `py()`/`main()` runs LibriSpeech only;
Loquacious-medium and pretrained model downloads are explicit opt-ins.
The default graphs use representative operating points; additional streaming
and tokenization variants remain callable in the recognition modules.

Configure the cluster/container in your Sisyphus settings. The neural examples
need RETURNN, i6_models and CUDA; exported scorers require ONNX Runtime with the
appropriate execution provider. Qwen-ASR additionally needs a Transformers
version exposing `Qwen3ASRForConditionalGeneration` (the known working runtime
is 5.16.1), Hugging Face Hub and SoundFile. It uses the pinned HF conversion
`Qwen/Qwen3-ASR-0.6B-hf`; public model weights replace private checkpoints.
Put Hugging Face caches on a suitable work filesystem via `HF_HOME` /
`HF_HUB_CACHE`, not a small home directory.

Standard STM/CTM scoring is used. Qwen's LibriSpeech example uppercases and
strips punctuation from hypotheses only, retaining raw outputs; it applies no
Hugging Face leaderboard reference normalization. AED word times are uniform
positions for scoring, not forced alignments. Output limits retain partial
hypotheses and report truncation; no EOS is fabricated.

Research OOV searches, delayed fusion, Open-ASR evaluation, leaderboard scoring,
results and research plans are maintained separately on `berger_asr_research`
under `users/berger/asr_research`.
