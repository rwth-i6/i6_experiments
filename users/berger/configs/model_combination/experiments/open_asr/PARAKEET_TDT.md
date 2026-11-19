# Parakeet TDT v3 development pilot

Run from the existing Sisyphus setup.

```bash
./sis m config/parakeet_tdt.py
```

This pilot uses the official NVIDIA `parakeet-tdt-0.6b-v3` Transformers
checkpoint at revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`. The current
official v2 release is a NeMo bundle. Using v3 lets the native control and
ONNX export share the checkpoint in the existing Transformers environment.
The pinned config has 8,192 SentencePiece tokens, blank ID 8192, a two-layer
predictor of width 640, durations 0–4 and an 80 ms encoder frame shift.
[Official model config](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/config.json)

On each dev set, the pilot compares native Transformers greedy, lexicon-free
RASR at beam 1 and beam 8, and RASR beam 8 with text-Qwen 1.7B Base or Speech-Qwen
1.7B at LM scales 0.1 and 0.25. The fusion scorer uses TDT's hypothesized casing.
The report also includes the existing same-corpus RNN-T and standalone
Speech-Qwen controls. This is a development sweep for choosing the settings
before a held-out evaluation.

RASR is built from `d4aa61a12fa4f33e873512678a4d204a90329748`. Python sends
unpadded projected encoder features in frame order. The three ONNX graphs
map the batch-first LSTM states and predictor output by metadata. The updater
accepts RASR's int32 token input. The joint graph returns separately normalized
negative log-probabilities for tokens and durations. ONNX sessions use the CPU
provider for the initial compatibility run. The encoder and Qwen models use
the GPU.

The ONNX job verifies native PyTorch and ONNX Runtime against real encoder
features and validates recurrent updates at a batch size greater than one.
An eight-utterance RASR beam-1 probe compares the native transcripts and records
encoder lengths and traceback endpoints. Monotone finite traces are checked.
Endpoints can differ from the encoder length because of acoustic EOS and
duration jumps. The probe reports native agreement without requiring an exact
match, since native greedy clips a zero blank duration to one and RASR selects
the best positive blank duration and adds duration costs to token costs.
Recognition uses benchmark JSONL hypotheses and the existing benchmark scorer.
Traceback frame times are retained. CTM export and streaming are outside this
development pilot.

Outputs are under `output/parakeet-tdt/f4212424889ee5d3/`. The published
report is `output/reports/asr-lm/parakeet-tdt.md`. Inspect `tools/onnx-parity.json`
and `tools/probe.json` for the validation details.
