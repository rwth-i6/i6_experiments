"""Unsupervised ASR (SAE) — sisyphus recipe code.

Grapheme/phoneme text-bottleneck unsupervised ASR (NLA-style) on a frozen BEST-RQ encoder.
This package holds the *sisyphus* pieces that are LLM-independent: the 𝒯_φ phoneme-LM pipeline
(text/phonemize/phoneme_lm) and the rVAD silence-removal preprocessing + §1.0 validation gate
(vad_port). Everything under recipe/ must be a sisyphus Job/graph.

Not here (see README):
  - standalone CPU probes (decipher / hsmm / GAN / §1.0 metric) -> workspace ``scripts/unsupervised_asr/``
  - frozen-encoder representation-quality audit (repr_audit / real_repr_probe) -> ``experiments/ssl/analysis``
  - the LLM autoencoder -> sisyphus config+recipe under ``recipe/2025-10-speech-llm``
"""
