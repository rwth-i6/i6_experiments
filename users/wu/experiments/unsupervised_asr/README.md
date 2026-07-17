# unsupervised_asr — sisyphus recipe

Unsupervised ASR via a phoneme text bottleneck on a frozen BEST-RQ encoder. **Everything in this
directory (i.e. anywhere under `recipe/`) is sisyphus** — Jobs and the `py()` graphs that wire them.
Standalone test scripts do **not** belong here.

## Layout of the project across the workspace

| Where | What | Version-controlled |
|---|---|---|
| `recipe/i6_experiments/users/wu/unsupervised_asr/` (**here**) | LLM-independent sisyphus: 𝒯_φ phoneme-LM pipeline (`text.py`, `phonemize.py`, `phoneme_lm.py`), rVAD preprocessing + §1.0 gate (`vad_port.py`) | i6_experiments `haotian` |
| `scripts/unsupervised_asr/` | standalone CPU probes: `decipher.py`, `hsmm_decipher.py`, `continuous_gan.py`, `unsup_metric.py` | no (workspace-local) |
| `experiments/ssl/analysis/` | frozen-encoder **representation-quality** audit: `repr_audit.py`, `real_repr_probe.py` | i6_experiments `haotian` |
| `recipe/2025-10-speech-llm/` | the **LLM autoencoder** — sisyphus config+recipe | speech-llm `haotian_modality_matching_jupiter` |

## Sisyphus entry points (workspace `config/`)

- `config/sae_0b_text.py` → `text.librispeech_phoneme_inventory`, `phonemize.phonemize_lm_corpus`
- `config/sae_1a_lm.py` → `phoneme_lm.phoneme_ngram_lm`
- `config/sae_1_0_vad.py` → `vad_port.register_rvad_validation`

## Convention

Prefer sisyphus Jobs for anything chained / large-scale / reusable. A one-off CPU probe is fine as a
standalone script, but it lives in `scripts/`, never under `recipe/`. When the GAN / PUSM / LLM
autoencoder graduate from probe to real training, they become sisyphus Jobs here (GAN/PUSM) or in
`recipe/2025-10-speech-llm` (LLM autoencoder).
