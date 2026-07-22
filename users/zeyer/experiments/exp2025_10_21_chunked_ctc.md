# Chunked CTC / Conformer streaming ASR: results

Code: `i6_experiments/users/zeyer/experiments/exp2025_10_21_chunked_ctc.py`.

## Overview / setup

Study: how close chunked streaming attention gets to the offline model.

Task: Loquacious, `large` subset (~25 kh).
1x training = `total_k_hours=100` = 4 full epochs (100 subepochs, split 25).

Model: Conformer encoder + AED (Transformer) decoder, with CTC aux heads.
- encoder: 16 layers, dim 1024, 8 heads, relu-square FF (no bias),
  `ConformerConvSubsample` /6 downsampling (`out_dims=[32,64,64]`).
- decoder: `TransformerDecoder`, 6 layers, dim 1024, RMSNorm, `FeedForwardGated`, rotary causal self-att.
- CTC aux: encoder layers `[4,10,16]`, decoder layer `[3]`; `feature_batch_norm`.

Train: bf16, batch 100k, weight_decay 1e-2, lrlin OCLR base_lr 0.5, max input 19.5 s.
Vocab: spm10k (sampling BPE, breadth_prob 0.01).

Recog:
- CTC-only (`ctc_model_recog`) = headline.
- AED+CTC time-sync (`aed_ctc_timesync_recog_recomb_auto_scale`).
- CTC+LM label-sync + prior (`ctc_recog_recomb_labelwise_prior_auto_scale`),
  LM = Trafo n32-d1024 spm10k.
Encoders: `nn_rf/encoder/chunked_conformer_v1`, `nn_rf/encoder/chunked_conformer_v2.py`.

Eval: Loquacious dev / test, aggregate + per-domain (voxpopuli, commonvoice, librispeech, yodas).
Metric below: CTC-only WER [%], dev / test aggregate, last epoch.

## Baselines

Offline (full-context) reference:

| model | dev | test |
| --- | --- | --- |
| base (offline conformer) | 7.32 | 8.10 |

Vocab size (chunked L80-C5-R4, v1; CTC-only dev):

| vocab | dev |
| --- | --- |
| spm1k | 10.66 |
| spm5k | 9.52 |
| spm10k | 9.56 |

Default: spm10k.

## Training scale

CTC-only WER, dev / test, last epoch.
base: offline recog.
dyn-rope-ctembed: streaming recog at the deployment chunk (C5, R4).

| scale | base (offline) | dyn-rope-ctembed (streaming) |
| --- | --- | --- |
| 1x | 7.32 / 8.10 | 9.41 / 10.29 |
| 2x | 6.58 / 7.39 | 8.52 / 9.25 |
| 4x | 6.08 / 6.72 | 7.82 / 8.59 |
