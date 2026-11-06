# Digital full-QAT observer-initialization probe

Run directory: `observer_wer_probe_singleprocess_gpu/`
Model: `qat_ffnn_transducer_full_quant_v2_lowbs`, W8/A8, epoch 2000.
First utterance: `dev-other/7601-291468-0006/7601-291468-0006`.
This is also the first utterance in the historical no-LM tree-decoding CTM.

| Measurement | Current | Frozen |
| --- | ---: | ---: |
| Encoder activation quantizers visited | 252 | 252 |
| Scorer activation quantizers visited | 8 | 8 |
| Observer ranges changed on first call | 0 | 0 |
| Quantization parameters changed on first call | 0 | 0 |
| Reference words | 74 | 74 |
| Word edit errors | 1 | 1 |
| Single-utterance WER | 1.351351% | 1.351351% |

The visited quantizer names and resulting scale/zero-point values match across
modes. The two CTM files are byte-identical. The sole word error is an extra
`A` before `COUNTRY SEAT`.

No regression is observed in this probe. Since these quantizers cache their
qparams after initialization, the suspected first-evaluation-update mechanism
has introduced no persistent grid difference in this digital model instance.
Later calls in the same eval session reuse those grids. A larger paired decode
is therefore not necessary just to demonstrate this initialization mechanism
for this first-input ordering.

This is not a measured full-corpus WER comparison. It does not establish
equivalence for different first inputs, checkpoints, shards/model reloads, or
memristor outputs. Numerical logits and gradients were not compared here.
