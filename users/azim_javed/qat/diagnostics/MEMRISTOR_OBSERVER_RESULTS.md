# Encoder+predictor memristor greedy probe: seed 17

Run directory: `observer_memristor_encpred_greedy_s17/`.
Model: `ffnn_transducer_qat_encoder_pred_bpe___ffnn_transducer_bpe`, W8/A8,
cycle-1 converted checkpoint. Floating-point joiner; cached greedy decoder.
Utterance: `dev-other/7601-291468-0006/7601-291468-0006`.

| Measurement | Current | Current repeat | Frozen |
| --- | ---: | ---: | ---: |
| Encoder activation quantizers visited | 132 | 132 | 132 |
| Predictor output quantizers visited | 2 | 2 | 2 |
| First-call observer range changes | 0 | 0 | 0 |
| First-call qparam changes | 0 | 0 | 0 |
| Reference words | 74 | 74 | 74 |
| Word edit errors | 2 | 2 | 2 |
| Single-utterance WER | 2.702703% | 2.702703% | 2.702703% |

All three CTMs are byte-identical. The visited quantizer names and resulting
scales/zero points match. All noise-report fields match, including starting and
ending CPU/CUDA RNG-state fingerprints, encoder input/output fingerprints, and
first-scorer input/output fingerprints.

The same-seed repeat control succeeded. Read noise remained enabled; its
realization did not trigger an observer-range update in this probe. No observer
initialization effect or WER regression was observed.

The two potentially updating predictor output quantizers initialized identical
cached grids. The encoder quantizers use training-only observer updates. Thus
there is no persistent observer-policy grid difference for this model instance
under this initialization, although inference itself continues to be noisy.

This is a one-utterance, one-seed, one-programmed-checkpoint result, not a
full-corpus WER comparison or an estimate over hardware/read-noise realizations.
