# Completed development diagnostics

These are development-set diagnostics, not held-out performance or deployable oracle systems. They describe candidate availability and final ranking, not within-word pruning.

## librispeech-dev-other

### Standalone complementarity

| Pair | AM WER | Speech WER | Unit-selection oracle WER |
| --- | ---: | ---: | ---: |
| Granite CTC + Speech Qwen 1.7B | 2.631 | 2.664 | 1.669 |
| Parakeet RNN-T + Speech Qwen 1.7B | 2.036 | 2.664 | 1.533 |

The oracle selects the better standalone hypothesis per scoring unit using the reference. It is neither executable recognition nor a lower bound on arbitrary fusion.

### Candidate availability versus ranking

| AM (+ Speech 1.7B) | First | Rescored | Pool oracle | Coverage gaps | Ranking gaps | Missed correct | Mean distinct candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite CTC | 2.631 | 2.491 | 1.337 | 170 | 151 | 345 | 36.1 |
| Parakeet RNN-T | 2.071 | 1.935 | 0.795 | 34 | 93 | 368 | 8.7 |

Coverage gaps: the pool oracle has more edits than standalone Speech-Qwen. Ranking gaps: the pool can match or beat Speech-Qwen, but rescoring chooses a worse output. Missed correct: a reference-correct candidate exists but is not selected. Counts are scoring units, not word errors. A nominal 50-best cap does not imply 50 distinct candidates. The n-best first-pass rows are controls of their own candidate generators; they can differ from the independently selected standalone controls above.

### Reference-word accuracy by LibriSpeech LM-text frequency

| Condition | Unseen N | Unseen acc. % | Frequent N | Frequent acc. % |
| --- | ---: | ---: | ---: | ---: |
| Granite CTC control | 380 | 35.26 | 45394 | 98.52 |
| Parakeet RNN-T control | 380 | 44.74 | 45394 | 99.01 |
| Granite CTC + Speech 1.7B 1-pass fusion | 380 | 43.16 | 45394 | 98.51 |
| Granite CTC + Speech 1.7B 2-pass (50-best) | 380 | 40.26 | 45394 | 98.54 |
| Parakeet RNN-T + Speech 1.7B 1-pass fusion | 380 | 47.63 | 45394 | 99.10 |
| Parakeet RNN-T + Speech 1.7B 2-pass (50-best) | 380 | 47.89 | 45394 | 99.09 |

Word-alignment units: 2775 included, 89 excluded for reference-normalization mismatch. Insertions are separate from reference-word accuracy. Unseen means absent from the LibriSpeech LM training text, not necessarily OOV to a decoder or unseen in pretraining; it is not a proper-name annotation.

## loquacious-dev.short

### Standalone complementarity

| Pair | AM WER | Speech WER | Unit-selection oracle WER |
| --- | ---: | ---: | ---: |
| Granite CTC + Speech Qwen 1.7B | 4.705 | 4.624 | 3.365 |
| Parakeet RNN-T + Speech Qwen 1.7B | 3.951 | 4.624 | 3.236 |

The oracle selects the better standalone hypothesis per scoring unit using the reference. It is neither executable recognition nor a lower bound on arbitrary fusion.

### Candidate availability versus ranking

| AM (+ Speech 1.7B) | First | Rescored | Pool oracle | Coverage gaps | Ranking gaps | Missed correct | Mean distinct candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite CTC | 4.705 | 4.263 | 2.478 | 145 | 195 | 434 | 34.9 |
| Parakeet RNN-T | 3.989 | 3.811 | 1.964 | 44 | 142 | 439 | 11.6 |

Coverage gaps: the pool oracle has more edits than standalone Speech-Qwen. Ranking gaps: the pool can match or beat Speech-Qwen, but rescoring chooses a worse output. Missed correct: a reference-correct candidate exists but is not selected. Counts are scoring units, not word errors. A nominal 50-best cap does not imply 50 distinct candidates. The n-best first-pass rows are controls of their own candidate generators; they can differ from the independently selected standalone controls above.

### Reference-word accuracy by LibriSpeech LM-text frequency

| Condition | Unseen N | Unseen acc. % | Frequent N | Frequent acc. % |
| --- | ---: | ---: | ---: | ---: |
| Granite CTC control | 1185 | 67.59 | 42055 | 97.16 |
| Parakeet RNN-T control | 1185 | 72.91 | 42055 | 97.81 |
| Granite CTC + Speech 1.7B 1-pass fusion | 1185 | 72.74 | 42055 | 97.34 |
| Granite CTC + Speech 1.7B 2-pass (50-best) | 1185 | 71.56 | 42055 | 97.40 |
| Parakeet RNN-T + Speech 1.7B 1-pass fusion | 1185 | 74.18 | 42055 | 97.91 |
| Parakeet RNN-T + Speech 1.7B 2-pass (50-best) | 1185 | 74.94 | 42055 | 97.87 |

Word-alignment units: 2932 included, 68 excluded for reference-normalization mismatch. Insertions are separate from reference-word accuracy. Unseen means absent from the LibriSpeech LM training text, not necessarily OOV to a decoder or unseen in pretraining; it is not a proper-name annotation.
