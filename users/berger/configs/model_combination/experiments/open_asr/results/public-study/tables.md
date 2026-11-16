# Frozen public evaluation: WER (%)

All settings were selected jointly on LibriSpeech dev-other and Loquacious dev.short. These are fixed operating points, not per-test-set winners. 1p = one-pass delayed fusion; 2p = n-best rescoring. Speech/Text identify the Qwen model type; sizes are explicit.

Granite is the selected AM-only open-tree first-pass control, not native greedy decoding. Parakeet is the selected AM-only RASR beam control. Speech-Qwen standalone uses beam 8, length penalty 0. The Granite two-pass first pass uses an extended LibriSpeech lexicon; one-pass fusion is lexicon-free. Parakeet fusion is lexicon-free.

## Main corpora

### Standalone controls

| Test set | Granite | Parakeet | Speech Qwen 0.6B | Speech Qwen 1.7B |
| --- | ---: | ---: | ---: | ---: |
| AMI | 7.363 | 13.295 | 9.336 | 8.367 |
| Earnings22 | 8.471 | 8.882 | 7.537 | 5.652 |
| GigaSpeech | 8.523 | 8.169 | 7.662 | 7.289 |
| LibriSpeech clean | 1.457 | 1.089 | 1.635 | 1.213 |
| LibriSpeech other | 2.596 | 2.099 | 3.895 | 2.862 |
| Loquacious | 4.936 | 4.353 | 6.144 | 4.973 |
| Monsoon | 4.286 | 4.421 | 4.114 | 3.425 |
| SPGISpeech | 1.841 | 2.679 | 2.705 | 2.569 |
| VoxPopuli | 4.635 | 3.776 | 2.960 | 2.852 |

### Granite fusion

| Test set | Speech 0.6B / 1p | Speech 0.6B / 2p | Speech 1.7B / 1p | Speech 1.7B / 2p | Text 0.6B / 1p | Text 0.6B / 2p | Text 1.7B / 1p | Text 1.7B / 2p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AMI | 7.093 | 7.190 | 7.062 | 7.169 | 7.495 | 7.597 | 7.476 | 7.522 |
| Earnings22 | 7.569 | 7.887 | 7.556 | 7.620 | 8.246 | 8.393 | 8.161 | 8.283 |
| GigaSpeech | 7.980 | 8.138 | 7.928 | 7.942 | 8.400 | 8.587 | 8.299 | 8.556 |
| LibriSpeech clean | 1.343 | 1.386 | 1.343 | 1.361 | 1.380 | 1.410 | 1.344 | 1.386 |
| LibriSpeech other | 2.428 | 2.426 | 2.360 | 2.443 | 2.587 | 2.557 | 2.524 | 2.517 |
| Loquacious | 4.614 | 4.661 | 4.561 | 4.603 | 4.783 | 4.785 | 4.724 | 4.758 |
| Monsoon | 3.576 | 3.645 | 3.355 | 3.410 | 4.280 | 4.537 | 4.133 | 4.509 |
| SPGISpeech | 1.705 | 1.752 | 1.691 | 1.820 | 1.782 | 1.821 | 1.760 | 1.814 |
| VoxPopuli | 3.741 | 3.836 | 3.741 | 3.701 | 4.848 | 4.882 | 4.809 | 4.826 |

### Parakeet fusion

| Test set | Speech 0.6B / 1p | Speech 0.6B / 2p | Speech 1.7B / 1p | Speech 1.7B / 2p | Text 0.6B / 1p | Text 1.7B / 1p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMI | 11.235 | 11.072 | 10.699 | 10.182 | 13.492 | 13.360 |
| Earnings22 | 7.900 | 8.421 | 7.483 | 7.952 | 9.310 | 9.128 |
| GigaSpeech | 7.680 | 7.709 | 7.526 | 7.449 | 8.450 | 8.366 |
| LibriSpeech clean | 1.018 | 1.040 | 0.993 | 0.989 | 1.123 | 1.097 |
| LibriSpeech other | 1.980 | 2.076 | 1.962 | 1.987 | 2.142 | 2.129 |
| Loquacious | 4.218 | 4.314 | 4.163 | 4.191 | 4.526 | 4.447 |
| Monsoon | 3.553 | 3.547 | 3.279 | 3.254 | 5.422 | 5.316 |
| SPGISpeech | 2.466 | 2.431 | 2.550 | 2.537 | 2.856 | 2.860 |
| VoxPopuli | 3.190 | 3.043 | 3.105 | 3.004 | 4.344 | 4.288 |

## Loquacious source subsets

These overlap the Loquacious total; its LibriSpeech subset also overlaps the separate LibriSpeech evaluation. Do not pool them.

### Standalone controls

| Test set | Granite | Parakeet | Speech Qwen 0.6B | Speech Qwen 1.7B |
| --- | ---: | ---: | ---: | ---: |
| commonvoice | 6.737 | 5.713 | 8.661 | 6.633 |
| librispeech | 2.581 | 2.098 | 3.919 | 2.846 |
| voxpopuli | 5.637 | 5.018 | 6.635 | 5.804 |
| yodas | 7.818 | 8.053 | 7.466 | 6.743 |

### Granite fusion

| Test set | Speech 0.6B / 1p | Speech 0.6B / 2p | Speech 1.7B / 1p | Speech 1.7B / 2p | Text 0.6B / 1p | Text 0.6B / 2p | Text 1.7B / 1p | Text 1.7B / 2p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| commonvoice | 6.161 | 6.262 | 6.041 | 6.063 | 6.150 | 6.156 | 6.095 | 6.111 |
| librispeech | 2.399 | 2.399 | 2.340 | 2.413 | 2.565 | 2.538 | 2.508 | 2.492 |
| voxpopuli | 5.442 | 5.455 | 5.430 | 5.434 | 5.548 | 5.537 | 5.496 | 5.521 |
| yodas | 7.051 | 7.292 | 7.042 | 7.145 | 7.930 | 8.087 | 7.826 | 8.155 |

### Parakeet fusion

| Test set | Speech 0.6B / 1p | Speech 0.6B / 2p | Speech 1.7B / 1p | Speech 1.7B / 2p | Text 0.6B / 1p | Text 1.7B / 1p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| commonvoice | 5.554 | 5.586 | 5.455 | 5.388 | 5.703 | 5.685 |
| librispeech | 1.977 | 2.084 | 1.956 | 1.984 | 2.154 | 2.132 |
| voxpopuli | 5.064 | 5.217 | 5.070 | 5.179 | 5.273 | 5.159 |
| yodas | 7.232 | 7.224 | 6.913 | 6.869 | 8.875 | 8.528 |
