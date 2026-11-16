# Corrected held-out error attribution

Selected Parakeet + Speech-Qwen 1.7B; counts come from the refreshed multi-fusion analysis. Fixed/new counts are corrected/introduced reference-word errors. Insertions are separate. Units whose normalized references disagree are excluded pairwise; therefore these counts do not reconstruct corpus WER. No totals are pooled across corpora/subsets.

## Relative to Parakeet

| Test set | Method | Fixed | New | Removed ins. | Added ins. | Units used | Excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AMI | 1p | 2303 | 495 | 507 | 222 | 7689 | 26 |
| AMI | 2p | 2641 | 397 | 559 | 270 | 7696 | 19 |
| Earnings22 | 1p | 147 | 67 | 47 | 14 | 3 | 3 |
| Earnings22 | 2p | 142 | 65 | 46 | 16 | 4 | 2 |
| GigaSpeech | 1p | 3445 | 1056 | 934 | 860 | 18626 | 131 |
| GigaSpeech | 2p | 3802 | 1013 | 922 | 973 | 18574 | 183 |
| LibriSpeech clean | 1p | 72 | 31 | 17 | 8 | 2609 | 11 |
| LibriSpeech clean | 2p | 83 | 38 | 16 | 10 | 2605 | 15 |
| LibriSpeech other | 1p | 125 | 66 | 17 | 10 | 2928 | 11 |
| LibriSpeech other | 2p | 150 | 97 | 21 | 22 | 2925 | 14 |
| Loquacious | 1p | 561 | 292 | 162 | 187 | 8053 | 34 |
| Loquacious | 2p | 637 | 379 | 192 | 240 | 8053 | 34 |
| commonvoice | 1p | 130 | 74 | 45 | 29 | 3035 | 9 |
| commonvoice | 2p | 188 | 102 | 52 | 45 | 3036 | 8 |
| librispeech | 1p | 120 | 60 | 14 | 10 | 2737 | 10 |
| librispeech | 2p | 142 | 91 | 18 | 20 | 2736 | 11 |
| voxpopuli | 1p | 129 | 127 | 70 | 97 | 1816 | 11 |
| voxpopuli | 2p | 132 | 160 | 86 | 128 | 1815 | 12 |
| yodas | 1p | 182 | 31 | 33 | 51 | 465 | 4 |
| yodas | 2p | 175 | 26 | 36 | 47 | 466 | 3 |
| Monsoon | 1p | 608 | 83 | 93 | 62 | 2098 | 4 |
| Monsoon | 2p | 631 | 106 | 109 | 66 | 2097 | 5 |
| SPGISpeech | 1p | 3649 | 1399 | 1583 | 2613 | 39162 | 179 |
| SPGISpeech | 2p | 4056 | 1282 | 1651 | 3067 | 39117 | 224 |
| VoxPopuli | 1p | 106 | 20 | 37 | 6 | 620 | 8 |
| VoxPopuli | 2p | 132 | 27 | 42 | 11 | 620 | 8 |

## Relative to Speech-Qwen 1.7B

| Test set | Method | Fixed | New | Removed ins. | Added ins. | Units used | Excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AMI | 1p | 855 | 2729 | 484 | 507 | 7675 | 40 |
| AMI | 2p | 858 | 2291 | 469 | 487 | 7673 | 42 |
| Earnings22 | 1p | 54 | 76 | 9 | 14 | 2 | 4 |
| Earnings22 | 2p | 52 | 91 | 10 | 13 | 2 | 4 |
| GigaSpeech | 1p | 2424 | 3513 | 1248 | 1116 | 18559 | 198 |
| GigaSpeech | 2p | 2355 | 3032 | 1225 | 1217 | 18547 | 210 |
| LibriSpeech clean | 1p | 212 | 83 | 14 | 22 | 2570 | 50 |
| LibriSpeech clean | 2p | 214 | 81 | 14 | 24 | 2568 | 52 |
| LibriSpeech other | 1p | 619 | 191 | 60 | 24 | 2884 | 55 |
| LibriSpeech other | 2p | 597 | 180 | 57 | 31 | 2879 | 60 |
| Loquacious | 1p | 1717 | 797 | 412 | 266 | 7997 | 90 |
| Loquacious | 2p | 1672 | 763 | 389 | 268 | 7992 | 95 |
| commonvoice | 1p | 635 | 359 | 117 | 64 | 3024 | 20 |
| commonvoice | 2p | 634 | 329 | 111 | 67 | 3024 | 20 |
| librispeech | 1p | 576 | 179 | 56 | 23 | 2696 | 51 |
| librispeech | 2p | 558 | 172 | 53 | 28 | 2693 | 54 |
| voxpopuli | 1p | 416 | 131 | 157 | 121 | 1817 | 10 |
| voxpopuli | 2p | 394 | 137 | 138 | 120 | 1814 | 13 |
| yodas | 1p | 90 | 128 | 82 | 58 | 460 | 9 |
| yodas | 2p | 86 | 125 | 87 | 53 | 461 | 8 |
| Monsoon | 1p | 421 | 349 | 89 | 86 | 2090 | 12 |
| Monsoon | 2p | 409 | 338 | 90 | 74 | 2089 | 13 |
| SPGISpeech | 1p | 2860 | 3333 | 2726 | 2066 | 39080 | 261 |
| SPGISpeech | 2p | 2789 | 2766 | 2390 | 2101 | 39059 | 282 |
| VoxPopuli | 1p | 115 | 123 | 27 | 64 | 622 | 6 |
| VoxPopuli | 2p | 118 | 108 | 27 | 64 | 624 | 4 |

Earnings22 has very few comparable recording-level units; do not infer stable word-error patterns from its attribution counts. Loquacious total and source rows overlap.
