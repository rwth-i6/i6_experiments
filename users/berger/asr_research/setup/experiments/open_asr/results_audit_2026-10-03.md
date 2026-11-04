# Open ASR results audit — 2026-10-03

Read from the currently registered output and alias links. The active neural_lms=False matrix has 78 scored recognitions and 8 failed recognitions (86 total). Seven earlier neural-LM recognitions also have completed scoring, giving 85 scored results overall. Failed and incomplete runs are not included as WER observations. No recognition jobs were changed or resubmitted during this audit.

## Main findings

- Granite native and RASR lexicon-free paths agree exactly in text and tokens on all eight datasets. Cohere agrees in tokens and normalized text on the six completed native/free pairs; raw text differs. Parakeet has small differences, with WER differences at most 0.033 percentage points.
- Closed-tree/LM conditions use the LibriSpeech lexicon, fallback disabled, initial beam 32 / threshold 12 / LM scale 0.3. These are not the previously tuned extended-tree conditions.
- Cohere +4gram has a serious failure pattern: Monsoon WER 3.43% native → 4.22% tree → 33.01% tree+4gram, including repeated generic phrases; Earnings22 tree → tree+4gram is 13.26% → 41.54%. These results need investigation, not interpretation as a tuned LM comparison.
- Corpus-specific lexica help: Granite VoxPopuli 6.15% → 4.97%; Cohere VoxPopuli 6.61% → 5.10% (native 5.34%). Granite GigaSpeech 10.08% → 9.02%; adding the corpus 4gram gives 9.53% (native 8.46%).
- Native AED token-limit failures also occur at the same progress positions as lexicon-free RASR. Strict EOS guards currently abort a whole dataset on these cases. Raising the limit alone is unlikely to resolve persistent token loops.
- Pair-dependent compound normalization causes slight reference word-count differences between conditions. Small WER differences need paired error analysis.

## granite-ctc

WER (%). — means no scored output.

| Dataset | native | rasr-lexfree | tree | tree-4gram | tree-trafo | tree-qwen |
|---|---:|---:|---:|---:|---:|---:|
| librispeech-test-clean | 1.46 | 1.46 | 1.83 | 1.83 | 1.80 | 5.72 |
| librispeech-test-other | 2.67 | 2.67 | 3.03 | 2.97 | 2.98 | — |
| ami-cleaned | 7.35 | 7.35 | 8.01 | 8.61 | 8.68 | — |
| gigaspeech-cleaned | 8.46 | 8.46 | 10.08 | 11.40 | — | — |
| spgispeech | 1.76 | 1.76 | 3.37 | 4.22 | — | — |
| voxpopuli-cleaned | 4.69 | 4.69 | 6.15 | 8.23 | — | 10.90 |
| monsoon-en-in | 4.32 | 4.32 | 4.79 | 6.41 | 6.55 | — |
| earnings22-cleaned | 8.34 | 8.34 | 15.74 | 21.33 | 23.29 | — |

All completed scores, error counts and measured RTF:

| Condition | Dataset | WER | Ins | Del | Sub | Ref words | RTF |
|---|---|---:|---:|---:|---:|---:|---:|
| corpus-tree-4gram-gigaspeech-cleaned | gigaspeech-cleaned | 9.529 | 5960 | 12559 | 19263 | 396500 | 0.0144 |
| corpus-tree-gigaspeech-cleaned | gigaspeech-cleaned | 9.015 | 7411 | 7680 | 20653 | 396484 | 0.0066 |
| corpus-tree-voxpopuli-cleaned | voxpopuli-cleaned | 4.972 | 160 | 322 | 402 | 17779 | 0.0075 |
| native | ami-cleaned | 7.354 | 1120 | 1884 | 3048 | 82294 | 0.0050 |
| native | earnings22-cleaned | 8.336 | 191 | 403 | 689 | 15391 | 0.0023 |
| native | gigaspeech-cleaned | 8.458 | 7080 | 7615 | 18837 | 396441 | 0.0033 |
| native | librispeech-test-clean | 1.456 | 53 | 49 | 669 | 52967 | 0.0035 |
| native | librispeech-test-other | 2.674 | 121 | 103 | 1188 | 52803 | 0.0037 |
| native | monsoon-en-in | 4.319 | 322 | 657 | 1131 | 48855 | 0.0028 |
| native | spgispeech | 1.758 | 3797 | 3166 | 10056 | 968249 | 0.0029 |
| native | voxpopuli-cleaned | 4.691 | 157 | 320 | 357 | 17777 | 0.0027 |
| rasr-lexfree | ami-cleaned | 7.354 | 1120 | 1884 | 3048 | 82294 | 0.0074 |
| rasr-lexfree | earnings22-cleaned | 8.336 | 191 | 403 | 689 | 15391 | 0.0043 |
| rasr-lexfree | gigaspeech-cleaned | 8.458 | 7080 | 7615 | 18837 | 396441 | 0.0053 |
| rasr-lexfree | librispeech-test-clean | 1.456 | 53 | 49 | 669 | 52967 | 0.0057 |
| rasr-lexfree | librispeech-test-other | 2.674 | 121 | 103 | 1188 | 52803 | 0.0066 |
| rasr-lexfree | monsoon-en-in | 4.319 | 322 | 657 | 1131 | 48855 | 0.0057 |
| rasr-lexfree | spgispeech | 1.758 | 3797 | 3166 | 10056 | 968249 | 0.0047 |
| rasr-lexfree | voxpopuli-cleaned | 4.691 | 157 | 320 | 357 | 17777 | 0.0048 |
| tree | ami-cleaned | 8.012 | 1206 | 1932 | 3457 | 82318 | 0.0081 |
| tree | earnings22-cleaned | 15.739 | 265 | 863 | 1295 | 15395 | 0.0055 |
| tree | gigaspeech-cleaned | 10.078 | 8225 | 7867 | 23899 | 396801 | 0.0064 |
| tree | librispeech-test-clean | 1.826 | 113 | 48 | 806 | 52968 | 0.0071 |
| tree | librispeech-test-other | 3.032 | 216 | 79 | 1306 | 52804 | 0.0077 |
| tree | monsoon-en-in | 4.791 | 379 | 646 | 1316 | 48863 | 0.0059 |
| tree | spgispeech | 3.370 | 6339 | 4438 | 21859 | 968437 | 0.0058 |
| tree | voxpopuli-cleaned | 6.148 | 190 | 388 | 515 | 17779 | 0.0055 |
| tree-4gram | ami-cleaned | 8.614 | 823 | 3209 | 3059 | 82320 | 0.0157 |
| tree-4gram | earnings22-cleaned | 21.325 | 170 | 1977 | 1136 | 15395 | 0.0117 |
| tree-4gram | gigaspeech-cleaned | 11.401 | 5989 | 17353 | 21899 | 396823 | 0.0146 |
| tree-4gram | librispeech-test-clean | 1.831 | 75 | 114 | 781 | 52969 | 0.0105 |
| tree-4gram | librispeech-test-other | 2.969 | 134 | 140 | 1294 | 52804 | 0.0126 |
| tree-4gram | monsoon-en-in | 6.414 | 252 | 1639 | 1243 | 48865 | 0.0142 |
| tree-4gram | spgispeech | 4.222 | 3961 | 16921 | 20005 | 968444 | 0.0126 |
| tree-4gram | voxpopuli-cleaned | 8.229 | 146 | 827 | 490 | 17779 | 0.0123 |
| tree-qwen | librispeech-test-clean | 5.723 | 65 | 2216 | 750 | 52966 | 1.9893 |
| tree-qwen | voxpopuli-cleaned | 10.900 | 159 | 1295 | 484 | 17780 | 1.7229 |
| tree-trafo | ami-cleaned | 8.675 | 872 | 3178 | 3091 | 82317 | 0.5522 |
| tree-trafo | earnings22-cleaned | 23.293 | 177 | 2302 | 1107 | 15395 | 0.4369 |
| tree-trafo | librispeech-test-clean | 1.803 | 64 | 115 | 776 | 52968 | 0.3923 |
| tree-trafo | librispeech-test-other | 2.979 | 120 | 204 | 1249 | 52804 | 0.3843 |
| tree-trafo | monsoon-en-in | 6.553 | 267 | 1687 | 1248 | 48865 | 0.4215 |

## cohere-aed

WER (%). — means no scored output.

| Dataset | native | rasr-lexfree | tree | tree-4gram | tree-trafo | tree-qwen |
|---|---:|---:|---:|---:|---:|---:|
| librispeech-test-clean | 0.97 | 0.97 | 1.25 | 1.35 | — | — |
| librispeech-test-other | 2.08 | 2.08 | 2.35 | 2.42 | — | — |
| ami-cleaned | 7.03 | 7.03 | 7.72 | 11.31 | — | — |
| gigaspeech-cleaned | — | — | 9.62 | — | — | — |
| spgispeech | 2.75 | 2.75 | — | — | — | — |
| voxpopuli-cleaned | 5.34 | 5.34 | 6.61 | 13.74 | — | — |
| monsoon-en-in | 3.43 | 3.43 | 4.22 | 33.01 | — | — |
| earnings22-cleaned | — | — | 13.26 | 41.54 | — | — |

All completed scores, error counts and measured RTF:

| Condition | Dataset | WER | Ins | Del | Sub | Ref words | RTF |
|---|---|---:|---:|---:|---:|---:|---:|
| corpus-tree-gigaspeech-cleaned | gigaspeech-cleaned | 7.734 | 7684 | 6007 | 16972 | 396466 | 0.2718 |
| corpus-tree-voxpopuli-cleaned | voxpopuli-cleaned | 5.101 | 243 | 339 | 325 | 17780 | 0.2365 |
| native | ami-cleaned | 7.035 | 1373 | 1591 | 2824 | 82275 | 0.0313 |
| native | librispeech-test-clean | 0.970 | 53 | 37 | 424 | 52972 | 0.0259 |
| native | librispeech-test-other | 2.075 | 99 | 87 | 910 | 52810 | 0.0264 |
| native | monsoon-en-in | 3.431 | 351 | 452 | 873 | 48850 | 0.0213 |
| native | spgispeech | 2.754 | 11157 | 3421 | 12090 | 968178 | 0.0233 |
| native | voxpopuli-cleaned | 5.343 | 271 | 333 | 346 | 17780 | 0.0229 |
| rasr-lexfree | ami-cleaned | 7.035 | 1373 | 1591 | 2824 | 82275 | 0.0355 |
| rasr-lexfree | librispeech-test-clean | 0.970 | 53 | 37 | 424 | 52972 | 0.0306 |
| rasr-lexfree | librispeech-test-other | 2.075 | 99 | 87 | 910 | 52810 | 0.0318 |
| rasr-lexfree | monsoon-en-in | 3.431 | 351 | 452 | 873 | 48850 | 0.0259 |
| rasr-lexfree | spgispeech | 2.754 | 11157 | 3421 | 12090 | 968178 | 0.0290 |
| rasr-lexfree | voxpopuli-cleaned | 5.343 | 271 | 333 | 346 | 17780 | 0.0278 |
| tree | ami-cleaned | 7.717 | 1433 | 1756 | 3163 | 82316 | 0.1835 |
| tree | earnings22-cleaned | 13.264 | 234 | 1094 | 713 | 15387 | 0.3001 |
| tree | gigaspeech-cleaned | 9.621 | 8297 | 10813 | 19063 | 396751 | 0.2710 |
| tree | librispeech-test-clean | 1.246 | 96 | 34 | 530 | 52973 | 0.1570 |
| tree | librispeech-test-other | 2.348 | 159 | 85 | 996 | 52809 | 0.1930 |
| tree | monsoon-en-in | 4.224 | 412 | 607 | 1045 | 48859 | 0.2056 |
| tree | voxpopuli-cleaned | 6.615 | 325 | 371 | 480 | 17778 | 0.2346 |
| tree-4gram | ami-cleaned | 11.312 | 1285 | 4329 | 3698 | 82318 | 0.1587 |
| tree-4gram | earnings22-cleaned | 41.538 | 244 | 4728 | 1422 | 15393 | 0.2577 |
| tree-4gram | librispeech-test-clean | 1.354 | 71 | 56 | 590 | 52972 | 0.0908 |
| tree-4gram | librispeech-test-other | 2.422 | 146 | 132 | 1001 | 52812 | 0.1323 |
| tree-4gram | monsoon-en-in | 33.007 | 421 | 9592 | 6115 | 48863 | 0.2108 |
| tree-4gram | voxpopuli-cleaned | 13.735 | 369 | 1090 | 983 | 17779 | 0.2557 |

## parakeet-rnnt

WER (%). — means no scored output.

| Dataset | native | rasr-lexfree |
|---|---:|---:|
| librispeech-test-clean | 1.14 | 1.14 |
| librispeech-test-other | 2.13 | 2.12 |
| ami-cleaned | 14.34 | 14.36 |
| gigaspeech-cleaned | 8.39 | 8.40 |
| spgispeech | 2.72 | 2.72 |
| voxpopuli-cleaned | 3.71 | 3.70 |
| monsoon-en-in | 4.80 | 4.81 |
| earnings22-cleaned | 10.04 | 10.01 |

All completed scores, error counts and measured RTF:

| Condition | Dataset | WER | Ins | Del | Sub | Ref words | RTF |
|---|---|---:|---:|---:|---:|---:|---:|
| native | ami-cleaned | 14.339 | 950 | 6557 | 4288 | 82258 | 0.0238 |
| native | earnings22-cleaned | 10.038 | 158 | 631 | 756 | 15392 | 0.0158 |
| native | gigaspeech-cleaned | 8.395 | 6259 | 10148 | 16865 | 396351 | 0.0201 |
| native | librispeech-test-clean | 1.142 | 59 | 73 | 473 | 52963 | 0.0192 |
| native | librispeech-test-other | 2.127 | 87 | 80 | 956 | 52800 | 0.0193 |
| native | monsoon-en-in | 4.796 | 247 | 1031 | 1065 | 48857 | 0.0166 |
| native | spgispeech | 2.716 | 6880 | 7084 | 12332 | 968011 | 0.0175 |
| native | voxpopuli-cleaned | 3.709 | 125 | 283 | 251 | 17769 | 0.0164 |
| rasr-lexfree | ami-cleaned | 14.358 | 943 | 6586 | 4282 | 82258 | 0.0468 |
| rasr-lexfree | earnings22-cleaned | 10.005 | 154 | 632 | 754 | 15392 | 0.0632 |
| rasr-lexfree | gigaspeech-cleaned | 8.395 | 6231 | 10195 | 16849 | 396351 | 0.0650 |
| rasr-lexfree | librispeech-test-clean | 1.142 | 59 | 75 | 471 | 52963 | 0.0482 |
| rasr-lexfree | librispeech-test-other | 2.125 | 86 | 80 | 956 | 52800 | 0.0434 |
| rasr-lexfree | monsoon-en-in | 4.808 | 246 | 1038 | 1065 | 48857 | 0.0444 |
| rasr-lexfree | spgispeech | 2.717 | 6855 | 7103 | 12341 | 968011 | 0.1035 |
| rasr-lexfree | voxpopuli-cleaned | 3.703 | 124 | 283 | 251 | 17769 | 0.0465 |

## Failed jobs

All eight failures are Cohere AED. Six are token-limit failures; two have no completed EOS path.

| Condition/dataset | Error | Job |
|---|---|---|
| tree/spgispeech | ValueError: RASR AED exceeded the native token limit | HfBenchmarkRecogJob.EdRm53iiNeNT |
| rasr-lexfree/earnings22-cleaned | ValueError: RASR AED exceeded the native token limit | HfBenchmarkRecogJob.I7FJFNqHifXc |
| rasr-lexfree/gigaspeech-cleaned | ValueError: RASR AED exceeded the native token limit | HfBenchmarkRecogJob.Ko2Xws9k0o9N |
| corpus-tree-4gram-gigaspeech-cleaned/gigaspeech-cleaned | ValueError: RASR AED returned no completed EOS path for an audio chunk | HfBenchmarkRecogJob.Th17f3wn9iQ0 |
| tree-4gram/spgispeech | ValueError: RASR AED returned no completed EOS path for an audio chunk | HfBenchmarkRecogJob.kw7jQ0V3xTrt |
| tree-4gram/gigaspeech-cleaned | ValueError: RASR AED exceeded the native token limit | HfBenchmarkRecogJob.uVXofhffFJOi |
| native/gigaspeech-cleaned | ValueError: Native AED hit its token limit without EOS | HfBenchmarkRecogJob.3UP1NEbBDevZ |
| native/earnings22-cleaned | ValueError: Native AED hit its token limit without EOS | HfBenchmarkRecogJob.IIJN6CiLKplA |

## Runtime and logging

Granite native RTF 0.002–0.005; closed-tree approximately 0.006–0.008; 4gram 0.011–0.016; Transformer 0.38–0.55; Qwen 1.72–1.99. Cohere native 0.021–0.031, lexicon-free 0.026–0.036, trees 0.09–0.30. Parakeet native 0.016–0.024, RASR 0.043–0.104. These are observed GPU job runtimes, not controlled hardware benchmarks.

Parakeet SPGISpeech rasr.recog.log has logical size 1.926 TB and approximately 138.86 GB allocated blocks; GigaSpeech has logical size 279.70 GB and approximately 20.17 GB allocated blocks. Avoid full-file reads. Disable verbose per-step diagnostics in subsequent production runs; retain sufficient segment summaries/tracebacks for comparisons.

## LibriSpeech Qwen status

Existing dev-other closed-tree scale sweep: scale 0 → 2.9% (1467 errors), 0.1 → 2.8% (1452), 0.3 → 2.8% (1422), 0.5 → 3.2% (1649). Deletions rise from 75 at scale 0 to 416 at 0.5. Text-variant lexica at scale 0.3/wordbeam 8 give 1423 errors, effectively no improvement; their RTF is around 6.2–7.1 versus 1.75–2.81 for the original lexicon. These dev scores use the historical scoring pipeline and must not be compared directly to Open ASR test scores.

Next priorities: verify complete-hypothesis Qwen score/tokenization parity and jointly tune LM scale and word-exit reward on LibriSpeech dev-other; only then broaden to extended/open search. For Cohere, define consistent native/RASR token-limit handling and inspect tree scoring/termination on the repeated-output counterexamples. Keep broad neural-LM benchmark runs held back while these issues are resolved.


