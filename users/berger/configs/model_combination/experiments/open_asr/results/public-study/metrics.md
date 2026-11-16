# Gains, uncertainty and runtime

Δ is fusion WER minus the better standalone constituent's WER, in percentage points (negative is better). Speech-Qwen baselines match the auxiliary model size. Text-Qwen has only the AM baseline. Intervals are the existing 10,000-replicate paired source-cluster bootstrap intervals; unadjusted and exploratory.

RTF is recorded recognition/rescoring time divided by audio duration. Historical hardware is not matched. Two-pass timing includes generation and auxiliary scoring, including the independent one-best parity decode where recorded; it excludes downloads and evaluation. Loquacious subset runtimes come from their parent run. The RTF <1.5 selection constraint applies to development data, not every test corpus.

## AMI

7,715 scoring units; 16 source clusters.

Few independent clusters; interval is fragile

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 7.363 | — | — | 0.201 |
| Granite + Speech 0.6B / 1p | 7.093 | -0.271 | [-0.373, -0.178] | 0.095 |
| Granite + Speech 0.6B / 2p | 7.190 | -0.173 | [-0.260, -0.095] | 0.343 |
| Granite + Speech 1.7B / 1p | 7.062 | -0.301 | [-0.380, -0.215] | 0.179 |
| Granite + Speech 1.7B / 2p | 7.169 | -0.194 | [-0.296, -0.097] | 0.528 |
| Granite + Text 0.6B / 1p | 7.495 | +0.132 | [-0.004, +0.261] | 0.114 |
| Granite + Text 0.6B / 2p | 7.597 | +0.233 | [+0.106, +0.342] | 0.236 |
| Granite + Text 1.7B / 1p | 7.476 | +0.113 | [-0.011, +0.228] | 0.157 |
| Granite + Text 1.7B / 2p | 7.522 | +0.159 | [+0.026, +0.287] | 0.267 |
| Parakeet | 13.295 | — | — | 0.109 |
| Parakeet + Speech 0.6B / 1p | 11.235 | +1.899 | [+1.551, +2.224] | 0.396 |
| Parakeet + Speech 0.6B / 2p | 11.072 | +1.737 | [+1.349, +2.068] | 0.504 |
| Parakeet + Speech 1.7B / 1p | 10.699 | +2.331 | [+1.901, +2.728] | 0.850 |
| Parakeet + Speech 1.7B / 2p | 10.182 | +1.814 | [+1.396, +2.196] | 0.664 |
| Parakeet + Text 0.6B / 1p | 13.492 | +0.197 | [+0.024, +0.350] | 0.544 |
| Parakeet + Text 1.7B / 1p | 13.360 | +0.065 | [-0.057, +0.206] | 0.673 |
| Speech Qwen 0.6B | 9.336 | — | — | 0.062 |
| Speech Qwen 1.7B | 8.367 | — | — | 0.097 |

## Earnings22

6 scoring units; 6 source clusters.

Few independent clusters; interval is fragile

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 8.471 | — | — | 0.217 |
| Granite + Speech 0.6B / 1p | 7.569 | +0.032 | [-0.865, +1.888] | 0.283 |
| Granite + Speech 0.6B / 2p | 7.887 | +0.349 | [-0.607, +2.366] | 0.281 |
| Granite + Speech 1.7B / 1p | 7.556 | +1.904 | [+0.152, +3.734] | 0.608 |
| Granite + Speech 1.7B / 2p | 7.620 | +1.968 | [+0.285, +3.745] | 0.369 |
| Granite + Text 0.6B / 1p | 8.246 | -0.225 | [-0.362, -0.102] | 0.154 |
| Granite + Text 0.6B / 2p | 8.393 | -0.077 | [-0.155, +0.000] | 0.237 |
| Granite + Text 1.7B / 1p | 8.161 | -0.310 | [-0.486, -0.134] | 0.324 |
| Granite + Text 1.7B / 2p | 8.283 | -0.188 | [-0.342, -0.059] | 0.251 |
| Parakeet | 8.882 | — | — | 0.198 |
| Parakeet + Speech 0.6B / 1p | 7.900 | +0.362 | [-0.539, +1.678] | 1.048 |
| Parakeet + Speech 0.6B / 2p | 8.421 | +0.883 | [-0.072, +2.186] | 0.792 |
| Parakeet + Speech 1.7B / 1p | 7.483 | +1.831 | [+0.929, +2.826] | 2.429 |
| Parakeet + Speech 1.7B / 2p | 7.952 | +2.299 | [+1.201, +3.515] | 0.923 |
| Parakeet + Text 0.6B / 1p | 9.310 | +0.428 | [-0.022, +0.969] | 0.575 |
| Parakeet + Text 1.7B / 1p | 9.128 | +0.245 | [-0.019, +0.497] | 1.034 |
| Speech Qwen 0.6B | 7.537 | — | — | 0.054 |
| Speech Qwen 1.7B | 5.652 | — | — | 0.086 |

## GigaSpeech

18,757 scoring units; 131 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 8.523 | — | — | 0.228 |
| Granite + Speech 0.6B / 1p | 7.980 | +0.317 | [+0.202, +0.427] | 0.153 |
| Granite + Speech 0.6B / 2p | 8.138 | +0.475 | [+0.365, +0.580] | 0.329 |
| Granite + Speech 1.7B / 1p | 7.928 | +0.639 | [+0.520, +0.774] | 0.316 |
| Granite + Speech 1.7B / 2p | 7.942 | +0.653 | [+0.537, +0.792] | 0.493 |
| Granite + Text 0.6B / 1p | 8.400 | -0.123 | [-0.207, -0.057] | 0.157 |
| Granite + Text 0.6B / 2p | 8.587 | +0.065 | [-0.006, +0.136] | 0.255 |
| Granite + Text 1.7B / 1p | 8.299 | -0.224 | [-0.311, -0.153] | 0.274 |
| Granite + Text 1.7B / 2p | 8.556 | +0.033 | [-0.051, +0.115] | 0.280 |
| Parakeet | 8.169 | — | — | 0.134 |
| Parakeet + Speech 0.6B / 1p | 7.680 | +0.018 | [-0.144, +0.159] | 0.562 |
| Parakeet + Speech 0.6B / 2p | 7.709 | +0.047 | [-0.113, +0.185] | 0.545 |
| Parakeet + Speech 1.7B / 1p | 7.526 | +0.237 | [+0.146, +0.335] | 1.339 |
| Parakeet + Speech 1.7B / 2p | 7.449 | +0.160 | [+0.082, +0.244] | 0.657 |
| Parakeet + Text 0.6B / 1p | 8.450 | +0.281 | [+0.205, +0.367] | 0.609 |
| Parakeet + Text 1.7B / 1p | 8.366 | +0.197 | [+0.128, +0.274] | 0.871 |
| Speech Qwen 0.6B | 7.662 | — | — | 0.062 |
| Speech Qwen 1.7B | 7.289 | — | — | 0.101 |

## LibriSpeech clean

2,620 scoring units; 87 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 1.457 | — | — | 0.158 |
| Granite + Speech 0.6B / 1p | 1.343 | -0.115 | [-0.208, -0.045] | 0.074 |
| Granite + Speech 0.6B / 2p | 1.386 | -0.072 | [-0.133, -0.018] | 0.267 |
| Granite + Speech 1.7B / 1p | 1.343 | +0.129 | [+0.012, +0.245] | 0.152 |
| Granite + Speech 1.7B / 2p | 1.361 | +0.148 | [+0.031, +0.264] | 0.439 |
| Granite + Text 0.6B / 1p | 1.380 | -0.077 | [-0.164, -0.011] | 0.087 |
| Granite + Text 0.6B / 2p | 1.410 | -0.047 | [-0.100, +0.000] | 0.185 |
| Granite + Text 1.7B / 1p | 1.344 | -0.113 | [-0.207, -0.039] | 0.122 |
| Granite + Text 1.7B / 2p | 1.386 | -0.072 | [-0.128, -0.020] | 0.212 |
| Parakeet | 1.089 | — | — | 0.129 |
| Parakeet + Speech 0.6B / 1p | 1.018 | -0.072 | [-0.118, -0.028] | 0.640 |
| Parakeet + Speech 0.6B / 2p | 1.040 | -0.049 | [-0.103, +0.004] | 0.471 |
| Parakeet + Speech 1.7B / 1p | 0.993 | -0.096 | [-0.138, -0.055] | 1.421 |
| Parakeet + Speech 1.7B / 2p | 0.989 | -0.100 | [-0.147, -0.055] | 0.521 |
| Parakeet + Text 0.6B / 1p | 1.123 | +0.034 | [-0.022, +0.090] | 0.579 |
| Parakeet + Text 1.7B / 1p | 1.097 | +0.008 | [-0.047, +0.065] | 0.882 |
| Speech Qwen 0.6B | 1.635 | — | — | 0.056 |
| Speech Qwen 1.7B | 1.213 | — | — | 0.092 |

## LibriSpeech other

2,939 scoring units; 90 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 2.596 | — | — | 0.189 |
| Granite + Speech 0.6B / 1p | 2.428 | -0.168 | [-0.251, -0.094] | 0.097 |
| Granite + Speech 0.6B / 2p | 2.426 | -0.170 | [-0.236, -0.107] | 0.314 |
| Granite + Speech 1.7B / 1p | 2.360 | -0.236 | [-0.326, -0.152] | 0.205 |
| Granite + Speech 1.7B / 2p | 2.443 | -0.153 | [-0.250, -0.066] | 0.480 |
| Granite + Text 0.6B / 1p | 2.587 | -0.010 | [-0.084, +0.063] | 0.097 |
| Granite + Text 0.6B / 2p | 2.557 | -0.040 | [-0.110, +0.028] | 0.220 |
| Granite + Text 1.7B / 1p | 2.524 | -0.072 | [-0.151, +0.003] | 0.151 |
| Granite + Text 1.7B / 2p | 2.517 | -0.080 | [-0.149, -0.014] | 0.247 |
| Parakeet | 2.099 | — | — | 0.121 |
| Parakeet + Speech 0.6B / 1p | 1.980 | -0.119 | [-0.177, -0.060] | 0.576 |
| Parakeet + Speech 0.6B / 2p | 2.076 | -0.022 | [-0.088, +0.048] | 0.446 |
| Parakeet + Speech 1.7B / 1p | 1.962 | -0.136 | [-0.199, -0.073] | 1.293 |
| Parakeet + Speech 1.7B / 2p | 1.987 | -0.112 | [-0.185, -0.038] | 0.504 |
| Parakeet + Text 0.6B / 1p | 2.142 | +0.044 | [-0.027, +0.116] | 0.569 |
| Parakeet + Text 1.7B / 1p | 2.129 | +0.030 | [-0.032, +0.099] | 0.856 |
| Speech Qwen 0.6B | 3.895 | — | — | 0.058 |
| Speech Qwen 1.7B | 2.862 | — | — | 0.094 |

## Loquacious

8,087 scoring units; 5,175 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 4.936 | — | — | 0.209 |
| Granite + Speech 0.6B / 1p | 4.614 | -0.322 | [-0.387, -0.258] | 0.105 |
| Granite + Speech 0.6B / 2p | 4.661 | -0.275 | [-0.328, -0.222] | 0.316 |
| Granite + Speech 1.7B / 1p | 4.561 | -0.375 | [-0.430, -0.276] | 0.214 |
| Granite + Speech 1.7B / 2p | 4.603 | -0.333 | [-0.391, -0.233] | 0.476 |
| Granite + Text 0.6B / 1p | 4.783 | -0.153 | [-0.212, -0.094] | 0.097 |
| Granite + Text 0.6B / 2p | 4.785 | -0.151 | [-0.208, -0.095] | 0.233 |
| Granite + Text 1.7B / 1p | 4.724 | -0.212 | [-0.274, -0.153] | 0.160 |
| Granite + Text 1.7B / 2p | 4.758 | -0.178 | [-0.236, -0.122] | 0.256 |
| Parakeet | 4.353 | — | — | 0.120 |
| Parakeet + Speech 0.6B / 1p | 4.218 | -0.135 | [-0.193, -0.077] | 0.519 |
| Parakeet + Speech 0.6B / 2p | 4.314 | -0.039 | [-0.108, +0.029] | 0.477 |
| Parakeet + Speech 1.7B / 1p | 4.163 | -0.190 | [-0.261, -0.121] | 1.142 |
| Parakeet + Speech 1.7B / 2p | 4.191 | -0.162 | [-0.233, -0.092] | 0.551 |
| Parakeet + Text 0.6B / 1p | 4.526 | +0.173 | [+0.099, +0.255] | 0.525 |
| Parakeet + Text 1.7B / 1p | 4.447 | +0.094 | [+0.028, +0.165] | 0.714 |
| Speech Qwen 0.6B | 6.144 | — | — | 0.048 |
| Speech Qwen 1.7B | 4.973 | — | — | 0.079 |

## commonvoice

3,044 scoring units; 3,044 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 6.737 | — | — | 0.209 (parent) |
| Granite + Speech 0.6B / 1p | 6.161 | -0.575 | [-0.731, -0.421] | 0.105 (parent) |
| Granite + Speech 0.6B / 2p | 6.262 | -0.474 | [-0.607, -0.343] | 0.316 (parent) |
| Granite + Speech 1.7B / 1p | 6.041 | -0.592 | [-0.785, -0.262] | 0.214 (parent) |
| Granite + Speech 1.7B / 2p | 6.063 | -0.570 | [-0.753, -0.251] | 0.476 (parent) |
| Granite + Text 0.6B / 1p | 6.150 | -0.586 | [-0.735, -0.442] | 0.097 (parent) |
| Granite + Text 0.6B / 2p | 6.156 | -0.581 | [-0.708, -0.449] | 0.233 (parent) |
| Granite + Text 1.7B / 1p | 6.095 | -0.642 | [-0.791, -0.493] | 0.160 (parent) |
| Granite + Text 1.7B / 2p | 6.111 | -0.626 | [-0.753, -0.496] | 0.256 (parent) |
| Parakeet | 5.713 | — | — | 0.120 (parent) |
| Parakeet + Speech 0.6B / 1p | 5.554 | -0.158 | [-0.260, -0.058] | 0.519 (parent) |
| Parakeet + Speech 0.6B / 2p | 5.586 | -0.127 | [-0.259, +0.007] | 0.477 (parent) |
| Parakeet + Speech 1.7B / 1p | 5.455 | -0.258 | [-0.373, -0.141] | 1.142 (parent) |
| Parakeet + Speech 1.7B / 2p | 5.388 | -0.325 | [-0.467, -0.183] | 0.551 (parent) |
| Parakeet + Text 0.6B / 1p | 5.703 | -0.010 | [-0.138, +0.138] | 0.525 (parent) |
| Parakeet + Text 1.7B / 1p | 5.685 | -0.027 | [-0.148, +0.099] | 0.714 (parent) |
| Speech Qwen 0.6B | 8.661 | — | — | 0.048 (parent) |
| Speech Qwen 1.7B | 6.633 | — | — | 0.079 (parent) |

## librispeech

2,747 scoring units; 90 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 2.581 | — | — | 0.209 (parent) |
| Granite + Speech 0.6B / 1p | 2.399 | -0.182 | [-0.271, -0.100] | 0.105 (parent) |
| Granite + Speech 0.6B / 2p | 2.399 | -0.182 | [-0.252, -0.117] | 0.316 (parent) |
| Granite + Speech 1.7B / 1p | 2.340 | -0.241 | [-0.337, -0.152] | 0.214 (parent) |
| Granite + Speech 1.7B / 2p | 2.413 | -0.168 | [-0.267, -0.077] | 0.476 (parent) |
| Granite + Text 0.6B / 1p | 2.565 | -0.016 | [-0.095, +0.060] | 0.097 (parent) |
| Granite + Text 0.6B / 2p | 2.538 | -0.042 | [-0.115, +0.026] | 0.233 (parent) |
| Granite + Text 1.7B / 1p | 2.508 | -0.073 | [-0.155, +0.006] | 0.160 (parent) |
| Granite + Text 1.7B / 2p | 2.492 | -0.089 | [-0.160, -0.023] | 0.256 (parent) |
| Parakeet | 2.098 | — | — | 0.120 (parent) |
| Parakeet + Speech 0.6B / 1p | 1.977 | -0.121 | [-0.182, -0.060] | 0.519 (parent) |
| Parakeet + Speech 0.6B / 2p | 2.084 | -0.014 | [-0.085, +0.061] | 0.477 (parent) |
| Parakeet + Speech 1.7B / 1p | 1.956 | -0.142 | [-0.208, -0.077] | 1.142 (parent) |
| Parakeet + Speech 1.7B / 2p | 1.984 | -0.113 | [-0.192, -0.035] | 0.551 (parent) |
| Parakeet + Text 0.6B / 1p | 2.154 | +0.057 | [-0.017, +0.134] | 0.525 (parent) |
| Parakeet + Text 1.7B / 1p | 2.132 | +0.034 | [-0.031, +0.105] | 0.714 (parent) |
| Speech Qwen 0.6B | 3.919 | — | — | 0.048 (parent) |
| Speech Qwen 1.7B | 2.846 | — | — | 0.079 (parent) |

## voxpopuli

1,827 scoring units; 1,827 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 5.637 | — | — | 0.209 (parent) |
| Granite + Speech 0.6B / 1p | 5.442 | -0.195 | [-0.293, -0.096] | 0.105 (parent) |
| Granite + Speech 0.6B / 2p | 5.455 | -0.182 | [-0.276, -0.089] | 0.316 (parent) |
| Granite + Speech 1.7B / 1p | 5.430 | -0.207 | [-0.302, -0.103] | 0.214 (parent) |
| Granite + Speech 1.7B / 2p | 5.434 | -0.203 | [-0.309, -0.087] | 0.476 (parent) |
| Granite + Text 0.6B / 1p | 5.548 | -0.089 | [-0.184, +0.007] | 0.097 (parent) |
| Granite + Text 0.6B / 2p | 5.537 | -0.100 | [-0.186, -0.016] | 0.233 (parent) |
| Granite + Text 1.7B / 1p | 5.496 | -0.141 | [-0.235, -0.046] | 0.160 (parent) |
| Granite + Text 1.7B / 2p | 5.521 | -0.116 | [-0.202, -0.032] | 0.256 (parent) |
| Parakeet | 5.018 | — | — | 0.120 (parent) |
| Parakeet + Speech 0.6B / 1p | 5.064 | +0.046 | [-0.051, +0.140] | 0.519 (parent) |
| Parakeet + Speech 0.6B / 2p | 5.217 | +0.198 | [+0.080, +0.314] | 0.477 (parent) |
| Parakeet + Speech 1.7B / 1p | 5.070 | +0.052 | [-0.049, +0.153] | 1.142 (parent) |
| Parakeet + Speech 1.7B / 2p | 5.179 | +0.161 | [+0.050, +0.270] | 0.551 (parent) |
| Parakeet + Text 0.6B / 1p | 5.273 | +0.255 | [+0.136, +0.386] | 0.525 (parent) |
| Parakeet + Text 1.7B / 1p | 5.159 | +0.141 | [+0.035, +0.248] | 0.714 (parent) |
| Speech Qwen 0.6B | 6.635 | — | — | 0.048 (parent) |
| Speech Qwen 1.7B | 5.804 | — | — | 0.079 (parent) |

## yodas

469 scoring units; 214 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 7.818 | — | — | 0.209 (parent) |
| Granite + Speech 0.6B / 1p | 7.051 | -0.415 | [-0.866, +0.099] | 0.105 (parent) |
| Granite + Speech 0.6B / 2p | 7.292 | -0.175 | [-0.593, +0.382] | 0.316 (parent) |
| Granite + Speech 1.7B / 1p | 7.042 | +0.300 | [-0.116, +0.820] | 0.214 (parent) |
| Granite + Speech 1.7B / 2p | 7.145 | +0.402 | [-0.028, +0.924] | 0.476 (parent) |
| Granite + Text 0.6B / 1p | 7.930 | +0.112 | [-0.155, +0.348] | 0.097 (parent) |
| Granite + Text 0.6B / 2p | 8.087 | +0.269 | [+0.002, +0.520] | 0.233 (parent) |
| Granite + Text 1.7B / 1p | 7.826 | +0.008 | [-0.271, +0.265] | 0.160 (parent) |
| Granite + Text 1.7B / 2p | 8.155 | +0.337 | [+0.039, +0.619] | 0.256 (parent) |
| Parakeet | 8.053 | — | — | 0.120 (parent) |
| Parakeet + Speech 0.6B / 1p | 7.232 | -0.234 | [-0.722, +0.273] | 0.519 (parent) |
| Parakeet + Speech 0.6B / 2p | 7.224 | -0.242 | [-0.809, +0.343] | 0.477 (parent) |
| Parakeet + Speech 1.7B / 1p | 6.913 | +0.170 | [-0.192, +0.544] | 1.142 (parent) |
| Parakeet + Speech 1.7B / 2p | 6.869 | +0.126 | [-0.317, +0.561] | 0.551 (parent) |
| Parakeet + Text 0.6B / 1p | 8.875 | +0.822 | [+0.277, +1.541] | 0.525 (parent) |
| Parakeet + Text 1.7B / 1p | 8.528 | +0.475 | [+0.001, +1.118] | 0.714 (parent) |
| Speech Qwen 0.6B | 7.466 | — | — | 0.048 (parent) |
| Speech Qwen 1.7B | 6.743 | — | — | 0.079 (parent) |

## Monsoon

2,102 scoring units; 2,102 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 4.286 | — | — | 0.232 |
| Granite + Speech 0.6B / 1p | 3.576 | -0.538 | [-0.696, -0.362] | 0.143 |
| Granite + Speech 0.6B / 2p | 3.645 | -0.469 | [-0.629, -0.296] | 0.338 |
| Granite + Speech 1.7B / 1p | 3.355 | -0.070 | [-0.213, +0.072] | 0.298 |
| Granite + Speech 1.7B / 2p | 3.410 | -0.015 | [-0.162, +0.128] | 0.507 |
| Granite + Text 0.6B / 1p | 4.280 | -0.006 | [-0.114, +0.103] | 0.120 |
| Granite + Text 0.6B / 2p | 4.537 | +0.251 | [+0.141, +0.364] | 0.254 |
| Granite + Text 1.7B / 1p | 4.133 | -0.153 | [-0.258, -0.047] | 0.211 |
| Granite + Text 1.7B / 2p | 4.509 | +0.223 | [+0.111, +0.331] | 0.277 |
| Parakeet | 4.421 | — | — | 0.126 |
| Parakeet + Speech 0.6B / 1p | 3.553 | -0.561 | [-0.754, -0.377] | 0.574 |
| Parakeet + Speech 0.6B / 2p | 3.547 | -0.567 | [-0.752, -0.384] | 0.539 |
| Parakeet + Speech 1.7B / 1p | 3.279 | -0.146 | [-0.294, -0.001] | 1.341 |
| Parakeet + Speech 1.7B / 2p | 3.254 | -0.171 | [-0.316, -0.023] | 0.695 |
| Parakeet + Text 0.6B / 1p | 5.422 | +1.001 | [+0.850, +1.158] | 0.530 |
| Parakeet + Text 1.7B / 1p | 5.316 | +0.894 | [+0.747, +1.046] | 0.751 |
| Speech Qwen 0.6B | 4.114 | — | — | 0.048 |
| Speech Qwen 1.7B | 3.425 | — | — | 0.081 |

## SPGISpeech

39,341 scoring units; 1,114 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 1.841 | — | — | 0.196 |
| Granite + Speech 0.6B / 1p | 1.705 | -0.136 | [-0.151, -0.121] | 0.077 |
| Granite + Speech 0.6B / 2p | 1.752 | -0.089 | [-0.102, -0.075] | 0.287 |
| Granite + Speech 1.7B / 1p | 1.691 | -0.150 | [-0.165, -0.135] | 0.158 |
| Granite + Speech 1.7B / 2p | 1.820 | -0.021 | [-0.037, -0.005] | 0.423 |
| Granite + Text 0.6B / 1p | 1.782 | -0.059 | [-0.075, -0.043] | 0.091 |
| Granite + Text 0.6B / 2p | 1.821 | -0.020 | [-0.034, -0.005] | 0.216 |
| Granite + Text 1.7B / 1p | 1.760 | -0.081 | [-0.097, -0.064] | 0.071 |
| Granite + Text 1.7B / 2p | 1.814 | -0.027 | [-0.041, -0.012] | 0.214 |
| Parakeet | 2.679 | — | — | 0.136 |
| Parakeet + Speech 0.6B / 1p | 2.466 | -0.213 | [-0.233, -0.190] | 0.581 |
| Parakeet + Speech 0.6B / 2p | 2.431 | -0.249 | [-0.271, -0.223] | 0.549 |
| Parakeet + Speech 1.7B / 1p | 2.550 | -0.019 | [-0.050, +0.012] | 1.318 |
| Parakeet + Speech 1.7B / 2p | 2.537 | -0.033 | [-0.060, -0.006] | 0.655 |
| Parakeet + Text 0.6B / 1p | 2.856 | +0.177 | [+0.152, +0.203] | 0.610 |
| Parakeet + Text 1.7B / 1p | 2.860 | +0.181 | [+0.156, +0.207] | 0.445 |
| Speech Qwen 0.6B | 2.705 | — | — | 0.052 |
| Speech Qwen 1.7B | 2.569 | — | — | 0.085 |

## VoxPopuli

628 scoring units; 292 source clusters.

| Condition | WER | Δ | 95% CI for Δ | RTF |
| --- | ---: | ---: | ---: | ---: |
| Granite | 4.635 | — | — | 0.186 |
| Granite + Speech 0.6B / 1p | 3.741 | +0.781 | [+0.496, +1.058] | 0.152 |
| Granite + Speech 0.6B / 2p | 3.836 | +0.876 | [+0.584, +1.159] | 0.280 |
| Granite + Speech 1.7B / 1p | 3.741 | +0.889 | [+0.643, +1.144] | 0.324 |
| Granite + Speech 1.7B / 2p | 3.701 | +0.849 | [+0.606, +1.098] | 0.405 |
| Granite + Text 0.6B / 1p | 4.848 | +0.213 | [+0.049, +0.375] | 0.135 |
| Granite + Text 0.6B / 2p | 4.882 | +0.247 | [+0.112, +0.378] | 0.204 |
| Granite + Text 1.7B / 1p | 4.809 | +0.174 | [+0.021, +0.325] | 0.239 |
| Granite + Text 1.7B / 2p | 4.826 | +0.191 | [+0.052, +0.329] | 0.224 |
| Parakeet | 3.776 | — | — | 0.309 |
| Parakeet + Speech 0.6B / 1p | 3.190 | +0.231 | [-0.043, +0.497] | 0.686 |
| Parakeet + Speech 0.6B / 2p | 3.043 | +0.084 | [-0.190, +0.349] | 0.576 |
| Parakeet + Speech 1.7B / 1p | 3.105 | +0.253 | [+0.037, +0.473] | 1.590 |
| Parakeet + Speech 1.7B / 2p | 3.004 | +0.151 | [-0.064, +0.368] | 0.672 |
| Parakeet + Text 0.6B / 1p | 4.344 | +0.568 | [+0.405, +0.735] | 0.563 |
| Parakeet + Text 1.7B / 1p | 4.288 | +0.511 | [+0.345, +0.682] | 0.854 |
| Speech Qwen 0.6B | 2.960 | — | — | 0.050 |
| Speech Qwen 1.7B | 2.852 | — | — | 0.082 |
