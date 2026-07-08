# Chunked CTC / Conformer streaming ASR: results

Loquacious (large subset), CTC-only recognition.
WER [%] on the multi-domain dev / test sets,
aggregated over the domains, at the last epoch.

| Model | Streaming | dev | test |
| --- | --- | --- | --- |
| base (offline conformer) | no | 7.32 | 8.10 |
| base, 2x training | no | 6.58 | 7.39 |
| chunked L80-C5-R4 (fixed chunk) | yes | 9.46 | 10.29 |
| chunked, dyn-rope-ctembed | yes | 9.41 | 10.29 |
| chunked, dyn-rope-ctembed, 2x training | yes | 8.52 | 9.25 |
