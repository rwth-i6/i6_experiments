# Final public experiment completion — 2026-11-14

The frozen selected evaluation (`f4212424889ee5d3`) now has **234 of 234 cells**
across 13 datasets/subsets and 18 model/method families, with zero missing
standalone baselines. Overlapping parent/subset datasets are not pooled.
This completes the two pending cells in the [November 10 review](COMPLETED_REPORT_REVIEW_20261110.md).
The results remain exploratory: these test sets have already been inspected.

## Last SPGISpeech results

| Condition | WER % | End-to-end RTF |
| --- | ---: | ---: |
| Granite AM-only selected control | 1.841 | 0.196 |
| Granite + Speech-Qwen 1.7B one-pass | 1.691 | 0.158 |
| Parakeet AM-only selected control | 2.679 | 0.136 |
| Speech-Qwen 0.6B standalone, beam 8/lp 0 | 2.705 | 0.052 |
| Parakeet + Speech-Qwen 0.6B one-pass | 2.466 | 0.581 |
| Parakeet + Speech-Qwen 0.6B two-pass | **2.431** | **0.549** |
| Speech-Qwen 1.7B standalone, beam 8/lp 0 | 2.569 | 0.085 |
| Parakeet + Speech-Qwen 1.7B one-pass | 2.550 | 1.318 |
| Parakeet + Speech-Qwen 1.7B two-pass | 2.537 | 0.655 |

The 0.6B two-pass condition improves by 0.249 WER points against its better
standalone constituent (paired 95% CI [-0.271, -0.223]). The 1.7B one-pass gain
is only 0.019 points (CI [-0.050, +0.012]); two-pass gains 0.033 points
(CI [-0.060, -0.006]). These are unadjusted exploratory cluster-bootstrap
intervals, using 1,114 identifier-derived source clusters and 39,341 scoring units.
The two sizes use their separately development-selected fusion settings; this
is not a size-only ablation. Historical runtimes are not hardware-matched.

The final result supplies a useful counterexample to uniform auxiliary-model
scaling: standalone Speech-Qwen 1.7B is stronger, yet Parakeet fusion with 0.6B
is more accurate and faster at these selected operating points. Granite is
stronger than either Parakeet/Speech-Qwen pair on this corpus. The broad conclusion
remains conditional gains, competitive two-pass rescoring, and important failures
when comparing against the stronger standalone constituent. No new experiment
grid is needed to finish this descriptive study.

## CTC scorer-API parity: exact match

The submitted standalone parity run completed on November 11. All 2,864
LibriSpeech dev-other segments match between `9804f5aa` and `d000386c`:

- Zero raw-text, normalized-text or token-path differences.
- Zero cumulative AM/LM score differences on those paths.
- Identical I/D/S = 94/68/1,107, reference words = 51,460.
- Identical unrounded WER = 2.465993004275165%.

The new job used FP32, the selected public prior at scale 0.1, Speech-Qwen 1.7B
and transformers 5.16.1. The complete machine-readable evidence and checksums are
in [ctc_api_parity_result.json](ctc_api_parity_result.json). This establishes parity
for that transferred configuration and environment, not for arbitrary settings,
other corpora or transformers 5.19.0. The API change itself is no longer an
unmeasured obstacle for the AppTek transfer.

## Repository/analysis next step

Once the manager and all workers are stopped, preserve the live checkout's dirty
state and switch to `berger_dev`, keeping settings, dependencies and work/output
locations. Recheck the graph against the original job-ID inventory before
resubmitting anything. Regenerate the CPU complementarity reports with the
multi-fusion attribution fix; the old report's per-fusion correction/regression
counters still need that fix, while its WERs and standalone selection oracles
remain valid. Continue consolidating the study and company-side setup; additional
calibration, union-pool or noise studies remain optional.
