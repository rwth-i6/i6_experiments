# Completed contraction rescoring — 2026-11-19

**No meaningful or consistently beneficial effect at the frozen settings.**
All four auxiliary scoring jobs and CPU analyses completed. The source summary
and results are pinned in [review-20261119.json](results/contraction-rescoring/review-20261119.json).
The generated full report is output/reports/asr-lm/contraction-rescoring.md.

| AM / development set | Literal WER | Best-of-three WER | Delta [95% paired CI], points | Outputs changed |
| --- | ---: | ---: | ---: | ---: |
| Granite / LibriSpeech other | 2.491 | 2.485 | -0.00583 [-0.01287, 0.00000] | 4 / 2864 |
| Granite / Loquacious short | 4.263 | 4.265 | +0.00201 [-0.00422, +0.00942] | 5 / 3000 |
| Parakeet / LibriSpeech other | 1.935 | 1.935 | 0.00000 [0.00000, 0.00000] | 3 / 2864 |
| Parakeet / Loquacious short | 3.811 | 3.813 | +0.00201 [0.00000, +0.00638] | 8 / 3000 |

Granite's small LibriSpeech change is three fewer word errors (1282 to 1279),
without a comparable Loquacious gain (one extra error). Parakeet retains the
same LibriSpeech error count and adds one Loquacious error. All intervals are
exploratory and unadjusted; this is not evidence of a useful improvement.

Forcing contracted wording slightly worsens all four aggregate results.
Expansion is a no-op for Granite and slightly worsens Parakeet. Untouched-pool
hypotheses remain unchanged in every condition. There is no clear rescue in the
Speech-Qwen-contraction slices either: Granite's dev-other WER there changes
4.540 to 4.530; dev.short changes 6.002 to 6.026. Parakeet's best-of-three
contraction-slice WER is unchanged on both sets.

## Interpretation

Choosing a different LM wording rarely changes the winning AM transcription.
For Granite, the winning LM rendering is contracted in 208 dev-other units and
150 dev.short units, but only four/five output hypotheses change. For Parakeet,
nonliteral renderings win only 18/25 units, and three/eight outputs change.
Lower Speech-Qwen NLL for a rendering therefore does not establish a useful
change in ranking among the available AM candidates.

The experiment does not show that transcription conventions are irrelevant.
It tests a finite set of unambiguous negative/auxiliary contractions, with fixed
AM pools, scales, formatting and original-AM-word rewards. Ambiguous 's/'d,
possessives, numbers, hyphens and arbitrary mixtures of local variants are not
covered. It cannot recover a correct transcription missing from the n-best pool.
Whole-string rescoring also does not diagnose effects of delayed scoring or
history choices on first-pass pruning. In particular, the AppTek convention
mapping remains necessary for a fair fixed-hypothesis score comparison; that is
a different question from whether changing only the auxiliary rendering improves
recognition.

## Validation and decision

All literal hypotheses exactly match the frozen controls. The four GPU token
replay gates passed; maximum token-cost differences were about 7e-5 to 1.1e-4,
well below their declared tolerance. Thus there is no observed implementation
or numerical parity failure that explains the negligible WER changes.

Keep literal rescoring as the default. These results do not justify implementing
multiword contraction rewriting/history rollback in first-pass search solely on
this evidence. Retain the finding as a negative ablation for the tested core
contractions, not a rejection of other representation mismatches.

## Follow-up: the large deficit is predominantly possessive 's

The user's follow-up exposed a limitation in the coarse cohort interpretation:
the original contraction detector includes possessive 's, whereas the new
rescoring rules deliberately leave ambiguous 's/'d unchanged. A breakdown by
actual supported contracted forms (using expansion rules only, not the presence
of an expanded phrase) gives:

| Speech-output cohort | Units | Granite native | Granite fusion | Speech-Qwen |
| --- | ---: | ---: | ---: | ---: |
| dev-other / core only | 188 | 1.804 | 1.804 | 2.188 |
| dev-other / ambiguous only | 282 | 6.648 | 6.104 | 3.504 |
| dev.short / core only | 136 | 3.689 | 3.149 | 3.288 |
| dev.short / ambiguous only | 252 | 8.301 | 7.558 | 5.332 |

The large error is already present in native Granite. On the aligned ambiguous
cohort, the dominant additional reference-word error relative to Speech-Qwen is
reference `s` recognized as `is`: 210 of 259 such Granite-fusion errors on
dev-other and 155 of 195 on dev.short. These counts exclude reference-grid
mismatches and insertions and do not represent all corpus errors.

For example, segment `3660-172183-0000` has reference **Arthur's court**, but
Granite emits **Arthur is court**. The shared normalizer produces `arthur s
court` versus `arthur is court`. Other examples are **Anderson's people** and
**the report's references**, rendered as **Anderson is people** and **the report
is references**. This is a systematic possessive rendering error/mismatch;
the data does not establish which training normalization procedure caused it.

Three inspected frozen Granite pools contain no correct possessive variant
(including joined-s and separate-s spellings): 37 candidates for Arthur's court,
42 for Anderson's people, 27 for the report's references. This is a spot check,
not a corpus-wide coverage finding. Changing only the auxiliary rendering while
holding AM output strings fixed cannot repair those missing output alternatives.

Consequently, the negative core-contraction ablation does not explain away the
observed mismatch. A targeted next investigation is possessive output
representation/reconstruction or ensuring correct possessive alternatives enter
the pool/search. A blind `is` to possessive rewrite would corrupt genuine verbs;
any candidate-based reconstruction must retain the original, account for the AM
surface separately and be evaluated as a distinct system change. The company-side
fixed-hypothesis diagnostic remains useful in parallel. No general alias/model
grid is justified by these three spot checks.

Numerical decomposition and inspection details:
[possessives-20261119.json](results/contraction-rescoring/possessives-20261119.json).
