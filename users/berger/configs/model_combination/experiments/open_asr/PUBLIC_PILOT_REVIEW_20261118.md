# Completed public pilots — 2026-11-18

Both opt-in configs completed. This review pins the source summaries and selected
numerical evidence in [review-20261118.json](results/public-pilots/review-20261118.json).
The original generated reports remain in output/reports/asr-lm/noise-pilot.md
and transcription-conventions.md. No new recognition run is proposed here.

## Controlled white noise

256 development utterances, 91 chapters, 0.54 hours per condition. All nine
recognitions completed. Checksums of all nine score/segment/runtime inputs match
the report manifest. Measured SNR deviations are below 2.8e-8 dB. Four utterances
have a common gain below one; that same gain was applied to clean and noisy copies.

| Condition | Parakeet WER | Speech-Qwen 1.7B WER | Fusion WER | Relative gain vs better | Absolute delta [95% CI] | Fusion RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 2.095 | 2.834 | 1.862 | 11.1% | -0.233 [-0.409, -0.062] | 1.377 |
| 20 dB | 2.367 | 3.531 | 2.249 | 5.0% | -0.117 [-0.326, +0.093] | 1.385 |
| 10 dB | 4.306 | 5.064 | 3.879 | 9.9% | -0.428 [-0.725, -0.118] | 1.367 |

Fusion retains a benefit on this synthetic perturbation, with intervals excluding
zero for clean and 10 dB. The 20 dB result is inconclusive. Parakeet stays the
stronger standalone at all levels; this does not test the weaker-AM/stronger-
Speech-Qwen regime responsible for many AppTek regressions.

There is **no demonstrated increase in fusion benefit with noise**. The larger
absolute gain at 10 dB is not a larger relative gain than clean. A joint chapter
bootstrap across all nine conditions gives the change in delta relative to clean:

- 20 dB: +0.116 points, CI [-0.092, +0.322].
- 10 dB: -0.194 points, CI [-0.510, +0.142].

Both intervals include zero. Primary and contrast intervals are exploratory and
unadjusted. Noise is stationary Gaussian with one frozen draw per utterance;
this is not evidence for general environmental-noise robustness. Fusion RTF is
stable around 1.37, versus 0.13 for Parakeet and 0.09-0.11 for Speech-Qwen on
these runs. Those are observed costs, not hardware-controlled speedups.

At 10 dB, fusion reduces substitutions relative to Parakeet (183 to 154) but
increases deletions (21 to 27) and insertions (18 to 19). The gain is therefore
not a general suppression of deletion errors.

The cross-condition contrasts can be reproduced with
[noise_gain_changes.py](../../scripts/noise_gain_changes.py), passing the completed
noise-pilot summary JSON. It verifies the per-segment evidence checksums and
jointly resamples the same source chapters across all conditions.

## Transcription conventions

All 105 records completed: 75 selected-condition records and 30 native controls.
The native controls confirm that uppercasing of Granite's selected tree control
comes from lemma orthography, rather than the native model. Native Granite and
Granite fusion have no contractions on the two public development sets; Parakeet
and Speech-Qwen do. This supports taking convention matching seriously when
forced-scoring a Speech-Qwen transcription under Granite.

The strongest descriptive pattern is Granite's deficit on units where Speech-
Qwen emits contractions:

| Dataset / S-output slice | Units | Granite control | Speech-Qwen | Granite fusion |
| --- | ---: | ---: | ---: | ---: |
| dev-other / contraction | 508 | 4.704 | 3.084 | 4.530 |
| dev-other / no contraction | 2356 | 2.113 | 2.559 | 1.950 |
| dev.short / contraction | 416 | 6.634 | 4.671 | 6.064 |
| dev.short / no contraction | 2584 | 4.326 | 4.614 | 3.881 |
| Loquacious test / contraction | 1105 | 6.869 | 5.096 | 6.594 |
| Loquacious test / no contraction | 6982 | 4.539 | 4.948 | 4.144 |

Fusion improves Granite within both groups, but Speech-Qwen remains much stronger
on the contraction group. These are whole-scoring-unit WERs conditioned on model
output, not errors attributed directly to a contraction; difficulty, lexical
content and model correctness can differ between groups. There are no subgroup
confidence intervals or causal conclusions.

This pattern does **not** explain every fusion failure. Parakeet on GigaSpeech
loses to Speech-Qwen by almost exactly the same amount with/without contractions
(+0.238/+0.235 points). On AMI it loses in both groups (+2.593/+2.117). Its
LibriSpeech and Loquacious gains also occur in both groups.

A separate target is SPGISpeech: on 2,751 units where Speech-Qwen emits digits,
Parakeet fusion is 2.229 versus Speech-Qwen's 1.913 (+0.316 points); without
digits it is 2.578 versus 2.627 (-0.049). Number formatting/lexical content is a
candidate for targeted investigation, not an established mechanism. The
number-word detector is lexical and includes ambiguous words such as 'oh'.

## Decision

Keep the completed white-noise pilot as a limited robustness result; its current
pattern does not justify a large noise grid or noise-specific tuning. Use the
convention audit to choose targeted public cases once the AppTek forced scorer
has passed its convention-variant gate. The main pending diagnostic remains
whether lower-cost, more accurate, realizable alternatives were lost by search
or are disfavored by the fusion objective. Reverse search, extra models and a
full-suite noise campaign remain deferred.
