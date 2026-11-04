# Granite Open ASR extended-tree audit, 2026-10-05

59 of the 60 newly registered conditions have scoring results. The remaining
condition, the matched closed-tree Transformer control on SPGISpeech, timed out.
Scores below are WER percentages with the current HF normalizer. Native and
RASR lexicon-free Granite scores agree. Tuning settings were transferred from
LibriSpeech development data; these test results should not be used for tuning.

## LibriSpeech lexicon and transferred word LMs

| Dataset | Native/free | Closed, no LM | Open, no LM | Closed, 4gram | Open, 4gram | Closed, Transformer | Open, Transformer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LibriSpeech test-clean | 1.46 | 2.04 | 1.46 | 1.78 | 1.43 | 1.74 | 1.39 |
| LibriSpeech test-other | 2.67 | 3.04 | 2.60 | 2.94 | 2.56 | 2.83 | 2.50 |
| AMI | 7.35 | 8.30 | 7.36 | 8.45 | 7.85 | 8.40 | 7.80 |
| GigaSpeech | 8.46 | 11.39 | 8.52 | 10.58 | 9.17 | 10.53 | 9.01 |
| SPGISpeech | 1.76 | 4.13 | 1.84 | 3.41 | 1.98 | timeout | 1.92 |
| VoxPopuli | 4.69 | 7.28 | 4.63 | 6.09 | 4.97 | 6.12 | 4.96 |
| Monsoon | 4.32 | 5.20 | 4.29 | 5.96 | 5.57 | 5.89 | 5.45 |
| Earnings22 | 8.34 | 21.43 | 8.46 | 13.97 | 9.04 | 16.48 | 8.70 |

Closed columns are matched controls using the same prepared lexicon and pruning
as the corresponding open condition, not the older tree runs. Preparation removes
unsupported multi-boundary pronunciation variants. No additional subword LM is used.

The open tree recovers most of the closed-tree degradation. External LibriSpeech
LMs improve LibriSpeech but still degrade every completed non-LibriSpeech dataset
relative to open decoding without an LM. On test-other the Transformer reduces
errors from 1412 (native) to 1318, approximately 6.7% relative WER reduction.
Native I/D/S: 121/103/1188; open Transformer: 92/119/1107. Small differences in
reference word counts arise from hypothesis-dependent compound alignment; tiny
WER differences should not be treated as significant without paired analysis.

## Corpus-specific lexica and available LMs

| Dataset / lexicon | LM | Matched closed | Open |
| --- | --- | ---: | ---: |
| GigaSpeech / GigaSpeech | none | 9.46 | 8.45 |
| GigaSpeech / GigaSpeech | GigaSpeech 4gram | 9.29 | 8.89 |
| GigaSpeech / GigaSpeech | LibriSpeech Transformer | 9.57 | 8.99 |
| VoxPopuli / VoxPopuli | none | 4.97 | 4.57 |
| VoxPopuli / VoxPopuli | LibriSpeech 4gram | **invalid: 99.90** | **invalid: 98.57** |
| VoxPopuli / VoxPopuli | LibriSpeech Transformer | **invalid: 4.94** | **invalid: 4.58** |

There is no usable VoxPopuli ARPA in this setup: the available public LM is a
KenLM binary unsupported by the current loader. There are no corpus-specific
Transformer conditions; those rows use the LibriSpeech Transformer.

The GigaSpeech-specific 4gram improves on the transferred LibriSpeech 4gram
(8.89 versus 9.17), but not on no-LM decoding (8.45).

### Invalid VoxPopuli LM mapping

VoxPopuli ordinary lemmata have lowercase orthographies and no explicit syntactic
sequence, so RASR derives lowercase LM tokens. The LibriSpeech LMs use uppercase.
Of 87,568 ordinary VoxPopuli LM tokens, **zero** match LibriSpeech ARPA vocabulary
exactly; 49,697 match after uppercasing. By contrast GigaSpeech has 125,856 exact
matches, unchanged by uppercasing.

The Transformer log explicitly reports `did not find: a using unknown word token
output 1`, and equivalent messages for ordinary lowercase words. Thus its seemingly
reasonable VoxPopuli scores do not demonstrate proper contextual word-LM fusion.
The extremely fast runtime of these conditions is also misleading.

Repair should preserve orthographies/acoustic pronunciations but explicitly map
syntactic tokens to the transferred LM vocabulary, including true OOVs to its
verified UNK token. Check ordinary-word coverage, not only singleton UNK coverage,
then rerun these four conditions. Avoid changing shared data fingerprints for
unaffected existing experiments.

## Runtime and failure

Valid open conditions have RTF ranges approximately 0.007–0.011 without an LM,
0.054–0.063 with the 4gram, and 0.38–0.51 with the Transformer. No completed new
condition shows the previous explosive Transformer stall.

`HfBenchmarkRecogJob.Rq7BXpKrvtM4` (SPGISpeech matched closed Transformer) was
cancelled by Slurm at its time limit, after 25,480/39,341 utterances with aggregate
RTF 0.629. Its log ends with `DUE TO TIME LIMIT`; there is no scoring result.
At this operating point a 48-hour allowance is insufficient. Existing AED failures
are older/backlogged conditions, not failures of these new Granite runs.

Source artifacts: `output/open-asr/granite-ctc/<condition>/<dataset>/scoring`,
`runtime`, `hypotheses`, and `segments`; job logs under corresponding
`alias/open-asr/granite-ctc/<condition>/<dataset>/` links.

Next steps: repair and validate VoxPopuli LM token mapping; rerun the timed-out
SPGISpeech control with sufficient time; retain these valid results and use
development splits for any domain-specific tuning.
