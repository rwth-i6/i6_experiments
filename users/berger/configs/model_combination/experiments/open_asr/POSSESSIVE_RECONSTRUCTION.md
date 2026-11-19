# Granite possessive reconstruction

Run `./sis m config/possessive_reconstruction.py` from the Sisyphus setup.
The report is published as `output/reports/asr-lm/possessive-reconstruction.md`.
The JSON summary, output hypotheses and Speech-Qwen edit scores are under
`output/possessive-reconstruction/f4212424889ee5d3/`.

There are two GPU jobs, one for LibriSpeech dev-other and one for Loquacious
dev.short, using the frozen Speech-Qwen 1.7B checkpoint and formatting policy.
For each frozen Granite two-pass winner, the scorer compares the original text
with alternatives that change one eligible `word is` to `word's`. It scores
whole transcripts with the original audio and EOS. The CPU report applies the
lowest-cost edit only if its NLL improvement exceeds 0, 2, or 5 nats. This is
output reconstruction guided by Speech-Qwen, with no new AM score for edited
text. It uses one edit per utterance and retains the original when the margin
is unmet.

Native Granite has a separate text-only baseline. The deterministic rule takes
the first eligible edit after a preposition, optionally with an intervening
`the`, excluding pronoun owners and common continuations of genuine `is`.
This heuristic requires neither audio nor model rescoring. On the existing
development outputs it gives the following exploratory results.

| Corpus | Native WER % | Native + rule WER % | Restored s | Damaged correct is |
|---|---:|---:|---:|---:|
| LibriSpeech dev-other | 2.732 | 2.635 | 56 | 13 |
| Loquacious dev.short | 4.778 | 4.720 | 49 | 25 |

The report includes paired WER intervals and checks the frozen 50-best pool
for a correct owner and suffix at anchored `s` → `is` errors in the two-pass
winner. This localized, alignment-based check found coverage at 44 of 190
sites on dev-other and 43 of 140 on dev.short. Word attribution uses a common
normalized reference grid. The `s` sites are a proxy for possessive rendering
and the coverage count is independent of the rest of the candidate's accuracy.

Native Parakeet already emits apostrophes and contractions. In the same
normalized alignment, where the preceding reference word matched, native
Parakeet produced zero `s` → `is` substitutions on both development sets.
Native Granite produced 197 on dev-other and 157 on dev.short. The pilot
therefore targets Granite.

## Public test evaluation

Run `./sis m config/possessive_evaluation.py` from the Sisyphus setup.
This evaluation uses a single 2-nat margin from development. It compares
native Granite, native Granite with the text rule, literal Granite two-pass,
Granite two-pass with Speech-Qwen reconstruction, and standalone Speech-Qwen
1.7B. The nine base datasets each have one new Speech-Qwen scoring job on the
48 GB queue. The four Loquacious source subsets reuse the parent edit scores.
All original hypotheses come from completed public runs.

The report uses the full test-set denominator. For chunked recordings it
reconstructs individual chunks and joins them in benchmark order before scoring.
It verifies the three frozen baseline scores and reports paired WER changes
against the original output and the gap between reconstruction and standalone
Speech-Qwen. It also records restored `s` and damaged correct `is` counts.

Read `output/reports/asr-lm/possessive-evaluation.md` for the report, or
`output/possessive-evaluation/f4212424889ee5d3/` for JSON and edited hypotheses.
