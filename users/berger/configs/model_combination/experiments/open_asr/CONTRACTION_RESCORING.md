# Contraction wording in Speech-Qwen rescoring

This development-only experiment tests whether presenting equivalent contraction
wordings to Speech-Qwen improves ranking, while leaving acoustic hypotheses
unchanged. It is independent of the company-side forced-scoring work.

```bash
./sis m config/contraction_rescoring.py
```

The graph reuses the completed frozen 50-best pools for Granite CTC and Parakeet
RNN-T on LibriSpeech dev-other and Loquacious dev.short. It schedules **four
additional Speech-Qwen 1.7B GPU scoring jobs** (48 GB queue, 24-hour limits),
plus CPU selection, scoring and one report. No AM forward/search, n-best
generation, builds, downloads, test-set evaluations or parameter sweeps are added.
The large main config is unchanged.

## Compared conditions

| Mode | Text shown to Speech-Qwen |
| --- | --- |
| literal | The exact frozen literal rendering |
| contracted | Deterministic phrase contractions, negative forms first |
| expanded | Apply all supported contraction-to-phrase rules |
| best | Lowest whole-string Speech NLL among those three styles; literal wins ties |

The LM rendering is **not** the output transcript. The selected candidate's
original AM text is emitted in every condition and scored normally. Its AM cost,
retained first-pass LM/fallback cost and word reward are unchanged:

```
cost = original_first_pass_cost
       + frozen_speech_scale * SpeechNLL(rendering | audio)
       + frozen_word_penalty * number_of_original_AM_words
```

Thus `you are right` keeps three AM words and its original acoustic score even
if Speech-Qwen sees `You're right.`. Every rendering is teacher-forced as a full
string under the same audio prompt, including one assistant-turn EOS. The
experiment does not replay the first-pass algorithm's greedy per-word history;
it tests a different rescoring objective before any search implementation change.

Frozen settings are taken directly from selection protocol
`f4212424889ee5d390df7a87d917b77b65d35c1a15e7c4920c3599c3c1769a56`:

| AM | Speech scale | Original-AM-word penalty | Formatting | Pool cap |
| --- | ---: | ---: | --- | ---: |
| Granite CTC | 0.5 | -1 | initial-capital-period | 50 |
| Parakeet RNN-T | 0.3 | -1 | initial-capital-period | 50 |

These are the frozen **two-pass** settings, not the one-pass scales. No
condition-specific tuning is done in this experiment.

## Rules and limits

Fixed, reference-independent rules include `you are`/`you're`, `we will`/`we'll`,
`I am`/`I'm`, explicit pronoun + `have`/`'ve`, and negative forms such as
`do not`/`don't`, `cannot`/`can't` and `will not`/`won't`. Case is preserved before
the existing frozen formatting policy is applied. Curly apostrophes are accepted
for these rules. Core aliases are checked for HF-normalized content equivalence.

Ambiguous `'s` and `'d`, possessives, chained contractions, hyphens, titles and
number forms are not changed. This intentionally does not cover every mismatch
seen on AppTek. The three styles are not an exhaustive search over mixtures of
local choices (for example, `you're not` versus `you aren't`). Taking a minimum is a diagnostic best-of-three score, not a
normalized probability marginal over equivalent spellings.

## Gates and reports

Cached literal NLLs are reused exactly. Any transformed string already present
as a literal in the same segment reuses that score too; only new strings need
additional scoring. The first eight segments of each GPU job also recompute the
literal scores and compare every token cost against the cache, with the same
atol=0.002/rtol=0.0002 tolerance as the existing backend gate. A discrepancy stops
the job. The backend's full-forward audio/EOS gate is also retained.

CPU selection must reproduce every frozen literal hypothesis exactly. Pools
without supported wording changes must produce identical hypotheses in all
conditions. Tests protect these gates and check that transformed LM word counts
never change the original AM word reward or output text.

The CPU preflight matched all 11,728 frozen literal hypothesis entries across
the four model/dev combinations. This does not replace the GPU token-score gate,
which runs when the new scoring jobs start.

Outputs:

- `output/reports/asr-lm/contraction-rescoring.md`
- `output/contraction-rescoring/f4212424889ee5d3/{report.md,summary.json}`
- Per AM/dev: scored renderings, gate results, selection ledger, original-AM-text
  hypotheses and WER summaries for all four modes.

The report includes I/D/S, paired cluster-bootstrap intervals against literal
rescoring and against the better matching standalone constituent, changed
hypothesis counts, winning rendering counts, and overlapping descriptive slices
for eligible/untouched pools and Speech-Qwen contraction/non-contraction outputs.
The AM-only comparator is the first pass of the same n-best generator, which can
differ from the independently selected standalone beam control.

Runtime is labeled **additional scoring**: it includes the new strings and
literal replay gate but excludes cached generation and original Speech scoring.
It must not be presented as an end-to-end speedup. All findings remain exploratory
development results. First assess whether ranking improves without degrading
untouched/non-contraction controls; only then consider broader variants or a
first-pass multiword-history implementation.
