# Public study: when does ASR model combination help?

Consolidated 2026-11-16. The frozen development-selected evaluation is complete:
**234 results, 18 model/method families, 13 datasets or subsets, no missing
standalone baselines**. The results support an empirical study of the conditions
under which combination helps, rather than a general recommendation to fuse
models or a claim of a new search algorithm.

## Tables and evidence

- [WER tables](results/public-study/tables.md): all standalone controls and
  text/speech fusion conditions, with Qwen size and one-/two-pass method explicit.
- [Gains, confidence intervals and RTF](results/public-study/metrics.md): each
  combination against its better standalone constituent of the matching size.
- [Development diagnostics](results/public-study/diagnostics.md): standalone
  complementarity, candidate availability/ranking and word-frequency slices.
- [Corrected error attribution](results/public-study/attribution.md): errors
  fixed and introduced by Parakeet + Speech-Qwen 1.7B, with exclusions explicit.
- [Full-precision results](results/public-study/results.tsv): edit counts,
  denominators, baselines, timings and source paths.
- [Provenance](results/public-study/provenance.json): input/output checksums,
  frozen protocol and renderer identity.
- [Frozen transfer conditions](frozen_public_conditions.json): selected public
  stage parameters and pinned model revisions for the AppTek handoff families.

These tables use fixed, jointly development-selected configurations, not the
best configuration on each test set. Earlier exploratory tuning tables are not
substituted for the frozen results. The earlier small word-LM/closed-tree studies
are background experiments, not matched conditions in this final comparison.

## Main findings

**Combination gives reproducible but conditional improvements over strong
standalone constituents.** Parakeet + Speech-Qwen 1.7B one-pass improves from
2.099% to 1.962% on LibriSpeech test-other, from 4.353% to 4.163% on Loquacious
test, and from 3.425% to 3.279% on Monsoon. These are about 6.5%, 4.4% and 4.3%
relative reductions, respectively. The corresponding paired intervals for the
absolute change are [-0.199, -0.073], [-0.261, -0.121] and [-0.294, -0.001]
percentage points. Monsoon two-pass reaches 3.254% (5.0% relative), so improvement
is possible even when Speech-Qwen is the stronger standalone constituent.

**Improving on the AM alone is insufficient evidence of useful combination.**
On Earnings22, standalone Speech-Qwen 1.7B obtains 5.652%; Parakeet fusion obtains
7.483% one-pass and 7.952% two-pass. On GigaSpeech the corresponding numbers are
7.289%, 7.526% and 7.449%. VoxPopuli also favors standalone Speech-Qwen: 2.852%
versus 3.105% and 3.004%. The VoxPopuli two-pass interval includes zero; the
Earnings22 and GigaSpeech intervals indicate degradation, subject to the
exploratory protocol and particularly few independent Earnings22 recordings.

**The primary recognizer matters.** On AMI, Granite + Speech-Qwen 1.7B one-pass
improves Granite from 7.363% to 7.062%. Parakeet fusion improves its AM from
13.295% to 10.699%, but loses to Speech-Qwen standalone at 8.367%. There is no
aggregate AMI degradation from the selected Parakeet beam relative to native
Parakeet: the audited native WER was 14.339%. The earlier apparent search
degradation came from misidentifying Speech-Qwen's result as Parakeet's.
Available Cohere native output was 7.035%, so Granite fusion is also not a claim
to beat every available recognizer on AMI.

**Speech-Qwen is generally more useful than text-Qwen at the selected operating
points.** For Parakeet on LibriSpeech other, text-Qwen 0.6B/1.7B gives
2.142%/2.129%, compared with 1.980%/1.962% for speech-Qwen one-pass and 2.099%
for the AM control. For Granite, text-Qwen 1.7B one-pass improves 2.596% to
2.524%, while speech-Qwen 1.7B reaches 2.360%. This supports the value of
audio-conditioned auxiliary scoring in this setup; it does not isolate audio
conditioning from differences in training or calibration.

**A larger auxiliary model need not produce a better fused system.** On
SPGISpeech, Parakeet + Speech-Qwen 0.6B two-pass obtains 2.431%, compared with
2.537% for 1.7B, although 1.7B is the stronger standalone Speech-Qwen
(2.569% versus 2.705%). The sizes have separately selected fusion parameters;
this is an operating-point comparison, not a size-only ablation. Granite alone
is stronger still at 1.841%, and Granite + Speech-Qwen 1.7B reaches 1.691%.
Gains over a pair's constituents must therefore be distinguished from gains
over the strongest recognizer across all models.

**Two-pass rescoring remains a competitive reference.** For Parakeet +
Speech-Qwen 1.7B it reaches 1.987% versus 1.962% one-pass on LibriSpeech other,
4.191% versus 4.163% on Loquacious, and 3.254% versus 3.279% on Monsoon.
Recorded RTFs are 0.504 versus 1.293, 0.551 versus 1.142, and 0.695 versus
1.341, respectively. These are observed operating points on historical
hardware, not controlled speedup measurements. Standalone Speech-Qwen is much
cheaper on these runs, so a small fusion gain can carry a substantial cost.
The development RTF constraint does not guarantee test-time RTF <1.5:
Parakeet + Speech-Qwen 1.7B one-pass reaches 2.429 on Earnings22 and 1.590 on
VoxPopuli.

## What the diagnostics establish

There is complementary information to exploit. The reference-dependent
per-scoring-unit selector for Parakeet and Speech-Qwen 1.7B reaches 1.533% on
LibriSpeech dev-other versus standalone 2.036%/2.664%, and 3.236% on Loquacious
dev.short versus 3.951%/4.624%. This oracle is a diagnostic; it is neither a
deployable method nor a lower bound on every possible fusion.

Candidate generation and ranking both leave opportunities unused. In Parakeet's
1.7B rescoring diagnostics, 34/44 scoring units on the two development sets
have a pool oracle worse than Speech-Qwen; 93/142 have a pool that can match or
beat Speech-Qwen but a selected result that does not. A correct candidate is
present but not selected in 368/439 units. The nominal 50-best lists contain
only 8.7/11.6 distinct texts on average. These observations establish both
coverage and ranking limitations of the tested rescoring pools. They do not
identify within-word pruning as the cause, or prove that reversing search
direction would solve the test-corpus regressions.

Frequency slices suggest a useful rare-word direction, with limited scope.
For Granite one-pass + Speech-Qwen 1.7B, accuracy on reference words absent
from the LibriSpeech LM training text rises from 35.26% to 43.16% on dev-other
and from 67.59% to 72.74% on dev.short; frequent-word accuracy changes much less.
Parakeet improves more modestly, from 44.74% to 47.63% and from 72.91% to 74.18%.
This is not evidence about pretraining exposure or proper names. Units with
different normalized references are excluded from word attribution, and
insertions are reported separately from reference-word accuracy.

## Scope and limitations

- Frozen selection: equal-weight mean WER/native-WER over LibriSpeech dev-other
  and Loquacious dev.short, requiring development RTF <1.5 on both. Protocol
  SHA-256: `f4212424889ee5d390df7a87d917b77b65d35c1a15e7c4920c3599c3c1769a56`.
- Granite's AM-only control is the selected open-tree first-pass control,
  not native greedy decoding. Parakeet's is the selected AM-only RASR beam.
  Speech-Qwen controls use native beam 8 and length penalty 0. Different search
  families are practical systems comparisons, not isolated algorithm ablations.
- Updated HF English normalizer revision
  `48219c6028db0517d704600d92f31edfc96e8c23`, kaldialign 0.12.0 and the existing
  compound-equivalence scoring are shared. These scores need not equal older
  leaderboard scores with a different normalizer.
- Intervals use 10,000 seeded paired source-cluster bootstrap replicates with
  per-system reference denominators. The better standalone is recomputed in
  each replicate. Intervals are unadjusted; the sets were already inspected,
  so this is exploratory evidence. AMI has 16 source clusters; Earnings22 only
  six, making its uncertainty particularly fragile.
- Loquacious total and its source subsets overlap; its LibriSpeech subset also
  overlaps the separate LibriSpeech evaluation. No pooled grand mean, overall
  significance claim or independent win count is reported across those rows.
- Historical hardware and runtime accounting differ. Loquacious subset RTFs
  are inherited from the parent run. Host RSS and requested GPU memory are
  not measured GPU peak usage. No controlled memory-efficiency claim is made.
- These evaluations do not establish behavior on genuinely long unsegmented
  audio, controlled noise or an independent proper-name benchmark. Long-recording
  scoring does not imply recognition used unsegmented inputs.

## Research conclusion and next checkpoint

The supported contribution is a study of **when complementary recognizers can
be combined effectively, and when additional scoring cost fails to improve on
the stronger constituent**. Useful gains coexist with clear negative results;
neither auxiliary model size nor AM-relative gains alone predict success.
Two-pass rescoring is an essential comparator to the more expensive one-pass
search, and candidate-pool diagnostics separate two concrete failure sources.

The public consolidation is complete. The next checkpoint is comparison with
the already running AppTek suite under its transferred frozen conditions,
keeping any company-side tuning separate. The public Granite scorer-API parity
check found identical text, token paths and AM/LM scores on all 2,864 dev-other
segments; that validates this transferred configuration, not arbitrary new
environments. Additional public searches, reverse fusion, noise experiments or
long-form studies remain optional follow-ups, not prerequisites for describing
the current results.

## Reproduction

From the Sisyphus setup directory, `./sis m config/study_summary.py` registers
one small CPU consolidation job and the four report pages
`output/reports/asr-lm/public-study-{tables,metrics,diagnostics,attribution}.md`.
It reads completed immutable producer outputs and does not reconstruct any
recognition or download jobs. The complete machine-readable collection is
registered at `output/public-study`.

To regenerate this checked-in snapshot from the repository root:

```bash
python users/berger/model_pipelines/common/study_summary.py \
  --output-root ../../output \
  --destination users/berger/configs/model_combination/experiments/open_asr/results/public-study
```

The renderer refuses incomplete results, mismatched protocols/normalizers,
test-selected replacements, incorrect constituent baselines or inconsistent
edit-count WERs. The November 10/14 reviews remain historical records; this
consolidation incorporates the completed CPU attribution refresh after cutover.
