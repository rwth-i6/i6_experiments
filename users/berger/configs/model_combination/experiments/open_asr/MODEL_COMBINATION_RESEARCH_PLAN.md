# Research plan: when does ASR model combination help?

Recorded: 2026-11-02.

## Research question and intended contribution

Under which conditions does inference-time combination improve on the strongest
standalone recognizer, and when is its additional computational cost unjustified?

The intended contribution is a controlled empirical study, not a claim that a new
fusion algorithm universally improves ASR. Positive and negative results should
establish the roles of model complementarity, search direction, evaluation
conditions and computational budget.

Current experiments are the exploratory foundation. In particular, the apparent
failure of AM-led fusion when Speech-Qwen is the stronger recognizer suggests a
search/objective asymmetry, but does not establish its cause. Distinguish missing
candidates from poor ranking and pruning before drawing that conclusion.

The steps below are ordered by expected insight per amount of work, with
dependencies and decision gates determining the more expensive follow-ups.

## 1. Consolidate the existing controlled comparison

**Effort:** low. **Expected insight:** high.

- Finish pending rescoring and prior runs.
- Select configurations using LibriSpeech dev-other and Loquacious dev.short,
  not held-out test performance.
- Retain strong standalone controls, including native beam search; AM-led
  one-pass fusion; and two-pass rescoring for each applicable model pair.
- Record WER, insertions/deletions/substitutions, runtime, memory and exact
  configuration provenance.
- Measure improvement over the better standalone component and report the
  accuracy–runtime trade-off.
- Separate slow historical configurations from operating points meeting
  RTF <1.5. Use the 48 GB GPU queue where appropriate; extra memory alone does
  not establish an acceptable RTF. Keep the 168-hour job limit and no sharding.
- Use paired confidence intervals at the appropriate utterance or recording
  level. Keep scoring, normalization and dataset versions explicit.

**Deliverable:** a reproducible comparison table. Existing test results remain
exploratory; later confirmation must include untouched data.

Implementation started 2026-11-03: `config/controlled_comparison.py` freezes
separate development/evaluation evidence and registers CPU-only selection/audit
reports. After its selection is complete, `config/controlled_evaluation.py`
reuses exact existing results and schedules missing fast-protocol evaluations.
Both export provenance and paired cluster-bootstrap intervals. This step is not
complete until the selected evaluations and their reports have finished. See
the baselines README for the selection rule, clustering and two-stage workflow.

## 2. Measure error complementarity

**Effort:** low; primarily analysis of existing hypotheses.
**Expected insight:** very high.

For every corpus and model pair, distinguish:

- Both models correct.
- Only the AM correct.
- Only Speech-Qwen correct.
- Both wrong, with identical or different errors.
- Errors corrected by fusion and errors newly introduced by fusion.

Calculate a reference-dependent utterance-selection oracle: select the better
standalone hypothesis for each scoring unit. Label it as a diagnostic, never a
deployable system. Include available Cohere outputs before implementing another
search.

**Question:** is there meaningful complementary information to exploit?

**Decision gate:** prioritize model pairs and corpora with substantial oracle
headroom rather than merely different aggregate WERs.

Implementation started 2026-11-03: `config/error_complementarity.py` registers
CPU-only native, fixed beam-control and exact dev-selected panels, including
available Cohere native outputs. It measures scoring-unit and reference-word
overlap, selection-oracle headroom, and fusion-corrected/new errors. It does
not select test winners or launch recognition. The selected-evaluation graph
also registers analysis on its completed outputs. Interpret the diagnostic
oracle as reference-dependent selection, not a lower bound on fusion. See
the baselines README for artifacts and normalization/alignment caveats.

## 3. Diagnose why current fusion misses the available headroom

**Effort:** low–medium. **Expected insight:** very high.

Use completed n-best lists to analyze cases where Speech-Qwen is correct and
fusion is wrong:

1. Is the correct transcription, or a substantially better alternative, present
   in the AM-generated candidate list?
2. If present, why does the combined objective rank it below the selected output?
3. If absent, does adding the standalone Speech-Qwen hypothesis substantially
   improve the candidate-pool oracle?

Separate candidate-generation limitations, score-calibration/ranking limitations,
text-formatting effects and pruning effects. Keep native n-best/one-best score
parity checks enabled.

Adding a Speech-Qwen hypothesis to a pool is initially an oracle diagnostic. A
deployable union-pool system must score every candidate consistently with both
models; do not compare candidates with missing or incomparable score components.

**Decision gate:** determine whether the next priority is changing search
direction or correcting the scoring objective/calibration.

Implementation started 2026-11-04: `config/candidate_ranking.py` (or the combined
`config/research_diagnostics.py`) checks exact fast dev-selected two-pass
settings against cached Speech-Qwen scores. It separates coverage gaps,
ranking gaps, first-pass score effects and word-reward effects. The augmented
oracle has no invented AM score. These CPU-only diagnostics require completed
development selection; they do not launch inference or diagnose online pruning.

## 4. Analyze difficult conditions already present in the data

**Effort:** low–medium. **Expected insight:** high.

Before preparing new datasets, stratify existing results by:

- Utterance duration.
- Independently estimated word frequency.
- Proper names, numbers and other transcription categories where reliably
  identifiable.
- Source/domain, especially the Loquacious sub-splits.

Measure rare-word accuracy together with surrounding ordinary-word errors.
Define frequency bins independently of recognition outputs. Do not pool
overlapping parent datasets and subsets.

**Question:** do modest aggregate gains hide substantial benefits or regressions
under particular conditions?

Treat these findings as exploratory associations, not causal evidence.

Implementation started 2026-11-04: `config/stratified_errors.py` (also included
in `config/research_diagnostics.py`) reuses existing outputs and recorded input
durations. It measures source/duration slices, LibriSpeech-training-frequency
and digit-word accuracy, neighboring common-word errors and paired fusion
corrections/regressions. Proper names are deferred without independent labels.
Grouped recording duration is distinguished from unsegmented model input.
See the baselines README for artifacts and common-reference-grid exclusions.

## 5. Run a controlled noise pilot

**Effort:** medium. **Expected insight:** high.

- Select a fixed development subset.
- Create several reproducible noise levels while preserving content, references
  and segmentation.
- Evaluate both standalone models, the selected fusion configuration and
  rescoring where available.
- Initially hold fusion parameters fixed across noise levels to test robustness,
  rather than condition-specific retuning.

**Question:** does combination become useful as acoustic evidence deteriorates,
and does the answer depend on which model deteriorates faster?

**Decision gate:** expand the noise evaluation only if the pilot reveals an
informative pattern.

## 6. Compare AM-led and Speech-LLM-led search

**Effort:** high. **Expected insight:** very high, particularly if Step 3
implicates candidate generation.

Implement Speech-LLM-led label-synchronous fusion for one existing model pair
first. Compare:

| Condition | Purpose |
| --- | --- |
| Strong standalone AM and Speech-Qwen | Required controls |
| AM-led fusion | Existing asymmetric search |
| Speech-LLM-led fusion | Reverse the asymmetry |
| Two-pass/union-pool rescoring | Separate proposal generation from ranking |

Keep model checkpoints, preprocessing and evaluation fixed. Tune each search on
development data and report its accuracy–runtime frontier. Do not introduce a
new ASR model simultaneously, since that would confound the comparison.

**Question:** does search direction determine whether combination can exploit the
stronger component?

## 7. Pilot natural long-form audio

**Effort:** medium–high. **Expected insight:** high.

Start with a small collection of long recordings, preferably including
representative AppTek material if available and suitable for research use.
Separate two factors:

1. **Segmentation:** fixed common segments versus the long-form processing
   pipeline.
2. **Context:** reset linguistic history versus carry context across segments,
   where supported.

Track WER, deletions, repetitions/hallucinations, runtime and memory as recording
duration grows. Apply explicit output-length limits, retain partial outputs when
limits are reached, and report truncations.

**Question:** does combination benefit from longer context or stabilize
long-form decoding, and are apparent gains actually segmentation effects?

**Decision gate:** establish the controls before scaling to a substantial
long-form evaluation.

AppTek preparation can start now on company servers, independently of the
public GPU queue. Begin native company-dev baselines after a data/scorer smoke
test; choose the expensive search implementation after Step 3, then freeze
company-dev settings before untouched evaluation. See
[APPTEK_HANDOFF.md](APPTEK_HANDOFF.md) for the separate-setup handoff.

## 8. Validate with a stronger additional recognizer

**Effort:** medium for native baselines; high for fusion integration.
**Expected insight:** medium–high.

Establish Parakeet TDT's native baseline and error complementarity first. Analyze
existing Cohere results similarly. Select one for fusion integration based on:

- Complementarity with Speech-Qwen.
- Whether it tests a question not answered by Granite CTC or Parakeet RNN-T.
- Implementation cost.

Do not infer usefulness from an overall leaderboard ranking alone.

**Question:** do the conclusions generalize beyond the original recognizers?

## 9. Consider three-model fusion only with evidence of added information

**Effort:** high. **Expected insight:** uncertain.

Before implementing CTC + RNN-T + Speech-LLM, show that the third model corrects
errors remaining in the best pair. Compare any resulting system against:

- Every constituent standalone model.
- The best two-model combination.
- A simpler selection/rescoring baseline.
- The additional computational cost.

**Decision gate:** skip this extension if the third model contributes little
complementary information.

## 10. Run confirmatory evaluation and decide the paper scope

**Effort:** medium.

- Freeze the hypotheses, configurations and primary metrics before evaluating
  untouched conditions.
- Report positive and negative findings, paired uncertainty and runtime.
- Keep exploratory test-selected results separate from confirmatory evaluation.
- State conclusions only for the tested model families and conditions unless
  additional evidence supports broader generalization.

The final study should answer whether model complementarity, evaluation
conditions and search direction explain when combination is worthwhile—not
merely whether some tuned combination can lower WER.

## Immediate execution order

1. Complete Steps 1–4 using existing outputs and pending experiments.
2. Run a small controlled noise pilot.
3. Let Step 3 determine the urgency of Speech-LLM-led search.
4. Develop long-form evaluation as the next substantial evaluation direction.
5. Add another recognizer only after inspecting its complementarity.
6. Leave three-model fusion until last.

Do not launch every branch at once. Use the decision gates to avoid expanding
the experiment grid without sharpening the research question.
