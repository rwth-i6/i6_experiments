# Modular Sisyphus entrypoints

All entrypoints provide a `py()` function, allowing filename-style invocation:

```bash
./sis m config/rnnt_qwen.py
```

This assumes `config` points to this baselines directory, as in the current
setup. The `.main` selectors below remain supported for compatibility.

Keep the existing combined entrypoint:

`i6_experiments.users.berger.configs.baselines.main`

To load a smaller graph, replace that config selector with one of:

| Config selector (same module prefix) | Experiments |
| --- | --- |
| `baselines.librispeech.main` | LibriSpeech group |
| `baselines.loquacious_small.main` | Loquacious-small group |
| `baselines.loquacious_medium.main` | Loquacious-medium group |
| `baselines.open_asr.main` | Currently enabled Open-ASR conditions |
| `baselines.rnnt_qwen.main` | Only Parakeet/Qwen dev.short tuning |
| `baselines.qwen_tuning.main` | Granite/Parakeet + text Qwen 0.6B/1.7B and Speech-Qwen, dev-other/dev.short tuning |
| `baselines.earnings_qwen.main` | Parakeet + Speech-Qwen pruning diagnostic on Earnings22 |
| `baselines.pretrained_corrections.main` | Standalone Speech-Qwen beams and pretrained prior/ILM tuning on dev-other/dev.short |
| `baselines.speech_qwen.main` | New dev-only Speech-Qwen 1.7B balance/prior tuning, 0.6B/1.7B n-best rescoring and pruning thresholds |
| `baselines.controlled_comparison.main` | CPU-only frozen dev selection, provenance audit, missing-cell inventory and paired confidence intervals |
| `baselines.controlled_evaluation.main` | Exact missing evaluations from a completed frozen fast protocol; no test-set tuning |
| `baselines.error_complementarity.main` | CPU-only paired-error overlap, selection oracles and fusion error attribution, including fixed Cohere native |
| `baselines.research_diagnostics.main` | CPU-only candidate coverage/ranking and duration/source/training-frequency analyses (Steps 3/4) |
| `baselines.candidate_ranking.main` | Step 3 only, using cached development n-best Speech-Qwen scores |
| `baselines.stratified_errors.main` | Step 4 only, using existing hypotheses, audio lengths and training transcripts |
| `baselines.wordwise_lm.main` | Parked own-model delayed-fusion pilots; opt-in only, no longer called by main |
| `baselines.results.main` | Reports only; reads existing outputs without loading recognition graphs |

For example, the tuning selector is
`i6_experiments.users.berger.configs.baselines.rnnt_qwen.main`.
Use it in place of the existing selector in your Sisyphus manager invocation;
keep the same working directory, settings and environment.

Importing the parent package no longer imports all corpus drivers. Selecting
one config registers only that group's jobs (and its necessary dependencies).
Aliases and output prefixes match the combined entrypoint, so existing jobs
can be reused. Use one selected entrypoint at a time to avoid duplicate output
registration; don't load both the combined config and its children together.

The combined entrypoint still runs all **enabled** groups. `./sis m config/open_asr.py`
now loads the eight-corpus held-out Qwen evaluation matrix plus Loquacious dev/test: standalone Granite,
Parakeet and Speech-Qwen, plus text-Qwen and Speech-Qwen fusion for each acoustic
model, reusing existing jobs where possible. Granite two-pass Qwen rescoring
with LibriSpeech and available corpus lexica is also registered. No specialized
word-LM grid or new held-out tuning is enabled.

All full-suite fusion conditions retain LM scale 0.3. Granite uses final/pre-LLM
beams 32/128; Parakeet + text-Qwen uses 8/32.
Final/pre-LLM thresholds are 6/6 for Granite, 2/2 for Parakeet + text-Qwen,
and 3/3 with beams 8/32 for Parakeet + Speech-Qwen (`speech-qwen-fast`).
The latter reached 3.93% WER / RTF 1.23 on dev.short; old expensive held-out
Speech-Qwen conditions are not scheduled. Granite preserves acoustic-model casing;
Parakeet scores lowercase and capitalized variants with both Qwen scorers.
The combined report is registered as `open-asr/qwen-combination/overview`.
No submitted jobs are cancelled or outputs deleted by switching entrypoints.

The full suite now includes text Qwen Base 1.7B alongside 0.6B for both ASR
models, and Granite two-pass 1.7B rescoring reuses the same candidate lists.
Speech-Qwen also compares 0.6B with the pinned
`Qwen/Qwen3-ASR-1.7B-hf` checkpoint
(`bcd2b5b7f32b480ab5790554cfa8347f246a14f3`): standalone greedy and beam 4/8,
Granite fusion, and Parakeet fusion. The full-suite 1.7B fusion runs transfer
scale 0.3 and the same efficient pruning/casing/batching policies as 0.6B;
these are transfer settings, not claimed 1.7B optima. `qwen_tuning.py` adds
1.7B scales 0.05/0.1/0.3/0.5 on dev-other/dev.short for both ASR models,
plus standalone beams and the development-only length-penalty ablation.
New 1.7B jobs request 48 GB GPU memory without changing inference precision.
Aliases/reports distinguish Speech-Qwen 1.7B from text-Qwen 1.7B, and fusion
deltas use standalone Speech-Qwen controls of the matching size.
Matched-budget ASR-only controls use zero external-LM scores with the same
search budgets. Standalone Speech-Qwen additionally runs native beams 4/8
(length penalty 1, unchanged 500-token limit). These are stronger standalone
controls, not fusion settings. Reports compare Speech-Qwen fusion with the
better of the displayed standalone ASR and standalone Speech-Qwen controls.

`./sis m config/pretrained_corrections.py` registers 56 corrected development
runs plus standalone Speech-Qwen beam controls on dev-other/dev.short. For
Granite, prior weights 0.1/0.2 subtract a frame-weighted log prior estimated
from 512 training utterances per corpus (LibriSpeech train.100 and Loquacious
small/train). The pinned training datasets are streamed; no dev/test audio or
text is used to estimate priors. For Parakeet, ILM weights 0.1/0.2 subtract
the zero-encoder predictor/joint distribution conditional on nonblank labels;
blank scores remain unchanged. This is an ILM approximation, not a separately
trained LM. Each weight is crossed with external-LM scales 0.1/0.3/0.5 for
text-Qwen and Speech-Qwen, alongside corrected ASR-only controls. Development
Speech-Qwen beams also compare length penalties 0 and 1. The main entrypoint
includes this group; held-out correction runs wait for development selection.

Correction jobs export validation summaries for vocabulary ordering, score
shapes, normalization and unchanged RNNT blank scores. Unit tests cover score
accounting and native multimodal beam generation; full-checkpoint validation
and correction WER measurements happen when the new jobs run.

## Speech-Qwen follow-up batch

`./sis m config/speech_qwen.py` loads the new batch alone (also included by main).
Only LibriSpeech dev-other and Loquacious dev.short are accepted. No new full-suite
runs are enabled by this batch.

- Parakeet + Speech-Qwen 1.7B: scales 0.05/0.1/0.15/0.2/0.25/0.3 crossed with
  word penalties 0/-0.5/-1, at beam/pre-beam 8/32 and threshold 3.
- Granite + Speech-Qwen 1.7B: prior weights 0/0.05/0.1/0.2 crossed with scales
  0.2/0.3/0.4. Reuses the existing 512-training-utterance priors, including their
  pinned source revisions; no evaluation audio is used for prior estimation.
- Speech-Qwen 0.6B and 1.7B n-best rescoring: reuses Granite's LS open-tree
  50-best lists and generates one Parakeet AM-only 50-best list per dev corpus
  (Viterbi, beam/pre-beam 64/128, threshold 6). All model sizes and scoring
  settings share their first-pass candidates. CPU topology and per-segment
  score/text parity checks guard the native RNN-T n-best interface.
- Cached whole-sentence audio-conditioned scoring uses batch 8, FP32, one EOS,
  and original/initial-capital/initial-capital-period surfaces. Selection compares
  scales 0.1/0.3/0.5, word penalties 0/-1, original vs initial-capital-period, plus
  a 10-best comparison at scale 0.3. Includes first-pass controls and explicit
  reference-dependent 10/50-best oracles. References never enter LM scoring or
  deployable selection. No token reward or length averaging is applied.
- Threshold stage: Speech-Qwen 1.7B + Parakeet, scales 0.1/0.3, beam/pre-beam
  32/128, thresholds 1.5/2/2.5/3. These generous-beam diagnostics may exceed RTF
  1.5. After selecting a threshold on dev, call `run_pruning(stage="beam",
  threshold=..., scale=..., word_penalty=...)` explicitly to compare beam/pre-beam
  4/16, 8/32, 16/64 and 32/128. The beam stage is not scheduled prematurely.

All new GPU jobs request the 48 GB queue and remain subject to the seven-day
cap. Final configurations must be selected for RTF <1.5, not assumed to satisfy
it from memory allocation alone. Existing inference runtime files are unchanged
so completed recognition jobs are reused.

Long unsharded runs use measured audio durations: GigaSpeech 35.143 hours,
SPGISpeech 100.002 hours. Requests target RTF <1.5 with 10% margin plus two hours:
60 and 168 wall-clock hours, respectively. SPGISpeech GPU runs use the 48 GB
queue. Resource-only changes preserve job identities. `settings.py` enforces
the 168-hour cap, including tasks carrying the retired `runtime_cap_hours`
override. Additional GPU memory does not guarantee RTF <1.5; slower conditions
need pruning tuning rather than longer requests. Already-submitted Slurm
requests must be updated separately; reloading the config does not modify them.
Timed-out/submitted jobs are not reset or resubmitted automatically.

`./sis m config/earnings_qwen.py` adds ten pruning settings and reuses the fast
and wider controls, all with Speech-Qwen scale 0.3 and unchanged casing/rewards.
Final/pre-beams and thresholds: 32/128 at 3/3.5; 24/128 at 3.5/4;
16/128 at 3/3.5/4; 8/128 at 4; 16/64 at 3.5/4. Controls are 8/32 at 3
and 32/128 at 4. This explicitly tunes on Earnings22 (target WER 8.3–8.4),
so results are marked as a corpus-specific diagnostic, not untouched held-out
evaluation. The main entrypoint includes this group; the dev-only entrypoint
does not. There is no sharding.

`./sis m config/qwen_tuning.py` runs focused Granite/Parakeet development grids
on LibriSpeech dev-other and Loquacious dev.short, including Qwen Base 0.6B/1.7B,
Speech-Qwen and cached n-best scale tuning. The combined entrypoint includes it.
`rnnt_qwen.py` selects only the focused Parakeet 0.6B dev.short portion.
Old delayed-fusion word-LM experiments are parked (explicit `wordwise_lm.py`
remains available for historical use, but main no longer calls it).

## Accumulated result collection

### Controlled comparison (research-plan Step 1)

1. Run `./sis m config/controlled_comparison.py` (also registered by main).
   This snapshots existing exports/aliases without reconstructing old experiment
   grids. Development and evaluation evidence are separate job inputs; selection
   cannot read test scores. Outputs live under `output/controlled-comparison/`.
2. Inspect `selection.json` and `comparison.md`. Each model/auxiliary/method
   pairing gets a single jointly selected condition: minimize equal-weight mean
   WER/native-WER across dev-other and dev.short. Ties prefer lower runtime,
   then stable condition id. The primary choice must have RTF <1.5 on both dev
   corpora; unrestricted choices are reported separately. Native/beam and
   first-pass no-LM controls compete as standalone controls; lexicon usage stays
   explicit. Missing dev timing/provenance or a different normalizer disqualifies
   a candidate. Test-specific Earnings tuning and staged pruning diagnostics are
   not transferable candidates. Pending paired-dev candidates make the selection
   provisional; rerun the existing tuning entrypoint rather than silently ignoring
   them. Text-Qwen RNN-T two-pass is not implemented and is not implied by the
   available Speech-Qwen two-pass results.
3. Once `ready` is true, run `./sis m config/controlled_evaluation.py`. It resolves
   the selection output alias to its immutable job artifact when constructing the
   graph. Optionally set `ASR_CONTROLLED_PROTOCOL` to an explicit frozen artifact.
   It reuses exact completed configuration matches and schedules only missing
   fast evaluations. It never substitutes a nearby/default/test-best setting.
   This second entrypoint is deliberately NOT automatically called by main.
   Original reviewed runtime classes are reused; no existing runtime sources are
   edited. Existing model/config artifacts must remain available.
4. Read `output/reports/asr-lm/controlled-selected-evaluation.md` or the versioned
   outputs under `output/controlled-evaluation/<protocol-digest>/`. The evaluation
   covers the existing full suite and Loquacious test, deriving source-subset
   scores without decoding twice. Speech-Qwen sizes share matching candidate
   generation; two-pass retains the recorded first-pass costs. For prior-based
   configurations, the frozen Loquacious recipe applies to Loquacious; all other
   corpora use the LibriSpeech calibration recipe. All new GPU jobs request 48 GB,
   remain unsharded and respect 168 hours. Dev budget eligibility is not an RTF
   guarantee on another corpus: evaluation budget failures are marked explicitly.

The reports retain I/D/S, per-system reference counts, runtime, job/config hashes,
model download revisions, dataset origins, host peak RSS and requested GPU memory.
Historical GPU peak usage is unknown, not inferred from host RSS. Subset runtime
is marked as parent runtime. Raw-reference, audio-origin, normalization and paired
coverage mismatches fail loudly. Compound-equivalence scoring can legitimately
give slightly different reference word counts, so paired bootstrap retains each
system's denominator. Ten thousand seeded paired bootstrap replicates group
LibriSpeech chapters, AMI meetings, GigaSpeech/SPGISpeech source recordings and
VoxPopuli sessions where identifiers support those groupings. Identifier-derived
groupings are explicit and should be reviewed before confirmatory work; unknown
origins remain scoring units. Earnings22 already has recording-level scoring
units. Few-cluster warnings and unadjusted 95% interval labels are explicit.
The better aggregate standalone is recomputed per replicate, not selected per
utterance; missing Speech-Qwen controls suppress that comparison. No cross-corpus
or parent/subset pooling is performed. These already inspected tests remain
exploratory: untouched data is still needed for confirmatory claims.

Audio-origin checks trace the audited selection-only `PrepareBenchmarkSubsetJob`
to its parent. This permits pre-decoding dev.short selection to be compared with
post-decoding filtering of the same parent audio, including mixed runtime/hypothesis
provenance. Raw-reference and scoring-unit coverage checks still apply. Arbitrary
audio-transform jobs are not collapsed, and genuinely different origins still fail.

### Error complementarity (research-plan Step 2)

`./sis m config/error_complementarity.py` runs analysis only, with no model
downloads or recognition jobs. It reuses Step 1's dev/evaluation snapshots and
selection jobs and includes Cohere's available fixed native outputs. Main also
registers this analysis. It can run before the missing selected fusion
evaluations finish. As usual, do not run two managers writing the same report
outputs concurrently; select the desired entrypoint in the existing setup.

Outputs are under `output/error-complementarity/` and the report is registered
as `reports/asr-lm/error-complementarity.md`:

- `index.md` / `pairs.tsv`: per-dataset/per-pair WER, scoring-unit selection
  oracle, headroom and asymmetric correct/error counts.
- `<dataset>/report.md` / `summary.json`: both-correct, only-A/B-correct and
  both-wrong units (same vs different normalized hypotheses); reference-word
  overlap; insertion overlap; fusion-corrected and newly introduced errors
  relative to each standalone; jointly correct outputs broken by fusion.
- `<dataset>/units.jsonl.gz`: compact per-unit counts and oracle choices.
  `<dataset>/examples.jsonl`: bounded examples, clipped to 1200 characters per
  transcript field, with IDs and source-score provenance for inspection.

There are three distinct panels, never selected by test WER:

1. **native:** each model's fixed native decoding.
2. **beam-controls:** fixed Parakeet beam 8/threshold 3 and Speech-Qwen beam 4;
   Granite and Cohere retain native decoding. Fixed-transfer fusion settings
   are compared against both this panel and the native controls. These are
   exploratory controls, not claimed optimal configurations.
3. **dev-selected:** exact fast selections from Step 1. Missing configurations
   remain missing; no default or best-on-test result is substituted. A
   provisional development selection suppresses this panel. Cohere stays
   fixed native because no dev-selected Cohere grid exists.

All available standalone pairs are analyzed, including Granite/Parakeet,
Speech-Qwen 0.6B/1.7B and Cohere. Text-only LMs have no standalone ASR output
and therefore are not model pairs in this analysis. Speech-fusion results
are attached only to their actual constituent model pair.

The oracle selects the lower edit count per official scoring unit, ties A,
and reports the selected hypothesis's reference denominator. It uses
references, is not deployable, and is not a lower bound on fusion: fusion can
combine corrections within a unit. For grouped corpora, the units are
recordings rather than utterances. Word attribution is performed only when
the normalized references agree; compound-equivalence mismatches are counted
explicitly instead of comparing incompatible word grids. Insertion events
are compared separately by reference boundary and token, with multiplicity.
Minimum-edit alignment ties can affect word attribution. These are descriptive
diagnostics, not significance claims; Step 1 supplies paired uncertainty.

The selected-evaluation entrypoint additionally registers an analysis against
its declared, completed results under
`controlled-evaluation/<digest>/error-complementarity/`, with report
`reports/asr-lm/error-complementarity-selected.md`. This adds CPU jobs only.
For a later standalone rerun, `ASR_CONTROLLED_PROTOCOL` can pin the selection
artifact and `ASR_COMPLEMENTARITY_EVIDENCE` can add its frozen evaluation
snapshot explicitly. Neither mechanism polls for test winners.

## Candidate ranking and difficult-condition analysis

With the existing configs in the same manager, add
`config/research_diagnostics.py` to its configuration list. This
registers Steps 3 and 4 together; the two smaller entrypoints above are
alternatives. The combined main also includes them. These jobs request CPU
resources only and reuse completed evidence; they launch no recognition,
model inference or audio downloads. The development selection must finish
before candidate diagnostics run.

Step 3 exports `output/candidate-ranking/<dev-set>/`:

- `summary.json` / `report.md`: first-pass, rescored, standalone Speech-Qwen
  and n-best oracle WER; candidate coverage gaps versus ranking gaps.
- `units.jsonl.gz`: individual choices and retained AM, first-pass LM,
  Speech-Qwen, EOS and word-reward costs.
- `examples.jsonl`: bounded examples for inspecting each failure category.

It checks the exact fast dev-selected two-pass settings for both AMs and both
Speech-Qwen sizes on dev-other and dev.short. Recomputed selections must match
the saved normalized hypotheses. Comparing the auxiliary-only choice with the
combined choice separates first-pass score effects from word-reward effects.
Adding the standalone output is an **oracle diagnostic only**: no missing AM
score is invented, and no deployable union-pool ranking is claimed. These
whole-sentence cached scores cannot attribute within-word pruning failures.

Step 4 exports `output/stratified-errors/<dataset>/summary.json` / `report.md`:

- Official WER and I/D/S by fixed duration, source and input-chunk slices.
- Reference-word accuracy by LibriSpeech train-other-960 transcript frequency
  (unseen, 1–5, 6–20, 21–100, >100), digit presence, and common words adjacent
  to rare words; paired corrections and newly introduced errors versus each
  constituent standalone control.

Frequency counts use the current normalizer on the existing training XML,
independently of recognition/evaluation text. They measure **LibriSpeech-training
frequency**, not general English rarity or pretrained-model exposure. The
source is recovered from the existing training-frequency job; optionally pin
an existing XML with `ASR_RARITY_TRAINING_CORPUS`. No new corpus download is
scheduled. Proper names require independent annotations and are deferred.

Duration bins are disjoint: [0,5), [5,10), [10,20), [20,30), [30,60),
[60,300), [300,infinity) seconds. For grouped scoring units, duration is
summed input-chunk audio, **not** evidence of unsegmented model processing.
Parent corpora and source subsets remain separate. All three Step-2 control
panels are retained; missing exact selections remain missing. Word statistics
exclude units with incompatible compound-normalized reference grids for every
system within the panel; official slice WER includes them. Insertions count in
WER, not reference-word accuracy. These slices are exploratory descriptions,
not significance tests or a new tuning criterion.

Reports are registered as `reports/asr-lm/candidate-ranking-<dataset>.md`
and `reports/asr-lm/stratified-<dataset>.md`. To include a completed frozen
selected-evaluation snapshot in Step 4, set
`ASR_COMPLEMENTARITY_EVIDENCE`; `ASR_CONTROLLED_PROTOCOL` pins its selection,
as for Step 2. No best-on-test result is substituted.

For company-side preparation and the decision gates for AppTek, see
[APPTEK_HANDOFF.md](../../asr_research/setup/experiments/open_asr/APPTEK_HANDOFF.md).

The full, Open-ASR, RNN-T/Qwen and wordwise-LM entrypoints register ten live
reports with `tk.register_report`, under `output/reports/asr-lm/`. Start with
`index.md`: the main pages are `librispeech.md`, `loquacious.md` and `open_asr.md`.
Each dataset (including Loquacious source sub-splits) has standalone/no-LM
controls and model/LM combinations together, with explicit ASR model names,
text-vs-speech Qwen and sizes, decoding method, lexicon, WER and RTF.
`models-and-methods.md` lists exact pretrained IDs and explains historical
condition names. The older combined/word-LM/own-model pages remain available,
along with the incomplete/excluded inventory and `all-results.tsv`.
Reports refresh every five minutes while the manager is running. They read
individual exported scoring/runtime summaries, not potentially stale overview
jobs. Missing exports do not block completed results. Existing inactive results
remain visible, but no archived recognition jobs are reconstructed or scheduled.

For a report-only graph: `./sis m config/results.py`. Run it from the usual
Sisyphus setup directory so `gs.OUTPUT_DIR` selects the existing output tree.
Do not run two managers writing the same report collection concurrently.
The historical corpus reports are linked from `own-models.md`. WER differences
are only calculated for matching normalizer revisions; test sets are never used
to pick the best condition, and overlapping subsets are not pooled.
