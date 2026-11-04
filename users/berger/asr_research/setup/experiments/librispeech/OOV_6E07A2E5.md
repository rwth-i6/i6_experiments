# Beta look-ahead: RASR 6e07a2e5

The active `oov_length.py` setup now pins
`6e07a2e503fb18cc4fcddc89fa01e2d9fbd4a5d9`, plus the approved XML separator
patch. The previous build getters are unchanged. New output aliases and
recognition descriptors use `6e07a2e5`; historical results are not overwritten.

## Regression being tested

3ed6d27b passed the delayed-word-cost probe but failed a first-piece test:
within-word acoustic pruning discarded DOG (cost 20) in favor of fallback-only
x (acoustic cost 0) before its beta=100 became visible at the piece exit.
The new commit anticipates this cost during within-word pruning.

Preflight now requires:

- The existing 34 native decoding/accounting cases.
- The existing four-condition delayed-cost probe: seven DOGs, total cost 35.
- **Eight first-piece checks**, four each for BPE and Granite: closed, open
  threshold 16 / beam 256, open threshold 16 / beam 4096, and open threshold
  200 / beam 4096. All must yield DOG at total cost 20.
- Real-lexicon monophone validation for all four boundary-flag combinations
  per acoustic label, including all 65,536 Granite checks.

The first-piece input is three frames: blank, competing emissions, blank.
Blank costs 0 on frames 0 and 2; on frame 1, fallback-only x costs 0 and DOG
costs 20. All remaining scores are 100. Beta is 100, alpha is zero, and the
LM contributes zero. Both tokenizations use the existing toy inventory and
boundary-independent monophone tying. The runner is hashed as a job argument.

`RequireOovPruningProbeJob` now depends on both probe reports and verifies
transcripts, costs and complete case coverage. All open-fallback lexica depend
on that combined acceptance output, so the new runs cannot silently reuse the
weaker 3ed6d27b acceptance result.

## Recognition grid

Retain the same limited dev-other grid to isolate the decoder change:

- BPE CTC + original 4-gram, scale 0.6: beta 10, 12, 20, 100 plus a same-file
  closed-tree control (5 runs).
- Granite Approach B, no LM and original 4-gram scale 0.3: beta 0, 2, 4, 8,
  16, 100 (12 runs).
- Rerun the nine Granite Approach A / closed / greedy controls on this binary,
  including beams 256 and 1024. Do not compare against old-build controls alone.

Initial stage: **26 recognition jobs**. Open-fallback runs keep beam 256, threshold 16,
word-end beam 256 / threshold 16, strict sentence-end handling, prior=0,
blank penalty=0, no word reward and Viterbi recombination. Alpha remains zero
for this regression round; the calibrated BPE alpha-sweep API remains available
for the next tuning stage. No subword LM, reduced lexicon, adapted LM or
held-out evaluation is added.

Keep the degeneracy guard enabled. Specifically revisit Granite no-LM beta=100
on `1585-131718-0024`: 3ed6d27b stopped after “THE FOURTH YEAR I GOT IT DOWN TO”;
the lower-beta conditions recognized “¢37” and the remainder. Passing toy
tests does not establish that this real-utterance failure is fixed.

Outputs under `output/librispeech/tools/rasr-oov-6e07a2e5/`:

- `decoder-smoke-report`, `decoder-smoke-logs`
- `delayed-penalty-probe`, `delayed-penalty-logs`
- `first-piece-report`, `first-piece-logs`
- `pruning-acceptance`

Dataset/lexicon validation outputs use `datasets/LibriSpeech/oov-6e07a2e5/`.
The AM margin measurement and historical scoring audit are reused: neither
depends on the decoder change.

## Completed regression results and next tuning stage

All native probes and full-corpus runs passed. BPE beta 20 and 100 reproduce
every closed-tree transcript; Granite beta 100 recovers closed-tree error
counts, with two raw punctuation-spacing differences. The previously truncated
utterance retains its suffix. Best dev-other WERs in the initial grid:

- BPE + 4-gram: beta 12 gives 5.280%, versus closed 5.286%. The gain is just
  three errors; substitutions fall by 33, insertions by 29, but deletions rise
  by 59. Unattested fallback runs reach 28 pieces.
- Granite no LM: beta 4 gives 2.648%, versus closed 2.850% and greedy 2.817%.
- Granite + 4-gram: beta 8 gives 2.604%, versus closed 2.722%.

`run_all` now appends `run_tuning`, adding **30 new dev-other recognitions**:

| Model / LM | Beta | Alpha | New runs |
|---|---|---|---:|
| BPE / 4-gram scale 0.6 | 6, 8, 10, 12, 14 | 0, 0.25, 0.5, 1, 2 | 23 |
| Granite / no LM | 1, 2, 3, 4, 5, 6 | 0 | 4 |
| Granite / 4-gram scale 0.3 | 4, 6, 8, 10, 12 | 0 | 3 |

BPE is a full beta-by-alpha Cartesian grid. Its positive alpha values are
exploratory absolute costs, not a measured alpha0 calibration. Including lower
beta lets the length penalty discourage long spellings without imposing the
same increase on short OOV words. Alpha=0 controls are retained. Interpret
results using deletions, fallback lengths and paired IV/OOV errors alongside WER.

Completed points retain their exact descriptors/configs and remain in
`run_limited_fallback`; the new entrypoint skips those rows rather than creating
duplicates. Both grids use the same prepared lexica and acceptance dependencies.
The build, LM scales, prior, blank penalty, word reward, pruning, normalization,
guards and standard lexicon remain unchanged. No new baselines, reduced lexica,
Approach C, neural/subword LMs, or held-out evaluation are added in this round.

## Granite alpha tuning (before beam tuning)

The preceding Granite runs **fixed** alpha=0; they did not establish that it
was optimal. Their best beta values were 3 without an LM (2.644% WER) and 8
with the 4-gram at scale 0.3 (2.604%). Greedy reached 2.817%. Despite the WER
gain, reference OOV-word accuracy was lower than greedy, so inspect IV/OOV
accuracy as well as overall WER when selecting the next settings.

`run_all` appends `run_granite_alpha_tuning`, using Approach B and the same
validated standard lexicon and pinned decoder:

| LM | Beta | Alpha | New runs |
|---|---|---|---:|
| None | 0, 1, 2, 3, 4 | 0, 0.25, 0.5, 1, 2 | 20 |
| Original 4-gram, scale 0.3 | 0, 2, 4, 6, 8 | 0, 0.25, 0.5, 1, 2 | 20 |

These are Cartesian grids in absolute natural-log cost units, not values
derived from the former alpha0 calibration. Lower beta settings allow the
length penalty to replace some of the per-word penalty. Ten alpha=0 controls
already exist in the preceding stages and retain their exact configurations;
the new entrypoint skips them, adding **40 dev-other recognition jobs**.

Within-word and word-end beams remain 256, both score thresholds remain 16.
LM scales, prior, blank penalty, word reward, normalization and degeneracy
guards are unchanged. No new BPE runs or held-out runs are added. Review this
grid first, then tune beam sizes and score thresholds at the selected costs;
beam tuning is intentionally not scheduled in this stage.

## Completed Granite alpha grid and pruning stage

All 40 new scoring outputs are present. Dev-other WER (%), including reused
alpha=0 controls:

| No LM: beta | alpha=0 | 0.25 | 0.5 | 1 | 2 |
|---|---:|---:|---:|---:|---:|
| 0 | 2.741 | 2.712 | 2.689 | 2.670 | 2.656 |
| 1 | 2.683 | 2.664 | 2.662 | 2.656 | 2.633 |
| 2 | 2.658 | 2.646 | 2.644 | 2.638 | 2.642 |
| 3 | 2.644 | 2.640 | 2.648 | 2.638 | 2.660 |
| 4 | 2.648 | 2.642 | 2.644 | 2.642 | 2.677 |

| 4-gram scale 0.3: beta | alpha=0 | 0.25 | 0.5 | 1 | 2 |
|---|---:|---:|---:|---:|---:|
| 0 | 3.085 | 3.008 | 2.959 | 2.852 | 2.699 |
| 2 | 2.806 | 2.761 | 2.720 | 2.677 | 2.609 |
| 4 | 2.677 | 2.658 | 2.640 | 2.609 | 2.586 |
| 6 | 2.629 | 2.613 | 2.607 | 2.584 | 2.580 |
| 8 | 2.604 | 2.596 | 2.588 | 2.592 | 2.611 |

Best observed settings, not established optima (both are at the alpha limit):

- No LM: beta=1, alpha=2 gives 1355 errors (1159 substitutions, 71 deletions,
  125 insertions), versus 1361 (1155/86/120) for beta=3, alpha=0. There are
  16 improved and 11 worsened utterances by total error count.
- 4-gram: beta=6, alpha=2 gives 1328 errors (1123/121/84), versus 1340
  (1117/139/84) for beta=8, alpha=0. There are 22 improved and 11 worsened
  utterances. Beta=6, alpha=1 is only two errors worse than the winner.

Both winners have 2864 utterance diagnostics, zero guard failures and at most
five pieces in an unattested fallback word. Small reference-count differences
(e.g. 51468 vs 51469) result from the HF pair-dependent compound-equivalence
rule; this is not a change of evaluation data.

### IV/OOV trade-off

Use the same original-reference-word partition in every condition: IV means
the raw word is in the ordinary closed Granite lexicon (case-insensitive).
Normalize the full utterance with the pipeline normalizer and map normalized
words back to their original reference words. For contextual normalization
changes, use the corresponding source group (all such groups here are IV).
Project Sclite matches back through compound-boundary equivalences, checking
that the reference character stream is unchanged. An original word counts as
correct only when all its scored normalized parts match. Exclude the 24 IV
words removed by normalization. This leaves 50,629 IV and 295 OOV words;
insertions are reported in WER, not reference-word accuracy.

| Condition | WER | Correct IV / 50,629 | Correct OOV / 295 |
|---|---:|---:|---:|
| Greedy | 2.817% | 49,564 (97.896%) | 54 (18.31%) |
| No LM, beta=3, alpha=0 | 2.644% | 49,665 (98.096%) | 49 (16.61%) |
| No LM, beta=1, alpha=2 | 2.633% | 49,683 (98.132%) | 43 (14.58%) |
| 4-gram, beta=8, alpha=0 | 2.604% | 49,654 (98.074%) | 43 (14.58%) |
| 4-gram, beta=6, alpha=2 | 2.580% | 49,669 (98.104%) | 37 (12.54%) |

Positive alpha improves total WER mainly through IV/deletion improvements;
it does **not** improve OOV recovery at these selected operating points. The
alpha=0 controls remain important. These small dev-set gains are not a
statistical-significance claim or a held-out result.

### Next: beam and threshold grid

`run_all` now appends `run_granite_beam_tuning` with the two lowest-WER pairs
above fixed, on dev-other only. Cross beams **64, 256, 1024** with score
thresholds **8, 16, 32, 64**. Within-word and word-end beam sizes are tied;
their thresholds are also tied, so a narrow second-stage cutoff cannot mask
the wide-threshold probe. This is not yet independent tuning of the two stages.

There are 24 grid points, **22 new recognition jobs**: beam=256/threshold=16
is already complete for both conditions. All other parameters, guarded
traceback formatting, validated lexicon and acceptance dependencies stay
unchanged. Existing jobs continue to register search/total RTF and RASR
statistics. Compare cost/WER, IV/OOV accuracy and runtime; wider pruning need
not improve WER. Do not call the current alpha range exhausted or select a
final OOV-robust system solely from the WER minimum.

## Completed pruning study and active Transformer comparison

Threshold 64 is excluded from the analysis. All nine no-LM runs at thresholds
8/16/32 completed. Beam 256 and 1024 have identical normalized hypotheses
(1355 errors, 2.63265% WER); beam 64 adds one deletion. All six successful
4-gram runs at thresholds 16/32 have 1328 errors (2.58019%). All three 4-gram
threshold-8 runs fail with an empty traceback on `4572-112381-0018`.
Widening pruning does not improve IV or OOV accuracy. The selected efficient
operating points are no-LM beam=256/threshold=8 and 4-gram beam=64/threshold=16.
Their search RTFs are 0.00544 and 0.06567, respectively (single-run measurements).

### New grid: original word Transformer LM

`run_granite_transformer_tuning` registers **18 new dev-other recognitions**:

| Condition | LM scales | Beta | Alpha | Runs |
|---|---|---|---|---:|
| Closed tree, fallback disabled | 0.3, 0.5 | 0 | not applied | 2 |
| Known-excluding fallback (B) | 0.3, 0.5 | 0, 4, 8, 12 | 0, 2 | 16 |

The fallback grid is Cartesian. Both stages use beam 256 / score threshold 16.
Do not assume that the 4-gram's cheaper pruning point transfers to the neural
LM. Wider-budget verification is deferred until selection on dev. Prior,
blank penalty, word reward, lexicon, scoring normalization, strict sentence-end
handling, native acceptance checks and fallback guards remain unchanged.
The closed controls use the matching prepared lexicon without fallback pieces.
No LM adaptation, subword LM, training or held-out evaluation is added.

The model and word-index vocabulary are pinned to the exact artifacts used by
the completed `tree_trafoLM_lm-scale-0.3` recognition, recorded in
`WriteRasrConfigJob.YIfkcUqOcSQo`:

- `AddOnnxMetadataJob.wJhl1BGaB8Lo/output/model.onnx`: original 96-layer,
  512-hidden-dimension KV-cached word Transformer; retained as the numerical
  reference, not the active recognition model.
- Since 2026-09-25, the active 18 conditions use a stateless export of the
  **same checkpoint**, `ReturnnTrainingJob.UCMntNBI6vDq/output/models/epoch.030.pt`.
  Dimensions match its original export `TorchOnnxExportJob.SLHrUJ7Gtfum`.
  RASR uses `onnx-stateless` with CUDA, forwarding full histories without KV
  cache inputs. Each job requests a 24-GB GPU, shared with the Granite encoder.
  The existing stateless LM batch cap is 3. Search beams/thresholds, LM scales,
  alpha and beta are unchanged. Descriptors end in `_gpu-stateless`; these
  replace the CPU/KV sweep registrations, not add a second sweep.
- `LmIndexVocabularyFromLexiconJob.lGRAjpScZIsf/output/lm.vocab.txt`:
  200,002 output indices, UNK=1 and shared BOS/EOS=0. The shared sentence-marker
  index is deliberate; acoustic label ids are a separate inventory.

`ValidateOovNeuralLmJob` checks the unknown lemma in both lexica, explicit word
indices, ONNX output dimension, score normalization and finite positive UNK
probability after BOS, BOS+UNK, and BOS+THE. It tests UNK as a history input as
well as an output column. The stateless validation additionally compares every
output column against the cached reference, including a 17-token history.
These short correctness probes run on CPU; recognition runs on GPU. All 18
jobs depend on its byte-identical vocabulary copy, so a failed check blocks
recognition without changing model scores. The new report is registered as
`tools/rasr-oov-6e07a2e5/word-trafo-gpu-stateless-validation`.

The real ONNX probe was also run locally: UNK costs were 6.31375, 2.85245 and
5.23520 nats for those three histories; probability masses were within 0.000028
of one. This finite smoke test is not a guarantee for every possible history.
Those measurements were for the original cached export. The new stateless
export and equivalence gate run as jobs; no LM training is scheduled.

### Active-pipeline cleanup

Historical experiment factories are preserved for explicit reruns, but
`run_all` no longer registers:

- Granite's old closed-tree LM-scale and text-normalized-LM sweeps;
- the old-build `oov_prototype.run` grid;
- Approach A, limited-fallback, alpha/beta and beam tuning grids in `oov_length`.

`run_retained_comparisons` instead keeps **nine already-completed references**:

- Granite same-build greedy, closed/no-LM and closed/4-gram;
- Granite open/no-LM (beta=1, alpha=2, beam=256, threshold=8) and open/4-gram
  (beta=6, alpha=2, beam=64, threshold=16);
- the alpha=0 OOV-recall references (no-LM beta=3 and 4-gram beta=8), both
  beam=256/threshold=16;
- BPE closed/4-gram and beta=10/alpha=1 open/4-gram, both at scale 0.6.

All nine scoring hashes were checked against the finished jobs and match.
Existing result files are untouched; removing registration does not cancel
Slurm jobs or delete their outputs. Unrelated recognition and reduced-lexicon
experiments are unchanged. Select the Transformer candidates using WER **and**
IV/OOV accuracy, retaining a better OOV-recall point if the WER minimum trades
away OOV recovery.

### 2026-09-27: Transformer pruning sweep

The completed GPU/stateless cost sweep selects LM scale **0.3**, beta **4**,
alpha **2**: 1281 errors (2.489% WER) versus 1360 (2.642%) for the matching
closed tree. Search RTF is still 2.50, motivating an independent beam-cap grid.

`run_granite_transformer_pruning_tuning` now replaces the completed 18-point
cost-sweep registration. On **dev-other only**, cross within-word beam caps
**64 / 128 / 256** with word-end caps **4 / 16 / 64 / 256**. Both score
thresholds remain **16**; threshold tightening is deferred because previous
Granite fallback runs failed the degeneracy guard at threshold 8.

This gives **11 new runs**, plus the existing 256/256 best open-tree point and
the existing scale-0.3 closed-tree control, with their original identities.
All other settings stay fixed: checkpoint, vocabulary, GPU/stateless execution,
24-GB GPU request, LM scale, beta/alpha, prior/blank penalty, lexicon, reference
normalization, strict sentence end and traceback guard. New descriptors contain
`_beam{B}_wordbeam{W}_thr16_gpu-stateless`. RTF and search-space statistics are
registered as before. Pick a speed/accuracy operating point against the wide
control before changing thresholds or evaluating held-out data.

Qwen remains closed-tree. Its matched closed-tree controls on 8c1ac793 score
2.722% (4-gram) and 2.642% (word Transformer), versus 2.763% for Qwen at scale
0.3. Thus the 4-gram comparison does not rely on giving only the 4-gram an open
tree. Open-vocabulary Qwen fusion is not enabled by this pruning change:
fallback spellings need their actual Qwen token histories, not a fabricated
word-level UNK token. No Qwen configurations are changed here.

### 2026-09-29: bypass the stateless LM speculative queue

The 13 active Transformer pruning/control conditions now set the LM's
`max-batch-size = 1` and append `_lm-batch1` to their descriptors. This disables
the pinned `OnnxStatelessLm`'s speculative history queue, which otherwise retains
discarded histories across utterances (the clear in `startHistory()` is after
the cached-history early return). Several old runs accumulated substantial
memory and stalled. This is a configuration workaround, not a RASR code fix.

The LM remains **GPU, stateless, full-history** with the same completed export
and validation job. All beam caps, thresholds, LM scale, beta/alpha, lexica and
scoring are unchanged. Even the open/closed controls are rerun with batch size 1
for a consistent runtime comparison. New RASR configs give distinct recognition
job identities. Historical factories and finished results remain available;
the active driver no longer schedules the old batch-3 pruning grid. No running
jobs are cancelled automatically.
