# SAE §4b — Sparse acoustic codes from frozen w2v2 features

## State

WEIGHTED-L1 FOLLOW-UP COMPLETE, AUDITED — `WeightedL1TrainingJob.1Z8e0Io2tv5b` reached the fixed
200,000-update endpoint and dependent `BatchTopKProbeJob.pyZdQ8mmOZgK` completed. Config:
`config/sae_4b_weighted_l1_w2v2.py`. The probe's legacy class name also serves weighted-L1 checkpoints.
The supervised phonetic-information gate passes; improvement over matched dense features and over
the earlier BatchTopK SAE fails. Measured sparsity, phoneme multiplicity, paired comparisons and
audit qualifications are in "Weighted-L1 result and sparsity" below.

No jobs or further arms remain queued in this budget. A new sparsity operating point or a cycle
comparison would require a separately registered protocol; labels cannot select the SAE setting.
The next experimental decision and phase closure remain with the user.

The completed BatchTopK run remains the comparison: `BatchTopKTrainingJob.1Z8e0Io2tv5b` and
`BatchTopKProbeJob.aghUh8TQoiem`, under `config/sae_4b_audiosae_w2v2.py`. Its audited results and
requested sparsity statistics are below. The follow-up's active-feature count must be measured,
not forced to 50. Gold-based SAE selection/tuning remains prohibited. Cycle substitution remains
outside this stage and phase closure remains with the user; shared constraints are in `SAE_ref.md`.

## Objective and hypothesis

Use the existing speech-only SSL representation, frozen w2v2 L15 at 50 Hz and 1024 dimensions.
A sparse autoencoder reconstructs each frame from a small set of active dictionary features. The
hypothesis is that these features separate useful phonetic variation sufficiently to improve an acoustic
target for the cycle. Reconstruction and sparsity alone do not prefer phonetic identity over speaker,
channel or prosody. Multiple active features also do not directly define one token per phoneme.

This is an acoustic-representation phase with its own comparison, not a standalone speech-to-text
initializer. No GAN, paired transcript supervision, supervised-ASR teacher or gold-fitted phone mapping
may enter representation fitting, code selection or training. The current w2v2 checkpoint is inherited;
supervised HuBERT/Whisper checkpoints appearing in the literature are not admissible replacements.

## Literature that shapes the plan

- [Parra, IJCNLP-AACL SRW 2025](https://aclanthology.org/2025.ijcnlp-srw.1.pdf) finds sparse events and
  broad phonetic associations in HuBERT activations. Its 4096 features have roughly 3% active per frame;
  features may mark transitions rather than stable phone interiors. The underlying HuBERT checkpoint
  was ASR-fine-tuned on 960 h of labeled LibriSpeech, and TIMIT labels interpret the features. This
  motivates temporal checks but does not establish label-free phoneme discovery on our SSL features.
- [Aparin et al., AudioSAE 2026](https://arxiv.org/html/2602.05027) includes self-supervised HuBERT:
  6144 sparse features with about 50 active per frame retain phonetic and nuisance information. Phone
  naming uses forced-aligned labels, and its 0.89 frame score accepts any matching active feature.
  Therefore multi-feature interpretability is evidence for phonetic information, not unsupervised PER.
  Speaker/noise dependence and temporal stability belong in the acoustic-code diagnostic.
- [Gundogdu et al., Interspeech 2020](https://www.isca-archive.org/interspeech_2020/gundogdu20_interspeech.pdf)
  combines sparse correspondence learning with temporal competition and vector quantization on PLP.
  English ZeroSpeech development ABX is 29.6 for K64 k-means versus 29.2 for sparse+WTA+VQ, at different
  bitrates; stronger compression is not a demonstrated large phonetic-discrimination gain. The full
  system's speaker-adversarial component is outside the proposed method. This motivates an explicit
  same-feature clustering baseline and matched code-rate interpretation.

Verified full-text conditions and limits:
`reports/codex_4a_sparse_w2v_codes_literature_2026-09-17.md`. None of these checked studies establishes
named-phone recovery from our exact frozen w2v2-L15 stream or success in our cold-start cycle.

## Original preliminary experimental sequence (superseded first-stage scope below)

1. Fit one speech-only sparse autoencoder on frozen L15 features. Select its settings using held-out
   acoustic criteria alone; architecture, sparsity and training budget remain unspecified until design
   review. Reconstruction quality is a diagnostic, not evidence of phonetic or ASR improvement.
2. Produce a K64 categorical stream by clustering the sparse activations. Compare it with direct K64
   clustering of identical L15 inputs and the existing MFCC K64 target, using the same speech and frame
   clock. K64 is inherited from the §4a content auxiliary. Inspect occupancy, run lengths, temporal
   stability and speaker dependence; distinguish vocabulary size from actual entropy/bitrate. Gold
   phone association or ABX, if later included, is a sealed diagnostic and selects no settings.
3. Only if the representation comparison warrants it, propose a fixed-target substitution in §4a's
   categorical phone-output content term. Keep the cycle, reverse K500 units, numerical precision,
   initialization, schedule and evaluation matched to its control. This later cycle comparison is not
   authorized by phase registration.

## Original preliminary gate and unresolved design

No scientific acceptance threshold or compute budget is approved yet; execution is explicitly held.
The future representation gate must measure acoustic-code usefulness against direct L15 clustering,
not merely sparse reconstruction or a phone map fitted to gold. A private acoustic code can satisfy an
intermediate objective without recovering phone names. A claim of improved unsupervised ASR additionally
requires the existing label-free selection protocol and paired downstream PER/acoustic-dependence
evaluation; the original cold-start gate in §4a is not weakened by this phase.

Before execution, settle whether the sparse code is temporally stable enough to quantize, how code rate
is controlled across baselines, which nuisance diagnostics remain label-free, and what improvement and
budget justify a cycle target swap. This document records a hypothesis and comparison, not a result.

## Execution authorization and first-stage scope (2026-09-17)

The user's later instruction is to start §4b by reproducing an SAE on wav2vec2 and extracting phoneme
information in a supervised way, following AudioSAE. This replaces the execution hold above; no
previous numerical gate existed. The first deliverable is the speech-only SAE and a disclosed,
held-out phoneme diagnostic. The direct-feature baseline must use the same labeled items and phone
inventory. AudioSAE's multi-feature coverage score must be distinguished from single-label frame
accuracy and sequence PER. The architecture, schedule, splits, budget and interpretation criteria
will be recorded here before launch; downstream code and cycle gates remain unresolved.

## First run: fixed operating point and readout

**Source and limits.** `reports/codex_4b_audiosae_protocol_2026-09-17.md` verifies the
[AudioSAE paper](https://arxiv.org/html/2602.05027) and released code at `5c6708f487244a581397e612e4fce73e70f127e0`.
This is a wav2vec2 adaptation, not numeric replication of HuBERT/Whisper results. The authors omit
training/probing utilities; ongoing decoder constraints, threshold calibration and the stated sparsity
warmup are unspecified. The speech-only corpus, fixed endpoint and explicit inference rule below are
declared adaptations; the published 0.89 phone coverage is not a comparable target.

**Inputs.** Frozen SSL-only `wav2vec2-large-lv60`, explicit `hidden_states[15]`, 1024 dimensions at
50 Hz, no pooling. Existing waveform normalization is unchanged; add per-frame L2 normalization
for both SAE and dense baseline. All 28,539 train-clean-100 utterances / 18,088,388 frames in
`L15FeatureHdfJob.e2athsQ218Og` train the SAE. Source is `AvStatesJob.Dsynh5MqmgjY`, with no AV
checkpoint. Dev features come from `AvStatesJob.c4Ak1rACchRC`; use only its 5,567 dev utterances,
excluding its 2,849 seed utterances. Concrete paths and extraction lineage:
`reports/codex_4b_inventory_2026-09-17.md`.

**SAE.** BatchTopK, 8192 features (8× expansion), average activity target 50. Encode
`ReLU(W_enc (x - b_dec) + b_enc)` and retain the global top `50 * B` entries per training batch;
decode linearly with `b_dec`. Decoder columns have unit norm at initialization, encoder copies
their transpose, biases start at zero. No ongoing norm constraint, auxiliary loss, dead-feature
resampling or undefined sparsity warmup. Reconstruction loss is mean squared L2 distance per
frame (sum across dimensions). Adam: lr 0.0002, betas (0.9, 0.999), default epsilon 1e-8,
weight decay zero. Uniform frame sampling with replacement, batch 2500, seed 0, 200,000 updates;
lr stays fixed through update 160,000, then decays linearly to zero. Select the final endpoint,
never a label-selected checkpoint. FP16 cache storage, float32 training. This presents 500 million
frames, about 27.6 passes over the cached pool. Fixed inference is whole-utterance BatchTopK:
top `50 * T` positive entries across an utterance, without a learned threshold. This permits
variable activity per frame; it is not per-frame top-50.

**Supervised diagnostic.** All 2,703 dev-clean utterances (40 speakers) fit feature naming and the
two probes; all 2,864 dev-other utterances (33 disjoint speakers) form the sealed held-out readout.
These memberships were checked from metadata without reading held-out phones. This clean-to-other transfer is
reported explicitly. No validation split is needed because settings and endpoint are fixed.
Gold is the existing MFA stress-free ARPAbet-39 plus SIL frame alignment, rasterized at the existing
frame centers `(t + 0.5) / 50` with the source overlap convention. Unknown/uncovered spans are excluded and counted, never silently made
SIL. Every comparison uses identical retained frames. A feature receives a phone name only when
strictly more than half its activated fit frames carry that phone. Freeze this map before scoring.
Report any-named-active-feature match coverage and strongest-named-active-feature single-phone
accuracy, with abstention counted wrong; these are distinct metrics.

Fit linear softmax probes to normalized dense L15 and to SAE activations, using the existing
`repr_audit_w2v2.probe` constants: fit-only feature mean/std (+1e-5), Adam lr 0.001, weight decay
1e-5, batch 4096, cross-entropy, 25 epochs, seed 0, final epoch only. Also freeze fit-majority
phone predictors for all frames and for nonsilence frames. Store per-utterance counts, speaker IDs,
feature naming counts and held-out feature precision/recall. Report both all-frame and nonsilence
scores, reconstruction, positive L0 and dead-feature fraction. Training reconstruction is measured
in 2500-frame batches; held-out reconstruction uses the fixed whole-utterance inference rule.

**Predefined evidence criteria.** Completion means the source-traced fixed run and held-out
diagnostics exist; no scientific success is presumed. Evidence of phonetic discrimination requires
the SAE linear probe's **nonsilence** accuracy gain over the fit-nonsilence-majority baseline to have
a paired speaker-clustered 95% CI strictly above zero. Improvement over dense L15 requires the
same criterion for SAE minus dense nonsilence accuracy. Use 2000 speaker resamples, seed 0, from
the existing `eval_jobs.py` convention. CIs condition on this one trained pair and exclude training
seed variability. SIL-only gains, permissive coverage, reconstruction and passed code checks cannot
satisfy the phonetic criterion. Neither criterion establishes unsupervised phone names or ASR gains.
An 8192-dimensional SAE probe beating the 1024-dimensional dense probe establishes easier linear
recovery at these fixed operating points, not an increase in intrinsic information.
Audit the result before drawing a consequential conclusion; phase closure remains with the user.

**Budget and review.** One 200,000-update SAE and two 25-epoch linear probes, existing caches only,
no parameter or seed sweep. Use one GPU per job within the inherited four-GH200 experiment allocation
and 11.5-hour scheduler task cap; resumable checkpoints preserve the same endpoint. No extra
experimental arm or downstream cycle run is included. First-phase review:
`reports/codex_4b_design_review_2026-09-17.md`. Its corrections are incorporated above. Before spend,
check a real cached minibatch for finite sparse loss/gradient, unit input norm and mean positive
activity at most 50, guarding against the released model's dense-default `forward()`.

## Run and evidence pointers

- Training: `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/sparse_autoencoder/BatchTopKTrainingJob.1Z8e0Io2tv5b`.
- Diagnostic: `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/sparse_autoencoder_probe/BatchTopKProbeJob.aghUh8TQoiem`.
- Registered outputs: `output/sae/4b/audiosae_w2v2/` — `checkpoint.pt`, `train_metrics.json`,
  `probe_metrics.json`, `per_utterance.json`, `features.npz`, `probes.pt`. Durable payloads live in
  `artifacts/sae_4b/<full-job-id>/`.
- Implementation commits: training `ccff17b1f`, probe `dc24de9e6` in `i6_experiments` on `haotian`.
  Review: `reports/codex_4b_code_review_2026-09-17.md` (no findings). Source-cache checks establish
  sparse execution/finite gradients only, not model quality. Submission evidence:
  `reports/codex_4b_launch_2026-09-17.md`.

## First-run result (2026-09-17)

**Verified completion.** Both graph nodes finished and all six registered outputs opened successfully.
The loaded checkpoint and training metrics agree on 200,000 updates / 500 million frame presentations
from the registered 28,539 utterances / 18,088,388 frames. Training Slurm ID `1860439`, diagnostic
`1860927`. Completion evidence: `reports/codex_4b_completion_2026-09-17.md`; verbatim result extraction:
`reports/codex_4b_result_extraction_2026-09-17.md`.

**Held-out phone readout.** Dev-clean fitting used 806,709 aligned frames from all 2,703 utterances.
Dev-other scoring used all 2,864 utterances / 33 speakers: 741,817 aligned frames, including 733,242
nonsilence frames. Of 919,980 raw dev-other frames, 178,163 (19.37%) were uncovered by usable alignment
and excluded; no unknown-phone frames were recorded. The arms use identical retained frames. The
excluded seed-cache payload was 2,849 utterances / 1,797,904 frames. Train, fit and test speakers
are disjoint. All scores below are percentages at the fixed final SAE/probe endpoints.

| Readout | Nonsilence frames | All aligned frames |
| --- | ---: | ---: |
| SAE linear probe | 73.64 | 72.88 |
| Normalized dense L15 linear probe | 80.19 | 79.45 |
| Fit-majority phone predictor (AH) | 7.41 | 7.32 |
| Strongest named active feature, single phone | 67.72 | 66.94 |
| Any named active feature matches gold, coverage | 79.79 | 78.87 |

SAE minus fit-nonsilence-majority accuracy is **+66.24 percentage points**, paired speaker 95% CI
**[+63.59, +68.75]**: the predefined phonetic-discrimination criterion is met. SAE minus normalized
dense L15 is **−6.54 points**, CI **[−8.18, −5.05]**: the predefined improvement criterion is not met.
Dense L15 yields 47,980 additional correct nonsilence frames (587,976 versus 539,996). Both contrasts
have the same respective sign on every one of the 33 held-out speakers.

**Sparse representation and naming.** Supervised fit-only naming assigned a phone to 2,730 of 8,192
features under the strict majority rule. `features.npz` contains feature-to-phone indices, phone names,
fit/held-out counts and held-out precision/recall; naming alone does not establish held-out feature
purity. Full-train mean squared L2 reconstruction error per frame was 0.1042833 with mean positive
L0 = 50 and two features inactive on that final pool read. Whole-utterance held-out inference also has mean L0 = 50;
dev-clean/dev-other MSE **per dimension** is 0.0001183709 / 0.0001514811, with dead-feature fractions
0.0203857 / 0.0144043. Training and held-out reconstruction use different reported reduction scales
and BatchTopK grouping; do not directly compare their raw numbers.

**Audited interpretation.** This single configuration retains substantial phonetic information accessible
to supervised labeling and probes, while its fixed linear readout is worse than the normalized dense
baseline. It provides neither improved phoneme recoverability at this operating point nor unsupervised
named-phone/ASR evidence. The weaker fixed readout does not establish that every SAE configuration or
later acoustic-code use must fail; no gold-informed retuning follows from this diagnostic.

Fresh-context audit: `reports/codex_4b_result_audit_2026-09-17.md` (`DONE_WITH_CONCERNS`). It independently
re-summed per-utterance scores, verified masks, speaker separation, checkpoint hash, label timing and
the sign of every speaker's paired delta. Exact bootstrap percentile endpoints were read from the
saved score file rather than independently regenerated; their side of zero follows independently
from the per-speaker signs. CIs remain conditional on this one trained pair, as preregistered.

## Sparsity and phone multiplicity (user request, 2026-09-17)

These are descriptive summaries of the same fixed SAE and supervised phone mapping, computed by
`analysis/sae_4b_sparsity_stats.py` from the three existing result files. Source hashes, exact numbers,
all 40 phone rows and all 8,192 feature rows are in `reports/sae_4b_sparsity_2026-09-17/`
(`summary.json`, `phones.csv`, `features.csv`, `phoneme_features.png` / `.pdf`). Execution and
fresh-context audit: `reports/codex_4b_statistics_execution_2026-09-17.md` and
`reports/codex_4b_statistics_audit_2026-09-17.md` (DONE). No model, label map or gate was changed.

**Activation sparsity versus inactive features.** The raw-frame mean L0 reported above corresponds to
0.61035% nonzero activation entries and **99.38965% zeros**. BatchTopK fixes the mean budget per group;
it does not require every frame to have the same activity. Dictionary usage is uneven: on the full
train pool, the median feature fires on 0.03799% of frames, the 90th percentile on 1.34492%, and the
most frequent feature on 54.88968%. These are quantiles across features, not across frames. Only
0.02441% of dictionary features are inactive on that final full-train read; inactivity on raw
dev-other is 1.44043%. Sparse activation therefore differs from an unused dictionary.

**Distinct dictionary features assigned to a phone.** A label requires strictly more than half of
the feature's activated **aligned dev-clean frames** to carry that phone. The fitted naming total
above consists of 2,729 speech-phone features and one SIL feature. Across all 39 speech phones,
including zero counts, the mean is **69.97 features per phone**, median **25**, range **0–264**.
There are assignments for 38 phones; ZH has none under this rule. The mean restricted to those 38
phones is 71.82. Across all 40 labels including SIL, it is 68.25. The 5,462 unassigned features
(66.67% of the dictionary) did not pass this naming rule; this does not establish absence of
phonetic information in them.

| Phone | Fixed dev-clean labels | Same label remains >50% on dev-other |
| --- | ---: | ---: |
| EH | 264 | 205 |
| AH | 259 | 177 |
| AE | 236 | 177 |
| S | 95 | 62 |
| CH | 3 | 2 |
| G | 2 | 1 |
| JH | 1 | 1 |
| ZH | 0 | 0 |

The held-out strict-majority criterion, applied to the **same fixed labels** with positive held-out
support, retains 1,967 speech features: **50.44 per speech phone**, median 16, range 0–205. The SIL
feature does not retain its majority. These are descriptive validation counts, without relabeling
or model selection. Forty-nine fit-named features have no aligned held-out activation support.

**How many matching features fire together?** On the 733,242 aligned nonsilence dev-other frames,
1,391,643 activations bear the current gold phone's fixed label: **1.898 matching active features
per frame**, averaged with frame weights. All features together contribute **52.525 active features
per frame on this subset**; the raw all-frame mean has a different denominator. For example, S
has 95 assigned dictionary features but only 3.611 matching features active on an average S frame;
EH has 264 assigned features and 1.349 matching features per EH frame. Distinct dictionary counts
and simultaneous activity answer different questions. Framewise activity quantiles and the fraction
of frames with no active features are unavailable from the stored aggregates.

## Weighted-L1 follow-up: preregistered protocol (user 2026-09-17)

**Authorization and source.** The user now requests the regularization from
[*Scaling Monosemanticity* (May 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/).
This funds one new fixed-endpoint SAE plus its supervised diagnostic, beyond the completed first
run. Verified method and linked April training defaults:
`reports/codex_4b_anthropic_protocol_2026-09-17.md`. This is an adaptation of that recipe to speech,
not a reproduction of Claude's proprietary corpus, million-feature widths or undisclosed selected
learning rate. The original BatchTopK results and gates above are preserved.

**Data, scale and model.** Reuse exactly the same raw frozen L15 train/dev caches, 1024 input
dimensions, 8192 dictionary features, 50-Hz clock and split membership. Replace individual frame
unit normalization with a single training-only scalar
`c = sqrt(1024 / mean_train(sum(raw_x ** 2)))`, estimated over all 18,088,388 train frames and saved
in the checkpoint. Apply `x = c * raw_x` unchanged on dev. This follows May's mean **squared** norm
rule, not April's different mean norm rule. No dev or gold contributes to `c`.

Encode `f = ReLU(W_enc x + b_enc)` directly, without subtracting the decoder bias and without
TopK or activation thresholding. Reconstruct `x_hat = W_dec f + b_dec`. Decoder columns start
with random directions and norm 0.1, encoder weights copy their transpose, and both biases start
at zero. Decoder norms remain unconstrained. The exact loss is
`mean_frames(sum_dimensions((x_hat-x)^2) + lambda * sum_features(f * decoder_column_norm))`.
The norm factor participates in backpropagation. Its target coefficient is **lambda = 5**, with
a linear warmup over the first 10,000 updates; an activity target of 50 is not imposed.

**Training budget and selection.** One seed-0 model, uniform frame sampling with replacement,
2500 frames/update, 200,000 updates, final checkpoint only. This retains the earlier 500-million
frame-presentation budget and differs from the source's batch defaults and single-pass data regime.
Use the linked April defaults: Adam lr 0.00005, betas (0.9, 0.999), weight decay zero, global gradient
norm clip 1, constant lr through update 160,000 followed by linear decay to zero. Epsilon 1e-8 and
float32 arithmetic are inherited local settings. No dead-feature resampling, auxiliary losses,
post-training adjustment or label-selected settings. Retain resumable checkpoints and the existing
one-GPU task / exclusive four-GH200 node allocation with 11.5-hour task limit; no sweep is funded.

**Diagnostic and comparisons.** Reuse the same dev-clean fit / dev-other held-out protocol, masks,
linear-probe settings, frame naming threshold and speaker-paired bootstrap as the first run. The
new dense baseline receives the same globally rescaled inputs as the new SAE. For interpretation
and the SAE probe, use the paper's contribution magnitudes `g_i = f_i * ||W_dec[:,i]||`; reconstruct
using raw `f`. Record raw ReLU and contribution activity separately if zero decoder columns make
them differ. Phone naming still uses positive activations and the same strict fit-majority rule.
Report reconstruction with its scale and the dimensionless `SSE / sum(||x||²)`, loss components,
decoder norms, mean L0, per-frame activity histograms/quantiles and inactive-feature counts. The
earlier aggregate-only limitation on framewise activity applies to the old run, not this new readout.

The existing phonetic criterion is reused: held-out nonsilence SAE-minus-fit-majority accuracy
must have paired speaker 95% CI strictly above zero. Improvement over the new matched dense
baseline uses the same criterion. Also compare the new SAE with the completed BatchTopK SAE
on identical per-utterance frames, using paired speaker intervals and the same positivity criterion.
That contrast is between complete recipes: normalization, initialization, encoder centering and
optimization differ alongside the sparsity mechanism. It cannot identify the penalty alone as a
cause. No loss, sparsity count or probe result alone establishes unsupervised phone discovery or
licenses cycle substitution. Audit results before drawing a consequential conclusion.

**Run and evidence.** The reviewed recipe is committed as `30afce9a5` (training) and
`91e165ad6` (shared diagnostic). Concrete jobs, relative to this setup:

- `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/sparse_autoencoder_l1/WeightedL1TrainingJob.1Z8e0Io2tv5b`;
- `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/sparse_autoencoder_probe/BatchTopKProbeJob.pyZdQ8mmOZgK`.

The six registered outputs are under `output/sae/4b/weighted_l1_w2v2/`: `checkpoint.pt`,
`train_metrics.json`, `probe_metrics.json`, `per_utterance.json`, `features.npz`, and `probes.pt`;
payloads persist under `artifacts/sae_4b/` by full job ID. Execution evidence:
`reports/codex_4b_weighted_l1_launch_2026-09-17.md`. Method review:
`reports/codex_4b_weighted_l1_code_review_2026-09-17.md`. The completed paired comparison helper is
`analysis/sae_4b_compare_weighted_l1.py`; its output is
`reports/sae_4b_weighted_l1_comparison_2026-09-17.json`, including resolved inputs and SHA256 hashes.

## Weighted-L1 result and sparsity (fixed endpoint, audited)

**Completion and operating point.** Both jobs finished with exit 0 and all six registered outputs
open successfully. Slurm training `1863139_1` took **23m06s** (recorded training compute 1202.727 s);
diagnostic `1863364_1` took **5m16s**. The actual checkpoint and probe source agree on 200,000 updates,
8192 features and train-only input scale **0.07542069287767447**, estimated from all 18,088,388 frames
in 28,539 train-clean-100 utterances. Mean scaled squared input norm is 1023.99999058. Final full-pool
mean reconstruction squared L2 is 182.80207906 and mean decoder-norm-weighted L1 before multiplying
by lambda is 29.62830216; lambda is 5. Dimensionless reconstruction SSE/input energy is 0.17851766
on train and 0.23163436 on raw dev-other. Raw reconstruction losses across recipes have different
input scales. Completion evidence: `reports/codex_4b_weighted_l1_completion_2026-09-17.md`.

**Held-out phone readout.** The paired comparison verifies identical 2,864 dev-other utterances,
33 speakers, raw input and gold paths, masks, clock, phone inventory and final probe settings.
The denominators remain 741,817 aligned frames including SIL and 733,242 nonsilence frames;
178,163 of 919,980 raw frames are uncovered/unknown and excluded. Accuracies are percentages:

| Representation/readout | Nonsilence | Including SIL |
| --- | ---: | ---: |
| Weighted-L1 SAE | 68.1164 | 67.4172 |
| Matched globally scaled dense L15 | 80.3085 | 79.5700 |
| Earlier BatchTopK SAE | 73.6450 | 72.8844 |
| Fit-majority baseline | 7.4055 | 7.3199 |

Nonsilence weighted-L1-minus-baseline differences, with the registered 2000-resample, seed-0,
speaker-paired 95% intervals, are **+60.71 pp [57.20, 64.18]** versus fit majority,
**−12.19 pp [−14.90, −9.67]** versus matched dense, and **−5.53 pp [−7.21, −4.03]** versus
BatchTopK. Thus the phonetic-information criterion passes, while both improvement criteria fail.
The old/new comparison changes the complete SAE recipe and cannot attribute this difference to
the L1 penalty alone. The intervals are conditional on these fixed trained models, not training-seed
variability. No result establishes unsupervised phone naming or authorizes cycle substitution.

**Activation sparsity.** Positive raw ReLU activations are measured, not constrained to a fixed L0.
All decoder columns have nonzero norm; raw and contribution activity histograms agree on dev.
The full-train contribution histogram was not stored. Zero entries are the fraction of the
frame-by-feature activation matrix that is zero, distinct from zero-active frames and unused features.

| Domain | Frames | Mean active/frame | Min / median / p90 / max | Zero entries | Zero-active frames | Inactive features |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Full train pool | 18,088,388 | 25.7610 | 1 / 26 / 37 / 266 | 99.6855% | 0% | 18 |
| Raw dev-clean | 968,057 | 24.6357 | 0 / 24 / 35 / 688 | 99.6993% | 0.0001033% | 39 |
| Raw dev-other | 919,980 | 25.0982 | 0 / 24 / 37 / 558 | 99.6936% | 0.0013044% | 27 |

**Features per phoneme.** The same fixed dev-clean strict-majority naming rule assigns **2253**
features to speech phones and **1** to SIL, leaving **5938** unnamed. Across all 39 speech phones,
including zero-count ZH, the mean is **57.7692 features/phone**, median **43**, range **0–183**;
38 phones have assignments. The most assigned are S 183, AH 180, IY 152, IH 125 and AE 119;
the fewest are ZH 0, JH 5, OY 5, G 6 and CH 8. These labels describe supervised associations.

The same fixed labels remain strict majorities with positive support on dev-other for **1804**
speech features and zero SIL features: **46.2564 per speech phone**, median 37, range 0–165.
This held-out retention read does not select a new map. Across the 733,242 aligned nonsilence
frames, there are **2.632139 correctly named active features per frame**, against **26.920816 total
active features per frame**. These simultaneous counts differ from distinct dictionary features
assigned to a phone and use a different frame subset from raw-frame sparsity above.

Exact statistics, all 40 phone rows, all 8192 feature rows and PNG/PDF plots are under
`reports/sae_4b_weighted_l1_sparsity_2026-09-17/`; helper:
`analysis/sae_4b_weighted_l1_sparsity_stats.py`. Execution and source/hash checks:
`reports/codex_4b_weighted_l1_sparsity_execution_2026-09-17.md`. Fresh result/statistics audit:
`reports/codex_4b_weighted_l1_result_audit_2026-09-17.md` (DONE_WITH_CONCERNS: both improvement
intervals lie wholly below zero; supervised-label and complete-recipe limitations remain).
