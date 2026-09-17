# SAE §4b — Sparse acoustic codes from frozen w2v2 features

## State

FIRST RUN COMPLETE, AUDITED — `BatchTopKTrainingJob.1Z8e0Io2tv5b` reached the fixed 200,000-update
endpoint; `BatchTopKProbeJob.aghUh8TQoiem` completed its supervised held-out readout. Config:
`config/sae_4b_audiosae_w2v2.py`; concrete artifacts are below. The predefined phonetic-discrimination
criterion is met; improvement over normalized dense L15 is not met. Exact results, operating points
and audit qualification are in "First-run result" below.

No jobs or additional arms are pending within this first-run budget. The next experimental decision
is whether to define a later acoustic-code usefulness comparison; this result does not establish
unsupervised phone naming or justify a cycle-target substitution. Gold-based SAE tuning remains
prohibited. Phase closure remains with the user. Shared constraints are in `SAE_ref.md`.

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
