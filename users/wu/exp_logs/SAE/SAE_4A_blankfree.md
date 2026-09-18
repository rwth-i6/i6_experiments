# SAE §4a — Blank-free cycle model

## State

User-authorized new model, 2026-09-18: evaluate, implement and execute autonomously. Main direction
is unsupervised silence removal, a stride-3 CNN as in the completed local wav2vec-U 2.0 reproduction,
and phone paths with adjacent repeat collapse and no CTC blank. Keep the cycle objective and exact
trigram marginalization. Cold model implementation and source review are complete (`323de02`,
with lattice `5cc9c72`); no model training has run. Joint VAD preparation
`BlankfreeVadHdfJob.SAjz8y1cT06g`, Slurm `1883648`, is running. The profile graph
`config/sae_4a_blankfree_profile.run` is launched; `BlankfreeCostProfileJob.3Nmqmdz20Wyi` waits
for that census and `FlatRecognizerInitJob.0J9d6wjrkRYH` (Slurm `1883714`). Next: verify the
actual-data cost/support gate, then launch the registered cold four-epoch experiment and inspect
outputs. Launch/artifact pointers: `reports/sae_blankfree_profile_launch_2026-09-18.md`.
Historical M512 and supervised-init results remain in `SAE_4A.md`; neither is a new-model result.

Follow-up user instruction: also run independent supervised initialization of both models on the
existing labeled 10 h, alongside cold trigram joint training. Preparation is shared, supervision
and checkpoints are not. The user explicitly chooses **separate initialization only**, with no
100 h adaptation after it. The support census `BlankfreeSeedSupportJob.ctjDDgnu43kW` is queued
behind the shared VAD job via `config/sae_4a_blankfree_seed_support.run`. Then inspect support
before releasing the independently implemented fits. Launch/artifacts:
`reports/sae_blankfree_seed_support_launch_2026-09-18.md`.
The original-M512 sampled-group replay is COMPLETE and audited: `GenericExecJob.awnR46wSTGCf`,
Slurm `1883777`, exit0:0 in19m19s. P6 importance weights concentrate despite diverse draws;
see "Sampled-group result" below and `reports/sae_sample_group_result_audit_2026-09-18.md`.
Event monitors are armed under `/e/scratch/spell/wu24/codex-sisyphus-monitor/`:
profile `monitor.4111620.Eq25JHJT` and seed support `monitor.4144463.6MdufYoD` (both verified
live at23:47). Sampled-group monitor `monitor.4154180.4DfP5FYL` is terminal DONE; no re-arm.
Resume the main branches from their terminal artifacts and release criteria; no new-model
training result is available yet.

## Objective and authority

The user proposes that handling alignment more simply may improve cold-start learning. This is a
new model within the existing cycle-consistency campaign, not a GAN reproduction. Their explicit
instruction to implement and execute authorizes this new round; the earlier hold on further M512
training does not block it. The old M512 job must not be restarted or altered.

Preserve `SAE_ref.md` label quarantine: unpaired train-clean-100 speech and unpaired text, existing
speech-only SSL encoder, no supervised or GAN-derived initialization. Gold phone sequences are for
evaluation and feasibility diagnostics only, never training, selection or parameter tuning. No WER
work is requested. Exact full-sum trigram is the preferred first implementation if its audited
topology and actual-batch cost permit it; higher-order sampled variants require a later release.

The cited paper is Liu et al., *Towards End-to-end Unsupervised Speech Recognition*,
https://arxiv.org/pdf/2204.02492. Verified paper facts and exact released-code settings will be
recorded in `reports/sae_blankfree_paper_2026-09-18.md`. Any departure, including removal of an
explicit SIL output, must be named; paper equivalence is not assumed from stride alone.

## Model question and gate

For independent categorical frame outputs a, define y=B(a) by replacing each identical-phone run
with one phone, without a blank symbol. Then q(y|x)=sum_{a:B(a)=y} product_t q(a_t|x) is normalized
over the image of B. The support excludes adjacent identical phones; this is a support restriction,
not by itself a normalization defect. The implementation must preserve this distinction and verify
the combined LM/reverse-model lattice rather than infer correctness from the frame distribution.

Keep the original G4a.3 scientific threshold (epoch-4 dev-other PER <0.50 and positive own-phi
speaker-matched derangement gap) as a reference criterion. A numerical or cost check is not evidence
of cold-ASR improvement. Full output-diversity diagnostics and literal examples are required even
when the threshold fails. This first bundled model change can assess viability; it cannot assign
any observed change separately to VAD, stride, blank removal or the alignment estimator.

Before compute, register exact VAD provenance/parameters, feature timing and utterance mapping,
phone/text inventory and LM treatment, reverse clock/duration support, repeat-aware forward sum,
initialization/schedule, representative actual-data checks and cumulative resource ceiling.

## Registered first model

Reference: completed local `FairseqW2vu2TrainJob.HOb2GgtYT7Bc`, resolved `output/train.log`.
Use the current training stack, not fairseq. The user explicitly prefers this. Reuse existing
`ConvRecognizer` with input 1024, valid-frame batch normalization initialized at scale 30,
dropout 0.1, residual 1024-to-1024 linear, one bias-free convolution, kernel 9, stride 3,
padding 4, dilation 1. These are verified reproduction settings, not newly tuned constants.

**Preprocessing:** the reproduction environment's rVADfast (module declares 0.0.3; installed
distribution metadata is 0.0.5), threshold 0.4, 25 ms window/10 ms shift, pairwise aggregation
to 50 Hz (ties retain speech), existing truncate/tail-pad-as-silence reconciliation. Recompute masks
from original continuous decoded audio; apply the identical mask to existing raw L15 features and
enc50 K500 reverse units, preserving original frame indices, original lengths and all utterance IDs.
The completed reproduction masks features AFTER full-waveform SSL extraction. This is the convention
adopted here; the paper's waveform-trimming-before-SSL description is a disclosed difference.
Its existing train/dev artifacts contain 28539/5567 items, 15427853/1612502 retained frames from
18088388/1888037 raw frames, without dropped utterances. The new joint mask must reconcile those
counts by utterance against the completed reproduction. Do not use its MFCC64 `.km` labels as enc50.
No new SSL forward is needed. Gold does not enter mask computation.

**Inventory:** 39 ARPAbet phones plus SIL (40 outputs, zero-based phone IDs), no CTC blank.
The paper and reproduction retain SIL at sentence edges and at word boundaries with probability
0.5. Keep that existing unpaired-text prior and the reverse SIL duration model. Silence removal
does not imply deleting the SIL vocabulary. Greedy inference collapses consecutive identical
predictions including SIL, then removes SIL for comparison with the existing phone references.

**Exact topology:** q has T=ceil(S/3) steps for S retained 50-Hz input/reverse frames. State
F_t(s,h) starts at F_0(0,BOS)=0. A prediction equal to last(h) consumes one recognizer step
with log q/tau only. A different prediction (or any from BOS) starts exactly one reverse segment,
advances the trigram state, consumes a legal duration d, and adds
[log q + beta log P3(k|h) + reverse_segment_score]/tau. This includes SIL: no same-token
new-segment exception. No EOS factor is added, matching the reference objective. The band is
|s−3t|≤25 retained frames; the final gather is s=S, offset S−3T+25, including all padding residues.
Keep d_min=2, phone D=25 and SIL D=50 on the retained 50-Hz reverse clock. Exact summation is
over this registered band/duration support; it is not the unrestricted q pushforward partition.

**Loss conventions:** cycle normalized by retained S, never the stride-3 T. Existing rate target
and weight retain their original-audio-duration denominator; save original frame counts and report
rates per original and retained second separately. Blank-free expected run-unigram counts are
q_0(k)+sum_{t>0} q_t(k)(1−q_{t−1}(k)); off-diagonal run-bigram counts are
sum_{t>0}q_{t−1}(j)q_t(k). Mask variable lengths. Reuse inherited text unigram targets;
zero the structurally unreachable diagonal of the inherited bigram target and renormalize.
This target projection is a topology-required objective change, not a learned parameter.

**Initialization and schedule:** cold shape-specific flat recognizer, independently random reverse
model, no old checkpoint transplant; inherited exact-P3 reference optimizer, batch/order, anneal
8→2 by subepoch 4, beta and regularizer weights. Concrete inherited values: tau by subepoch
(8, 5.04, 3.17, 2, 2, 2, 2, 2); rate rho=9.6619373279 Hz and lambda_rate=3; beta=1;
lambda_agg=0.1, EMA decay=0.99; batch 88000 padded frames/max_seqs 128; theta LR=1e-4,
phi LR=3e-3, Adam betas=(0.5,0.98), epsilon=1e-6, weight decay=0, gradient clipping=5;
partition_epoch=4. Exact P3 uses matmul reduction/checkpoint stride 32. Sources:
`config_sae_4a_s3b_rate_v1.py` rate/trigram settings, `emc_train_jobs.py` training defaults,
and `s3_jobs.py` anneal. Implementation must resolve these source basenames in its report.
Initially execute four subepochs; continue to eight only if the unchanged epoch-4 G4a.3 passes.
Checkpoint and inspect epochs 1 and 4 (and 8 only if released); no best-PER selection.

Store SIL-inclusive collapsed strings for reverse/derangement scoring. For phone evaluation,
remove SIL after collapse without a second collapse: A SIL A must remain A A. Original IDs and
gold remain unchanged. The new decoder must not treat label zero as a blank.

**Release checks:** independent tiny exhaustive path/segmentation enumeration must match logZ and
recognizer/segment gradients at tau 1/2/8, including repeated SIL, initial/final transitions,
stride-3 padding residues, band boundaries and variable lengths. Separately verify unrestricted
q pushforward mass sums to one; that does not establish the banded cycle partition is one. Old CTC
defaults must retain their tested behavior. Inspect all preprocessing IDs/lengths and legal support;
no silent removal, padding or fallback for invalid/empty VAD items is permitted.

Profile the inherited four actual random/stress batches after applying the joint mask, at tau 8
and 2. During the new loader inventory, add its earliest batch of maximum retained padded length
if its tag set is not already among those four. These are cold-weight temperature checks, not
trained-state timing guarantees. Inventory the new loader's actual number of updates N for eight
subepochs; release only if all measured complete
updates are finite, both parameter families receive finite nonzero gradients, and
N × max_measured_update_seconds × 1.25 ≤ 8 hours. The 1.25 allowance is inherited from the cost
protocol; this is an engineering proxy, not a promised runtime. Cap this screen at one GPU/30 min
and the training round at one GPU/8 h cumulative, with no automatic timeout restart. CPU-only VAD
preparation is capped at one 8-hour allocation. Numerical/profile passing is not an ASR result.

Design/source evidence: `reports/sae_blankfree_code_inventory_2026-09-18.md`,
`reports/sae_blankfree_paper_2026-09-18.md`, and
`reports/sae_blankfree_design_review_2026-09-18.md`.

## Implementation evidence

Exact topology implementation `5cc9c72` passes three focused blank-free CPU tests (independent
enumeration, posterior/gradient checks, mixed lengths and normalized pushforward/counts) and two
existing CTC identity/checkpoint tests. Independent source review finds no open issue. These establish
the tested numerical contracts, not runtime or recognition. Reports:
`reports/sae_blankfree_lattice_impl_2026-09-18.md`,
`reports/sae_blankfree_lattice_review_2026-09-18.md`.

Joint-mask preparation `7f09df6` has a two-utterance HDF fixture and mismatch failure check;
the actual corpus reconciliation is pending. Report:
`reports/sae_blankfree_data_impl_2026-09-18.md`. No new-model training result exists.

The model/train/profile/evaluation integration is committed as `323de02` and independently
source-reviewed. A tiny end-to-end CPU step has finite loss and nonzero gradients in both
models; actual-data runtime/support remain pending. Sources and exact settings:
`reports/sae_blankfree_model_impl_2026-09-18.md`,
`reports/sae_blankfree_model_review_2026-09-18.md`. Separate selectors are
`config/sae_4a_blankfree_prepare.run`, `config/sae_4a_blankfree_profile.run`, and
`config/sae_4a_blankfree.run`; the final selector is not released until the profile passes.

## Supervised 10 h separate initialization only

User explicitly requests this alongside the cold joint model and explicitly excludes subsequent
100 h adaptation. Fit theta and phi independently on the existing labeled 10 h; no cold parameter
or target may depend on these fits. Share only speech-only preprocessing/architecture resources.
This branch is a supervised reference, not cold unsupervised progress.

Fixed membership: `SeedGoldPhonesJob.zii9E9tvr51e/output/{seed_ids.json,seed_gold_phones.json}`,
2849 utterances; fixed 2821-train/28-held split from `CvHoldoutSplitJob.sD7U6CYs8ACM`.
These are the reference's 10 h/seed-42 selection and seed-0 holdout, not a new sample.
Theta inherits 24 complete passes, Adam LR5e-5, betas(0.5,0.98), epsilon1e-6, clipping5,
100000-frame/max512 batching and final-checkpoint evaluation; one allocation capped11.5h.
Phi inherits eight passes, Adam LR3e-3/default betas, clipping5, batch8, seed42 and final
checkpoint; one allocation capped4h. Both tasks must be nonresumable automatically. Source:
`reports/sae_blankfree_seed_inventory_2026-09-18.md`; these limits come from the existing
independent supervised fits, not from the cold training allowance.
Independent design review: `reports/sae_blankfree_seed_design_2026-09-18.md`.

Theta's proposed sequence loss is exact blank-free log Q(y|x), tau1, without an LM, reverse
factor or cycle band. Apply adjacent-run collapse to SIL-free reference phones **for training
targets only**. Keep all original evaluation references unchanged and report how many target
phones this removes. SIL remains a real output; do not repurpose it as a hidden CTC blank.
Phi retains the reference's original uncollapsed phone sequence with one SIL at either end,
and learns exact reverse marginal likelihood per retained 50-Hz frame, without the q-lattice band.
Its independently supervised target support need not equal theta's collapsed-string support.

Before either fit, a CPU support census (one allocation, at most30min) must cover all2849 masked
items and preserve the fixed split. Theta requires nonempty targets and U_collapsed≤ceil(S/3).
Phi requires 2U_phi≤S≤sum_i D(y_i), with the registered phone/SIL duration maxima. Record all
infeasible IDs/counts; no silent filtering, zero-infinity suppression or artificial padding.
If sequence support fails, do not launch the fits as currently specified. The next decision is
the supervised target/loss definition, based on that census; existing MFA-timing frame supervision
is a candidate to assess, not an automatic fallback. This does not block the cold joint branch.

Completion is both independent supervised fits and their final evaluations under the registered
protocol. Report theta's original-reference phone errors, output diversity and literal examples,
and phi's held-out likelihood/frame with its precise target convention. No best-PER selection,
supervised-to-cold initialization or speech-only adaptation follows from this branch.

Support source review clears the CPU census only:
`reports/sae_blankfree_seed_support_review_2026-09-18.md`. The support report and collapsed-target
HDF must both be complete, with `supported_all=true`, before the fit graph can be registered;
both fits retain a Sisyphus dependency on that completed census.
Implementation `db3f932` and independent full source review are complete; three exact-likelihood,
gradient and support tests pass. These do not establish actual-data feasibility or fit quality.
Evidence: `reports/sae_blankfree_seed_impl_2026-09-18.md`,
`reports/sae_blankfree_seed_review_2026-09-18.md`. After the terminal support census passes,
the fit/evaluation selector is `config/sae_4a_blankfree_seed.run`.

## Independent side task: diversity within sampled groups

Submission: `GenericExecJob.awnR46wSTGCf`, Slurm `1883777`, via
`config/sae_4a_sample_group_diversity.run`; output `output/sae/4a/m512_sample_group_diversity.json`.
Source review and result audit DONE. Launch/provenance:
`reports/sae_sample_group_relaunch_2026-09-18.md`. The replay imports immutable original M512
sources from `artifacts/m512_frozen_source/src` (recipe commit `e1352581`); all 15 pinned source
hashes and original input/config/checkpoint hashes are preserved. Independent source verification:
`reports/sae_sample_group_repair_review_2026-09-18.md`. Actual completed replay allocation is
19m19s; cumulative side allocation19m55s is within the original30min ceiling.

The latest matched M512 P3/P6 run samples complete paths in both arms; historical DP64 trigram
used a full sum. Check actual sampling groups, not only final greedy hypotheses. Reuse registered
real batches (global steps 173, 417, 146, 144), cold state and each current M512 arm's final epoch-8
checkpoint, M=512 and the unchanged 384+128 target/hot proposal mixture. No training updates.

Measure per utterance: unique joint alignments and collapsed phone strings, duplicate multiplicity,
largest unweighted group, normalized importance-weight maximum and ESS, ESS after string aggregation,
component-specific counts, and weighted/unweighted phone and length diversity. Preserve SIL-sensitive
training strings separately from any evaluation collapse. Predeclared descriptive concentration
scales are ESS<2 and maximum normalized weight>0.9, not scientific pass/fail thresholds. Report
distribution across utterances and cases; one extreme cannot establish prevalence or cause.

One GPU, at most 30 minutes, using the inherited profile resource envelope, no automatic repeat.
Script: `analysis/m512_sample_group_diversity.py`; implementation report:
`reports/sae_sample_group_impl_2026-09-18.md`. Source review precedes execution, independent audit
precedes a causal interpretation. This side task must not hold up a launch-ready main experiment.

### Sampled-group result (2026-09-18, audited)

All16 registered cases completed: four actual batches of124/103/113/71 items, 411 groups per
arm/state and1644 rows total. Cold uses tau8; final uses each arm's epoch8/step477 checkpoint at
tau2. Every group contains512 draws, with384 target-temperature and128 hot draws. Every group
has512 unique joint paths and512 unique recognizer paths. Means below weight utterance groups
equally, not batches; strings retain the sampler's token-start SIL convention.

| Arm / state | Unique strings, mean | Path ESS, mean | String-aggregated ESS, mean | Groups ESS<2 | Groups max weight>0.9 |
|---|---:|---:|---:|---:|---:|
| P3 cold | 512.00 | 395.00 | 395.00 | 0/411 | 0/411 |
| P3 final | 510.06 | 384.14 | 379.35 | 0/411 | 0/411 |
| P6 cold | 512.00 | 3.74 | 3.74 | 106/411 | 35/411 |
| P6 final | 493.97 | 3.50 | 3.45 | 165/411 | 76/411 |

ESS=1/sum(normalized_importance_weight²). P6 final median path ESS is2.39; its string count
is324–512, median506. Thus P6 has strong finite-sample importance-weight concentration despite
distinct alignments and generally many distinct training strings. In40.1% of its final groups,
ESS is below2; in18.5%, one path carries over90% of normalized weight. Weight concentration is
already present in the cold replay. These descriptive scales do not establish true-posterior
coverage or prove a cause of the final greedy-output pattern.

The matched M512 arms both sample paths. Their final checkpoints differ; cold-to-final also
changes temperature. Cold deterministic seeding supports an order-dependent concentration
observation, but exact cold state hashes/draw keys were not saved for a strict same-path causal
contrast. Historical exact-P3 DP64 is not an ESS comparator. No P6 accuracy or intrinsic-prior
quality conclusion follows. Keep the already registered new-model exact-P3 experiment unchanged.

Ground truth: `output/sae/4a/m512_sample_group_diversity.json` (15191900 bytes); completion and
all case summaries: `reports/sae_sample_group_completion_2026-09-18.md`; independent audit:
`reports/sae_sample_group_result_audit_2026-09-18.md`. Side task complete; no additional run.
