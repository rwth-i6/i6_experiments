# SAE §4b — Sparse acoustic codes from frozen w2v2 features

## State

REGISTERED, NO EXECUTION. The user requests a separate phase with a preliminary literature-based
description (2026-09-17) and explicitly withholds execution. No model, codebook, extraction or training
job is launched for this phase. Active higher-context rescoring and the categorical-content experiment
remain in `SAE_4A.md`.

Objective: determine whether a sparse autoencoder of the existing frozen w2v2 features yields more
useful acoustic codes than direct clustering of those same features, and whether such codes can improve
cold-start cycle learning. A useful private code counts as intermediate progress; named-phone recovery
requires separate evidence. Next action is design refinement only. Before any execution, specify the
architecture/sparsity, acoustic train/held-out split, code-rate controls, numerical gates and budget,
then obtain the user's execution authorization. Shared constraints are in `SAE_ref.md`.

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

## Preliminary experimental sequence

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

## Preliminary gate and unresolved design

No scientific acceptance threshold or compute budget is approved yet; execution is explicitly held.
The future representation gate must measure acoustic-code usefulness against direct L15 clustering,
not merely sparse reconstruction or a phone map fitted to gold. A private acoustic code can satisfy an
intermediate objective without recovering phone names. A claim of improved unsupervised ASR additionally
requires the existing label-free selection protocol and paired downstream PER/acoustic-dependence
evaluation; the original cold-start gate in §4a is not weakened by this phase.

Before execution, settle whether the sparse code is temporally stable enough to quantize, how code rate
is controlled across baselines, which nuisance diagnostics remain label-free, and what improvement and
budget justify a cycle target swap. This document records a hypothesis and comparison, not a result.
