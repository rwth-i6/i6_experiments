# PLAN §3a sub-plan — psi_align, the monotonic-alignment reconstruction scorer

Planner-owned design document (normative), 2026-08-05. PLAN.md §3a holds the decision and the
gate summary and points here; this file holds the full design. The implementer's execution log is
`SAE_3A.md` (open it on first build; this file records *what to build and why*, that one records
*what happened*). Renamed from `SAE_3A_SCORER_DESIGN.md` the same day it was written (user call:
PLAN.md stays lean, sub-plans carry the detail).

Fired by the user 2026-08-05 ("do a research on what exact neural architecture to use for an ideal
AR, read TTS and TTE papers") — this is the escalation §3a pre-registered on 2026-07-18
("an explicit monotonic-alignment scorer, duration/MAS-style").

---

## 1. Requirements — each anchored to a measurement

- **R1 Rank near-misses.** The reward's only job is ordering G=12 samples of one utterance
  (`SAE_2S.md` approach 15: within-group WER spread 0.044 exists at T=0.7; incumbent gap_true
  +0.0124 nats).
- **R2 No unit-history channel.** mask=none: 27.2 % top-1 with >99 % of it from unit continuity
  (`SAE_2S.md` approach 10, conclusions 17-19) — a smoothness detector, not a reconstruction scorer. The channel must be
  structurally absent, not masked.
- **R3 Alignment-aware.** The units are structured (27.2 % predictable from history) and the
  encoder states carry phone identity (linear probe 0.145 PER, SAE_0), yet text+position buys only
  +0.79 top-1 points — the model knows *what* units the text makes, not *where* they land. RoPE
  relative position cannot express durations.
- **R4 Exact, deterministic, comparable likelihood.** Score = properly normalized p(u | text(z))
  for fixed u across candidate z; no sampling noise, no unnormalized energies. Marginalize the
  alignment (full-sum), never max it: near-miss candidates differ exactly where the 1-best path is
  ambiguous, and Viterbi throws away the distinguishing mass.
- **R5 Cheap to retrain.** The target stream changes per B0 arm (§3b) and per outer loop; the
  incumbent costs a 2.2 h 4-GPU SFT per stream. Minutes-scale retraining makes the scorer×target
  matrix affordable.
- **R6 Price length natively.** The loop's documented failure mode is unbounded generation; the
  hinge is a patch with measured slack (hyp/ref 1.61 on short dev utts).

## 2. Literature — four families, three eliminated by our own measurements

**Family 1: teacher-forced attention AR** (Tacotron2; TTE = Hori et al. arXiv:1811.01690;
token-LM TTS like VALL-E). The TTE model specifically: Tacotron2 with location-aware attention,
character input, predicting **ASR encoder states** instead of spectrogram (deliberately dropping
speaker/para-linguistics — same rationale as our unit/state target), L1+MSE+stop-BCE, used as a
cycle-consistency loss over 5 sampled ASR hypotheses via REINFORCE — i.e. *their reward is our
reward*: group-sampled reconstruction likelihood. Their 100 h-seed + 360 h-audio result
(WER −14.7 % rel) is the external validation of the §3d regime argument, and their seed loss never
leaving the objective is §3c. But the family conditions the decoder on its own acoustic past — the
R2 channel. We have run this family twice: mask=none (99 % history), and the candidate-3
prenet-history continuous regressor (eta < 0, measured OUT 2026-07-31). TTE contributes the target
choice and the replay mechanism, not the decoder architecture.

**Family 2: NAR + point durations** (FastSpeech 1/2, FastPitch). Kills the history channel — the
TTS lesson §3a already cites — but durations are a point estimate, and the family needs an external
aligner to train at all (FastSpeech distilled Tacotron attention; FastSpeech 2 used MFA). As a
*scorer* this is p10 plus a guessed alignment: one duration error shifts every downstream frame and
the likelihood collapses for reasons unrelated to transcript quality. No marginalization → fails
R4. The incumbent p10 is this family's degenerate member (uniform/no alignment): +0.79 points.

**Family 3: alignment-marginalized exact likelihood — the pick.**
- *Glow-TTS* (arXiv:2005.11129): monotonic alignment by MAS = Viterbi (max, not sum), exact flow
  likelihood given the alignment, duration predictor fit to the extracted durations. Establishes
  that learned monotonic alignment needs no external aligner.
- *RAD-TTS aligner, "One TTS Alignment To Rule Them All"* (arXiv:2108.10447): the general recipe —
  soft alignment from pairwise distances of small conv encoders, **forward-sum over all monotonic
  paths via the standard CTC DP**, a beta-binomial diagonal prior early in training, Viterbi
  hardening + bin loss only to extract durations. Improved every TTS architecture it was attached
  to, AR and NAR alike.
- *UnitY2 aligner* (Seamless, arXiv:2312.05187): **the RAD-TTS aligner with mel replaced by
  discrete k-means speech units** (10k XLS-R units, character-level text), trained jointly with a
  NAR text-to-unit model. The closest published object to ours: text-to-kmeans-units monotonic
  alignment learning works at industrial scale. We differ in *use*: normalized generative scorer,
  not duration teacher.
- *Neural-HMM TTS / OverFlow* (arXiv:2108.13320, 2211.06892): left-right no-skip HMM, neural
  emissions AND neural transition probabilities, trained by the exact forward algorithm. Their
  emissions are autoregressive on previous acoustic frames — the one part we delete (R2). Deleting
  it costs them naturalness; costs us nothing (we score, we never synthesize).
- *Transducer with stateless prediction network* (Ghodsi et al., ICASSP 2020): exact DP marginal,
  monotonic, label history cut to the last symbol at near-parity with full history. The
  bounded-leak fallback (section 8).

**Family 4: implicit/diffusion likelihoods** (flow matching etc.): no cheap exact likelihood —
fails R4. Not considered further.

All of family 3 is one mathematical object: a latent monotonic alignment lattice (the left-right
HMM state structure) with neural emission scores, handled by either Viterbi (max) or the forward
algorithm (sum). CTC is the same lattice with a specific topology. psi_align = RAD-TTS's
forward-sum + Neural-HMM's learned transitions + categorical unit emissions, minus every
autoregressive part.

## 3. Why the LLM is the wrong body for this organ

The AR's task is phonetic-durational. Everything Qwen brings is invisible to an audio-derived
target (spelling, semantics — the homophone channel, `SAE_2S.md` approach 15, conclusions 31-33), banned (own-side
sequence modeling, R2), or already from-scratch anyway (the 502-row sentinel overlay is the only
real unit head; the 152k text logits leak 0.015 nats of mass). The measured ceiling agrees:
gold-text-trained 0.409 nats text-explained ≈ pseudo-text AR_G 0.419 — saturated regardless of
training text (`SAE_2S.md` approach 10, conclusions 17-19). The AV half keeps its LLM: AV^G beating its CTC teacher by
~4 WER *is* the language prior working.

## 4. psi_align v1 — normative specification

A conditional neural HMM over the G2P phone string: "RAD-TTS aligner turned into a normalized
generative scorer", equally "neural-HMM TTS minus acoustic autoregression".

### 4.1 Generative model

- **State graph** from phones(z): closed-lexicon G2P (the §3e reward's lookup, one canonical
  pronunciation; stress markers stripped in v1 → ~45-symbol inventory incl. silence and
  word-boundary marks). Left-to-right phone states; **optional silence states** at word boundaries
  and utterance edges (enterable or skippable — standard Kaldi-style topology); a final absorbing
  end state. The text-side symbol inventory is itself an axis (§5c — characters are the funded
  lexicon-free arm; phones are the v1 reference); everything below is inventory-agnostic.
- **Transitions**: per state j, a 3-way softmax over outgoing arcs {self-loop, advance, skip} from
  the state vector s_j (skip = advance-by-2, the phone-deletion arc; silence arcs from the graph).
  Self-loops give geometric durations (v1); explicit HSMM duration heads are the v1.1 escalation
  (section 8).
- **Emissions**: per state j, log-softmax over the unit inventory only — K=500, nothing else. No
  152k-vocab denominator; the leak channel of `SAE_2S.md` approach 10 (conclusions 17-19) does not exist here (its
  measured 6.8 % mass-term signal is an artifact of the incumbent's shared vocab, not a feature to
  reproduce — a native head expresses "does the transcript explain the audio" inside the 500-way
  distribution).
- **Likelihood**: exact forward algorithm over the T × U lattice in log space,
  p_psi(u_1..T, reach-end-at-T | phones(z)). Because outgoing-arc probabilities and emissions are
  both normalized per state, this is a proper joint distribution over (length, unit sequence)
  given the text, so length is taxed **inside the likelihood** (R6): a too-long hypothesis pays
  the skip-arc price, a too-short one pays improbable self-loop runs. Stated honestly (verifier,
  2026-08-05): this is a *learned tax, not a wall* — if training drives the skip rate up to absorb
  the real 12.5 Hz squeeze, an inserted ~4-phone word costs only ~6–9 nats (~0.04–0.06 nats/frame
  at T≈160). The two jobs of the skip arc (absorb the genuine squeeze; price insertions) are
  anti-synergistic, which is why the hinge is NOT retired on faith — §7 step 3. Cost O(T·U·3) ≈ tens of
  thousands of lattice cells per utterance — trivial, vectorized over the batch.

### 4.2 Network

| component | spec | params |
|---|---|---|
| phone embedding | ~45 symbols × d=384 | ~17 k |
| text encoder | 6-layer pre-norm bidirectional transformer, d=384, 6 heads, FFN 1536, dropout 0.1, RoPE or relative bias | ~10.6 M |
| emission head | linear 384 → 500, log-softmax | ~0.19 M |
| arc head | linear 384 → 3 softmax (+ silence-arc params) | ~1 k |
| **total** | | **~11 M, all trainable, from scratch** |

Bidirectionality is free and correct (the text is fully observed) and gives triphone-and-beyond
context to every state's emissions without any history channel.

### 4.3 Reward definition and CI status

R_recon(z) = (1/T) · log p_psi(u_1..T, T | phones(z)) — a drop-in for the §3e recon slot; T is
fixed within a GRPO group so the 1/T is a shared constant. **In-lexicon** homophone spellings
collapse to one phone string → byte-identical recon reward by construction (today those groups
get pure-noise advantage differences; after collapse, zero — strictly better under
group-normalized advantages). Scope caveat (verifier, 2026-08-05): the *measured* within-group
spelling variance (`SAE_2S.md` approach 15, conclusions 31-33 — cosett/kosetz/so'st, literal underscores) is
largely OUT-of-lexicon, where collapse cannot fire; those candidates route through the single
pre-registered OOV convention (§9 risk 4) instead. The de-noising is real but narrower than "the
§5 channel is deleted". Ranking spellings is the LM/lexicon side's job and no acoustic scorer's.

CI status, stated precisely: emissions and transitions are functions of (text, state) only —
there is **no parameterized dependence on u_<t** (stronger than p10's masking: nothing to mask).
The *score's* predictive factorization p(u_t | u_<t, z) does retain a bounded history channel
through the forward state posterior (sharp emissions + self-loops can reproduce repeated-unit
continuity through any text that latches the current segment); it is gated by the
text-conditioned state bottleneck and must be **priced by the same shuffled-text contrast this
design demands of fallback B** — symmetry the verifier correctly required. Hinge, KL anchor,
lambda_4 OOV term: unchanged at swap time.

### 4.4 Feasibility precondition (P0 — before building anything)

At 12.5 Hz the stream runs roughly 1.3 frames/phone (158 units vs an *estimated* ~110–120 phones
on a ~12.7 s utterance — the estimate is exactly what P0 replaces with a measurement). Monotonic
alignment without deletions needs T ≥ U. One CPU pass over the seed bed and tc100, reporting the
verifier-hardened statistic set (the raw median alone is the wrong gate: silence inflates T, and
fast *regions* break locally while the utterance median looks fine): (a) the T/U_phones
distribution; (b) the fraction of utterances with T < U_phones; (c) a silence-corrected
frames/phone estimate (speech-only T via the existing rVAD); (d) the phone-rate tail — p90 of
U_phones/duration against the 12.5 Hz frame rate. Skip arcs absorb a small tail; **stop and
escalate if the silence-corrected median is squeezed (≲ 1.1) or the T < U fraction is non-trivial
(> ~10 %)** — syllable-level states, or the 25 Hz assign of the same codebook (that is a §3b
*target* question and goes to the planner, not a silent scorer-side change).
**Measured and ruled (2026-08-05, `SAE_3A.md` P0)**: the phone arm straddles the line —
silence-corrected median 1.091 (seed) / 1.100 (tc100) vs the 1.1 bar, while every direct
feasibility statistic is comfortable (frac T<U 2.7–3.3 %, forced skips ≈ 0). Planner ruling: a
knife-edge straddle with comfortable direct feasibility reads **proceed, with the words-arm
silence hedge**; the squeeze verdict belongs to the G1/G3 read and the M5 (25 Hz) trigger, not
to this precondition. Noted honestly: on a speech-only basis ~23 % of utterances have fewer
speech frames than phones — the skip/insertion monitors are the eyes on that. P0 also reports
the same statistic set for **characters** (T/U_chars — needs no lexicon, just letter counts; the §5c
arm is expected ~1.15–1.25 chars/phone, i.e. a tighter squeeze, and gets its own go/stop read)
and measures the rollout-side OOV rate (section 9, risk 4).

### 4.5 Training recipe

- **Objective**: NLL of the forward likelihood (full-sum), flat start. For the first ~3–5 epochs
  add the beta-binomial diagonal alignment prior of arXiv:2108.10447 to the emission scores,
  annealed to zero — the known fix for degenerate early alignments. No Viterbi hardening in the
  training path (full-sum IS the product; hardening is only a diagnostic).
- **Numerics**: DP strictly in fp32 log-space (logsumexp), whatever autocast wraps the encoder —
  the `SAE_2S.md` approach 23 bf16 lesson, applied in advance.
- **Optimization**: AdamW, lr 1e-3, wd 0.01, ~500-step warmup then cosine; batch ~32 utterances
  bucketed by T·U; ~30 epochs over the 2849 seed pairs ≈ minutes per epoch on one GH200.
- **Checkpoint (planner call 2026-08-05, replaces blind `checkpoint_last` for THIS model; user
  may veto)**: an ~11 M from-scratch model on 2849 sentences can overfit inside the epoch budget,
  and an overfit pin would fire the "target wall" branch on a training artifact. The fix stays
  inside the label rules: hold out a **seed-internal 5 % split** (these labels are already
  sanctioned training input in this quarantined arm; evaluation gold is untouched), stop at
  held-out-NLL plateau, pin that checkpoint, criterion pre-registered here. This is selection on
  train-side sanctioned data, not on evaluation gold — the thing the project convention actually
  forbids. Other arms keep `checkpoint_last` unchanged.
- **Monitors**: train/dev NLL; per-utterance alignment entropy (must fall); mean self-loop prob;
  skip-arc rate; silence-state frame occupancy; Viterbi duration histogram vs the P0 frames/phone
  distribution.
- **Data**: the sanctioned 10 h seed pairs (same source dir as the incumbent AR SFT), incumbent
  k500 unit stream (same lineage the incumbent scores — apples-to-apples), G2P via the LibriSpeech
  lexicon (training-side OOV ≈ 0). Dev = the same 5000-utterance subset and derangement the FER
  job used, so every gate number lands on the incumbent's own instrument. A pseudo-text twin
  (G-track init) trains later iff §3d resumes.

### 4.6 Free diagnostics

Viterbi path → aligned frame accuracy (the FER-instrument analogue), per-phone duration stats,
alignment entropy; the true-vs-shuffled usage gate applies verbatim; per-state emission entropy
maps which phones the units resolve.

## 5. Execution plan (all jobs sisyphus; model-forward = GPU per standing rule)

| phase | job(s) | compute | artifact |
|---|---|---|---|
| P0 feasibility | `PhoneStatsJob` (CPU): T/U_phones AND T/U_chars distributions on seed bed + tc100; OOV rate over the existing n=512 `rollouts.jsonl` texts | minutes, CPU | stats.txt → go / escalate (per inventory) |
| P1 build + G0 | `speech_llm/sae/psi_align.py` + tests: forward-sum == brute-force path enumeration on toy lattices (T,U ≤ 6); per-state normalization sums to 1; batch-size invariance; fp32-under-autocast; prior anneal | login-node pytest | green tests |
| P2 train | `PsiAlignTrainJob`, seed pairs per §4.5 | ≤ 1 GPU-h | ckpt + monitors |
| P3 gate G1 (+ G2 diagnostic) | `PsiAlignInfoGateJob`: dev-5000 true + length-matched deranged pairings → ce_loo usage gate and text-explained (§6 convention as amended), NLL/frame secondary, ce_emis reported-only, length-only-null control, Viterbi + leave-one-out frame accuracy | ~0.5 GPU-h | gate table |
| P4 gate G3 — **runs unconditionally, not gated behind P3** | `PsiAlignRerankJob`: re-score the **existing** n=512 × G=12 rollout dumps (theta_0 bed primary; AR_G bed secondary) — no new sampling; UNK-primary OOV convention; gap_true, spearman, audio margin, eta-at-G for the record | ~0.5–1 GPU-h | PLAN.md §2.5(d) table |

Total < 3 GPU-h to the decisive read. B0 synergy: retraining psi_align per candidate stream
({k100, brown_k100, perutt_k500}) is minutes each — those slot into the matrix below as
emission-vocabulary variants of cell M1.

## 5b. The scorer×target matrix — the axes the new family reopens (user, 2026-08-05)

The scorer changed, so two target-side closures made under the p10 LLM-AR are
scorer-family-conditional, and the user has reopened both for psi_align: **continuous vs
discrete** (candidate-3's 2026-07-31 closure covered *no-alignment* regressors only) and **raw
vs subsampled+adapted substrate** (the "post-adapter beats pre-adapter, eta 0.145 vs 0.225"
reading was measured with the old scorer AND through the since-withdrawn eta instrument).
"w2v2 encoder state before vs after SFT" — **CORRECTED 2026-08-05 (verifier; the earlier
"resolved" text here was factually wrong, caught by the implementer and by the user)**: the
AV's w2v2 encoder transformer **IS fine-tuned during SFT** (`config_sae_2s_av_sft_w2v2_v1.py`
passes `encoder_trainable=True`; only the conv front end stays frozen; theta_0 `OLzy9Q2oC3mU`
is hash-pinned as the *trainable*-policy checkpoint). The frozen-encoder sites are the GRPO
**loop** and the unit-dump encoder; the wrong claim conflated those sites with SFT (the
BEST-RQ-era SFT default was also frozen — two eras, two phases, one flag name). So the
before/after-SFT encoder contrast exists, and M1 differs from raw in **two** ways at once:
a fine-tuned encoder AND a learned concat-×4 adapter instead of a mean pool. M6 is
**REINSTATED** as the midpoint that decomposes them (the user's drop was conditioned on
"encoder frozen", which is false; their prior stance — M6 makes sense if the encoder trains —
stands).

| cell | emissions | substrate (all deterministic transforms of audio) | isolates | independence status | cost |
|---|---|---|---|---|---|
| **M1** (critical path, §5) | categorical k500 | post-adapter AV states, 12.5 Hz | the scorer family itself | seed-arm (adapter is 10 h-seed-trained) | in §5 |
| **M2 = psi_align-C** | Gaussian; label-free PCA to d≈64–128; fixed/floored diagonal variance | same as M1 | continuous vs discrete, substrate held fixed | same as M1 | ~1 GPU-h |
| **M3** | categorical k500, codebook refit | RAW frozen w2v2-lv60 L15, ×4-pooled to 12.5 Hz | adapted vs raw, rate and K held fixed | fully label-free | refit + ~1 GPU-h |
| **M4** (contingent — user descope 2026-08-05) | Gaussian (as M2) | RAW frozen L15, subsampled | the fully label-free continuous cell — the TTE-style target on our SSL encoder | fully label-free | fires only if BOTH single-axis contrasts (M2, M3) move |
| **M5** (contingent) | best of M1–M4 | raw L15 at 25 Hz (×2 pool) | frame rate — dissolves the G0 squeeze | fully label-free | fires on the G0 escalation or planner call |
| **M6** (REINSTATED 2026-08-05 — the drop premise "encoder frozen during SFT" was false) | as the substrate axis runs (categorical k500 refit, or the M2 winner) | theta_0's **SFT-fine-tuned** encoder L15, mean-pooled ×4 to 12.5 Hz — encoder tap, no adapter | the decomposition midpoint: **M1→M6 prices the adapter, M6→M3 prices the SFT fine-tuning** (without it the M1-vs-M3 contrast conflates both) | seed-arm (encoder is 10 h-seed-SFT-trained) | one GPU dump + re-quantize (`AvStatesJob(tap="encoder")` + `PoolStatesJob(factor=4)` exist) |

**Comparability rule (extends the G1 convention).** Absolute nats compare only *within* a
representation class — a categorical CE and a Gaussian log-density are not the same currency —
so across cells the comparators are (i) each cell's own usage gate (true vs length-matched
shuffled text, same density), and (ii) **G3's rank statistics and audio margin, which are fully
representation-agnostic**: same rollouts, same WERs, only the scorer swaps. The per-cell G3
re-rank is the matrix's scoreboard. The Gaussian cells carry the §8 guard (variance floor, so
no candidate is ever ranked by variance collapse).

**Order (user descope, 2026-08-05: "keep M1–M3 at the beginning is enough")**: the funded
initial batch is **M1 → M2 + M3** — the family test plus the two clean single-axis contrasts.
M4 is contingent (fires only if both M2 and M3 move — the combined cell is only informative once
each axis matters alone), M5 stays on its G0 trigger, and **M6 rides the substrate-axis batch**
(user order 2026-08-05: M2 first, then the substrate axis; M6 is that axis's midpoint, not an
extra batch). The *other* historical M6 reading — states from the §1d student's GAN-shaped
weights — stays unfunded pending a user independence ruling ([[gan-not-a-teacher]]); reinstated
M6 uses only theta_0, which is seed-arm-clean. Every funded cell is minutes-to-1-GPU-h; the
initial batch stays under ~3 GPU-h on top of §5.

**Emission capacity — "is ~11 M with a Gaussian head strong enough?" (user question,
2026-08-05).** Split the worry in two. Parameter count is not the risk: the published aligners
this design descends from are ~1 M-scale objects; the information G1 asks for is half a nat per
frame; and the 1.7B incumbent already demonstrated that scale is not the binding factor in this
task family (its gold-text-trained ceiling, 0.409 nats, equals its pseudo-text one, 0.419). The
underfit signature — train NLL high with train ≈ dev — is on the §4.5 monitor list, and risk 6's
escalation (d=512 / 8 layers, ~25 M) is minutes-scale. The **Gaussian emission family is the
honest weak point**, and the user is right to poke it: a single diagonal Gaussian per state is
exactly the density that classical ASR had to rescue with mixtures and context-dependent states,
because per-phone acoustics are multimodal (speaker, channel, coarticulation). Three things keep
the funded cells M2/M4 meaningful anyway. (i) The transformer state already supplies
context-dependent means — the CD-states half of the classical fix comes free. (ii) The scorer is
a *within-group comparator on shared audio*: the G=12 candidates score the same frames, so
utterance-level nuisance modes (speaker, channel) are common-mode and largely cancel under
group-normalized advantages; what survives is per-frame phonetic mismatch, which is the signal.
With the fixed/floored variance the log-density reduces to a scaled negative MSE — precisely
TTE's objective (L1+MSE on encoder states), which demonstrably ranked ASR hypotheses well enough
to drive REINFORCE (WER −14.7 % rel). A weak *density* can be a good *comparator*; calibration
is not what G3 measures. (iii) A pre-registered escalation ladder if M2/M4 fail their own usage
gate while M1 passes: first a mixture-of-K diagonal Gaussians head (K = 4–8, ~0.2 M extra
params — the classical fix), then low-rank-plus-diagonal covariance, then a conditional
normalizing-flow emission — Glow-TTS's own answer to exactly this weakness (its flow exists to
map frames into a space where the diagonal Gaussian holds, and it preserves the exact likelihood
R4 requires). If the continuous axis needs flow-grade machinery to compete at all, that is
itself the matrix's answer: the categorical cells win on pragmatics.

## 5c. The text-side axis — lexicon-free inventories (user, 2026-08-05: "I still really like
lexicon free approach if possible")

The human lexicon enters psi_align at exactly **one** point: the state-graph construction
phones(z) via G2P. The lattice, transitions, emission heads, training recipe and every gate are
symbol-inventory-agnostic — swapping the text side is a one-line change to graph construction
plus a minutes-scale retrain. Three inventories, two funded:

| text side | human lexicon? | expected T/U (P0 measures) | status |
|---|---|---|---|
| phones via G2P (§4.1) | yes (LibriSpeech lexicon) | ~1.3 frames/phone silence-corrected | reference arm — prices what the lexicon buys |
| **characters** | **none** | ~1.05–1.15 frames/char (~1.15–1.25 chars/phone) | **funded: M1-char** — same substrate and emissions as M1, only the graph changes |
| text-BPE, ~1–2 k merges learned on the sanctioned *unpaired* LM text corpus (machine-derived, no human pronunciation input) | none | ~3–4 frames/token | contingent — fires if P0-char stops on the squeeze, or as squeeze relief alongside/instead of M5 |

**Why characters should work at all**: UnitY2's discrete-unit aligner — the closest published
object to psi_align (§2) — is *character-level*, aligning letters to k-means units at industrial
scale; TTE itself took character input. The bidirectional text encoder simply becomes an implicit
G2P learned from the seed pairs: silent letters ride the skip arc, digraphs spread two states over
one phone's frames. Two consequences to keep in view:

- **The squeeze tightens.** At ~1.2 chars/phone the silence-corrected frames-per-symbol drops
  toward 1.05–1.15 — more skip traffic than the phone graph. P0 measures T/U_chars (no lexicon
  needed) and applies the same per-inventory stop rule (§4.4). If chars stop but phones pass,
  BPE is the lexicon-free escape.
- **OOV dissolves — and imports a channel.** Every rollout spelling gets a real state graph (the
  UNK convention of risk 4 is phone-arm-only), so the cosett/kosetz spelling variants the phone
  arm collapses or UNKs are scored directly. The flip side: the learned spelling-to-sound map
  carries an orthographic-plausibility prior — partially an LM-side channel inside the "audio"
  scorer. The G3 audio-margin bar prices exactly this; and because scoring OOV groups at all
  flatters the char arm on the full set, the phone-vs-char G3 comparison is **also reported on
  the OOV-free subset** (pre-registered here, so nobody picks the friendlier number later).

**BPE mechanics, pre-registered for the contingent arm**: one state per token is too coarse (a
3-phone token emits different units across its span), so each token expands to S left-to-right
sub-states with per-sub-state emissions (S = 3 fixed, or proportional to token length in
characters); token embeddings are composed from their characters by a small char-CNN so rare
types share statistics — 2849 utterances cannot train ~2 k independent token embeddings. T/U ≈
3–4 dissolves the G0 squeeze entirely, which makes text-BPE the *text-side* alternative to the
25 Hz *target-side* escalation (M5) — cheaper, because it changes no target stream. This is
text-side BPE only; target-side unit-BPE stays on §3b's ledger under the planner-fallback rule.

**Decision rule**: M1-char runs right after M1 with the identical G1/G3 instruments (≤ 1 GPU-h).
If it matches M1-phone within noise on both gates, **lexicon-free becomes the default text side
for every later cell** — the user's preference honored at measured-zero cost. If it clearly
loses, the gap is the measured price of the lexicon, and whether to pay it is the user's call.

**OUTCOME (2026-08-05, `SAE_3A.md` P0): the char arm is MEASURED OUT at P0** — 0.855 unit frames
per character (the table above estimated ~1.05–1.15; wrong because word separators are real
emitting states in the char graph, verified consistent with the actual char model), 82–84 % of
utterances with T < U_chars, ~17 % forced skips. That is far past the stop rule and M1-char never
trains. **The text-BPE contingency trigger has therefore FIRED**; planner ruling: the BPE arm is
built **only after the phone arm's G3 lands and shows the family has ranking signal** — BPE
changes the text side, not the scorer class, so it cannot rescue a family that is flat, and at
T/U ≈ 3–4 it is the lexicon-free escape worth wiring the moment the family works.

## 6. Gates (pre-registered) — what each one asks, and what happens either way

**G0 — is the model built right, and is the alignment problem even solvable on this stream?**
Two cheap checks before any training. Correctness first: the forward-algorithm code has to
reproduce a brute-force enumeration of all monotonic paths on tiny toy lattices, and its
per-state probabilities have to sum to one — if the DP is wrong, every later number is noise.
Then feasibility: at 12.5 Hz there is only about one unit frame per phone, and a monotonic
aligner needs at least one frame for every phone it doesn't skip. One CPU pass over the bed
(P0, §4.4) measures frames-per-phone with silence removed, the fraction of utterances that have
fewer frames than phones, and the fast-speech tail. If the silence-corrected median is already
squeezed (≲ 1.1) or more than ~10 % of utterances have fewer frames than phones, we stop and take
the question to the planner — the fix would be syllable states or a 25 Hz unit stream, which is a
target (§3b) decision, not something to patch quietly inside the scorer.

**G1 — does the new scorer extract more text information from the units than the old family
ever did?** Train psi_align on the same 10 h seed pairs the incumbent AR used, then ask it the
same two questions every AR has been asked: how much better does it predict the units given the
*true* transcript rather than a random one (the usage gate), and how much of the stream's entropy
does the true transcript explain (text-explained)? One correction the verifier forced: psi_align's
raw likelihood contains alignment and length terms the incumbent's masked CE simply doesn't have,
so raw numbers are not comparable — the usage gate would flatter psi_align (a random transcript
also mismatches in *length*, which the HMM punishes structurally) while text-explained would
short-change it (the transition terms tax it against the unigram-entropy baseline). The
convention was amended once at build time (implementer finding, verifier-adopted 2026-08-05):
the originally mandated emission-component statistic (ce_emis, emission CE under the alignment
posterior) has the very defect this section disqualifies G2 for — the posterior is inferred from
the frame being scored, so sharpening the emissions drives it toward zero without predicting
anything. The gated statistic is therefore **ce_loo**, the CE of p(u_t | u_-t, text) with frame
t's own emission removed from the posterior — a genuine predictive likelihood — with the random
transcripts drawn **length-matched** and a length-only null model reported as a control (the
deltas also cancel most of ce_loo's bounded unit-context channel, which is present in both
pairings). NLL/frame is the secondary read — verified legitimately comparable, since the
incumbent scores EOU and so also prices termination — and ce_emis stays in the table,
reported-only, its inflation stated beside it. The bar: beat **AR_G, the best measured member of
the old family — 0.564 usage-gate / 0.419 text-explained nats** (a "family wins" headline needs ≥ 2× the incumbent: 0.67 / 0.53). Passing
means alignment unlocked information the LLM-AR could not see. Failing on k500 triggers a cheap
re-run across the other B0 streams (k100, brown, per-utt — minutes each, §5), the reopened
matrix cells (§5b — the continuous and raw-substrate axes the user reopened 2026-08-05), and the
text-side arm (§5c): flat across the whole matrix, *together with a flat G3*, is the program's
answer that the wall is the target, not the scorer — that escalates to the user (unit-BPE,
bottleneck rebuild).

**G2 — the intuitive number, deliberately NOT a gate.** "Given the true text, how often is the
aligned state's most likely unit the actual unit?" is the easiest statistic to read, and we
report it — but it cannot be a bar. The Viterbi path is inferred from the very units being
predicted, so the statistic gets to pick each state using the answer's neighborhood, and a
comparison against the incumbent's masked **+0.79 points** (which never sees any unit) would
flatter psi_align by construction. It is reported in two flavors (Viterbi and leave-one-out
state-posterior), used to understand the model, never to fund it.

**G3 — does any of that turn into ranking the loop's actual candidates? This is the gate that
decides funding, and it runs no matter how G1 went.** Take the existing rollout dumps — 512
utterances × 12 samples that theta_0 actually generated at T=0.7, no new sampling — and score
every candidate with psi_align (OOV words handled by the single pre-registered UNK convention,
§9 risk 4, so nobody picks the friendlier number afterwards). Three bars, all at once:
(1) it must prefer the true transcript to its average rollout at least **twice as strongly as
the incumbent** (gap_true ≥ 2 × 0.0124 nats); (2) its ordering must correlate with quality at
least as well (**spearman > 0.170**); and (3) — the audio-free-null lesson of
`SAE_2S.md` approach 16 — it must demonstrably beat a ranker that never hears the audio at all.
That null (lm_prior − hinge) already reaches eta +0.115 on this bed purely because English is
predictable; the incumbent's margin over it is +0.149 [+0.031, +0.278] and AR_G's is **negative**
(−0.074 [−0.143, −0.006] — more informative than the incumbent, yet redundant with the LM). So
psi_align's **audio margin must be positive with a bootstrap CI excluding zero** — computed at
the ONE weight pair `SAE_2S.md` approach 16 pre-registered (shaped `+0.075·prior − 1.0·hinge`
vs null `prior − 100·hinge`, fixed across every temperature and arm; a per-cell maximum is a
winner's curse) — or the "scorer" is just re-deriving the language model. Why G3 doesn't wait for G1: information and ranking
dissociate — AR_G was the best scorer by information and the worst by ranking, and the reverse
(an 11 M model with modest information concentrated exactly where candidates differ) is just as
possible; at under 1 GPU-h on existing dumps, this is the cheapest decisive measurement in the
plan. G3 passing → the swap protocol (§7), whatever G1 said. G3 failing with G1 passing → the
scorer sees text but the candidates don't differ where it looks: the sampler/homophone side
leads (G2P-equivalence ceiling, sampler diversity), no swap. And per the gate-vs-measurement
rule, a failed gate declines funding — it never concludes the loop would have failed.
Two G3 amendments from the build (2026-08-05, verifier-adopted): (1) a fourth control, the
**OOV-count null** — candidates ranked by −n_oov alone — forced by P0's measurement that 51 % of
rollout candidates carry an out-of-lexicon word (16× the gold rate): OOV words are mostly
misspellings, misspellings correlate with WER, so a lexicon-bearing scorer could outrank the
incumbent on lexicon information alone; any psi advantage the null also shows is not audio.
(2) The audio-margin bar at the fixed weight pair is valid as a *beat-the-null* test, but its
**magnitude is not comparable to the incumbent's +0.149** — the pair's weights are
scale-dependent and psi's recon column has a different dynamic range.

## 7. Loop-swap protocol (only after G3)

1. **Regression bed first**: the 10 h joint arm re-run with psi_align as the recon scorer AND as
   the decoupled AR-update (full-sum NLL on all G pairs per step — same decoupling as today's AR
   CE). Pre-registered bar: reach the incumbent trajectory band (≤ 13.15/16.13 at ep4) — the
   working configuration must not regress, else the swap is rejected regardless of gate wins.
2. Then the 100 h replay arm (§3c) with psi_align, and the §3d G-track re-gate at the AV^G init
   (the track resumes iff psi_align's G3 passes there too).
3. Hinge stays wired initially; a hinge-off single-variable arm runs only after (1) passes —
   the direct test of R6's learned length tax. That arm explicitly monitors **insertion rate and
   skip-arc usage on rollout text** (not aggregate WER alone): the specific exploit the learned
   skip price leaves open is short acoustically-plausible filler whose phones ride skips and emit
   nothing (verifier, 2026-08-05).
4. lambda recalibration per bed via the within-group monitors, as always.

## 8. Variants — fallback, escalation, rejected, deferred

**Scorer class by target rate (user question, 2026-08-05: "is the neural HMM still optimal for
k-means / highly subsampled targets?").** The lattice machinery is target-agnostic — emissions are
a swappable head — but which scorer class is *optimal* tracks one number, T/U_phones:

| regime | example target | right scorer |
|---|---|---|
| T/U ≫ 1 | fine units, 25–50 Hz | HMM/HSMM at full strength — durations are real signal |
| T/U ≈ 1 | the current 12.5 Hz (~1.3 frames/phone) | the HMM degrades gracefully to a **learned monotonic edit distance** (emissions = substitutions, loops/skips = ins/del); duration modeling adds little (geometric suffices, v1.1 unlikely to fire) but marginalization stays the point — at coarse rates a ±1-position misalignment is a whole phone, exactly what breaks position-indexed scoring |
| T/U < 1 | aggressive dedup / pooling / unit-BPE (§3b options) | per-phone states stop being the right granularity: go segmental on the text side (syllable/span states) or promote **fallback B to primary** — the transducer handles all ratios natively at the cost of the bounded leak |

So P0's T/U statistic is both the feasibility check and the scorer-class selector, and any §3b
target adoption re-runs it.

- **Fallback B — stateless-prediction transducer** (text encoder + unit joiner, prediction net =
  last-unit embedding only, Ghodsi et al.): exact DP, monotonic, bounded unit-bigram leak.
  Triggers: (i) G1 marginal on k500 with Viterbi diagnostics suggesting within-segment unit order
  carries signal the per-state categorical cannot; (ii) §3b adopts a T/U < 1 target (table above).
  The bigram leak must be priced by the shuffled-text contrast before any reward use.
- **v1.1 HSMM explicit durations** (categorical duration head, d ∈ 1..D_max, segment-lattice DP):
  trigger = duration-histogram mismatch or G1 marginal with otherwise healthy alignments. Note the
  regime table: at 12.5 Hz this is unlikely to be worth firing.
- **psi_align-C, the continuous-emission variant — REOPENED and funded by the user 2026-08-05
  (matrix cells M2/M4, §5b)**: Gaussian emissions on (subsampled) L15/AV states — same lattice,
  one head swap. The measured case FOR it: quantization is the pipeline's lossy step (hard
  k-means 0.63 PER vs linear-probe 0.145, `real-ceiling` finding), and TTE's external validation
  scored continuous encoder states. The 2026-07-31 continuous closure covered the *no-alignment*
  regressors (context-independent = position-only; prenet-history = the banned channel) — they
  lacked exactly what psi_align adds, and the user has explicitly reopened the axis for this
  family. Guard specific to scored regression: a variance floor (or fixed per-dim variance) so
  candidates are never ranked by variance collapse; PCA is fit label-free on unlabeled audio.
- **Rejected**: FastSpeech-as-scorer (point durations — fails R4); MAS/Viterbi-only likelihood
  (max ≠ sum exactly on the near-misses the reward exists to rank); any teacher-forced attention
  AR or unit-LM (R2, measured twice on our bed).

## 9. Risks and mitigations

1. **12.5 Hz squeeze** (T/U ≈ 1): P0 measures before anything is built; skip arcs from v1;
   syllable/25 Hz escalation path named.
2. **Degenerate early alignments** (all mass on one state): beta-binomial diagonal prior, annealed
   — the arXiv:2108.10447 recipe, adopted wholesale.
3. **Silence states hogging frames**: monitored (sil occupancy); cap sil self-loop prob if needed.
4. **OOV asymmetry in G3**: rollouts contain OOV/artifact tokens ("so'st", literal underscores —
   `SAE_2S.md` approach 15, conclusions 31-33); the incumbent scores raw text, psi_align needs phones. **One
   primary convention, pre-registered (verifier: two unregistered variants at the decisive gate =
   the per-cell winner's curse §6 bans): OOV word → a single UNK-word state with fixed
   unit-unigram emissions** — deterministic, needs no training (UNK never occurs in training
   text), and neutral by construction. "Drop-OOV-word" is the sensitivity read ONLY (it changes
   the phone count and re-enters the length channel of §4.1). P0 measures the OOV rate either
   way. lambda_4 already penalizes OOV in-loop, so the swap-time semantics are unchanged. The
   character arm (§5c) has no OOV at all — this convention is phone-arm-only, and phone-vs-char
   G3 comparisons run on the OOV-free subset as well (§5c).
5. **Reward-variance shrink from homophone collapse**: intended de-noising (section 4.3), but
   watch `reward/std_within_group` — groups differing only in spelling legitimately collapse to
   zero advantage.
6. **11 M underfits triphone context**: cheap to escalate (d=512 / 8 layers ≈ 25 M, still
   minutes-scale). Not a reason to start bigger.

## 10. What this sub-plan deliberately does not do

No loop compute before G3. No new rollout sampling for G3 (the dumps exist). No target-stream
changes (that is §3b's ledger; psi_align is target-agnostic by design). No claim that a better
scorer fixes homophone orthography — that ceiling is measured separately (queue item 5) and
belongs to the LM/lexicon side.

## 11. Sources

- Hori et al., Cycle-consistency training for end-to-end ASR (TTE): https://arxiv.org/abs/1811.01690
- Baskar et al., Semi-supervised seq2seq ASR with TTE: https://arxiv.org/abs/1905.01152
- Badlani et al., One TTS Alignment To Rule Them All (RAD-TTS aligner): https://arxiv.org/abs/2108.10447
- Kim et al., Glow-TTS (MAS): https://arxiv.org/abs/2005.11129
- Seamless (UnitY2 NAR T2U + discrete-unit aligner): https://arxiv.org/abs/2312.05187
- Mehta et al., Neural HMMs are all you need: https://arxiv.org/abs/2108.13320 ; OverFlow: https://arxiv.org/abs/2211.06892
- Ghodsi et al., RNN-T with stateless prediction network: https://ieeexplore.ieee.org/document/9054419
- Ren et al., FastSpeech 2: https://arxiv.org/abs/2006.04558
