# SAE — Speech AutoEncoder: Unsupervised ASR via a Text Bottleneck

Reconstruction-through-text unsupervised ASR, structurally following the NLA training loop
(transformer-circuits.pub/2026/nla): an **AV** (audio verbalizer, speech->text policy) and an **AR** (audio
reconstructor, text->speech-unit channel model) trained jointly — AR by supervised CE on AV samples, AV by
GRPO against the AR's reconstruction likelihood. The pair is an autoencoder over speech whose bottleneck is
a grapheme transcript and whose reconstruction score is an exact discrete likelihood, so the AV-optimal
policy is amortized noisy-channel decoding. In the adopted live `psi_align` system the channel conditions on
the scorer's own orthographic BPE states, not G2P: z_hat = argmax_z p_LM(z) * p_psi(u | BPE_states(z)); G2P
survives only in evaluation and probes.

> Index scope: objective, hard constraints, live queue, phase pointers. Results, per-phase State, gate reads
> and run catalogs live in the `SAE_<phase>.md` documents below; pre-unification subplans are frozen under
> `archive/` with their registered gates as provenance only, and reopened work restates its live gate in the
> phase document before producing a new result.

## North star & hard constraints

- **North star (user ruling 2026-08-01).** Real unsupervised ASR with the **autoencoder as the single main
  mechanism**; an adversarial init as the load-bearing mechanism would demote the autoencoder to a refiner.
  The mainline initialization question is §1e (pairing-free); GAN/§1d is the working label-free fallback
  init (§3d hierarchy).
- **Label quarantine.** True transcripts appear in exactly three quarantined places: evaluation metrics
  (PER/WER, probes, gate measurements on dev), the §0c architecture toplines, and the §2S anchor arm (1 h/10
  h paired seeds; its artifacts never feed the unsupervised ladder). In the unsupervised arm no training
  signal, checkpoint selection, or hyperparameter choice may depend on them; checkpoint selection uses dev
  reward + LM score only. Disclosed exception (NLA-style): loop mechanics and lambda ranges may be developed
  on §2S and reused. Amendment (USER 2026-08-14, strengthened 2026-08-16 — replaces the trigger-gated form):
  speaker IDs, previously never-train (2026-07-16 ruling), MAY train and may be tried first-line, disclosed
  as supervision cost; transcripts and alignments stay absolute (tier menu `archive/SAE_3g_spec_legacy.md`
  Z3).
- **Independence rule (GAN is not a teacher).** Admissible AR targets are *measurements of the audio*
  (deterministic transforms of encoder states), never another model's hypotheses; passing the label rule
  does not make a target admissible. One bounded carve-out (user 2026-08-03): GAN/§1d output as
  *initialization only* in the G-track (§3d) — never as in-loop teacher, reward, or selection signal.
- **Framing: usability, not superiority.** Matching at lower supervision cost is the win; pre-register
  non-inferiority margins; count circularity as a cost. "Unpaired" = no paired audio–text; Qwen3's
  pretraining almost surely contains the Gutenberg books underlying LibriSpeech — disclosed, controlled
  (§4), never hidden.
- **Evaluation discipline (USER rulings 2026-08-18, 2026-08-23).** Every model-evaluation comparison uses
  PAIRED data: both arms score the same items, read as per-item paired deltas with a resampled/clustered CI,
  never two pooled numbers. Constructed clause batteries (corruption ladders, proxy discrimination
  statistics) may gate spend inside a phase but never close one: a phase-closing better-or-worse verdict
  requires direct measurement of the real target quantity — ranking quality eta (equal, on shared groups, to
  the paired selection-WER delta over the shared oracle headroom) for scorers, plain WER for policies — in a
  fair paired comparison, and the closure decision then rests with the user.
- **Min-duration topology is standing (USER 2026-08-15):** every new scorer plan carries `d_min >= 2`; a
  given, not a revisitable choice. **Lambdas are per-bed:** sweep at <= 100 h, never 960 h; recalibrate
  against the within-group-std monitors, never carry values across beds (off-seed the LM prior decides
  converge-vs-turn and its share grows with bed size).
- **Storage placement (user decision 2026-08-22).** Jobs creating many small files put that payload on
  `$SCRATCH`; checkpoints and every durable/decision-bearing artifact stay in the project fileset; when
  relocating a job dir move only the payload subdir, never the `finished` marker.

## The live reward (index is its sole home)

Reward per sampled transcript z (utterance units u, duration D):

    r(z) = (1/|u|) log p_psi(u | BPE_states(z))   reconstruction (psi_align forward-sum;
                                                   graphemic bpe512 sub-states, 1.5 chars/state,
                                                   SIL at word boundaries)
         + lam_1 * lm_prior(z)                     LM prior, p_base; lm_prior_norm="units"
         - lam_2 * KL_hat(z)                       anchor to theta_0 (frozen SFT snapshot)
         - lam_3 * length_hinge(n_chars(z), D)     chars/s hinge (nu 14.55, len_eps 0.4);
                                                   lam_len 0 in the G-track arms, 0.5 in Z4

The live reward contains **no G2P anywhere** (corrected 2026-08-17, verified at source; verbatim correction
with source references in `SAE_ref.md`): the orthographic channel is LIVE, homophone spellings are NOT
reward-invariant, and the scorer carries a per-state price on orthographic length (the minimal-state
exploit's substrate). `lm_prior_norm="units"` is the standing fix because the per-token mean pays for length
(22:1 trade); `len_eps` 0.4 leaves a 49 % free band if the hinge is ever load-bearing. Updates are decoupled
(NLA shape): sample G=8–12 at the bed's T; the scorer is frozen in-loop by construction and any update goes
through §3e.1; AV by GRPO with group-normalized advantages.

## Priority queue (revision 2026-08-21; status read through 2026-08-26)

1. **§1g simple weak initialization — WITH THE USER.** 1g.2's own-minus-donor selector gate fired NEGATIVE
   (reference loses to the strongest content-free control by 5.02): H4 unresolved, maxima frozen, final
   refits and the 1,112-ID evaluation CLOSED, so no PER of that route can exist. 1g.9, the 1g.10/10a/10b/10c
   family, 1g.11, 1g.12 and 1g.13 are all closed on their own gates and are jointly evidence toward the
   training paradigm as the binding constraint. Closing 1g.12/1g.13 and the phone-versus-character route
   direction are the USER's word. `SAE_1g.md`.
2. **§1f — entry 9.1a running** (A9a `uni+bi+tri` seed 0 on the c5 merged stream, pinned 40,000 updates, ~17
   GPU-h), USER-funded 2026-08-26 as an explicit OVERRIDE of gate 9.0's registered "9.1 does not run". A9b
   and the audio-swap control are NOT run, so gate 9.1's SIGNATURE and CONTENT clauses do not fire and no
   reading may report 9.1 as passed or failed; only its HEALTH clause carries over, as a report. Gate 9.0 is
   not reopened and 9.2 is not licensed. With the USER: the TIMIT/ruling-4 fork and the stopping-rule
   question gate 9.0 left open. `SAE_1f.md`.
3. **§3e.1 D9 — banked, nothing running, AWAITS THE USER'S WORD**: close D9 under the joint license ("scorer
   refitting is not funded on this loop family at cold or evolved operating points") and decline the
   D8.3-style policy-leg assay (planner recommendation; the evolved policy's sampling collapse is banked as
   the family's newest fact). `SAE_3E1.md`.
4. **G-track loop — D6-PERIODIC/GAN960-FROZEN in flight** (`SAE_3E1.md` approach 33): gate is leg 8 beating
   this arm's own init 13.11/16.82 on both splits; matched-leg deltas against GAN-FROZEN are reported and
   select nothing. Everything else downstream of theta_0^G960 — loop, scorer refit, D6 branch, adopting it
   as an init elsewhere — stays unauthorized without a new preregistered decision. `SAE_3D_GTRACK.md`.
5. **§0d gate (ii) awaits the USER's blessing**: pass proposed for theta0-bed loop use at lam=1 only, NOT
   the re-swept peak lam=0.3, no G-track use licensed. The adapted reward sweeps mix EOS conventions, so
   their small margin claims await the registered re-score (direct SFT and WER endpoints unaffected); §2a
   rescorer and lam_1/lam_2 recalibration stay deferred behind §1g. `SAE_0d.md`.
6. **Two §3e.1 blessings pending**: the CI-vs-point convention pin (it also decides D4' round acceptance;
   clause tables stay dual-reported until confirmed) and gate v2 clause (i) floor-only.
7. **`archive/SAE_3a_spec_legacy.md` matrix wrap-up**: M4 contingency call; collapse when closed.
8. **§1e §2.5(d) + usage gates on the ep50 pins** — the §3d init upgrade path. `SAE_1e.md`.
9. **G2P-equivalence ceiling** on existing rollouts.jsonl (CPU): phone-reachable vs orthography-only
   oracle-gap split.
10. **Rung repair (Rung S, 1 h / 10 min)**: first attempt VOID (budget artifacts, not seed-size verdicts);
    extend AV budgets through the phase transition, ARs get full budget, then per-rung §2.5(d). `SAE_2S.md`.
11. **§2a unblocked but deferred** behind §1g: Qwen rescoring of the §1d lattices cannot resolve the
    north-star initialization question.
12. **§3b B0 gate table** — read under psi_align only if the target axis reopens.

**Parked**: G-track D4 round 1 and with it the bad-init self-repair read (revive on the user's word); D3. Do
not retry the closed offline D7-v2 graph (exact admission 56 rows/two speakers, necessary-core bound 120/four,
against the 6,778/201 floor); no solver retry, support-floor relaxation or graph amendment is authorized.

## Phase pointers (objective — gate status — file)

- **Phase 0 foundations** (0a representation audit, 0b lexicon/phonemization) — CLOSED: tuple frozen; linear
  probe 0.145 vs oracle-map ~0.53-0.60, so the *units*, not the encoder, cap hard assignment — the bound
  that closed §1a. `SAE_0.md`.
- **§0d LM-prior domain adaptation** — complete; pre-check (i) PASSED, gate (ii) open (queue 5). theta_0'
  re-SFT alone 11.43/15.54 dev, 11.99/14.34 test vs stock 16.91/20.64, 15.28/20.78 — better than anything
  any loop earned from stock theta_0. `SAE_0d.md`.
- **§1a decipherment** — CLOSED permanently on a bound (LL anti-aligned with PER; §0a oracle-map ceiling);
  scope amendment 2026-08-18 recorded there. `SAE_1a.md`.
- **§1c wav2vec-U 2.0 GAN** — PASSED on wav2vec2 and decided the encoder: perplexity-selected seed 0 =
  0.173/0.214 dev-clean/dev-other PER (0.137/0.168 oracle-best, diagnostic only); BEST-RQ flat 0.75-0.92.
  `SAE_1c.md`.
- **§1d Rung 0 self-training** — CLOSED: 0.172 dev-other phone PER; fixed lexicon/4-gram word decode
  17.96/21.87 dev WER, 2,703/2,864 utterances, zero empty hypotheses
  (`Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks`). `SAE_1d.md`.
- **§1e pairing-free initialization (mainline)** — UNDECIDED, gated on §2.5(d) (queue 8); kill-switch if all
  arms gate flat: non-adversarial output-distribution matching, then §1c/§1d stays the init of record.
  `SAE_1e.md`.
- **§1f statistics-matching initialization** — the fixed low-order family is CLOSED ON THIS BED by gate
  9.0's measurement ("not funding it here", never "it could not have worked"), qualified by the
  undeliverable second read; entry 9.1a running (queue 2). `SAE_1f.md`.
- **§1g simple weak starting point** — 1g.2 gate NEGATIVE, all sub-probes closed, direction with the user
  (queue 1); only a lexicon-free result supports the main claim, and the 1f 0.05/0.05 cliff is recorded but
  is not the future admission bar. `SAE_1g.md`.
- **Phase 2S anchor arm (quarantined)** — role complete at 10 h. Gate: loop beats identical-seed
  self-training by >= 0.5 dev-other, unsupervised-selected; at the fixed four-epoch endpoint joint AR wins
  by 1.61 (16.13 vs 17.74) with no label-based checkpoint choice (the earlier +1.24 was INVALID for this
  gate). Shuffled reward DECISIVE (ep1 207.59 vs 16.87). `SAE_2S.md`.
- **§3a psi_align reconstruction scorer** — ADOPTED (G1 + G3 passed 2026-08-05); text side `bpe512_cps15`;
  M2 and substrate closed; frozen within each policy leg, sha-verified; best bed is 100 h, shaped final
  6.06/10.31 dev, 6.33/10.84 test. `SAE_3A.md`; `archive/SAE_3a_spec_legacy.md`.
- **§3d G-track (GAN-init label-free; 960 h loop bed)** — scale gate PASSED: theta_0^G960 13.11/16.82 vs
  theta_0^G 13.89/18.34, the project's best label-free AV start; one-generation fresh-label gate FAILED both
  starts, no second generation authorized; no durable loop gain yet. `SAE_3D_GTRACK.md`.
- **§3e.1 scorer trainability without collapse** — D5 closed (continuous joint psi catastrophic); D6's
  one-shot `d_min=2` repair passed its matched continuation; the D6-PERIODIC/GAN recency A/B decided against
  refresh; D7 CLOSED on clause 2; D8 closed, reopened for the paired eta read, then CLOSED on the user's
  word (control retained); D9 banked (queue 3). `SAE_3E1.md`; `archive/SAE_3e1_spec_legacy.md`.
- **§3g Z-track (from-scratch, no GAN)** — all four arms closed; Z4 FAILED its gate with earnable variance
  remaining (not an exhausted loop); no Z5 funded, and the recommendation is a content-bearing §1g seed
  before any further no-pairs loop. `SAE_3G.md`.
- **Build/setup records** — 960 h loop build `sae_960h_loop_build.md`; reward side-inputs lam_1/lam_2 `SAE_ref.md`.

## Standing gates for phases with no separate document

- **§0c supervised topline of the exact AV architecture (PENDING, unscheduled).** Healthy: dev-other <= ~10
  %. Blocker: > 14.33 % — worse than the LS100 CTC baseline means the architecture, not unsupervision, is
  broken. Delta_input = WER(AV-U) − WER(AV) decides whether the token-only AV-U can carry mainline
  experiments.
- **§2a Rung 1 and §2b Rung 2 (PENDING).** Rung 1 <= WER of the 4-gram WFST decode of the *same* lattices,
  with a 4-gram-prior-only control separating "better prior" from memorization; Rung 2 <= Rung 1 + 1 abs AND
  dev insertion rate <= 1.5x the teacher's.
- **§2c AR SFT SUPERSEDED for the reward** by psi_align (old Delta-CE usage screen superseded by
  §2.5(c)/(d); measured 2026-07-17: full-history Delta-CE ~ +0.005, a target wall, which started the scorer
  program). **§3b target SETTLED at avunits k500**: admissible targets are measurements of the audio only,
  compared same-set under §2.5(d) against the incumbent stream. **§3e protocol**: checkpoint selection by
  dev reward + LM score only; monitor reward components, ins/del and within-group std separately; a
  degrading run is reverted, not compounded.
- **§2.5 go/no-go instruments — IN ACTIVE SERVICE; (d) is decisive for every new scorer, target or init.**
  (d) reward-RANK probe: replay the loop step on real theta_0 rollouts (G~12, T in {0.3, 0.5, 0.7}; T=1.0
  logged, never evidence). Gate: within-group spearman with CI > 0, gap_true = r(z_true) − mean r(z_i) > 0,
  reward-selected WER <= group mean. Read discipline (2026-08-05): **absolute-eta bars withdrawn** —
  same-bed/same-n/same-G, gap_true + spearman lead, plus the audio margin over the audio-free null.
  Calibrate any new diagnostic on the §2S paired-init models first (failure there indicts the instrument,
  not the signal); (c) is a known-optimistic synthetic proxy and (b) is superseded by (d).
- **§3f exit gate (Rung 3; pre-registered, unchanged) — NOT FIRED.** All of: (1) dev-other <= min(Rung 0,
  Rung 2) − 0.5 abs; (2) the winning checkpoint is the one the **unsupervised** criterion selects; (3) sign
  reproduced by a second RL seed; (4) stable over the last third, ins/del within 1.5x SFT, §4 probes clean;
  (5) reported head-to-head vs Rung 3-BT — if RL loses, BT becomes the headline and RL the reported negative
  arm. If (1) fails with §2.5 passed, the failure localizes to the loop (lambda balance, scorer drift,
  anchor) — iterate there, not in Phase 1.
- **Phase 3B backtranslation (NOT STARTED, pending Phase 2).** Unit-level iterative backtranslation between
  AV-U and the AR; invariant: each model always trains toward a REAL target, only sources are synthetic, ~50
  % previous-round data retained, unsupervised stopping. Gate: >= 1 round of positive unsupervised-score
  gain, and Rung 3-BT <= Rung 2 − 0.5 abs, unsupervised-selected.
- **Phase 4 controls and ablations — all probes reported, no numeric gate** except speaker leakage (linear
  speaker-ID probe on AV states, pre vs post RL: accuracy gain <= 2 abs). Dev probes: orthographic homophone
  swap / case-punctuation jitter (reconstruction, LM-prior and composed-reward deltas reported separately —
  the live BPE scorer is not homophone-invariant, so this is an attribution diagnostic, not a pass
  condition); word-boundary resegmentation at equal lexical content; content sensitivity by random
  BPE-distinct word substitution (more-negative reward is better, scaled against within-group reward std).
  Shuffled-reward control DONE and DECISIVE (2026-08-04) — the reward is load-bearing. Remaining 100 h
  ablations: scorer-frozen-vs-updated (now §3e.1), lam_1 = 0, lam_2 = 0, pure-phoneme Option A, warm-start
  degradation sweep, confabulation check, contamination control (log p_base of true dev transcripts vs
  length-matched LM-corpus sentences; 4-gram-only prior deltas).
- **Phase 5 refinement (NOT STARTED, gated on Rung 3 > Rung 0).** (a) Qwen3-8B warm-started from the winning
  branch's pseudo-labels; (b) label-free speaker embedding + quantized F0/energy streams conditioning the AR
  (usage-gated); (c) 8B n-best noisy-channel rescoring tuned on dev by reward. Gate: Rung 4 dominates Rung 3
  with the side-channel delta isolated.

## Resources, notation, anchors

| Item     | Value |
|----------|-------|
| Audio    | LibriSpeech 960 h (no transcripts in the unsupervised arm) |
| Text     | LibriSpeech LM corpus (`get_librispeech_normalized_lm_data()`) |
| Prior knowledge | Pronunciation lexicon + G2P (allowed); MFA gold alignments (evaluation only) |
| Encoder  | **wav2vec2-Large-lv60, layer 15** (SSL-only ckpt; decided 2026-07-18, §1c). 1024-d @ 50 Hz, per-utterance norm; units = k-means K=500 on 50->25 Hz pooled states; AV adapter stride x4 -> 12.5 Hz. Frozen for unit dumps and the GAN; AV SFT trains the transformer (conv extractor frozen); frozen inside the GRPO loop. BEST-RQ = documented negative (`SAE_1c.md`). lv60 pretrains on 60 kh LibriLight audio, zero transcripts. |
| LLM      | Qwen3-1.7B (Phases 0–4), Qwen3-8B (Phase 5 only) |
| Compute  | 4xGH200 96 GB per experiment |

**Notation.** x waveform; h = E_l(x) encoder features; u = dedup(kmeans_K(h)) unit sequence; z grapheme
transcript; phi = G2P(z), stress-free ARPAbet, one canonical pronunciation per word, no word-boundary
symbols in AR inputs. AV: p_theta(z|x) = base LLM + LoRA-A + conv downsampler/projector. AR/scorer:
p_psi(u|phi). AV-U: p(z|u), unit-token-input verbalizer (LoRA-A'), the §3B vehicle. p_base(z): frozen
adapterless base LLM as grapheme prior. T: text corpus; T_phi = G2P(T).

**Code anchors** (relative to `recipe/`; `ssl/` = `i6_experiments/users/wu/experiments/ssl/`, fixed
2026-08-17 — the bare `ssl/` base does not exist under `recipe/`): AV SFT recipe under
`2025-10-speech-llm/.../librispeech/configs/` (w2v2 variant `config_sae_2s_av_sft_w2v2_v1.py`); GRPO loop
`train_steps/sae_grpo.py` + configs `config_sae_3a_*`; psi_align `sae/psi_align.py` +
`sae/psi_align_jobs.py`; HF downloads `hf_models.py`; k-means
`ssl/experiments/pretrain_two_level/kmeans.py`; LM corpus / lexicon / G2P
`i6_experiments/common/datasets/librispeech/{language_model,lexicon}.py`; gold alignments
`ssl/analysis/seg_diag.py` (eval only); external refs fairseq `examples/wav2vec/unsupervised`, ESPUM
arXiv:2310.02382, Hori et al. arXiv:1811.01690; surveys `ssl/LITERATURE_REVIEW.md`, `ssl/SPEECH_UNIT_BPE_REVIEW.md`.

## Deliverables ladder

| Rung | Claim | Must dominate |
|------|-------|---------------|
| 0    | bootstrap + self-training + WFST decode (standard recipe) | — |
| 1    | LLM rescoring of the same lattices | same-lattice 4-gram decode |
| 2    | AV SFT distillation of Rung 1 | Rung 1 |
| 3-BT | iterative backtranslation, distilled back, no RL | Rung 2 |
| 3    | reconstruction-reward GRPO, identical bootstrap | Rung 0, Rung 2; head-to-head vs 3-BT |
| 4    | 8B + side channels | best of 3 / 3-BT |
| S    | anchor arm: RL from {1 h, 10 h} seed vs self-training from the identical seed | separate supervision axis |

Publish from the highest rung that holds; the BT branch and Rung S hedge the RL and bootstrap axes
respectively. The SAE story survives either head-to-head outcome — both branches instantiate the
text-bottleneck autoencoder.
