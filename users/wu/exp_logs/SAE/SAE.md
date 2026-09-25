# SAE — Speech AutoEncoder: Unsupervised ASR via a Text Bottleneck

Reconstruction-through-text unsupervised ASR, structurally following the NLA training loop
(transformer-circuits.pub/2026/nla): an **AV** (audio verbalizer, speech->text policy) and an **AR** (audio
reconstructor, text->speech-unit channel model) trained jointly — AR by supervised CE on AV samples, AV by GRPO
against the AR's reconstruction likelihood. The pair is an autoencoder over speech whose bottleneck is a grapheme
transcript and whose reconstruction score is an exact discrete likelihood, so the AV-optimal policy is amortized
noisy-channel decoding. In the adopted live `psi_align` system the channel conditions on the scorer's own
orthographic BPE states, not G2P: z_hat = argmax_z p_LM(z) * p_psi(u | BPE_states(z)); G2P survives only in
evaluation and probes.

> Results and State live in `SAE_<phase>.md`; frozen `archive/` gates are provenance only.
> Reopened work restates its live gate in the phase document before a new result.

## North star & hard constraints

- **North star (user priority 2026-09-16).** Pure unsupervised ASR **without GANs**, improving **cold start
  within §4a cycle consistency**. The user also reopens 10 h supervised-init refinement as a parallel analysis;
  its results remain separate from cold-start progress. No GAN fallback; scope and amendment: `SAE_ref.md`.
- **Label quarantine.** True transcripts appear in exactly three quarantined places: evaluation metrics (PER/WER,
  probes, gate measurements on dev), the §0c architecture toplines, and the §2S anchor arm (1 h/10 h paired seeds;
  its artifacts never feed the unsupervised ladder). In the unsupervised arm no training signal, checkpoint
  selection or hyperparameter choice may depend on them; checkpoint selection uses dev reward + LM score only.
  Disclosed exception (NLA-style): loop mechanics and lambda ranges may be developed on §2S and reused. Amendment
  (USER 2026-08-14, strengthened 2026-08-16 — replaces the trigger-gated form): speaker IDs, previously
  never-train (2026-07-16 ruling), MAY train and may be tried first-line, disclosed as supervision cost;
  transcripts and alignments stay absolute (tier menu `archive/SAE_3g_spec_legacy.md` Z3).
- **Independence rule.** Admissible AR targets are *measurements of the audio*
  (deterministic transforms of encoder states), never another model's hypotheses; passing the label rule does not
  make a target admissible. The former GAN-initialization carve-out is superseded by the current no-GAN priority.
- **Framing: usability, not superiority.** Matching at lower supervision cost is the win; pre-register
  non-inferiority margins; count circularity as a cost. "Unpaired" = no paired audio–text; Qwen3's pretraining
  almost surely contains LibriSpeech's Gutenberg books — disclosed, controlled (§4), never hidden.
- **Evaluation discipline (USER rulings 2026-08-18, 2026-08-23).** Every model-evaluation comparison is PAIRED:
  both arms score the same items, read as per-item paired deltas with a resampled/clustered CI, never two pooled
  numbers. Constructed clause batteries (corruption ladders, proxy discrimination statistics) gate spend inside a
  phase, never close one: a phase-closing better-or-worse verdict requires direct measurement of the target
  quantity — ranking quality eta (equal, on shared groups, to the paired selection-WER delta over the shared
  oracle headroom) for scorers, plain WER for policies — in a fair paired comparison; closure rests with the user.
- **Min-duration topology is standing (USER 2026-08-15):** every new scorer plan carries `d_min >= 2`, a given,
  not revisitable. **Lambdas are per-bed:** sweep at <= 100 h, never 960 h; recalibrate on within-group-std
  monitors, never across beds (off-seed the LM prior decides converge-vs-turn, its share growing with bed size).
- **Storage placement (user decision 2026-08-22).** Jobs creating many small files put that payload on `$SCRATCH`;
  checkpoints and every durable/decision-bearing artifact stay in the project fileset; when relocating a job dir
  move only the payload subdir, never the `finished` marker.

## The live reward (index is its sole home)

Reward per sampled transcript z (utterance units u, duration D):

    r(z) = (1/|u|) log p_psi(u | BPE_states(z))   reconstruction (psi_align forward-sum;
                                                   graphemic bpe512 sub-states, 1.5 chars/state,
                                                   SIL at word boundaries)
         + lam_1 * lm_prior(z)                     LM prior, p_base; lm_prior_norm="units"
         - lam_2 * KL_hat(z)                       anchor to theta_0 (frozen SFT snapshot)
         - lam_3 * length_hinge(n_chars(z), D)     chars/s hinge (nu 14.55, len_eps 0.4);
                                                   lam_len 0 in the G-track arms, 0.5 in Z4

The live reward contains **no G2P anywhere** (corrected 2026-08-17, verified at source; verbatim correction and
source refs in `SAE_ref.md`): the orthographic channel is LIVE, homophone spellings are NOT reward-invariant, and
the scorer carries a per-state price on orthographic length (the minimal-state exploit's substrate).
`lm_prior_norm="units"` is the standing fix because the per-token mean pays for length (22:1 trade); `len_eps` 0.4
leaves a 49 % free band if the hinge is ever load-bearing. Updates are decoupled (NLA shape): sample G=8–12 at the
bed's T; the scorer is frozen in-loop by construction and any update goes through §3e.1; AV by GRPO with
group-normalized advantages.

## Priority queue (revision 2026-09-15 adds item 0; other statuses read through 2026-08-26)

Older entries retain historical state; the current no-GAN, cold-start priority governs new work.

0. **§4a exact-marginal cycle (EMC) — ACTIVE, ahead of the rest by user direction 2026-09-15.** Phone-level,
   no LLM: phone recognizer + frozen phone m-gram + semi-Markov unit reverse model, marginalized alignment.
   Prior cold failures, completed S2e/S2f/S2g supervised refinements and audited M512 output inspection:
   `SAE_4A.md`. GAN-lineage initialization remains retired; S3d remains stopped.
   New main direction: VAD + stride-3 blank-free model, `SAE_4A_blankfree.md`: cold trigram joint
   training and separate supervised 10 h fits (no adaptation); sampled-group diagnostic is complete/audited.
   Blankfree cold ep4 0.865 FAILS G4a.3. **Attribution 2x2 (user 2026-09-19), `SAE_4A_attrib.md`:**
   training-free mode-seeking read, no-reverse arm, K=64 reverse arm, GAN+reverse arms, in parallel.
   Attribution closed on its rule (no branch fires). **Budget round (user 2026-09-20), N = 50 arms READ 2026-09-21 (FAIL, all six at PER
   0.89–0.91, no treatment or schedule effect beyond 0.016), N = 100 arms KILLED by the user
   2026-09-21 at epoch 65–67 (phase CLOSED), `SAE_4A_budget.md`:** 50 / 100 sub-epochs with LR + tau schedule; control, lattice prior + BT
   (jointly trained), coverage + LM prior; gate G4a.4. **InfoMax round (user 2026-09-20), CLOSED 2026-09-21 FAIL,
   `SAE_4A_infomax.md`:** conditional-entropy penalty (decayed / held) x augmentation invariance,
   four arms on one node, paired against ctrl_50; gate G4a.5 read at sub-epoch 50: PER 0.89–0.92
   in every arm, no band exit (max 0.021 below own chance null), ent_50 0.019 worse than ctrl_50;
   confidence and consistency do not move the cold bed out of the content-free band. Objective derivation: `SAE_4A_objective.md`.
   **Context-dependent reverse model (user 2026-09-20), DEFERRED without limit by the user the same
   day, `SAE_4A_cdrev.md`:** emission `p_phi(x_seg | k, previous phone)`; gate G4a.6; design reviewed
   (approve with amendments, applied), nothing built. Literature is against it
   (`reports/lit_cdrev_2026-09-20.md`).
   **Prior strength (user 2026-09-20), OPEN, `SAE_4A_prior.md`:** the private code already satisfies
   the trigram nearly as well as phones (−3.98 vs −3.20 per token), so weight is not the lever;
   Step 0 is a CPU prior-gap diagnostic (orders 1-8 and the exact lexicalised prior) with a fixed
   decision table between a 4-gram importance-sampled correction and a lexicon score-function
   term. Step 0 read (audited): the lexicon, not order or weight, identifies the private code
   (gap 2.32 vs trigram 1.39); IS closed; the arm is a score-function term with a strong scorer
   (G4a.7 defined, design-reviewed, amendments applied). Step 0b (neural phone LM as the
   scorer): 3.3 M / 3 epochs was a partial proxy (gap 1.71 < 2.01 bar); the 30-epoch rerun at
   3.3 M and 10.9 M plateaus at gap 1.85 (no-improvement abort: the 1 M-line window is the
   ceiling); the 25 M / 10 M-line instance (c) reaches held ppl 3.96 but gap 1.81, so Step 0b
   CLOSED 2026-09-21: no phone LM trained on this text reaches the bar (fluent-but-wrong decodes
   are rewarded by any phone-sequence model; only the word constraint discriminates). Pre-launch
   falsifiers read 2026-09-21: the amended reward clears the sign trap, but gold beats every
   sampled string in 100 / 100 / 97 % of utterances at ep1 / ep4 / ep10, so under the
   pre-registered rule the score-function arm is NOT funded (audited CONFIRMED). The phase's
   successor is the lexicon inside the marginalised lattice (GPU trie DP; survey and literature
   banked), own phase `SAE_4A_lexlat.md`: E-1 PASS, E0 read (C = 4096), **E1 FAIL on both cost
   clauses (C = 4096 OOM at 95 GiB against the 80 GiB bar; C = 1024 at 802–912 s per step, about
   41× the 1202 s per sub-epoch bar)**; user ruling 2026-09-21 made it the main line (cost work
   until E1 passes, no moved bar); the k2 second-graph form (amendment 9) PASSED E1 and the
   over-count read, twelve arms ran 20 sub-epochs on three nodes 2026-09-22, gate G4a.9 read
   CANNOT_TELL with PASS unreachable (every arm PER 0.81-0.86, the deranged null matches the
   treatment), diagnosis D1-D8 in the phase file; user ruling 2026-09-22: train longer to the PER
   plateau (`k2lat_20_ma3000` continued to 60 sub-epochs, phase file "Extension E60"), other
   changes wait for the diagnosis write-up. **User ruling 2026-09-21: the 3.3 M transformer phone LM is approved as
   the scorer despite the Step 0b bar**, so the prior phase's soft (straight-through) arm is
   funded at N = 20 with four arms (two seeds, a 0.3× strength point, a permuted-identity null),
   spec in `SAE_4A_prior.md` "Training arm, reopened by user ruling"; build in progress.
   Other follow-up candidates: K = 64 reverse units; an explicit end-of-word symbol in the phone
   set (user idea 2026-09-21; SIL already marks word boundaries with p = 0.5 in the prior text, and
   a hard EOW the audio never realises is a free symbol for the private code, so it is a training
   arm read by the derangement gap and the prior-gap table, not a CPU screen).
   **Cold-line successor (user decision 2026-09-23), own phase `SAE_4A_lexlat_v2.md`:** phi-first EM
   decipherment from a random reverse model with likelihood-selected restarts and a competence ladder,
   then a bridge into the joint run; pure unsupervised and GAN-free by ruling; gates G4a.L2.1-4
   registered before any job. Next-step proposal phase `SAE_4A_rename.md` (user request 2026-09-24):
   what phi training needs so EM can correct wrong phone names; analyses only, training arms proposed.
   Phi-init plan `SAE_4A_phiinit.md` (user request 2026-09-25; runs on i6, unfunded): coarse-to-fine named
   alphabet with binary splits and a SIL anchor; oracle split arm first.
   **User directive 2026-09-20 (execute autonomously):** wav2vec-U 2.0 is the role model. (1) Fix
   the sub-epoch count for future arms from the label-free behaviour of the running budget arms
   (50 is suspected too long after the stall). (2) Paper-faithful silence handling: the bed masks
   features after full-waveform SSL extraction, the paper cuts the waveform with rVAD before
   extraction (verified, `reports/lit_w2vu2_preprocessing_2026-09-20.md`); one ablation arm with
   the paper's cut, own phase `SAE_4A_prepro.md` -- READ 2026-09-21, G4a.8 FAIL: all three
   N = 20 arms at PER 0.875–0.889, the cut +0.014 worse than the masked control at ep20, outside
   the seed band (−0.001); the silence convention is not the stall's cause; bed stays masked
   (orchestrator ruling, user may overturn). (3) Train a phone LM that reaches the Step 0b
   gate and launch the strong-scorer arm (`SAE_4A_prior.md`).
   **§4b weighted-L1 follow-up COMPLETE, audited:** sparsity and phone statistics reported;
   improvement over dense and BatchTopK not met. No further arm queued. `SAE_4B.md`.
1. **§1g simple weak initialization — WITH THE USER.** 1g.2's own-minus-donor selector gate fired NEGATIVE
   (reference loses to the strongest content-free control by 5.02): H4 unresolved, maxima frozen, final refits and
   the 1,112-ID evaluation CLOSED, so no PER of that route exists. 1g.9, the 1g.10/10a/10b/10c family, 1g.11,
   1g.12 and 1g.13 are closed on their own gates, jointly evidence for the training paradigm as the binding
   constraint. Closing 1g.12/1g.13 and the phone-vs-character route direction are the USER's word. `SAE_1g.md`.
2. **§1f — entry 9.1a running** (A9a `uni+bi+tri` seed 0 on the c5 merged stream, pinned 40,000 updates, ~17
   GPU-h), USER-funded 2026-08-26 as an OVERRIDE of gate 9.0's registered "9.1 does not run". A9b and the
   audio-swap control are NOT run, so gate 9.1's SIGNATURE and CONTENT clauses do not fire and no reading may
   report 9.1 passed or failed; only its HEALTH clause carries over, as a report. Gate 9.0 is not reopened; 9.2 is
   not licensed. With the USER: the TIMIT/ruling-4 fork and the stopping-rule question gate 9.0 left open.
   `SAE_1f.md`.
3. **§3e.1 D9 — banked, nothing running, AWAITS THE USER'S WORD**: close D9 under the joint license ("scorer
   refitting is not funded on this loop family at cold or evolved operating points") and decline the D8.3-style
   policy-leg assay (planner recommendation; the evolved policy's sampling collapse is banked as the family's
   newest fact). `SAE_3E1.md`.
4. **G-track loop — D6-PERIODIC/GAN960-FROZEN in flight** (`SAE_3E1.md` approach 33): gate is leg 8 beating this
   arm's own init 13.11/16.82 on both splits; matched-leg deltas against GAN-FROZEN are reported and select
   nothing. Everything else downstream of theta_0^G960 — loop, scorer refit, D6 branch, adopting it as an init
   elsewhere — stays unauthorized without a new preregistered decision. `SAE_3D_GTRACK.md`.
5. **§0d gate (ii) awaits the USER's blessing**: pass proposed for theta0-bed loop use at lam=1 only, NOT the
   re-swept peak lam=0.3, no G-track use licensed. The adapted reward sweeps mix EOS conventions, so their small
   margin claims await the registered re-score (direct SFT and WER endpoints unaffected); §2a rescorer and
   lam_1/lam_2 recalibration stay deferred behind §1g. `SAE_0d.md`.
6. **Two §3e.1 blessings pending**: the CI-vs-point convention pin (it also decides D4' round acceptance; clause
   tables stay dual-reported until confirmed) and gate v2 clause (i) floor-only.
7. **`archive/SAE_3a_spec_legacy.md` matrix wrap-up**: M4 contingency call; collapse when closed.
8. **§1e §2.5(d) + usage gates on the ep50 pins** — the §3d init upgrade path. `SAE_1e.md`.
9. **G2P-equivalence ceiling** on rollouts.jsonl (CPU): phone-reachable vs orthography-only oracle-gap split.
10. **Rung repair (Rung S, 1 h / 10 min)**: first attempt VOID (budget artifacts, not seed-size verdicts); extend
    AV budgets through the phase transition, ARs get full budget, then per-rung §2.5(d). `SAE_2S.md`.
11. **§2a unblocked but deferred** behind §1g: Qwen rescoring of the §1d lattices cannot resolve the north-star
    initialization question.
12. **§3b B0 gate table** — read under psi_align only if the target axis reopens.

**Parked**: G-track D4 round 1 and with it the bad-init self-repair read (revive on the user's word); D3. Do not
retry the closed offline D7-v2 graph (exact admission 56 rows/two speakers, necessary-core bound 120/four, against
the 6,778/201 floor); no solver retry, support-floor relaxation or graph amendment is authorized.

## Phase pointers (objective — gate status — file)

- **Phase 0 foundations** (0a representation audit, 0b lexicon/phonemization) — CLOSED: tuple frozen; linear probe
  0.145 vs oracle-map ~0.53-0.60, so the *units*, not the encoder, cap hard assignment — the bound that closed
  §1a. `SAE_0.md`.
- **§0d LM-prior domain adaptation** — complete; pre-check (i) PASSED, gate (ii) open (queue 5). theta_0' re-SFT
  alone 11.43/15.54 dev, 11.99/14.34 test vs stock 16.91/20.64, 15.28/20.78 — better than anything any loop earned
  from stock theta_0. `SAE_0d.md`.
- **§1a decipherment** — CLOSED permanently on a bound (LL anti-aligned with PER; §0a oracle-map ceiling); scope
  amendment 2026-08-18 recorded there. `SAE_1a.md`.
- **§1c wav2vec-U 2.0 GAN** — PASSED on wav2vec2 and decided the encoder: perplexity-selected seed 0 = 0.173/0.214
  dev-clean/dev-other PER (0.137/0.168 oracle-best, diagnostic only); BEST-RQ flat 0.75-0.92. `SAE_1c.md`.
- **§1d Rung 0 self-training** — CLOSED: 0.172 dev-other phone PER; fixed lexicon/4-gram word decode 17.96/21.87
  dev WER, 2,703/2,864 utterances, zero empty hypotheses (`Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks`). `SAE_1d.md`.
- **§1e pairing-free initialization (mainline)** — UNDECIDED, gated on §2.5(d) (queue 8); kill-switch if all arms
  gate flat: non-adversarial output-distribution matching, then §1c/§1d stays the init of record. `SAE_1e.md`.
- **§1f statistics-matching initialization** — the fixed low-order family is CLOSED ON THIS BED by gate 9.0's
  measurement ("not funding it here", never "it could not have worked"), qualified by the undeliverable second
  read; entry 9.1a running (queue 2). `SAE_1f.md`.
- **§1g simple weak starting point** — 1g.2 gate NEGATIVE, all sub-probes closed, direction with the user (queue
  1); only a lexicon-free result supports the main claim, and the 1f 0.05/0.05 cliff is recorded but is not the
  future admission bar. `SAE_1g.md`.
- **Phase 2S anchor arm (quarantined)** — role complete at 10 h. Gate: loop beats identical-seed self-training by
  >= 0.5 dev-other, unsupervised-selected; at the fixed four-epoch endpoint joint AR wins by 1.61 (16.13 vs 17.74)
  with no label-based checkpoint choice (the earlier +1.24 was INVALID for this gate). Shuffled reward DECISIVE
  (ep1 207.59 vs 16.87). `SAE_2S.md`.
- **§3a psi_align reconstruction scorer** — ADOPTED (G1 + G3 passed 2026-08-05); text side `bpe512_cps15`; M2 and
  substrate closed; frozen within each policy leg, sha-verified; best bed is 100 h, shaped final 6.06/10.31 dev,
  6.33/10.84 test. `SAE_3A.md`; `archive/SAE_3a_spec_legacy.md`.
- **§3d G-track (GAN-init label-free; 960 h loop bed)** — scale gate PASSED: theta_0^G960 13.11/16.82 vs theta_0^G
  13.89/18.34, the project's best label-free AV start; one-generation fresh-label gate FAILED both starts, no
  second generation authorized; no durable loop gain yet. `SAE_3D_GTRACK.md`.
- **§3e.1 scorer trainability without collapse** — D5 closed (continuous joint psi catastrophic); D6's one-shot
  `d_min=2` repair passed its matched continuation; the D6-PERIODIC/GAN recency A/B decided against refresh; D7
  CLOSED on clause 2; D8 closed, reopened for the paired eta read, then CLOSED on the user's word (control
  retained); D9 banked (queue 3). `SAE_3E1.md`; `archive/SAE_3e1_spec_legacy.md`.
- **§3g Z-track (from-scratch, no GAN)** — all four arms closed; Z4 FAILED its gate with earnable variance
  remaining (not an exhausted loop); no Z5 funded, and the recommendation is a content-bearing §1g seed before any
  further no-pairs loop. `SAE_3G.md`.
- **§4a exact-marginal cycle (EMC)** — OPEN; G4a.1 read (S1a, spend control only); G4a.2 read and audited
  (S2b/S2c: refines vs own init at best -2.0 WER, none usable vs §1d 21.87); G4a.3 CLOSED FAIL (S3 flat start).
  S2d FAIL; iterated factorized target fitting drifts, without an established limiting posterior. Frozen/joint
  reverse-model diagnostics remain active; CT CLOSED FAIL on its audited endpoint. `SAE_4A.md`.
- **Build/setup** — 960 h loop build `sae_960h_loop_build.md`; reward side-inputs lam_1/lam_2 `SAE_ref.md`.
- **§4b sparse acoustic codes from w2v2** — BatchTopK and Scaling Monosemanticity weighted-L1
  arms complete and audited, with sparsity and phoneme statistics. Phase remains open. `SAE_4B.md`.

## Standing gates for phases with no separate document

- **§0c supervised topline of the exact AV architecture (PENDING, unscheduled).** Healthy: dev-other <= ~10 %.
  Blocker: > 14.33 % — worse than the LS100 CTC baseline means the architecture, not unsupervision, is broken.
  Delta_input = WER(AV-U) − WER(AV) decides whether the token-only AV-U can carry mainline experiments.
- **§2a Rung 1 and §2b Rung 2 (PENDING).** Rung 1 <= WER of the 4-gram WFST decode of the *same* lattices, with a
  4-gram-prior-only control separating "better prior" from memorization; Rung 2 <= Rung 1 + 1 abs AND dev
  insertion rate <= 1.5x the teacher's.
- **§2c AR SFT SUPERSEDED for the reward** by psi_align (old Delta-CE usage screen superseded by §2.5(c)/(d);
  measured 2026-07-17: full-history Delta-CE ~ +0.005, a target wall, which started the scorer program). **§3b
  target SETTLED at avunits k500**: admissible targets are measurements of the audio only, compared same-set under
  §2.5(d) against the incumbent stream. **§3e protocol**: checkpoint selection by dev reward + LM score only;
  monitor reward components, ins/del and within-group std separately; a degrading run is reverted, not compounded.
- **§2.5 go/no-go instruments — IN ACTIVE SERVICE; (d) is decisive for every new scorer, target or init.** (d)
  reward-RANK probe: replay the loop step on real theta_0 rollouts (G~12, T in {0.3, 0.5, 0.7}; T=1.0 logged,
  never evidence). Gate: within-group spearman with CI > 0, gap_true = r(z_true) − mean r(z_i) > 0,
  reward-selected WER <= group mean. Read discipline (2026-08-05): **absolute-eta bars withdrawn** —
  same-bed/same-n/same-G, gap_true + spearman lead, plus the audio margin over the audio-free null. Calibrate any
  new diagnostic on the §2S paired-init models first (failure there indicts the instrument, not the signal); (c)
  is a known-optimistic synthetic proxy and (b) is superseded by (d).
- **§3f exit gate (Rung 3; pre-registered, unchanged) — NOT FIRED.** All of: (1) dev-other <= min(Rung 0, Rung 2)
  − 0.5 abs; (2) the winning checkpoint is the one the **unsupervised** criterion selects; (3) sign reproduced by
  a second RL seed; (4) stable over the last third, ins/del within 1.5x SFT, §4 probes clean; (5) reported
  head-to-head vs Rung 3-BT — if RL loses, BT becomes the headline and RL the reported negative arm. If (1) fails
  with §2.5 passed, the failure localizes to the loop (lambda balance, scorer drift, anchor) — iterate there, not
  in Phase 1.
- **Phase 3B backtranslation (NOT STARTED, pending Phase 2).** Unit-level iterative backtranslation between AV-U
  and the AR; invariant: each model always trains toward a REAL target, only sources are synthetic, ~50 %
  previous-round data retained, unsupervised stopping. Gate: >= 1 round of positive unsupervised-score gain, and
  Rung 3-BT <= Rung 2 − 0.5 abs, unsupervised-selected.
- **Phase 4 controls and ablations — all probes reported, no numeric gate** except speaker leakage (linear
  speaker-ID probe on AV states, pre vs post RL: accuracy gain <= 2 abs). Dev probes: orthographic homophone swap
  / case-punctuation jitter (reconstruction, LM-prior and composed-reward deltas reported separately — the live
  BPE scorer is not homophone-invariant, so this is an attribution diagnostic, not a pass condition);
  word-boundary resegmentation at equal lexical content; content sensitivity by random BPE-distinct word
  substitution (more-negative reward is better, scaled against within-group reward std). Shuffled-reward control
  DONE and DECISIVE (2026-08-04) — the reward is load-bearing. Remaining 100 h ablations: scorer-frozen-vs-updated
  (now §3e.1), lam_1 = 0, lam_2 = 0, pure-phoneme Option A, warm-start degradation sweep, confabulation check,
  contamination control (log p_base of true dev transcripts vs length-matched LM-corpus sentences; 4-gram-only
  prior deltas).
- **Phase 5 refinement (NOT STARTED, gated on Rung 3 > Rung 0).** (a) Qwen3-8B warm-started from the winning
  branch's pseudo-labels; (b) label-free speaker embedding + quantized F0/energy streams conditioning the AR
  (usage-gated); (c) 8B n-best noisy-channel rescoring tuned on dev by reward. Gate: Rung 4 dominates Rung 3 with
  the side-channel delta isolated.

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
transcript; phi = G2P(z), stress-free ARPAbet, one canonical pronunciation per word, no word-boundary symbols in
AR inputs. AV: p_theta(z|x) = base LLM + LoRA-A + conv downsampler/projector. AR/scorer: p_psi(u|phi). AV-U:
p(z|u), unit-token-input verbalizer (LoRA-A'), the §3B vehicle. p_base(z): frozen adapterless base LLM as grapheme
prior. T: text corpus; T_phi = G2P(T).

**Code anchors** (relative to `recipe/`; `ssl/` = `i6_experiments/users/wu/experiments/ssl/`, fixed 2026-08-17 —
the bare `ssl/` base does not exist under `recipe/`): AV SFT recipe under
`2025-10-speech-llm/.../librispeech/configs/` (w2v2 variant `config_sae_2s_av_sft_w2v2_v1.py`); GRPO loop
`train_steps/sae_grpo.py` + configs `config_sae_3a_*`; psi_align `sae/psi_align.py` + `sae/psi_align_jobs.py`; HF
downloads `hf_models.py`; k-means `ssl/experiments/pretrain_two_level/kmeans.py`; LM corpus / lexicon / G2P
`i6_experiments/common/datasets/librispeech/{language_model,lexicon}.py`; gold alignments
`ssl/analysis/seg_diag.py` (eval only); external refs fairseq `examples/wav2vec/unsupervised`, ESPUM
arXiv:2310.02382, Hori et al. arXiv:1811.01690; surveys `ssl/LITERATURE_REVIEW.md`,
`ssl/SPEECH_UNIT_BPE_REVIEW.md`.

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

Publish from the highest rung that holds; the BT branch and Rung S hedge the RL and bootstrap axes respectively.
The SAE story survives either head-to-head outcome — both branches instantiate the text-bottleneck autoencoder.
