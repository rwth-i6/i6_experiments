# SAE — Speech AutoEncoder: Unsupervised ASR via a Text Bottleneck

Reconstruction-through-text unsupervised ASR, structurally following the NLA training loop
(transformer-circuits.pub/2026/nla): an **AV** (audio verbalizer, speech→text policy) and an **AR**
(audio reconstructor, text→speech-unit channel model) trained jointly — AR by supervised CE on AV
samples, AV by GRPO against the AR's reconstruction likelihood. The pair is an autoencoder over
speech whose bottleneck is a grapheme transcript; the reconstruction score is an exact discrete
likelihood, so the AV-optimal policy is amortized noisy-channel decoding:
z_hat = argmax_z p_LM(z) * p_AR(u | G2P(z)).

> Restructured 2026-08-07 (planner): this file holds live decisions, gates, and one status
> snapshot; results and history live in the SAE_*.md logs; `PLAN_3A.md` is the normative psi_align
> sub-plan. Every sub-phase carries the same five fields — Purpose / Approach / Experiments /
> Gate / Status.

## North star & hard constraints

- **North star (user ruling 2026-08-01).** Real unsupervised ASR with the **autoencoder as the
  single main mechanism**. GAN-based initialization is not the goal: an adversarial init as the
  load-bearing mechanism would demote the autoencoder to a refiner. The mainline initialization
  question is §1e (pairing-free); GAN/§1d is the working label-free fallback init (§3d hierarchy).
- **Label quarantine.** True transcripts appear in exactly three quarantined places: evaluation
  metrics (PER/WER, probes, gate measurements on dev), the §0c architecture toplines, and the §2S
  anchor arm (1 h/10 h paired seeds; its artifacts never feed the unsupervised ladder). In the
  unsupervised arm no training signal, checkpoint selection, or hyperparameter choice may depend
  on them; checkpoint selection uses dev reward + LM score only. Disclosed exception (NLA-style):
  loop mechanics and lambda ranges may be developed on §2S and reused. Amendment (USER
  2026-08-14, strengthened USER 2026-08-16 — replaces the trigger-gated form): speaker IDs,
  previously never-train (2026-07-16 ruling), MAY train and may be tried first-line;
  disclosed as supervision cost; transcripts and alignments stay absolute. Tier menu:
  `PLAN_3G.md` Z3.
- **Independence rule (GAN is not a teacher).** Admissible AR targets are *measurements of the
  audio* (deterministic transforms of encoder states), never another model's hypotheses. Passing
  the label rule does not make a target admissible. One explicit, bounded carve-out (user,
  2026-08-03): GAN/§1d output as *initialization only* in the G-track (§3d) — never as in-loop
  teacher, reward, or selection signal.
- **Framing: usability, not superiority.** Matching at lower supervision cost is the win;
  pre-register non-inferiority margins; count circularity as a cost. "Unpaired" = no paired
  audio–text; Qwen3's pretraining almost surely contains the Gutenberg books underlying
  LibriSpeech — disclosed, controlled (§4), never hidden.

## Status & priority queue (current read 2026-08-19)

**Where we are.** psi_align (§3a, `PLAN_3A.md`) is the adopted reconstruction scorer — frozen in
all live arms (sha-verified, `SAE_3A.md` §6.10). Current results, all label-free-selected:

- **10 h seed bed** (§6.5/§6.9): psi_align beats the token-LM AR on all four sets at equal
  supervision (last-epoch −3.4/−3.2/−2.8/−2.2); extended run final: shaped 9.59/12.84 dev.
- **100 h bed is the BEST bed** (§6.9/§6.10): shaped monotone to 8/8, final **6.06/10.31 dev,
  6.33/10.84 test** — same init/scorer as 10 h, nothing added but unlabeled audio. The 2S-era
  off-seed collapse was a *reward* problem, not a bed problem. Insertions FALL with no length
  term (the topology prices length natively). `recon` turns at ep2: off-seed the LM prior is the
  difference between converging and turning, and its share grows with bed size — lam values are
  per-bed, never carried over.
- **G-track 960 h: both arms HELD at sub-ep4** (§6.7/§6.10): `recon` diverges via an *inherited*
  filler token (init and scorer share the §1d pseudo-text — correlated errors are rewarded, not
  caught); `shaped` plateaus at init then slips. The suspected fix — outer scorer re-estimation
  between passes — is RUNNING since 2026-08-17 as D6-PERIODIC/GAN (`PLAN_3E1.md`; gate deleted by
  the user's label-hygiene ruling, so the arm reads as a trajectory).
- **960 h supervision-axis arm RUNNING**: theta_0 init + gold scorer + all 960 h audio, shaped
  only, 3 passes (`ReturnnTrainingJob.22Ntu7y0O6iW`, ~5.5 h/sub-epoch, ~7 days). Separates
  "more audio" from "contaminated bootstrap".
- **§0d donor swap VERIFIED 2026-08-08**: the LBS-adapted donor's theta_0' re-SFT alone is
  11.43/15.54 dev, 11.99/14.34 test vs stock 16.91/20.64, 15.28/20.78 (one-argument A/B) —
  better than anything any loop earned from stock theta_0. `_lbslm` loop arms RUNNING (100 h
  `fFp8sXTA5Wug`, 960 h 1-pass `vhyvv2waeU16`) at unchanged lam=1/T=0.7: the controlled
  donor-axis A/B vs the stock shaped arms. Gate (ii) read in §0d Status awaits the user's
  blessing.
- 2S incumbent reward program: superseded by psi_align; history in `SAE_2S*.md`.
- Reward hygiene: the LM-prior per-token mean pays for length (§6.6) — `lm_prior_norm="units"`
  is the standing fix; `len_eps` 0.4 leaves a 49 % free band if the hinge is ever load-bearing.

**Priority queue (current revision, 2026-08-19):**
1. **960 h 3-pass read** — the supervision axis at fixed audio/target/schedule.
2. **§3e.1 — D4+D5 on the best bed (USER REDIRECTS 2026-08-08/09)**: the user overrode the
   D3-plateau trigger (2026-08-08, rationale: rate-matching is a targeted heuristic, which
   D2's own read supports — d2_rate failed the paired read, d2_contrast is the conditional
   winner), then redirected BOTH phases to the best arm — the theta_0' lbs 960 h 1-pass
   shaped loop `vhyvv2waeU16` (running; gold-seed psi, 2849-pair train set, 281k-utt bed).
   One label-free fork checkpoint feeds THREE update-rule arms at matched remaining
   schedule: FROZEN continuation (the running arm, free) / CONTINUOUS JOINT psi (D5(b), the
   collapsed form, 4-6 sub-epochs, stop regardless) / GATED DISCRETE REFRESH (D4' —
   iterative psi refit on curated own-decodes anchored by the gold seed at 50 % floor).
   Specs + pre-registered gates in `PLAN_3E1.md` D4'/D5; D5(a) collapse forensics on the §3c
   run's existing checkpoints go first (cheap). Planner read 2026-08-09: D5(a) COMPLETE —
   the collapse is pure over-generation with the scorer's preference migrating to its own
   padded decodes, and pinned-policy eta flips negative after ONE sub-epoch (share-based
   monitors are blind; counts + in-run eta/CE_true are the mandatory instruments — folded
   into D4'/D5(b) as dated amendments); the in-flight G-track D4 round-1 curation read
   independently supports the park (curated picks dirtier than the anchor; refit finishes
   for the record only). Planner read 2026-08-10: fork PINNED at sub-ep 2 (the count screen
   vetoed the reward's own argmax pick); D5(b)'s faithful single-knob form is INFEASIBLE on
   the node (OOM + 2.6x step time) — re-specced to a 4-of-12 psi-CE subsample with a
   pre-registered lower-bound caveat; D4' round-1 has NO admissible curation view on this
   bed by measurement (psi filler-positive at matched WER, suspect derivation empty) —
   round 1 re-specced UNCURATED (gold anchor 50 % + one greedy decode per utterance); both
   amendments dated in `PLAN_3E1.md`, user may override. Planner read 2026-08-11: the
   4-of-12 re-spec is SUPERSEDED — the implementer's batch-halving fix ran the FAITHFUL
   joint arm; one sub-epoch of co-training is the bed's best-ever WER (5.12/9.27, beating
   the matched frozen control 6.56/11.15) and the next destroys it (17.35/21.97, insertions
   ~16x) — gate verdict pending CE_true forensics and sub-ep 3, but the shape (one good step
   then a cliff) is the strongest case yet for the gated discrete refresh (D4' round 1)
   over any continuous update rule. User-directed 2026-08-11: NEW TRACK D6 (`PLAN_3E1.md`)
   — general insertion repair, three rungs (offline price steering; corruption-trained arc
   prices; min-duration topology), goal a scorer ranking as well as the incumbent without
   the insertion cheapness. Planner read 2026-08-12: D5 CLOSED — collapse confirmed by its
   gate (sub-ep 3 = 41.8/50.9, CE_true monotone rising; new finding: the allegiance GAP,
   not the CE_true level, is the leading in-loop alarm). D6 read: rung 3 (min-duration
   topology, d_min=2) passes all four pre-registered clauses and is the scorer-swap
   candidate; rung 1 failed its bar, rung 2 refuted (learned its own negative
   distribution). D4' round-1 clause table NO WINNER (c33) — moot for production, the swap
   candidate is the D6 refit on the same corpus. NEXT FUNDED STEP, gated on the user's
   CI-vs-point pin: swap-in continuation (d6_mindur frozen) vs the frozen control at
   matched sub-epochs, confirmation read = the insertion regression shrinking; cheap
   parallel: d_min=3 refit through the same clause table. USER 2026-08-12: swap-in
   approved and extended to BOTH beds — best bed as registered; G-track via a
   min-duration refit of its own round-1 refresh recipe on its own corpus (topology
   transfers, checkpoints don't; spec in `PLAN_3E1.md` D6 Status). CI-vs-point blessing
   PENDING CONFIRMATION (user asked for the plain-words definition first); clause tables
   stay dual-reported until confirmed. USER 2026-08-12, new parallel front — real
   unsupervised without GAN: (a) §3g Z-track, from-scratch joint loop (LBS-SFT text donor +
   min-duration psi co-trained from zero, full D5 forensics; deliverable = failure-mode
   classification, taxonomy pre-registered in §3g); (b) §1f, statistics-matching init
   revisited (1b was never run — superseded, not refuted; two kill-condition prerequisites
   registered before any matching arm). Z-track (now `PLAN_3G.md`): base arm CLOSED (A)
   2026-08-13; Z2 STOPPED by user 2026-08-14 mid sub-ep 5 — coupling ladder duration ->
   density, no phone content; Z3 (tempo/noise/pitch perturbation-consistency package)
   REGISTERED AND FUNDED 2026-08-14, close-out battery on Z2's checkpoint first. Planner
   read 2026-08-15 (Z3 live in sub-ep 4/6): primary clause failing so far — dur-matched gap
   flat negative, same duration code rebuilt more purely (98.7% stem-times-k), no content
   signal; LM-prior demotion WITHDRAWN 2026-08-15 (user pushback + mechanics review — prior
   is the posterior's own term, and the code is recon-funded). Z4 (discrete psi refresh +
   within-seq repetition price + lam_len activation; lam_lm kept) REGISTERED AND FUNDED
   2026-08-15 on the user's word — build order and pre-registered gate in `PLAN_3G.md` 3g.4;
   Z3 runs untouched to its registered end as the like-for-like comparison. USER
   2026-08-14: best-bed swap-in continuation ("D4' with min duration") GREENLIT to start
   now, CI pin still pending and non-blocking (spec in `PLAN_3E1.md` D6 Status). Planner
   read 2026-08-15: that continuation is COMPLETE and passes its confirmation outright
   (4.73/9.31 vs control 6.46/11.41 at sub-epoch 10, dev-other insertions halved, 933 vs
   1964; log c39) — but the whole gain lands in the first post-swap sub-epoch and then
   plateaus. USER 2026-08-15: the periodic version REGISTERED AND FUNDED — refit the
   min-duration scorer from scratch at EVERY sub-epoch boundary on the current policy's
   decodes, per-round acceptance gate, re-forked from the same parent checkpoint so the
   finished one-refit arm is the matched control; spec in `PLAN_3E1.md` D6-PERIODIC.
   Same message, STANDING RULE: every new scorer plan carries the min-duration topology
   (d_min>=2). The
   G-track full-bed read closes
   the reward question there: ar_recon eta -0.1103 (argmax worse than random) while the
   oracle-random gap is ~6 WER points — the scorer, not group degeneracy, is the G-track
   binding defect. Requires the additive-only trainable-psi
   build (the running arm re-imports the recipe tree on resume — no executed frozen-path
   line may change). PARKED by these redirects: the G-track D4 round-1 (and with it the
   bad-init self-repair read — revive on the user's word); D3 stays parked. Still needs
   blessing: the CI-convention pin (now decides D4' round acceptance too) + gate v2 (i)
   floor-only. USER 2026-08-17: D6-PERIODIC extends to the gan-init bed —
   **D6-PERIODIC/GAN launched** (theta_0^G init, 8 rounds, per-boundary from-scratch
   d_min=2 refits on the policy's own greedy decodes; anchor-free pool and NO acceptance
   gate — both gold touchpoints deleted for label hygiene, user-directed in the
   implementer session; c37 planner-verified same day, so the one-shot G-track swap-in
   does NOT proceed and is superseded by this arm), plus a **homophone-diversity SFT
   arm** on the same bed as the one-argument A/B against it (specs, ratifications and
   pre-registered reads in `PLAN_3E1.md` D6-PERIODIC/GAN and /GAN+HOM).
   USER 2026-08-17: 1f fork resolved — entry 5 (ESPUM statistics-matching init) FUNDED
   as one contained simplicity-constrained batch; spec pre-registered in `PLAN_1F.md`;
   BPE-level ESPUM registered as conditional follow-up on a phone-level pass.
   2026-08-17: entry 5 RAN AND FAILED THE GATE, both clauses (label-free pick 0.8580
   dev-other PER vs the 0.8446 bar; audio-swap rise 0.0466 vs 0.05, close). Health
   passed (no collapse); failure is identity, not rate. Best 1f arm to date (unary
   solve 0.8809; margins tripled) but 0.44 above the memoryless ceiling. Entry 5
   CLOSED per its gate; verdict in `PLAN_1F.md` entry-5 Status; table in SAE_1f.md.
   USER 2026-08-17 (later): ruling 6 — "try your best to make a PUSM-like approach
   work; reproduction accepted" — 1f does NOT close. Reproduce-then-bridge registered
   as entry 7 in `PLAN_1F.md`: stage A reproduces the released ESPUM stack on TIMIT
   unmatched (anchor 0.473); stage B swaps one component at a time toward our setup
   (frozen segmentation / our 500-way units / LibriSpeech bed) to localize the killer;
   stage C transplants the fix and takes the unchanged arm gate. Ruling 4's TIMIT ban
   lifted for reproduction only; simplicity yields to fidelity inside entry 7.
   Implementer: TIMIT availability check (step zero), then stage A build.
   2026-08-17 (later): step zero found NO TIMIT on the cluster; USER clarifies ruling 6
   — reproduction is APPROACH-wise, no TIMIT at all. Entry 7 amendment 1: reference
   pipeline verbatim (wav2vec2 features, k-means-128, learned segmenter, relabeling)
   on our LibriSpeech seed bed, full + bigram-only arms, signature-based read
   (bigram-only worse by >= 0.10); swaps then localize on the same bed. TIMIT returns
   only as a user option if the signature is absent.
3. **LM-prior domain adaptation (§0d) — RUN AND VERIFIED 2026-08-08** (`SAE_0d.md`; replaces
   the pre-run item because the phase executed): pre-check (i) PASSED; gate (ii) read — planner
   verdict in §0d Status **awaits the user's blessing** (margin over the audio-free null is
   statistic/lam/bed-dependent; pass proposed for theta0-bed lam=1 only, no lam=0.3, no G-track).
   theta_0' re-SFT alone beats every stock-theta_0 loop result. Open: donor-axis loop reads when
   `_lbslm` arms finish; §2a-rescorer and lam_1/lam_2 recalibration deferred until then.
4. **G-track round-2 self-training** (AV^G decodes → AV^G2, ~2.5 h): the §3d operator baseline.
5. **PLAN_3A matrix wrap-up**: M4 contingency call; collapse the sub-plan when closed.
6. **§1e §2.5(d)+usage gates on the ep50 pins** — the §3d init upgrade path.
7. **G2P-equivalence ceiling** on existing rollouts.jsonl (CPU): phone-reachable vs
   orthography-only oracle-gap split.
8. **Rung repair** (Rung S 1 h/10 min): first attempt VOID (budget artifacts, `SAE_2S.md` approaches 3-4);
   extend AV budgets through the phase transition, ARs get full budget, then per-rung §2.5(d).
9. **§1d → Rung 0**: word-level self-training pipeline to completion.
10. **B0 gate table** (§3b) — role shrunk by the PLAN_3A closures; read under psi_align only if
    the target axis reopens.
11. **§1g simple weak initialization — TOP OF THE QUEUE FOR NEW SPEND (rewritten 2026-08-19
    after the USER clarified Phase 1's role).** H1 is accepted: the construction-only topology read
    selected two states for both live routes and fixed the phone duration at `p=0.23560298`; do not
    rerun it. H3 calibration is also complete and valid: the corrected 715,099-run stream selected
    full-loss ESPUM seed 0/update 30,000 on the exact 6,414/890 roles, its strict update-population
    `Q`/`B` projection is materialized, and the GH200 resume trajectory is bit-exact. H2's numerical
    engine, actual wired start, strict input parsing, evidence, and shard merge now pass, and its
    isolated 48-cell timing preflight completed cleanly. One material H2 issue remains: repair allows
    a duration to bridge deleted silence while scoring/decoding force a new duration. Propagate the
    same boundary law through repair. Eight persisted alternatives are accepted as an output-only cap
    because one-best and confidence use the complete beam. H3's CUDA, projection, and all-family final
    wiring pass; its final graph is launch-ready because no old final directory exists and the worker
    verifies runtime code hashes. Run that graph in parallel with the H2 fix, and never relaunch the
    accepted calibration graph. H4 waits for both.
    Reuse 1g.4's spectral and hard-descriptor failures; the unrun six-factor product is corrected to
    not answerable and stays parked. Reuse the fixed 1f recipes and original artifacts as provenance,
    but not as held-out inputs: both banked seeds saw the evaluation audio. The first E5 job remains an
    engineering rehearsal and cannot fire a gate. Once H2/H3 close, run the corrected phone repair,
    decoder, and score assay, then test policy-side and scorer-side SAE handoffs separately.
    Characters are the first lexicon-free candidate. Use the loop's exact BPE only for a demonstrated
    scorer-interface need. Resegmentation, repeated-speech mining, synthetic speech, and adaptive
    restart searches remain deferred. Prospective admission compares uncertainty-aware gains over identically
    treated content-free controls and then measures downstream usefulness; it does not reuse the
    historical absolute 0.05/0.05 cliff. The implementer-facing corrective package is Phase 1g.H in
    the canonical specification, `PLAN_1G.md`.

*Read 2026-08-07 (planner): the §3c 100 h replay arm FAILED its matched-compute read — ep2
23.94/29.22 vs the 10 h arm's 13.15/16.13, never beat its init, killed at ep4 (18.79→46.71). It
survives only as the artifact-backed 2S bar for the 100 h bed (§6.8).*

*Read 2026-08-07 (planner): §3e.1 fan-out closed — ranking noise refuted (recon within-group std
0.1112→0.0276), correlated bias live (the scorer rewards the inherited filler), group blindness
untested; gate v1 found gold-conditioned as instrumented and sign-blind to the filler mode →
gate v2 registered pre-verdict; ladder D0–D4 pre-registered in `PLAN_3E1.md`.*

*Read 2026-08-07 (planner, post-diagnostics): `SAE_3E1.md` verified clean against the job
outputs. Noise refuted again in-group at the operating point; bias ~70% psi_align-family /
~30% shared text (gold control pays 0.167 vs the loop's 0.243); group blindness partial and
binding (23%/9% contrast coverage) — the sampling sweep is a co-requirement, not a
contingency. The replay collapse is re-diagnosed as scorer DRIFT off the gold domain (contrast
rose 86% while CE_true crossed uniform) — gate v2 gains an absolute unit-marginal floor.
Fork presented to the user: sweep + D1/D2 (contrastive co-primary) + D3, ~20-35 GPU-h.*

*Read 2026-08-07 (planner, D1 read): D1's power check FAILED as pre-registered, and the audit
shows why no filler statistic could have separated the arms — the probe's headline insertion
discount was majority a state-count artifact (LM control unmatched in length), while the real
invariant is the lattice's ~0.03 nats/frame price per inserted emitting state, in the gold
control too. The cheap-insertion exploit is a property of the alignment lattice open to every
minimal-state word; contamination chose which word, not whether. Text repair demoted to hygiene;
insertion pricing (contrastive term; lambda reprice bounded by the audio-free share) is the
load-bearing lever. Gate v2 (i)'s improvement clause is domain-confounded by the held set's
provenance — floor-only amendment for changed-text candidates, user's blessing pending. D3 cost
corrected 9-18 -> ~85 GPU-h (planner's 100 h-bed assumption; the bar pins the 960 h bed). Two
numeric slips found, direction-neutral.*

## Resources, notation, anchors

| Item     | Value |
|----------|-------|
| Audio    | LibriSpeech 960 h (no transcripts in the unsupervised arm) |
| Text     | LibriSpeech LM corpus (`get_librispeech_normalized_lm_data()`) |
| Prior knowledge | Pronunciation lexicon + G2P (allowed); MFA gold alignments (evaluation only) |
| Encoder  | **wav2vec2-Large-lv60, layer 15** (SSL-only ckpt; decided 2026-07-18, §1c). 1024-d @ 50 Hz, per-utterance norm; units = k-means K=500 on 50→25 Hz pooled states; AV adapter stride ×4 → 12.5 Hz. Frozen for unit dumps and the GAN; AV SFT trains the transformer (conv extractor frozen); frozen inside the GRPO loop. BEST-RQ = documented negative (`SAE_1c.md`). lv60 pretrains on 60 kh LibriLight audio, zero transcripts. |
| LLM      | Qwen3-1.7B (Phases 0–4), Qwen3-8B (Phase 5 only) |
| Compute  | 4×GH200 96 GB per experiment |

**Notation.** x waveform; h = E_l(x) encoder features; u = dedup(kmeans_K(h)) unit sequence;
z grapheme transcript; phi = G2P(z), stress-free ARPAbet, one canonical pronunciation per word, no
word-boundary symbols in AR inputs. AV: p_theta(z|x) = base LLM + LoRA-A + conv
downsampler/projector. AR/scorer: p_psi(u|phi). AV-U: p(z|u), unit-token-input verbalizer
(LoRA-A'), the §3B vehicle. p_base(z): frozen adapterless base LLM as grapheme prior. T: text
corpus; T_phi = G2P(T).

**Code anchors** (relative to `recipe/`; `ssl/` = `i6_experiments/users/wu/experiments/ssl/`,
fixed 2026-08-17 — the bare `ssl/` base does not exist under `recipe/`): AV SFT recipe
`2025-10-speech-llm/src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/`
(w2v2 variant `config_sae_2s_av_sft_w2v2_v1.py`); GRPO loop `train_steps/sae_grpo.py` + configs
`config_sae_3a_*`; psi_align `sae/psi_align.py` + `sae/psi_align_jobs.py`; HF downloads
`hf_models.py`; k-means `ssl/experiments/pretrain_two_level/kmeans.py`; LM corpus / lexicon / G2P
`i6_experiments/common/datasets/librispeech/{language_model,lexicon}.py`; gold alignments
`ssl/analysis/seg_diag.py` (eval only); external references: fairseq
`examples/wav2vec/unsupervised`, ESPUM arXiv:2310.02382, Hori et al. arXiv:1811.01690; survey
numbers in `ssl/LITERATURE_REVIEW.md`, `ssl/SPEECH_UNIT_BPE_REVIEW.md`.

---

## Phase 0 — Foundations

### 0a. Representation audit

**Purpose.** Measure what the frozen encoder and unit inventory can support — the information
ceiling for any downstream mapper, and the calibration for Phase 1.
**Approach.** k-means per layer/K on ~100 h scored against MFA gold (eval-only): PNMI/purity,
CTC-probe PER, oracle-assignment PER, H(phi|u), plus a label-free utterance-separability probe.
**Experiments.** Layer × K sweep + probes; freeze the winning (layer, K, centroids) tuple.
**Gate.** Proceed regardless — values calibrate rather than block; CTC-probe PER > ~25 % or
oracle-map PER > ~45 % means the unit inventory is the constraint (shorten §1a, expect §1c).
**Status: CLOSED.** Tuple frozen; linear probe 0.145 vs oracle-map ~0.53–0.60 — the *units*, not
the encoder, cap hard assignment, which is the bound that closed §1a. Log: `SAE_0.md`.

### 0b. Phoneme/grapheme-adapted LLM (CPT)

**Purpose.** A phoneme-aware LLM for decipherment LMs, neural P2G, and phoneme priors.
**Approach.** Extend Qwen3-1.7B-Base with ARPAbet tokens; CPT on mixed phonemized / grapheme /
synthetic-P2G streams rendered from the text corpus.
**Experiments.** None run.
**Gate.** Phoneme-LM ppl stabilized; grapheme ppl regression ≤ 5 %; P2G robustness curve
(theta_P2G = max input PER with output WER ≤ 40 %).
**Status: DEFERRED (2026-07-18), never run — consumers dissolved.** Revival triggers: §2a shows
lexicon/word-LM-limited headroom, or the Phase-4 pure-phoneme arm runs. If revived, drop `<wb>`.

### 0c. Supervised topline of the exact AV architecture

**Purpose.** The architecture-gap denominator for every rung, and Delta_input = WER(AV-U) −
WER(AV), which decides whether the token-only AV-U can carry mainline experiments.
**Approach.** SFT on true LS960 transcripts through the AV path (quarantined), feature-input and
unit-input twins.
**Experiments.** The two SFTs.
**Gate.** Healthy: dev-other ≤ ~10 %. Blocker: > 14.33 % — worse than the LS100 CTC baseline
means the architecture, not unsupervision, is broken.
**Status: PENDING, unscheduled** — not run on the wav2vec2 stack; the 2S/G-track SFTs have
served the calibration role in the meantime.

### 0d. LM-prior domain adaptation to LibriSpeech text (USER-proposed 2026-08-06)

**Purpose.** Close the domain gap of the lam_1 prior (and §2a rescorer): stock Qwen3-1.7B-Base is
multilingual web text, the candidates are 19th-century prose under LibriSpeech normalization —
genre and surface convention both off-distribution.
**Approach.** Full fp32 finetune of Qwen3-1.7B-Base on lowercased `librispeech-lm-norm` at
one-pass volume, then re-run of the 10 h AV SFT on the adapted donor (theta_0') and `_lbslm`
shaped loop arms on both beds at unchanged lam=1 / T=0.7. (Replaces "short finetune, text-only
swap", 2026-08-08, because the AV checkpoint carries the 2.03 B donor weights over anything
`av_args()` names — a donor swap only lands through re-running the AV SFT, so the phase is a
retraining and its main effect appears there; `SAE_0d.md` concl. 1.)
**Experiments.** (i) Blocking pre-check: dev/test disjointness of the LM corpus by 8-gram
content scan with train-clean-100 as positive control and a >=20 % control-power clause.
(Replaces "book-level disjointness", 2026-08-08, because the norm corpus carries no book
boundaries; the positive control is what gives the scan power.) (ii) Offline re-rank of the
existing n=512 rollout dumps with lam_1 under base vs finetuned, same bed/n/G (~1 GPU-h); loop
use only after that read.
**Gate — NOT perplexity.** gap_true + spearman + the **audio margin over the audio-free null**;
if the margin shrinks, reject regardless of perplexity (a better English prior raises the null
too, and the over-generation exploit gets stronger, not weaker). After any swap: re-sweep lam_lm
per bed, keep `lm_prior_norm="units"`, recalibrate the lam_1/lam_2 balance.
**Status: LIVE — built, run, and read; verified 2026-08-08** (`SAE_0d.md`; planner audit same
day: all numbers reproduce; the "one exact pass" claim is false as run — rank-dependent shuffle
seeds gave one-pass volume over 68.4 % of distinct lines; no number invalidated; pin a common
`random_seed_offset` on any future donor iteration).
- 2026-08-08, (i) **PASSED**: dev/test 8-gram overlap 1.59–3.36 % vs the train-clean-100
  positive control's 42.98 % (13–27x separation); the eval books are not in the corpus.
- 2026-08-08, (ii) **READ by the planner** (the sweeps existed, finished, unlogged:
  `RewardShapeSweepJob.{yCfwZSr3huv7,GUbnTUM2ggiv}` stock vs `.{oaOqWCrd3ZPO,FPIKAU6TkEK4}`
  adapted): absolute ranking improves everywhere (theta0 bed, T=0.7, lam=1: spearman
  0.6684 -> 0.7132, sel_wer 0.1578 -> 0.1187, eta 0.1583 -> 0.5644) AND the audio-free null
  strengthens everywhere too (null spearman 0.4668 -> 0.5258 theta0, 0.5216 -> 0.5952 gtrack) —
  the gate's feared mechanism is real. The gate text pinned neither statistic nor lam: at the
  loops' operating point (theta0 bed, lam=1) the sel_wer and eta margins over the null WIDEN and
  spearman's shrinks by 0.014; at each column's best lam (0.3) and on the gtrack bed at every lam
  the margins SHRINK. Planner verdict, **needs the user's blessing**: pass for theta0-bed loop
  use at lam=1; do NOT chase the re-swept peak lam=0.3 (that is where the free-English share
  grows); no G-track use of the adapted prior is licensed by this read.
- 2026-08-08, the unregistered but verified main effect: theta_0' re-SFT alone is
  **11.43 / 15.54 dev, 11.99 / 14.34 test** vs stock 16.91 / 20.64, 15.28 / 20.78 — a clean
  one-argument A/B at ep50 both, psi-view deltas <= 0.06 — better than anything any loop earned
  from stock theta_0. Loop use began before the (ii) read (procedural violation, noted); at
  unchanged lam=1 / T=0.7 the running `_lbslm` arms (100 h `fFp8sXTA5Wug`, 960 h 1-pass
  `vhyvv2waeU16`) are the controlled donor-axis A/B against the stock shaped arms, so they stand.

---

## Phase 1 — Bootstrap

Policy: decipherment first (novelty-preserving), PUSM and GAN as gated fallbacks; resolved
2026-07-18 with the GAN passing on wav2vec2 and the encoder decision falling out of it.

### 1a. Decipherment program

**Purpose.** Classical LM-guided decipherment of unit streams — the primary, adversarial-free
bootstrap the paper's story wants.
**Approach.** Hard CDF/ICM assignment; fertility-HMM channel model trained by Baum–Welch; OT
embedding-cloud init; unsupervised LL selection with a restart-agreement diagnostic.
**Experiments.** Ran on the BEST-RQ-era units with multiple inits/restarts.
**Gate.** dev-other PER ≤ 50 % under the §1.0 unsupervised ppl-selection metric.
**Status: CLOSED permanently, on a bound** — decipherment LL anti-aligned with PER, and hard-unit
decipherment is capped by §0a's oracle-map ceiling on *either* encoder; do not revisit. `SAE_1a.md`.
AMENDED IN SCOPE 2026-08-18 (planner, `PLAN_1G.md`; replaces the unqualified "do not revisit",
because both legs were read back to `SAE_1a.md` and neither covers the discrete case): the closure
stands as written for CONTINUOUS generative maximum likelihood over features — the configuration that
produced the anti-alignment — while the DISCRETE channel decoded through a language model on the
pooled stream is reopened as §1g, since the ceiling leg bounds memoryless lookup decodes only and
`SAE_1a.md` approach 4 measured the discrete objective WELL-aligned and init-limited. The only
real-data discrete evidence is one row — Gromov-Wasserstein init collapsing on the RAW stream — and
§1g re-runs 1a's own anti-alignment test on the pooled stream before funding any fit.

### 1b. Fallback A — PUSM

**Purpose.** Positional/skipgram distribution matching — the escalation aimed at unbroken
permutation symmetries, decipherment's known failure mode.
**Approach.** ESPUM recipe: frame one-hots → CNN generator + boundary segmenter, unigram +
skipgram L1 objectives, length-matched text batches.
**Experiments.** None run.
**Gate.** Same §1.0-metric selection, PER ≤ 50 %.
**Status: NOT EXERCISED** — superseded by 1c's pass; no spec beyond the Approach line
survives (pre-restructure backup deleted) — re-derive from the ESPUM paper if ever reopened.

### 1c. Fallback B — wav2vec-U 2.0 GAN

**Purpose.** Feature-level distribution matching that bypasses the unit inventory — the fallback
matched to §0a's "units are the constraint" verdict, and the encoder-discrimination instrument.
**Approach.** Faithful fairseq w2vu2 reproduction (rVAD trim, batch-normed features, CNN
generator to ~12.5 Hz, discriminator + gp/sp/pd/ss terms); selection by the §1.0 unsupervised
metric only.
**Experiments.** Full grid × seeds on both encoders, identical pipeline.
**Gate.** §1.0 metric with a non-empty converged filter set.
**Status: PASSED on wav2vec2** — 0.168 dev-other PER ppl-selected (paper anchor 0.136) vs BEST-RQ
flat at 0.75–0.92; this run decided the encoder. Tables and mechanism: `SAE_1c.md`.

### 1d. Rung 0 self-training

**Purpose.** The standard-recipe baseline (Rung 0) from the winning bootstrap — the number the
loop must beat.
**Approach.** WFST pseudo-label decode → CTC finetune of wav2vec2 on pseudo-labels (HMM-GMM stage
skipped, no-Kaldi route), last checkpoint, lexicon + 4-gram word decode.
**Experiments.** CTC student and word decode done; the completed Rung-0 pipeline read is queue 9.
**Gate.** §1.0-metric selection throughout; Rung 0 = word WER of the final system.
**Status: student stands, pipeline unfinished** — 0.172 dev-other phone; word decode 17.96/21.87
WER (`output/sae/1d/word_wer.json`). `SAE_1d.md`.

### 1e. Pairing-free initialization (mainline; USER priority 2026-08-01)

**Purpose.** Initialize AV and AR from unpaired audio + text only — no GAN, no decipherment; if
it clears its gates it replaces the G-track's GAN init and restores full independence.
**Approach.** SFT on length-paired / random-paired / audio-continuation pseudo-pairs; any 1e loop
runs joint AR with the lam_1 + lam_2 anchors mandatory (no seed pins the text side).
**Experiments.** SFT screens done at gold budget; §2.5(d) + usage gates on the ep50 pins are
queue 6. Kill-switch if all arms gate flat: non-adversarial output-distribution matching, then
§1c/§1d stays the init of record.
**Gate.** §2.5(d) on vanilla-unit rollouts before any loop compute.
**Status: UNDECIDED** — the length-vs-random contrast is real (209.8 vs 307.0 dev-clean) but AR
CE is length ≈ random (Δ0.019 nats); whether the reward carries rank is exactly the pending
§2.5(d) question. `SAE_1e.md`.

### 1f. Statistics-matching initialization, revisited (USER-directed 2026-08-12)

**Purpose.** Answer whether FIXED-statistic (frequency / n-gram) distribution matching can
replace the GAN as the bootstrap. Honest history: 1a closed DECIPHERMENT (generative ML/EM over
features) on measured evidence — LL anti-aligned with PER, §0a unit-information ceiling — but 1b,
the moment-matching form the user is asking about, was NEVER RUN: it was superseded when the GAN
passed, i.e. dismissed by supersession, not by evidence. Reopening 1b's question is legitimate;
1a's "do not revisit" covers EM-decipherment only and stands.
**Approach.** Two cheap prerequisites BEFORE any matching run, each a registered kill condition:
(i) the §0a information audit re-run on the CURRENT enc50 unit inventory (oracle-map PER,
H(phone|unit)) — the old inventories capped ANY unit-level token mapping at PER 0.53-0.63, and a
static matching init cannot beat the oracle map by construction; if the current units are
similarly capped, the arm moves to FEATURE level (the 1b/ESPUM shape: low-capacity segmental
generator over features, unigram + skipgram / n-gram objectives) or dies; (ii) the
channel-structure read from 1a c6, measured not assumed: correlation of the unit co-occurrence
graph with the phone-bigram graph (on the old units, real co-occurrence was acoustic-flicker- and
coarticulation-dominated — the phone-LM signal the matcher needs was swamped; simulated units
whose co-occurrence mirrored the bigram recovered 0.97 of the map, real units 0.146).
**Experiments.** Prerequisites first (CPU-cheap, existing MFA/probe machinery); the matching arm
itself only if both clear. Literature-pinned modern form (planner scan 2026-08-12): ESPUM-style
positional-unigram + n-skipgram L1 matching (Wang/Hasegawa-Johnson/Yoo, ICASSP 2024,
arXiv:2310.02382 — small-batch-stable where the GAN diverges; positional unigrams are
load-bearing, bigrams-only collapses; 4/5-grams HURT), not Empirical-ODM's coverage-KL (its
corpus-frequency-inside-the-log needs ~50k-token batches, arXiv:1812.09323). Two theory anchors
map onto our prerequisites: identifiability needs (a) the channel to factorize as the model
assumes — prerequisite (ii) tests exactly this, and 1a's flicker/coarticulation finding is that
condition failing — and (b) spectral genericity of the text statistics (Wang et al. ACL 2023,
arXiv:2306.07926; Yang/Schlueter/Ney 2026, arXiv:2603.02285). Reference verdicts (2026-08-16,
replaces the verify-FLAG of 2026-08-12; all three flagged references verified first-hand):
coarse/syllable granularity ADOPTED as the default target unit (the 2510.03639 ablation
stands) while that paper's pipeline and its bootstrap claim are REJECTED; the 2306.07926
closed-form estimator enters only ridge-regularized and sigma_min-gated (never run on real
speech); 2603.02285's rank condition is kept, its training loss REJECTED as 1a's decipherment
likelihood in gradient form. Details and candidate ladder: `PLAN_1F.md`.
**Gate.** REPLACED 2026-08-16 (was: 1b's dev-other PER <= 50 % bar, registered 2026-08-12 —
no matcher result existed, so replaceable) because the USER re-set the criterion: the single
requirement is that the init be BETTER THAN RANDOM/UNPAIRED initialization. Operational form:
dominate the strongest content-free nulls — a marginal-matched random unit-to-phone map, and
the 1e pseudo-pair init — on plain PER as scored (labels eval-only) AND the audio-swap
content-dependence control; margin pre-registered in `PLAN_1F.md` before the first matcher
read. The prerequisite kill conditions stand as registered; verdicts in Status.
**Status: REGISTERED 2026-08-12.** Awaiting prerequisite runs; literature scan DONE (planner,
same day — citations inline above). 2026-08-16 planner fan-out (five-agent workflow, 28
candidates screened): design space pinned in `PLAN_1F.md` — the prerequisite screen becomes a
per-representation battery (raw / deduped / segment-pooled / Brown-K100 / unit-BPE; adds
sigma_min(P_X), Laplacian eigen-similarity, spectrum overlay; calibrated on the simulated-unit
generator), plus a six-entry candidate ladder each with a pre-investment kill-test — new
front-runners are fixed-core tri-factorization (fit only the emission matrix against a
text-pinned phone-bigram core; method-of-moments, not EM) and ridge positional-unigram least
squares (first real-speech run of the 2306.07926 estimator). The screen battery remains the
first fundable step; nothing above relaxes the registered gate or kill conditions.
2026-08-16 (later): prerequisites RAN (`SAE_1f.md`). Kill (i) FIRED — oracle-map PER 0.832
dev-other vs the 0.50 bar — but localized to over-segmentation (ins 0.692; subs 0.132 and
PNMI 0.682 both program-best), so the fork is staged: the battery's pooled rows on the
current codebook decide the representation, feature-level ESPUM is the fallback, death only
if all cap. Kill (ii) measured: the observable graph carries bigram signal (PMI spearman
0.373/0.370 vs floor ~0.215, ceiling ~0.41) but the matcher's own TV objective separates
truth from no-correspondence by only 9-11 % relative — a separability bar for
transition-consuming matchers is now pre-registered in `PLAN_1F.md`; transition-free entries
are unaffected and lead the queue. Same day the USER ruled: simplest-possible init (ladder
re-ranked — pooled-rows screen first, then fingerprint assignment, then the ridge solve;
ESPUM last) and the gate replacement recorded above. Detail: `PLAN_1F.md`.
2026-08-16 (battery): kill (i) CLEARED at the unit level — data-driven segment pooling
passes the bar on every rung (`seg12.5` 0.414 / `seg16` 0.452 / `seg9` 0.481 dev-other vs
0.50, program-best ceilings), the feature-level fallback is not exercised, and inventory
coarsening at fixed rate is catastrophic (`brown100` 1.152) so the 500-way codebook stays.
Entry 2 (ridge positional-unigram) CLOSED by its own sigma_min gate, structurally (0 on all
pooled rows; a simulated perfect channel also reads 0). The kill-(ii) separability bar is
VOID AS MEASURED (the real stream beats its seg_swap ceiling on pooled rungs — coarticulation
inverts the control), so entries 1/4 stay parked behind the transition-free entries with no
post-hoc replacement bar. Arm-gate margin pre-registered in `PLAN_1F.md` before any matcher
run: beat min(random-map, 1e-pseudo-pair) by >= 0.05 dev-other PER AND degrade >= 0.05 under
audio-swap. NEXT FUNDABLE STEP: entry 3 (fingerprint assignment) + its two nulls on
`seg16`/`seg12.5`/`seg9`, entry 6 kill-test on `ubpe12.5` — CPU-cheap; a funded init later
needs the pooling pass on the assign-side shards. Verdicts: `PLAN_1F.md`; rows: `SAE_1f.md`.
2026-08-16 (USER ruling 3): the screens run TWO text-side arms per the 3a section-5c
pattern — phone-level reference (statistics from T_phi) vs lexicon-free (text-BPE-512 /
frequent-word statistics from the raw corpus; entry 6's function-word kill-test is that
arm's precondition) — and the gap is reported as the measured price of the lexicon. The
phone arm's extra lexicon touchpoint (pseudo-labels need lexicon + word decode to become
SFT text; the lexicon-free arm outputs text directly) is disclosed in its supervision
cost. Gate and margins unchanged, applied per arm. Detail: `PLAN_1F.md` ruling 3.
2026-08-16 (entry 3): the fingerprint assignment FAILS the arm gate on every
representation — best margin over the stronger null +0.015 vs the registered 0.05, and
audio-swap movement at the random null's own level, i.e. content-free by the control;
both of its own kill-tests also fail. Measured cause: the matchable transition-free
statistics rank the true phone only ~2x above chance — too diffuse for a 39-way
assignment. NOT FUNDED (licenses not funding, not "could never work"). Remaining
simple-family step: entry 6's function-word kill-test + ruling-3's lexicon-free arm;
entries 1/4 stay parked; entry 5 (ESPUM, GPU training, the one entry with published
real-speech evidence) stays last — funding it after entry 6 is the USER's call.
Verdict detail: `PLAN_1F.md`; table: `SAE_1f.md` approach 4.
2026-08-16 (USER ruling 4): no TIMIT bed — the staged TIMIT reproduction proposed for
entry 5 is declined; entry 5, if ever funded, is judged directly on LibriSpeech against
the arm gate. NEXT STEP (funded, dispatched to the implementer): entry 6's function-word
kill-test on `ubpe12.5` + the ruling-3 lexicon-free text-side screens — CPU-cheap, on
existing artifacts; gate and margins as registered.
2026-08-16 (later): entry 6's kill-test CLEARS (verified) — the lexicon-free arm keeps
its precondition, with the recorded scope that the signature is positional only and an
utterance-onset acoustic confound remains (eval-only oracle read on the hitting units
green-lit to resolve it). Ruling-3 screens launched (4 LexFreeMatchJob, one per
representation); frame ratified except the oracle ceiling, overturned to the candidate's
own restricted map space — re-run required, gate reads unaffected. Verdicts and both
frame rulings: `PLAN_1F.md`; numbers: `SAE_1f.md` approach 5, conclusions 19-23.
2026-08-16 (later still): onset control DONE and verified — the confound resolves
LINGUISTIC on `seg12.5` (unit 403 is a genuine THE-like unit) while `ubpe12.5`'s
headline hit was a missed all-silence unit (proxy defect recorded; one genuine YOU-like
hit remains), so the lexicon-free arm's precondition stands on direct evidence; hit
counts corrected under `SAE_1f.md` conclusions 19/22, amended verdict and the proxy-
defect consequence for the running `ubpe12.5` screen in `PLAN_1F.md`.
2026-08-17 (ruling-3 batch amendments, planner-verified): the `ubpe12.5` open-ceiling
screen died at wall clock unwritten — the restricted re-runs (queued) carry BOTH
ceilings from one pass, with a pre-registered bit-for-bit reproduction check against
the finished seg runs; the `ubpe12.5` STREAM itself was found budget-stopped at a
default (8000 merges, measured 14.08 tok/s vs the 12.5 target) — no rebuild this
batch, the matched-rate contrast with `seg12.5` is retired, a true-12.5 rebuild is a
conditional follow-up; the words text side is unreachable at 2.8 words/s on every rung
(screened at each rung's floor with the mismatch printed, pre-registered as a frame
limitation). Rulings and the resume-change ratification: `PLAN_1F.md` 2026-08-17.
2026-08-17 (ruling-3 batch close, planner-verified): the screens FAIL the arm gate
in all twelve cells (best dev-other M2 0.0252 vs 0.05; M1 negative in 10 of 12) —
the lexicon-free arm is NOT FUNDED and the phone-reference side fails the same
gate, so NO 1f init is fundable from the screens run to date. The words cells price
a rate-mismatched arm (no rung reaches 2.8 words/s, pre-registered), but the
rate-matched cells fail equally, so no retry is proposed. All pre-registered
determinism checks passed bit-for-bit across three job generations; one hash-label
swap in the log Catalog (seg12.5/seg16 audio jobs) is being corrected — labels
only, no numbers. USER FORK NOW OPEN: fund entry 5 (last unkilled ladder entry,
LibriSpeech-direct per ruling 4, raised bar), register a new screen for parked
entries 1/4, or close 1f. Detail: `PLAN_1F.md` amendment (7).
2026-08-17 (USER ruling 5): the fork resolves — entry 5 FUNDED, with a second
instruction that the whole process stay as simple as possible. Spec registered same day
pre-run (`PLAN_1F.md` entry-5 funded batch): the ESPUM reference mechanism (ICASSP 2024,
verified first-hand including its released code) with seven traceable deviations — fixed
measured boundaries instead of the learned segmenter, our 500-way one-hot units, the
ruling-3 silence convention, LABEL-FREE selection (the released config selects by error
rate against test references — quarantine-incompatible, so the deviation is mandatory),
the screens' eval protocol so the banked seg12.5 nulls price the candidate directly (M1
bar: dev-other PER <= 0.8446), 3 seeds plus the bigram-only collapse control as the
health pair — one contained batch on the 20.5 h seed stream. Honest anchor: the paper's
UNMATCHED-text TIMIT column (PER 0.451-0.473); LibriSpeech is unanchored, the
research-bet framing stands. A failed gate closes entry 5 and returns 1f with no
unkilled entry. Post-close defect disclosed (`PLAN_1F.md` (7b)): the ruling-3/entry-3
text statistics sampled only the first 60.6% of the alphabetically sorted corpus —
nulls and candidates shared the sample so the verdicts stand; a standing full-coverage
sampling rule is registered and entry 5 pins the proven full-coverage sample.

### 1g. A simple weak starting point for the SAE loop (rewritten 2026-08-19; sub-plan `PLAN_1G.md`)

**Purpose.** Produce a label-free, audio-dependent seed that gives the speech autoencoder loop a
better starting point than an identically treated content-free control. Phase 1g does not need to
solve ASR by itself.

**Approach.** Estimate `P(audio unit | text symbol)` and decode it jointly with a
text language model. Test phones first because two real phone seeds, two controls, and an oracle
already exist; this is a reference and mechanics check that pays for a pronunciation lexicon.
Characters are the first primary lexicon-free route. Use the loop's exact BPE vocabulary only when a
direct scorer handoff requires it. The channel may seed either the audio-to-text policy through
pseudo-transcript cross-entropy training or the reconstruction scorer directly; test those paths separately before
combining them.

**Experiments.** Reuse 1g.0's label-free one-segment rejection; keep its old full-dev and
gold-duration cells diagnostic. Fit each prospective shared duration and recompute the live
dependence read on update audio only; choose the smaller admissible one-state or two-state form.
Reuse the 1g.4 spectral/hard-descriptor not-funded verdicts. The unrun six-factor product is not
answerable and stays parked. First run the corrected phone assay on construction-only
rebuilds of the proxy-silence masks, ESPUM, fingerprint, and both controls, with common held-out data, preprocessing,
empirical channels, full text coverage, nested local decoder, and fixed repair counts. Keep the
original full-bed seeds as transductive provenance rows only. Validate that the fitting and selection
scores follow speech content. Then run separate phone policy-side and scorer-side handoffs and start
the character route once one is valid; a combined phone loop is optional and must not delay
characters. The lexicon-free candidate receives the fixed combined test. Preserve one-best text,
alternatives, posteriors, confidence, per-utterance gate statistics, donor tables, and uncertainty
inputs. Full corrective handoff: `PLAN_1G.md` Phase 1g.H.

**Gate.** From now on, separate two questions. A seed is content-bearing when paired, uncertainty-
aware comparisons show that it beats treated content-free controls under both plain error and
same-speaker audio-swap dependence. A separate policy or scorer handoff can identify a promising
component; a usable Phase-1 initialization requires the fixed combined path to beat its matched
controls without materially degrading from its start. A failed path-specific positive control makes
that assay unresolved, not the seed negative.
The historical 1f (0.05/0.05) failure remains recorded but is not the future admission cliff.
Phone results validate mechanics. The phone-versus-character difference bundles several design
changes, including pronunciation-lexicon cost; only a lexicon-free result supports the main claim.

**Status.** **Active; H1 and H3 implementation accepted, one H2 algorithmic fix remains
(2026-08-19).** The first E5 job remains exploratory and non-decisive. The accepted H1 artifact
freezes the split, masks, two-state topology, and phone `p=0.23560298`; no further H1 run is required.
H3's final-refit graph is ready to run in parallel with propagation of the deleted-silence boundary
through H2 repair. The H2 timing preflight completed cleanly and must not be rerun.
H4--H6 remain blocked until the H2 law is consistent and H3 final artifacts exist. Details and all
future gates: `PLAN_1G.md`. Evidence: `SAE_1g.md`.

---

## Phase 2 — Warm start (SFT)

### 2a. Offline decode → Rung 1

**Purpose.** The LLM-decoding claim with no CPT: Rung 1.
**Approach.** Frozen-base Qwen3 n-best rescoring of §1d's WFST lattices (kappa from a
dev-disjoint unsupervised sweep), with a 4-gram-prior-only control separating "better prior"
from Gutenberg memorization.
**Experiments.** Rescoring vs same-lattice 4-gram 1-best; pending on Rung 0 (queue 9).
**Gate.** Rung 1 ≤ WER of the 4-gram WFST decode of the *same* lattices.
**Status: PENDING.**

### 2b. AV SFT → Rung 2

**Purpose.** Distill Rung 1 pseudo-labels into the feature AV (Rung 2); the AV-U twin is the §3B
init.
**Approach.** §0c recipe on (audio → pseudo-transcript) pairs, LoRA-A; AV-U twin with LoRA-A'.
**Experiments.** The two SFTs; pending on Rung 1.
**Gate.** Rung 2 ≤ Rung 1 + 1 abs AND dev insertion rate ≤ 1.5× the teacher's.
**Status: PENDING** — the G-track AV^G (13.89/18.34 from §1d labels at 960 h) is evidence the
distillation step itself works.

### 2c. AR SFT

**Purpose.** Warm-start a text→units channel model for loop use.
**Approach.** Boundary-free phonemes → deduped units CE (LoRA-B), no speaker/F0 conditioning.
**Experiments.** Pending in-phase; in practice superseded by psi_align training (§3a) for the
reward role.
**Gate.** The old ΔCE usage screen is superseded — the binding screen is §2.5(c)/(d) (measured
2026-07-17: full-history ΔCE ≈ +0.005, a target wall, which started the scorer program).
**Status: SUPERSEDED for the reward; retained only if an LLM-AR is ever revived in-loop.**

---

## Phase 2S — Semi-supervised anchor arm (quarantined)

**Purpose.** Loop-validity control with a known-good init, validation of the validators, the
Rung-S hedge, and the supervision-equivalence framing. Seed artifacts never enter Rungs 0–4;
mechanics and lambda ranges transfer, disclosed.
**Approach.** Paired-seed SFTs (1 h/10 h) → full GRPO loop vs self-training from the *identical*
seed, plus the shuffled-reward and frozen-vs-joint controls.
**Experiments.** 10 h loop + controls done; the 1 h/10 min rung repair is queue 8 (first attempt
VOID — budget artifacts, not seed-size verdicts).
**Gate.** Loop beats identical-seed self-training by ≥ 0.5 dev-other, unsupervised-selected.
**Status: role complete at 10 h.** Gate PASSED (+1.24); shuffled control DECISIVE (ep1 202.54 vs
14.74 — the reward is load-bearing); joint AR beats frozen run-to-completion; the measured
0.27-nat information cap of the token-LM reward drove the §3a escalation. Logs: `SAE_2S*.md`.

---

## Phase 2.5 — Go/no-go instruments

**Purpose.** Cheap verdicts before any RL compute; instrument (d) is decisive for every new
scorer, target, or init.
**Approach.** (a) rerank test and (c) graded-corruption ladder (SNR ≥ 1) as pre-screens — (c) is
a known-optimistic synthetic proxy; (b) deadlock test superseded by (d); **(d) reward-RANK
probe**: replay the loop step on real theta_0 rollouts (G≈12, T ∈ {0.3, 0.5, 0.7}; T=1.0 logged,
never evidence).
**Experiments.** Run (d) for every candidate before loop compute; calibrate any new diagnostic on
the §2S paired-init models first (a failure there indicts the instrument, not the signal).
**Gate.** Within-group spearman with CI > 0, gap_true = r(z_true) − mean r(z_i) > 0,
reward-selected WER ≤ group mean. Read discipline (2026-08-05): **absolute-eta bars withdrawn** —
same-bed/same-n/same-G, gap_true + spearman lead, plus the audio margin over the audio-free null.
**Status: in active service** — governed the §3a adoption, the §3d funding decision, and gates
§1e next.

---

## Phase 3 — Joint RL loop (Rung 3, the central claim)

### 3a. Reconstruction scorer — psi_align

**Purpose.** A reward whose score moves with transcript quality on real rollouts — where the
teacher-forced AR (text-blind on fine units) and the CI-given-text LLM scorer (family-capped)
failed; alignment is the missing expressivity.
**Approach.** Conditional neural HMM over the text symbol string: ~11 M from-scratch
bidirectional text encoder, per-state categorical emissions over the 500-unit inventory, 3-way
{self-loop, advance, skip} transitions, exact forward-sum p(units, T | symbols); CI given (text,
alignment) with the alignment marginalized, unit history structurally absent, length priced
natively. Frozen in-loop by construction (`train_steps/sae_grpo.py:153` forces it).
**Experiments.** G0–G3 gates + the scorer×target matrix (§5b) and text-side axis (§5c), all per
`PLAN_3A.md`; remaining: M4 contingency call (queue 5).
**Gate.** G1 usage gate and G3 re-rank as pre-registered in PLAN_3A §6 (same-bed/same-n/same-G,
audio margin over the audio-free null).
**Status: ADOPTED.** G1 + G3 passed decisively 2026-08-05; §5c BPE 12/12 cells — carry-forward
text side `bpe512_cps15` (lexicon-free, zero OOV); M2 CLOSED (discrete k-means-500 stands);
substrate CLOSED (post-adapter 12.5 Hz); frozen-scorer state sha-verified across all six arms.
Normative: `PLAN_3A.md`; log: `SAE_3A.md`.

### 3b. Reconstruction target

**Purpose.** Choose what the scorer reconstructs; the target's information content bounds every
reward (the 2S collapse root cause was a 0.27-nat target-information cap).
**Approach.** Candidate ledger gated by §2.5(d), select the finest that passes; admissible
targets are measurements of the audio only (independence rule — the withdrawn GAN-phone stream
is the precedent).
**Experiments.** The remaining B0 gate table is queue 10, read under psi_align only if the target
axis reopens.
**Gate.** Same-set §2.5(d) comparisons against the incumbent stream.
**Status: SETTLED at avunits k500** by the PLAN_3A M2/substrate closures. History:
`SAE_2S.md` approach 13 (conclusions 23-25).

### 3c. Seed-replay

**Purpose.** Stabilize the quarantined 2S arm by keeping seed supervision in the objective (the
Hori/TTE mechanism).
**Approach.** Mix seed paired CE into AV/AR objectives; lambdas calibrated against within-group
monitors (measured lambda_av 0.466 / lambda_ar 0.996 — the AR term near-duplicates in-loop
ar_ce, so supervision enters via AV).
**Experiments.** The 100 h replay arm ran to ep4.
**Gate.** Matched-compute read: ep2 vs the 10 h arm's final 13.15/16.13.
**Status: FAILED its read (2026-08-07)** — ep2 23.94/29.22, never beat its init, 46.71 by ep4;
survives as the artifact-backed 2S bar (§6.8). Replay anchors to a good seed — the wrong anchor
for the bad-init regime (§3e.1); admissible in the 2S arm only.

### 3d. G-track — GAN-init fully-unsupervised 960 h track

**Purpose.** The operator question: does the autoencoder loop beat plain self-training as the
refinement operator, from the same label-free init?
**Approach.** Init = SFT on §1d-student pseudo-labels under the **init-only carve-out** (AV:
audio→pseudo-text; scorer conditioning only — targets stay audio-derived units). Arms: (1) real
reward, (2) shuffled reward (= iterated pseudo-labeling, the built-in pivot result), (3) no-loop
baselines; (1)−(2) = reward contribution, (2)−(3) = distillation contribution. Init hierarchy:
§1e (goal) → GAN/§1d (working fallback) → 10 h seed (2S only).
**Experiments.** AV^G and psi_align^G built and gated; both 960 h loop arms ran to sub-ep4;
round-2 self-training (AV^G2) is queue 4 — the operator baseline either way.
**Gate.** §2.5(d) at the init before loop compute (passed under psi_align^G; the earlier AR_G
attempt declined funding on the same read).
**Status: both loop arms HELD at sub-ep4** (§6.7/§6.10) — `recon` diverges through an
*inherited* `to` filler (init and scorer share byte-identical §1d pseudo-text, so correlated
defects are rewarded); `shaped` plateaus then slips. Standing suspicion: the frozen scorer
cannot leave the shared bad prior; the admissible fix is outer-EM re-estimation between passes,
gated on the §3e.1 rule — deferred, user's call.
2026-08-17: the deferral ENDS — the outer re-estimation runs as `PLAN_3E1.md` D6-PERIODIC/GAN
(per-boundary from-scratch d_min=2 refits on the policy's own greedy decodes; the §3e.1
acceptance-gate clause is DELETED on this track by the user's label-hygiene ruling — a gold-read
gate selects what trains the next leg, and no annotation may train or select here). The
homophone-diversity SFT arm rides the same bed as its one-argument A/B.

### 3e. Reward and update protocol

**Purpose.** The loop itself: reward composition, decoupled updates, monitoring, selection.
**Approach.** Reward per sampled transcript z (utterance units u, duration D):

    r(z) = (1/|u|) log p_psi(u | BPE_states(z))   reconstruction (psi_align forward-sum;
                                                   graphemic bpe512 sub-states, 1.5 chars/state,
                                                   SIL at word boundaries)
         + lam_1 * lm_prior(z)                     LM prior, p_base; lm_prior_norm="units"
         - lam_2 * KL_hat(z)                       anchor to theta_0 (frozen SFT snapshot)
         - lam_3 * length_hinge(n_chars(z), D)     chars/s hinge (nu 14.55, len_eps 0.4);
                                                   lam_len 0 in the G-track arms, 0.5 in Z4

(Formula corrected 2026-08-17, replaces the G2P form — because the live reward contains NO G2P
anywhere, verified at source: psi re-encodes the decoded string under its own graphemic BPE
(`psi_scorer.py:141-146`, `psi_align_jobs.py:87-104`; the "phones" branch exists but no live arm
sets it), the hinge is `len(decoded_string)` (`train_steps/sae_grpo.py:205-212`; `reward.py:14-15`
documents the deviation), and the old lam_4 OOV term is unwired dead code that raises if enabled.
The G2P map — first pronunciation, stress-free — survives in probes and analyses as phi = G2P(z),
NOT in the reward. Consequence, load-bearing for §3e.1 D6-PERIODIC/GAN+HOM and queue 7: the
orthographic channel is LIVE — homophone spellings are NOT reward-invariant; the scorer carries a
per-state price on orthographic length (the minimal-state exploit's substrate) plus any
spelling-specific emissions it learned.)

`lm_prior_norm="units"` because the per-token mean pays for length (22:1 trade measured, §6.6).
Updates are decoupled
(NLA shape): sample G=8–12 at the bed's T; scorer frozen under psi_align (any update goes
through §3e.1); AV by GRPO with group-normalized advantages (the group shares one utterance, so
speaker/prosody cancel). As built: RETURNN `train_steps/sae_grpo.py`, one ReturnnTrainingJob per
arm, per-sub-epoch recogs.
**Experiments.** Lambda sweeps per bed at ≤ 100 h, never 960 h; lambdas are bed-size-dependent
(§6.9/§6.10: off-seed the prior decides converge-vs-turn and its share grows with bed size) —
recalibrate against the within-group-std monitors, never carry values across beds.
**Gate.** Checkpoint selection by dev reward + LM score only; monitor reward components, ins/del,
and within-group std separately; a degrading run is reverted, not compounded.
**Status: LIVE** — 10 h and 100 h rounds complete, 960 h 3-pass running (queue 1).

### 3e.1 Scorer trainability without collapse (USER-directed 2026-08-06; sub-plan `PLAN_3E1.md`)

**Purpose.** The bad-init north star needs a scorer that repairs itself in-loop; a scorer that
must start good imports the bootstrap problem into the reward. Both endpoints fail: frozen
Goodharts (2S) or cannot leave a contaminated prior (G-track), and training on the policy's whole
sample set collapses the scorer — the trainable 100 h replay arm (`freeze_ar=False`) went
18.79 → 46.71 by DRIFT off the gold domain, not text-blindness (re-diagnosed 2026-08-07,
`SAE_3E1.md` c1-2: its text contrast rose 86% while CE_true crossed the unit marginal and
uniform) — so the update *rule*, not trainability, is the question. Attribution of that
collapse to co-training itself is REOPENED 2026-08-09 (user question): no frozen-scorer
control has ever run on that bed and the 10 h matched pair went the other way — `PLAN_3E1.md`
D5 carries it: (a) forensics on the collapsed run's own checkpoints, (b) a USER-redirected
joint-psi control arm on the current best 960 h bed (the running frozen arm is its matched
control).
**Approach.** The evidence splits the failure three ways (`PLAN_3E1.md`): ranking NOISE is
refuted (twice — recon within-group std, and in-group spearman ~0.50/0.56 at the loop's own
operating point); correlated BIAS is confirmed but ~70% is a psi_align FAMILY property (the
gold-text control also pays for the filler, beta 0.167 vs 0.243 — only the differential is
contamination, `SAE_3E1.md` c4); GROUP BLINDNESS is measured partial and binding (23%/9%
contrast coverage — ~77% of "to"-groups unsteerable for ANY scorer, c6). Admissible shape: discrete gated OFFLINE refresh rounds at sisyphus-job granularity
— no in-loop psi channel exists (`grpo/psi_scorer.py:153`), the loop always runs on the last
accepted frozen scorer, so rollback is free; ladder D0–D4 (discriminator → probes → round-0 text
repair without co-training → frozen-repaired control arm → gated outer refresh) pre-registered in
`PLAN_3E1.md`. Old candidates posterior-weighted CE and emissions-pinned text refresh are
withdrawn (published collapse mode; not a real parameter partition — one trunk feeds all heads).
**Experiments.** Diagnostics, the coverage/steerability read, and D1 are DONE and
audit-verified 2026-08-07 (`SAE_3E1.md`). D1's pre-registered power check FAILED: no filler
statistic separates the contaminated scorer from the gold-text control, and the audit shows the
probe's headline discount was majority a state-length artifact — the lattice charges ~0.03
nats/frame per inserted emitting state in every scorer, so the cheap-insertion exploit is open
to any minimal-state word (`PLAN_3E1.md` D1 verdict). Direction change: text repair is hygiene;
mechanism-level insertion pricing (contrastive term, bounded lambda reprice at prior-variance
share <= ~46 %) is primary. D2 candidates are mid-training; the T=0.9 presumptive point is
withdrawn (oracle degrades 0.107 -> 0.150, c11) — the (scorer, lambda, T) point is selected
jointly on the D0 dump with the D2 winner. Next, the user's call: length-matched probe variant +
D2 admission reads (~1-2 GPU-h), then D3 at its corrected cost (~85 GPU-h as wired on the 960 h
bed, ~42 trimmed to 2 arms x 2 sub-epochs); D4 stays unfunded.
**Gate v2 (replaces the two-sided gate, 2026-08-07 — amended BEFORE any verdict was read against
v1, because v1's `text_explained_loo` arm is gold-conditioned as instrumented
(`config_sae_3a_enc50_units_v1.py:233-243`) and has the wrong sign against the filler mode, and
its held-NLL arm is a per-round redraw, not comparable across rounds).** Accept a scorer update
only if, label-free on a frozen external held pair set outside the candidate's curated pairs:
held unit NLL improves vs the last accepted scorer AND `text_explained_loo` ≥ the pre-loop floor
AND filler-contrast probes do not degrade AND paired rank stability vs the last accepted scorer
holds; `PsiScorerParityJob` before any live use. Full battery in `PLAN_3E1.md`; the gold-text G1
stays a reported diagnostic that can never flip a G-track decision.
**Status: OPEN** — 2026-08-07 planner fan-out closed (14-agent, 6-lens literature review); gate
v2 registered; D0 queued beside the usage diagnostic; no mechanism funded. 2026-08-07 (later,
post-diagnostics): both reads verified clean; fork picked in `PLAN_3E1.md` (bias-dominant with
partial group blindness); gate v2 amended pre-verdict (absolute unit-marginal floor on arm (i);
selector filler-affinity admissibility for D4). 2026-08-07 (D1 read): power check FAILED,
verified + audited (two direction-neutral numeric slips; conclusion 8's "token-specific"
corrected to a state-count artifact); probe battery demoted to mechanism meter; gate v2 (i)
improvement clause found domain-confounded — floor-only for changed-text candidates, amendment
flagged for the user's blessing; D3 cost corrected to ~85 GPU-h (~42 trimmed). AWAITING the
user: gate amendment blessing + funding shape for the admission reads and D3.

### 3f. Exit gate (Rung 3)

**Purpose.** The central claim's acceptance test.
**Approach/Experiments.** Read on the loop's final arms from the identical bootstrap; the
Phase-4 probes and the 3-BT head-to-head are inputs to it.
**Gate (pre-registered, unchanged).** All of: (1) dev-other ≤ min(Rung 0, Rung 2) − 0.5 abs;
(2) the winning checkpoint is the one the **unsupervised** criterion selects; (3) sign reproduced
by a second RL seed; (4) stable over the last third, ins/del within 1.5× SFT, §4 probes clean;
(5) reported head-to-head vs Rung 3-BT — if RL loses, BT becomes the headline and RL the
reported negative arm.
**Status: NOT FIRED.** If (1) fails with §2.5 passed, the failure localizes to the loop (lambda
balance, scorer drift, anchor) — iterate there, not in Phase 1.

### 3g. Z-track — from-scratch fully-unsupervised joint loop (USER-directed 2026-08-12)

(Moved to `PLAN_3G.md` 2026-08-14 — replaces the inline block, because the track outgrew a
page: two closed arms and one live registration. Gate text carried verbatim there.)
**Purpose.** Real unsupervised ASR without GAN: run the joint loop from zero paired data and
classify the failure mode against the pre-registered (A)/(B)/(C) taxonomy; labels evaluate
only.
**Approach / Experiments / Gate.** `PLAN_3G.md`; log `SAE_3G.md`.
**Status.** 3g.1 base arm CLOSED 2026-08-13, outcome (A) — mode collapse to one constant
sentence by step 346; the per-utterance joint objective's optimum sits at zero coupling.
3g.2 (Z2: diversity price + pseudo-pair init + derangement hinge) STOPPED BY USER
2026-08-14 mid sub-ep 5 — verdict: escaped zero coupling via a nuisance-channel ladder,
duration (0.856) then speech density (0.252), held gap +0.085 = ~2 % of a real scorer's, no
phone content, lexicon churn 14/2703; close-out battery on its last checkpoint pins the Z3
baseline. 3g.3 (Z3: tempo/noise/pitch perturbation-consistency package + hardened hinge
negatives + raised lam_div) REGISTERED AND FUNDED 2026-08-14; planner mid-run read
2026-08-15: collapse beaten but the primary clause failing — the same duration code rebuilt
more purely (98.7% stem-times-k) under the live co-trained scorer; runs untouched to its
registered end as Z4's comparison. 3g.4 (Z4: discrete psi refresh replacing co-training +
within-seq repetition price + lam_len activation; lam_lm kept at 1.0 units-norm by the
2026-08-15 ruling) REGISTERED AND FUNDED 2026-08-15. Z4 GATE VERDICT 2026-08-16 (six
rounds, planner-verified): FAILS — primary above bar only at round 1, speaker-meter
secondary fails as written, repetition price binds; the registered exhaustion reading does
NOT fire (within-group spread recovers past start), so this is a gate failure with
earnable variance remaining, not a loop that ran dry. Track pauses; follow-up space
(any Z5 vs standing on §1f alone) is the USER's call. Detail: `PLAN_3G.md` 3g.4 Status.

---

## Phase 3B — Backtranslation branch (no RL; parallel)

**Purpose.** An RL-independent second shot at beating Rung 2 — upgrades the program's worst case;
independent of the §2.5 gate.
**Approach.** Unit-level iterative backtranslation between AV-U and the AR (shared token space).
Invariant: each model always trains toward a REAL target (real text for AV-U, real units for AR);
only sources are synthetic; ~50 % previous-round data retained; unsupervised stopping.
**Experiments.** 2–4 rounds; headline = distill-back (final AV-U decodes → feature-AV SFT =
**Rung 3-BT**).
**Gate.** ≥ 1 round of positive unsupervised-score gain, and Rung 3-BT ≤ Rung 2 − 0.5 abs,
unsupervised-selected.
**Status: NOT STARTED** (pending Phase 2).

---

## Phase 4 — Controls and ablations

**Purpose.** The paper's credibility section: leakage probes, attribution controls, ablations.
**Approach.** Probes (dev, frequent during Phase 3):

| Probe | Transform | Pass condition |
|-------|-----------|----------------|
| Orthographic | homophone swap, case/punct jitter → same phi | mean Δr within 0.1 sigma_r |
| Word-boundary | resegmentations with identical phone string | mean Δr within 0.1 sigma_r |
| Content sensitivity | random G2P-distinct word substitution | mean Δr ≤ −1 sigma_r |
| Speaker leakage | linear speaker-ID probe on AV states, pre vs post RL | accuracy gain ≤ 2 abs |

**Experiments.** Remaining ablations at 100 h scale: scorer-frozen-vs-updated (now the §3e.1
program), lam_1 = 0, lam_2 = 0, pure-phoneme Option A, warm-start degradation sweep; plus the
confabulation check and the contamination control (log p_base of true dev transcripts vs
length-matched LM-corpus sentences; 4-gram-only prior deltas).
**Gate.** All probes reported; no numeric gate.
**Status.** Shuffled-reward control DONE and DECISIVE (2026-08-04, `SAE_2S.md` approach 19): the
reward is load-bearing. Probes and ablations pending on the Phase-3 endgame.

---

## Phase 5 — Refinement (gated on Rung 3 > Rung 0)

**Purpose.** Scale and side channels as measured deltas, then the headline decode.
**Approach.** (a) Qwen3-8B warm-started from the winning branch's pseudo-labels, shortened
rerun; (b) label-free speaker embedding (mean-pooled frozen states, else crop-contrastive
InfoNCE; usage-gated) + quantized F0/energy streams conditioning the AR; (c) 8B n-best
noisy-channel rescoring tuned on dev by reward.
**Experiments.** (a) then (b) then (c), each reported as a delta over Rung 3.
**Gate.** Rung 4 dominates Rung 3 with the side-channel delta isolated.
**Status: NOT STARTED** (gated).

---

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

Publish from the highest rung that holds; the BT branch and Rung S hedge the RL and bootstrap
axes respectively. The SAE story survives either head-to-head outcome — both branches
instantiate the text-bottleneck autoencoder.
