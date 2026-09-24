# SAE_i6 P0 — port to i6: reproduce the banked baselines and verify every component

## State

No manager, watcher or Slurm job is live; no training has run. Done: env; i6 `settings.py` (absolute
IMPORT_PATHS, `FFMPEG_PIN_ACCEPT` label after the pin FAIL); package amendments (ctrl_20_s1, p0 PER read,
report-only VAD counts; `reports/impl_p0_amendments_2026-09-24.md`); configs `config/sae_i6_p0_screen.py`
(75 jobs: inputs + ctrl_20 `ReturnnTrainingJob.GiT88bxzoZbZ`) and `config/sae_i6_p0.py` (139 jobs);
tests P1-P3 + gaps: 503 passed, 12 strict xfail, 0 failed (oracle review PASS_WITH_NOTES). Package
code/test/config changes are UNCOMMITTED until the launch review passes.
In flight at the last save: the launch code review (`code-reviewer`), writing
`reports/review_p0_launch_2026-09-24.md`. If that file is absent on resume, re-dispatch the review of
the current state (settings.py, the amendments, the screen launch).
NEXT: review PASS -> commit package changes -> launch `config/sae_i6_p0_screen.py` per the sis skill's
section for this setup (manager + watcher) -> cost screen at ctrl_20 sub-epoch 1 (<= 1800 s per
sub-epoch, <= 40 GiB) + step-1 triple + ep1 PER -> run gpu-marked tests on a gpu_48gb node -> release
`config/sae_i6_p0.py`.

## Objective

Establish a trusted i6 baseline for the SAE_i6 campaign before any new experiment: (1) the ported
package (`recipe/i6_experiments/users/wu/experiments/unsupervised_asr`) builds its inputs from raw
public sources and trains on i6 hardware; (2) its banked JUPITER numbers reproduce within the bed's
known noise; (3) every component, above all the loss terms, computes what `SAE_i6_ref_objective.md`
says. Everything later compares against the i6 controls produced here, not against JUPITER numbers.

## Constraints

- Label quarantine and the other standing constraints of `SAE_i6_ref.md` section 2.
- No change to the method. Permitted i6 adaptations, each disclosed below when made: interpreters and
  tool paths, Slurm resources (partition, memory, time), the audio generation label if the ffmpeg pin
  check fails, and a batch-shape change ONLY if the reference shape (88,000 padded frames, 128 seqs)
  does not fit a 46 GB L40S; a batch change is a new operating point and voids the step-1 clause of
  G0.R1.
- GPU pool: gpu_48gb L40S (5 GPUs per user). The reference was one GH200 96 GB per arm.

## Runs

| run | entry point | why |
|---|---|---|
| input graph | `config/base.py` inputs (download, ffmpeg pin, ogg, rVAD, w2v2 L15, units, eta, CV split, phone text, prior, duration prior, gold) | everything downstream |
| `ctrl_20` | `config/base.py` | the control of every later pack; banked PER 0.874568 |
| `ctrl_20_s1` | seed replicate of ctrl_20 (flat_seed 1, random_seed 1, offset 1000) | the i6 seed band B for every later pack; a second step-1 identity point |
| `k2lat_20_ma3000` | `config/k2_word_lm.py` preset | the only banked arm clearly off the control (-0.0560); exercises the k2 path |
| gold phi and p0 (analysis only, disclosed label use) | `config/supervised_init.py` | p0's dev-other PER 0.1894 is a near-deterministic end-to-end check of features, VAD, gold, recognizer and PER chain; the gold phi exercises the reverse model |

`ctrl_20_x60`, `k2lat_20_ma3000_x60`, `off4_k2lat_20` and the never-run default `k2_word_lm` are
not part of P0 (E60 already showed a plateau; they cost 3x). Reads: the package's registered reads
(greedy PER at kept epochs, derangement and decode gaps and JS rows at the final epoch, paired delta).

## Gates (pre-registered 2026-09-24, before any job; amended the same day before any job)

The first registration (G0.R1-R3 on PER +-0.03, step-1 l_tau +-0.01 / tokens 2 %, gold-phi NLL
3.2888 +-0.02, G0.V on priority-1 tests) was superseded before launch by the design review
(`reports/design_review_p0_2026-09-24.md`): as written, R1/R3 could pass a broken port (every
content-free arm lands in the PER band; 3.2888 is untraceable in the JUPITER logs) and R2 could fail a
correct one (the banked identical-config k2 replicate sits at the window edge). Banked per-utterance
hypotheses are not available on i6, so reproduction reads are tolerance reads against banked scalars.

Tier A: a miss is FAIL (debugger before any rerun). Tier B: a miss triggers a debugger read before the
verdict; REPRODUCED may still be declared if the debugger attributes the miss to hardware or audio
with evidence recorded here. "Audio label" = the run uses `FFMPEG_PIN_ACCEPT` (a different audio
generation); widened tolerances in brackets apply then.

- **G0.R0 input graph (A).** Pin check verdict recorded first. HLG (in-house, word-boundary) states in
  [23.8 M, 24.0 M], arcs in [98.1 M, 99.1 M]. Prior held-out ppl 9.56 +-0.02; rho = 9.6619373279
  (1e-9 rel). VAD totals equal `BANKED_VAD_COUNTS` (audio label: within 0.5 %, report-only).
- **G0.R1 ctrl_20 (A).** dev-other PER ep1 0.855 +-0.01 [+-0.015]; ep4 / 10 / 20 0.875 / 0.869 / 0.874568
  +-0.03; step 1 l_tau -0.350 +-0.002, prior per token -5.657 +-0.005, expected tokens 63.821 +-1.0
  [+-0.005 / +-0.01 / +-3 %] (a batch-shape change voids the step-1 clause); ep20 emitted greedy rate
  9.16 +-0.3 /s, derangement gap 4.27 +-0.6. (B): dev l_tau ep20 1.811 +-0.05, dev agg 1.516 +-0.15,
  dev reverse per frame -3.26 +-0.3, ep1 rate 3.30 +-0.3.
- **G0.R1s ctrl_20_s1 (A; new run: flat_seed 1, random_seed 1, random_seed_offset 1000).** Step 1
  -0.347 / -5.671 / 57.498 at the R1 tolerances; PER ep20 0.8751 +-0.03. Report the i6 band ctrl_20 -
  ctrl_20_s1 at ep1/4/10/20 (banked +0.004 / +0.002 / -0.011 / -0.001): it is the B of every later pack.
- **G0.R2 k2lat_20_ma3000 (A).** Step 1 identical to ctrl_20's (pre-on-set path); PER ep20 0.818615
  +-0.03; paired vs the i6 ctrl_20 at ep20: CI upper bound < -0.015 and point in [-0.086, -0.026]
  (banked -0.0560; the banked identical-config replicate reads about -0.036); dev
  `lexlat_k2_expected_words` at ep20 in [24, 38]; escape share (expected escape words / expected words)
  <= 0.01; `lexlat_k2_stability` <= 0.05 from sub-epoch 11; `lexlat_k2_empty_frac` <= 0.002; no abort
  marker. (B): dev lexicon term ep20 0.310 +-0.03; derangement gap 3.61 +-0.6; emitted rate in
  [7.0, 8.5] /s.
- **G0.R3 supervised inits (analysis only).** (A): p0 dev-other greedy PER 0.1894 +-0.02 at its selected
  checkpoint (banked: selected at pass 1). (B, report-only under an audio label): gold phi
  `dev_loss_nll_per_frame` at epoch 8 = 3.2888 +-0.02 (package-banked value, not in the JUPITER logs).
- **G0.V components.** All priority-1 tests (T1.1-T1.23) and T2.1-T2.4 green, the gpu- and k2-marked
  tests run on a gpu_48gb node, plus a CPU-vs-CUDA parity assert for `log_z_hlg` and `log_z_h` on the
  T1.19 fixture (1e-5). Strict xfails are allowed only for defects outside what the train step
  computes, or for banked behaviour the owner has accepted, and each is listed in Results. Owner
  decisions S1, S2, S7: recorded (`SAE_i6_ref_objective.md` section 10: the code defines the bed).
- **Launch order (cost screen).** The ffmpeg pin check runs first, alone (CPU). Then the input graph and
  ctrl_20 alone among the trainings; after its sub-epoch 1: wall time per sub-epoch <= 1800 s (3x
  GH200) and peak GPU memory <= 40 GiB at the reference batch shape, else stop and decide (the lattice
  GEMM is always float64, `model/lattice.py:651-672`, and Ada GPUs run fp64 at about 1/64 of fp32; V100
  runs it at 1/2). Only then are k2lat_20_ma3000, ctrl_20_s1, the gold phi and p0 released.

Verdict: REPRODUCED if every Tier-A clause and G0.V pass and every Tier-B miss is attributed; a
Tier-A miss goes to the debugger before any rerun; P0 closes on REPRODUCED or on a user decision.

## Deviations from the reference (filled as they are made)

- **Audio generation (G0.R0 pin clause read 2026-09-24): the ffmpeg pin check FAILS** — the i6 conda
  ffmpeg 7.1.1 (x86_64) reproduces only 14 of the 2864 dev-other reference PCM digests (aarch64 build;
  `reports/exec_pincheck2_2026-09-24.md`). No aarch64 emulation is available. Decision (orchestrator):
  run P0 on this generation under `FFMPEG_PIN_ACCEPT = "x86_64-conda-ffmpeg-7.1.1-i6"` (every downstream
  hash moves; README: "a reproduction on other audio"); the gates' audio-label tolerances apply and the
  VAD job reports its totals against the banked ones instead of raising.
- Hardware: L40S 46 GB (sm_89) instead of GH200 96 GB (sm_90); x86_64 instead of aarch64.
- Setup: `IMPORT_PATHS` made absolute in `settings.py` (tasks run in `<job>/work`; lazy recipe imports
  failed otherwise); hash-neutral.
- Env (`reports/env_build_2026-09-24.md`): every environment.yml pin unchanged; BLAS MKL instead of
  OpenBLAS; k2 source build of the pinned commit for sm_70/sm_86 (SASS sm_86 runs on the L40S);
  librosa 0.11.0 added (i6_core imports it; the spec omitted it). Recipe checkouts newer than the README
  pins: i6_core 4537aaf (pin ca161b7 is an ancestor; three later commits: JAX checkpoint support in
  ReturnnTrainingJob, a new ExtractOovWordsFromTextJob, an optional prettify), sisyphus a567fa7 (pin ddcd028 plus later fixes); RETURNN for jobs
  is cloned at the pinned commit with the shipped patch.

## Results

### G0.V priority-1 tests (2026-09-24; CPU on the desktop, sae env; GPU parity T1.8 not yet run)

Suite after P1: 394 passed, 5 skipped, 11 strict xfail, 0 failed (`reports/impl_tests_*_2026-09-24.md`).
Every loss term the `ctrl_20` bed uses matches an independent brute-force oracle on tiny instances:
lattice log Z, gradients, posteriors and statistics over the grid of bands, histories, tau, beta,
reductions, checkpointing and fp32/fp64 (T1.1-T1.7; fp32 vs fp64 log Z within 1.0e-7 to 1.4e-6
relative at production shape); reverse-model tables and DP (T1.9-T1.10); Witten-Bell tables to 1e-12
(T1.12-T1.13); expected non-SIL counts and the rate term's FD surrogate (T1.14-T1.15; at the bed's
eps 0.25 the gradient bias is at most 1.9e-3 relative, E[N] 3.4e-4); aggregate run counts, EMA and
targets (T1.16-T1.17); the k2 term against an explicit L∘G enumeration (T1.18-T1.22). The strict
xfails pin recorded defects, none of which changes what the banked runs computed:
- SIL runs may split into several SIL tokens in the train step's lattice (prior history built for
  "ctc"; T1.6) — the banked behaviour; magnitude on real batches to be measured (queue).
- `_logmm` floors impossible products at about -708 (T1.4c, T1.5) — only S = 1 is infeasible on the bed,
  and it is caught elsewhere.
- One-token strings double-counted by the order-3 prior scorer (T1.12/T1.13) — statistic only.
- k2: pruned G unnormalised (S3; official 4-gram arms only), the lexicon term is a tempered unnormalised
  weight that can be positive (S5), trailing back-off over-count; S4 exact for the production placement,
  S6 refuted. Details: `SAE_i6_ref_objective.md` section 10.
Code-vs-note decisions S1, S2, S7: the code is authoritative; the note is corrected (section 10). T1.11
measures the S1 gap at tau = 2 as +0.0622 nats on its instance (0 at tau = 1).

### G0.V priority-2 and -3 tests (2026-09-24, CPU)

Suite: 495 passed, 14 skipped (7 `artefact` tests wait for P0 outputs; gpu-marked tests wait for a GPU
node), 12 strict xfail, 0 failed (`reports/impl_tests_p2_2026-09-24.md`, `impl_tests_p3_2026-09-24.md`).
T2.1, the train step's assembly on an enumerable batch (S [6, 5, 1], tau 2): l_tau and rate equal the
oracle to about 1e-16, agg to float32 rounding, total = 1 l_tau + 3 rate + 0.1 agg, kept-row
normalisation right (S = 1 flagged z_zero); theta's gradient equals the exact oracle to 4.5e-9 at
eps 1e-4, and at the bed's eps 0.25 it deviates from the exact gradient by 1.03e-5 (1.3e-4 of the
largest component); the reverse model receives gradient from l_tau only (bitwise). Schedules as read by
the model, checkpoint slicing, the posterior-dump -> PER chain (real RETURNN CPU forward), paired CI,
Viterbi decode, genmarg marginals, JS, flat init and duration-prior init/freeze all match their oracles.
Data side: rVAD rule, the L15 tap and normalisation (SpecAugment off in eval, which both dump paths use;
S11), eta PCA, SIL-insertion statistics, official lexicon reader all pass. New strict xfail: the null
recognizer's -log 40 is float32-rounded (T_rec 3.6e-8/tau nats; negligible). rho (S8) is 2.7 x phones
per word of the FULL phonemised corpus (2,784,159,269 / 778,025,128), not the prior window, consistent
with the objective note; the artefact test checks the numerator on the P0 outputs.
Open for G0.V: the gpu- and k2-marked tests (T1.8, CUDA parity of log_z_hlg / log_z_h) on a gpu_48gb
node.
