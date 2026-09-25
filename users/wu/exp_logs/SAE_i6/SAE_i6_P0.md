# SAE_i6 P0 — port to i6: reproduce the banked baselines and verify every component

## State

LIVE (2026-09-25 13:00). One manager, pid 1646677, runs the FULL graph `config/sae_i6_p0.py` (pid file
`log/sae_i6_p0.manager.pid`). Never start a second manager on it. Re-arm the watcher first after a resume (from the setup dir):
`SIS_LAUNCHER="/work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis" PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH bash ~/.claude/skills/sis/sis_watch.sh 1646677 config/sae_i6_p0.py 60`
Trainings route to V100 (gpu_32gb) from 2026-09-25 (user; Deviations: GPU partition; Gates: hardware amendment).
- ctrl_20 `GiT88bxzoZbZ`: L40S, Slurm 4346718, finishes there at about 19:00.
- ctrl_20_s1 `DvVfxf1LrCBi` (Slurm 4359756) and ctrl_20_rc `llSFybyKXkbL` (4359755): cancelled after their sub-epoch 3,
  resuming at sub-epoch 4 on V100 (about 41 min per sub-epoch, ends about 01:00 on 2026-09-26).
- k2lat `jcKXbLMDk4hl`: Slurm 4359832, on V100 since 13:05. The trie and HLG finished (Results, G0.R0: HLG
  clause FAIL, attributed). Job starts spend about 2 min per HDF input in the cache-manager (`cf`) timeout before
  reading directly; this is slow, not a failure.
Decided: the i6 prior and i6 phone text are the bed (user, 2026-09-25). The G0.R0 prior clause is a Tier-A FAIL,
attributed in Results; the T0 check stays open. G0.R3 PASS. ctrl_20 ep1 PASS.
NEXT:
- Resume verified (12:57): both logs show a Tesla V100-SXM3-32GB, epoch.003 model and optimizer loaded, and
  "start epoch 4 global train step 171".
- The G0.V suite on a V100 is green (Results: G0.V repeated on a V100). k2 on sm_70 is cleared.
- OOM at the first V100 sub-epoch: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` through
  DEFAULT_ENVIRONMENT_SET. OOM of k2lat at sub-epoch 8: `LEXLAT_K2_CHUNK_SEQS=8`, then `-p gpu_48gb`, resuming from
  epoch 7 (`reports/review_v100_routing_2026-09-25.md`, sections d and e).
- Read k2lat's "ep 1 train, step 0" line against ctrl_20's (`log.run.1:670`) under the amended G0.R2 step-1
  clause.
- Then the ctrl_20 PER at ep4/10/20, ctrl_20_s1, ctrl_20_rc and k2lat.
The package's `model/`, `training/` and `analysis/` stay FROZEN until the trainings end. Push only after ctrl_20
passes G0.R1 and the audit is done (user, 2026-09-24).

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
| `ctrl_20_rc` (added 2026-09-24, user decision) | `ctrl_20` + the hashed option that rebuilds the prior history from the blank-free cfg (SIL runs collapse to one token, the run-collapse definition of `SAE_i6_ref_objective.md` section 4.1) | the fixed control. From P0 on, it is the base of every new arm. The reproduction arms keep the banked (SIL-split) behaviour, so G0.R1-R3 stay readable. Open design point (review `reports/review_silfix_2026-09-24.md`, 3(a)): under rc one token spans at most 17 recognizer frames (1.02 s), so a longer retained pause must hold a non-SIL token that gets gradient and counts in E[N]. The banked split instead read it as several SIL tokens. Before rc becomes the base, measure the share of retained SIL runs over 17 frames after VAD and compare the rc rate and PER to ctrl_20. Measured on the VAD job `RLrgIh6lFv9m`'s masks, which equal the package recomputation exactly (`analysis/long_sil_after_vad.py`; MFA gaps incl. spn; a diagnostic, disclosed label use that never feeds training; `reports/exec_long_sil_2026-09-24.md`): after VAD, 1.22 % of dev-other utterances hold such a run (38 runs, excess 0.12 % of retained frames, max 32 frames) and 0.33 % of dev-clean (10 runs, 0.017 %, max 28). Before VAD: 3.98 % / 2.26 %. The trailing count is a lower bound: the model caps trailing runs at 10 frames, and the script counts only runs over 17. Train not measured (its MFA download is 6.5 GB). Small, but the G0.RC reads should look for insertions in these utterances |
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
  Amended 2026-09-24, before any job: under the audio label the VAD job only reports, and the gate
  reads `utterances` and `original_frames` as exact and only `kept_frames` within 0.5 %. The first two
  do not depend on the audio (the decode length equals the FLAC length on 24 of 24 probed files). A
  different ffmpeg build moved the kept count of 2 of 120 probed utterances
  (`reports/review_p0_launch_2026-09-24.md`, issue 2).
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
- **Hardware amendment to G0.R2, G0.R1s and G0.RC (2026-09-25, before k2lat started and before any V100
  sub-epoch ended; the user moved the P0 trainings to V100, Deviations: GPU partition).** Hardware split:
  ctrl_20 all L40S; ctrl_20_s1 and ctrl_20_rc sub-epochs 1-3 on L40S and 4-20 on V100; k2lat all V100.
  - What differs between the two GPUs:
    - reduction order in every cuBLAS and cuDNN kernel, the float64 lattice GEMM included;
    - the recognizer's dropout masks, which come from the CUDA RNG, whose element mapping depends on the SM count;
    - possibly the recognizer's Conv1d precision: cuDNN may use TF32 on the L40S by PyTorch default, and the
      V100 has none. Not verified.
    Bit-identity was already absent on one GPU (atomics in backward).
  - G0.R2 "Step 1 identical" now reads:
    - same step-1 batch (sequences and frames);
    - l_tau, prior per token and expected tokens within the G0.R1 step-1 tolerances of ctrl_20's
      (+-0.002 / +-0.005 / +-1.0), a cross-hardware tolerance already registered;
    - a difference above 1e-4 relative is recorded and gets a debugger read before the verdict.
    A lexicon term leaking in before the on-set moves l_tau by order 1.
  - G0.R1s band and G0.RC deltas at sub-epochs 4/10/20: the i6 band ctrl_20 - ctrl_20_s1 and the G0.RC deltas
    carry a hardware term besides the seed. With one replicate the two cannot be separated. ep1 is all L40S.
  - G0.R2 paired read and PER: k2lat on V100 vs ctrl_20 on L40S. Thresholds unchanged.
- **Confound note on G0.R0 HLG and G0.R2 (recorded 2026-09-25, before the i6 trie, HLG or k2lat gave any
  number; thresholds unchanged).**
  - What changed: the lexlat word set comes from a word LM trained on the i6 prior window (the user's i6-text
    decision). It has 182,215 words, not 151,731, and 3,414,449 in-line bigram types, not 3,302,936. Its word
    LM has 182,218 / 3,510,073 / 10,623,445 n-grams, against 151,734 / 3,393,577 / 10,419,405.
  - HLG size: the reviewer (`reports/review_trie_reportonly_2026-09-25.md`) PREDICTS about 24.9-25.1 M states
    and 101-104 M arcs. The G0.R0 HLG clause is read as registered; a miss is a Tier-A FAIL, attributed to
    this deviation only if the size tracks the vocabulary.
  - G0.R2: its thresholds were set on JUPITER's 151,731-word graph. A G0.R2 miss therefore cannot be blamed
    on the port alone.
  - k2lat GPU memory: estimated 35-38 GiB of 45 GiB, not measured; it would show at the k2 on-set, sub-epoch 8.
- **G0.R3 supervised inits (analysis only).** (A): p0 dev-other greedy PER 0.1894 +-0.02 at its selected
  checkpoint (banked: selected at pass 1). (B, report-only under an audio label): gold phi
  `dev_loss_nll_per_frame` at epoch 8 = 3.2888 +-0.02 (package-banked value, not in the JUPITER logs).
- **G0.RC fixed control `ctrl_20_rc` (registered 2026-09-24, before any job).**
  - (A, correctness):
    - The option-on T1.6 test matches the run-collapse oracle to 1e-10.
    - With the option off, every existing job id and test result is unchanged.
    - Step 1 of `ctrl_20_rc` has the same batch and inputs as `ctrl_20`; its log Z is <= ctrl_20's,
      because it sums over a subset of the latents.
  - (Report-only; no pass/fail, a new operating point): the paired dev-other PER delta
    `ctrl_20_rc` - `ctrl_20` at sub-epochs 1/4/10/20, read against the i6 seed band from G0.R1s.
    Also its emitted rate and derangement gap at 20.
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
  `reports/exec_pincheck2_2026-09-24.md`). No aarch64 emulation is available. Decision (orchestrator; the user confirmed on 2026-09-24,
  declining an import of the banked JUPITER Ogg audio):
  run P0 on this generation under `FFMPEG_PIN_ACCEPT = "x86_64-conda-ffmpeg-7.1.1-i6"` (every downstream
  hash moves; README: "a reproduction on other audio"); the gates' audio-label tolerances apply and the
  VAD job reports its totals against the banked ones instead of raising.
- Hardware: L40S 46 GB (sm_89) instead of GH200 96 GB (sm_90); x86_64 instead of aarch64.
- Slurm time: `settings.py` raises the ReturnnTrainingJob `run` task from the package's 11.5 h to 72 h
  (not hashed). At the screen's 1800 s per sub-epoch ceiling, 20 sub-epochs would exceed 11.5 h, and
  the TIMEOUT-resume path is checked only statically (`reports/impl_settings_time_2026-09-24.md`).
- Setup: `IMPORT_PATHS` made absolute in `settings.py` (tasks run in `<job>/work`; lazy recipe imports
  failed otherwise); hash-neutral.
- Env (`reports/env_build_2026-09-24.md`): every environment.yml pin unchanged; BLAS MKL instead of
  OpenBLAS; k2 source build of the pinned commit for sm_70/sm_86 (SASS sm_86 runs on the L40S);
  librosa 0.11.0 added (i6_core imports it; the spec omitted it); matplotlib-base 3.11.2 added (the
  ReturnnTrainingJob `plot` task imports it; without it every training ends in ERROR and p0's
  checkpoint pick never runs; 20 packages added, none changed). Recipe checkouts newer than the README
  pins: i6_core 4537aaf (pin ca161b7 is an ancestor; three later commits: JAX checkpoint support in
  ReturnnTrainingJob, a new ExtractOovWordsFromTextJob, an optional prettify), sisyphus a567fa7 (pin ddcd028 plus later fixes); RETURNN for jobs
  is cloned at the pinned commit with the shipped patch.
- RETURNN in the setup's `recipe/` (2026-09-24): `recipe/returnn` is a symlink to the pinned clone
  (`CloneGitRepositoryJob.KQ3NuCaDE6QH`, 00171dfe + patch). It used to be an upstream master
  clone (5f752be49), which the job config's sys.path sends DataLoader workers to. Two w2v2 forwards died
  because the worker could not unpickle the parent's config (`reports/debug_w2v2_forward_2026-09-24.md`).
  The reference used one checkout for both. No id moved and no finished output is affected
  (`reports/review_returnn_pin_2026-09-24.md`: PASS_WITH_NOTES).
- GPU partition (user rule 2026-09-24; `settings.py` `gpu_route_partition`, CLI `./gpu_route`): a GPU
  task needing <= 24 GB goes to whichever of gpu_24gb (A10) and gpu_48gb (L40S) has more GPUs usable
  for it now (a free GPU counts only on a node with enough free CPUs and memory; capped by QoS
  headroom). Larger tasks, i.e. every training, go to gpu_48gb, and a training keeps its GPU type
  across resubmits. Forwards may therefore run on A10 or L40S. Hash-neutral. Open note: a
  flexible training ignores its sticky type after a lock timeout; no P0 training is flexible.
  Changed 2026-09-25 (user decision): every ReturnnTrainingJob run task goes to gpu_32gb (V100-SXM3 32 GB,
  sm_70), resumes included. No stickiness; an explicit `-p` still wins. `settings.py` `GPU_ROUTE_TRAIN`.
  Hash-neutral (164 ids identical). Reports: `reports/impl_v100_routing_2026-09-25.md`, review
  `reports/review_v100_routing_2026-09-25.md`. ctrl_20_s1 and ctrl_20_rc were cancelled after their
  sub-epoch 3 checkpoints and resumed on V100. ctrl_20 finishes on L40S.
- Phone text (found 2026-09-25; the user decided the same day to keep the i6 prior as the bed): the i6 phonemised LM corpus keeps 40,418,258
  lines, where JUPITER's kept 39,630,169 (G0.R0 prior clause, Results). The same window also feeds the lexlat
  word LM, trie and HLG: 182,215 words vs 151,731 (confound note under Gates; `LexiconTrieBuildJob` now records
  its banked checks instead of asserting them, hash-neutral). g2p defect in the environment,
  not yet fixed: Sequitur's `adjustHigherOrder` calls `np.sometrue`, which numpy 2.4.6 removed.
  `TrainG2PModelJob.pD4nbqFLWtbi` therefore aborted ramp-ups 2 and 3 early (at iterations 41 and 25), yet
  reported success. Final dev symbol error 5.53 %, string error 22.79 %. Only the pronunciations of g2p words
  are touched (about 0.25 % of word tokens). Fix: `num.any` in the env's `sequitur.py`, then clear the
  g2p job; every job down to the prior reruns under the same hashes.

## Results

### G0.R0 input graph (read as the jobs finish)

- Pin clause: FAIL, and the audio label was accepted (Deviations).
- VAD clause (amended): PASS. Source: `BlankfreeVadHdfJob.RLrgIh6lFv9m` `output/counts_vs_expected.json`
  (2026-09-24 22:08). Utterances and original frames equal the banked counts exactly on dev-clean, dev-other
  and train (2703/968057, 2864/919980, 28539/18088388). Kept frames 831360 / 781125 / 15427887 against
  831372 / 781130 / 15427853, a relative difference of at most 1.4e-5 (tolerance 0.5 %).
- Prior clause: FAIL. The held-out ppl (order 3) is 9.601838 against a banked 9.561056 (+0.041; tolerance
  +-0.02). Source: `PhoneNgramPriorJob.qJxXHgXLe31S/output/prior.json`: 1,000,000 lines counted, 10,000 held,
  82,740,447 tokens, SIL token rate 0.1383. Debugger attribution (`reports/debug_prior_ppl_2026-09-25.md`):
  the fit, sample and held-out code and parameters equal the reference, and the difference is in the text.
  JUPITER's `PhonemizeWithSilJob.DbFgvZOGZQ8F` kept 39,630,169 of 40,418,261 LM lines: it dropped 788,092
  lines whose words had no pronunciation, because its g2p lexicon held only about half of the 773,673
  non-bliss types (inferred; the cause is not in the logs). i6 kept 40,418,258 lines. `SampleLinesJob`
  draws from the line count, so the two windows share no lines. Measured split (Slurm 4349993,
  `analysis/prior_gap/prior_window_spread.py`, output in `/work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/prior_window_spread/`;
  the seed-0 self-check reproduces 9.601838 exactly):
  - Window noise: sample seeds 0-4 on the i6 text give 9.6018 / 9.6005 / 9.5882 / 9.5909 / 9.6080, mean 9.5979,
    SD 0.0082. JUPITER sits 0.037 (4.5 SD) below, so the gap is not window noise.
  - Emulation of JUPITER's drop (a random 50 % of the non-bliss types dropped, 3 seeds; 39.618-39.620 M lines
    kept, against JUPITER's 39.630 M): 9.5785 / 9.6010 / 9.5592, mean 9.5796, SD 0.0209. This brackets 9.561.
  - Bliss-only corpus (38.95 M lines): 9.4988, a lower bracket.
  Reading: the gap is consistent with corpus makeup. Which types JUPITER lost moves the ppl by about 0.02.
  UNRESOLVED AUDIT (`reports/audit_prior_gap_2026-09-25.md`, UNDETERMINED):
  - JUPITER lies outside the 95 % prediction interval of the 5 i6 windows (p about 0.015; "4.5 SD" overstates it
    at n = 5).
  - JUPITER's banked window tokens (81,559,944 counted, 808,146 held) match the drop emulations, not the i6 windows.
  - T2b cannot tell JUPITER from i6 apart (t = 1.45).
  - JUPITER's loss was not a uniform half of the types: it lost at least 388,780 types but only 788,092 lines.
  - The seed-0 self-check compares the port with itself only; JUPITER's sampling / Witten-Bell source is not on i6.
  - So a residual port effect of about +-0.02 is not excluded. The decisive test is T0: the port's fit of JUPITER's
    window must give 9.561056.
  - The G0.R0 prior clause stays a Tier-A FAIL as pre-registered.
  This can be reproduced exactly only
  by importing JUPITER's `g2p.lexicon` (`ApplyG2PModelJob.myTIGtmrUIFq`); no JUPITER artefact is on i6.
- rho clause: PASS. `rate_rho_hz = 9.6619373279` (ctrl_20 `returnn.config:51`) is hard-coded at
  `training/config.py:316`, not computed from the text. The i6 text would give about 9.679 (debugger).
- HLG clause: FAIL as registered. k2lat's HLG, `LexlatHLGBuildJob.avjHv1Xvjyqd`, has 24,949,308 states and
  103,206,474 arcs; the gate is [23.8 M, 24.0 M] states and [98.1 M, 99.1 M] arcs.
  - Build settings: theta 0.0 (the full trigram; the ladder stopped at its first rung); peak RSS 14.5 GiB.
  - Components: H 42 / 1,681; L 1,076,120 / 1,440,630; G 3,692,293 / 21,216,609 (states / arcs).
  - Source: `output/summary.txt` and `build.json`; extract `reports/extract_hlg_k2lat_step1_2026-09-25.md`.
  - The size lies inside the reviewer's prediction from the i6 vocabulary, made before the build: 24.9-25.1 M states,
    101-104 M arcs (confound note under Gates). The trie recorded 182,215 words vs 151,731 banked (MISMATCH, as
    expected).
  - Attribution: the size tracks the vocabulary, i.e. the phone-text deviation. Not audited.
- Trie word set: the gap is JUPITER's g2p drop, as for the prior. Source: Slurm 4358262,
  `analysis/prior_gap/window_word_types.py`, output
  `/work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/window_word_types/window_word_types.txt`.
  - Self-check PASS: the i6 seed-0 window equals the trie job's replayed words byte for byte.
  - Identity PASS: each window's sample sha256 and ppl3 equal the prior-gap run.
  - Word types, in-line bigram types and tokens:
    - i6 seed-0 window: 182,215 / 3,414,449 / 19,885,331.
    - Drop-emulated windows: 151,228 / 151,515 / 151,660 types; 3,300,564 / 3,295,442 / 3,302,838 bigram types;
      19.58-19.63 M tokens.
    - JUPITER: 151,731 / 3,302,936 / 19,629,091.
  - So the 30,484 extra i6 words are the words JUPITER's lexicon lacked. It is one deviation (Deviations:
    phone text), not a second port defect. The trie task was cleared on this reading (State plan).

### Cost screen and G0.R1 step 1 (ctrl_20 sub-epoch 1, 2026-09-25, `ReturnnTrainingJob.GiT88bxzoZbZ`)

- Wall time: FAIL. 3237 s of training in sub-epoch 1 (`work/learning_rates` `epoch_train_time_secs`; 57 steps;
  99.7 % computing time), 3537 s from job start to the checkpoint. The bound is 1800 s. This is 1.8x the bound,
  about 18 h of training for 20 sub-epochs.
- Peak GPU memory: PASS. `nvidia-smi` on the node after sub-epoch 1 reads 35,091 MiB (34.3 GiB) of 46,068 MiB,
  at 100 % utilisation (read through `srun --overlap`; `log/gpumem_probe.4348532.out`). The reference batch
  shape is unchanged (88,000 frames, 128 seqs). At this shape a 32 GB V100 cannot host the run.
- Step 1 (RETURNN "step 0", `log.run.1:670`), audio-label tolerances: l_tau -0.352 (banked -0.350 +-0.005, PASS);
  prior per token -5.637 (banked -5.657 +-0.01, FAIL); expected tokens 63.854 (banked 63.821 +-3 %, PASS). The
  recognizer is flat-initialised, so the miss is expected from the flatter prior. It is confounded with the
  audio label, which changes the units and eta.
- ep1 dev-other greedy PER 0.855180 (banked 0.855 +-0.015 under the audio label, PASS). S/D/I = 34109 / 117437 / 56
  over 177,275 phones. Emitted rate 3.255 /s (Tier B 3.30 +-0.3, PASS). Source: `BlankfreeGreedyPerJob.xGeNddosJKVH`
  (`output/sae/4a/ctrl_20/ep1/dev-other/per.json`). This is on the i6 prior.
- Decision (orchestrator, 2026-09-25). The time miss has a known cause (the float64 lattice GEMM on Ada;
  the GPU is saturated), and no other i6 pool fits this batch. The prior-dependent arms (ctrl_20_s1, k2lat,
  ctrl_20_rc) stay held until the user decides on the prior, and weighs the cost there. gold phi and p0
  are released on their own entry point, `config/sae_i6_p0_supinit.py` (manager started 2026-09-25 01:27).
  gold phi does not depend on the prior. p0 carries it in its hash and loads it, but its values enter
  no loss, pick, dump or PER (`reports/review_supinit_launch_2026-09-25.md`), so p0's PER holds under either
  prior. ctrl_20 keeps running.
- V100 benchmark (2026-09-25, user request; Slurm 4358701 on cn-32, V100-SXM3-32GB, and 4358702 on cn-508, L40S;
  `analysis_out/v100_bench/{v100,l40s}/summary.txt`; review `reports/review_v100_bench_2026-09-25.md`).
  - Method: the real ctrl_20 step, resumed from epoch.011 (sub-epoch 12), on 12 seeded uniform-random train batches
    (not laplace), the same batches on both GPUs, with 2 untimed warm-ups.
  - 88,000 / 128: median step 51.68 s on V100 [51.39, 51.89] vs 68.14 s on L40S [66.29, 68.80]; ratio 0.758.
    The earlier "does not fit a V100" was wrong. Peak memory, identical on both: 20.28 GiB allocated. Reserved:
    28.22 GiB on V100, 36.8 GiB on L40S; the nvidia-smi 34.3 GiB was mostly the allocator cache.
  - 44,000 / 64: 25.71 s vs 31.72 s. Per padded frame it matches the full batch (0.589 ms on V100), so halving
    buys nothing.
  - Projection: 3237 s x 0.758 = 2455 s of training per sub-epoch (41 vs 54 min); a 20-sub-epoch run takes
    about 15 h instead of 19 h.
  - The float64 matrix multiply was expected to dominate, which would predict a much larger V100 gain; where the
    time goes is not profiled.
  - Decision (orchestrator): P0's remaining trainings stay on L40S. G0.R2 pairs k2lat with the L40S ctrl_20, and
    a GPU switch inside a Tier-A pair adds a hardware difference to the seed noise. Later packs run on V100.
    SUPERSEDED the same day by the user ("start them on V100"). The hardware amendment is under Gates.

### G0.R3 supervised inits (2026-09-25; both finished at 01:41-01:43 on L40S)

- p0 (A): PASS. Dev-other greedy PER 0.189626 (banked 0.1894 +-0.02), S/D/I 8673 / 17916 / 7027 over 177,275
  phones. Selected epoch 1 (banked: pass 1). Epoch 1 trained 353 steps in 39 s. Source
  `output/sae/4a/analysis_only/p0/best/dev-other/per.txt`, training `ReturnnTrainingJob.Y1vbqR6KeJSx`.
- gold phi (B, report-only under the audio label): PASS. `dev_loss_nll_per_frame` at epoch 8 is 3.274249
  (banked 3.2888 +-0.02). Epochs 1-8: 3.4343 / 3.3285 / 3.2878 / 3.2845 / 3.2820 / 3.2731 / 3.2806 / 3.2742.
  Training `ReturnnTrainingJob.Ac2eioZbRX7d`.
- Reading: features, VAD, gold, the recognizer, the reverse model and the PER chain reproduce on the i6
  audio generation. Together with ctrl_20's ep1 PER, this makes a port defect outside the text prior unlikely.

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

### G0.V gpu- and k2-marked tests (2026-09-24, one L40S)

Slurm 4334384 on gpu_48gb (cn-508; NVIDIA L40S, capability 8.9; torch 2.7.1, CUDA 12.6), full suite:
506 passed, 11 skipped, 12 strict xfail (the same 12 as the CPU run, no XPASS), 0 failed, 120 s.
Skips: 7 `artefact` tests (P0 outputs not yet built) and 4 ffmpeg-pin tests (they need the
reference-cluster ffmpeg); none for CUDA or k2. T1.8 GPU parity passed; the CPU-vs-CUDA parity of
log Z_HLG and log Z_H on the T1.19 fixture passed at tau 1 and 2 (agreement about 1e-15, tolerance
1e-5). First confirmed k2 kernels (sm_86 build) on sm_89. RETURNN-importing tests ran on the training
clone (00171dfe + the shipped patch). Report: `reports/exec_gpu_tests_2026-09-24.md`; logs in
`/work/asr4/hwu/sae_i6_tests/gpu_2026-09-24/`.
G0.V: every clause is met. Several strict xfails sit inside the train step: the SIL-run split (T1.6)
and the `_logmm` floor (T1.4c, T1.5). They count as banked behaviour only through the orchestrator's
decision "the code defines the bed" (`SAE_i6_ref_objective.md` section 10). The user accepted T1.4c and
T1.5 on 2026-09-24; T1.6 is pending the user's decision.

### G0.V repeated on a V100 (2026-09-25, before any k2 use on V100)

- Run: Slurm 4359759 on cn-32, `analysis/v100_bench/run_k2_tests.sh`, which runs the same full pytest command.
  The log reads "device Tesla V100-SXM3-32GB capability (7, 0)". Output:
  `/work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/v100_k2_tests/`.
- Outcome: 529 passed, 11 skipped, 12 xfailed, 0 failed, 849 s. The suite now collects 552 tests; the L40S run
  collected 529. The 11 skips are the same artefact-dir and reference-ffmpeg skips; none is a CUDA or k2 skip.
- Passed on the V100:
  - T1.8 GPU parity;
  - the CPU-vs-CUDA parity of log Z_HLG and log Z_H at tau 1 and 2;
  - T1.18 and T1.19;
  - the T2.5 k2 train-step plumbing.
- Reading: the sm_70 k2 kernels are exercised and agree with the CPU oracles. k2lat may use k2 on V100.
  Its memory at sub-epoch 8 is still unmeasured (Gates, hardware amendment).
