# SAE_i6 P1 — completion and small extensions of the JUPITER framework

## State

OPEN (2026-09-25 14:25, re-scoped by the user). Runs in parallel with P0 and never touches P0's manager or graph. No job yet.
- Task A (core phi reads): G1.G PASS (Results). Fix 3 (`WAVE_*` durinit at 12) is still open; it waits until
  the P0 trainings end, because the P0 graph imports `reverse_model/phi_first.py`. No Task A manager is live.
- Task B (lift ladder): gates amended from the design review before any job (`reports/design_review_p1_2026-09-25.md`).
  - Fits DONE, G1.F PASS (Results). No fits manager is live. r70 / r80 / r90 = `ReturnnTrainingJob.ruLnJFWyifwp` /
    `6IkAAuzcBwBR` / `uO8wkodbR2uh`, epoch 8.
  - Arms: the per-chunk backward (`reverse_model/rt_chunked_backward.py`, selected by a hash-neutral config epilog)
    and `config/sae_i6_p1_ladder.py`. Review PASS_WITH_FIXES (`reports/review_p1_arms_chunked_backward_2026-09-25.md`):
    exact on CPU (deviation 0.0), one delta per arm. The memory-log fix and the NaN-aware CUDA compare are
    re-reviewed PASS (`reports/review_p1_probe_memlog_2026-09-25.md`, 39 rt tests pass, job ids unchanged). Its F1:
    with `stop_on_nonfinite_train_score = True`, a failed stability read (NaN monitor) makes RETURNN stop the arm,
    against the read's design. Decision: a diagnostic may not end an arm, so the guard is being fixed before the
    parity re-run (`reports/impl_p1_stability_nan_guard_2026-09-25.md`, in progress). CUDA gradients are still
    unverified (Slurm 4361250 stopped at the NaN-vs-NaN monitor).
NEXT: review of the NaN guard, then the GPU parity re-run (CUDA gradients), then commit the code, then the rt_r90
probe (STAGE=probe, one L40S, G1.M) under a single manager that builds the same fit ids; then rt_r70 and rt_r80.
The second-seed builder needs a `seed=` argument in `ladder.py`; it is needed only if G1.L's second-seed rule fires.

## Objective

Complete the core pieces of the JUPITER framework that the port lacks, and run small, label-using extensions of
JUPITER's established experiments on i6. Each task has its own pre-registered gate. New cold-start methods are
not in P1. The port scope (what is core, what is ported only on demand, what is not ported) is in `SAE_i6.md`,
Queue 2.

## Task A: core phi reads (registered 2026-09-25 as P0 G0.G; moved here the same day, before any code or run)

Three fixes are needed before any phi run: (1) generative PER on the D4 dev-other set (direct, Hungarian, NMI),
a standing read for every phi run; (2) ladder competence reads on the 260 set, since the 285 set holds 25 fit
items; (3) the wave default durinit at 12 sub-epochs (`WAVE_*` is None).
- **G1.G (A):** the i6 gold phi's genmarg posterior decode of the D4 dev-other set (500 utterances) under the
  trigram gives Hungarian PER 0.193 +-0.02 (banked R1, `SAE_i6_ref_lexlat_v2.md`, A15 table, row gold). The i6 gold
  phi is a refit on other hardware (P0 G0.R3), hence the tolerance of the p0 clause.
- **G1.G (A):** unit tests for the D4 sample, the Hungarian map and NMI against hand-computed oracles; the 260 set
  is the 285 CV-holdout set minus the fit items shared by every ladder phi, and there are 25 of them.
- **G1.G (B, report-only):** direct PER, the uniform-prior Hungarian PER (banked R2 0.276), NMI(symbol, phone),
  E[d].
- Every P0 job id stays unchanged by the change.
- Read conventions against JUPITER's (code review, `reports/review_g1g_core_phi_reads_2026-09-25.md`). MATCH:
  the Hungarian map (40x40 with a zero-gain DELETE column, after an identity-label edit alignment); direct PER (SIL
  dropped, no merging); token NMI; the D4 sample (2863 eligible); the 260 set. Also MATCH, settled from JUPITER's
  source (`exp_logs/SAE/reports/reply_genmarg_conventions_2026-09-25.md`, origin bb33ab6d1): R2's prior is a
  normalised 1/40 table at prior weight 1, as on i6 (the same job hash, `UniformPhonePriorJob.n7Yd9EbtVi13`); E[d]
  is the unweighted mean of phi's duration law over the 39 non-SIL types, SIL separate, as `duration_means`
  computes it. Frame NMI is i6-only.

## Task B: corrupted-phi lift ladder — reproduce rt_r70, extend to rt_r80 and rt_r90 (user, 2026-09-25)

JUPITER's L2-0 node R showed that a supervised phi fitted to corrupted gold strings lifts a cold theta to near-gold
PER by sub-epoch 8. It LIFTs up to rho = 0.7 and does NOT LIFT at 1.0 (rho*_lift = 0.7; `SAE_i6_ref_lexlat_v2.md`,
"L2-0 ladder"). Where the boundary lies inside (0.7, 1.0) is unknown. Task B reruns rt_r70 to check that the regime
reproduces on i6, then runs rt_r80 and rt_r90 to place the boundary to 0.1. It is a disclosed LABEL-USING
diagnostic, analysis only: never cold-start progress, a route or a fallback (`SAE_i6_ref.md` section 2).

### Design (JUPITER node R exactly; `reverse_model/ladder.py` docstring; source `SAE_4A_lexlat_v2.md` L2-0)

- phi_rho: the gold phi's fit (`supervised.blankfree_supervised_reverse_init`, final epoch 8) on the 10 h seed's gold
  strings, with exactly round(rho n_u) tokens per utterance substituted.
  - Each substitute is drawn from the seed gold unigram without the original symbol; `CorruptSeedGoldJob`, seed 0.
  - The draw depends on seed and tag only, so the ladder is nested: r70's substitutions are the first ones of r80's,
    and r80's the first ones of r90's.
- Arms rt_r70, rt_r80, rt_r90:
  - D10e's `supphi_k2lat` with tau held at 2.0 and N = 20 learning rates truncated to 8 sub-epochs, kept 1/2/4/8;
  - the k2 block at rung 1000, on-set 1, ramp 3, lam 1; both models trainable;
  - theta at the cold flat init (`FlatRecognizerInitJob`, seed 0); phi from phi_rho; `lexlat_k2_chunk_seqs` 4.
  - rt_r70 is JUPITER's arm unchanged. rt_r80 and rt_r90 differ from it only in rho.
- Reads: dev-other greedy PER at ep1/2/4/8 per arm; the corruption report of each fit (`corruption.json`).
- Known deviations inherited from P0 (Deviations there): the i6 prior and i6 phone text, the i6 HLG (182,215 words,
  not 151,731), the audio label, i6 hardware, and the i6 gold phi recipe refit (G0.R3).
- The lattice keeps JUPITER's banked SIL split (`sil_run_collapse` off), not P0's rc base: Task B is a reproduction.
- GPU: the arms run on `gpu_48gb` (L40S 46 GB), one arm per job, not JUPITER's 4-GPU pack. JUPITER's arms peaked
  at 80-84 GB reserved at steps 1-2 on a 95 GB GH200 (A13), and the training path keeps every k2 chunk's graph
  (`model/lexlat_k2_train.py:480-490`), so chunking cannot lower the peak. The arms therefore use a per-chunk k2
  backward with gradient accumulation, the fix A13 planned and never applied. It is implementation-only and exact
  up to float64 rounding, pinned to the held path by a unit test (log Z 1e-9, gradient 1e-7). It is not a new
  operating point (design review, `reports/design_review_p1_2026-09-25.md`, item 1).
- Run collapse, RESOLVED (JUPITER source, `exp_logs/SAE/reports/reply_corruption_collapse_2026-09-25.md`, origin
  f9becb0f6). JUPITER's `CorruptSeedGoldJob` never collapsed runs and asserted equal length, as the port does. The
  seed gold itself keeps 1705 adjacent repeats. JUPITER's banked r70 (`CorruptSeedGoldJob.mnzDC7XJX00r`, seed 0,
  2849 utterances, 351,312 tokens): realised rate 0.699905, adjacent repeats 1705 gold / 14293 corrupted. The draw
  depends only on the seed, the tag and the seed gold strings, so equal i6 seed strings give identical counts.

### Gates (pre-registered 2026-09-25, before any job; amended the same day from the design review, before any job)

The registration text of each amended clause is kept under "Original".

- **G1.F phi fits (A).** The realised substitution rate of r70 / r80 / r90 on all seed utterances is 0.700 / 0.800 /
  0.900 +-0.005. The nesting holds, read by a positions check (`analysis/`): each smaller rho's substituted positions
  are a subset of the larger's and carry the same symbols. Each fit ends at epoch 8 without error. Amendment (A):
  each fit's dev NLL per frame at epoch 8 is above the i6 gold phi's 3.2742 (G0.R3) and rises with rho, r70 < r80
  < r90 (< r100 if run); this catches a fit wired to the gold strings or checkpoint. (B): the competence statistic S
  on the 260 set for each phi (JUPITER: r70 4.411, r100 4.690), expected to rise with rho. The i6 r70 corruption
  counts against JUPITER's (Design, run collapse): identical if the i6 seed gold strings equal JUPITER's; any
  difference is traced to the seed strings.
  - Original: "Its final dev NLL is report-only (B)."
- **G1.M memory and time (A for launching the arms).** A probe runs the FULL config of the most exposed arm, rt_r90
  (rt_r100 if it is run), on one L40S for all of sub-epoch 1, with the per-chunk backward. It records per-step peak
  allocated memory (`torch_log_memory_usage` in `post_config`), the k2 monitor `lexlat_k2_peak_reserved_gib`,
  `nvidia-smi` on the node, the step time and the sub-epoch wall time. PASS = no OOM, no k2 int32 error, no
  `lexlat_k2_ABORT.json`, the sub-epoch 1 stability read completes, and peak allocated <= 40 GiB over the whole
  sub-epoch. A passed probe may continue as rt_r90. On a miss, no arm launches; the miss goes to the debugger.
  - Amended 2026-09-25 from the arms code review (`reports/review_p1_arms_chunked_backward_2026-09-25.md`), before
    any arm job. The peak is the max over each step of the pre-k2 peak (which includes the stability read) and the
    k2 leg's peak. Both are allocated memory: `lexlat_k2_pre_peak_allocated_gib` and the step's `mem_usage:cuda:0`
    line; the reserved-memory monitors are context only (clarified 2026-09-25 from the memlog re-review, before any run). Device-level used memory (`torch.cuda.mem_get_info`, logged per step) stands in for
    `nvidia-smi`, which cannot run beside a batch job here. A probe over 40 GiB that has not crashed is cancelled at
    the read and does not continue as rt_r90.
  A change of batch shape, or of anything that moves a score or gradient, is a new operating point: it needs the
  design review and voids the G1.R70 comparison. `lexlat_k2_chunk_seqs` and `expandable_segments` are launch
  granularity and move nothing; both are disclosed if used.
  - Original: "(report) Record peak allocated and reserved memory over the first sub-epoch's steps 0-3 ... and the
    step time." A13 set its peaks at steps 1-2, but the longest batches came at steps 7-8.
- **G1.R70 reproduction.** (A): rt_r70 reads LIFT (< 0.50) at ep8, and ep8 < ep1. (B): ep8 0.1909 +-0.02, ep1
  0.2533 +-0.03. A B miss goes to the debugger before the verdict and may be attributed with evidence to the
  disclosed deviations (prior, graph, audio label, refit, hardware). On an attributed miss G1.L is still read, and
  the phase records that the i6 boundary is not comparable to JUPITER's at the 0.02 level.
  - Original: "(A) ... reads LIFT (< 0.50) and lies within 0.1909 +-0.02. (B): ep1 0.2533 +-0.03." The window
    0.171-0.211 also contains JUPITER's r0 / r30 / r50 (0.178 / 0.183 / 0.186), so a pass cannot single out r70,
    while a miss could be an attributable bed effect that would void r80 and r90.
- **G1.L boundary (the question; read only if G1.R70 passes).** Class at ep8 by JUPITER's A4 bands: LIFT < 0.50,
  PARTIAL < 0.8164, NO LIFT otherwise. rho*_lift on i6 is the largest rho in {0.7, 0.8, 0.9} that LIFTs with every
  smaller one lifting. r30 and r50 LIFT on JUPITER and are not rerun here.
  - r80 not LIFT, r90 not LIFT: boundary in (0.7, 0.8].
  - r80 LIFT, r90 not LIFT: boundary in (0.8, 0.9].
  - Both LIFT (amended): rt_r100 is run on i6 (same recipe, rho 1.0). NO LIFT puts the boundary in (0.9, 1.0];
    PARTIAL or LIFT voids the branch and goes to the debugger. Original: "resting on JUPITER's r100".
  - Added: r80 not LIFT with r90 LIFT is CANNOT_TELL. Both rungs then get a second seed (corruption seed 1 and
    theta init seed 1).
  - Added: the first rung that does not LIFT gets the same second seed before the boundary is final. If the two seeds
    disagree, that rung reads CANNOT_TELL.
  - Added: an arm with `lexlat_k2_ABORT.json` or without an ep8 checkpoint reads VOID, never NO LIFT.
  - PARTIAL at ep8 is reported with its ep1-8 trajectory. It is not read as LIFT; rho*_lift is defined at the
    8-sub-epoch budget.
  - If G1.R70 fails its Tier A, the r80 / r90 classes are reported but no boundary is claimed; the miss goes to the
    debugger.
- **Caveat (audit N1, carried).** The corruption is independent of the acoustics, so each phone's most likely unit
  stays right below rho 1. rho*_lift does not transfer to an EM phi, whose errors are structured.

## Runs

- Fits (2026-09-25 14:51): manager on `config/sae_i6_p1_fits.py`; job ids in State. The corruption on the i6 seed
  gold (2849 utterances, 351,312 tokens, 1705 repeats) gives realised rates 0.699905 / 0.800081 / 0.899946 when
  the reviewer re-runs the code. The r70 rate, PER vs gold 0.695288 and 14,293 repeats equal JUPITER's banked
  `CorruptSeedGoldJob.mnzDC7XJX00r` exactly, so the i6 seed strings are JUPITER's. These are the reviewer's code
  reads; the jobs' own corruption.json are read at G1.F.

## Deviations from the reference (filled as they are made)

- The fits run on gpu_24gb (A10). The i6 gold fit ran on L40S and JUPITER's on GH200. The fit configs differ from
  the gold fit's only in the train/dev HDF paths (the corrupted targets) and the model dir (review). The GPU peak
  is estimated at 2-3 GiB from a worst-case batch (229 phones, 823 units) measured on CPU.

## Results

### G1.F corrupted-phi fits (2026-09-25, 14:51-15:05)

Read `reports/read_g1f_2026-09-25.md` (`analysis/p1_nesting.py`, the fits' learning_rates files and logs).

| rho | corruption job | realised rate | adjacent repeats gold / corrupted | fit | dev NLL/frame ep8 | GPU, wall |
|---|---|---|---|---|---|---|
| 0.7 | `CorruptSeedGoldJob.aj2XQaDlED2E` | 0.699905 | 1705 / 14293 | `ReturnnTrainingJob.ruLnJFWyifwp` | 5.051149 | A10, 8:42 |
| 0.8 | `H9GbhLPtIthV` | 0.800081 | 1705 / 14706 | `6IkAAuzcBwBR` | 5.117938 | A10, 9:17 |
| 0.9 | `wsKq52Dk68b4` | 0.899946 | 1705 / 14942 | `uO8wkodbR2uh` | 5.127337 | A10, 9:02 |
| gold | (none) | 0 | 1705 | `Ac2eioZbRX7d` (P0) | 3.274249 | L40S, 4:48 |

- (A) Realised rates within 0.700 / 0.800 / 0.900 +-0.005: PASS.
- (A) Nesting (r70 in r80, r80 in r90): 0 positions outside the subset, 0 symbol mismatches, 0 length mismatches:
  PASS.
- (A) Each fit ends at epoch 8 without error (no Traceback, ERROR or nan): PASS.
- (A) Epoch-8 dev NLL above gold 3.2742 and r70 < r80 < r90: PASS. The step from r80 to r90 is small (+0.009); at
  epoch 4 r80 (5.1339) sat above r90 (5.1331). Only epoch 8 enters the clause.
- (B) The r70 corruption equals JUPITER's banked `CorruptSeedGoldJob.mnzDC7XJX00r` on every field (0.699905;
  1705 / 14293). S on the 260 set is not yet read.
- The fits train on the corrupted targets: the launch review diffed the written configs, which differ from the gold
  fit only in the train/dev HDF paths and the model dir. Their dev NLL sits 1.8 nats per frame above gold's.
- Verdict: G1.F PASS.

### G1.G core phi reads (2026-09-25, 14:38-14:44; V100 cn-32)

Code 8280d6e1f (review PASS, `reports/review_g1g_core_phi_reads_2026-09-25.md`); launch
`reports/launch_g1g_2026-09-25.md`. The i6 gold phi is P0's `ReturnnTrainingJob.Ac2eioZbRX7d` at epoch 8. Its D4
dev-other genmarg posterior decodes are `ReturnnForwardJobV2.RFaMyWRYUUQN` (trigram, Slurm 4360692, 1 min 13 s)
and `4hVXB1LWkw5p` (uniform, 4360693, 2 min 27 s); the reads are `DevOtherPhoneReadJob.4eT8Gj5YOGWO` / `Vk8VRJVfKaZd`,
`output/phone_read.txt`. 500 of 500 utterances decoded, 0 impossible, N = 29,690 gold phones.

| read | trigram (R1) | uniform (R2) | JUPITER banked |
|---|---|---|---|
| Hungarian PER | 0.1960 | 0.2806 | 0.193 / 0.276 |
| direct PER | 0.1960 (S 3004, D 2149, I 666) | 0.2806 (S 3547, D 4126, I 657) | |
| Hungarian map | identity, 40 of 40 symbols | identity, 40 of 40 | |
| token NMI(symbol, phone) | 0.8291 (27,794 tokens) | 0.7953 (25,806) | |
| frame NMI (i6 only) | 0.5648 | 0.5351 | |
| E[d] of phi's table, 39 phones / SIL | 5.0966 / 4.9991 | same phi | |
| E[d] of the decode, non-SIL / SIL | 4.4065 / 5.3420 | 4.8172 / 6.1766 | |

- (A) R1 Hungarian PER 0.1960, within 0.193 +-0.02 (+0.003): PASS.
- (A) Unit tests: 92 CPU tests pass, 20 of them new (D4 sample, Hungarian map, NMI and E[d] against hand oracles).
  The 260 set is 285 minus 25, on disk: PASS.
- (A) Every P0 job id unchanged (164 ids; code review): PASS.
- (B) R2 0.2806 against 0.276 (+0.005). The map is the identity, as expected of a gold phi, so direct PER equals
  Hungarian PER under both priors. The uniform prior mainly adds deletions (2149 to 4126).
- Verdict: G1.G PASS. The dev-other generative phone read is ported and reproduces JUPITER's gold-phi row to
  0.005. It is the standing read for every phi run from now on.
