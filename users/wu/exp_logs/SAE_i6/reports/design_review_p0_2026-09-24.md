# Design review, P0 (port to i6): reproduction and verification gates, 2026-09-24

Verdict: **APPROVE_WITH_AMENDMENTS**. The three runs and the four gates are the right shape, and
nothing in the spec is wrong by construction; but as written, G0.R1 and G0.R3 can pass for a
broken port and G0.R2 can fail for a correct one, and one input-graph clause stalls the whole
graph under the audio deviation the phase already anticipates. Read-only review; nothing on
disk changed except this file.

Read: `SAE_i6.md`, `SAE_i6_P0.md`, `SAE_i6_ref.md` s2-4, `SAE_i6_ref_blankfree.md` s1, 3, 6,
`SAE_i6_ref_lexicon.md` B6-B8, B11, B13, `SAE_i6_ref_lexlat_v2.md` (Open at the move),
`SAE_i6_ref_objective.md` s10, `reports/test_plan_2026-09-24.md` s6, `reports/impl_settings_p0_2026-09-24.md`,
`reports/impl_tests_reverse_prior_agg_2026-09-24.md`, `reports/audit_ref_distill_2026-09-24.md`,
`reports/env_build_2026-09-24.md`; package: `README.md`, `inputs.py`, `data/vad.py`,
`training/{arms,jobs,config}.py`, `config/{common,supervised_init}.py`, `reverse_model/{p0,supervised_steps}.py`,
`model/{train_step,lattice,reverse}.py`, `tests/`; JUPITER provenance `exp_logs/SAE/SAE_4A_lexlat.md`
(l.490, 527-535, 558, 750-756), `SAE_4A_prepro.md` (l.13, 222-236).

## Findings, most material first

**F1. Under `FFMPEG_PIN_ACCEPT` the VAD job raises on the banked frame counts; the graph stalls
after the GPU feature dumps.** `inputs.py:139` passes `expected_counts=BANKED_VAD_COUNTS`
unconditionally and `data/vad.py:140` raises on any mismatch. The pin check is expected to fail on
x86_64 (README l.131-135); a different Ogg generation moves rVAD decisions on some of the 18 M frames,
so the totals will not be equal. The VAD job takes the L15 feature HDFs as inputs, so the 7 GPU
dumps (A10, hours) run first, then the graph stops. P0 names "the audio generation label" as a
permitted deviation but has no clause for what the VAD count check does under it. Predicted
failure: a stalled input graph and a manual fix after compute is spent. Owner decision needed
before launch (an implementer change: report-only counts under an accept label, or a documented
plan to let it fail and relaunch).

**F2. G0.R2's paired-delta window fails the banked identical-config replicate.** Window
[-0.0760, -0.0360]. D15 (`_lexicon` B13, l.421) reran `k2lat_20_ma3000` with an identical config:
`k2lat_rep` 0.8390 against the parent 0.8186 (+0.0204), i.e. a delta against ctrl_20 (0.8746) of
about -0.036, at or just outside the window. So a correct port has a real chance to FAIL R2 on
the bed's own noise. The +-0.02 is not traced to any measurement; the measured k2-arm ep20
replicate spread is 0.02. Also R2 does not test the lexicon path's content: every deranged-lexicon
null gave the same PER drop (B8: k2shuf_20_ma3000 -0.0604 vs treatment -0.0560), so a wrongly built,
shuffled or mis-parsed HLG passes R2. The banked quantities that do separate treatment from null are
expected words per utterance (27-34 vs 16-20, D2/B8 l.533), escape share (in-house treatment 0.0001)
and the graph size (in-house word-boundary HLG 23.9 M states / 98.6 M arcs, B6). These are logged
(`lexlat_k2_expected_words`, `lexlat_k2_expected_escape_words`, `build.json`) and cost nothing.

**F3. G0.R3's reference 3.2888 has no provenance; its tolerance has no noise measurement; the
quantity is audio-dependent.** The distillation audit (row 26, "What would be needed to close")
found the number in no JUPITER log; it exists only in the package README. n = 1, no replicate;
the fit's held-out set is 28 utterances; its units come from k-means on the audio, so under a
different audio generation the read cannot be interpreted. Meanwhile the graph already trains
p0 (`ReturnnTrainingJob.CqZ9Wr1y9sxl`), whose dev-other PER 0.1894 IS logged on JUPITER
(`_lexicon` B10; `SAE_4A_lexlat.md` l.490 names the file), is a supervised, near-deterministic
read (selected at pass 1, 24 Adam steps), and exercises features, VAD, gold, the recognizer and
the PER chain end to end; but `reverse_model/p0.py:36-37` states its PER reads "are not built
here". A wrong feature layer, VAD rule or collapse rule would move 0.19 far more than it moves
any content-free arm. Predicted failure as written: R3 passes or fails on an untraceable number
while the traceable supervised anchor is never read.

**F4. The step-1 clause is the only sharp part of R1 and its tolerances do not separate the
reference batch from the seed-replicate batch on two of three numbers.** ctrl_20 step 1:
l_tau -0.350 / prior per token -5.657 / expected tokens 63.821; ctrl_20_s1 (other flat seed,
other data order): -0.347 / -5.671 / 57.498 (`SAE_4A_lexlat.md` l.490). The clause reads l_tau
+-0.01 (includes -0.347) and omits prior per token; only "tokens within 2 %" discriminates. On
JUPITER twelve arms matched these to 1e-4 (step lines to three decimals, ep1 means to 2e-7,
l.535); phi is initialised on the CPU generator, so cross-hardware differences are fp64 GEMM
ordering only. The clause can therefore be much tighter. It also survives a different audio
generation better than the spec assumes: at the flat init the features enter only through the
random phi's emission lookups, so l_tau, prior per token and expected tokens depend on the batch's
retained lengths and the prior, not on feature values. The clause should stay a gate under a
different audio label, at a widened tolerance, rather than be dropped.

**F5. No cost or memory screen on the L40S before four 20-sub-epoch trainings, with an fp64
GEMM in every lattice step.** `model/lattice.py:651-672`: "the exp / GEMM / log accumulation is
ALWAYS float64". GH200 (Hopper) runs fp64 GEMM on tensor cores; Ada (L40S) has no fp64 tensor
cores and runs fp64 at about 1/64 of fp32. The 601 s per sub-epoch and the 11.5 h `TIME_RQMT`
(`training/jobs.py:20`) are GH200 figures; `impl_settings_p0` flags this but no clause reads it.
Also `GPU_MEM_RQMT = 96` on a 46 GB card; the banked whole-step peak is about 32 GiB for the k2 arm
(B6), so the batch shape probably fits, but it is unmeasured. Predicted failure: repeated
wall-clock kills and RETURNN resumes on every arm (each resume a disclosed trajectory break), or a
5-10x cost that changes what the campaign can afford. The reads exist: `blankfree_frames_per_sec`
in `learning_rates`, per-step timing in the log at verbosity 5, peak memory in the log.

**F6. The i6 seed band is not produced, although every later gate's M is built on it.** Later
packs pair against the i6 ctrl_20 with M = max(|B|, |F|, 0.010); B is the ctrl_20 - ctrl_20_s1 band.
The banked band is +0.004 / +0.002 / -0.011 / -0.001 at ep1/4/10/20 (`SAE_4A_prepro.md` l.232;
the ref's sign label is flipped, audit row 124). ctrl_20_s1 = flat_seed 1 / random_seed 1 /
random_seed_offset 1000 (`SAE_4A_prepro.md` l.13; `training/config.py:214-215` and
`training/init.py` support all three). It also gives a second, independent step-1 identity point
(-0.347 / -5.671 / 57.498) with a different first batch and phi init, which is the single best
port check available. Cost: one of the five L40S slots, no wall-time extension (P0 has four
trainings).

**F7. G0.V names a bar whose k2 half does not exist yet and omits the assembly tests.** No
`tests/test_model_lexlat_k2.py` (T1.18-T1.23) exists; those are the tests that measure S3-S6 and
the only ones that check the source-built k2 numerically. T2.1-T2.3 (assembly and weights,
configured constants, schedule reading) are labelled P2 but are where a lam scale or a tau/LR
off-by-one would hide; none of P1 catches that. The gpu-marked tests (T1.8) and the k2 tests must
run on a `gpu_48gb` node (sm_86 SASS on sm_89, `env_build` l.40), and there is no CPU-vs-CUDA
parity for `log_z_hlg`. `impl_tests_reverse_prior_agg` pinned a real defect (FINDING 1,
`model/prior.py:246`, one-token double count) with strict xfails; G0.V must say how such xfails
count (outside the train step: allowed and listed; inside: blocker).

**F8. Banked scalars beyond PER that R1 can read at no cost** (CV-holdout dev columns in
`learning_rates`, `per.json`, `derangement_gap.json`): dev l_tau at ep20 1.811 (E60 l.751), dev
agg 1.516 (l.753), reverse per frame -3.26 (s1 -3.11; D2 addendum l.562), emitted greedy rate
9.16 /s (s1 9.03) and derangement gap 4.27 (s1 4.60) at ep20; ep1 rate 3.30, gap -0.007
(`SAE_4A_prepro.md` l.226-227); prior held-out ppl 9.56 and rho 9.6619373279 (T3.7). For the k2
arm: dev lexicon term 0.310 at ep20 (E60 l.752), stability <= 0.044 from sub-epoch 11,
empty_frac <= 0.001, derangement gap 3.61, emitted rate 7.3-8.1 /s (l.533). The seed spread is
known only for gap (0.33), rate (0.13) and reverse (0.15); l_tau and agg have no replicate, so
their tolerances must be registered as a debugger trigger, not a FAIL.

## The five questions

1. **Do R1-R3 discriminate?** Not as written. PER +-0.03 at ep4-20 spans most of the 0.83-0.91
   band; the k2 PER delta is lexicon-agnostic; R3 rests on an untraceable number. The sharp
   reads are: the step-1 triple (both seeds), ep1 PER (twelve arms within 0.0007 on JUPITER),
   p0 PER 0.1894, the HLG size, expected words / escape share, and the ep20 dev terms. With those,
   a broken port has to hit five independent banked scalars by luck.
2. **Tolerances.** Per quantity and epoch, from the measured spreads: ep1 PER +-0.01; ep4 +-0.03
   (identical-config spread up to 0.026); ep10/20 +-0.03 (k2 replicate 0.020; ctrl seed 0.001, no
   identical-config replicate); step-1 l_tau +-0.002, prior/token +-0.005, tokens +-1.0 (JUPITER
   agreement 1e-4; the s1 batch sits 0.003 / 0.014 / 6.3 away); paired delta banked +-0.03, CI
   upper bound < -0.015; gap +-0.6, rate +-0.3, reverse +-0.3 (2x the seed spread); l_tau, agg,
   lexicon term +-0.05 / +-0.15 / +-0.03 as debugger triggers.
3. **If the pin fails:** first, the pin check itself is cheap (1 min CPU) and is the first job;
   read it before deciding anything. If it fails and no passing x86 build is found: (a) F1 must be
   resolved; (b) hashes move, results are "reproduction on other audio" (README); (c) clauses that
   stay valid unchanged: PER +-0.03 at all epochs, the paired delta, HLG size, expected words,
   escape share, prior ppl, rho, k2 monitors; widened: ep1 PER +-0.015, step-1 triple
   +-0.005 / +-0.01 / +-3 %; report-only: gold-phi NLL, VAD totals (expect within 0.5 % of banked;
   more means the VAD rule, not the audio, differs); p0 PER stays a gate at +-0.02 (supervised,
   robust to 62 dB perturbations).
4. **G0.V.** Right in spirit; amend per F7. The owner decisions for S1, S2, S7 are already
   recorded (`_objective` s10: the code defines the bed), so that part is satisfied now.
5. **Missing for later:** ctrl_20_s1 (F6, add). p0's PER read (F3, add). x60 runs: not for P0
   (E60 plateau stands); but the L2-1 wave needs phi_c = `k2lat_20_x60` ep60, which has no preset
   and no banked checkpoint on i6, so if the user re-runs the wave here it is a 60-sub-epoch k2
   job to plan then, not now. `k2_word_lm` default: never run, no banked number, and S3 (pruned-G
   normalisation) is open; not a reproduction target. An identical-config ctrl_20 rerun is the
   debugger's first tool if a Tier-A clause fails, not a P0 run.

## Amended gate text (proposed; the orchestrator registers before launch)

Two tiers. Tier A: a miss is FAIL (debugger before any rerun). Tier B: a miss is a debugger read
before the verdict; REPRODUCED may still be declared if the debugger attributes the miss to
hardware or audio with evidence, recorded in the phase file.

1. **G0.R0 input graph (Tier A).** Pin check result recorded first. HLG `build.json` states in
   [23.8 M, 24.0 M] and arcs in [98.1 M, 99.1 M] (banked 23.9 M / 98.6 M). Prior held-out ppl
   9.56 +-0.02; rho = 9.6619373279 (1e-9 rel). VAD totals: equal to `BANKED_VAD_COUNTS` on
   reference audio; within 0.5 % under an accept label (F1 resolved so the job can report them).
2. **G0.R1 ctrl_20.** Tier A: PER at ep1 0.855 +-0.01 (+-0.015 under an audio label); ep4 / 10 / 20
   0.875 / 0.869 / 0.874568 +-0.03; step 1 l_tau -0.350 +-0.002, prior per token -5.657 +-0.005,
   expected tokens 63.821 +-1.0 (audio label: +-0.005 / +-0.01 / +-3 %; batch change voids it);
   ep20 emitted rate 9.16 +-0.3 /s, derangement gap 4.27 +-0.6. Tier B: dev l_tau ep20 1.811 +-0.05,
   dev agg 1.516 +-0.15, dev reverse per frame -3.26 +-0.3, ep1 rate 3.30 +-0.3.
3. **G0.R1s ctrl_20_s1 (new run; flat_seed 1 / random_seed 1 / random_seed_offset 1000).** Tier A:
   step 1 -0.347 / -5.671 / 57.498 at the R1 tolerances; PER ep20 0.8751 +-0.03. Report the i6
   band ctrl_20 - ctrl_20_s1 at ep1/4/10/20 (banked +0.004 / +0.002 / -0.011 / -0.001) as the B of
   every later pack.
4. **G0.R2 k2lat_20_ma3000.** Tier A: step 1 identical to ctrl_20's (pre-on-set path, l.490);
   PER ep20 0.818615 +-0.03; paired vs i6 ctrl_20 at ep20: CI upper bound < -0.015 and point in
   [-0.086, -0.026]; from sub-epoch 8: `lexlat_k2_expected_words` (dev, ep20) in [24, 38],
   escape share (`expected_escape_words / expected_words`) <= 0.01, `lexlat_k2_stability` <= 0.05
   from sub-epoch 11, `lexlat_k2_empty_frac` <= 0.002, no abort marker. Tier B: dev lexicon term
   ep20 0.310 +-0.03; gap 3.61 +-0.6; emitted rate in [7.0, 8.5] /s.
5. **G0.R3 supervised inits (analysis only).** Tier A: p0 dev-other greedy PER 0.1894 +-0.02
   (needs the posterior-dump + PER wiring on p0's exported checkpoint; report the selected epoch,
   banked 1). Tier B: gold phi `dev_loss_nll_per_frame` at epoch 8 = 3.2888 +-0.02 on reference
   audio, report-only under an audio label; state its provenance as package-banked (audit row 26).
6. **G0.V.** All P1 tests (T1.1-T1.23) and T2.1-T2.4 green; gpu- and k2-marked tests run on a
   `gpu_48gb` node; add a CPU-vs-CUDA parity assert for `log_z_hlg` and `log_z_h` on the T1.19
   fixture (1e-5). Strict xfails are allowed only for defects outside the train step and are
   listed in the phase file (FINDING 1 is one). Owner decisions S1, S2, S7: already in
   `_objective` s10.
7. **Launch order (cost screen, F5).** ctrl_20 launches first and alone among the trainings. After
   its sub-epoch 1: wall time per sub-epoch <= 1800 s (3x GH200) and peak memory <= 40 GiB at the
   reference batch shape, else stop and decide (fp64 GEMM on Ada; batch change; time rqmt). Only
   then k2lat_20_ma3000, ctrl_20_s1, gold phi and p0 are released. The same read gives the step-1
   triple and ep1 PER within the first hour.

## Cheapest falsifying check

Submit the pin check and ctrl_20 only. Within about one hour on one L40S the log gives: the
pin verdict, step-1 l_tau / prior per token / expected tokens against -0.350 / -5.657 / 63.821,
peak memory against 46 GB, seconds per sub-epoch against 601, and ep1 PER against 0.855. Any one of
these off settles whether the cohort should run at all, before the other four GPU jobs and the k2
graph build are spent.
