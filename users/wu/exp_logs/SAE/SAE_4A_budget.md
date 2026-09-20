# SAE §4a — Training budget on the cold blank-free bed

## State

Phase opened 2026-09-20 on the user's direction: every cold arm so far stopped at one epoch
(4 sub-epochs, 228 updates); the hyperparameter audit (`SAE_4A_attrib.md`, "Hyperparameter
review") read that budget as a claim-scoping defect. This phase funds 50 and 100 sub-epochs with a
learning-rate and temperature schedule for three arms: the lattice-prior bed alone (control),
lattice prior + back-translation auxiliary (jointly trained), and coverage term + lattice prior.
Built and reviewed 2026-09-20 (commits 75c41af schedule knobs + config, e2b3f59 pack job, 83bd84c
BT port on the blank-free topology; reviews `reports/sae_budget_review2_2026-09-20.md`,
`sae_budget_pack_review_2026-09-20.md`, `sae_budget_bt_review_2026-09-20.md`, all PASS or
PASS_WITH_CONCERNS; design review APPROVE_WITH_AMENDMENTS, amendments applied below).
**Node A RUNNING** since 2026-09-20 13:43: `PackedBlankfreeTrainJob.ks7CbtlvpcIL`
(alias `sae/4a/budget_pack/node_a/training`, Slurm 1907749_1, node jpbo-008-46; arms ctrl_50,
odmprior_50, ctrl_100, odmprior_100). **Node B RUNNING** since ~14:02: `.reEI2Nd0S77A`
(Slurm 1908003_1, jpbo-015-35; bt_50, odmbt_50, bt_100, odmbt_100). **Node C RUNNING** since
~14:02: `.4QzmftNlbErt` (Slurm 1908002_1, jpbo-015-24; nosched_ctrl_50/100, nosched_odmprior_50/100
= constant LR, original 4-sub-epoch anneal held at 2: the schedule-vs-budget control, design review
item 3). Wiring commit ea487f0 (`reports/sae_budget_wiring_2026-09-20.md`; 12 arms, paired
PairedPerDeltaJob reads registered per kept epoch: each arm vs ctrl_N, nosched vs its scheduled twin,
ctrl_N ep4 vs banked priorshuf ep4). One manager `sae_4a_budget_pack` pid 2945974
(`log/sae_4a_budget_pack.manager.20260920T135909.log`), watcher
`bash ~/.claude/skills/sis/sis_watch.sh 2945974 config/sae_4a_budget_pack.py 600` (re-arm first
after any session resume). First 11.5 h kills expected 2026-09-21 ~01:14 (A) / ~01:33 (B, C); the
100-arms resume then (first real test of the pack resume path: check each node's log.run.2 for
"skipped" on the finished 50-arms and each 100-arm's start epoch). Do not edit the blank-free
modules while nodes run (re-imported on resume). Job dirs:
`work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.{ks7CbtlvpcIL,reEI2Nd0S77A,4QzmftNlbErt}`.
NEXT: read ep1 wall time per arm off each node's learning_rates (packed step rate; bt arms carry an
extra BT step per EMC step); ep4 / ep10 reads as the eval jobs land; the 11.5 h resume check.

## Objective

Does any cold blank-free arm leave the content-free band (dev-other greedy PER 0.83-0.91) when the
budget is raised 12.5x / 25x and the optimiser is given a schedule? Every earlier FAIL reads "no
take-off within 228 updates"; this phase measures the budget axis directly. It does not change the
objective, the bed, the prior fit or the recognizer.

The objective's relation to the reverse-KL distribution-matching target, and why tau ends at 2, is
derived in `SAE_4A_objective.md` (reference note, 2026-09-20).

## Bed and constants (inherited, `SAE_4A_blankfree.md` "Registered first model" and `SAE_4A_attrib.md` Step 5)

priorshuf bed: cold blank-free model (40 outputs, stride 3, VAD-masked L15, K500 reverse units),
exact trigram lattice prior with lower-order backoff (`PhoneNgramPriorJob.RtzbESkOedsT` on the
uniform-sample window `SampleLinesJob.orN768ARKwlt`), prior_weight 1, lam_rate 3 (rho 9.66/s),
lam_agg 0.1 order 2 (KL form) on the control, batch 88000 padded frames / max_seqs 128 (57 updates
per sub-epoch, partition_epoch 4), Adam (0.5, 0.98), theta LR 1e-4, phi LR 3e-3 (multiplier 30),
clip 5, weight decay 0. Reference 4-sub-epoch corners: priorshuf ep4 0.8829, odm3_prior ep4 0.8756
(both dev-other, `SAE_4A_attrib.md` Step 5b table).

## Design (pre-registered 2026-09-20, before any launch)

Six runs, n = 1 seed each (disclosed), N in {50, 100} sub-epochs:

| arm | delta from the priorshuf bed | question |
|---|---|---|
| ctrl_N | schedule only | budget/schedule alone (attribution control for the other two) |
| bt_N | + BT auxiliary, lam_bt 0.1, depth full, ramp 0 -> 0.1 over the first 20 % of N | user item 1: lattice prior + BT, jointly trained |
| odmprior_N | + order-3 ratio coverage term, lam_agg 0.009 (the banked odm3_prior config) | user item 2: coverage + LM prior |
| odmbt_N | odmprior_N + the BT auxiliary of bt_N (user 2026-09-20) | the two mechanisms feeding each other: coverage moves phi, BT amplifies it |

User decisions 2026-09-20: combined arm added; arms packed four per node (the partition allocates
whole 4-GPU nodes); max_seqs stays 128 because the coverage term is a per-batch n-gram estimate
(Liu et al. 2017: batch and estimator dominate ODM) and a smaller batch would starve it. If ODM is
batch-limited, the follow-up lever is accumulating coverage counts over k steps, not a smaller batch.

Schedule (same for every arm; all lists are per sub-epoch, length N):
- temperature: geometric 8 -> 2 over the first ceil(0.2 N) sub-epochs (10 / 20), held at 2 after
  (the 4-sub-epoch anneal was designed for an 8-sub-epoch run; the stretch is proportional-ish and
  is a stated choice, not a tuned value).
- base learning rate (theta; phi rides the inherited 30x multiplier): linear warmup 0.1x -> 1x over
  the first 5 % (3 / 5), hold 1x until 60 % of N (30 / 60), linear decay to 0.1x at N. Peak values
  unchanged (1e-4 / 3e-3): the hyperparameter audit found no scalar worth retuning.
- kept checkpoints: N=50: 1, 4, 10, 25, 50; N=100: 1, 4, 10, 20, 25, 50, 75, 100; keep_last_n 1 for
  resume. Registered evaluation (PER S/D/I, greedy rate, derangement gap, dev-other and dev-clean)
  at every kept checkpoint; label-free selection (weighted LM perplexity, as the S3b lam3 read)
  reported alongside, never best-PER.
- job: a resumable variant of `BoundedBlankfreeTrainingJob` (resume="run"), time rqmt 11.5 h (the
  engine cap); N=100 needs one auto-resume (~10 min per sub-epoch measured on the control:
  ~8.5 h / ~17 h per run). The partition allocates whole 4-GPU nodes, so arms run four per node in
  `PackedBlankfreeTrainJob` (per-arm configs byte-identical to the single-arm jobs, resume after the
  11.5 h kill via each arm's newest checkpoint; reviewed `reports/sae_budget_pack_review_2026-09-20.md`:
  the resume path is verified by reading, not yet by a real kill; a node whose remaining arms all
  crash lands in error state and needs manual clearing).
- BT on the blank-free topology (port, second implementer round): the S3b `bt_aux` step is CTC on
  41 outputs at stride 1 and cannot run here. Same mechanism as S3b-BT-aux (`SAE_4A.md`, "S3b-BT-aux"):
  after every EMC step one separate BT optimizer step on theta; units for real T_phi sentences from
  the live phi under no_grad, collage-rendered from the retained-frame pool with sampled durations
  and speaker; BatchNorm stats frozen on the synthetic batch; phi gets no BT gradient. Loss: the
  blank-free run-collapse full-sum (the tested unrestricted pushforward of q onto the target
  string), targets with adjacent identical phones collapsed to one (the recognizer cannot emit them;
  disclosed), pairs with fewer than |y| recognizer output frames masked out, with a feasible-fraction
  monitor. Frame pool and collage are clock-agnostic and are reused.

Gate **G4a.4** (per arm, dev-other, read at the final sub-epoch OR at the label-free-selected
checkpoint if a selector exists for this bed; never best-PER over the kept set): greedy PER < 0.50
AND greedy emitted rate in [5.80, 14.49]/s; health clause: speaker-matched derangement gap > 0 (point
estimate; the gap job emits no CI, and the gap is positive in every content-free arm, so it cannot
carry the decision). Paired reads (PairedPerDeltaJob, registered): each bt_N / odmprior_N / odmbt_N
vs ctrl_N (same N), and ctrl_N vs the banked priorshuf ep4 (budget PLUS schedule: ctrl_N's ep4 sits
inside the warmup/anneal). Within-band paired deltas are not content (attrib step 6 item 4). A PASS
in any arm is audited from a fresh context before it is written up and needs a second seed before
it is claimed; a negative reads "no take-off in this seed" (n = 1).
Pre-registered prediction (from the hyperparameter audit, its "M6 signature"): PER >= 0.80 at every
kept checkpoint while the health statistics (coverage CE, gap, rate) improve.
Design review 2026-09-20 (`reports/sae_budget_design_review_2026-09-20.md`, APPROVE_WITH_AMENDMENTS;
amendments above): SPDG deferred (step 5b shows the coverage objective is satisfiable content-free,
so estimator bias is not shown to bind; odmprior_N's KL3 trajectory decides); a free slot goes to a
schedule-free budget control (constant LR, 4-sub-epoch anneal) before a second ctrl seed.
Abort rule per arm: NaN or |trained surrogate| > 100 in any sub-epoch; or expected phone rate
(`emc/expected_phone_rate_hz` monitor) < 0.6 rho for 5 consecutive sub-epochs after the anneal
ends. An aborted arm is read FAIL (collapse) and not restarted with a different constant.

Levers explicitly NOT in this round (candidates if every arm stays in the band):
- more updates per epoch by a smaller batch (max_seqs 128 -> 32) for the ctrl / bt arms only; rejected
  for the coverage arms (see the user decision above); coverage-count accumulation over k steps
  as the ODM-side alternative (the working GAN reference ran 150k updates);
- tau ending at 1 instead of 2; phi/theta LR ratio other than 30; lam_agg 0.1 with the prior on.
- SPDG (Liu et al. 2017 primal-dual Empirical-ODM) as the coverage estimator, user question
  2026-09-20. Literature (`reports/lit_spdg_2026-09-20.md`): the current "ratio" form is plain SGD on
  the forward CE with an EMA denominator, i.e. the biased family of their Table 1 (SGD 56 % at batch
  10k vs SPDG 9.6 %); the paper's dual constants (nu ~ U(-1,0), mu_v 1e-4, C = 29, N = 2, linear
  softmax) do not transfer to C = 40, N = 3 where |nu*| = 1/u is 1e3-1e6 (the dual would sit at its
  init), the printed Eq. 14 has a sign error, and SPDG has never been run on a segmental model. An
  SPDG arm needs its own reparametrisation (nu = -exp(a), init at -1/p_text) and a small screen
  before it takes a slot; deferred until odmprior_N's KL3 trajectory is read.

## Results

(none yet)
