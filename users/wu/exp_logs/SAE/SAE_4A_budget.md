# SAE §4a — Training budget on the cold blank-free bed

## State

Phase opened 2026-09-20 on the user's direction: every cold arm so far stopped at one epoch
(4 sub-epochs, 228 updates); the hyperparameter audit (`SAE_4A_attrib.md`, "Hyperparameter
review") read that budget as a claim-scoping defect. This phase funds 50 and 100 sub-epochs with a
learning-rate and temperature schedule for three arms: the lattice-prior bed alone (control),
lattice prior + back-translation auxiliary (jointly trained), and coverage term + lattice prior.
Plan shown to the user 2026-09-20 (this file); no job launched. Implementation of the schedule
knobs, the resumable job class and the budget config is in progress (implementer); the BT port to
the blank-free topology follows in a second implementer round.
NEXT: code review of the schedule/config change, then launch ctrl_50, odmprior_50, ctrl_100,
odmprior_100; BT arms launch after the port is reviewed and its tests pass.

## Objective

Does any cold blank-free arm leave the content-free band (dev-other greedy PER 0.83-0.91) when the
budget is raised 12.5x / 25x and the optimiser is given a schedule? Every earlier FAIL reads "no
take-off within 228 updates"; this phase measures the budget axis directly. It does not change the
objective, the bed, the prior fit or the recognizer.

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
  ~8.5 h / ~17 h per run). Each run occupies an exclusive 4-GPU node with one GPU used, as every
  earlier blank-free arm did; no blank-free pack job exists.
- BT on the blank-free topology (port, second implementer round): the S3b `bt_aux` step is CTC on
  41 outputs at stride 1 and cannot run here. Same mechanism as S3b-BT-aux (`SAE_4A.md`, "S3b-BT-aux"):
  after every EMC step one separate BT optimizer step on theta; units for real T_phi sentences from
  the live phi under no_grad, collage-rendered from the retained-frame pool with sampled durations
  and speaker; BatchNorm stats frozen on the synthetic batch; phi gets no BT gradient. Loss: the
  blank-free run-collapse full-sum (the tested unrestricted pushforward of q onto the target
  string), targets with adjacent identical phones collapsed to one (the recognizer cannot emit them;
  disclosed), pairs with fewer than |y| recognizer output frames masked out, with a feasible-fraction
  monitor. Frame pool and collage are clock-agnostic and are reused.

Gate **G4a.4** (per arm, read at the final sub-epoch AND at the label-free-selected checkpoint,
dev-other): greedy PER < 0.50 AND speaker-matched derangement gap > 0 with CI excluding 0 AND
greedy emitted rate in [5.80, 14.49]/s. Paired reads: each bt_N / odmprior_N vs ctrl_N (same N), and
ctrl_N vs the banked priorshuf ep4 (budget effect alone). A PASS in any arm is audited from a fresh
context before it is written up and needs a second seed before it is claimed.
Pre-registered prediction (from the hyperparameter audit, its "M6 signature"): PER stays in
0.83-0.91 at every kept checkpoint while the health statistics (coverage CE, gap, rate) improve.
Abort rule per arm: NaN or |trained surrogate| > 100 in any sub-epoch; or expected phone rate
(`emc/expected_phone_rate_hz` monitor) < 0.6 rho for 5 consecutive sub-epochs after the anneal
ends. An aborted arm is read FAIL (collapse) and not restarted with a different constant.

Levers explicitly NOT in this round (candidates if every arm stays in the band):
- more updates per epoch by a smaller batch (max_seqs 128 -> 32: 4x updates at roughly equal
  GPU-time per epoch, more gradient noise; the working GAN reference ran 150k updates);
- tau ending at 1 instead of 2; phi/theta LR ratio other than 30; lam_agg 0.1 with the prior on.

## Results

(none yet)
