# SAE 4A InfoMax: conditional-entropy penalty and augmentation invariance on the cold blank-free bed

## State

Phase opened 2026-09-20 on the user's direction (InfoMax proposal, then "augmentation invariance
could help there; consider this a new arm in phase 4A, implement and execute now"). Four arms,
one packed node (node_d), N = 50 sub-epochs, all reads paired against the budget round's ctrl_50
(`SAE_4A_budget.md`, node_a, `PackedBlankfreeTrainJob.ks7CbtlvpcIL`). Status: spec written,
implementer and design review dispatched in parallel; no job launched yet.
NEXT: code review of the implementer's change, then launch node_d through the executor; then the
ep1 / ep4 / ep10 reads.

## Objective

`SAE_4A_objective.md` sections 3 and 5 argue that the content-free band is a stationary point of
the exact reverse-KL bound, not only of our approximation: with an uninformative reverse model the
generative posterior is the prior, independent of the audio, and the reverse model trained on that
recognizer's output stays uninformative. Nothing in the current loss has a gradient out of that
point. This phase tests the user's InfoMax proposal as the symmetry breaker:

    min_theta  D(Q^Y, P_Y) + lambda * H_theta(Y | X),   lambda > 0,

where the divergence is the existing lattice + agg + rate machinery and the new term is the
recognizer's per-frame conditional entropy, minimized. Prediction (Regularized Information
Maximization, Krause, Perona, Gomes 2010): the penalty sharpens each frame toward its own current
preference, which is input-dependent from initialization, and the marginal terms forbid the one
input-independent sharp solution, so the recognizer leaves the band toward a confident partition.
The penalty selects confident partitions, not correct ones: a partition tracking speaker or loudness
is also confident. The second factor, augmentation invariance (the registered S3b-C consistency
term, `emc/consistency.py`, clean teacher detached, KL(clean || aug) per frame, BatchNorm running
statistics frozen on the augmented pass), removes nuisance-tracking partitions from the confident
set. Together the two are IMSAT (Hu et al. 2017, information maximization with self-augmented
training).

No labels anywhere (standing). The phase changes the objective; it does not change the bed, the
prior fit or the schedules of the budget round.

## Bed and constants (inherited from `SAE_4A_budget.md` ctrl_50, unchanged)

priorshuf bed, cold blank-free model (40 outputs, stride 3, 1-layer conv kernel 9 on 1024-dim
VAD-masked L15 at 50 Hz, dropout 0.1, BatchNorm), exact trigram lattice prior, prior_weight 1,
lam_tau 1, lam_rate 3 (rho 9.66/s), lam_agg 0.1 order 2 KL form, batch 88000 padded frames /
max_seqs 128 (57 updates per sub-epoch), Adam (0.5, 0.98), theta LR 1e-4 with the budget warmup /
hold / decay, phi LR 3e-3, clip 5, temperature geometric 8 -> 2 over the first 10 sub-epochs then
2, kept checkpoints 1, 4, 10, 25, 50. Operating point of ctrl_50 at sub-epoch 19 (node_a,
`reports/extract_nodeA_ep1_2026-09-20.md`): l_tau 1.79 nats per unit frame (about 5.4 per output
frame), agg 1.29, reverse -3.1 per frame, phone rate 9.0/s, 617 s per sub-epoch, 10.6 s per step,
so N = 50 fits one 11.5 h allocation without a resume.

## Design (pre-registered 2026-09-20, before any launch)

Arms, N = 50, one seed each (disclosed), a 2 x 2 on (schedule of lambda) x (invariance):

| arm | lambda_ent per sub-epoch | invariance | question |
|---|---|---|---|
| ent_50 | geometric 0.1 -> 0.001 over sub-epochs 1..10, then 0 | none | does the penalty alone break the stationary point, and does the band survive once it is withdrawn |
| enthold_50 | 0.1 held for all 50 | none | hysteresis / over-confidence of a held penalty |
| entaug_50 | as ent_50 | S3b-C consistency, specaug view + statistic-swap view | does invariance steer the confident partition toward content |
| entaughold_50 | as enthold_50 | as entaug_50 | the held variant with invariance |

Not run, with the reason: invariance alone (theory: its gradient is zero at an input-independent
recognizer, so it cannot break the point; S3b-C's own caveat says the same); a lambda sweep (the
user asked for a modest lambda; 0.1 is sized below).

The entropy term: on the clean recognizer log posterior the lattice already ran on, per output
frame `H_t = -sum_k q_t(k) log q_t(k)` over the 40 outputs, mean over the valid output frames of the
batch, marked as loss `ent` with scale lambda_ent(sub-epoch) from a per-sub-epoch list handled like
`temperature_schedule`. Sizing: the entropy is at most log 40 = 3.69 nats per output frame and near
3 in the band; at lambda 0.1 the term is under 7 % of the lattice value per output frame, and its
per-logit gradient (at most about q_k |log q_k + H|, order 1, normalized per output frame) is of
the same order as the lattice's (1/tau)(q - post) normalized per unit frame. At the stationary
point the lattice gradient is near zero, so any nonzero lambda dominates there; 0.1 is chosen so
that away from it the marginal terms can still win. Reported as_error for these arms only:
entropy per output frame.

The invariance term: `consistency.consistency_loss` as registered for S3b-C (teacher = clean view
detached, KL(clean || aug), mean over valid clean frames pooled over views, augmented passes under
`frozen_batch_norm_stats`), weight lam_cons = 1.0, constant. Views, both frame-aligned so the KL
is frame for frame:
- specaug: the S3b-C `DEFAULT_SPECAUG` on the 1024-dim features;
- statswap (new, the speaker / loudness proxy): per utterance, replace the utterance's own
  per-dimension mean and standard deviation over its valid frames by those of a different
  utterance of the same batch (a fixed derangement of the batch; a batch of one skips the view),
  `x' = (x - mu_i) / sd_i * sd_j + mu_j`. The utterance-level feature statistics carry speaker and
  channel (SAE `speaker.py` builds its speaker code from the utterance-mean L15), so invariance to
  swapping them is invariance to that code without touching the frame sequence.
The speed view of S3b-C is not used: it needs a second feature dump the VAD-masked bed does not
have.

Job: `PackedBlankfreeTrainJob` node_d in a new config `config_sae_4a_infomax_pack_v1.py`
(workspace `config/sae_4a_infomax_pack.py`), per-arm RETURNN config byte-identical to ctrl_50's
except the new model arguments; alias `sae/4a/infomax_pack/<arm>/...`. Registered reads at every
kept checkpoint as in the budget round (PER S/D/I, greedy rate, derangement gap, dev-other and
dev-clean, label-free selector). Paired reads (PairedPerDeltaJob): each arm vs ctrl_50 at the same
sub-epoch; entaug_50 vs ent_50 and entaughold_50 vs enthold_50 (the invariance effect); enthold_50
vs ent_50 (the schedule effect).

Constraint on the code: nodes a, b, c re-import the blank-free modules at their 11.5 h resume
(2026-09-21 about 01:14 / 01:33). Every edit to a module they import must be behavior-neutral at
the default arguments and import-safe; new logic lives in a new module.

## Gate

**G4a.5** (per arm, dev-other, read at sub-epoch 50, or at the label-free-selected checkpoint if a
selector exists; never best-PER over the kept set): the budget round's G4a.4 thresholds, greedy PER
< 0.50 AND greedy emitted rate in [5.80, 14.49]/s, health clause derangement gap > 0. Read against
ctrl_50 at the same sub-epoch through the paired job.

Pre-registered early read at sub-epoch 10 (not a gate): symmetry breaking is declared when the
entropy per output frame is below 1.0 nat AND the paired dev-other PER delta vs ctrl_50 is outside
the band, i.e. PER below 0.80. Entropy below 1.0 with PER still in 0.83-0.91 reads "confident but
content-free" and is the case the invariance arms exist for.

Abort rule per arm (the budget round's, plus one): NaN or |trained surrogate| > 100 in any sub-epoch;
expected phone rate outside [4, 20]/s at the end of any sub-epoch after the fifth; entropy per
output frame below 0.05 nats together with a rate outside that window (collapse to a constant
output). An aborted arm reads FAIL (collapse) and is not restarted with another constant.

Pre-registered predictions: entropy leaves 2.5-3 nats and drops below 1 within 10 sub-epochs in all
four arms; ent_50 either takes off or drifts back toward the band after sub-epoch 10; the
invariance arms have lower PER than their non-invariance twins if the confident partition without
invariance tracks a nuisance. A PASS is audited from a fresh context and needs a second seed before
it is claimed.

## Results

(none yet)
