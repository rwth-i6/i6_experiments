# SAE 4A InfoMax: conditional-entropy penalty and augmentation invariance on the cold blank-free bed

## State

Phase opened 2026-09-20 on the user's direction (InfoMax proposal, then "augmentation invariance
could help there; consider this a new arm in phase 4A, implement and execute now"). Four arms,
one packed node (node_d), N = 50 sub-epochs, all reads paired against the budget round's ctrl_50
(`SAE_4A_budget.md`, node_a, `PackedBlankfreeTrainJob.ks7CbtlvpcIL`). Arm set amended after the
code review (Design, "Amendment after code review"): ent_50, entaug_50, aug_50, aughi_50. Code:
2025-10-speech-llm commits 35bcf39 (mechanism, `sae/emc/infomax.py`) and 68549f9 (arm table);
reviews `reports/review_infomax_2026-09-20.md` (PASS_WITH_CONCERNS, concerns applied),
`reports/design_review_infomax_2026-09-20.md`. node_d job `PackedBlankfreeTrainJob.YtvrSez8z9Wf`,
config `config/sae_4a_infomax_pack.py`; **submitted 2026-09-20 17:53** as Slurm 1912407 (pending
at submission; `reports/exec_launch_infomax_2026-09-20.md`). Manager `sae_4a_infomax_pack` pid
1355841 (`log/sae_4a_infomax_pack.manager.20260920T181107.log`, restarted 18:11 to load the reads;
`reports/exec_restart_infomax_mgr_2026-09-20.md`), watcher
`bash ~/.claude/skills/sis/sis_watch.sh 1355841 config/sae_4a_infomax_pack.py 600`; the budget
watcher `bash ~/.claude/skills/sis/sis_watch.sh 2945974 config/sae_4a_budget_pack.py 600` is armed
in the same session (re-arm both first after any session resume). Checkpoint reads (eval-mode entropy from the registered posterior dump, symbol-usage entropy,
chance null; dev-other, 5 arms x 5 kept epochs) registered into the same config (commits speech-llm
de72d86, i6 c8d01d2a3; `reports/impl_infomax_reads_2026-09-20.md`); census 315 jobs / 536 targets,
node_d hash unmoved; ctrl_50 ep1/4/10 reads submitted 18:11. Private-code analysis banked 2026-09-20 19:05 (manager
`sae_4a_private_code`, exited clean; 8 jobs under `work/speech_llm/sae/emc/private_code/`:
`PrivateCodeAnalysisJob.{xuCBRVe3CzMR,DMAmCpn0L5mB,VAYV87DbdDwU}` ep1/4/10 dev-other,
`.{x79av2mvTxo3,eLdlRYU1YtQZ,ssh9wJM8pxjI}` dev-clean, `SymbolDeciphermentJob.{36NfY7XDOL3f,FkH0wprbfjaN}`
ep4/ep10 dev-other; `reports/exec_launch_private_code_2026-09-20.md`); fresh-context audit
dispatched before the tables go into Results.
NEXT: run the private-code config when the implementer reports; read the ctrl_50 ep1/4/10 entropy
and usage-null reads (banked numbers replace the wiring-check values), then the ctrl_50 ep10
private-code table, then node_d ep1 / ep4 / ep10.

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
| ent_50 | 0.1 held over sub-epochs 1..10 (the tau 8 -> 2 anneal window), geometric 0.1 -> 0.001 over 11..20, 0 from 21 | none | does the penalty alone break the stationary point, and does the band survive once it is withdrawn |
| enthold_50 | 0.1 held for all 50 | none | hysteresis / over-confidence of a held penalty |
| entaug_50 | as ent_50 | S3b-C consistency, specaug view + statistic-swap view | does invariance steer the confident partition toward content |
| entaughold_50 | as enthold_50 | as entaug_50 | the held variant with invariance |

Not run, with the reason: invariance alone (theory: its gradient is zero at an input-independent
recognizer, so it cannot break the point; S3b-C's own caveat says the same); a lambda sweep (the
user asked for a modest lambda; 0.1 is sized below).

**Amendment after code review, before launch (2026-09-20 evening; original table above kept).**
The review (`reports/review_infomax_2026-09-20.md`) measured ctrl_50's own eval-mode dev-other
entropy per output frame at 3.25 / 2.71 / 0.30 / 0.19 nats at sub-epochs 1 / 4 / 10 / 21, while
the arm stays in the band. The tau 8 -> 2 anneal alone makes the recognizer confident; the band is
therefore a confident, input-dependent, content-free partition (a theta-phi private code), not the
diffuse stationary point of the Objective. Consequences: the entropy penalty has nothing to bind
on after sub-epoch 10, so the held-penalty arm answers nothing; invariance alone, dropped above
because it has no gradient at a diffuse recognizer, has gradient at a confident one and is now the
live test of "the confident partition tracks a nuisance". The review also measured lam_cons 1.0 at
6.4x the gradient norm of the whole objective (cons 0.85 nats per output frame at ep10: specaug
1.41, statswap 0.29; the term is normalized per output frame, three times the lattice's unit-frame
normalization), so 1.0 is replaced. Launched arm set:

| arm | lambda_ent | lam_cons (specaug + statswap) | question |
|---|---|---|---|
| ent_50 | 0.1 held 1..10, geometric to 0.001 over 11..20, 0 after | 0 | does sharpening before the anneal completes land in a different partition than the anneal's own (path dependence) |
| entaug_50 | as ent_50 | 0.03 | the user's InfoMax + invariance combination |
| aug_50 | 0 | 0.03 (about 20 % of the objective's gradient norm at ep10) | invariance alone, modest |
| aughi_50 | 0 | 0.1 (about 60 %) | invariance alone, strong enough to move the partition if it can |

Paired reads: each arm vs ctrl_50 at matched sub-epochs; entaug_50 vs ent_50 and entaug_50 vs aug_50
(each factor given the other); aughi_50 vs aug_50 (weight). The Objective section's stationary-point
framing stands as the account of the first sub-epochs (entropy 3.25 at sub-epoch 1) and is
superseded from sub-epoch 10 by the private-code reading; `SAE_4A_objective.md` section 6 notes it.

The entropy term: on the clean recognizer log posterior the lattice already ran on, per output
frame `H_t = -sum_k q_t(k) log q_t(k)` over the 40 outputs in nats (no division by log 40), summed
over the valid output frames of an utterance and divided by that utterance's retained unit-frame
count exactly as the lattice term is, then averaged over utterances; marked as loss `ent` with
scale lambda_ent(sub-epoch) from a per-sub-epoch list handled like `temperature_schedule`, so
lambda is in the units of lam_tau. Sizing: the entropy is at most log 40 = 3.69 nats per output
frame; its value in the band is unmeasured (design review amendment 1; banked by the eval-mode read
below). At lambda 0.1 the term is at most 7 % of the lattice value at the same normalization. Its
per-logit gradient q_k |log q_k + H| is small at a diffuse posterior (0.02-0.1, design review), and
the lattice gradient is near zero at the stationary point, so the balance there is not predictable
from constants; the held arms exist to answer whether 0.1 is enough. The decayed arms hold 0.1
through the tau anneal (which flattens q while the penalty sharpens it) and withdraw it only after
tau reaches 2 (amendment 3). Reported as_error for these arms only: entropy per output frame
(pooled over valid output frames, nats) and lambda.

The penalty is a deliberate mode-seeking departure from the reverse-KL bound of
`SAE_4A_objective.md` (whose section 5 item 3 rejects the opposite-sign bonus): the bound's
minimizer is the generative posterior; the penalty targets a sharpened posterior. It is a
cold-start device, which is why one schedule withdraws it.

The invariance term: `consistency.consistency_loss` as registered for S3b-C (teacher = clean view
detached, KL(clean || aug), mean over valid clean frames pooled over views, augmented passes under
`frozen_batch_norm_stats`), weight lam_cons = 1.0, constant. Views, both frame-aligned so the KL
is frame for frame:
- specaug: the S3b-C `DEFAULT_SPECAUG` on the 1024-dim features;
- statswap (new, the speaker / loudness proxy): per utterance, replace the utterance's own
  per-dimension mean and standard deviation over its valid frames by those of a different
  utterance of the same batch (a fixed derangement of the batch; a batch of one skips the view),
  `x' = (x - mu_i) / sd_i * sd_j + mu_j`, std floored at 1e-3, statistics over valid frames only,
  swapped only when both partners have at least 200 valid frames (otherwise the utterance keeps its
  own features), result asserted finite (amendment 6). The utterance-level feature statistics carry
  speaker and channel (SAE `speaker.py` builds its speaker code from the utterance-mean L15), so
  invariance to swapping them is invariance to that code without touching the frame sequence.
The speed view of S3b-C is not used: it needs a second feature dump the VAD-masked bed does not
have.

Job: `PackedBlankfreeTrainJob` node_d in a new config `config_sae_4a_infomax_pack_v1.py`
(workspace `config/sae_4a_infomax_pack.py`), per-arm RETURNN config byte-identical to ctrl_50's
except the new model arguments; alias `sae/4a/infomax_pack/<arm>/...`. Registered reads at every
kept checkpoint as in the budget round (PER S/D/I, greedy rate, derangement gap, dev-other and
dev-clean, label-free selector). Paired reads (PairedPerDeltaJob): each arm vs ctrl_50 at the same
sub-epoch; entaug_50 vs ent_50 and entaughold_50 vs enthold_50 (the invariance effect); enthold_50
vs ent_50 (the schedule effect).

Two reads added on the design review (amendments 1, 5, 7), registered for every kept checkpoint of
the four arms AND of ctrl_50 (its checkpoints exist; ctrl_50 is the band reference the entropy
clause needs): (i) eval-mode dev-other mean per-output-frame posterior entropy in nats (a forward
job over the checkpoint, not the train-mode as_error); (ii) greedy symbol-usage entropy of the
dev-other decode in bits over the 40 outputs (a reader on the decode output; `emc_hyp_inspect`
already computes the usage histogram) together with the arm's own length-and-unigram-matched
chance null (the attrib step 6 null script). The consistency KL is reported per view. Second
implementer round, after the training launch; the ep10 early read waits for it.

Constraint on the code: nodes a, b, c re-import the blank-free modules at their 11.5 h resume
(2026-09-21 about 01:14 / 01:33). Every edit to a module they import must be behavior-neutral at
the default arguments and import-safe; new logic lives in a new module.

## Private-code analysis (user 2026-09-20: "analyze the output / private code when ready";
"say if deciphering can be done")

Registered in its own config `config/sae_4a_private_code.py` on frozen decode outputs of ctrl_50
(ep4, ep10, ep25 when present, dev-other), later on node_d's arms. Parts, each with its
pre-registered reading:
- A. Phone identifiability (label-using diagnostic, never selection): PER as scored, PER after the
  best one-to-one relabeling of the 40 symbols (Hungarian), PER after the best many-to-one
  relabeling, aligned-pair normalized mutual information. Oracle one-to-one PER well below the
  band = the content is in the code and only the labeling is wrong; oracle PER still high = the
  code is not a relabeling of phones.
- B. Unit code: H(symbol | unit), H(unit | symbol), NMI(symbol, unit) against the same for gold
  phones. Symbol nearly a function of the reverse unit while the phone is not = the recognizer
  learned a coarse clustering of the reverse units, the "easy reconstruction" code of
  `SAE_4A_objective.md` section 5 item 2.
- C. Nuisance: NMI(symbol histogram, speaker) against NMI(phone histogram, speaker); symbol-usage
  entropy; run lengths against gold durations.
- D. Sequence statistics: symbol bigram against text bigram after relabeling.
- E. Label-free decipherment: an HMM with phones as states under the banked prior and a 40 x 40
  substitution emission table fitted by EM on the collapsed symbol strings (no gold in the fit;
  fit on even-indexed utterances, decode both halves), then PER of the deciphered strings.
  Readings: deciphered PER close to the Hungarian oracle and low = decipherable, and a label-free
  repair exists (permute the recognizer's output layer by the deciphered map and continue
  training); prior score per token under the deciphered labeling much better than under the
  identity labeling = the objective preferred the deciphered labeling and training was stuck
  (optimization); equal prior scores = the trigram prior cannot separate the two labelings
  (identifiability at this order).

Amendments before launch (2026-09-20, after the implementer's wiring run on ctrl_50 ep10, whose
numbers are not banked and not recorded): (i) the Hungarian one-to-one map has no delete option,
so a symbol that mostly covers silence stays in and scores as insertions, and its PER can exceed
the identity PER; a one-to-one-with-drop variant (each symbol may map to "remove", priced as the
PER prices removal) is reported next to it. (ii) The prior-score comparison of part E is read on
the HARD relabeling (each symbol to its maximum-a-posteriori phone under the fitted emission
table, applied to the raw collapsed output string, re-collapsed), not on the Viterbi phone
sequence, which is prior-optimized by construction. Its trigram score per token and PER are read
against a null of 20 uniformly random permutations of the 40 output symbols (seed 0, mean / sd /
max): "a relabeling the prior prefers exists" reads only if the hard relabeling's prior score
exceeds the null maximum on both halves. EM defaults, disclosed: emission noise 0.1, 3 restarts,
text bigram for EM = the banked prior's unigram times bigram, trigram rescoring. Jobs:
`PrivateCodeAnalysisJob` (A-D, ep1/4/10, dev-other and dev-clean) and `SymbolDeciphermentJob`
(E, ep4 and ep10 dev-other) in `sae/emc/private_code.py`; ep25 is added when its decode lands.
The banked table is audited from a fresh context before it is written into Results.

## Gate

**G4a.5** (per arm, dev-other, read at sub-epoch 50, or at the label-free-selected checkpoint if a
selector exists; never best-PER over the kept set): the budget round's G4a.4 thresholds, greedy PER
< 0.50 AND greedy emitted rate in [5.80, 14.49]/s, health clause derangement gap > 0. Read against
ctrl_50 at the same sub-epoch through the paired job.

Pre-registered early read at sub-epoch 10 (not a gate), as amended after the code review: the
entropy clause is descriptive only (ctrl_50 itself is at 0.30 nats by sub-epoch 10, so "below 1.0
while ctrl_50 is not" can never fire; the eval-mode entropy of every arm is still banked at each
kept checkpoint). Band exit is declared when the arm's dev-other PER is below its own
length-and-unigram-matched chance null by more than 0.05 (amendment 7; the null sits at 0.84-0.87
for a length-faithful string, `SAE_4A.md:984`, so a PER of 0.80 alone is a screening number, not
band exit). The sub-epoch 4 checkpoint is read the same way for the ent arms, where the penalty
acts before the anneal has sharpened ctrl_50 (2.71 nats at sub-epoch 4).

Low-inventory FAIL (amendment 5; the S3b-C consistency arms collapsed onto 3-4 symbols,
`SAE_4A.md:984`): greedy symbol-usage entropy below 3 bits with the rate inside the window reads
FAIL (inventory collapse) at any kept checkpoint from sub-epoch 10 on. Column pinned 2026-09-20
before any arm's read: the entropy is over the recognizer's 40 output symbols (the `SymbolUsageNullJob`
"bits(40)" field, `blankfree_infomax_read_jobs.py`); the 39-phone column is reported alongside for
comparison with the banked gold numbers only. The two straddle 3.0 on ctrl_50 at sub-epoch 1
(2.82 vs 3.03), which the clause does not read; at sub-epoch 10 ctrl_50 sits at 5.0 bits on both.

Abort rule per arm (the budget round's, plus one): NaN or |trained surrogate| > 100 in any sub-epoch;
expected phone rate outside [4, 20]/s at the end of any sub-epoch after the fifth; train-mode entropy
per output frame below 0.05 nats together with a rate outside that window (collapse to a constant
output). An aborted arm reads FAIL (collapse) and is not restarted with another constant.

Pre-registered predictions: the held arms' eval-mode entropy drops below ctrl_50's from the first
kept checkpoint and below 1 nat by sub-epoch 10; ent_50's tracks the held arm through sub-epoch 20
and then either stays (take-off) or drifts back toward ctrl_50's; the invariance arms have lower PER
than their non-invariance twins if the confident partition without invariance tracks a nuisance;
the low-inventory failure, if it appears, appears in the held arms first. A PASS is audited from a
fresh context and needs a second seed before it is claimed.

Design review 2026-09-20 (`reports/design_review_infomax_2026-09-20.md`, APPROVE_WITH_AMENDMENTS;
amendments 1-8 applied above and in the implementer's brief).

## Results

### Private-code analysis of ctrl_50 (banked 2026-09-20; audit `reports/audit_private_code_2026-09-20.md`, DONE_WITH_CONCERNS)

Jobs: State pointers. dev-other, 2864 utterances; dev-clean tells the same story (audit item 1).
Audit corrections applied to the numbers below: the part A many-to-one row omitted run collapse
(banked 0.874 at ep10; recomputed 0.855, code fix pending re-bank); the with-drop oracle is an
upper bound under the fixed alignment and has no null; the E5 permutation null is bijections
against a many-to-one statistic and the identity labeling itself clears it, so the E5 rule as
pre-registered decides nothing (fix pending: matched many-to-one null plus the identity row).

| ctrl_50 dev-other | ep1 | ep4 | ep10 |
|---|---|---|---|
| PER as scored | 0.855 | 0.869 | 0.897 |
| PER, best many-to-one relabeling (label-using; audit-corrected at ep10) | 0.820 | 0.827 | 0.855 |
| PER, one-to-one-with-drop (label-using upper bound) | 0.822 | 0.862 | 0.841 |
| PER, label-free decipherment (part E, held half) | - | 0.842 | 0.855 |
| NMI(symbol, phone), aligned tokens | 0.115 | 0.063 | 0.056 |
| NMI(symbol, phone), frames | 0.087 | 0.057 | 0.256 |
| frame error, best many-to-one (identity 0.92 throughout) | 0.898 | 0.869 | 0.699 |
| NMI(symbol, unit) (gold phone vs unit: 0.461) | 0.148 | 0.092 | 0.377 |
| NMI(symbol, speaker) (gold: 0.0035) | 0.018 | 0.041 | 0.0065 |
| trigram log p per token, identity labeling, SIL kept (gold, no SIL: -3.20; identity SIL-free at ep10: -4.62) | - | -7.43 | -3.98 |
| symbol-usage entropy, tokens, bits (of 40) | 2.82 | 4.56 | 4.99 |

Readings, against the pre-registered ones:
- Not a relabeling of phones at the token level: every label-using relabeling stays in the band
  (0.82-0.86) and token NMI falls during training. Deciphering "can be done" only up to that
  ceiling: the label-free cipher matches the label-using oracles (0.855 vs 0.841-0.855 at ep10),
  so no information is lost by the absence of labels, and there is nothing phone-like to recover.
  The output-layer permutation repair is therefore not a lever.
- Not a speaker code (NMI with speaker at the gold level by ep10). The invariance arms' premise
  (a nuisance partition that specaug / statistics swap would break) is not supported by this
  table; they remain the empirical test.
- Supported (audit item 5d): a frame-level acoustic code whose token sequence is not phone-like.
  Frame-level phone information rises to 1.3 of 5 bits (frame error 0.70 after the best
  many-to-one map, floor about 0.93) while token-level information falls; unit tracking rises with
  it but stays below the gold phones' (0.377 vs 0.461), so "coarse clustering of the reverse
  units" as written fails.
- Withdrawn (audit items 3 and 6): the reading "the labeling is already near the prior-best, so
  the prior is satisfied by a non-phone sequence (identifiability, not stuck optimization)". What
  the table shows is only that the identity labeling's prior score rose from -7.43 to -3.98
  between ep4 and ep10 (gold -3.20, like-for-like SIL-free identity -4.62); whether a labeling the
  prior prefers exists is undecided until the matched null is banked.
