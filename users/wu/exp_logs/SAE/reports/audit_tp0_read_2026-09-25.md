# Audit: TP0 read (`Tp0ReadJob.t7gzZhB2fWtn`), 2026-09-25

**CONFIRMED_WITH_CORRECTIONS.** The rule outcome EM LOCKED re-derives exactly from the artifacts with independent
code, under the registered rule, with AN-5's identity. The two full-derangement arms carry it. Corrections below
scope what it licenses.

Inputs: registration `SAE_4A_rename.md` TP0 bullet (unchanged since commit c9641e31d, 10:30; the pack started 11:02);
reader `work/speech_llm/sae/emc/rename_tp0_jobs/Tp0ReadJob.t7gzZhB2fWtn/output/{tp0.json,report.txt}`; pack
`PackedBlankfreeTrainJob.21tww6QQK0tH`; executor `reports/exec_tp0_check_2026-09-25.md`; extraction
`reports/extract_tp0_read_2026-09-25.md`. Scratch script (not banked):
`/tmp/claude-34349/.../scratchpad/rederive.py`.

## 1. Re-derived numbers and the verdict

This audit used its own code and none of the reader's helpers for the quantities:
- pi: accumulated from each cell's `unit_symbol.npz` over the 260 disjoint tags.
- key: argmax_s log m + log pi over `m_phi.npz`.
- weights: recounted from the four `units.train.shard*.hdf` over `train.segments` (28,254 tags; total 15,275,716 = A20's).
- best 1:1: scipy `linear_sum_assignment` on the 40 x 40 weighted table.
- S: the mean `nll_tau1_per_frame` over the common tag set.
- PER: re-read from `report.json`.

Across all 16 cells the keys equal the reader's, the identity and best-1:1 differences are 0, and S differs by at most
3.6e-15.

| arm | id0 | id4 | id8 | id12 | DELTA | 1:1 at 0/4/8/12 |
|---|---|---|---|---|---|---|
| full_s1 | 0.0777 | 0.1059 | 0.0996 | 0.1016 | +0.0239 | 0.9998 / 0.5973 / 0.5650 / 0.5541 |
| full_s2 | 0.0778 | 0.0928 | 0.0917 | 0.0903 | +0.0125 | 0.9999 / 0.5546 / 0.5364 / 0.5177 |
| 5pair_s1 | 0.5931 | 0.6266 | 0.5981 | 0.5880 | -0.0051 | 1.0000 / 0.7014 / 0.6786 / 0.6501 |
| control (beside) | 1.0000 | 0.8169 | 0.7929 | 0.7798 | -0.2202 | 1.0000 / 0.8169 / 0.7929 / 0.7798 |

Rule: DELTA >= 0.2 on 0 of 3 arms, and DELTA <= 0.05 on 3 of 3, so the outcome is **EM LOCKED**. The reader's
thresholds (0.2 and 0.05, each needing 2 arms, signed) and its rule text match the registration verbatim.

## 2. The identity is AN-5's rule
- The TP0 reader calls `An5ReadJob.cell` and `An5ReadJob.weights` directly.
- `rename_an5_jobs.py`, `key_search_jobs.py` and `rename_emstep_jobs.py` did not change between the AN-5 commit
  40cb0854 and the TP0 commit 64e65cb2. TP0 added new files only.
- The independent re-implementation above reproduces every key.

## 3. Inits, derangements and recipe
- The four arm configs in the pack differ only in `reverse_checkpoint_path` and the model output path. The four init
  paths are:
  - full_s1: `PhiDerangeJob.QRIjZnmG0boo`
  - full_s2: `PhiDerangeJob.DLoD5WXxaOx6`
  - 5pair_s1: `PhiDerangeJob.PhNxF9aMQxAJ`
  - control: `PhiFromKeyInitJob.f0jaGuiJVe6A`
- Each arm loaded its own init:
  - The step-0 losses differ by arm. Prior per token is -3.93 / -3.83 / -3.49 / -2.84.
  - At epoch.001, each arm's `emb_type` is nearest to its own init: distance 2.5-2.7, against 5.2 or more to the others.
- Derangements:
  - Every `derange.json` g equals this audit's re-implementation of AN-0's `derangement()`. The pool is the 37
    non-SIL symbols that hold units; OY and ZH are outside it.
  - The fixed points are OY, ZH and SIL.
  - The 5-pair seed-1 draw is AH-T, AO-N, K-OW, M-Y, P-S, the same as AN-0's `rename_emstep.json`.
- Deranged checkpoints:
  - Against the gold-key phi, only `emb_type.weight` differs, with its rows permuted by g. Every other tensor is
    equal. `dur_logits` is unchanged because all non-SIL rows are identical.
  - At epoch 0, each deranged arm's m_phi equals the control's rows permuted by g exactly.
- Recipe:
  - Each arm's config equals A17 (ii)'s `t1_gold_s01` (`PhiFirstProbeTrainingJob.nVpD2O3xpfcJ`) and `t1_r70_s01`
    (`YtsRkvAl7Kl8`) configs, except the init path, the output path, and the logging-only `torch_log_memory_usage`.
  - Settings: tau = 1 for all 12 sub-epochs, prior_weight 1, phi lr 1e-4 x 30, random_seed 1.
  - All arms finished 12 sub-epochs (684 steps). The logs contain no NaN.

## 4. Epoch 0 is the inits, and every cell uses the same sets
- All 64 per-cell jobs were traced to the expected checkpoint, with 0 mismatches: the content (`content.json`
  phi_checkpoint), the anatomy forward, the CV marginal forward and the dev-other decode forward. Epoch 0 is the init;
  epochs 4/8/12 are the pack's `epoch.004/008/012.pt`.
- The reused jobs point to the right checkpoints with the same settings:
  - control epoch 0: `an5/gold_key_ep0`, `WbIBVwEM8Ev6`, and the keyinit epoch-0 read;
  - the deranged arms' epoch-0 marginals: AN-2's.
- Forward configs: all 16 anatomy, 16 marginal and 16 decode configs are identical apart from the checkpoint and the
  name.
- `PhiContentJob` settings are identical in every cell: dev-other, 500 utterances, gold phi `16v7R6ztSq1u` ep8.
- Evaluation sets:
  - pi uses the 260 disjoint utterances in every cell.
  - S uses one common set of 260 tags, none excluded; the marginal settings are identical, with shuffle_seed None.
  - PER uses the 500 dev-other utterances in all cells.

## 5. What could make the verdict unsafe
- **The measure sees names.** Identity at epoch 0 equals the value g implies (0.0779 / 0.0779 / 0.5932 / 1.0000),
  within the pi-tie share (at most 1.9e-4). The posterior-key variant is also flat.
- **The full arms are sensitive.** Their own epoch-12 partitions admit identity up to 0.554 and 0.518, so a DELTA of
  +0.48 or +0.44 was reachable. They read +0.024 and +0.013, and their A15-F own-label counts fall (3 -> 2, 3 -> 1).
  The verdict needs only these two arms, and EM RENAMES could not have fired without both of them.
- **The 5-pair arm cannot show renaming at this bar.** This is audit arithmetic on the reader's numbers, not a
  registered statistic.
  - A perfect rename followed by the control's drift gives about 0.780 - 0.593 = 0.187, below 0.2.
  - On its own epoch-12 partition, the best naming gives 0.650 - 0.593 = 0.057.
  - Its LOCKED (-0.005) therefore carries no information about renaming.
- **The 5-pair arm's A15-F emission map disagrees with its flat key identity.** This is read off the reader's own
  `content.json` inputs.
  - At epoch 4, the nearest gold type of symbols AH, T, N, M and Y is their own name. The own-label set gains AH, M,
    N, T and Y, and the reader prints OWN 30 -> 32.
  - At epoch 12, the nearest gold type of AH, T, N, M and Y is still their own name. The own-label set is AH, M, N, Y.
  - AO, K, OW, P and S keep their partner's type or move elsewhere, and AO becomes unclaimed.
  - These files cannot tell whether the frame mass of the swapped phones followed the rows. A per-gold-phone breakdown
    of the key, from a registered reader, would settle it.
- **The "right partition" does not hold during the run.** In the deranged arms, best-1:1 falls to 0.60 / 0.55 / 0.70
  by sub-epoch 4 and to 0.55 / 0.52 / 0.65 by 12. The control is at 0.82 and 0.78.
- **The control behaves as a control.** It has the same recipe, seed and batches.
  - Its names stay the best names for its partition at every epoch: identity equals best-1:1 within 3.5e-5, and
    own-label goes 40 -> 38.
  - Its identity drift of -0.220 comes from the partition only.
  - This is consistent with AN-5's gold-key arm (1.000 -> 0.766 at 48, on a different schedule).

## Descriptive facts: control arm and best 1:1 column
- Control:
  - Identity is 1.000 / 0.817 / 0.793 / 0.780 and equals its best-1:1 at every epoch.
  - DUP goes 1 -> 3. R4 emis goes 0.612 -> 0.557.
  - Hungarian PER is 0.385 / 0.330 / 0.360 / 0.379.
  - S is 4.573 -> 3.252, the lowest of the four arms at every epoch. At 12 the others are 5-pair 3.292, full_s2
    3.370 and full_s1 3.387.
- Best 1:1 column:
  - Every deranged arm starts at a partition of about 1.0 and loses most of it by sub-epoch 4 (0.60 / 0.55 / 0.70).
    It then declines slowly to 0.55 / 0.52 / 0.65, always below the control.
  - Named identity stays near its starting value: 0.10 / 0.09 against 0.55 / 0.52 in the full arms, and 0.588 against
    0.650 in the 5-pair arm.

## Corrections for recording
1. Record EM LOCKED as carried by full_s1 and full_s2. The 5-pair arm's LOCKED is uninformative at the 0.2 bar.
2. Scope the consequence wording.
   - Licensed: from the right partition with all non-SIL names deranged, 12 sub-epochs of A17 (ii)'s tau = 1 EM
     (lambda 1, one training seed per arm) do not move key identity toward gold. The partition degrades instead, more
     than in the control.
   - Not licensed: "keeps wrong names on the right partition" for the 5-pair condition. There the emission map shows
     partial row-level movement to the right names (AH, T, N, M, Y) that key identity does not register.
3. Not tested: other seeds, the LM-led schedule or lambda > 1, a flat-channel start, and longer runs.

The registered consequence of EM LOCKED (the naming levers go back to the user beside the proposal) follows from the
rule outcome. Nothing in this audit overturns it.
