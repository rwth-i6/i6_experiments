# Audit: A18 (a) L2-2 bridge read, G4a.L2.4 (dec_joint on em_s13 against cold_ctl), 2026-09-24

**CONFIRMED_WITH_CORRECTIONS.** I re-derived the registered read from the per-utterance forward outputs and it
holds: G4a.L2.4 = LOWER -- OBJECTIVE ONLY. The code is not broken, because greedy PER on dev-other stays in the
chance band for every arm at every kept epoch. The corrections are about the frame and how the result should be
worded. None of them changes a number or the verdict class.

Read-only audit. Nothing was edited, launched or cleared. The scripts used are in the session scratchpad
(`/tmp/claude-34349/.../scratchpad/{phicmp,paired,per,perpair,gtrace,k2,trace,sums}.py`). They read artifacts and
ran on the login node (CPU, all bounded).

## 1. The verdict: delta, interval, B, margin

Source: `per_utterance.json` of the five crosseval forwards under `work/i6_core/returnn/forward/`: dec_joint
`1TOnF9PzjkUS`, dec_distil `ptp4A0Jhx34O`, dec_frz `GLqP0aok8X6R`, dec_joint_s2 `czNp2CXhwumu`, and cold_ctl
`sVxuayfU6CYi` (SLURM 1998211, confirmed in its engine file). Each forward loads its run's `epoch.008.pt` for both
theta and phi. The five forward configs differ only in the checkpoint paths and the cell name, so all cells are read
at one setting: tau 2.0, lam 1 / 0.1 / 3 / 1, and the same HLG.

The per-utterance L is my own recomputation: l_tau + lexlat_k2 + 3 x rate, each term per retained frame. It covers
all 285 CV-holdout utterances, all kept in every cell, from 164 speakers. Retained frames are identical across cells.

| contrast vs cold_ctl | delta L | 95 % CI (speaker bootstrap, seed 0, 2000) | CI (seed 7, 5000) | utterances lower |
|---|---|---|---|---|
| dec_joint | -0.115829 | [-0.125691, -0.103458] | [-0.126281, -0.103974] | 268 / 285 |
| dec_distil | -0.107139 | [-0.116958, -0.095885] | | 259 / 285 |
| dec_frz | -0.044921 | [-0.055303, -0.031795] | | 217 / 285 |
| dec_joint_s2 | -0.106491 | [-0.116282, -0.094954] | | 262 / 285 |

- B = |mean(L_dec_joint - L_dec_joint_s2)| = 0.009338, so the margin is max(B, 0.01) = 0.01.
- The interval lies wholly below zero, and |delta| = 0.116 is about 11.6 x the margin, so the class is **LOWER**.
- CODE BROKEN needs dev-other greedy PER < 0.50 at some kept epoch. The minimum over all arms and epochs is 0.8420,
  so the flag is **OBJECTIVE ONLY**.
- Every value matches `BridgeReadJob.v6SONiAZL7rH/output/bridge_read.json` to 6 decimals.
- The identity check passes: each forward's cell mean equals the run's logged ep8 dev score, with a difference of at
  most 6e-17.

**The phi of dec_joint is em_s13's selected checkpoint.**
- `SelectedCheckpointJob.uZtbsDtueCrI/output/model.pt` is a link to
  `PackedBlankfreeTrainJob.KCj5mptWgBqb/output/em_s13/models/epoch.012.pt`.
- Its `selected.json` reads selected em_s13, verdict SIGNAL, with the 16 restarts as candidates.
- `ExtractSubmoduleCheckpointJob.xYJt2MEmS2No` (prefix `reverse.`, 12 keys, epoch 12) matches em_s13's reverse block
  with a max-abs difference of 0.0.
- dec_joint's `returnn.config` loads that extract as `reverse_checkpoint_path`.
- I loaded the checkpoints and compared the reverse blocks:
  - dec_frz's reverse block equals em_s13 exactly at ep1, 2, 4 and 8.
  - dec_distil's equals it exactly at ep1.
  - dec_joint's phi moves: max-abs difference 0.022, 0.25, 0.55 and 1.08 at ep1, 2, 4 and 8.
  - The launch review showed "dec_frz = the selected phi" from the code only; it is now verified on disk.

**cold_ctl against dec_joint.**
- The cold_ctl run is `PackedBlankfreeTrainJob.UdhhxiGIMBob/output/cold_ctl` (L2-0 R2, run 2026-09-23 22:17 to
  09-24 00:06). `sVxuayfU6CYi` is only its crosseval forward.
- A diff of the two written `returnn.config` files shows exactly one line besides the output model path:
  dec_joint's `reverse_checkpoint_path`. Flat init `0J9d6wjrkRYH`, seed, tau [2.0]*8, lr [1e-5, 1e-4 x7], the k2
  block and the prior/eta/HLG paths are all equal.
- Both ran the same RETURNN build (`00171dfe.dirty`).
- No commit or working-tree change touched the training-path code (`definitions/sae_blankfree.py`,
  `train_steps/sae_blankfree.py`, `lexlat_k2_train.py`, `sae_emc`, `reverse`) between cold_ctl's start and
  dec_joint's start. The last such commit is 6fd3d02e at 09-23 20:31.
- The configs import the live checkout, so this check matters. The exact identity reproduction of cold_ctl's logged
  scores under today's code corroborates it.
- dec_joint ran 8/8 sub-epochs (SLURM 1998206) with no NaN or traceback in any arm log.

## 2. PER and generative PER

**Greedy PER.** I recomputed it with my own Levenshtein from each job's `greedy_phones.json` against
`GoldPhonesJob.ZGSp0hxyd2YP` dev-other: all 2,864 utterances and 177,275 reference phones. I traced each of the 20
BlankfreeGreedyPerJobs through its forward and recognizer extract to the right arm and epoch checkpoint.

| run | ep1 | ep2 | ep4 | ep8 |
|---|---|---|---|---|
| dec_joint | 0.8615 | 0.8536 | 0.8464 | 0.8420 |
| dec_distil | 0.8537 | 0.8526 | 0.8471 | 0.8461 |
| dec_frz | 0.8623 | 0.8585 | 0.8432 | 0.8463 |
| dec_joint_s2 | 0.8629 | 0.8549 | 0.8426 | 0.8460 |
| cold_ctl | 0.8881 | 0.8864 | 0.8554 | 0.8487 |

- All values equal the jobs' `per.json`. Every run is inside the chance band 0.83-0.91 at every kept epoch, and all
  read NO LIFT at ep8.
- Paired contrasts at ep8, with a speaker bootstrap over 33 dev-other speakers:

  | contrast | delta PER | 95 % CI | reading |
  |---|---|---|---|
  | dec_joint - cold_ctl | -0.0067 | [-0.0101, -0.0033] | |
  | dec_joint_s2 - cold_ctl | -0.0027 | [-0.0065, +0.0010] | includes 0 |
  | dec_frz - cold_ctl | -0.0024 | [-0.0059, +0.0010] | includes 0 |
  | dec_joint - dec_frz | -0.0043 | | |
  | dec_joint - dec_joint_s2 | -0.0040 | | the seed spread, same size |

- At ep1-2 all four arms, including the frozen one, are 0.025-0.034 below cold_ctl. The gap closes by ep8 as
  cold_ctl catches up.
- The same cold_ctl PERs appear in A14 (i). There the A10 phis at sub-epoch 48 reached 0.820-0.842 at ep8, with
  deltas of -0.006 to -0.029. em_s13 lifts no more than they do.

**Generative PER.** Direct / Hungarian / NMI(symbol, phone) on the 500-utterance D4 dev-other set. I traced all 20
GenDecodeReportJobs to the right checkpoints.
- **em_s13, from dec_frz, identical at ep1, 2, 4 and 8:** 0.8566 / 0.8587 / 0.0761, band IN.
- Against the A10 phis at sub-epoch 48 (doc line 448; I checked it against
  `PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO` report.txt):
  - Direct 0.832-0.856: em_s13 is 0.0006 above the top.
  - Hungarian 0.838-0.861: em_s13 is inside.
  - NMI 0.079-0.113: em_s13 is 0.003 below the bottom.
  - So em_s13 sits at the worst edge of the A10 band. It is no better than A10, and it is inside the chance band.
- The jointly trained phis stay at 0.855-0.858 direct from ep1 to ep8 (dec_joint ep8: 0.8558 / 0.8526 / 0.0779).
  Joint training does not move phi's content.
- cold_ctl's phi, reported only:
  - ep1 NMI 0.399 is degenerate: 657 substitutions and 27,177 deletions out of 29,690 phones.
  - ep2 reads 0.8262, BELOW the band, and is deletion-heavy (12,982 deletions).
  - Neither is content in the joint arms.

## 3. What carries the objective drop

Paired per-utterance term contrasts against cold_ctl, scaled, with 95 % CIs:

| arm | l_tau | lexlat_k2 | 3 x rate | sum |
|---|---|---|---|---|
| dec_joint | -0.2401 [-0.2495, -0.2281] | +0.0916 [+0.0873, +0.0959] | +0.0326 [+0.0257, +0.0401] | -0.1158 |
| dec_frz | -0.2041 [-0.2138, -0.1916] | +0.1094 [+0.1052, +0.1137] | +0.0497 [+0.0420, +0.0582] | -0.0449 |
| dec_joint - dec_frz | -0.0360 [-0.0402, -0.0318] | -0.0178 [-0.0210, -0.0147] | -0.0171 [-0.0204, -0.0139] | -0.0709 [-0.0766, -0.0651] |

- **The whole drop is l_tau.** The word-lexicon term and the rate term both get worse, in every arm. So LOWER must
  not be read as a better fit to the lexicon.
- lexlat_k2 = (z_h - z_hlg) / retained frames exactly. Its rise comes from z_hlg per frame falling: -0.269 for
  dec_joint against -0.184 for cold_ctl. The recognizer's emissions fit the word graph worse than cold_ctl's.
- l_tau = -log Z_tau over a lattice whose segment table is phi's unit log-likelihood, so a better unit-fitting phi
  lowers it whatever theta learns.
  - The frozen em_s13 alone gets 85 % of dec_joint's l_tau drop (-0.204 of -0.240).
  - The cell diagnostics are descriptive expectations under each cell's own posterior; the reader prints no
    per-utterance values for them:
    - reverse per frame -3.174 against -3.951: phi explains the units 0.78 nats/frame better;
    - phone-trigram log-probability per token -3.521 against -2.514: the posterior strings are 1.0 nat/token less
      likely under the prior.
- **The joint drop is larger than the frozen one:** -0.116 against -0.045, a difference of -0.071 [-0.077, -0.065].
  - About half of that difference is further phi adaptation in l_tau (-0.036).
  - The rest is k2 and rate recovering part of the ground dec_frz loses.
  - None of it shows up as content:
    - ep8 PER, dec_joint - dec_frz: -0.0043, the same size as the seed difference (-0.0040);
    - dec_joint_s2 and dec_frz against cold_ctl: both CIs include 0;
    - the jointly trained phi's generative PER is unchanged (0.8558 against 0.8566);
    - k2 remains worse than cold_ctl.
- agg is a point contrast only, and I did not recompute it; per A6 it is not gated. The reader reports -2.108
  (cold_ctl 3.187 against 1.079). The weighted total including 0.1 x agg is -0.329 and points the same way, so
  whether agg is included does not change the class.

## 4. What could change the verdict (corrections)

1. **The baseline deviates from A9's registered text.** A9 (doc line 62) says: "L2-2's cold_ctl baseline takes the
   wave's duration setting, so its single delta stays phi's emissions."
   - The reused cold_ctl has uniform duration logits. em_s13 carries durinit-initialised, EM-trained durations.
   - A18's build choices re-classed this as part of phi's init, which A6 lets differ. That choice was recorded
     before any result and is disclosed in report.txt.
   - The extraction's wording, "only in phi's init ... per A9/A6", is therefore inaccurate for A9. It should say
     "a pre-result A18 build choice relaxing A9".
   - Whether a durinit cold_ctl would shrink the 0.116 delta, and so possibly change the class to TIE, is not
     measured. This cannot be told from these files; it would need a durinit cold_ctl at L2-2 constants.
   - It cannot turn the result into CODE BROKEN. The PER clause fails by 0.34 on its own.
2. **Record the result with its decomposition.** The drop is l_tau only (-0.240), while lexlat_k2 (+0.092) and
   rate (+0.033) worsen. 85 % of the l_tau drop comes from the frozen phi.
3. **B comes from one seed pair** and tests theta-init noise only, with the same phi in both. It does not cover the
   choice of EM restart. That is as registered and does not matter at an 11x margin.
4. **Minor:**
   - The brief's doc line numbers are off by one: the gate row is at line 46 and A6 at line 59.
   - "cold_ctl sVxuayfU6CYi" in the brief names cold_ctl's crosseval forward, not the run, which is UdhhxiGIMBob.
