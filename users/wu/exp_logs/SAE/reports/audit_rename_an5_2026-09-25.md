# Audit: AN-5 name tracking in stage 2 (SAE_4A_rename.md), 2026-09-25

Verdict: CONFIRMED_WITH_CORRECTIONS. **EM LOCKED** holds under the registered text. Every one of the 20 identity
cells recomputes exactly with independent code. The verdict survives every admissible change of pi, of the key
construction and of the sign convention that I could compute. The largest DELTA found on any key arm under any
variant is +0.017, against the 0.05 lock bar and the 0.2 rename bar. The corrections below concern the wording
of the write-up, not the verdict. One robustness question, m without the duration weighting, is not measured.

Audited from a fresh context. Inputs: `An5ReadJob.2q262c8lhmdT` (`output/report.txt`, `an5.json`), the 20
`PhiContentJob` `m_phi.npz` files, the 20 `ReturnnForwardJobV2` `unit_symbol.npz` files, the gold key
`GoldUnitKeyJob.sLnMRRd2qO0t/output/key.json`, `KeyArmsReadJob.STcxhF0w4kpq/output/keyinit_arms.json`,
`KeyInitControlReadJob.JH7zzrX3Egfm/output/keyinit_control.json`, the code `rename_an5_jobs.py` (commit 40cb0854,
clean in the checkout), and the review and launch reports. Nothing was edited or rerun. My scripts are in this
session's scratchpad (`recompute.py`, `ident.py`, `pivar.py`); they import no project code.

## 1. Key identity recomputed (own code)

- **Weights.** I counted units over the 28,254 `train.segments` utterances directly from
  `BlankfreeVadHdfJob.SAjz8y1cT06g/output/units.train.shard{0-3}.hdf` with h5py. The total is 15,275,716, which is
  A20's `weights_frames`.
- **pi.** The posterior mass of `unit_symbol.npz`, summed over the 260 tags of
  `CvDisjointSegmentsJob.PvgJ79Qc1Nro/output/disjoint.segments`, with column sums normalised. All 260 tags are
  present in every forward, which holds 285 tags. The negative entries are at most -8.6e-15 and are clipped.
- **KEY and IDENTITY.** KEY = argmax_s (log m[s,u] + log pi[s]). IDENTITY = the w-weighted share of units whose
  KEY equals the gold key.

All 20 cells equal the printed values (difference 0.0e+00):

| arm | ep0 | ep4 | ep12 | ep48 | DELTA |
|---|---|---|---|---|---|
| rank1 | 0.115057 | 0.071254 | 0.075366 | 0.074689 | -0.0404 |
| rank2 | 0.143350 | 0.165338 | 0.142417 | 0.138650 | -0.0047 |
| rank3 | 0.123739 | 0.127007 | 0.122236 | 0.122911 | -0.0008 |
| rank4 | 0.065251 | 0.057434 | 0.058185 | 0.052136 | -0.0131 |
| gold_key (not counted) | 0.999975 | 0.827717 | 0.777510 | 0.766406 | -0.2336 |

The posterior-key VARIANT (argmax_s N(u,s)) also reproduces all 20 printed values to 4 decimals.

## 2. Verdict under the registered text

- **EM RENAMES** (DELTA >= 0.2 on at least 2 of 4 arms): 0 of 4 arms qualify. It does not fire.
- **EM LOCKED** (DELTA at most 0.05 on at least 3 of 4 arms): 4 of 4 arms qualify.
- **Sign convention.** The literal text "at most 0.05" is the signed reading, as coded. Under the alternative
  reading |DELTA| <= 0.05, the largest |DELTA| is 0.040 (rank1), so the verdict is the same. Across all variants
  the largest |DELTA| is 0.0499 (rank1, uniform pi), which still passes. Even if that arm failed, 3 of 4 arms
  would remain.
- **"Same arm" licence.** The licence needs EM RENAMES with S48 < 3.289 on the same arm. No arm has
  DELTA >= 0.2, so the licence is NOT LICENSED.
  - S48 from `keyinit_arms.json` `comparison.s_arms`, paired over 260 tags: rank1 3.27275, rank2 3.29550,
    rank3 3.28529, rank4 3.33540. The bar is 3.28903.
  - rank1 and rank3 are both below the bar, but their DELTAs are -0.040 and -0.001.
  - The printout names only the best arm (rank1). That changes nothing.

## 3. Checkpoints (no substitution; m and pi from one checkpoint)

For each of the 20 cells I read the `info` INPUT lines of the PhiContentJob and of the forward. In every cell
both name the same file:

- epoch 0: the init `phi.pt`. rank1-4 use `PhiFromKeyInitJob` 86Dq, AiES, G2XQ and 9G5t; the gold-key arm uses
  f0jaGuiJVe6A.
- epochs 4, 12 and 48: `PackedBlankfreeTrainJob.G0Vzzokj5PQC/output/rank{r}/models/epoch.{004,012,048}.pt` for the
  key arms, and `ge1MKcAPmZIV/output/gold_key/models/...` for the gold-key arm.

Further checks:

- Each rank's `returnn.config` in the pack preloads its own init: rank1 86Dq, rank2 AiES, rank3 G2XQ, rank4 9G5t.
  Those inits are built from `KeySearchSelectJob.g9wsznNnqmyO/selected_{1-4}.json`.
- `used_epochs` is the identity map for every arm (from `job.save`). The printout reports "substitutions: none".
- Epochs 1-48 are all on disk.
- The forwards run lambda = 1 and tau = 1 (`anatomy()` asserts prior_weight 1.0 and calls with temperature 1.0)
  on the `cv_holdout` set of 285 utterances.

## 4. Robustness (DELTA per arm: rank1 / rank2 / rank3 / rank4)

| variant | DELTA | arms <= 0.05 |
|---|---|---|
| registered (pi from the 260-tag forward) | -0.040 / -0.005 / -0.001 / -0.013 | 4/4 |
| pi from all 285 forward tags | -0.040 / -0.005 / -0.001 / -0.013 | 4/4 |
| pi uniform (maximum-likelihood key) | -0.050 / -0.043 / -0.005 / -0.015 | 4/4 |
| pi = the bed LM unigram (R4's pi) | -0.027 / +0.007 / +0.017 / -0.004 | 4/4 |
| pi = gold frame share (label-derived, sensitivity only) | -0.028 / -0.007 / +0.017 / -0.003 | 4/4 |
| pi frozen at ep0 | -0.032 / -0.005 / +0.008 / -0.005 | 4/4 |
| pi frozen at ep48 | -0.039 / -0.005 / -0.001 / -0.013 | 4/4 |
| VARIANT posterior key (260 tags) | -0.038 / -0.009 / -0.006 / -0.007 | 4/4 |
| VARIANT posterior key (285 tags) | -0.038 / -0.011 / -0.009 / -0.007 | 4/4 |
| type identity (unweighted, printed) | -0.034 / +0.016 / -0.008 / -0.024 | 4/4 |
| R4 direct (dev-other MFA frames, printed) | -0.014 / -0.003 / +0.006 / -0.004 | 4/4 |

- **Size.** No variant moves any key arm's DELTA by more than 0.04 from the registered value. The maximum DELTA
  is +0.017. Reaching PARTIAL would need DELTA > 0.05 on 2 arms, and EM RENAMES would need +0.2 on 2 arms.
- **m without the duration weighting: NOT MEASURED.** It needs the phi's `emission_log_probs` on each checkpoint,
  which is a model forward. I did not run one. Bounding evidence:
  - At epoch 0, m is the same by construction, because the init phis have no cells.
  - At epoch 48, the two computable extremes agree to within 0.008 on each key arm: the duration-weighted model
    m, and the empirical m = N(u,s)/N(s), which weights cells by the forward's own segmentation.
  - A third weighting would have to add 0.05-0.09 at epoch 48 on 2 arms to reach PARTIAL, and 0.19-0.26 to reach
    EM RENAMES.
  - To settle it: one PhiContentJob variant with uniform (c, j) weights on the 8 key-arm checkpoints at 0 and 48.

## 5. Separation of partition from names (supports "locked", and bears on TP0)

- **Best 1:1 renaming of each cell's own key partition** (Hungarian on the w-weighted 40 x 40 table, own code):
  - ep0: 0.465 / 0.473 / 0.394 / 0.428 (rank1-4)
  - ep48: 0.492 / 0.465 / 0.427 / 0.426
  - many-to-one at ep48: 0.51-0.58
  - The partition keeps room for a pure-renaming gain of 0.30-0.42, yet identity stays at 0.05-0.14.
- **The keys are not frozen.** The KEY changes between ep0 and ep48 on 0.49 / 0.51 / 0.36 / 0.28 of train frames.
  The share that moves onto gold (0.018 / 0.041 / 0.025 / 0.008) is at most the share that moves off gold
  (0.058 / 0.046 / 0.025 / 0.021).
- **Gold-key arm.** Identity equals the 1:1 oracle at every epoch (0.828 / 0.778 / 0.766). The fall of 0.234 is
  therefore all partition change, with the names still gold. EM keeps whatever names it starts with, from either
  init.
- **Consequence.** AN-5 does separate partition from names, so TP0's "cannot separate partition from names" clause
  is not triggered.

## 6. Decision-table rows (all reads to date, precedence amendment applied)

| Row | Condition | Fires? |
|---|---|---|
| 1 | needs AN-3, which was dropped after AN-0 DEAD | no |
| 2 | S PREFERS FOUND NAMES with H2 REFUTED | **yes** |
| 3 | needs P1a and P1c; P1a is False | no |
| 4 | needs GOLD FIRST; AN-1 read NOT GOLD FIRST | no |
| 5 | needs AN-3 | no |
| 6 | needs EM RENAMES with KEY BASIN; AN-5 reads EM LOCKED | no, whatever the separate KEY BASIN audit finds |
| 7 | row 2 fires and H2 is REFUTED | no |

- **Precedence amendment.** On EM LOCKED, row 2's proposal stands. That selects outcome (b): TP-B (relocation
  moves under S with refit) and the coarse-to-fine unit inventory.
- TP-A2 is not brought, because P1a is False.
- The rename levers (TP-A1, TP0's lambda arms) stay dropped.
- TP0 is not listed, because the read is not PARTIAL.
- Row 2's read is then reported as a property of the pre-EM key partitions (audit correction to AN-2).

## 7. Operating point and licence

**Measured.** The stage-2 recipe of A16 (b): the A10 recipe verbatim except the key init, durinit, tau 4 then 1,
48 sub-epochs, trigram prior at lambda = 1, one seed per key. It starts from the four stage-1 selected keys,
whose identity is 0.065-0.143. m comes from the D4 dev-other etas. pi comes from the CV-disjoint 260 at lambda 1,
tau 1. Weights come from the train side. All of these constants trace to the lexlat_v2 registration (lines
272-281, 789) and to the phase file lines 89-95.

**Licensed.** Under this recipe and from these four keys, phi EM does not move key names toward gold over 48
sub-epochs. DELTA is -0.040 to -0.001, even though the partitions admit 1:1 renamings at 0.43-0.49 and the keys
change on 28-51 % of frames. The premise holds for found partitions under stage 2.

**Not licensed:**
- other inits, in particular keys at or above K70's 0.30;
- other recipes: TP-A1's LM-led schedule, a sparse channel prior, relocation moves;
- more seeds (one seed per key) or longer training;
- "EM cannot rename" in general;
- that TP-B or the inventory will work. The failed row 6 is a decision, not a measurement of the alternatives.

**KEY BASIN.** rank1 reaches S below the bar while its identity falls (0.115 to 0.075) and its PER stays at chance
(0.858 / 0.861). KEY BASIN on S therefore carries no name information here. This agrees with the stage-2 audit's
"S only".

## Corrections

1. `SAE_4A_lexlat_v2.md`, stage-2 read: "EM LOCKED, key identity 0.05-0.14 throughout" is not quite right. The
   range over 0/4/12/48 is 0.052-0.165 (rank2 at ep4 is 0.165). At ep0 and ep48 the range is 0.052-0.143.
2. Write-up wording: EM LOCKED means identity does not rise. It does not mean the keys are frozen: they change on
   28-51 % of train frames, with net movement toward gold of zero or less.
3. Write-up wording: the gold-key arm's -0.234 is partition change (identity equals the 1:1 oracle). It is not
   renaming away from gold, and under the signed rule it would read "LOCKED". It is not counted, as registered.
4. Gap: m without the duration weighting is not measured (section 4 states what would settle it).
