# S3b-R mode-pricing companion -- implementer report (2026-09-15)

Status: **DONE_WITH_CONCERNS**.  Everything dispatched is written, tested and committed: the entropy
term, the one-sided hinge, the two held config arms, the tests and the hash census.  Nothing was
launched, no GPU work was run, `settings.py` untouched.  Two things the orchestrator has to know:
the `lam_ent` UNIT choice I had to make for 0.1 / 0.3 to satisfy the brief's calibration
(section 3), and a DEFECT in the census script the rate-term report names, which made its
"before" side enumerate the after tree (section 5).

Commit `e74d561` on `haotian_modality_matching_jupiter` in `recipe/2025-10-speech-llm/`
(explicit paths, no `git add -A`, not pushed).  Base was `1f05ca9` as the brief said; the
content-term implementer committed `6ec6b8a` and the packing implementer `69efa16` while I worked; the
former touches only
`content_term.py` / `mfcc_codes.py` / `recognizer.py` / their tests, i.e. no file of mine, and I
did not touch any file of theirs (`consistency.py`, `recognizer.py`, `content_term.py`,
`mfcc_codes.py` are all unmodified by this commit).

---

## 1. (a) `sae/emc/entropy_term.py` -- the term

    L_ent = mean over NON-PADDED frames of H(q_t) / ln C,     H(q_t) = -sum_c q_t[c] log q_t[c]

* **The tensor is `log_q`**, the variable assigned at `train_steps/sae_emc.py` from
  `model.recognizer(feats, feat_lens)` and handed verbatim to `lattice_loss(log_q, seg_table,
  **dp_kwargs)`.  It is the UNTEMPERED posterior: the DP applies `1/tau` to its own arc weights
  inside `lattice.py` (`scale = 1.0 / float(temperature)`, `scaled_seg_pad`), never to the tensor
  the caller passes.  A test asserts the reported entropy equals the entropy of
  `model.recognizer`'s own output and DIFFERS from the tau-tempered copy's (3.5196 vs 3.6234 nats
  in the fixture) -- so it cannot silently drift onto the tempered tensor.
* **The sub-epoch source is `epoch = int(rf.get_run_ctx().epoch)`** in `train_step`, the SAME local
  that feeds `model.temperature(epoch)` (the tau anneal) and `model.anchor_weight(epoch)`.  No
  second counter, no state: `lam_ent_effective(lam_ent, epoch, ramp_epochs)` is a pure function, so
  a resumed run gets the weight its sub-epoch had.
* **The ramp** is `lam_ent * min(1, max(0, (epoch - 1) / ramp_epochs))`: 0, 0.25, 0.5, 0.75 at
  sub-epochs 1-4 and `lam_ent` from 5 on, at the default `ent_ramp_epochs = 4`.  The endpoint is
  first ATTAINED at sub-epoch 5, which is the shape this loop's other per-sub-epoch schedule already
  has: `emc_train_jobs.ARM_B_ANCHOR_SCHEDULE = (1.0, 0.75, 0.5, 0.25, 0.0, 0.0)` for "alpha 1 -> 0
  over sub-epochs 1-4" (a test asserts `1 - ramp` reproduces that tuple).  This is the reading of
  "ramped 0 -> lam_ent over sub-epochs 1-4 and held at lam_ent from sub-epoch 5" I took; the
  alternative (full weight already at sub-epoch 4) would make "held from 5" say nothing and would
  make three of the brief's five test points identical.
* **Cost**: one elementwise reduction over the `[B, T, C]` tensor the step already has.  No lattice,
  no DP, no second recognizer pass.
* **Monitors**: `emc/entropy_per_frame` (NATS, so it reads against `ln 41 = 3.714` uniform and 0
  one-hot) and `emc/lam_ent_effective` (the ramped weight).  Both marked ONLY when `lam_ent != 0`.
* The ramped weight is the loss's `scale`, so at sub-epoch 1 RETURNN reports the term and leaves it
  out of the optimized total (`run_ctx.total_loss` skips `scale == 0`) -- the column set is the same
  at every sub-epoch of the run.
* Padding is excluded by the `feat_lens` mask; `Z = 0` utterances are NOT excluded (this term has no
  lattice, so a frame's entropy is defined whatever the segmentation can do with it) -- documented.

Container keys: `lam_ent` (default 0.0) and `ent_ramp_epochs` (default 4, written only together with
`lam_ent`).  At `lam_ent = 0` neither key reaches `model_args`, the term is skipped entirely and the
step is bit-identical.

## 2. (b) the one-sided hinge in `rate_term.py`

`rate_loss(..., hinge: bool = DEFAULT_HINGE)`, `DEFAULT_HINGE = False`.  With it on,

    value = (max(0, rho - r) / rho)^2,   coef = d value / d E[N] * 1/T = -2 max(0, rho - r)/(rho^2 T)

replaces the two-sided pair, in the same normalized units.  Above rho both are exactly 0; below rho
they are the two-sided form to the last bit (`-2 (rho - r) == 2 (r - rho)` there) -- both asserted.

**Both FD paths carry it, by construction**: `g = (post_q(+eps) - post_q(-eps)) / (2 eps)` is
`d post_q / d b`, a property of the lattice and not of the loss shape, so `_fd_passes` /
`_stacked_fd_call` are UNCHANGED and the hinge reaches the gradient through the single coefficient
`coef` that multiplies `g` on both paths.  The test therefore checks the hinged gradient against the
autograd oracle with `batched=True` (1 DP call, stacked) AND `batched=False` (2 sequential calls),
in float64 and float32: cosine 1.000000, relative norm error 0.09 % in all four.  The tilted passes
still run when the hinge is inactive, so the term costs exactly what the two-sided one costs
(no data-dependent DP-call count).

The deviation is documented in the module docstring ("ONE-SIDED HINGE ... A DISCLOSED DEVIATION"),
in the container's comment, in `emc_train_jobs.EMC_RATE_HINGE` and in the config: a hinged arm has
NOTHING holding the rate from above and must be read that way.  `RateStats.hinge` records which form
ran; the container refuses `rate_hinge = True` at `lam_rate = 0` (a silent no-op).

## 3. (c) the config, and the units of `lam_ent` (a decision, stated)

`config_sae_4a_s3b_rate_v1.py`: `LAM_RATE_ARMS` gains `(3.0, "lam3_ent_a")` and
`(3.0, "lam3_ent_b")`; the per-arm extras live in a new `COMPANION_ARMS` dict
(`{"lam3_ent_a": dict(rate_hinge=True, lam_ent=0.1), "lam3_ent_b": dict(..., lam_ent=0.3)}`) which
the build loop splats into `build_emc_train_config` (`**COMPANION_ARMS.get(tag, {})`, an empty dict
for every other arm).  `ACTIVE_ARMS = ("lam3",)` is unchanged -- adding a tag there is the ONLY
change either arm needs.  `LAM_RATE_ARMS` entries stay 2-tuples ON PURPOSE: three other files
unpack them (`scripts/sae_4a_cons_census.py:69`, `config_sae_4a_s3b_cons_v1.py:116`,
`test_pack_jobs.py:193`), two of them another implementer's.

**The units of `lam_ent`.**  The brief: "0.1 x max-entropy (ln of the number of recognizer output
classes incl. blank) is about 10 % of the current dev_loss_emc of 1.23".  C = 41
(`recognizer.N_OUT` = blank + 39 ARPAbet + SIL), `ln 41 = 3.7136`, and 10 % of 1.23 is 0.123.

* in NATS, `lam_ent = 0.1` buys at most `0.1 x 3.7136 = 0.371` = **30.2 %** of 1.23 -- a factor
  **3.02** over the calibration, i.e. just past the brief's factor-3 bound (and 0.3 would be 90 %);
* so I took the first clause of the brief ("pick lam_ent units so that ...") and made the term's
  units the NORMALIZED entropy `H / ln C` in [0, 1].  Then `lam_ent` IS the term's contribution at a
  maximally diffuse posterior: **0.1 -> 0.100 = 8.1 %** of 1.23 (the ~10 % asked for, 0.81x) and
  **0.3 -> 0.300 = 24 %** (2.44x the anchor -- the deliberately stronger arm, inside the factor 3).

**The dispatched values 0.1 / 0.3 are therefore kept**, and the normalization is the thing that was
chosen.  The monitor `emc/entropy_per_frame` is still reported in nats (it is the interpretable
number); the loss is that over `ln 41`.  If the orchestrator wants lam_ent in nats instead, the arms
have to become 0.033 / 0.1 and the two job hashes below change.

## 4. (d) tests -- ALL PASSED

`sae/emc/test_entropy_term.py` (NEW, 7 tests), conda `speech_llm`, CPU, tiny shapes:

| test | result |
|---|---|
| the ramp at sub-epochs 1, 2, 4, 5, 8 | 0, 0.075, 0.225, 0.300, 0.300 at lam_ent = 0.3; mirrors `ARM_B_ANCHOR_SCHEDULE`; `ramp_epochs = 0` refused |
| padded frames excluded | 11 = 2T - 3 frames counted, value bit-identical when the padding is replaced by uniform AND by one-hot |
| the value is the normalized entropy | uniform -> 1.000000, one-hot -> 2.4e-20, one gradient step lowers it 0.795454 -> 0.795193 |
| the tensor is the UNTEMPERED posterior | reported 3.519641 nats == the recognizer's own, != the tau = 1.5 copy (3.623372) |
| `lam_ent = 0` leaves the step bit-identical | 15 losses/monitors identical bit for bit, no `ent` key, no monitors, `entropy_penalty` never called |
| `lam_ent = 0.3` prices at the RAMPED weight | scale 0.000 / 0.075 / 0.300 at sub-epochs 1 / 2 / 5, both monitors reported, total backward runs |
| container + builder | three keys named and refused when impossible; omitted at their defaults (`sis_hash_helper` equal), written when on (hash moves), `rate_hinge` without `lam_rate` refused |

`sae/emc/test_rate_term.py` (13 tests now, 21 PASS lines; +2 tests):

| new test | result |
|---|---|
| hinge zero above rho / two-sided below | above rho (r 0.2360-0.2785 vs rho 0.1180): loss 0 AND gradient 0 while the two-sided form is 0.1893-strong; below rho: max abs delta of loss and of gradient = 0 |
| hinged FD gradient vs autograd oracle, both FD paths | float64 and float32 (production dtype) x stacked (1 DP call) and sequential (2): cosine 1.000000, rel 0.09 % each; phi grad None |

Regression suites re-run because the train step / container changed: `test_sae_emc.py` (6 checks),
`test_consistency.py` (all), `test_emc_train_jobs.py` (all), `test_pack_jobs.py` (all 4, incl.
`config equivalence (lam3)`, which re-derives `ReturnnTrainingJob.DF6blPpto23t`).  `ruff check` on
the eight touched files: the only findings are the 4 that pre-exist in the shadow tree
(`typing.List`, `c_r`, `sys`, `numpy`); my new files are clean.

## 5. (e) hash census -- and a DEFECT in the census script

Method: a shadow tree = the LIVE `src` copied, my six modified files restored from `git show HEAD:`
and my two new files deleted, so the delta enumerated is mine alone (the content-term implementer's
live changes sit on both sides).

**`scripts/sae_4a_cons_census.py`'s `SAE_CENSUS_SRC` override does not work.**  It inserts the
shadow at `sys.path[0]` and THEN imports `sisyphus.tk`, which loads the setup's `settings.py`, whose
line 191 does `sys.path.insert(0, recipe/2025-10-speech-llm/src)` -- the LIVE tree goes back in
front.  My first run "proved" the before side already contained `lam3_ent_a` / `lam3_ent_b`, i.e. it
had enumerated the after tree.  **The rate-term report's census (98/98, 1021/1021) was run through
that same script and mechanism, so its before side is not evidence of anything.**  I did not edit
the script (it is not my file); I ran a corrected runner
(`<scratch>/census_runner.py`) that imports `tk` first, then drops the live `src` from `sys.path`,
puts the shadow in front and PRINTS which tree was used.  Fix, if the script should be repaired:
move the `SAE_CENSUS_SRC` insert after the `from sisyphus import tk` line and drop the live entry.

Result (shadow-before vs live-after, ids sorted, proven tree on each side):

| graph | before | after | ids |
|---|---|---|---|
| `config_sae_4a_s3_v1` | 98 | 98 | identical |
| `config/sae_4a_phase.py` (s2b + s2c + s3 + decode_temp) | 1021 | 1021 | identical |
| `config_sae_4a_s3b_rate_v1` (ACTIVE_ARMS = lam3) | 91 | 91 | identical; the only diff in the file is the two new `# arm ... not built` comment lines |

Arms, unchanged: `lam1 WCh1fMyr88yu`, **`lam3 DF6blPpto23t` (the LAUNCHED job)**, `lam10
axf11lrsap5i` -- all three as the rate-term report banked them.  `test_pack_jobs.test_config_
equivalence` re-derives `DF6blPpto23t` independently, from the packed and the single config.

**The new arms** (built only with `ACTIVE_ARMS` extended, which is NOT committed):

| arm | ReturnnTrainingJob | model_args it adds |
|---|---|---|
| `lam3_ent_a` | `TWh8GPa8ItGi` | `rate_hinge: True, lam_ent: 0.1, ent_ramp_epochs: 4` |
| `lam3_ent_b` | `HdP6u4MSWRpk` | `rate_hinge: True, lam_ent: 0.3, ent_ramp_epochs: 4` |

Both carry lam3's own `lam_rate: 3.0, rate_rho_hz: 9.6619373279, rate_fd_eps: 0.25,
rate_fd_mode: 'central'` and nothing else; enabling all five arms adds 256 jobs and removes none
(91 -> 347).

## 6. What is NOT done / worth naming

* `config/sae_4a_s3b_rate.py` (the setup-dir wrapper, untracked) still says "three arms" in its
  docstring.  It is outside the files I was given; not touched.
* The entropy term is NOT coupled to `lam_rate` in the builder: `lam_ent` alone is a legal
  (if, per the design review, pointless) arm.  Only `rate_hinge` demands `lam_rate != 0`.
* No measurement of any kind was taken here.  The companion's effect on the greedy rate is an
  empirical question for the arm, and a confident output is not a content claim (S3's own
  sub-epochs 5-8 are confident and content-free).
* This report file is written into the i6_experiments checkout but NOT committed: the brief names
  only the speech-llm repository and branch for commits.

* The packing config `config_sae_4a_s3b_pack_v1.py` (commit `69efa16`, not mine) already reads
  `getattr(rate, "COMPANION_ARMS", {}).get(tag, {})` and packs only `lam1` / `lam10`, so the two
  new `LAM_RATE_ARMS` entries leave its graph and hashes untouched.
