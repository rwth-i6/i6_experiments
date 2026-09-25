# A18 builder: the keyinit lift pack, the L2-2 bridge and the stage-1 key wiring (implementer, 2026-09-24)

Status: DONE_WITH_CONCERNS. Everything is built and tested. Nothing was launched, and no manager was started or restarted.

## Commit
`1e05689f` is on `haotian_modality_matching_jupiter` in `recipe/2025-10-speech-llm`. It is not pushed. Paths were staged explicitly.
- The commit leaves other people's uncommitted edits untouched: `config_sae_1g_v1.py`, `key_search.py` and `key_search_jobs.py` (the stage-1 fix, which is in progress elsewhere), and `config_sae_3e1_d6_swap_cont_v1.py`.

## Files
New files:
- `src/speech_llm/prefix_lm/model/train_steps/sae_blankfree_distil.py`: dec_distil's train step. For sub-epochs `1..distil_subepochs` it takes the cross-entropy (CE) of theta's log_q to the stopped-gradient generative posterior; after that it runs the unchanged `sae_blankfree.train_step`.
  - The target is `lattice_loss` on `null_log_q` plus a detached segment table, using the joint step's `dp` with the temperature swapped for `distil_tau`.
- `src/speech_llm/sae/emc/a18_selected_checkpoint_jobs.py`: `SelectedCheckpointJob`.
  - At run time it resolves the arm named by a label-free selection json and links that arm's checkpoint.
  - A `None` value, a missing key or an unknown arm fails the job.
- `src/speech_llm/sae/emc/a18_keyinit_lift_jobs.py`: `DurinitBasinLiftReadJob` (A18 (b); the rule is quoted verbatim in its docstring).
- `src/speech_llm/sae/emc/l22_bridge_read_jobs.py`: `BridgeReadJob`, the G4a.L2.4 reader as amended by A6 and A9.
  - It runs D16's identity check, then a paired speaker-clustered bootstrap.
  - B = |mean(L_dec_joint − L_dec_joint_s2)| and the margin is max(B, 0.01).
  - It prints LOWER, HIGHER or TIE and flags CODE BROKEN or OBJECTIVE ONLY.
  - Beside the verdict it prints PER at ep1/2/4/8 with the A4 class, and the phi generative PER (direct, Hungarian, NMI).
- `configs/config_sae_4a_lexlat_v2_l22_v1.py`: `build_bridge(phi, ...)`, `cold_baseline()` and `selected_phi(...)`.
- `configs/config_sae_4a_lexlat_v2_a18_v1.py`: `build_keyinit_lift(ki)`.
- Tests: `train_steps/test_sae_blankfree_distil.py` and `sae/emc/test_a18_l22_readers.py`.

Edited files:
- `config_sae_4a_lexlat_v2_keyinit_v1.py`: `py()` now also builds `out["a18_lift"]`. This is not behind a flag, as dispatched.
- `config_sae_4a_lexlat_v2_em_v1.py`: adds `BRIDGE_WAVE = False` and `build_wave_bridge()`. The phi is the `selected` entry of `GenMargSelectionJob`, taken at `WAVE_NUM_SUBEPOCHS`.
- `config_sae_4a_lexlat_v2_keyarms_v1.py`:
  - `stage1_keys()` returns `{rank1..rank4: KeySearchSelectJob.out_selected[i]}`.
  - Adds `BRIDGE_KEYARMS = False` and `build_keyarms_bridge()`. The phi is `KeyArmsReadJob` `comparison.best_arm`, the arm with the lowest S at 48.
- `key_search*.py` was not edited, and nothing was written to the setup's `config/` folder. The keyarms shim already existed.

## New jobs
- keyinit (always built): 89 new jobs.
  - Lift pack `PackedBlankfreeTrainJob.ZUZypSQn7qc0` (gpu 4, 11.5 h, K2 exe).
  - `DurinitBasinLiftReadJob.VyOgC0BZvRA9`.
  - `PairedPerDeltaJob` `2wMw96BmGytD` (g_dur vs gold_em), `UnT62NW3k2M9` (r30) and `neMyJu1lkoSl` (r70).
  - 20 extract jobs, 16 greedy PER jobs, 16 generative decode reports and 32 forwards.
- Bridges (flags True; the hashes below were built with TEST-ONLY distil values lr 1e-5 and tau 2.0, and will move once the real values are set):
  - The shared `cold_ctl` is `BoundedBlankfreeTrainingJob.O4lQfx6Qz1K7` (gpu 1). Its hash does not depend on the distil values.
  - wave: `SelectedCheckpointJob.uZtbsDtueCrI`, pack `63nbj6Jgiegj`, reader `MyKUYB4On1Dz`. 111 jobs in total.
  - keyarms (tau4): `SelectedCheckpointJob.PAV7erhRs8wq`, pack `SCtv4DjzFQ50`, reader `isLznrpvjldx`. 110 jobs in total.
  - The two bridges share 22 jobs (the baseline and its reads).

## Checks
- Hash check: job ids were dumped from each setup shim before and after the change.

  | config | before | after | removed | added |
  |---|---|---|---|---|
  | keyinit | 1807 | 1982 | 0 | 175 |
  | em | 56 | 56 | 0 | 0 |
  | a14 | 1570 | 1570 | 0 | 0 |
  | a17 | 1451 | 1451 | 0 | 0 |
  | keysearch_s1 | 39 | 39 | 0 | 0 |
  | em_ext | 795 | 795 | 0 | 0 |

  - Of keyinit's 175 added jobs, 86 are A17's existing jobs, pulled in by the paired rows (see the concerns below); 89 are new.
  - keyarms builds 1326 jobs with the flag False, and all 39 keysearch_s1 jobs are among them.
  - With the flags True, every flag-False job is kept (em 56/56, keyarms 1326/1326).
  - With the flags True and the distil values None, the build raises the N7 ValueError, as intended.
- dec_joint against an A14 (i) arm: the written configs were serialized and diffed. The only difference is `reverse_checkpoint_path`.
  - dec_frz adds only `freeze_reverse=True`.
  - dec_distil differs only in the train_step partial and `learning_rates[0]`. With lr 3e-5 the result was `[3e-5, 1e-4 x7]`; the other arms stayed at `[1e-5, 1e-4 x7]`.
  - dec_joint_s2 differs only in the flat init seed-1 checkpoint, `random_seed = 1` and `random_seed_offset 1000`.
  - cold_ctl drops `reverse_checkpoint_path` and adds `reverse_duration_prior` + mode `init`.
- cold_ctl verdict: **do not reuse** L2-0's `cold_ctl` (`UdhhxiGIMBob`).
  - A value-level diff against its written config shows only the added `reverse_duration_prior` / `_mode init`. The other keys that differ are the ones the training job adds itself.
  - A9 requires the wave's duration setting, so the new baseline is its own run.
- `test_sae_blankfree_distil` (k2 python, real model from `T18RrTNTdg65/a17_gold_em` returnn.config, real CV-holdout utterances): all ok.
  - Target rows sum to 1 within 4e-14 (float64).
  - `tau·∂logZ/∂log_q` from `forward_log_z` autograd matches the posterior within 1e-14. The target does not change when theta is perturbed.
  - On 4 utterances (T 253–587) the CE matches the manual value (1.234523).
  - The gradient with respect to log_q matches −post/retained/n within 6e-11. Every recognizer parameter has a gradient and every phi gradient is None.
  - The `dp` is identical to the joint step's except for the temperature.
  - Sub-epochs 2 and 8 delegate to the joint step.
- `test_a18_l22_readers` (constructed records, real bootstrap): all ok.
  - classify; the LOWER, HIGHER and TIE cases; CODE BROKEN and OBJECTIVE ONLY; B swallowing a delta; identity FAIL giving VOID.
  - The pairing set; refusal of a mismatched setting or checkpoint.
  - The DURINIT clauses and band edges, and the rule text; the resolver's `resolve` and `run()`.

## Undetermined (not chosen; reported)
1. N7: `DISTIL_LR` and `DISTIL_TAU` in l22_v1 are `None`, so `build_bridge` refuses to build.
   - The literature note on alpha = 0 gives tau 1; the arms hold 2.0.
   - `DISTIL_SUBEPOCHS = 1` comes from the registered "one sub-epoch" (sub-epoch 1 of 8).
2. G4a.L2.4 does not name which arm decides the gate, so the reader prints all four against cold_ctl.
3. A18 (b) registers no outcome for the case where G-dur and r30-dur read NO LIFT and r70-dur lifts. The reader prints `NOT CLASSIFIED`.

## Assumptions (my choices)
- The distil CE uses l_tau's normalisation (divided by retained frames, mean over kept utterances).
- The distil target is the plain l_tau lattice with no k2 lexical term.
- The eval runs at ONE setting: dec_joint's model args without the init paths. The duration prior is init-only, so this changes nothing for cold_ctl.
- The identity tolerance of 0.005, N_BOOT 2000 and seed 0 come from D16.
- The stage-1 arms are named `rank1..rank4`.

## Concerns
1. The lift pack is wired into keyinit `py()` without a flag.
   - Any restart of the keyinit manager (pid 1192448) would also run the unreviewed lift pack. It waits for ge1MKcAPmZIV to finish.
   - A17 (i)'s jobs (T18RrTNTdg65 and its reads) would then sit in two managers' graphs.
   - Code review should happen before any restart.
2. keyarms `py()` now pulls in the stage-1 key-search jobs.
   - Its hashes depend on `KeySearchSelectJob.g9wsznNnqmyO`, which will move if the in-progress stage-1 fix edits `key_search_jobs.py`.
3. `blankfree_training` always adds the generic alias `sae/4a/blankfree/training`. The cold_ctl job will repoint that alias (it currently points at `5lBwcDjv2ItL`). This is cosmetic.
4. `cold_ctl` is a gpu-1 job on OverSubscribe=EXCLUSIVE nodes. Whether gpupack routes it or it idles a node needs checking before launch.
5. The crosseval forwards pass `expected_utterances=None`. The reader instead asserts identical tags across cells; the CV holdout has 285 utterances and 164 speakers.
6. With a `durfrz` wave setting, cold_ctl would freeze durations while the dec arms would not. The current setting is `durinit`.

Untested: nothing ran on GPU and no sisyphus job was run end to end. `BridgeReadJob.run` was not run against real crosseval files, and the D16 identity was not run on a real ep8 checkpoint.

## Round 2 (coordinator decisions 2026-09-24), commit 84b77283

Decisions applied:
1. The keyinit graph no longer pulls in A17. `DurinitBasinLiftReadJob(pers, phi_reports)` depends only on the lift pack's reads. The ep0 phi comes from keyinit's own report. The three PairedPerDeltaJobs against A17 (i) are built by `_paired_rows` in a18_v1, behind `A18B_PAIRED_ROWS = False`; their arguments are unchanged, so flipping the flag should give 2wMw96BmGytD, UnT62NW3k2M9 and neMyJu1lkoSl (not rebuilt this round).
2. The cold_ctl baseline is L2-0's banked `PackedBlankfreeTrainJob.UdhhxiGIMBob/cold_ctl`, pinned with `_pin` (no finished job enters the graph). The pins cover the ep1/2/4/8 checkpoints, the learning_rates file and the banked dev-other PER reads (MRzJUdkdFw0i, X85IkX4ih0rj, RQR5XVWnbe8g, sQ3EqBbi53lJ). Its ep8 checkpoint is cross-evaluated at the arms' one setting in ReturnnForwardJobV2.sVxuayfU6CYi, which is shared by both bridges. Its phi generative PER is read afresh. BoundedBlankfreeTrainingJob.O4lQfx6Qz1K7 is gone. The difference in phi's init is disclosed in the l22 module docstring and the reader docstring.
3. DISTIL_TAU = 2.0 and DISTIL_LR = 1e-5 are now the module defaults. The l_tau normalisation and the absence of a k2 term in the target are documented in sae_blankfree_distil.py.
4. The G4a.L2.4 verdict is decided by dec_joint vs cold_ctl only, with B = |dec_joint - dec_joint_s2|. The other arms are printed as "reported, deciding nothing".
5. The uncovered A18 (b) case reads MIXED.

BRIDGE_WAVE and BRIDGE_KEYARMS stay False. Nothing was launched.

New hashes (flag-on builds, real values):
- wave bridge: SelectedCheckpointJob.uZtbsDtueCrI, PackedBlankfreeTrainJob.63nbj6Jgiegj, BridgeReadJob.v6SONiAZL7rH, GenMargSampleJob.zXJjnNqU7kTa.
- key-arms bridge: SelectedCheckpointJob.PAV7erhRs8wq, PackedBlankfreeTrainJob.SCtv4DjzFQ50, BridgeReadJob.HWuOTb0Tefo1.
- The pack ids equal Round 1's (Round 1's test values were the same numbers).
- keyinit: DurinitBasinLiftReadJob.uRzbIh0EfQXG (was VyOgC0BZvRA9).

Checks (from the setup dir; files in the session scratchpad a18/r2_*):
- Hash stability: the em, a14, a17, keysearch_s1 and em_ext job sets are identical to the pre-change dumps (0 removed, 0 added).
- keyinit: 0 removed and 86 added (1 pack, 16 PER, 20 extract, 16 gendecode, 32 forwards, 1 reader). None of the added ids is in the A17 graph, and no A17-only id is in the keyinit graph. No T18RrTNTdg65 and no PairedPerDeltaJob.
- Flag-on builds for em and keyarms both load, and neither contains O4lQfx6Qz1K7.
- In the real bridge, every arm's lrs are [1e-5, 1e-4 x7], the dec_distil train_step has distil_tau 2.0 and distil_subepochs 1, and the pack rqmt is 4 GPU and 11.5 h.
- Config diff of dec_joint's config against L2-0 cold_ctl's written returnn.config, compared value by value: the only model difference is get_model.reverse_checkpoint_path. The remaining differences are runtime keys the job writes (device, log, model, num_epochs, save_interval, task, target, learning_rate_file, cleanup keep). The optimizer entries match except for a function address; the comparison covered only the first 200 characters.
- Tests:
  - test_a18_l22_readers: all ok. It covers MIXED, the dec_joint-only verdict, a LOWER on another arm not deciding, and the reader taking no paired input.
  - New sae/emc/test_l22_pins: all ok. The pinned pack is the ladder's pack_r2, and every pinned file and PER id is the ladder's own cold_ctl output.
  - test_sae_blankfree_distil (k2 env): all ok.

Open: none from the spec. PAIRED_WITH remains in a18_keyinit_lift_jobs.py, used by `_paired_rows`.
