# Review: keyinit launch (A16 (b) gold-key control + A17 (iii) G-dur / r30-dur / r70-dur), 2026-09-24

Verdict: APPROVE. There are no blocking findings. Launch with the live manager form (section 7), not the proposed `--log_level 30 m -io` form.

Scope: commit 572b59ca in recipe/2025-10-speech-llm, config `config/sae_4a_lexlat_v2_keyinit.py`, which is a shim over
`configs/config_sae_4a_lexlat_v2_keyinit_v1.py`. Implementer report: `reports/impl_a16b_keyinit_2026-09-24.md`.
The review was static and bounded. The graph was loaded read-only (`sis console -s -c`), the arm configs were written to the scratchpad and diffed, and the implementer's CPU suite
`speech_llm.sae.emc.test_lexlat_v2_keyinit` was re-run. It returned ALL PASS with exit 0: table, reader edges, graph, init (constructed and gold key), 3 swaps,
diagnostics, arm configs, control tau1, key arms, readers on the banked A10 files, boundaries on real MFA data, and the forward pass at sub-epochs 1 and 2 with finite gradients.

## 1. Init construction and the single delta
- The written arm configs were diffed against the A10 durinit s1 config. The only change in get_model is that `reverse_duration_prior` + `reverse_duration_prior_mode: "init"` are replaced by
  `reverse_checkpoint_path`. The post-config lines `model` and `torch_log_memory_usage` are unhashed. The following are identical: temperature_schedule (4 at sub-epoch 1, then 1), lr 3e-3,
  random_seed 1, num_epochs 48, prior RtzbESkOedsT (weight 1.0), null/frozen recognizer, and data.
  Against A14 (ii)'s gold-floor run fTBdXD0SwBaA, the only difference is reverse_checkpoint_path. The four arms differ from each other only in that path.
- No prior is dropped or doubled. Durinit is not set a second time through the prior (reverse_duration_prior is absent; the model asserts refuse to have both
  set). The durations therefore come only from the loaded checkpoint, and `_load_state` is strict.
- The test output shows dur_logits == durinit_logits == A10 durinit s1's constructed tensor. E[d] is 4.4138 for phones and 26.0 for SIL in all 4 arms, which matches the start log of A10
  durinit s1. The SIL row is 0, the same as A10 (reverse.py zero init).
- Swaps: 11 non-dur tensors are bit-equal to the sources BlankfreeSupervisedReverseInitJob 16v7R6ztSq1u (gold),
  HpmOCSCklRsB (rho 0.3) and OLIUO3BGy0ug (rho 0.7). These are the A14 (ii) floor inits.
- Control: the key is GoldUnitKeyJob.sLnMRRd2qO0t/output/key.json (finished; 2821 L2-0 fit utterances, so it is disjoint from the 260 set).
  The emission is 0.9 ML + 0.1/500 (the A12 smoothing), with a uniform row for the empty symbols OY and ZH. JS(marginal || table) is at most 2.25e-12.

## 2. Scoring
- S at 0 (the init output), at 1..48 on the CV holdout, and dev-other decodes plus reports at 0, 4, .., 48. It is computed on the 260 set (A13 disjoint.segments,
  asserted len 260), paired over tags possible in all compared jsons, with the A14 (ii) `_s` / `_possible_in`. Generative PER is reported beside S.
  Labels (MFA, gold) enter only reports and readers, never training or selection.
- KeyInitControlReadJob and A17SegmentationReadJob carry the registered rules verbatim, and the edge tests pass.

## 3. Off switches
- `CONTROL_TAU1 = False`. The graph dump shows control_tau1 None and no PhiFirstProbeTrainingJob.
- keyarms_v1, key_search*, config_sae_1g and config_sae_3e1_d6 are not imported by the keyinit graph (checked in sys.modules). The uncommitted
  key_search_jobs.py / test_key_search.py / config_sae_1g_v1.py edits therefore do not reach this launch.

## 4. Unfinished jobs, routing and packing
The graph has 1805 jobs. The 505 unfinished jobs are: 444 ReturnnForwardJobV2, 52 GenDecodeReportJob, 3 PhiDurinitSwapJob, 1 PhiFromKeyInitJob,
1 PackedBlankfreeTrainJob ge1MKcAPmZIV, 1 each of KeyInitControlReadJob / A17SegmentationReadJob / SegmentationBoundaryJob, and 1 PhiFirstA10DiagnosticsDisjointJob.
None is in an error state, and none has an existing job dir.
- Each arm has 111 forwards: 3 at ep0 (CV marginal, CV decode, dev-other decode) + 96 (CV marginal + decode at 1..48) + 12 (dev-other decode at 4..48), for 4 x 111 = 444.
  None duplicates a finished job. The boundary job reuses A14 (ii)'s finished decodes.
- Routing by `check_engine_limits`:
  - Forwards: run goes to gpupack (1 GPU, 2 h, 32 GB, 4 cpu; queued le2h bucket) and create_files goes to short.
  - The pack goes to long: 4 GPU, 64 cpu, 256 GB, 4.19925 h (= 1.5 x (29.64 s + 48 x 209.3 s)), --exclusive.
  - Every other job is a mini task and goes to short (login).
- Estimate from the A14 (ii) floor analog (432 forwards):
  - That run used 48 gpupack packs of 8-12 tasks, one per sub-epoch as checkpoints appear (path_available). Each pack spent about 90 s on the node, 1.20 node-h in total.
  - For keyinit, expect about 49-52 packs, about 1.2-1.3 node-h, about 5 GPU-h.
- Pack wall: the wave packs ran at 1.03x the 1-GPU projection, which gives about 2.9 h for 48 sub-epochs against the 4.2 h allocation (about 11.6 of 16.8 GPU-h used).
- Total: one pack node for about 3 h, plus about 50 short gpupack nodes overlapping training. About 17 GPU-h.

## 5. Overlap with live managers
- None of the 505 keyinit-unfinished jobs appears in the em (4111121), a14 (2080167) or a17 (1096118) graphs, and the reverse also holds.
- Checked against the live graphs: the A17 ids T18RrTNTdg65, nVpD2O3xpfcJ and YtsRkvAl7Kl8 and the A14 (i) id e9xZa5ElF16P are not in keyinit.

## 6. Settings and hashes
- settings.py was not touched by this work. Its mtime is 2026-09-24 00:47, before the commit and before every live manager started. JOB_AUTO_CLEANUP = True at line 285.
- The A10 / A14 (ii) ids keep their hashes and are finished: fTBdXD0SwBaA, T2V8nn5obzj9, MU6Q3RanOQqF, lR4CfDiHyAvH, PiYQ1OCFD4ot, 8S8gTgbRFKU6,
  JhiT3D0MyfPg. The A17 ids are kept.

## 7. Launch form
- `m` is the alias for `manager`, and `-io` (ignore_once) is valid. With no errored jobs, `-io` is a no-op: it only skips the first-loop clear prompt and clears nothing.
  It does, however, drop the fail-closed exit that sis_managers.sh relies on, so it is better left out.
- `--log_level 30` hides the INFO lines (gpupack buffer/submit, state changes) that the executor needs to diagnose the 50 forward packs. `>` also overwrites an
  untimestamped log. Use the live form:
  `setsid nohup /e/project1/spell/wu24/env/sis_env/bin/python tools/sisyphus/sis --log_level 20 manager -r config/sae_4a_lexlat_v2_keyinit.py < /dev/null >> log/sae_4a_lexlat_v2_keyinit.manager.<ts>.log 2>&1 &`
  Run it from the setup dir on a login node.

## Notes (not blocking; no failure path at the expected operating point)
- keyinit_read_jobs.py:238,242: the control bar is recomputed as paired S_min - 0.01 = 3.28903 on 260 tags. The registration (SAE_4A_lexlat_v2.md:263) and the A17
  reader (line 98, fixed 3.289) use 3.289. The two readings differ only if S_c falls in [3.28903, 3.289), or if T < 260 moves the paired S_min.
  The stand-ins pair 260/260.
- keyinit_read_jobs.py:196,294: the A17 (iii) joint tag set includes the report-only r30_dur arm, as in A14 (ii). If r30_dur were impossible on a tag, G-dur and r70-dur S
  would be computed on fewer than 260 tags, while the bar stays fixed at 3.289. The implementer disclosed this.
- phi_from_key.py:81 `PREACT_ON = 2.0` traces to neither the registration nor the reference. It is a free constant of the control's key-to-phi conversion (disclosed), and it affects
  only the control arm's dynamics, not the A17 (iii) arms.
