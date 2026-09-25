# A14 (i) rt_em lift read extraction — 2026-09-24

Job dir: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_a14_jobs/A14LiftReadJob.GAKp6IJ5pA3l
(resolved from output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_em/a14/rt_em/read/{a14_lift.json,report.txt})
Pack dir: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.e9xZa5ElF16P

1. Verdict string printed: "VERDICT: EM PHI DOES NOT LIFT" (report.txt), json field "verdict": "EM PHI DOES NOT LIFT"

2. Per-arm PER (dev-other greedy) and class@ep8:
- cold_ctl (baseline): ep1 0.8881, ep2 0.8864, ep4 0.8554, ep8 0.8487
- rt_em_durinit_s1: ep1 0.8639740516147228, ep2 0.8521562544069948, ep4 0.8343534057255676, ep8 0.8397687209138345, class NO LIFT
- rt_em_durinit_s2: ep1 0.8394076999012833, ep2 0.8309237061063319, ep4 0.8241884078409251, ep8 0.8200479481032295, class NO LIFT
- rt_em_durfrz_s1: ep1 0.8662529967564518, ep2 0.8642899450007051, ep4 0.8404681991256522, ep8 0.8423015089550134, class NO LIFT
- rt_em_durfrz_s2: ep1 0.845071217035679, ep2 0.8293160344098153, ep4 0.8198900014102384, ep8 0.8206233253419828, class NO LIFT
Bands: lift_per 0.5, partial_per 0.8164 (from json "bands").

3. Paired row rt_em minus cold_ctl at ep8 (delta_per, 95% CI, method PairedPerDeltaJob, speaker-clustered CI, 2864 utt each):
- durinit_s1: -0.008929629107319093 [-0.012758423548980604, -0.005179547876558449]
- durinit_s2: -0.028650401917924118 [-0.03177339167869429, -0.025635140770651918]
- durfrz_s1: -0.006396841066140135 [-0.010349339140190905, -0.0025701725142685604]
- durfrz_s2: -0.02807502467917078 [-0.03174446046413384, -0.024765500420644192]
(per_cold_ctl=0.8486983500211536 for all rows; baseline_name "cold_ctl")

4. Phi generative PER (direct/Hungarian) or NMI per arm/epoch: MISSING (searched: a14_lift.json full content, report.txt — neither field present in this job's output; job reports only greedy dev-other PER and paired deltas)

5. Init phi's S (held-out tau=1 NLL/frame, 260 set) per arm: MISSING (searched: a14_lift.json, report.txt, log.run.1 — not printed by A14LiftReadJob). A registered (not per-arm-printed) value exists in config docstring/constant: REGISTERED_S_MIN = 3.2990, "the lowest of the six A10 restarts at 48" — recipe/2025-10-speech-llm/src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_lexlat_v2_a14_v1.py line 79; docstring lines 36-37 state FLOOR_SEED=1 is "the seed of the S_min restart (durinit s1)" — i.e. S_min is associated with durinit s1, not each of the four arms individually.

A10 restart source checkpoint hash each arm initialised from (ExtractSubmoduleCheckpointJob, sub-epoch 48, resolved via alias sae/4a/lexlat_v2/em/a14/rt_em/<arm>/phi_extract):
- rt_em_durinit_s1: ExtractSubmoduleCheckpointJob.c5WVHZvlSuhF
- rt_em_durinit_s2: ExtractSubmoduleCheckpointJob.sPCV5PXQWpJJ
- rt_em_durfrz_s1: ExtractSubmoduleCheckpointJob.H7YC2Iub8bOh
- rt_em_durfrz_s2: ExtractSubmoduleCheckpointJob.vM4xvAHEdVM0
(all under work/speech_llm/sae/emc/emc_train_jobs/)

6. Warnings/CANNOT_TELL/NaN: none found in job's own log.run.1 or in the report/json (no "CANNOT_TELL" verdict, no NaN emitted). Pack per-arm log.run.1 files contain repeated "Cache manager: Error occurred, using local file" lines (non-fatal, dataset caching) and one "Learning-rate-control: error key 'train_loss_agg'" informational line per arm (RETURNN LR-control lookup notice, not an error/NaN in losses). Strict grep for standalone "nan"/"inf" tokens (excluding "information"/"infinite") in each arm's log.run.1: 0 matches for all four arms.

7. Checkpoint / sub-epoch completeness: confirmed. All four arms have epoch.001.pt, epoch.002.pt, epoch.004.pt, epoch.008.pt (plus epoch.008.opt.pt) under PackedBlankfreeTrainJob.e9xZa5ElF16P/output/<arm>/models/. No NaN/inf found in bounded grep of each arm's log.run.1 (see item 6).
