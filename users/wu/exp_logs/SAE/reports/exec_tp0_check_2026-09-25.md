# TP0 finish check (2026-09-25) -- DONE
Config config/sae_4a_rename_tp0.py. Graph status (console, skip_finished=False): finished 92, no other state (no error/queue_error/retry_error/interrupted/input_missing). No manager restarted, nothing edited.

Pack: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.21tww6QQK0tH
- finished.tar.gz present (contains finished, finished.run.1). sacct 2011293_1 COMPLETED 0:0, 00:47:03.
- Arms output/{full_s1,full_s2,5pair_s1,control}: each 12 checkpoints epoch.001-012.pt (all sub-epochs kept), learning_rates 12 entries, log.run.1 "Finished training at epoch 12, global train step 684" (same step count all arms). Epoch 0 = init phi (loaded from init job, not a pack checkpoint, per reader docstring).
- Log scan: 0 NaN/inf lines, no OOM, no Traceback. Only "error" lines: 9x "Cache manager: Error occurred, using local file" (file-cache fallback, benign) and 2x "Learning-rate-control: error key ..." (RETURNN info line). No retries (single run.1 attempt).

Reader: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/rename_tp0_jobs/Tp0ReadJob.t7gzZhB2fWtn
- finished marker present; outputs output/report.txt (11874 B), output/tp0.json (139223 B); tp0.json epochs = [0,4,8,12], has deranged_arms/control_arm/verdict fields.
- log.run.1: no NaN/warning/error/traceback (only INFO lines).
Results not interpreted.
