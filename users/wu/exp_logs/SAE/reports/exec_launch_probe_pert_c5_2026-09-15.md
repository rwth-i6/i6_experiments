# Executor launch report: bt_probe, pert_dump, c5a/c5b, emc_profile_step -- 2026-09-15

## Managers started (manual form, sis_env activated; neither config is in sis_managers.sh IN_SCOPE list so the wrapper was bypassed per skill instructions)
- sae_4a_bt_probe: pid 2037231, `config/sae_4a_bt_probe.py`. Alive at report time (~15 min elapsed),
  manager.log has no error/traceback (only the pre-existing HF-cleanup WARNING, unrelated), still in
  recipe-import phase. None of the 8 target BtProbeJob hashes (1E6mfTahsKLZ, 3BFG7tATDxqI,
  5owmxhkPXkdf, 91VDNBrLup2B, 96K3LFZ7brhJ, ZXlljMDUea5m, rNJbMa2sG9ia, uXVX60ASzhKI) had appeared
  under work/ yet at exit. No error state confirmed either -- config load for this setup routinely
  takes minutes.
- sae_4a_pert_dump: pid 2037343, `config/sae_4a_pert_dump.py`. Manager exited cleanly (no traceback).
  Graph status confirmed: 5/5 jobs `finished` (DownloadHuggingFaceRepoJob.JcEANaYZr2oe,
  TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb, L15FeatureHdfJob.OTO33H7dWkfy,
  AvStatesJob.pBRPNg7F2C8c, SpeedPerturbFrameCheckJob.oqcrbnKKCd2H) -- all reused/already banked, DONE.

## SLURM (sbatch, per analysis/out/*.cmd)
- c5a+c5b (single 3h alloc, sequential): job 1818371, `cold_c5`. sacct: COMPLETED 0:0. Outputs written:
  analysis/out/cold.c5a_flat_fitphi_ratetilt.bigram.dev-other.{txt,json},
  analysis/out/cold.c5b_flat_fitphi_pindur_ratetilt.bigram.dev-other.{txt,json};
  log analysis/out/cold_bigram.c5.run.log.
- emc_profile_step (coordinator addition): job 1818385, `emc_profile_step`. Submitted, running at
  report time (not polled to completion per "do not wait"). Will write
  analysis/out/emc_profile_step.{b128,b32,b256}.{txt,json}, log analysis/out/emc_profile_step.run.log.

## Nothing failed to submit.
