# Launch sae_4a_private_code manager -- 2026-09-20

Status: DONE

- Activated sis_env first (`which black` resolved to `.../sis_env/bin/black`).
- No prior manager for config/sae_4a_private_code.py; not in sis_managers.sh IN_SCOPE, so used the
  manual nohup form (matching the infomax restart precedent):
  `nohup .../sis_env/bin/python tools/sisyphus/sis --log_level 20 m -r config/sae_4a_private_code.py`.
  Manager pid: 1737485. Log: log/sae_4a_private_code.manager.20260920T185634.log
  (log/sae_4a_private_code.manager.log symlink points to it). Started plain, no -co/-cio.
- Census at graph load: exactly 8 runnable jobs, all in
  work/speech_llm/sae/emc/private_code/ -- no training/forward job was created or rerun. Nothing
  to escalate as BLOCKED.
  - PrivateCodeAnalysisJob x6: x79av2mvTxo3 (ctrl_50/ep1/dev-clean), xuCBRVe3CzMR (ep1/dev-other),
    eLdlRYU1YtQZ (ep4/dev-clean), DMAmCpn0L5mB (ep4/dev-other), ssh9wJM8pxjI (ep10/dev-clean),
    VAYV87DbdDwU (ep10/dev-other).
  - SymbolDeciphermentJob x2: 36NfY7XDOL3f (ep4, matches expected hash), FkH0wprbfjaN (ep10,
    matches expected hash).
  All submitted to the short/CPU engine immediately (SLURM ids 1913601-1913608).
- Foreground poll (60 s x up to 15 iters): 4/8 finished by iter 2, 8/8 finished by iter 10
  (~10 min elapsed) -- consistent with expected ~36 s per analysis job and ~10 min per decipher
  job. Zero errors throughout.
- All 6 PrivateCodeAnalysisJob dirs have output/private_code.json + private_code.md; both
  SymbolDeciphermentJob dirs have output/decipher.json + decipher.md.
- Manager 1737485 exited cleanly after all 8 jobs finished (no longer in ps). Other managers
  untouched and still running: 1355841 (sae_4a_infomax_pack), 2945974 (sae_4a_budget_pack).
- No anomalies.

Full report path: recipe/i6_experiments/users/wu/exp_logs/SAE/reports/exec_launch_private_code_2026-09-20.md
