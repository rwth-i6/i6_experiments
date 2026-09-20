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

## Re-run v2

Status: DONE

- Activated sis_env; no prior manager for config/sae_4a_private_code.py running; other managers
  untouched (2945974 sae_4a_budget_pack, 1355841 sae_4a_infomax_pack, confirmed alive throughout).
- Started manager plainly (no -co/-cio):
  `nohup .../sis_env/bin/python tools/sisyphus/sis --log_level 30 m -r config/sae_4a_private_code.py`.
  Manager pid: 2103202. Log: log/sae_4a_private_code.manager.20260920T194359.log
  (log/sae_4a_private_code.manager.log symlink repointed to it).
- Census at graph load (console -s -c, skip_finished): exactly the 8 expected NEW jobs, all
  queued/CPU, all under work/speech_llm/sae/emc/private_code/ -- no training/forward/PER job:
  PrivateCodeAnalysisJob j5ybPkSFOSBq, 1bNrZdet9JcH, UPuAcKSAI5oK, QY8blUARrLxQ, cQbcIJtOamLm,
  OIDSbcHXTzsP; SymbolDeciphermentJob GpiTxaoRCZXG, m8EsFhL6ysqu. Matches the dispatch exactly.
  Nothing to escalate as BLOCKED.
- Foreground checks (per-dir finished/error marker, sleep 60 loop): all 6 PrivateCodeAnalysisJob
  dirs finished within the first check window; both SymbolDeciphermentJob dirs finished by
  iteration 10 of the follow-up 10x60s loop (~10-11 min), matching the ~11 min estimate. Zero
  errors observed on any of the 8 job dirs throughout.
- Verified output artifacts on disk: all 6 PrivateCodeAnalysisJob dirs have
  output/private_code.json + private_code.md; both SymbolDeciphermentJob dirs have
  output/decipher.json + decipher.md.
- Manager 2103202 exited cleanly after all 8 jobs finished (confirmed absent from ps). Other
  managers (2945974, 1355841) confirmed still alive and untouched.
- No anomalies.

Full report path: recipe/i6_experiments/users/wu/exp_logs/SAE/reports/exec_launch_private_code_2026-09-20.md
