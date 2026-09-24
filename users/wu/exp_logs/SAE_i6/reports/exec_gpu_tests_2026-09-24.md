# GPU test run, SAE_i6 P0 G0.V (gpu/k2 part), 2026-09-24

Status: DONE. Slurm job 4334384, gpu_48gb, node cn-508, COMPLETED, elapsed 00:02:07 (pytest 120.07 s), pytest exit code 0.

## Edits applied to /work/asr4/hwu/sae_i6_tests/gpu_2026-09-24/run_tests_gpu.sbatch (review fixes only)
1. PYTHONPATH starts with the real path of the training RETURNN clone
   /work/asr4/hwu/setups/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn
   (HEAD 00171dfe2; modified engine.py, updater.py, task_system.py). Script prints returnn.__file__ and exits 2 unless it resolves under that clone.
2. k2 preflight asserts k2.with_cuda and runs get_tot_scores(log_semiring=True, use_double_scores=True) and get_forward_scores on a 2-Fsa vector on cuda; exit 2 on failure.
3. pytest gets --basetemp=$OUT/pytest_tmp_<jobid>.
4. README records the expected result (506 passed, 11 skipped = 7 artefact + 4 ffmpeg pin, 12 xfailed, 0 failed, 0 CUDA/k2 skips).
bash -n passed; submitted from /work/asr4/hwu/sae_i6_tests/gpu_2026-09-24.

## Results
- Device: NVIDIA L40S, capability (8, 9); torch 2.7.1 / CUDA 12.6; k2 with_cuda True; k2 cuda smoke: tot 1.0759394 on cuda:0, ok.
- returnn.__file__ = .../CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn/returnn/__init__.py (the training clone).
- Summary: 506 passed, 11 skipped, 12 xfailed in 120.07s. Matches the review's expectation exactly.
- FAILED/ERROR tests: none. (One captured log line "ERROR root:job.py:90 Wrong input arguments or missing __init__ function?" belongs to the passing test_job_ctor_checks; it is expected output, not a test error.)
- Skips (11, none CUDA/k2): test_artefacts.py:131,152,161,184,201,229,256 "SAE_ARTEFACT_DIR is not set"; test_data_ffmpeg_pin.py:88,109,123,141 "reference ffmpeg or dev-other FLAC not available".
- xfail: 12, same count as the CPU run's 12 strict xfails; no XPASS.
- test_t1_8_gpu_parity: PASSED (no deviation printed).
- test_t119_cpu_cuda_parity_log_z_hlg_and_log_z_h[1.0] and [2.0]: both PASSED. Recorded values agree CPU vs CUDA to ~1e-15 (e.g. tau=1 log_z_hlg -2.8015962830847903 vs -2.801596283084792; tau=2 log_z_h 2.5734434696836317 vs 2.573443469683...), far inside 1e-5.

## Artifacts
- /work/asr4/hwu/sae_i6_tests/gpu_2026-09-24/slurm-4334384.out
- /work/asr4/hwu/sae_i6_tests/gpu_2026-09-24/pytest_4334384.log
- /work/asr4/hwu/sae_i6_tests/gpu_2026-09-24/junit_4334384.xml
- /work/asr4/hwu/sae_i6_tests/gpu_2026-09-24/pytest_tmp_4334384/
