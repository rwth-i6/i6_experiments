# Review: G0.V GPU test script (run_tests_gpu.sbatch), 2026-09-24

Verdict: PASS_WITH_NOTES. If you submit the script as written, it gives a trustworthy G0.V read of
the gpu- and k2-marked tests. There is no route by which a gpu- or k2-marked test can be skipped
while the job still exits 0 with the preflight passed. None of the notes below can turn a failure
into a pass. Read-only review; nothing was submitted.

Reviewed: `/work/asr4/hwu/sae_i6_tests/gpu_2026-09-24/run_tests_gpu.sbatch` (37 lines) and
`README.md` beside it; package `tests/conftest.py`; every test with a skip, importorskip or skipif,
or a gpu or k2 marker; `default_tools.py`; `settings.py`; `training/config.py` (serialization);
the RETURNN checkouts.

## Checks that pass
- **Resources.** The script requests `gpu_48gb` (nodes cn-[506-509], AllowAccounts=hlt,
  MaxTime 7 d), `--gres=gpu:1`, account hlt, 8 CPUs, 48G and 2 h. `gpu_check.sbatch` used the same
  account and gres syntax and ran on this partition (job 4333502, L40S cc 8.9, cn-508;
  `reports/env_build_2026-09-24.md`). The suite takes 66 s on the CPU, so 2 h is ample. One GPU
  stays well inside the QoS cap of 5.
- **Preflight abort.** `bash -n` passes. I simulated the `python - <<'PY' || { ...; exit 2; }`
  construct in a scratch script: when the heredoc raises, it prints FATAL, exits 2 and never
  reaches the next line. `nvidia-smi` failing gives exit 3. A missing CUDA gives `sys.exit(1)`,
  and a failing `import k2` or `.to("cuda")` raises; both lead to exit 2 before pytest.
- **No silent skip.** conftest.py:55-64 skips a gpu-marked test only when
  `torch.cuda.is_available()` is False. It skips a k2-marked test only when `import k2` raises.
  The module-level `pytest.importorskip("k2")` / `("torch")` calls (test_model_lexlat_k2.py:22-23,
  test_lm_prune.py:24-25) skip under the same condition. The preflight checks exactly these
  conditions in the same interpreter with the same PATH and PYTHONPATH. Only two tests carry the
  gpu marker: test_model_lattice.py:809 (T1.8) and test_model_lexlat_k2.py:493 (x2 tau). No
  unmarked test or package function picks CUDA by itself (grep of `cuda.is_available` and
  `"cuda"`; only CLI `_cmd_gpu_check` and a bench do). In pytest 9.1.1 a broken (non-ModuleNotFound)
  k2 import errors instead of skipping.
- **Full collection.** The only pytest config candidate is `recipe/i6_experiments/pyproject.toml`,
  which has no pytest section, so there are no addopts. No pytest plugins are installed (pytest
  9.1.1 only). No `-m` or `-k` is given, and no conftest.py sits above tests/. Slow tests are
  therefore collected. A collection error aborts with rc 2 (no `--continue-on-collection-errors`).
- **Exit code.** `rc=${PIPESTATUS[0]}` then `exit $rc`. Simulated: an rc of 1 through `| tee`
  comes out as job exit 1.
- **Output.** Slurm output, pytest log and junit all go to `/work/asr4/hwu/sae_i6_tests/gpu_2026-09-24`
  (shared). The pytest cache is off. The script writes nothing under log/, work/ or config/.
- **Interpreter and package code.** Training runs `SAE_PYTHON` = `/work/asr4/hwu/conda/envs/sae/bin/python`
  (settings.py:19). K2_PYTHON is unset, so the k2 arms use it too (default_tools.py:114). The
  training configs import the package live from `$S/recipe` (`Collection(make_local_package_copy=False)`,
  training/config.py:345-371). The tests use the same interpreter and the same tree.

## Notes (none produces a false green)
1. **RETURNN differs from training.** The PYTHONPATH at run_tests_gpu.sbatch:17 resolves
   `import returnn` to `recipe/returnn/returnn` at 5f752be4 (verified). Training uses
   CloneGitRepositoryJob.KQ3NuCaDE6QH = 00171dfe plus a working-tree diff that I verified is
   identical to `training/returnn_local_fixes.patch`. The two are 712 commits apart; 153 files
   under returnn/ differ, including `frontend/run_ctx.py` (`Loss.get_inv_norm_factor`) and
   `torch/frontend/_backend.py` (+314 lines).
   - **The G0.V open items are unaffected.** The gpu- and k2-marked tests import no RETURNN:
     test_model_lattice, test_model_lexlat_k2, test_lm_prune, lattice_oracle, k2_oracle,
     model/{lattice,reverse,lexlat,lexlat_k2,lexlat_k2_train} and lm/lexlat_k2_official use only
     torch, numpy and k2.
   - **These tests do run on 5f752be4, not the training RETURNN:**
     - test_model_blankfree.py:14-16 (T2.1-T2.5), through model/train_step.py:24-25
       (`rf.get_run_ctx().mark_as_loss`);
     - test_data_speaker.py:180/225/249 (rf);
     - test_data_ogg_zip.py:67 (OggZipDataset);
     - test_w2v2_features.py:80;
     - test_analysis_gaps.py:153-185, a real `rnn.py` forward that resolves to
       `recipe/returnn/rnn.py`.

     The CPU baseline had the same property. For the scalar losses the train step marks, the
     changed `get_inv_norm_factor` gives the same value (1 element).
   - **Fix, if exact parity is wanted.** Set
     `PYTHONPATH=$S/work/i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn:$S/recipe:$S/sisyphus`.
     The clone must come BEFORE `$S/recipe`, because `$S/recipe/returnn/__init__.py` is a regular
     package and would otherwise win.
2. **The k2 preflight never runs a k2 kernel** (run_tests_gpu.sbatch:29). `Fsa.to("cuda")` only
   copies arrays (k2/fsa.py:1111-1124), so the comment "fails if k2 lacks this arch" is not true,
   and `with_cuda` is printed but not asserted. No false green follows: a missing kernel would
   FAIL the parity test loudly, and k2 kernels already ran on an L40S (job 4333502). Fix: assert
   `k2.with_cuda`, and call `.get_tot_scores(use_double_scores=True, log_semiring=True)` on the
   cuda Fsa, expecting 0.5.
3. **README read-out.** README.md:14 says "only artefact skips allowed". The 4 tests in
   test_data_ffmpeg_pin.py are also skipped here (skipif at :23-25 on reference-cluster paths).
   Expected tally: 506 passed, 11 skipped (7 artefact + 4 ffmpeg_pin), 12 xfailed, 0 failed.
   Anything else is a finding. The grep at README.md:15 would miss a module-level importorskip
   reason ("could not import 'k2'"). The preflight excludes that case, but checking for
   `PASSED` on the 3 gpu test IDs is the direct check.
4. **Minor.** pytest's tmp_path is on the node-local /tmp, so the files of a failing test cannot
   be inspected afterwards. Optional fix: `--basetemp=$OUT/tmp_${SLURM_JOB_ID}`.
