# Review: GAN port fix round (2026-09-25)

Verdict: APPROVE. None of the five checks has a blocking finding. The port can be pushed, and the
i6 session can launch phase 0.

Inputs:
- the previous review: reports/review_gan_port_2026-09-25.md;
- the fix report: reports/impl_gan_port_fixes_2026-09-25.md;
- the worktree /e/project1/spell/wu24/worktrees/i6_experiments_cycle_consistency at 7e7c38aee. This
  equals the remote branch head (checked with `git ls-remote`). The package is
  users/wu/experiments/unsupervised_asr (PKG below).
- the i6 routing reports on the branch: users/wu/exp_logs/SAE_i6/reports/impl_v100_routing_2026-09-25.md
  and review_v100_routing_2026-09-25.md, plus SAE_i6_P0.md:184-194.

I read the cited lines directly. The files are untracked, so there is no git diff to compare against.

## 1. GAN mem 100, not hashed: PASS; the resubmit doubling is acceptable without a cap

**The value.** PKG/training/w2vu2_gan.py:254 sets `GAN_RQMT` mem to 100. It reaches the job through
`rqmt=dict(GAN_RQMT)` at line 383.

**Not hashed.**
- `FairseqHydraTrainingJob.hash` hashes only command_line_args, the hydra config, the python exe and
  the root.
- i6 runs i6_core 4537aaf. Its fairseq/training.py, fetched from GitHub, is byte-identical to this
  setup's ca161b7 (`diff`).
- I built the graph twice, once with GAN mem 60 and once with 100. The ids of all 120 jobs are
  identical.

**The doubling on resubmit.**
- Upstream sisyphus master runs `get_rqmt`, which keeps the logged value for each key the recipe did
  not change, and then `update_engine_rqmt`. The latter doubles mem when mem - rss < 0.25. Its rss is
  the sum of RSS over the process tree (engine.py:131-148), about 307.7 GiB for the GAN.
- The request therefore goes 100 -> 200 -> 400 and then stays at 400, because 400 - 307.7 = 92 is
  above 0.25. The shared mmap bound (6 train workers x about 40 GB, 6 valid workers x about 5 GB, and
  the main process) puts the sum well below 400.
- 400 GB is 409,600 MB. A gpu_32gb node has 1,546,786 MB, so every step fits a node. Time doubles
  only after a timeout (11.5 -> 23 -> 46 h).
- The CTC student does not double: 60 - 49.8 is above 0.25.
- Conclusion: no failure path, so no cap is needed. The cost is packing. After two resubmits the 5
  seeds hold up to 2 TB of the 3 TB on cn-32/33.
- If the i6 owner wants tighter packing: cap mem at about 128 in i6's `check_engine_limits` for this
  class. The capped value then stays stable across resubmits. This change is not in the port's files.

## 2. Partition routing: PASS

**Code.**
- PKG/config/w2vu2.py:192 defines `I6_TRAIN_PARTITION_SETTING = "GPU_ROUTE_TRAIN"`.
- `_i6_train_rqmt` (200-207) sets gpu_mem 32. It adds `sbatch_args = ["-p", gs.GPU_ROUTE_TRAIN]` only
  when the loaded settings define that constant with a truthy value.
- It is applied at line 239 (each GAN seed, via `train.rqmt.update`) and line 300 (CTC student, via
  the constructor's rqmt).

**How it reaches gs.** sisyphus execs settings.py into the gs module globals (global_settings.py
`update_global_settings_from_file`, the same code upstream), so the constant is visible at graph
build.

**i6 side.**
- impl_v100_routing sets `GPU_ROUTE_TRAIN = "gpu_32gb"` (V100, cn-32/33).
- An explicit `-p` in `sbatch_args` wins over every rule: impl and review (a), and the updated
  gpu_route_test asserts it.
- `is_train_run` matches only the exact class ReturnnTrainingJob, so without the explicit `-p` the
  fairseq trainings would go to gpu_48gb. The port's explicit `-p` is therefore needed.

**Live check.** I built `w2vu2.py()` twice: once with gs.GPU_ROUTE_TRAIN="gpu_32gb" and once without
it. For each job I read the Task objects that `job.tasks()` creates, which is what the manager
submits.
- Job ids: identical in both builds (120 jobs, 55 of them new).
- Only 6 tasks differ between the builds: the `run` tasks of the 5 GAN trainings and the CTC student.
  - GAN: `{gpu 1, cpu 8, mem 100, time 11.5, gpu_mem 32, sbatch_args ['-p','gpu_32gb']}`.
  - CTC: `{gpu 4, cpu 16, mem 60, time 11.5, gpu_mem 32, sbatch_args ['-p','gpu_32gb']}`.
  - The in-place GAN update is seen, because i6_core builds the Task from self.rqmt when `tasks()` is
    called.
- No other job has sbatch_args, so there is no second or conflicting `-p`.
- The sisyphus Slurm engine appends the list once (`options()`).

**Fallback, when the constant is absent.**
- No sbatch_args are added.
- On JUPITER, settings.py's `check_engine_limits` then adds `-A spell -p booster --exclusive` as for
  any job.
- On i6 without the constant, the trainings go to gpu_48gb (L40S). That is a hardware change, not a
  crash, and the docstring (74-77) tells the reader to check the first `submit_log.run`.

**Decodes.** Every GPU task of the forwards and decodes has gpu_mem 40 and no sbatch_args. That covers
the 11 new ReturnnForwardJobV2, the CtcPhoneDecodeJob `run` task and the CtcWordDecodeJob `decode`
task. i6's rule gpu_mem > 24 sends them to gpu_48gb (L40S).

**Not verifiable from JUPITER.**
- The exact test in i6's settings.py for an explicit `-p` (the file is not reachable from here). Our
  list has the same form that `check_engine_limits` itself emits.
- Whether `./gpu_route --rebalance` ignores pending FairseqHydraTrainingJobs. It is documented to act
  on flexible (flex24) jobs and to skip ReturnnTrainingJob.
- The first `submit_log.run` settles both.

## 3. CUDA gate, build_w2vu_env.sh:159-167: PASS

The whole block sits under `if os.environ["GATE_CUDA"] == "1"`. The script passes GATE_CUDA and
CUDA_ARCH on the wrapper's command line, and the wrapper execs python with them inherited.
`is_available()` is asserted before `get_arch_list()`.

I extracted lines 159-167 and ran them:

| Condition | Result |
|---|---|
| GATE_CUDA=0, GPU hidden (`CUDA_VISIBLE_DEVICES=`) | passes |
| GATE_CUDA=1, GPU hidden | fails with the new "CUDA not available through the wrapper" message |
| GATE_CUDA=1, GPU visible | passes |

Other checks:
- `set -euo pipefail` makes the build abort when the gate fails.
- `bash -n` passes.
- The documented CUDA_ARCH=70 matches the torch that step 2 installs (torch==2.6.0 from the cu126
  index). pytorch v2.6.0 .ci/manywheel/build_cuda.sh builds x86_64 12.6 for
  5.0;6.0;7.0;7.5;8.0;8.6;9.0, so sm_70 is present.

## 4. env/w2vu2_port_extras.sh: PASS

- The file is executable, `bash -n` passes, and it contains no /e/, /u/, /work/ or user paths. The
  prefix is an argument, and conda comes from CONDA_BIN or PATH.
- **torchaudio.** It installs `torchaudio==<env torch version without local tag>` with `--no-deps`,
  so pip cannot replace torch. The gate then asserts that the torch version is unchanged.
- **Build match on i6.**
  - environment.yml pins conda-forge `pytorch=2.7.1=cuda126*`, so torch.version.cuda is 12.6.
  - PyPI torch 2.7.1 on x86_64 requires nvidia-cuda-runtime-cu12==12.6.77, so the matching
    torchaudio 2.7.1 x86_64 wheel is the cu126 build. torchaudio's CUDA-version check on import
    therefore matches.
  - The wheel is manylinux_2_28 and requires torch==2.7.1, a requirement `--no-deps` skips.
- **scikit-learn.** 1.8.0, installed from conda-forge only if the env holds a different version.
- **Gate.** It checks the versions and calls kaldi.mfcc, compute_deltas and MiniBatchKMeans.
- **Not exercised.** A real pip download on x86_64. A mismatch there would fail loudly in the script's
  own CPU gate, before any GPU job.

## 5. Tree and tests: PASS

**Tree.** `git status --porcelain` shows 15 entries, all `??` and all under PKG: the 14 files from
before plus env/w2vu2_port_extras.sh. There are no exp_logs entries and no tracked modifications. The
same holds after my test runs (run with `-p no:cacheprovider` and PYTHONDONTWRITEBYTECODE).

**Tests.** speech_llm env, login-node CPU, with scratch PYTHONPATH symlinks to the worktree:

| Run | Result |
|---|---|
| w2vu2 test files, `-m "not slow"` | 34 passed, 7 skipped, 7 deselected |
| Full package suite | 498 passed, 1 failed, 23 skipped, 7 xfailed, 223 s |

- The only failure is test_model_reverse.py::test_segment_scores_explicit_sum[3], the known
  pre-existing one: the difference is 1.46e-5 against a tolerance of 1e-5.
- The new test test_i6_trainings_carry_the_settings_partition_and_nothing_else_does passes.

Scratch /e/project1/spell/wu24/worktrees/port_checks_gan/review2/ was deleted after the checks.
