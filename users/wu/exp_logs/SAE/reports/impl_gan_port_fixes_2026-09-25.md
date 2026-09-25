# GAN port: the code-review fixes (2026-09-25)

Status: DONE_WITH_CONCERNS. All seven deltas are applied. Delta 2 could not be done through `gpu_mem`
alone, because i6's routing rule does not let a fairseq training reach V100 through `gpu_mem`. The
trainings therefore get an explicit partition, taken from i6's own settings constant, and not
hard-coded. No job id moved. Nothing was committed.

Worktree: /e/project1/spell/wu24/worktrees/i6_experiments_cycle_consistency (branch
haotian_cycle_consistency_unsupervised), package users/wu/experiments/unsupervised_asr/ (PKG).
Inputs:
- review: reports/review_gan_port_2026-09-25.md;
- implementer reports: reports/impl_gan_port_{A,B,C,wire}_2026-09-25.md;
- i6 routing: users/wu/exp_logs/SAE_i6/reports/impl_v100_routing_2026-09-25.md, impl_settings_time_2026-09-24.md,
  impl_settings_p0_2026-09-24.md, and SAE_i6_P0.md:185-193.

## Files (all untracked; `git status --porcelain` shows only `??`, 15 entries = the 14 before + 1 new)

- PKG/training/w2vu2_gan.py:244-254: `GAN_RQMT` mem 60 -> 100. The comment above it gives the measured
  need and its source.
- PKG/config/w2vu2.py:
  - docstring:
    - Resources (54-83): GAN mem, the CTC mem evidence, the i6 routing rule and the partition;
    - Environment (87-111): CUDA_ARCH, and the extras script;
    - new section "Checks on i6" (113-127);
  - code:
    - `I6_TRAIN_PARTITION_SETTING` (183-192);
    - `_i6_train_rqmt` (200-207), applied to the 5 GAN trainings (239) and the CTC student (300).
- PKG/env/build_w2vu_env.sh:
  - lines 34-37: the GATE_CUDA comment;
  - lines 161-167: `assert torch.cuda.is_available()` under GATE_CUDA=1, placed before the arch assert.
- PKG/env/w2vu2_port_extras.sh (new, executable): installs torchaudio==<env torch> with pip `--no-deps`,
  ensures scikit-learn 1.8.0, and runs a gate.
- Tests:
  - PKG/tests/test_w2vu2_gan_inputs.py:189: the expected GAN mem is now 100;
  - PKG/tests/test_w2vu2_config.py:
    - `GAN_RQMT` mem 100;
    - the fixture was split into `built` (settings constant removed) and `built_i6`, plus the helper `_build`;
    - new test `test_i6_trainings_carry_the_settings_partition_and_nothing_else_does`.

## Per delta

1. **GAN memory.** mem is now 100.
   - Measured need, from review F1 (s0 usage.run.1 and sacct 968302): about 42 GB of mmap'd file
     working set plus up to about 45-57 GB of anonymous memory. The 60 was never enforced on JUPITER
     (`mem=0`, `--exclusive`).
   - Hash: rqmt is not hashed (`FairseqHydraTrainingJob.hash` hashes only the command-line args, the
     config, the python exe and the root). The check is below.
   - CTC student: kept at 60. I read its banked job myself. usage.run.1 max rss is 49.78 GiB, and
     sacct MaxRSS for 976159 is 42985792K, about 41.0 GiB. Both are per-process sums, so both are
     upper bounds, and both are below 60. The request therefore has evidence behind it, and I did not
     change it.
2. **GPU routing.**
   - i6's `check_engine_limits` rules:
     - `-p gpu_32gb` (GPU_ROUTE_TRAIN) applies only to the `run` task of the exact class
       `i6_core.returnn.training.ReturnnTrainingJob` (impl_settings_time: matched on module and class
       name, not isinstance);
     - any other GPU job goes to flex24 (gpu_24gb A10, or gpu_48gb) when gpu_mem <= 24, and to
       gpu_48gb (L40S) otherwise;
     - an explicit `-p` in the rqmt's `sbatch_args` wins over all of these.
   - Consequence: no `gpu_mem` value puts a FairseqHydraTrainingJob on V100.
   - Change: `_i6_train_rqmt` sets gpu_mem 32 and, when the loaded settings define `GPU_ROUTE_TRAIN`,
     `sbatch_args = ["-p", gs.GPU_ROUTE_TRAIN]`. sisyphus execs settings.py into the gs module
     (global_settings.py:437), so the constant is readable as gs.GPU_ROUTE_TRAIN. On JUPITER the
     constant is absent and nothing is added.
   - Decodes and forwards are unchanged. gpu_mem 40 is above 24, so they go to gpu_48gb (L40S).
   - The docstring now says the partition comes from i6's settings.
3. **CUDA_ARCH.**
   - Use 70, on a V100 node. That is the GPU of the trainings, and those are the jobs that must not
     fall back to CPU.
   - Why not 89: pytorch v2.6.0 .ci/manywheel/build_cuda.sh builds the x86_64 cu126 wheel for
     5.0;6.0;7.0;7.5;8.0;8.6;9.0, so sm_89 is absent. sm_86 covers the L40S.
   - On a CPU-only host, set GATE_CUDA=0. The CUDA-banner check then becomes the only CUDA evidence.
   - Note: the JUPITER aarch64 w2vu env lists [sm_50, 80, 86, 89, 90, 90a] and no sm_70. That is a
     different wheel from the one i6 will install.
4. **CUDA gate.** Under GATE_CUDA=1 the gate now asserts `torch.cuda.is_available()`. The assert comes
   first, because `get_arch_list()` returns [] without CUDA.
5. **Extras script.** It is env/w2vu2_port_extras.sh.
   - torchaudio: the version equals the env's torch (2.7.1). It is installed with pip `--no-deps`,
     which is how the reference env has it: a pip wheel on top of conda-forge pytorch 2.7.1.
   - scikit-learn 1.8.0: this is A's version. environment.yml:37 already pins it, so the script
     installs it from conda-forge only if the env holds a different version.
   - It is referenced in the docstring of config/w2vu2.py.
6. **Plot task.** i6_core's own `FairseqHydraTrainingJob.plot` was called, imported and not copied,
   with each banked log placed at `./outputs/x/hydra_train.log`.
   - GAN HOb2GgtYT7Bc train.log: OK, 984 `[train]`/`[valid]` lines, svgs written.
   - CTC BI1uYgPyTeQ0 train.log: OK, 159 lines.
   - Only a matplotlib legend warning was printed. No subclass was added, and the hashes are unchanged.
7. **Docstring notes.** The new "Checks on i6" section covers three points:
   - the artefact tests skip on i6, so the evidence is the JUPITER reports;
   - the CUDA banner ("CUDA enviroments for all 1 workers"; the banked s0 log has it at line 39, and
     the CTC log shows 4 workers);
   - fairseq shadowing, with a one-line check command.

   Flip wording: config/w2vu2.py:135-137 already said "10 utterances (6 dev-clean, 4 dev-other)", which
   matches B's report. No other file in the port mentions flips, so there was nothing to change.

## Checks

- **Job ids.** I dumped the sorted (class, job_id) of all 55 jobs that `w2vu2.py()` adds, three times:
  before the edits, after the edits, and after the edits with gs.GPU_ROUTE_TRAIN="gpu_32gb". All three
  dumps are byte-identical (cmp).
  - After the edits, the GAN rqmt is `{gpu 1, cpu 8, mem 100, time 11.5, gpu_mem 32}` and the CTC rqmt
    is `{gpu 4, cpu 16, mem 60, time 11.5, gpu_mem 32}`.
  - With the constant set, all 6 trainings also carry `sbatch_args ['-p','gpu_32gb']`, and no other job does.
- **Env gate.** I extracted the gate from build_w2vu_env.sh and ran it with the JUPITER w2vu env
  (torch 2.6.0+cu126, aarch64), with the wrapper's env vars set:
  - GPU visible, CUDA_ARCH 90 or 89: OK;
  - GPU visible, CUDA_ARCH 70: fails at the arch assert (this wheel has no sm_70);
  - `CUDA_VISIBLE_DEVICES=` (GPU hidden), any arch: fails with the new "CUDA not available through
    the wrapper" message;
  - GATE_CUDA=0 with the GPU hidden: OK, prints `cuda: False`.
  - `bash -n` passes on both scripts.
- **Extras script.** I ran it against a scratch venv with `--system-site-packages` over the speech_llm
  env. pip reported "Requirement already satisfied: torchaudio==2.7.1", so nothing was installed and
  the base env was not modified. The gate printed "OK torch 2.7.1 torchaudio 2.7.1 sklearn 1.8.0".
  - Not exercised: a real pip download on x86_64, and the conda branch for scikit-learn.
- **w2vu2 test files, fast** (`-m "not slow"`): 34 passed, 7 skipped, 7 deselected.
- **With SAE_ARTEFACT_DIR=1** (`-m "artefact and not slow"`): 7 passed.
- **test_w2vu2_config.py alone:** 10 passed, rerun after the last docstring edit.
- **Full package suite** (speech_llm env, CPU, 294 s): 497 passed, 2 failed, 23 skipped, 7 xfailed.
  - `test_model_reverse.py::test_segment_scores_explicit_sum[3]` is the known pre-existing failure.
  - `test_analysis_per.py::test_t2_6_posterior_dump_to_per_chain` failed because `black` was not on
    PATH in my shell (i6_core ReturnnConfig.write: `_black_path` None). This is an artefact of how I
    called pytest. Rerun with the env's bin on PATH, it passed.
  - Net result: 498 passed, which is the wire report's 497 plus the 1 new test, and 1 known failure.

## Undetermined, assumptions, proposals

- **Assumption.** i6's settings.py still defines `GPU_ROUTE_TRAIN`, and `check_engine_limits` honours an
  explicit `-p`. I read both from the i6 reports and did not see the file, which is not reachable from
  JUPITER. If the name changes, the trainings go to L40S without any error. The docstring asks the
  reader to check the first `submit_log.run`.
- **Proposal for the i6 settings owner** (outside my files): extend `is_train_run` to the
  FairseqHydraTrainingJob `run` task. That would also give the fairseq trainings the 72 h time floor.
  Today they keep 11.5 h and rely on resume.
- **New finding, not implemented: resubmits double the GAN request.** On a resume, sisyphus's default
  `update_engine_rqmt` (global_settings.py:112-133; SAE_i6 review_v100_routing_2026-09-25.md:65-69 applies the same rule on i6) compares the
  request with `usage.run.N` max rss. That value is the per-process RSS sum: about 307 GiB for the GAN.
  - The effect: every resubmit doubles mem, 100 -> 200 -> 400 GB. If the run hit its time limit,
    time also doubles, 11.5 -> 23 h.
  - A V100 node has about 1.5 TB, so a few resubmits still fit, but each one holds much more memory
    than the job needs. The CTC student (sum 49.8 < 60) does not double.
  - The same would happen with mem 60. Deciding whether to cap this, for example in i6's
    check_engine_limits, is outside my files.
- **i6_core version.** i6 runs i6_core 4537aaf; the plot check used this setup's ca161b7. The review
  states that FairseqHydraTrainingJob is identical upstream, but I did not diff 4537aaf.
- Scratch /e/project1/spell/wu24/worktrees/port_checks_gan/fix/ was deleted after the checks.
