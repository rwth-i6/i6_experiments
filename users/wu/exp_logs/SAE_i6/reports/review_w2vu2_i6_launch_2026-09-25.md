# Review: i6 launch of the wav2vec-U 2.0 GAN reproduction (w2vu2), 2026-09-25

Reviewer: code-reviewer (read-only; nothing applied, nothing submitted).
Code: 68b39418a, merged locally as 692d6e55a on haotian_cycle_consistency_unsupervised. Gate: G0.GAN (SAE_i6_P0.md:154-170).
Inputs: reports/impl_w2vu2_i6_setup_2026-09-25.md; exp_logs/SAE/reports/*gan_port*_2026-09-25.md; config/w2vu2.py.
S = /u/hwu/setups/librispeech-960/2026-09-24-unsupervised, P = S/recipe/i6_experiments/users/wu/experiments/unsupervised_asr.

## Verdict: PASS_WITH_FIXES

Steps 1 and 3 hold as proposed. Step 2 (the env build) does not hold as written: MUST 1 below. Step 4 holds once
the env passes a gate that pins torch.

## MUST

1. P/env/build_w2vu_env.sh:76-81 (step 3) - the i6 build installs torch 2.8.0 (cu128) and torchaudio 2.8.0 over
   the reference torch 2.6.0+cu126, and the gate still passes.
   - Step 3 runs `pip install --ignore-installed fairseq==0.12.2 ...` without --no-deps and without a torch pin.
   - fairseq v0.12.2 setup.py:199-213 lists the unpinned dependencies `torch` and `torchaudio>=0.8.0`.
   - With --ignore-installed, pip's resolver does not see installed distributions.
     pip/_internal/resolution/resolvelib/factory.py:122-129 sets `_installed_dists = {}`. That code is the same in
     pip 23.3.2 and 26.1.1. So the torch 2.6.0+cu126 installed at step 2 is no candidate, and pip resolves torch
     again from PyPI.
   - No pip config exists that would change the index (none in ~/.config/pip, ~/.pip, /etc/pip.conf or /etc/xdg/pip;
     no PIP_* variables). build.sbatch sets only PIP_NO_CACHE_DIR.
   - PyPI has cp39 x86_64 wheels for torch and torchaudio up to 2.8.0 at most.
   - A dry-run resolve with the same pip semantics picks torch 2.8.0, torchaudio 2.8.0, triton 3.4.0,
     nvidia-cuda-runtime-cu12 12.8.90 and nvidia-cudnn-cu12 9.10.2.21. The command was
     `--dry-run --ignore-installed --only-binary=:all: --python-version 3.9`, manylinux x86_64, for
     `torch 'torchaudio>=0.8.0'`.
   - --ignore-installed overwrites without uninstalling (the script's own numpy note, lines 73-75). The result is a
     torch 2.8.0 tree laid over 2.6.0's files, with two torch dist-infos.
   - Why the gate cannot catch it: pytorch v2.8.0 .ci/manywheel/build_cuda.sh:59-60 builds the cu128 wheel with sm_70
     (7.0;7.5;8.0;8.6;9.0;10.0;12.0). So "cuda available" and "sm_70 in arch list" both pass, and so do
     fairseq 0.12.2, numpy 1.23.5 and batch_by_size. The gate only prints torch.__version__ (line 180); it does not
     assert it.
   - Effect: all five GAN seeds and the student train on torch 2.8.0 / CUDA 12.8 / cuDNN 9.10 instead of the
     reference env. The reference env is torch 2.6.0+cu126: extract_gan_port_inventory:38, and
     impl_gan_port_fixes:106, which ran the gate against it.
   - This is an undisclosed second difference in the baseline run. On a gate miss it would confound the reading of
     the seed-collapse count and the PER.
   - Fix: done by the implementer before step 2, with a narrow re-check of the diff.
     - In step 3, keep torch at the version step 2 installed. Either:
       - add `torch==2.6.0+cu126` with `--extra-index-url https://download.pytorch.org/whl/cu126`. Checked: both
         torch-2.6.0+cu126 and torchaudio-2.6.0+cu126 cp39 linux x86_64 wheels exist there, so torchaudio resolves
         to the matching build. Or
       - use a constraints file with the same pin.
     - Add `assert torch.__version__ == "2.6.0+cu126"` to the gate.
   - The JUPITER env measured 2.6.0+cu126 and reportedly had no torchaudio. So either the reference env was not built
     in this step order, or its pip saw a different index. Either way, on i6 x86_64 with PyPI this step resolves to
     2.8.0.
   - Not run: a full step-3 dry run on a py3.9 interpreter. The fairseq and kenlm sdists cannot be resolved with
     --only-binary. No other package in the step-3 list constrains torch.

## SHOULD

1. P/env/build_w2vu_env.sh:133-147 - the wrapper exists before the gate (line 152) runs.
   - A creator-less input path counts as available as soon as it exists (sisyphus job_path.py:150-151).
   - So a manager started on a failed or partial env submits the trainings. A seed could then run with a broken env,
     or on CPU if CUDA is hidden: fairseq trains on CPU without an error.
   - Start the w2vu2 manager only after log/w2vu_env_build.<jobid>.out shows "OK 2.6.0+cu126 0.12.2 1.23.5 |
     cuda: True" and "== done". On any failure, remove /work/asr4/hwu/conda/envs/w2vu immediately.
2. SAE_i6_P0.md:165 - "fairseq 0.12.2 is imported from the w2vu env" cannot be read from anything the jobs write.
   - fairseq 0.12.2 logs neither its version nor its file path. hydra_train.log has the CUDA banner (line 163), and
     submit_log.run has `-p gpu_32gb` (line 164), but neither shows where fairseq came from.
   - Run the probe from the config/w2vu2.py docstring once CloneGitRepositoryJob.rnfyLkSbJUoz has finished and before
     the first GAN starts (command below), and file its output as the evidence for line 165.
3. P/env/w2vu2_port_extras.sh:37-40 - the scikit-learn branch would run `conda install -p <live sae env>` with
   whatever conda is on PATH, which is /usr/local/bin/conda here.
   - sklearn is 1.8.0 today, so the branch is skipped.
   - Running the script with CONDA_BIN=/bin/false makes sure the live env can never get a conda transaction from it.

## Notes (no action)

- torchaudio install (step 3 of the launch):
  - The wheel installs torchaudio/, torio/ and their dist-infos. The implementer report says only torchaudio/; the
    extra torio/ is new and imported by nothing.
  - --no-deps leaves torch 2.7.1 alone, and the extras gate asserts that.
  - No live code imports torchaudio: not the P0 or P1 package paths, RETURNN or i6_core. A running or resuming job
    cannot see a half-written package it never imports, so the install need not wait for idle.
  - It must finish before the w2vu2 manager starts, because MfccKmeansJob and W2vu2FeatureDataJob import torchaudio.
- The dev gold phone symbols reach dict.phn.txt through FairseqCtcDataJob (w2vu2_ctc.py:176-179). This is the
  symbol inventory only and is identical to production. No dev label reaches training or selection:
  - selection is argmin weighted_lm_ppl from checkpoint_last extra_state["best"];
  - the feature data write no .phn;
  - the text data have only a train split.
- The w2vu2 forwards and decodes on gpu_48gb share the 5-GPU L40S cap with P1. They may take slots ahead of P1 arm
  resubmissions. This is acceptable as queueing.
- Plot task (i6_core fairseq/training.py:331-450): a failure is visible and the checkpoints are intact.
  - The banked JUPITER hydra logs parse with 0 failures.
  - matplotlib mini tasks work here: a ReturnnTrainingJob finished.plot.1 exists.
  - Manual recovery by touching finished.plot.1 is acceptable. No edit is needed before launch.
- The implementer's patched dry load diffed w2vu2 and P0 only. For P1: a fake-task test of check_engine_limits and a
  grep find 0 FairseqHydraTrainingJob in the P1 all, arms and probe lists.

## Evidence per dispatch check

- Settings patch (step 1).
  - settings.py sha256 is 743b4f7eaf45d82f25ea378a9fb2dd4fe095d5e7a927d39565e45d205028d752, and
    `patch --dry-run` applies cleanly.
  - The delta is exactly W2VU_PYTHON plus the `is_fairseq_train_run` term. That term forces time into [72, 168] h
    for exact-class i6_core.fairseq.training.FairseqHydraTrainingJob run tasks only, before the -p early return.
  - A fake-task comparison against the unpatched file:
    - GAN run, CTC run and a 23 h resume leg go to 72 h, with `-p gpu_32gb` kept.
    - Unchanged: the plot task, ReturnnTrainingJob run, the forwards, the phone and word decodes, big-memory CPU
      jobs, and a same-named class from another module.
  - Live managers: P0 (pid 1646677) and P1 (pid 1726329) are both alive.
    - Each exec'd settings.py once at start, and check_engine_limits runs in the manager, so neither changes
      behaviour.
    - Workers re-exec settings.py, but nothing in P0 or P1 reads W2VU_*, and neither graph has a
      FairseqHydraTrainingJob.
    - Settings are not hashed (GLOBAL_SETTINGS_FILE_CONTENT is unused), so no job id moves.
    - GNU patch replaces the file by rename, so a worker never reads a half-written file.
- Graph. 101 jobs.
  - The 49 shared jobs are all finished on disk.
  - None of the 52 new jobs has a directory, and none of their hashes appears in any P0 or P1 job list: the latest
    P0 list after the cs8 edits, and P1 all, arms and probe.
- Resources.
  - 5 GAN seeds: gpu 1, cpu 8, mem 100, gpu_32gb, 72 h after the patch.
  - Student: 4 GPUs, cpu 16, mem 60 (i6_core multiplies the rqmt by gpu). It fits one gpu_32gb node (16 V100,
    96 CPUs, about 1.5 TB; cn-33 idle at review time).
  - 11 forwards: gpu_48gb, 2 h, cpu 4, mem 24. CtcPhoneDecode: gpu_48gb, 2 h. CtcWordDecode: gpu_48gb, 11.5 h,
    cpu 8, mem 64. CPU jobs: cpu_modern.
  - No gpu_11gb, A100 or log_gpu_80gb anywhere.
- CPU and fairseq-origin risk.
  - DEFAULT_ENVIRONMENT_KEEP keeps CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH.
  - The wrapper replaces LD_LIBRARY_PATH with the w2vu libs and sets PYTHONNOUSERSITE=1.
  - The run prepends the sparse clone to PYTHONPATH. Its fairseq/ is a namespace portion, so the env's regular
    fairseq 0.12.2 package wins.
  - Remaining risks are those in MUST 1 and SHOULD 1 and 2.
- G0.GAN readability.
  - Readable from what the jobs write: selection and PER from the select and decode jobs, the CUDA banner from
    hydra_train.log (fairseq re-applies job_logging on the master rank), and -p from submit_log.run.
  - Not readable that way: the fairseq origin (SHOULD 2).
- Disk. /work/asr4/hwu has 959G free. Home is at 8.08 of 10G, which PIP_NO_CACHE_DIR protects.

## Launch commands (run from S; step 2 only after MUST 1 is fixed and re-checked)

1. sha256sum settings.py    # expect 743b4f7e...d752
   patch -p1 --dry-run < analysis_out/w2vu2_settings.patch && patch -p1 < analysis_out/w2vu2_settings.patch
2. sbatch analysis/w2vu_env_build/build.sbatch
3. CONDA_BIN=/bin/false bash recipe/i6_experiments/users/wu/experiments/unsupervised_asr/env/w2vu2_port_extras.sh /work/asr4/hwu/conda/envs/sae
4. Only after log/w2vu_env_build.<jobid>.out shows "OK 2.6.0+cu126 0.12.2 1.23.5 | cuda: True" and "== done",
   and after step 3's gate passed:
   (PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH setsid nohup /work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis --log_level 30 m -r config/sae_i6_w2vu2.py > log/sae_i6_w2vu2.manager.log 2>&1 < /dev/null &)
   Then write the pid to log/sae_i6_w2vu2.manager.pid, and start the watcher:
   SIS_LAUNCHER="/work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis" PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH bash ~/.claude/skills/sis/sis_watch.sh <pid> config/sae_i6_w2vu2.py 60
   Fairseq-origin probe (SHOULD 2), after CloneGitRepositoryJob.rnfyLkSbJUoz has finished and before the first GAN starts:
   env -i HOME=$HOME PATH=/work/asr4/hwu/conda/envs/sae/bin:/usr/bin:/bin PYTHONPATH=/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/work/i6_core/tools/git/CloneGitRepositoryJob.rnfyLkSbJUoz/output/fairseq: /work/asr4/hwu/conda/envs/w2vu/bin/w2vu-python -c 'import fairseq,fairseq_cli;print(fairseq.__file__,fairseq.__version__,fairseq_cli.__file__)'
   Expect fairseq.__file__ under /work/asr4/hwu/conda/envs/w2vu/lib/python3.9/site-packages, version 0.12.2, and
   fairseq_cli from the clone.

---

## Addendum (re-review of the fix): PASS

Fix: P/env/build_w2vu_env.sh, +26/-3. sha256 is now 9526dd61...a6176, matching the implementer report.
Implementer report: S/reports/impl_w2vu2_env_torch_pin_2026-09-25.md.
MUST 1 and SHOULD 1 are resolved. SHOULD 2 is resolved at build time; the job-time probe stays optional (below).

### What I verified

- **Only this file changed.**
  - `git diff --stat` shows this one file, +26/-3, and the diff is exactly the three hunks in the implementer report.
  - The package has no other change since 692d6e55a: the three commits after it touch only documents.
  - Unchanged: settings.py (743b4f7e...d752), build.sbatch (mtime 17:47), w2vu2_settings.patch and
    w2vu2_port_extras.sh.
  - `bash -n` passes.
- **torch and torchaudio resolve to 2.6.0+cu126, and nothing later replaces them.**
  - The implementer's dry-run is in the session scratchpad: report_new.json and report_old.json, with the command
    in step3_new.sh. I read the JSON reports myself.
  - The run used the script's pip, 23.3.2, on Python 3.9.23 x86_64. It covered the full step-3 package list,
    including the sdists that my torch-only dry run could not cover.
  - The command matches the script's step 3 byte for byte, apart from `--dry-run --report`.
  - Result: torch 2.6.0+cu126, torchaudio 2.6.0+cu126 and triton 3.2.0 from download.pytorch.org;
    nvidia-cuda-runtime-cu12 12.6.77 and nvidia-cudnn-cu12 9.5.1.17 (pypi.nvidia.com); sympy 1.13.1;
    fairseq 0.12.2 from the PyPI sdist.
  - These are the pins of step 2's torch, so the step-3 reinstall writes the same dist-info names.
  - The old run reproduces the torch 2.8.0 finding.
  - Later steps cannot replace torch:
    - the numpy step installs only numpy==1.23.5, without --ignore-installed, and numpy has no dependencies;
    - step 4 runs only `pip download --no-deps` and an in-place build_ext;
    - nothing after step 4 calls pip.
- **The gate checks the required items, in this order, through the wrapper.**
  - It runs with PYTHONPATH unset by build.sbatch, LD_LIBRARY_PATH replaced and no user site. In order:
    - numpy 1.23.5;
    - `torch.__version__ == "2.6.0+cu126"` (new);
    - `torch.cuda.is_available()`;
    - `sm_70` in `get_arch_list()` (CUDA_ARCH=70 from build.sbatch; the torch 2.6.0 cu126 x86_64 wheel is built
      for 7.0);
    - `fairseq.__version__ == "0.12.2"`;
    - the imports of examples, kenlm and flashlight;
    - batch_by_size;
    - "OK".
  - Then (new) `fairseq.__file__` and `__version__` must equal exactly `<site-packages>/fairseq/__init__.py 0.12.2`.
- **No false failure from the path match.**
  - `ls -d $PREFIX/lib/python3.*/site-packages` gives a single path for a Python 3.9 conda-forge env: the
    implementer's mamba-built py39 has only lib/python3.9. The `python3.1 -> python3.11` symlink that makes the sae
    env list two dirs exists only for Python 3.10 and later.
  - /work/asr4/hwu/conda contains no symlinked component. In the sae env, `readlink -f` equals the path, and
    numpy.__file__ matches the ls form.
  - fairseq 0.12.2 has no module-level print on `import fairseq`.
  - The build's cwd is analysis/w2vu_env_build, which holds only build.sbatch, so nothing can shadow fairseq.
- **The wrapper appears only after the gate.**
  - It is written as `w2vu-python.ungated`. The gate and the fairseq-origin check run through that file.
  - `mv` to `w2vu-python` is the last step before "== done".
  - The Python gate runs inside `if !`, so `set -e` does not preempt `gate_fail`, which exits 1 with the
    remove-the-prefix message.
  - A timed-out or killed build therefore leaves no `w2vu-python`. W2VU_PYTHON names exactly that file, so
    sisyphus cannot treat a partial env as ready.
  - The wrapper's content is unchanged, so no hash moves (get_w2vu_python has a fixed hash_overwrite).

### torchaudio 2.6.0+cu126 versus production's w2vu env

- The JUPITER reports do not record torchaudio in production's w2vu env from a measurement.
  - The only statement is impl_gan_port_wire_2026-09-25.md:52: "the w2vu env neither has nor needs it".
  - The reference script's step-2 comment says torchaudio was "deliberately omitted" (no aarch64 2.6.0+cu126 wheel).
  - The production torch is measured: 2.6.0+cu126 (impl_gan_port_fixes:106).
  - Match therefore cannot be confirmed. The likely state is that production had no torchaudio, so the i6 env
    carries one extra package.
- It is inert for these jobs.
  - Every torchaudio import in fairseq 0.12.2 (excluding examples/) is inside a function: audio_utils.py:44 and 215,
    feature_transforms/delta_deltas.py:25, tasks/text_to_speech.py:478 and models/speech_to_text/hub_interface.py:43.
  - In examples/wav2vec/unsupervised it appears only in scripts/remove_silence.py, which the jobs do not run.
  - It appears in none of the unpaired_audio_text, audio_finetuning or kaldi_decoder paths.
  - On x86_64 it cannot be avoided without `--no-deps` on fairseq, because fairseq lists it as a dependency.
- Record it in the P0 phase file with the other i6 deviations: torchaudio 2.6.0+cu126 present, x86_64/sm_70 wheel
  versus aarch64.

### Notes (no action for this launch)

- The script header (line 7) and the step-2 comment (lines 65-66, "torchaudio ... deliberately omitted") are now
  stale.
- The script no longer builds on aarch64: there is no torchaudio==2.6.0+cu126 aarch64 wheel. A JUPITER rebuild would
  fail loudly at step 3. No run depends on it.
- The build-time fairseq-origin check runs without the jobs' clone root on PYTHONPATH.
  - With the root prepended, the sparse clone's fairseq/ has no `__init__.py`, so it is only a PEP 420 portion and
    the env's regular package still wins.
  - The one-line job-time probe below is therefore optional extra evidence for SAE_i6_P0.md:165, not a blocker.
- Per the project rules, commit build_w2vu_env.sh with the implementer report and this review now that the review
  passes, before step 2.

### Final launch commands (from S)

1. sha256sum settings.py   # expect 743b4f7e...d752
   patch -p1 < analysis_out/w2vu2_settings.patch
2. sbatch analysis/w2vu_env_build/build.sbatch
3. CONDA_BIN=/bin/false bash recipe/i6_experiments/users/wu/experiments/unsupervised_asr/env/w2vu2_port_extras.sh /work/asr4/hwu/conda/envs/sae
   # must end with its "OK torch 2.7.1 torchaudio 2.7.1 sklearn 1.8.0" gate line
4. Only after log/w2vu_env_build.<jobid>.out shows "OK 2.6.0+cu126 0.12.2 1.23.5 | cuda: True",
   "fairseq: /work/asr4/hwu/conda/envs/w2vu/lib/python3.9/site-packages/fairseq/__init__.py 0.12.2" and "== done":
   (PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH setsid nohup /work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis --log_level 30 m -r config/sae_i6_w2vu2.py > log/sae_i6_w2vu2.manager.log 2>&1 < /dev/null &)
   Then write the pid to log/sae_i6_w2vu2.manager.pid, and start the watcher:
   SIS_LAUNCHER="/work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis" PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH bash ~/.claude/skills/sis/sis_watch.sh <pid> config/sae_i6_w2vu2.py 60
   Optional job-time probe, after CloneGitRepositoryJob.rnfyLkSbJUoz has finished: the command in the launch
   commands section above.
