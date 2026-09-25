# review: w2vu env build, bounded numpy-uninstall loop (2026-09-25)

Verdict: PASS. No findings. The change can be committed and the env rebuilt with the commands at the end.

- Change: P/env/build_w2vu_env.sh, step 3, lines 89-104. P = recipe/i6_experiments/users/wu/experiments/unsupervised_asr.
  It is uncommitted; the working-tree sha256 is 6c357079...ee56.
- Implementer report: reports/impl_w2vu2_env_numpy_loop_2026-09-25.md.
- Hung run: log/w2vu_env_build.4364831.out.

## 1. The diff is step 3 only

- `git diff --numstat` shows one file, +11/-2. The hunk sits between the step-3 install and the `rm -rf`, and it
  matches the implementer report byte for byte.
- The base is the reviewed and committed script: `git show HEAD:<file> | sha256sum` gives 9526dd61...a6176 (commit
  0dbe9cf7f). That is the sha the launch-review addendum approved.
- The file mode is unchanged (100755), and `bash -n` passes.
- build.sbatch and settings.py are untouched: the sbatch mtime is still 17:47.
- The hung job started at 18:32:26, after 0dbe9cf7f was committed at 18:31:35. So it ran this base.

## 2. The loop terminates in every case

Source checked in the partial env's pip 23.3.2 (read-only):
- commands/show.py:44-49 and print_results. `run` returns ERROR when nothing prints, and status_codes.py gives
  ERROR = 1. Measured directly with the env's python (`-B`, user site off): `pip show numpy` returns rc=1 with numpy
  absent, and `pip show pip` returns rc=0.
- req_install.py:717-720 and commands/uninstall.py:113. Uninstalling an absent package logs "Skipping" and
  returns SUCCESS. This is the hang, and step-2 line 220 of the log confirms step 2 installed no numpy.
- show and uninstall inherit the no-op `handle_pip_version_check` of base_command.py:82. Neither makes a network
  call, so no loop iteration can stall on the network.
- Python 3.9 uses the pkg_resources metadata backend (metadata/__init__.py:45). show and uninstall both enumerate
  get_default_environment(), so whenever show finds numpy, uninstall finds one copy.

Every pass through the loop body breaks, dies or increments n_uninst. At n_uninst=10, the next passing `show` dies.
So there are at most 10 uninstalls and 11 shows. The loop text, extracted with the implementer's sed and run under
`set -euo pipefail` with a stub pip (scratchpad review_loop/), gave:

| case | result |
|---|---|
| absent | 0 uninstalls, rc 0 |
| one copy | 1 uninstall, rc 0 |
| three stacked copies | 3 uninstalls, rc 0 |
| uninstall fails | break, 0 uninstalls, rc 0; the rm -rf follows |
| uninstall exits 0 without removing anything (the "outside environment" case, req_uninstall.py:446-448) | die at the cap, rc 1, message carries Version and Location |

The implementer's real-pip tests T1-T5 cover the same cases.

`PYTHONNOUSERSITE=1` is a per-command prefix, so it does not persist:
- The later `pip install numpy==1.23.5` runs as before. There is no --user, no PIP_* variables and no pip.conf in
  ~/.config/pip, ~/.pip, /etc or the prefix, and the prefix is writable. It installs into the env's site-packages.
- The gate imports numpy through the wrapper (PYTHONNOUSERSITE=1). Its sys.path ends at the same env
  site-packages.
- The user site is enabled for the env's python, but ~/.local/lib holds only python2.7. The flag therefore changes
  nothing today, and it only guards against a future user-site numpy.

## 3. set -euo pipefail

- The while condition is exempt from errexit.
- `(( n_uninst < 10 )) || die` is the left side of an `||` list, so it does not trip errexit.
- The uninstall ends with `|| break`.
- `n_uninst=$((n_uninst + 1))` is an assignment and always returns 0. It avoids the `((n++))` trap, which returns 1
  at 0.
- n_uninst is set before use (set -u).
- The pipeline inside die's `$(...)` cannot change the exit code, because die exits 1 regardless.
- Nothing continues where it should stop:
  - a still-present numpy after the cap dies;
  - an uninstall failure falls through to the `rm -rf`, which removes numpy/, numpy-*.dist-info and numpy.libs
    (lines 102-103);
  - a failed `pip install numpy==1.23.5` aborts under -e;
  - a wrong numpy fails the gate's `np.__version__ == "1.23.5"` (line 176).

## 4. The rest of the script, from line 104 to "== done"

- No other loop. The `for n in (...)` at line 129 sits inside a Python heredoc and has two items.
- Every pip call that touches the network (lines 104 and 120) runs with pip's defaults: --timeout 15 s and --retries
  5 (cmdoptions.py:280-294). No config overrides them. A stalled connection therefore fails loudly within minutes,
  and `set -e` aborts the script.
- A stall that keeps trickling bytes is bounded only by `--time=04:00:00`.
- The missing explicit timeout does not matter.
- Line 117 (`import fairseq` without the wrapper, under the submit shell's LD_LIBRARY_PATH):
  - In the partial env, `import torch` under this desktop's inherited LD_LIBRARY_PATH (cuda-9.1, cudnn-7.1, acml)
    loads 2.6.0+cu126. No soname collision.
  - fairseq's Cython imports are lazy, so this line is not expected to fail.
  - It is not yet measured on the node with numpy 1.23.5, and any failure would be a loud abort.
- Step 4's subshell inherits -e, so a failed download, build or cp aborts the script.
- The gate runs inside `if !`, as does the fairseq-origin check. The wrapper is renamed only after both pass.
- Python asserts would vanish under PYTHONOPTIMIZE, but the submit shell does not set it. That is not a live path.
- Nothing can pass silently: the numpy, torch, CUDA/sm_70, fairseq version and origin, and batch_by_size checks are
  all asserted.

## 5. Relaunch plan

- **Removing the partial env is safe.**
  - The only references to /work/asr4/hwu/conda/envs/w2vu are settings.py:33 (W2VU_PYTHON = <prefix>/bin/w2vu-python)
    and config/sae_i6_w2vu2.py. The wrapper file does not exist.
  - The live managers run sae_i6_p0.py and sae_i6_p1_ladder.py, and neither mentions w2vu.
  - The hwu Slurm jobs are all RETURNN train and forward jobs.
  - No symlink from the sae or kenlm_build envs points into w2vu, and no process on this host has the path open.
  - The prefix holds only the partial env: no numpy and no wrapper.
- **The launch wrapper is unchanged and adequate.**
  - build.sbatch sets -p gpu_32gb, --gres=gpu:1 and CUDA_ARCH=70, and preflights sm_70, g++ and an absent prefix.
  - It writes its log to S/log/w2vu_env_build.%j.out, not /var/tmp.
  - The partition limit is 7 days.
  - --time=04:00:00 is ample: steps 1-3 took about 8 min in 4364831 (18:32:26 to the bin mtime 18:40:13), and step
    4 plus the gate take minutes.
  - pip's build temp goes to node-local TMPDIR=/var/tmp. That is scratch, not job output.
- **Order.** Per the project rules, commit the reviewed script with both reports before the rebuild, so the env is
  built from a commit.

## Launch commands

From S = /u/hwu/setups/librispeech-960/2026-09-24-unsupervised:

1. `git -C recipe/i6_experiments branch --show-current`
   Expect haotian_cycle_consistency_unsupervised.
2. Stage the script and both reports:
   `git -C recipe/i6_experiments add users/wu/experiments/unsupervised_asr/env/build_w2vu_env.sh users/wu/exp_logs/SAE_i6/reports/impl_w2vu2_env_numpy_loop_2026-09-25.md users/wu/exp_logs/SAE_i6/reports/review_w2vu2_env_numpy_loop_2026-09-25.md`
3. Commit, with no attribution line:
   `git -C recipe/i6_experiments commit -m "unsupervised_asr: w2vu env build ends the numpy uninstall loop on pip show, capped at 10; SAE_i6 impl and review reports"`
4. Check that no build job is running: `squeue -u hwu -n sae_i6_w2vu_env_build`. The expected output is empty.
5. Remove the partial env: `rm -rf /work/asr4/hwu/conda/envs/w2vu`
6. Submit the build: `sbatch analysis/w2vu_env_build/build.sbatch`
7. Pass when log/w2vu_env_build.<jobid>.out shows all three lines:
   - "OK 2.6.0+cu126 0.12.2 1.23.5 | cuda: True"
   - "fairseq: /work/asr4/hwu/conda/envs/w2vu/lib/python3.9/site-packages/fairseq/__init__.py 0.12.2"
   - "== done"

   After that, continue with the addendum's step 4 in reports/review_w2vu2_i6_launch_2026-09-25.md.
