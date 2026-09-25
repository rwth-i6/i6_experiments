# review: w2vu env build, step 4 sdist fetch (2026-09-25)

Verdict: PASS. No findings.

Reviewed: the uncommitted change in recipe/i6_experiments/users/wu/experiments/unsupervised_asr/env/build_w2vu_env.sh
(branch haotian_cycle_consistency_unsupervised, diff against HEAD 45c20acef), the implementer report
reports/impl_w2vu2_env_sdist_fetch_2026-09-25.md, and the launcher analysis/w2vu_env_build/build.sbatch (mtime 17:47, unchanged).
Read-only: no script run, env untouched, no job cancelled.

## (a) Single delta
`git diff --stat` for the checkout: this file only, 2 insertions and 1 deletion, no mode change. Old line 120 (`pip download ... --no-binary :all:`)
is replaced by new lines 120 (urllib fetch, socket default timeout 300 s) and 121 (sha256 check with `|| die`). No other line differs.

## (b) Abort semantics
The subshell `( cd "$BUILD" ... )` (l.119-135) is a standalone command, not in an if/&&/|| context, so a non-zero exit aborts the parent under `set -e`.
It inherits errexit and the `die` function (l.49). A failed urlretrieve raises, python exits 1, errexit ends the subshell, and the parent aborts.
On a hash mismatch, `sha256sum -c` exits 1 and `die` runs `exit 1` inside the subshell, which also aborts the parent.
Truncated bodies: urlretrieve raises ContentTooShortError, and the sha256 check covers the rest.

## (c) Downstream expectations
The hash matches the PyPI JSON for fairseq 0.12.2. I fetched it independently: the only sdist is fairseq-0.12.2.tar.gz, sha256 34f1b184...b524, 9,595,935 bytes.
I downloaded the file with the same one-liner into my scratchpad. The sha256 check passed.
Tar members: one top-level dir, fairseq-0.12.2/. It contains fairseq/data/data_utils_fast.pyx and token_block_utils_fast.pyx.
The sdist ships no pre-generated .cpp. pip download fetched this same single sdist before the change, so the input to tar, build_ext_only.py and cp is byte-identical to the old path.
The URL answers HTTP 302 to /packages/30/36/.../fairseq-0.12.2.tar.gz. Python 3.9 urllib follows 301/302/303/307, so the redirect is handled.
Network from cn-32: step 3 of job 4365689 downloaded the same 9.6 MB sdist from PyPI at 20.6 MB/s (log line 234).
SSL: the partial prefix has conda-forge python 3.9.23, openssl 3.6.4, ca-certificates 2026.7.22 and $PREFIX/ssl/cert.pem. No SSL_CERT_FILE or proxy override is set in the submitting environment.

## (d) Steps 4-7: no remaining isolated build or network wait
Lines 106-215 contain no pip call. The only network access in steps 4-7 is the new l.120, which the socket timeout bounds.
build_ext_only.py is run directly, so pyproject's build-system.requires is not consulted. The sdist's pyproject.toml has only [build-system]. Its setup.cfg has only [flake8]/[egg_info], so setuptools gets no setup_requires and fetches nothing.
setuptools 80.9.0 (in the prefix), Cython 3.1.5 (step 3) and numpy 1.23.5 (step 3b) are already installed. Steps 5-7 are local file ops and imports.

## (e) Launch
- build.sbatch:23 calls the recipe-checkout script by absolute path (the file just reviewed, not a copy) through `bash "$SCRIPT"`.
- The launch runs on partition gpu_32gb (V100). The preflight at l.38 requires compute_cap 7.0, with CUDA_ARCH=70 and GATE_CUDA=1. No apptainer, gpu_11gb or A100 is used.
- `-o` writes to the shared log/ dir (l.19), not /var/tmp. mktemp's BUILD lands in node-local TMPDIR=/var/tmp. That is scratch space and is removed at l.136.
- PIP_NO_CACHE_DIR=1 (l.30) keeps pip from writing a cache, and PYTHONPATH/CONDA_PREFIX are unset (l.28). Nothing writes to the sae env.
- The only writes outside the prefix are mamba's additions to the Miniforge package cache and to the node-local tmp. These are additive, unchanged from the two earlier runs, and not part of this delta.
- Preflight l.35 refuses the existing prefix. /work/asr4/hwu/conda/envs/w2vu exists now (partial), so it must be removed after 4365689 is cancelled, as planned.
- Time: steps 1-3 took 8 min in 4365689 (19:51 to 19:59), so the 04:00:00 limit is ample.

## Notes (not findings)
- Job 4365689 is still running bash on the file edited at 20:29. If its pip download returns before the cancel, bash resumes at old byte offset 8825.
  In the new file that offset falls mid-line (`umpy.get_include()], ...`), so bash hits a syntax error and the job exits.
  The job would touch nothing outside the prefix. Read any such failure as an artifact of the edit, not a build defect.
- I did not run the fetch with the env's own python 3.9. The implementer used system python 3.10 and so did I. A TLS or redirect failure there would abort loudly, not hang.
