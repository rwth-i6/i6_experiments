# impl: w2vu env build, step 4 sdist fetch (2026-09-25)

Status: DONE. Not committed (per dispatch); not run; /work/asr4/hwu/conda/envs/w2vu untouched; no jobs submitted.

## Change
File: recipe/i6_experiments/users/wu/experiments/unsupervised_asr/env/build_w2vu_env.sh, step 4, inside the `( cd "$BUILD" ... )` subshell.
The `pip download ... --no-binary :all:` line is replaced by a direct fetch with the env's python (300 s socket timeout) plus a sha256 check that calls the script's existing `die` helper on mismatch. Nothing else in the file changed.

```diff
-  "$PY" -m pip download fairseq==0.12.2 --no-deps --no-binary :all: --no-cache-dir -d . >/dev/null
+  "$PY" -c 'import socket,sys,urllib.request; socket.setdefaulttimeout(300); urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' https://files.pythonhosted.org/packages/source/f/fairseq/fairseq-0.12.2.tar.gz fairseq-0.12.2.tar.gz
+  echo "34f1b18426bf3844714534162f065ab733e049597476daa35fffb4d06a92b524  fairseq-0.12.2.tar.gz" | sha256sum -c --quiet - || die "fairseq-0.12.2.tar.gz sha256 mismatch"
```

The timeout is per socket operation (connect/read), not a total wall-clock limit. A download failure raises in python, so it exits non-zero and `set -e` aborts the script.

## Checks
- `bash -n build_w2vu_env.sh`: rc 0.
- URL/hash from the login node: the same python one-liner (system python3) fetched the file into a scratch dir under /var/tmp. Size 9,595,935 bytes, `sha256sum -c` OK. The tarball contains fairseq-0.12.2/fairseq/data/data_utils_fast.pyx and token_block_utils_fast.pyx, the two sources build_ext_only.py compiles. The scratch dir was removed afterwards.
- Mismatch path: I corrupted the scratch file and ran the same echo|sha256sum||die construct inside a `( ... )` subshell under `set -euo pipefail` with an equivalent die. It printed the die message, exited with rc 1, and neither the inner nor the outer follow-on command ran. So die inside the subshell aborts the whole script.
- Not checked: the fetch on the compute node itself (cn-32). The dispatch says compute nodes have internet.

## Other pip calls in the script (unchanged, listed only)
- l.62 `pip install -U pip==23.3.2`: wheel.
- l.67 `pip install torch==2.6.0 --index-url .../cu126`: wheel.
- l.80-87 step 3 `pip install --ignore-installed ... fairseq==0.12.2 ... kenlm==0.3.0 ...`: fairseq 0.12.2 and kenlm 0.3.0 have **no cp39 linux x86_64 wheel on PyPI (sdist only)**, so pip builds both from source in an isolated build env. This is not a `--no-binary :all:` call, so their build requirements (setuptools, wheel, cython, numpy) come as wheels and do not trigger the compile-everything behaviour. The other pins checked (flashlight-text, editdistance, bitarray) have cp39 manylinux wheels. The rest were not checked. Step 3 has already completed in earlier builds, since step 4 was reached.
- l.97/99 `pip show numpy` / `pip uninstall -y numpy`: no build.
- l.104 `pip install --no-cache-dir numpy==1.23.5`: wheel.
- There is no other `--no-binary`, `--no-build-isolation` or `pip wheel` in the script.
