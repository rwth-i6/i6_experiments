# Implementer: torch pin, gated wrapper and fairseq provenance in build_w2vu_env.sh (2026-09-25)

Status: DONE. Not committed; the env was not built and nothing was submitted.
Brief: fix MUST 1 and SHOULD 1 and 2 of reports/review_w2vu2_i6_launch_2026-09-25.md.
S = /u/hwu/setups/librispeech-960/2026-09-24-unsupervised, P = S/recipe/i6_experiments/users/wu/experiments/unsupervised_asr.

## File touched

P/env/build_w2vu_env.sh only (+26/-3 lines). sha256 before 57bbb69a...43b27c4, after 9526dd61...a6176.
S/analysis/w2vu_env_build/build.sbatch is unchanged. It needs nothing new: its closing `ls -l $PREFIX/bin/w2vu-python`
runs only after the script exits 0, and that is when the rename has happened.

1. MUST (step 3, the fairseq pip install). Added `--extra-index-url https://download.pytorch.org/whl/cu126` and
   `torch==2.6.0+cu126 torchaudio==2.6.0+cu126` to the `--ignore-installed` install, with a comment explaining why.
   The gate now asserts `torch.__version__ == "2.6.0+cu126"`, right after `import torch` and before the
   `cuda.is_available()` and sm_70 asserts.
   - torchaudio: the script names it nowhere, but on x86_64 step 3 installs it as fairseq's dependency
     (`torchaudio>=0.8.0` in fairseq 0.12.2 setup.py install_requires). So it is pinned to torch's build.
   - Later pip steps cannot replace torch. The numpy step runs without `--ignore-installed`, numpy has no
     dependencies, and step 4's `pip download --no-deps` only downloads. Step 2 is unchanged. Step 3 now reinstalls
     the same 2.6.0+cu126 over it, which gives the same dist-info names, so nothing gets stacked. The cost is a second
     download of about 765 MB.
2. SHOULD (steps 6-7). The wrapper is now written as `<prefix>/bin/w2vu-python.ungated`, and the gate runs through that
   file. It is renamed with `mv` to `w2vu-python` only after every gate check passes. If a check fails, `gate_fail`
   exits 1 with this message: "gate failed[: reason]; <prefix> is not usable (no w2vu-python was written): remove it
   (rm -rf <prefix>) before rebuilding". The wrapper's content is byte-identical.
3. SHOULD (fairseq provenance, after the Python gate). The script runs the config/w2vu2.py docstring check through the
   wrapper: `-c 'import fairseq; print(fairseq.__file__, fairseq.__version__)'`. It prints "fairseq: <output>" and
   requires exactly "<prefix>/lib/python3.*/site-packages/fairseq/__init__.py 0.12.2". The existing
   "OK <torch> <fairseq> <numpy> | cuda: ..." line is unchanged, so the review's launch condition still reads
   "OK 2.6.0+cu126 0.12.2 1.23.5 | cuda: True" followed by "== done". A new "fairseq: ..." line comes between them.

## Checks

- `bash -n` on the edited script: OK.
- The dry run below resolves step 3 exactly as written in the old and the new script. It used a throwaway Python 3.9.23
  env with pip 23.3.2 (the script's pip) in the session scratchpad, on this x86_64 host with PyPI. It covered the full
  package list, including the fairseq, kenlm and antlr4 sdists, not the torch-only subset that the review ran.
  The flags were `--dry-run --report --ignore-installed`, the package list was taken from the script by sed, and
  PYTHONPATH was unset. Both runs exited 0.
  - New: torch 2.6.0+cu126 and torchaudio 2.6.0+cu126, both from download-r2.pytorch.org/whl/cu126. Also
    triton 3.2.0, nvidia-cuda-runtime-cu12 12.6.77, nvidia-cudnn-cu12 9.5.1.17, sympy 1.13.1, fairseq 0.12.2,
    numpy 2.0.2 (which the numpy step then replaces, as before). 47 packages.
  - Old, for contrast: torch 2.8.0, torchaudio 2.8.0, triton 3.4.0, nvidia-cuda-runtime-cu12 12.8.90,
    nvidia-cudnn-cu12 9.10.2.21, sympy 1.14.0. 51 packages. This reproduces the review's finding.
  - The two resolves differ only in torch, torchaudio, triton, sympy, the nvidia-*-cu12 set, and in importlib_metadata,
    zipp, setuptools and nvidia-cufile, which only the old resolve pulls. Every other package has the same version in
    both. Through the extra index, only torch, torchaudio, triton, jinja2 3.1.6 and colorama 0.4.6 come from
    download.pytorch.org. jinja2 and colorama are at the same versions as in the PyPI resolve.
- Stub test of the steps 6-7 control flow. The harness was the script's `die` plus steps 6-7 verbatim, run on a fake
  prefix with a stub python.
  - Gate passes: rc 0, bin/ holds `w2vu-python`, and the output is "OK ...", then "fairseq: ...", then "== done".
  - Python gate fails: rc 1, the remove-the-prefix message, and bin/ holds only `w2vu-python.ungated`.
  - fairseq from elsewhere: rc 1, "gate failed: fairseq is not 0.12.2 from <site-packages>; ...", and only
    `.ungated` exists.
  - This checks the control flow only. The real gate has not run: the env was not built.

## Note: w2vu2_port_extras.sh with CONDA_BIN=/bin/false (review SHOULD 3)

Confirmed: the script dies without installing anything through conda. If scikit-learn is missing or at another
version, line 38 keeps CONDA_BIN=/bin/false because it is non-empty. Line 40 then runs `/bin/false install -y -p ...`,
which exits 1, and `set -e` stops the script before the gate. This was simulated on a fake prefix whose stub python
reports torch 2.7.1 and fails `import sklearn`: rc 1.
Two caveats:
- The script prints no message of its own. The only sign is rc 1 and the missing "OK torch ..." gate line.
- Step 1, the pip `--no-deps` torchaudio install, has already run by then.
scikit-learn in /work/asr4/hwu/conda/envs/sae is 1.8.0 today, checked by import, so the branch is skipped.

## Left open or assumed

- Assumed: torchaudio pinned to 2.6.0+cu126. On x86_64 it gets installed either way. The reference env (aarch64)
  reportedly had no torchaudio, so its presence in the i6 env is a disclosed difference, as before. No w2vu job imports it.
- The build-time provenance check runs without the jobs' fairseq root on PYTHONPATH (build.sbatch unsets PYTHONPATH).
  The job-time probe with the CloneGitRepositoryJob root (review SHOULD 2) is still needed as evidence for
  SAE_i6_P0.md:165.
- Not updated: the script header (line 7 says step 6 writes w2vu-python) and the step 2 comment (torchaudio is
  "deliberately omitted"). Both are now slightly stale. They were kept byte-identical as the brief asked.

## Full diff

```diff
--- a/env/build_w2vu_env.sh
+++ b/env/build_w2vu_env.sh
@@ -73,7 +73,13 @@
 # numpy is deliberately NOT in this list: --ignore-installed overwrites without uninstalling, and for
 # numpy that leaves a Frankenstein (numpy-1.23.5.dist-info AND numpy-2.0.2.dist-info side by side,
 # pip reporting 1.23.5 while `import numpy` says 2.0.2). It is pinned separately below.
+# torch is pinned here again, from the cu126 index: --ignore-installed hides step 2's torch from the
+# resolver, and fairseq 0.12.2 lists `torch` and `torchaudio>=0.8.0` unpinned, so without the pin pip
+# re-resolves them from PyPI (torch 2.8.0 cu128 on x86_64) and lays that tree over step 2's.
+# torchaudio is installed here as fairseq's dependency on x86_64, so it is pinned to torch's build.
 "$PY" -m pip install --ignore-installed \
+  --extra-index-url https://download.pytorch.org/whl/cu126 \
+  torch==2.6.0+cu126 torchaudio==2.6.0+cu126 \
   fairseq==0.12.2 omegaconf==2.0.6 hydra-core==1.0.7 \
   typing_extensions==4.15.0 cffi==2.0.0 soundfile==0.13.1 bitarray==3.7.2 \
   editdistance==0.8.1 sacrebleu==2.5.1 regex==2026.1.15 Cython==3.1.5 \
@@ -132,7 +138,10 @@
 # The prefix and the torch lib dir are resolved now and written into the wrapper.
 TORCH_LIB="$(ls -d "$PREFIX"/lib/python3.*/site-packages/torch/lib)"
 WRAPPER="$PREFIX/bin/w2vu-python"
-cat > "$WRAPPER" <<EOF
+# Written under a temporary name; step 7 renames it to w2vu-python only after the gate passes, so a
+# failed or partial env never has a wrapper that looks usable (sisyphus takes the existing path as ready).
+WRAPPER_UNGATED="$WRAPPER.ungated"
+cat > "$WRAPPER_UNGATED" <<EOF
 #!/usr/bin/env bash
 # The w2vu env's python with the env isolation of the reference setup's settings.py
 # (_w2vu_env_overrides, selftrain.py:295): its own lib dirs only, the \`examples\` shim, no user
@@ -144,12 +153,13 @@
 export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
 exec "$PREFIX/bin/python" "\$@"
 EOF
-chmod +x "$WRAPPER"
+chmod +x "$WRAPPER_UNGATED"
 
 # ---- 7. gate ------------------------------------------------------------------------------------
 # Every import must resolve with user-site suppressed and the main env's LD_LIBRARY_PATH replaced,
 # i.e. as the job will see it: so the gate runs through the wrapper (which adds the shim).
-GATE_CUDA="$GATE_CUDA" CUDA_ARCH="$CUDA_ARCH" "$WRAPPER" - <<'EOF'
+gate_fail() { die "gate failed${1:+: $1}; $PREFIX is not usable (no w2vu-python was written): remove it (rm -rf $PREFIX) before rebuilding"; }
+if ! GATE_CUDA="$GATE_CUDA" CUDA_ARCH="$CUDA_ARCH" "$WRAPPER_UNGATED" - <<'EOF'
 import logging, os
 assert hasattr(logging, "getLogger"), f"stdlib logging is shadowed: {logging.__file__}"
 
@@ -157,6 +167,7 @@
 assert np.__version__ == "1.23.5", f"numpy is {np.__version__}; fairseq 0.12.2 needs <1.24 (np.int)"
 
 import torch
+assert torch.__version__ == "2.6.0+cu126", f"torch is {torch.__version__}; the reference env is 2.6.0+cu126"
 if os.environ["GATE_CUDA"] == "1":
     # fairseq trains on CPU without an error when CUDA is not visible (trainer.py:61);
     # checked first: get_arch_list() is [] then
@@ -179,4 +190,16 @@
 assert len(b) == 5, b
 print("OK", torch.__version__, fairseq.__version__, np.__version__, "| cuda:", torch.cuda.is_available())
 EOF
+then
+    gate_fail
+fi
+# fairseq provenance: the check of the config/w2vu2.py docstring, through the wrapper. fairseq must be
+# 0.12.2 from this env's site-packages (here without the jobs' fairseq root on PYTHONPATH).
+SITE="$(ls -d "$PREFIX"/lib/python3.*/site-packages)"
+FAIRSEQ_ORIGIN="$("$WRAPPER_UNGATED" -c 'import fairseq; print(fairseq.__file__, fairseq.__version__)')" \
+    || gate_fail "import fairseq"
+echo "fairseq: $FAIRSEQ_ORIGIN"
+[[ "$FAIRSEQ_ORIGIN" == "$SITE/fairseq/__init__.py 0.12.2" ]] \
+    || gate_fail "fairseq is not 0.12.2 from $SITE"
+mv "$WRAPPER_UNGATED" "$WRAPPER"
 echo "== done: W2VU_PYTHON = \"$WRAPPER\""
```
