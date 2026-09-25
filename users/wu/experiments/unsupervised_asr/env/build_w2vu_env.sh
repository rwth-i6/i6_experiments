#!/usr/bin/env bash
# Build the `w2vu` conda env: fairseq 0.12.2 + CUDA torch, for the wav2vec-U 2.0 GAN (section 1c).
#
# Ported from i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/w2vu2/build_w2vu_env.sh
# (the env of every reference GAN run).  Port changes: the prefix is a required argument (no cluster
# default); conda comes from CONDA_BIN; the GPU arch the gate checks is CUDA_ARCH (the reference
# asserted sm_90, GH200); and step 6 writes `<prefix>/bin/w2vu-python`, the wrapper that
# `settings.py`'s W2VU_PYTHON names (see w2vu2_tools.py).
#
# Usage: build_w2vu_env.sh <env_prefix>
#
# Why a second env at all: the main env's torch is 2.7.1 on py3.11, and fairseq 0.12.2 pins a
# py<=3.10 stack (omegaconf 2.0.6, hydra-core 1.0.7), so it cannot be installed there.
#
# The wrapper: jobs run under the sisyphus worker's environment, whose LD_LIBRARY_PATH holds the
# main env's lib dirs.  A py3.9 interpreter that inherits it dlopens the py3.11 libtorch_python and
# dies with `undefined symbol: PyObject_CallOneArg`.  The reference setup set the env of its w2vu jobs
# in settings.py's worker_wrapper (`_w2vu_env_overrides`, `requires_env = "w2vu"`) and in its 1d
# fine-tuning job (selftrain.py:295); the port may not touch the worker wrapper, and i6_core's
# FairseqHydraTrainingJob has no env hook, so the env ships a wrapper that sets all of them:
#     LD_LIBRARY_PATH=<prefix>/lib:<prefix>/lib/python3.*/site-packages/torch/lib   (replaced)
#     PYTHONPATH=<prefix>/fairseq_shim:$PYTHONPATH   (prepended: the top-level `examples` package)
#     PYTHONNOUSERSITE=1
#     TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   (torch 2.6 defaults torch.load to weights_only=True, and
#                                          fairseq 0.12.2 cannot load the wav2vec2 LV-60 checkpoint
#                                          that way; the reference set it for the 1d fine-tuning job,
#                                          the wrapper sets it for every w2vu job)
#
# Cluster-specific settings (environment variables; the JUPITER values are the examples):
#   CONDA_BIN   conda or mamba executable. Default: mamba, else conda, from PATH.
#               JUPITER: the mamba of the project conda install
#   CUDA_ARCH   the GPU compute capability the gate asserts in torch's arch list, without the dot.
#               Default: 90.  JUPITER (GH200): 90.  Set it to the GPUs the GAN will run on.
#   GATE_CUDA   1 (default): the gate asserts CUDA_ARCH and torch.cuda.is_available() through the
#               wrapper, so it must run on a GPU node of the CUDA_ARCH type (a GPU hidden by the
#               wrapper's LD_LIBRARY_PATH would otherwise let fairseq train on CPU without an error);
#               0 skips both asserts (a CPU-only build host).
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <env_prefix>" >&2
    exit 2
fi
PREFIX="$1"
CONDA_BIN="${CONDA_BIN:-$(command -v mamba || command -v conda || true)}"
CUDA_ARCH="${CUDA_ARCH:-90}"
GATE_CUDA="${GATE_CUDA:-1}"

die() { echo "build_w2vu_env.sh: $*" >&2; exit 1; }
[[ -n "$CONDA_BIN" ]] || die "no conda/mamba on PATH; set CONDA_BIN"
[[ ! -e "$PREFIX" ]] || die "$PREFIX exists; this script only creates a new env"

# ---- 1. the env ---------------------------------------------------------------------------------
"$CONDA_BIN" create -y -p "$PREFIX" python=3.9
PY="$PREFIX/bin/python"
export PATH="$PREFIX/bin:$PATH"

# fairseq 0.12.2's deps list `PyYAML>=5.1.*`, which is not valid PEP 440. Modern pip rejects the
# whole resolve with a misleading "No matching distribution for omegaconf==2.0.6"; pip 23.x still
# tolerates it. Do not "fix" this by relaxing the omegaconf pin -- fairseq 0.12.2 needs exactly 2.0.6
# (hydra-core 1.0.7 is likewise not interchangeable with 1.1+).
"$PY" -m pip install -U 'pip==23.3.2'

# ---- 2. torch -----------------------------------------------------------------------------------
# aarch64 has no torchaudio==2.6.0+cu126 wheel. It is not needed: the w2v-U path never imports it
# (the features are written by a job of the main env), so it is deliberately omitted.
"$PY" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# ---- 3. fairseq and its stack -------------------------------------------------------------------
# --ignore-installed is load-bearing: jobs run with PYTHONNOUSERSITE=1, but pip still *sees*
# ~/.local/lib/python3.9/site-packages and silently skips anything already there. Without this,
# typing_extensions/cffi/soundfile resolve on the login node and then ImportError inside the job.
# numpy is deliberately NOT in this list: --ignore-installed overwrites without uninstalling, and for
# numpy that leaves a Frankenstein (numpy-1.23.5.dist-info AND numpy-2.0.2.dist-info side by side,
# pip reporting 1.23.5 while `import numpy` says 2.0.2). It is pinned separately below.
# torch is pinned here again, from the cu126 index: --ignore-installed hides step 2's torch from the
# resolver, and fairseq 0.12.2 lists `torch` and `torchaudio>=0.8.0` unpinned, so without the pin pip
# re-resolves them from PyPI (torch 2.8.0 cu128 on x86_64) and lays that tree over step 2's.
# torchaudio is installed here as fairseq's dependency on x86_64, so it is pinned to torch's build.
"$PY" -m pip install --ignore-installed \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  torch==2.6.0+cu126 torchaudio==2.6.0+cu126 \
  fairseq==0.12.2 omegaconf==2.0.6 hydra-core==1.0.7 \
  typing_extensions==4.15.0 cffi==2.0.0 soundfile==0.13.1 bitarray==3.7.2 \
  editdistance==0.8.1 sacrebleu==2.5.1 regex==2026.1.15 Cython==3.1.5 \
  scipy==1.13.1 PyYAML==6.0.3 npy-append-array==0.9.19 kenlm==0.3.0 \
  flashlight-text==0.0.7

# fairseq 0.12.2 needs numpy<1.24 -- fairseq/data/data_utils.py:488 still uses `np.int`.
# Uninstall in a loop: a previous --ignore-installed run can leave several dist-infos stacked, and
# each `pip uninstall` removes only one. The loop runs while `pip show numpy` finds it (pip 23.3.2
# exits 1 when absent), not on uninstall's status: uninstalling an absent package logs "Skipping ...
# as it is not installed" and still exits 0, so looping on it never ends. A failing uninstall ends the
# loop (the rm -rf below clears what it left); still present after 10 uninstalls is an error.
# PYTHONNOUSERSITE=1: only this env's numpy counts (a user-site numpy is outside the env).
n_uninst=0
while PYTHONNOUSERSITE=1 "$PY" -m pip show numpy >/dev/null 2>&1; do
    (( n_uninst < 10 )) || die "numpy still installed after 10 pip uninstalls: $(PYTHONNOUSERSITE=1 "$PY" -m pip show numpy 2>&1 | grep -E '^(Version|Location):' | tr '\n' ' ')"
    PYTHONNOUSERSITE=1 "$PY" -m pip uninstall -y numpy >/dev/null 2>&1 || break
    n_uninst=$((n_uninst + 1))
done
rm -rf "$PREFIX"/lib/python3.*/site-packages/numpy "$PREFIX"/lib/python3.*/site-packages/numpy-*.dist-info \
       "$PREFIX"/lib/python3.*/site-packages/numpy.libs
"$PY" -m pip install --no-cache-dir "numpy==1.23.5"

# ---- 4. the two Cython extensions ---------------------------------------------------------------
# Rebuild fairseq's two Cython extensions against *that* numpy. Neither pip mode can do this itself:
#   with build isolation    setup.py's `from torch.utils import cpp_extension` raises ImportError in
#                           the isolated env, so the libbase extension is skipped and the wheel builds
#                           -- but the isolated env holds the newest numpy, so the .so targets the
#                           numpy-2 ABI and every run dies in batch_by_size with
#                           "numpy.dtype size changed ... Expected 96 from C header, got 88".
#   without build isolation torch is importable, so setup.py adds libbase, which needs
#                           fairseq/clib/libbase/balanced_assignment.cpp -- a file the 0.12.2 sdist
#                           does not ship. The build cannot succeed at all.
# So build exactly the two extensions w2v-U needs, and drop them into the installed package.
SP="$(PYTHONNOUSERSITE=1 "$PY" -c 'import fairseq, os; print(os.path.dirname(fairseq.__file__))')"
BUILD="$(mktemp -d)"
( cd "$BUILD"
  "$PY" -m pip download fairseq==0.12.2 --no-deps --no-binary :all: --no-cache-dir -d . >/dev/null
  tar xf fairseq-0.12.2.tar.gz
  cd fairseq-0.12.2
  cat > build_ext_only.py <<'PY'
from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize
exts = [Extension(f"fairseq.data.{n}", sources=[f"fairseq/data/{n}.pyx"],
                  include_dirs=[numpy.get_include()], language="c++")
        for n in ("data_utils_fast", "token_block_utils_fast")]
setup(name="fairseq_ext_only", ext_modules=cythonize(exts, language_level="3"))
PY
  PYTHONNOUSERSITE=1 "$PY" build_ext_only.py build_ext --inplace
  cp fairseq/data/*.so "$SP/data/"
)
rm -rf "$BUILD"

# ---- 5. the `examples` shim ---------------------------------------------------------------------
# The unsupervised task imports `from examples.speech_recognition...`, so `examples` must be a
# top-level package. Upstream's repo root has fairseq/ and examples/ as siblings; the wheel nests
# examples *inside* fairseq/, next to logging/, data/ and tasks/ -- so putting site-packages/fairseq
# on PYTHONPATH would shadow the **stdlib** `logging` for every process that inherits it. This shim
# dir exposes `examples` and nothing else.
mkdir -p "$PREFIX/fairseq_shim"
ln -sfn "$SP/examples" "$PREFIX/fairseq_shim/examples"

# ---- 6. the wrapper -----------------------------------------------------------------------------
# The prefix and the torch lib dir are resolved now and written into the wrapper.
TORCH_LIB="$(ls -d "$PREFIX"/lib/python3.*/site-packages/torch/lib)"
WRAPPER="$PREFIX/bin/w2vu-python"
# Written under a temporary name; step 7 renames it to w2vu-python only after the gate passes, so a
# failed or partial env never has a wrapper that looks usable (sisyphus takes the existing path as ready).
WRAPPER_UNGATED="$WRAPPER.ungated"
cat > "$WRAPPER_UNGATED" <<EOF
#!/usr/bin/env bash
# The w2vu env's python with the env isolation of the reference setup's settings.py
# (_w2vu_env_overrides, selftrain.py:295): its own lib dirs only, the \`examples\` shim, no user
# site-packages, torch.load without the weights_only default.
# Written by build_w2vu_env.sh; settings.py W2VU_PYTHON points here.
export LD_LIBRARY_PATH="$PREFIX/lib:$TORCH_LIB"
export PYTHONPATH="$PREFIX/fairseq_shim\${PYTHONPATH:+:\$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
exec "$PREFIX/bin/python" "\$@"
EOF
chmod +x "$WRAPPER_UNGATED"

# ---- 7. gate ------------------------------------------------------------------------------------
# Every import must resolve with user-site suppressed and the main env's LD_LIBRARY_PATH replaced,
# i.e. as the job will see it: so the gate runs through the wrapper (which adds the shim).
gate_fail() { die "gate failed${1:+: $1}; $PREFIX is not usable (no w2vu-python was written): remove it (rm -rf $PREFIX) before rebuilding"; }
if ! GATE_CUDA="$GATE_CUDA" CUDA_ARCH="$CUDA_ARCH" "$WRAPPER_UNGATED" - <<'EOF'
import logging, os
assert hasattr(logging, "getLogger"), f"stdlib logging is shadowed: {logging.__file__}"

import numpy as np
assert np.__version__ == "1.23.5", f"numpy is {np.__version__}; fairseq 0.12.2 needs <1.24 (np.int)"

import torch
assert torch.__version__ == "2.6.0+cu126", f"torch is {torch.__version__}; the reference env is 2.6.0+cu126"
if os.environ["GATE_CUDA"] == "1":
    # fairseq trains on CPU without an error when CUDA is not visible (trainer.py:61);
    # checked first: get_arch_list() is [] then
    assert torch.cuda.is_available(), ("CUDA not available through the wrapper (LD_LIBRARY_PATH="
                                       + os.environ.get("LD_LIBRARY_PATH", "") + "); run the gate on a GPU "
                                       "node, or GATE_CUDA=0 on a CPU-only host")
    arch = "sm_" + os.environ["CUDA_ARCH"]
    assert arch in torch.cuda.get_arch_list(), (arch, torch.cuda.get_arch_list())

import fairseq
assert fairseq.__version__ == "0.12.2", fairseq.__version__
from examples.speech_recognition.kaldi.kaldi_decoder import KaldiDecoderConfig  # noqa: F401
import kenlm  # noqa: F401  (the unpaired_audio_text task's LM)
from flashlight.lib.text.decoder import KenLM, LexiconDecoder  # noqa: F401  (the 1d word decode)
assert os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") == "1"

# The numpy-ABI check: importing the extension and actually calling it are different failures.
from fairseq.data import data_utils
b = data_utils.batch_by_size(np.arange(20), num_tokens_fn=lambda i: 10, max_tokens=50, max_sentences=4)
assert len(b) == 5, b
print("OK", torch.__version__, fairseq.__version__, np.__version__, "| cuda:", torch.cuda.is_available())
EOF
then
    gate_fail
fi
# fairseq provenance: the check of the config/w2vu2.py docstring, through the wrapper. fairseq must be
# 0.12.2 from this env's site-packages (here without the jobs' fairseq root on PYTHONPATH).
SITE="$(ls -d "$PREFIX"/lib/python3.*/site-packages)"
FAIRSEQ_ORIGIN="$("$WRAPPER_UNGATED" -c 'import fairseq; print(fairseq.__file__, fairseq.__version__)')" \
    || gate_fail "import fairseq"
echo "fairseq: $FAIRSEQ_ORIGIN"
[[ "$FAIRSEQ_ORIGIN" == "$SITE/fairseq/__init__.py 0.12.2" ]] \
    || gate_fail "fairseq is not 0.12.2 from $SITE"
mv "$WRAPPER_UNGATED" "$WRAPPER"
echo "== done: W2VU_PYTHON = \"$WRAPPER\""
